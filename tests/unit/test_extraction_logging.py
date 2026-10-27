import json
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest
from litellm import ModelResponse

from deet import custom_hierarchical, main_hierarchical
from deet.estimate_extraction_cost import (
    estimate_extraction_cost,
    main,
    summarize_extraction_costs,
)
from deet.hierarchical_mvp.extraction_logging import (
    ExtractionUsageLog,
    TokenUsage,
    UsageCall,
    UsageCallback,
    extraction_usage_log,
    normalize_usage,
)


def _response(usage=None, text="[[ ## answer ## ]]\nyes\n[[ ## completed ## ]]"):
    return ModelResponse(
        model="test-model",
        choices=[{"message": {"role": "assistant", "content": text}}],
        usage=usage
        if usage is not None
        else {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    )


def _read_log(tmp_path):
    paths = list(tmp_path.glob("extraction_log_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "raw",
    [
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 100},
        },
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 100},
        },
        {
            "input_tokens": 500,
            "output_tokens": 100,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 100,
        },
    ],
)
def test_normalize_usage(raw):
    usage = normalize_usage(raw)
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 100
    assert usage.cached_input_tokens == 400
    assert usage.non_cached_input_tokens == 600
    assert usage.cache_write_input_tokens == 100


def test_missing_usage_is_unknown():
    assert normalize_usage({}).input_tokens is None
    usage = normalize_usage({"prompt_tokens": 100, "completion_tokens": 5})
    assert usage.cached_input_tokens is None
    assert usage.non_cached_input_tokens is None
    assert usage.cache_write_input_tokens is None


@pytest.mark.parametrize(("cached", "writes"), [(101, 0), (50, 51), (-1, 0)])
def test_invalid_cache_counts_rejected(cached, writes):
    with pytest.raises(ValueError, match=r"exceed|greater than or equal"):
        normalize_usage(
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": cached,
                    "cache_write_tokens": writes,
                },
            }
        )


def test_real_dspy_calls_are_attributed_and_summed(tmp_path):
    class FirstStage(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class SecondStage(FirstStage):
        """A different task."""

    lm = dspy.LM("openai/test-model", cache=False)
    with (
        dspy.context(lm=lm),
        patch.object(lm, "forward", side_effect=[_response(), _response()]),
        extraction_usage_log(tmp_path, "paper", "PrognosticStudy", lm.model),
    ):
        assert dspy.Predict(FirstStage)(question="private document").answer == "yes"
        assert dspy.Predict(SecondStage)(question="private document").answer == "yes"
    log = _read_log(tmp_path)
    assert [call["stage"] for call in log["calls"]] == ["FirstStage", "SecondStage"]
    assert log["totals"]["input_tokens"] == 2000
    assert log["totals"]["cached_input_tokens"] == 800
    assert log["totals"]["output_tokens"] == 200
    assert log["totals"]["reasoning_tokens"] == 40
    assert log["usage_complete"] is True
    assert log["status"] == "completed"
    assert "private document" not in json.dumps(log)


def test_local_cache_hit_has_zero_new_tokens(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)
    cached = _response()
    cached.cache_hit = True
    cached.usage = {}
    with (
        dspy.context(lm=lm),
        patch.object(lm, "forward", return_value=cached),
        extraction_usage_log(tmp_path, "paper", "RCT", lm.model),
    ):
        lm("question")
    log = _read_log(tmp_path)
    assert log["calls"][0]["local_cache_hit"] is True
    assert log["totals"]["input_tokens"] == 0
    assert log["totals"]["output_tokens"] == 0
    assert (
        estimate_extraction_cost(log, {"input": 1, "cached_input": 0.1, "output": 5})
        == 0
    )


def test_failed_calls_and_partial_usage_persist(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)
    with (  # noqa: PT012
        dspy.context(lm=lm),
        patch.object(lm, "forward", side_effect=[_response(), RuntimeError("failure")]),
        pytest.raises(RuntimeError, match="failure"),
        extraction_usage_log(tmp_path, "paper", "RCT", lm.model),
    ):
        lm("first")
        lm("second")
    log = _read_log(tmp_path)
    assert log["status"] == "failed"
    assert log["calls"][0]["usage"]["input_tokens"] == 1000
    assert log["calls"][1]["status"] == "failed"
    assert log["calls"][1]["usage"]["input_tokens"] is None
    assert log["totals"]["input_tokens"] is None
    assert log["usage_complete"] is False
    with pytest.raises(ValueError, match="usage is unknown"):
        estimate_extraction_cost(log, {"input": 1, "cached_input": 0.1, "output": 5})


def test_adapter_fallback_records_both_calls(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)
    with (
        dspy.context(lm=lm),
        patch.object(
            lm,
            "forward",
            side_effect=[
                _response(text="unparseable"),
                _response(text='{"answer": "yes"}'),
            ],
        ),
        extraction_usage_log(tmp_path, "paper", "RCT", lm.model),
    ):
        result = dspy.Predict("question -> answer")(question="q")
        assert result.answer == "yes"
    log = _read_log(tmp_path)
    assert len(log["calls"]) == 2
    assert [call["attempt"] for call in log["calls"]] == [1, 2]
    assert log["totals"]["input_tokens"] == 2000


def test_logs_do_not_overwrite_or_leak_callbacks(tmp_path):
    callbacks = list(dspy.settings.callbacks)
    for _ in range(2):
        with extraction_usage_log(tmp_path, "paper", "RCT", "model"):
            pass
    assert len(list(tmp_path.glob("extraction_log_*.json"))) == 2
    assert dspy.settings.callbacks == callbacks


def _priced_log():
    log = ExtractionUsageLog(
        study_name="paper", study_type="RCT", model="model", status="completed"
    )
    log.calls.append(
        UsageCall(
            call_id="1",
            stage="stage",
            attempt=1,
            model="model",
            timestamp="now",
            status="completed",
            local_cache_hit=False,
            usage=TokenUsage(
                input_tokens=1_000_000,
                cached_input_tokens=400_000,
                non_cached_input_tokens=600_000,
                cache_write_input_tokens=100_000,
                output_tokens=200_000,
                reasoning_tokens=50_000,
            ),
        )
    )
    return log


def test_cost_calculation_and_cache_write_rate():
    prices = {"input": 2, "cached_input": 0.2, "output": 10}
    log = _priced_log()
    assert estimate_extraction_cost(log, prices) == pytest.approx(3.28)
    assert estimate_extraction_cost(
        log, {**prices, "cache_write_input": 2.5}
    ) == pytest.approx(3.33)
    log.calls[0].usage.cache_write_input_tokens = None
    with pytest.raises(ValueError, match="requires reported cache-write"):
        estimate_extraction_cost(log, {**prices, "cache_write_input": 2.5})


def test_folder_cost_summary(tmp_path, capsys):
    for name in ("extraction_log_b.json", "extraction_log_a"):
        (tmp_path / name).write_text(_priced_log().model_dump_json(), encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("invalid", encoding="utf-8")
    nested = tmp_path / "extraction_log_subfolder"
    nested.mkdir()
    (nested / "extraction_log_nested.json").write_text("invalid", encoding="utf-8")
    total = summarize_extraction_costs(
        tmp_path, {"input": 2, "cached_input": 0.2, "output": 10}
    )
    assert total == pytest.approx(6.56)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0].split() == ["Name", "Price"]
    assert lines[1].split() == ["extraction_log_a", "3.28000000"]
    assert lines[2].split() == ["extraction_log_b.json", "3.28000000"]
    assert lines[3].split() == ["Overall", "total", "6.56000000"]


@pytest.mark.parametrize("invalid", ["incomplete", "malformed"])
def test_folder_summary_rejects_bad_logs_without_partial_total(
    tmp_path, capsys, invalid
):
    log = _priced_log()
    (tmp_path / "extraction_log_a.json").write_text(
        log.model_dump_json(), encoding="utf-8"
    )
    log.logging_errors.append("callback failed")
    (tmp_path / "extraction_log_b.json").write_text(
        log.model_dump_json() if invalid == "incomplete" else "invalid JSON",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"extraction_log_b\.json"):
        summarize_extraction_costs(
            tmp_path, {"input": 2, "cached_input": 0.2, "output": 10}
        )
    assert capsys.readouterr().out == ""


def test_folder_summary_requires_folder_and_logs(tmp_path):
    prices = {"input": 2, "cached_input": 0.2, "output": 10}
    with pytest.raises(NotADirectoryError, match="not a directory"):
        summarize_extraction_costs(tmp_path / "missing", prices)
    with pytest.raises(FileNotFoundError, match="No files starting"):
        summarize_extraction_costs(tmp_path, prices)


@pytest.mark.parametrize("include_finished", [True, False])
def test_folder_summary_warns_and_skips_unfinished_logs(
    tmp_path, capsys, include_finished
):
    log = _priced_log()
    prices = {"input": 2, "cached_input": 0.2, "output": 10}
    if include_finished:
        (tmp_path / "extraction_log_finished.json").write_text(
            log.model_dump_json(), encoding="utf-8"
        )
    log.status = "running"
    (tmp_path / "extraction_log_unfinished.json").write_text(
        log.model_dump_json(), encoding="utf-8"
    )
    total = summarize_extraction_costs(tmp_path, prices)
    expected = 3.28 if include_finished else 0
    assert total == pytest.approx(expected)
    displayed = capsys.readouterr().out
    assert (
        "Warning: extraction_log_unfinished.json is unfinished; "
        "excluded from the total." in displayed
    )
    assert "Overall total (excluding unfinished logs)" in displayed
    assert displayed.rstrip().endswith(f"{expected:.8f}")
    if include_finished:
        assert "extraction_log_finished.json" in displayed
    with pytest.raises(ValueError, match="unfinished"):
        estimate_extraction_cost(log, prices)


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_invalid_prices_rejected(rate):
    with pytest.raises(ValueError, match=r"greater than or equal|finite"):
        estimate_extraction_cost(
            _priced_log(), {"input": rate, "cached_input": 0.2, "output": 10}
        )


def test_price_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "log.json"
    path.write_text(_priced_log().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "estimate_extraction_cost",
            str(path),
            "--prices",
            '{"input": 2, "cached_input": 0.2, "output": 10}',
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["estimated_cost"] == pytest.approx(3.28)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "study_type",
    [
        "RCT",
        "RCT_pairwise",
        "CochraneRCT",
        "CochraneRCT_pairwise",
        "ObesityRCT",
        "ObesityRCT_pairwise",
        "AnimalRCT",
        "AnimalRCT_pairwise",
        "PrognosticStudy",
        "ClimateCarbonPricing",
        "Programme",
    ],
)
def test_predefined_runners_log_all_study_types(
    tmp_path, monkeypatch, study_type, batch
):
    paper = tmp_path / "paper.md"
    paper.write_text("paper", encoding="utf-8")
    output = tmp_path / "output"
    config = {
        "study_type": study_type,
        "llm_model": "openai/test-model",
        "max_tokens": 100,
        "dspy_cache": False,
        "input_paths": [str(paper)],
        "input_folder": str(tmp_path),
        "output_parent_dir": str(output),
        "export_csv": False,
        "export_xlsx": False,
        "export_json": False,
    }
    lm = dspy.LM("openai/test-model", cache=False)

    def fake_extract(**kwargs):
        lm("question")
        return SimpleNamespace()

    monkeypatch.setattr(main_hierarchical, "configure_lm", lambda *_a, **_k: None)
    monkeypatch.setattr(main_hierarchical, "extract", fake_extract)
    monkeypatch.setattr(main_hierarchical, "save_data", lambda **_k: None)
    monkeypatch.setattr(
        main_hierarchical,
        "load_batch_config" if batch else "load_config",
        lambda _path: config,
    )
    with dspy.context(lm=lm), patch.object(lm, "forward", return_value=_response()):
        if batch:
            main_hierarchical.run_predict_batch("config.json")
        else:
            main_hierarchical.run_predict_single_study("config.json")
    log = _read_log(output)
    assert log["study_type"] == study_type
    assert log["totals"]["input_tokens"] == 1000


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "study_type",
    [
        *main_hierarchical.RCT_PIPELINES,
        "PrognosticStudy",
        "ClimateCarbonPricing",
        "Programme",
    ],
)
def test_custom_runners_log_all_study_types(tmp_path, monkeypatch, study_type, batch):
    paper = tmp_path / "paper.md"
    paper.write_text("paper", encoding="utf-8")
    schema = tmp_path / "schema.csv"
    schema.write_text("placeholder", encoding="utf-8")
    output = tmp_path / "output"
    config = {
        "study_type": study_type,
        "llm_model": "openai/test-model",
        "max_tokens": 100,
        "dspy_cache": False,
        "input_paths": [str(paper)],
        "input_folder": str(tmp_path),
        "output_parent_dir": str(output),
        "export_csv": False,
        "export_xlsx": False,
        "export_json": False,
    }
    lm = dspy.LM("openai/test-model", cache=False)

    class FakePipeline(dspy.Module):
        def forward(self, context):
            lm(context)
            return SimpleNamespace()

    monkeypatch.setattr(custom_hierarchical, "configure_lm", lambda *_a, **_k: None)
    monkeypatch.setattr(custom_hierarchical, "_load_prompt_schema", lambda _p: {})
    monkeypatch.setattr(
        custom_hierarchical,
        "_build_dynamic_pipeline_for_study_type",
        lambda *_a: FakePipeline,
    )
    monkeypatch.setattr(
        custom_hierarchical,
        "_write_dynamic_study_outputs",
        lambda *_a, **_k: output / "study.json",
    )
    monkeypatch.setattr(
        custom_hierarchical,
        "_load_runtime_batch_config" if batch else "_load_runtime_config",
        lambda _path: config,
    )
    with dspy.context(lm=lm), patch.object(lm, "forward", return_value=_response()):
        if batch:
            custom_hierarchical.run_dynamic_batch_extraction_from_csv_schema(
                schema, "config.json"
            )
        else:
            custom_hierarchical.run_dynamic_extraction_from_csv_schema(
                schema, "config.json"
            )
    log = _read_log(output)
    assert log["study_type"] == study_type
    assert log["totals"]["input_tokens"] == 1000


def test_invalid_provider_usage_is_recorded_without_interrupting_extraction(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)
    response = _response(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 200},
        }
    )
    with (
        dspy.context(lm=lm),
        patch.object(lm, "forward", return_value=response),
        extraction_usage_log(tmp_path, "paper", "RCT", lm.model),
    ):
        lm("question")
    log = _read_log(tmp_path)
    assert log["status"] == "completed"
    assert log["logging_errors"]
    assert log["usage_complete"] is False
    with pytest.raises(ValueError, match="logging is incomplete"):
        estimate_extraction_cost(log, {"input": 1, "cached_input": 0.1, "output": 5})


def test_price_file_cli_and_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "log.json"
    path.write_text(_priced_log().model_dump_json(), encoding="utf-8")
    prices = tmp_path / "prices.json"
    prices.write_text(
        '{"input": 2, "cached_input": 0.2, "output": 10}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        ["estimate_extraction_cost", str(path), "--prices-file", str(prices)],
    )
    main()
    assert json.loads(capsys.readouterr().out)["estimated_cost"] == pytest.approx(3.28)
    prices.write_text('{"input": -1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "Cost estimation failed" in capsys.readouterr().err


def test_missing_cache_usage_keeps_log_incomplete(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)
    with (
        dspy.context(lm=lm),
        patch.object(
            lm,
            "forward",
            return_value=_response(
                usage={"prompt_tokens": 100, "completion_tokens": 5}
            ),
        ),
        extraction_usage_log(tmp_path, "paper", "RCT", lm.model),
    ):
        lm("question")
    log = _read_log(tmp_path)
    assert log["status"] == "completed"
    assert log["usage_complete"] is False
    assert log["totals"]["input_tokens"] == 100
    assert log["totals"]["cached_input_tokens"] is None


def test_failed_pipeline_retains_known_usage_and_can_be_priced(tmp_path):
    lm = dspy.LM("openai/test-model", cache=False)

    def run():
        with extraction_usage_log(tmp_path, "paper", "RCT", lm.model):
            lm("question")
            message = "Later extraction validation failed"
            raise RuntimeError(message)

    with (
        dspy.context(lm=lm),
        patch.object(lm, "forward", return_value=_response()),
        pytest.raises(RuntimeError, match="Later extraction validation"),
    ):
        run()
    log = _read_log(tmp_path)
    assert log["status"] == "failed"
    assert log["usage_complete"] is True
    assert estimate_extraction_cost(
        log, {"input": 2, "cached_input": 0.2, "output": 10}
    ) == pytest.approx(0.00228)


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "initial_write",
        "final_write",
        "initial_serialization",
        "final_serialization",
        "aggregation",
        "normalization",
        "callback_start",
        "callback_end",
        "callback_setup",
    ],
)
def test_prediction_files_saved_despite_logging_failure(  # noqa: C901, PLR0912, PLR0915
    tmp_path, monkeypatch, custom, batch, failure
):
    paper = tmp_path / "paper.md"
    paper.write_text("paper", encoding="utf-8")
    schema = tmp_path / "schema.csv"
    schema.write_text("placeholder", encoding="utf-8")
    output = tmp_path / "output"
    config = {
        "study_type": "PrognosticStudy",
        "llm_model": "openai/test-model",
        "max_tokens": 100,
        "dspy_cache": False,
        "input_paths": [str(paper)],
        "input_folder": str(tmp_path),
        "output_parent_dir": str(output),
        "export_csv": False,
        "export_xlsx": False,
        "export_json": True,
    }
    lm = dspy.LM("openai/test-model", cache=False)

    class FakePipeline(dspy.Module):
        def forward(self, context):
            lm(context)
            return SimpleNamespace(model_dump=lambda: {"prediction": "saved"})

    def write_prediction(*_args: object, **_kwargs):
        path = output / "prediction.json"
        path.write_text('{"prediction": "saved"}', encoding="utf-8")
        return path

    module = custom_hierarchical if custom else main_hierarchical
    monkeypatch.setattr(module, "configure_lm", lambda *_a, **_k: None)
    if custom:
        monkeypatch.setattr(module, "_load_prompt_schema", lambda _p: {})
        monkeypatch.setattr(
            module, "_build_dynamic_pipeline_for_study_type", lambda *_a: FakePipeline
        )
        monkeypatch.setattr(module, "_write_dynamic_study_outputs", write_prediction)
        config_loader = (
            "_load_runtime_batch_config" if batch else "_load_runtime_config"
        )
    else:
        monkeypatch.setattr(module, "extract", lambda **_k: FakePipeline()(context="q"))
        config_loader = "load_batch_config" if batch else "load_config"
    monkeypatch.setattr(module, config_loader, lambda _p: config)

    if failure in {"initial_write", "final_write"}:
        original_write = type(output).write_text
        writes = 0

        def fail_write(path, *args: object, **kwargs):
            nonlocal writes
            if path.name.startswith("extraction_log_"):
                writes += 1
                if writes == (1 if failure == "initial_write" else 2):
                    message = "Injected log write failure"
                    raise PermissionError(message)
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(type(output), "write_text", fail_write)
    elif failure in {"initial_serialization", "final_serialization"}:
        original_dump = ExtractionUsageLog.model_dump_json
        dumps = 0

        def fail_dump(self, **kwargs):
            nonlocal dumps
            dumps += 1
            if dumps == (1 if failure == "initial_serialization" else 2):
                message = "Injected serialization failure"
                raise TypeError(message)
            return original_dump(self, **kwargs)

        monkeypatch.setattr(ExtractionUsageLog, "model_dump_json", fail_dump)
    elif failure == "aggregation":
        monkeypatch.setattr(ExtractionUsageLog, "summarize", _raise_logging_error)
    elif failure == "normalization":
        monkeypatch.setattr(
            "deet.hierarchical_mvp.extraction_logging.normalize_usage",
            _raise_logging_error,
        )
    elif failure == "callback_start":
        # Fail inside the guarded callback before a call record can be created.
        monkeypatch.setattr(
            "deet.hierarchical_mvp.extraction_logging.UsageCall", _raise_logging_error
        )
    elif failure == "callback_end":
        original_start = UsageCallback.on_lm_start

        def forget_pending(self, *args: object, **kwargs):
            original_start(self, *args, **kwargs)
            self._pending.clear()

        monkeypatch.setattr(UsageCallback, "on_lm_start", forget_pending)
    else:
        monkeypatch.setattr(UsageCallback, "__init__", _raise_logging_error)

    with dspy.context(lm=lm), patch.object(lm, "forward", return_value=_response()):
        if custom:
            if batch:
                module.run_dynamic_batch_extraction_from_csv_schema(
                    schema, "config.json"
                )
            else:
                module.run_dynamic_extraction_from_csv_schema(schema, "config.json")
        elif batch:
            module.run_predict_batch("config.json")
        else:
            module.run_predict_single_study("config.json")

    predictions = [
        path
        for path in output.glob("*.json")
        if not path.name.startswith("extraction_log_")
    ]
    assert len(predictions) == 1
    assert json.loads(predictions[0].read_text(encoding="utf-8")) == {
        "prediction": "saved"
    }
    # Even failures before call creation must not become a zero-cost estimate.
    with pytest.raises(ValueError, match=r"incomplete|unfinished|unknown"):
        estimate_extraction_cost(
            next(output.glob("extraction_log_*.json")),
            {"input": 1, "cached_input": 0.1, "output": 5},
        )


def _raise_logging_error(*_args: object, **_kwargs):
    message = "Injected token logging failure"
    raise KeyError(message)


@pytest.mark.parametrize("failure", ["aggregation", "write"])
def test_logging_failure_does_not_mask_extraction_exception(
    tmp_path, monkeypatch, failure
):
    if failure == "aggregation":
        monkeypatch.setattr(ExtractionUsageLog, "summarize", _raise_logging_error)
    else:
        monkeypatch.setattr(type(tmp_path), "write_text", _raise_logging_error)

    def run():
        with extraction_usage_log(tmp_path, "paper", "RCT", "model"):
            message = "Original extraction failure"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="Original extraction failure"):
        run()
