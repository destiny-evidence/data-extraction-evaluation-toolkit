"""Three-pass programme extraction with a shared report-containing prompt prefix."""

# ruff: noqa: N999

from __future__ import annotations

from types import GenericAlias
from typing import TYPE_CHECKING, Any, Literal

import dspy
from dspy.utils.exceptions import ContextWindowExceededError
from pydantic import BaseModel, Field, create_model

from .programme_framework import clean_outcome_classifications, load_indicator_framework
from .ProgrammeWithCachingModel import (
    PROGRAMME_WITH_CACHING_INSTRUCTIONS,
    ProgrammeWithCaching,
    ProgrammeWithCachingStep,
)

if TYPE_CHECKING:
    from pathlib import Path

ExtractionStage = Literal["programme", "quantitative", "qualitative"]

STAGE_INPUTS_MARKER = "[[ ## extraction_stage ## ]]"


class _SplitStageInputsMixin:
    """
    End a user message after the report and framework.

    Stage-specific inputs are sent in a separate final message. For OpenAI, the
    shared message also carries an explicit cache breakpoint: implicit caching
    only writes at the end of the latest message, which differs between stages.
    """

    cache_breakpoint: bool = False

    def format(
        self,
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = super().format(  # type: ignore[misc]
            signature, demos, inputs
        )
        last = messages[-1]
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            shared, marker, stage_inputs = last["content"].partition(
                STAGE_INPUTS_MARKER
            )
            if marker and shared.strip():
                shared_content: str | list[dict[str, Any]] = shared.rstrip()
                if self.cache_breakpoint:
                    shared_content = [
                        {
                            "type": "text",
                            "text": shared_content,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ]
                messages[-1:] = [
                    {"role": "user", "content": shared_content},
                    {"role": "user", "content": marker + stage_inputs},
                ]
        return messages


class ProgrammeWithCachingJSONAdapter(_SplitStageInputsMixin, dspy.JSONAdapter):
    """JSONAdapter with the report and framework in their own message."""

    def __init__(self, *, cache_breakpoint: bool = False) -> None:
        """Optionally mark the shared message with an OpenAI cache breakpoint."""
        super().__init__()
        self.cache_breakpoint = cache_breakpoint


class ProgrammeWithCachingChatAdapter(_SplitStageInputsMixin, dspy.ChatAdapter):
    """ChatAdapter whose JSON fallback keeps the cache-friendly message split."""

    def __init__(self, *, cache_breakpoint: bool = False) -> None:
        """Disable DSPy's plain JSONAdapter fallback in favour of the split one."""
        super().__init__(use_json_adapter_fallback=False)
        self.cache_breakpoint = cache_breakpoint

    def __call__(
        self,
        lm: Any,  # noqa: ANN401
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call the LM, retrying parse failures with the split JSON adapter."""
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except ContextWindowExceededError:
            raise
        except Exception:  # noqa: BLE001
            return ProgrammeWithCachingJSONAdapter(
                cache_breakpoint=self.cache_breakpoint
            )(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: Any,  # noqa: ANN401
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Asynchronously call the LM with the split JSON adapter fallback."""
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except ContextWindowExceededError:
            raise
        except Exception:  # noqa: BLE001
            return await ProgrammeWithCachingJSONAdapter(
                cache_breakpoint=self.cache_breakpoint
            ).acall(lm, lm_kwargs, signature, demos, inputs)


def _cache_friendly_adapter() -> dspy.ChatAdapter:
    """Keep a configured JSON adapter choice while splitting stage inputs."""
    lm = dspy.settings.lm
    cache_breakpoint = str(getattr(lm, "model", "")).startswith("openai/")
    if isinstance(dspy.settings.adapter, dspy.JSONAdapter):
        return ProgrammeWithCachingJSONAdapter(cache_breakpoint=cache_breakpoint)
    return ProgrammeWithCachingChatAdapter(cache_breakpoint=cache_breakpoint)


class ExtractProgrammeWithCaching(dspy.Signature):
    """Use a fixed schema and place changing inputs after the report and framework."""

    __doc__ = PROGRAMME_WITH_CACHING_INSTRUCTIONS

    context: str = dspy.InputField(desc="UN programme evaluation report text.")
    classification_framework: str = dspy.InputField(
        desc="CSV category IDs, groups, statements and inclusion/exclusion criteria."
    )
    extraction_stage: ExtractionStage = dspy.InputField(
        desc="The single extraction stage to perform."
    )
    previous_extraction: ProgrammeWithCachingStep = dspy.InputField(
        desc="Previously extracted information; all fields empty in the first stage."
    )
    extraction: ProgrammeWithCachingStep = dspy.OutputField(
        desc="Only the requested stage's results in the shared response envelope."
    )


class ProgrammeWithCachingExtractionPipeline(dspy.Module):
    """Keep instructions, schemas, report and framework identical across stages."""

    programme_model: type[BaseModel] = ProgrammeWithCaching
    step_model: type[ProgrammeWithCachingStep] = ProgrammeWithCachingStep
    extraction_signature: type[dspy.Signature] = ExtractProgrammeWithCaching
    extract_programme_info: dspy.Predict
    extract_quantitative: dspy.Predict
    extract_qualitative: dspy.Predict

    def __init__(self, classification_framework_path: Path | None = None) -> None:
        """Configure the same signature for each of the three named predictors."""
        super().__init__()
        self.classification_framework = load_indicator_framework(
            classification_framework_path
        )
        # Distinct names preserve stage attribution without changing rendered prompts.
        for attribute, signature_name in (
            ("extract_programme_info", "ExtractProgrammeWithCachingInfo"),
            ("extract_quantitative", "ExtractProgrammeWithCachingQuantitative"),
            ("extract_qualitative", "ExtractProgrammeWithCachingQualitative"),
        ):
            signature = type(
                signature_name,
                (self.extraction_signature,),
                {
                    "__doc__": self.extraction_signature.instructions,
                    "__module__": __name__,
                },
            )
            setattr(self, attribute, dspy.Predict(signature))

    def _extract_stage(
        self,
        predictor: dspy.Predict,
        stage: ExtractionStage,
        context: str,
        previous: ProgrammeWithCachingStep,
    ) -> ProgrammeWithCachingStep:
        with dspy.context(adapter=_cache_friendly_adapter()):
            prediction = predictor(
                context=context,
                classification_framework=self.classification_framework,
                extraction_stage=stage,
                previous_extraction=previous,
            )
        result = self.step_model.model_validate(prediction.extraction)
        for field in self.step_model.model_fields:
            expected = (
                field in {"programme_characteristics", "interventions"}
                if stage == "programme"
                else field == f"{stage}_outcomes"
            )
            if not expected and getattr(result, field):
                message = f"The {stage} stage returned unexpected {field}."
                raise ValueError(message)
        if stage == "programme" and result.programme_characteristics is None:
            message = "The programme stage must return programme_characteristics."
            raise ValueError(message)
        return result

    def forward(self, context: str) -> BaseModel:
        """Extract each stage, cleaning numerical results before qualitative use."""
        empty = self.step_model(
            programme_characteristics=None,
            interventions=[],
            quantitative_outcomes=[],
            qualitative_outcomes=[],
        )
        programme = self._extract_stage(
            self.extract_programme_info, "programme", context, empty
        )
        quantitative = self._extract_stage(
            self.extract_quantitative, "quantitative", context, programme
        )
        clean_outcome_classifications(
            quantitative.quantitative_outcomes, self.classification_framework
        )
        previous = programme.model_copy(
            update={"quantitative_outcomes": quantitative.quantitative_outcomes}
        )
        qualitative = self._extract_stage(
            self.extract_qualitative, "qualitative", context, previous
        )
        clean_outcome_classifications(
            qualitative.qualitative_outcomes, self.classification_framework
        )
        return self.programme_model(
            programme_characteristics=programme.programme_characteristics,
            interventions=programme.interventions,
            quantitative_outcomes=quantitative.quantitative_outcomes,
            qualitative_outcomes=qualitative.qualitative_outcomes,
        )


def build_programme_with_caching_pipeline(
    runtime_models: dict[str, type[BaseModel]],
) -> type[ProgrammeWithCachingExtractionPipeline]:
    """Use custom CSV field descriptions without changing schemas between stages."""
    fields = ProgrammeWithCachingStep.model_fields
    step_model = create_model(
        "DynamicProgrammeWithCachingStep",
        __base__=ProgrammeWithCachingStep,
        programme_characteristics=(
            runtime_models["Programme_Characteristics"] | None,
            Field(description=fields["programme_characteristics"].description),
        ),
        interventions=(
            GenericAlias(list, runtime_models["Intervention"]),
            Field(description=fields["interventions"].description),
        ),
        quantitative_outcomes=(
            GenericAlias(list, runtime_models["Quantitative_Outcome"]),
            Field(description=fields["quantitative_outcomes"].description),
        ),
        qualitative_outcomes=(
            GenericAlias(list, runtime_models["Qualitative_Outcome"]),
            Field(description=fields["qualitative_outcomes"].description),
        ),
    )
    signature = ExtractProgrammeWithCaching.with_updated_fields(
        "previous_extraction", type_=step_model
    ).with_updated_fields("extraction", type_=step_model)
    return type(
        "DynamicProgrammeWithCachingExtractionPipeline",
        (ProgrammeWithCachingExtractionPipeline,),
        {
            "programme_model": runtime_models["Programme"],
            "step_model": step_model,
            "extraction_signature": signature,
        },
    )
