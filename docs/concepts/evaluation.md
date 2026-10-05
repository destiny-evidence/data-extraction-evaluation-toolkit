# Evaluation

## Comparing human and AI-extracted data

To evaluate an [experiment](experiment.md), deet compares compares LLM-extracted values to <term: Ground Truth> annotations
for the same documents and attributes.
In the CLI this happens at the end of `deet experiments evaluate`, which
writes a set of [experiment artefacts](experiment.md#experiment-artefacts)
under `data-extraction-experiments/<run_id>/`.

Artefact paths are defined by
[`deet.data_models.project.ExperimentArtefacts`](../reference/api.md#deet.data_models.project.ExperimentArtefacts).
Metric registries live in
[`deet.evaluators.metrics`](../reference/api.md#deet.evaluators.metrics).

## What is compared

For each attribute in the prompt CSV (non-empty `prompt`), and for each gold
document, DEET collects:

- **Gold** `output_data` from the gold-standard annotation (or the attribute's
  missing-annotation default when the attribute is absent for that document)
- **LLM** `output_data` from the model annotation for the same document and
  attribute (`None` if the document failed extraction or produced no usable
  annotation)

Those parallel lists are scored with the metrics registered for the attribute's
[`AttributeType`](../reference/api.md#deet.data_models.base.AttributeType).

## Output artefacts

Each successful `deet experiments evaluate` run creates a new folder and writes
the files below. `deet experiments predict` extracts without scoring, so it
does not write `metrics.csv`, `metrics.json`, or `goldstandard_llm_comparison.csv`.

### `metrics.csv`

Wide metrics table (one row per attribute, metric names as columns). Produced by
[`GoldStandardLLMEvaluator.write_metrics_to_csv`](../reference/api.md#deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator.write_metrics_to_csv)
via [`AttributeMetric`](../reference/api.md#deet.data_models.evaluation.AttributeMetric).

| Column | Meaning |
|--------|---------|
| `extraction_run_id` | Run folder / run id |
| `attribute_id` | Attribute identifier |
| `attribute_label` | Human-readable attribute name |
| `attribute_type` | Output type (`BOOL`, `STRING`, `INTEGER`, `FLOAT`, etc.) |
| `n_gold_instances` | Number of document-attribute instances scored for this attribute |
| `n_good_source_instances` | Number of instances where the gold value is found in parsed `context` (STRING/INTEGER/FLOAT only) |
| `n_good_citation_instances` | Number of instances where the gold value is found in citation text (`additional_text` and/or `item_attribute_full_text_details`) (STRING/INTEGER/FLOAT only) |
| metric columns (e.g. `accuracy`, `precision`, `mean_absolute_error`) | Score value, or empty if not applicable / not computable |

#### Metrics by attribute type

| Attribute type | Default metrics |
|----------------|-----------------|
| **BOOL** | `accuracy`, `precision`, `recall`, `f1_score`, `n_labels` |
| **STRING** | `accuracy`, `edit_distance_match_rate`, plus stratified variants `*_given_good_source` and `*_given_bad_source` |
| **INTEGER** / **FLOAT** | `accuracy`, `mean_absolute_error`, `mean_absolute_percentage_error`, plus stratified variants `*_given_good_source` and `*_given_bad_source` |
| **LIST** / **DICT** | No default metrics yet |

- **`accuracy`**: fraction of exact matches between gold and LLM `output_data`.
- **`precision` / `recall` / `f1_score`**: standard binary classification metrics
  (BOOL attributes).
- **`n_labels`**: count of positive gold labels for the attribute (BOOL).
- **`edit_distance_match_rate`**: fraction of pairs whose normalised Levenshtein
  similarity is at least a configurable threshold (default `0.90`). Threshold
  can be set as `edit_distance_match_threshold` in the extraction config YAML.
- **`mean_absolute_error` / `mean_absolute_percentage_error`**: magnitude of
  numeric error. These complement exact-match accuracy; they do not replace it.
  Missing or invalid LLM predictions (e.g. failed document extraction or
  duplicate annotations → `None`) cause the metric to fail for that attribute,
  as with binary metrics — the CSV cell is left empty rather than scoring
  only the successful subset. MAPE is undefined when a gold value is zero
  (sklearn behaviour); that also leaves the cell empty.

For STRING / INTEGER / FLOAT, metrics are exported in three views:

- **Unconditional** (all scored instances), e.g. `accuracy`
- **Given good source** (`gold value in context`), e.g. `accuracy_given_good_source`
- **Given bad source** (`gold value not in context`), e.g. `accuracy_given_bad_source`

When a stratified subset is empty (for example every instance is good-source, so
there are no bad-source rows), those `*_given_*` cells are left **empty**. Score
metrics raise on empty inputs (as with sklearn), and the evaluator records
`None` rather than a misleading `0`.

You can also pass extra sklearn metric names with
`--custom-evaluation-metrics` on `deet experiments evaluate`.

### `metrics.json`

Machine-readable metric export with the same values as `metrics.csv`, grouped by
attribute. Produced by
[`GoldStandardLLMEvaluator.write_metrics_to_json`](../reference/api.md#deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator.write_metrics_to_json).

Top-level fields:

- `extraction_run_id`
- `format_version`
- `attributes` (list)

The file is serialised and de-serialised by
[`RunMetricsReport`](../reference/api.md#deet.data_models.evaluation.RunMetricsReport)
(one [`AttributeMetricsReport`](../reference/api.md#deet.data_models.evaluation.AttributeMetricsReport)
per attribute). `attribute_type` reuses
[`AttributeType`](../reference/api.md#deet.data_models.base.AttributeType).
Inapplicable keys are omitted (not written as `null`).

Each attribute object includes:

- `attribute_id`, `attribute_label`, `attribute_type`
- `counts` (e.g. `n_gold_instances`, `n_good_source_instances`)
- `metrics` (e.g. `accuracy`, `accuracy_given_good_source`, etc.)

### `goldstandard_llm_comparison.csv`

Side-by-side gold vs LLM values for every document × attribute pair evaluated.
This is the main file for debugging failures and inspecting citations /
verbatim grounding. Written by
[`GoldStandardLLMEvaluator.export_llm_comparison`](../reference/api.md#deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator.export_llm_comparison).

#### Identifiers

| Column | Meaning |
|--------|---------|
| `document_id` | Internal document id |
| `external_id` | External identifier when available (e.g. from the gold import) |
| `document_name` | Document title / name |
| `attribute_id` | Attribute identifier |
| `attribute_label` | Human-readable attribute name |
| `extraction_run_id` | Run id |

#### Presence and gold support text

| Column | Meaning |
|--------|---------|
| `attribute_presence` | <term: attribute presence>: `"True"` if a gold annotation for this attribute exists on the document, otherwise `"False"` |
| `human_additional_text` | Gold verbatim / supporting text from the annotation (`additional_text`), when present |
| `item_attribute_full_text_details` | Raw EPPI full-text detail string(s) joined for the cell (empty for non-EPPI gold) |

#### EPPI citation fields

| Column | Meaning |
|--------|---------|
| `citation_page` | <term: citation page>: page number(s) parsed from EPPI citation markup |
| `citation_highlight_text` | <term: citation highlight>: highlight text extracted from EPPI markup after cleaning |

!!! note "Related work"
    `citation_page` and `citation_highlight_text` are added with EPPI citation
    parsing / text-normalisation work. Older comparison CSVs may omit them;
    raw `item_attribute_full_text_details` remains available either way.

#### Extractions and LLM extras

| Column | Meaning |
|--------|---------|
| `human_extraction` | Gold `output_data` used for scoring |
| `llm_extraction` | LLM `output_data` (may be empty if extraction failed for the document) |
| `llm_reasoning` | Model reasoning text, or an explanatory message when no LLM annotation was produced |
| `llm_verbatim_text` | Model `additional_text` (verbatim / citation-style support text) |

#### Verbatim grounding scores

| Column | Meaning |
|--------|---------|
| `human_verbatim_fuzzy_match_pct` | <term: verbatim fuzzy match>: how well `human_additional_text` is grounded in the document context used for extraction (0–100) |
| `llm_verbatim_fuzzy_match_pct` | Same score for `llm_verbatim_text` against the same context |

Scores measure how well the snippet appears to be grounded in the document
text used for extraction. Empty snippets score `0.00`. `human_additional_text`
comes from the gold annotation's supporting / verbatim text (often from EPPI);
`llm_verbatim_text` is the model's corresponding support text.

#### Source-fidelity and match status fields

| Column | Meaning |
|--------|---------|
| `gold_value_in_citation` | `True` / `False` for STRING/INTEGER/FLOAT: whether the gold value is found in citation text (`additional_text` and/or `item_attribute_full_text_details`) |
| `gold_value_in_context` | `True` / `False` for STRING/INTEGER/FLOAT: whether the gold value is found in parsed `context` (what the LLM read) |
| `match_status` | Row-level outcome label for STRING/INTEGER/FLOAT (see below) |

`match_status` values:

| Value | Meaning |
|-------|---------|
| `exact_match` | LLM `output_data` equals gold `output_data` |
| `near_match` | STRING only: not exact, but normalised Levenshtein similarity is at least the edit-distance threshold (default `0.90`) |
| `missing_prediction` | No usable LLM value (`None`) |
| `extraction_error_bad_source` | Mismatch, and gold was **not** found in parsed context |
| `extraction_error_good_source` | Mismatch, and gold **was** found in parsed context |

BOOL rows leave these fields empty (source-fidelity checks are out of scope for
bool values).

For INTEGER / FLOAT source checks, numeric tokens are parsed from free text
without thousand separators (e.g. gold `1000` is not matched to `"1,000"`),
because comma forms are locale-ambiguous.

### `llm_annotations.json`

Full structured LLM output for the run: annotated documents (including
document context and per-attribute annotations with `output_data`, reasoning,
and additional text) plus run metadata. This is the machine-readable record
of what the model returned.

`deet experiments predict` also writes this file (without evaluation CSVs).
A flatter CSV export of LLM rows may be written as `llm_annotations.csv` on
predict (`ExperimentArtefacts.llm_annotation_csv`).

### `config.yaml` and `prompts_used.csv`

Snapshots of the <term: experiment configuration> and the prompt CSV actually
used for the run. Use these to reproduce or compare runs.

- `config.yaml` — model, provider, temperature, context options, etc.
  (`ExperimentArtefacts.config_snapshot`)
- `prompts_used.csv` — attributes and prompts retained for the run
  (`ExperimentArtefacts.prompts_snapshot`)

### `extraction_metadata.json`

Run-level cost, token, and timing summary written after extraction:

- Totals such as `total_cost_usd`, `total_input_tokens`, `total_output_tokens`,
  `total_pipeline_duration_seconds`
- `stage_durations_seconds` for pipeline stages (annotation conversion, prompt
  population, document preparation, LLM extraction, artefact export)
- `per_document` entries with per-document token counts and timings
  (parsing skip flags, `llm_call_seconds`, etc.)

See
[`ExtractionRunMetadata`](../reference/api.md#deet.data_models.extraction.ExtractionRunMetadata)
and related models in `deet.data_models.extraction`.

### `deet.log`

A copy of log output for the run directory (useful when investigating failed
documents or metric warnings).

## Reading results

1. Start with **`metrics.csv`** (or the CLI metrics table) to see which
   attributes scored well.
2. Open **`goldstandard_llm_comparison.csv`** for mismatches: compare
   `human_extraction` vs `llm_extraction`, check <term: attribute presence>,
   <term: match status>, citation/context source-fidelity flags, and verbatim
   fuzzy scores.
3. Use **`metrics.json`** when consuming metrics programmatically.
4. Use **`llm_annotations.json`** and **`extraction_metadata.json`** for full
   model payloads and cost/timing.

For how artefacts fit into the wider experiment workflow, see
[Data Extraction Experiment](experiment.md).

## Good evaluation practice

Deet is designed to encourage good evaluation practices.

This means separating data used to develop a pipeline (e.g. by engineering prompts or selected best-performing models) from data used to evaluate a pipeline.

### Evaluation strategies

To do this, deet implements evaluation strategies to manage how data is used for evaluation, in order to maximise efficiency while avoiding leaking.

Evaluation strategies are in general managed by running

```sh
deet experiments splits
```

which will trigger an interactive wizard that prompts you with options managing your evaluation.

#### The null evaluation strategy

By default, the null evaluation strategy is selected for new projects. With the null evaluation strategy, data extraction with deet is run on all documents within a [project](project.md). This is appropriate if you have separated your evaluation data manually into a separate "evaluation" project, or if you just want to test something informally. In this case running `deet experiments splits` will simply print a message informing you that the current strategy has no further options.

#### The dev-val-test strategy

The dev-val-test strategy explicitly handles the separation of data used for training and evaluation, by creating three splits of the data ([more info](https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data)). Assigning documents to those splits is managed by running

```sh
deet experiments splits
```

The first time this is run, the only available option will be to assign documents to the development set. These documents will be used to iteratively try out and improve upon prompts and other experiment configuration options. Each time you run `deet experiments evaluate`, you will see performance metrics for the chosen experiment configuration.

Once you have reached an experiment configuration which achieves performance you are happy with, you can run `deet experiments splits` again to re-enter the wizard. You will now see an option to "Move to validation". Selecting this will prompt you to select a number of documents to be used for validation, that is, for a first out-of-sample test of your chosen configuration.

If you choose to move to validation, you will be prompted to choose the model configuration you wish to validate. The prompts and other settings from this model will then be used to extract data from the documents in the newly created validation set, and metrics will be reported.

Users are then prompted to choose between:

- Accepting the configuration and doing a final test of the pipeline on all remaining documents; and
- Rejecting the configuration (because performance was dissappointing), and moving back to prompt development.

If the former option is selected, the project is finished, and the results are to be seen as the final evaluation scores.

If the latter option is selected, documents from the validation set are passed to the development set, and you are invited to continue iterating prompts.
