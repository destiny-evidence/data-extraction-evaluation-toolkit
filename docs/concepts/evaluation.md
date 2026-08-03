# Evaluation

Evaluation compares LLM-extracted values to <term: Ground Truth> annotations
for the same documents and attributes.
In the CLI this happens at the end of `deet experiments evaluate`, which
writes a set of [experiment artefacts](experiment.md#experiment-artefacts)
under `data-extraction-experiments/<run_id>/`.

Artefact paths are defined by
[`deet.data_models.project.ExperimentArtefacts`](../reference/api.md#deet.data_models.project.ExperimentArtefacts).
Metric registries live in
[`deet.data_models.evaluation`](../reference/api.md#deet.data_models.evaluation).

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
does not write `metrics.csv` or `goldstandard_llm_comparison.csv`.

### `metrics.csv`

One row per attribute × metric. Produced by
[`GoldStandardLLMEvaluator.write_metrics_to_csv`](../reference/api.md#deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator.write_metrics_to_csv)
via [`AttributeMetric`](../reference/api.md#deet.data_models.evaluation.AttributeMetric).

| Column | Meaning |
|--------|---------|
| `attribute_id` | Attribute identifier |
| `attribute_label` | Human-readable attribute name |
| `value` | Metric score, or empty if the metric could not be computed (e.g. incompatible `None` predictions for classification metrics) |
| `extraction_run_id` | Run folder / run id |
| `metric_name` | Name of the metric (see below) |

#### Metrics by attribute type

| Attribute type | Default metrics |
|----------------|-----------------|
| **BOOL** | `accuracy`, `precision`, `recall`, `f1_score`, `n_labels` |
| **STRING** | `accuracy`, `edit_distance_match_rate` |
| **INTEGER** / **FLOAT** | `accuracy`, `mean_absolute_error`, `mean_absolute_percentage_error` |
| **LIST** / **DICT** | No default metrics yet |

- **`accuracy`**: fraction of exact matches between gold and LLM `output_data`.
- **`precision` / `recall` / `f1_score`**: standard binary classification metrics
  (BOOL attributes).
- **`n_labels`**: count of positive gold labels for the attribute (BOOL).
- **`edit_distance_match_rate`**: fraction of pairs whose normalised Levenshtein
  similarity is at least a configurable threshold (default `0.90`). Threshold
  can be set as `edit_distance_match_threshold` in the extraction config YAML.
- **`mean_absolute_error` / `mean_absolute_percentage_error`**: magnitude of
  numeric error on coerced float values. These complement exact-match accuracy;
  they do not replace it. Pairs with `None` LLM predictions are dropped for
  these regression-style metrics. MAPE skips pairs where gold is zero and logs
  a warning.

You can also pass extra sklearn metric names with
`--custom-evaluation-metrics` on `deet experiments evaluate`.

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
   and inspect citations / verbatim fuzzy scores when diagnosing grounding.
3. Use **`llm_annotations.json`** and **`extraction_metadata.json`** for full
   model payloads and cost/timing.

For how artefacts fit into the wider experiment workflow, see
[Data Extraction Experiment](experiment.md).
