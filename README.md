# data-extraction-evaluation-toolkit

The Data Extraction and Evaluation Toolkit (DEET) is a suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

[Docs](https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/)

## .env when using hierarchical extraction

For Azure OpenAI models (model names beginning with `azure/`), use:

```
AZURE_API_KEY=yourkeyhere
AZURE_API_BASE=https://yourprojecthere.openai.azure.com/
```

For standard OpenAI models (model names beginning with `openai/`), use:

```
OPENAI_API_KEY=yourkeyhere
# Optional, for an OpenAI-compatible custom endpoint:
OPENAI_BASE=https://api.openai.com/v1
```

When using OpenAI's default endpoint, omit `OPENAI_BASE` or leave it empty. For
example, set `"llm_model": "openai/gpt-5"` in extraction configs or pass
`llm_model="openai/gpt-5"` to `evaluate_fields`.

You can also use an Anthropic endpoint, by adding AZURE_API_BASE_ANTHROPIC:

```
AZURE_API_KEY=yourkeyhere
AZURE_API_BASE_ANTHROPIC=https://yourprojecthere.services.ai.azure.com/anthropic
```

## Extraction token logs and cost estimates

The predefined and custom CSV-schema hierarchical runners automatically write
one `extraction_log_<study_name>_<run_id>.json` in `output_parent_dir` per study
run, including batch runs and failed extractions. Logging is independent of the
CSV/XLSX/JSON export switches and supports every implemented hierarchical study
type, including pairwise variants and Programme.

Each log contains per-LM-call records attributed to the predictor signature,
attempt numbers, requested/returned models, timestamps, status, raw provider
usage, and normalized token totals:

- `input_tokens`: total input, including cached reads and cache writes.
- `cached_input_tokens`: provider cache-read tokens, a subset of input.
- `non_cached_input_tokens`: input minus cache reads; includes cache writes.
- `cache_write_input_tokens`: cache-write tokens, when exposed by the provider.
- `output_tokens`: total output; reasoning tokens are not added a second time.
- `reasoning_tokens`: the reasoning subset, when exposed.

Token logging is best-effort: logging initialization, callback, aggregation,
serialization, or file-write errors are reported but do not prevent extraction
or prediction export. Genuine extraction failures still propagate normally.
When a log can be saved, accounting errors appear in `logging_errors` and mark
`usage_complete` as false. If log persistence fails, a log may be absent or
unfinished; check the application error messages.

The counts are under `calls[].usage` and `totals`. Missing counts remain `null`;
a total is `null` if any call lacks that count. `usage_complete` indicates whether
input, cached-read, non-cached-input, and output totals are all known.
DSPy local response-cache hits have `local_cache_hit: true` and zero new tokens;
they are distinct from discounted provider cache reads. Logs contain no prompt
or document text. Adapter fallback calls are included, but provider-internal
retries and failures that return no usage cannot be fully accounted for.
Direct pipeline calls outside these runners can opt in using the
`extraction_usage_log` context manager in
[`extraction_logging.py`](deet/hierarchical_mvp/extraction_logging.py).

Estimate the price using your deployment's rates **per million tokens**, all in
the same currency. For example, save this illustrative dictionary in a JSON
file (these are not current model prices):

```json
{"input": 2.0, "cached_input": 0.2, "output": 10.0}
```

Then run:

```powershell
python -m deet.estimate_extraction_cost "C:\outputs\extraction_log_example_RUNID.json" --prices-file "C:\config\token_prices.json"
```

Alternatively, pass a JSON dictionary with `--prices`, or call from Python:

```python
from deet.estimate_extraction_cost import estimate_extraction_cost

price = estimate_extraction_cost(
    r"C:\outputs\extraction_log_example_RUNID.json",
    {"input": 2.0, "cached_input": 0.2, "output": 10.0},
)
```

To display each log's filename and estimated price plus an overall total:

```python
from deet.estimate_extraction_cost import summarize_extraction_costs

total = summarize_extraction_costs(
    r"C:\outputs",
    {"input": 2.0, "cached_input": 0.2, "output": 10.0},
)
```

This scans files starting with `extraction_log` directly in the folder (not
subfolders), displays prices to eight decimal places, and returns the numeric
total. All files use the same supplied pricing schedule. Unfinished logs
(`status: running`) print a warning naming the file and are excluded from the
total; the other records are still displayed and priced. If all logs are
unfinished, the returned total is zero and explicitly excludes those logs.
Other invalid logs or logs with incomplete usage raise an error naming the file
before any table or total is displayed; missing folders and folders with no
matching files also raise errors.

The single-log function returns a numeric estimate; the CLI prints
`{"estimated_cost": ...}`. Both reject unknown required counts, unfinished logs,
logs with accounting errors, and invalid rates. Failed runs can be priced only
when all their call usage is known. Rates apply to all recorded calls, so use a single pricing schedule per
log. No prices are fetched or inferred from model/deployment names.

For deployments with separate cache-write pricing, add `cache_write_input`:
this is the **full write rate**, not an additional surcharge. It replaces the
ordinary input rate for reported write tokens and requires known write counts.
Without that key, writes use the ordinary input rate and a warning states that
additional cache-write charges are excluded. Mixed cache-write lifetimes/rates,
taxes, and other non-token charges are not modeled; estimates are not invoices.

## Implemented Study Types

The pre-defined hierarchical extraction pipeline (`deet/main_hierarchical.py`) supports the following values for `study_type` in its config files:

- `RCT`
- `CochraneRCT`
- `PrognosticStudy`
- `ObesityRCT`
- `AnimalRCT`
- `ClimateCarbonPricing`

The RCT types also have preserved pairwise variants: `RCT_pairwise`,
`CochraneRCT_pairwise`, `ObesityRCT_pairwise`, and `AnimalRCT_pairwise`.
The dynamic, CSV-schema-driven pipeline (`deet/custom_hierarchical.py`) supports
the same study types, including the pairwise variants and the distinct animal
induction/assessment intervention structure.

### One arm per RCT outcome

`RCT`, `CochraneRCT`, `ObesityRCT`, and `AnimalRCT` extract one intervention arm
per outcome object. Create a separate object for each reported outcome, time
point, and group, including controls and baseline results. A three-arm trial
reporting MADRS at one follow-up therefore produces three continuous outcome
objects, not one object containing multiple arms or repeated pairwise contrasts.
Each has a `group_label` identifying its intervention and only that group's
result fields (for example, `group_N`, `group_summary_statistics`, and
`group_variability_statistics`). Animal outcomes retain the induction subgroup.
Cochrane outcomes produce one RevMan row per object.
No unreported arm results are invented from a
between-arm estimate.

JSON keeps each arm in a separate list entry; CSV and XLSX exports keep a
separate row per arm. Cochrane post-processing supports both formats.
To continue using two-arm outcomes, select a `_pairwise` study type.
Their models and extraction modules are retained with the same suffix; continuous
statistics use the named-statistic representation described below in all variants.
Existing pairwise JSON is not automatically converted into single-arm records.
Regenerate prompt CSVs for the selected study type. Custom single-arm
pipelines reject old two-arm prompt schemas instead of silently mixing formats.

For all RCT types and their `_pairwise` variants, baseline outcome measurements
belong in separate outcome objects, including measurements reported in baseline-characteristics
tables. Do not substitute baseline numbers in `supplementary_info` or
`baseline_imbalances` for structured baseline results. The latter field is for
explicitly reported imbalances or assessments of baseline comparability, not
routine baseline measurements. For example, a three-arm `RCT` reporting baseline
MADRS mean (SE) and Week 8 mean change (SE) should produce six continuous outcome
objects: three baseline and three follow-up. `RCT_pairwise` retains two groups
per object, with both groups measured at the same time point; baseline and
follow-up still require separate objects. Do not infer baseline values or copy
follow-up sample sizes or imputation methods unless explicitly applicable.
Animal outcomes preserve the induction/assessment combination and whether the
reported baseline is before induction or before assessment treatment. Cochrane
schemas identify the object's time point in `footnotes`, while keeping numerical
baseline results in structured result fields, not in notes.
Regenerate existing custom prompt CSVs to pick up revised field descriptions,
then reapply any study-specific edits.

### Named continuous statistics

All `Continuous_Outcome` models, including the `_pairwise` variants, extract
lists of named summary and variability statistics instead of fixed mean/SD
fields. This preserves studies reporting medians, geometric means, within-arm
changes, standard errors, interquartile ranges, ranges, confidence intervals,
or multiple measures together.

For single-arm outcomes (and the animal induction/assessment structure):

```json
{
  "group_summary_statistics": [
    {"measure": "Median", "value": "12", "basis": "Endpoint", "details": ""}
  ],
  "group_variability_statistics": [
    {"measure": "Interquartile range", "value": "8-17", "basis": "Endpoint", "details": "Q1-Q3 for median"}
  ]
}
```

Two-arm human RCT variants use `group_a_summary_statistics`,
`group_b_summary_statistics`, `group_a_variability_statistics`, and
`group_b_variability_statistics`. Animal variants retain their induction and
assessment metadata and use the single-group statistic lists.

Each statistic has string fields `measure`, `value`, `basis`, and `details`.
For non-pairwise study types, each list contains at most one statistic. Additional
statistic sets must be separate outcome objects (and spreadsheet rows), retaining
the same arm, outcome, time point, and other metadata. Pair a summary only with
its reported variability measure; if the same summary has both SD and a confidence
interval, repeat that summary on separate rows for the two variability measures.
Unreported summaries or variability measures remain empty lists. These limits
also apply to custom-prompt pipelines.
Suggested measure names are prompt guidance, not enums: edit the
`ReportedStatistic` rows in generated prompt CSVs to customize them.
`basis` distinguishes endpoint values from change-from-baseline statistics;
`details` records confidence levels, the corresponding summary measure,
adjustment, or other reported context. Preserve values and interval bounds as
reported, without converting or calculating missing statistics. Use an empty
list when no statistic is reported. A between-arm mean difference is not an
arm-level average; preserve it in `effect_estimates` (or Cochrane `footnotes`),
not as a group's summary statistic.

JSON preserves these lists as objects. Non-pairwise CSV/XLSX exports have exactly
four `group_summary_statistics_*` columns (`measure`, `value`, `basis`, `details`)
and four corresponding `group_variability_statistics_*` columns, without numbered
blocks. Pairwise exports retain indexed columns for all statistics.
Existing non-pairwise JSON containing multiple statistics per outcome must be
re-extracted or explicitly migrated to separate outcome objects; it is rejected
rather than silently truncating values or guessing associations.
Cochrane exports add these statistic columns alongside the RevMan columns. They populate
`Mean`, `SD`, `SE`, and `Variance` only for an unambiguous matching endpoint
statistic with no additional context. Medians, change scores, confidence
intervals, adjusted estimates, unknown bases, and ambiguous repeated measures
remain in the full lists rather than being coerced into those numeric columns.
The fixed `CI level`, `CI start`, and `CI end` columns remain blank for the
new representation; interval bounds and confidence levels are preserved in
the statistic's `value` and `details`. When importing into software requiring
the exact RevMan template, review/map the full lists and remove supplemental
columns as appropriate.

This is a schema change: regenerate prompt CSVs and migrate existing continuous
outcome JSON/gold standards before reuse. Old mean/SD prompt CSVs are explicitly
rejected rather than silently used. Existing JSON is not automatically migrated.
The standalone Cochrane postprocessor still supports legacy two-arm JSON.

## Hierarchical RCT extraction (pre-defined schema)

DEET includes an experimental hierarchical extraction workflow that uses pre-defined schemas for the study types mentioned above. This workflow is config-driven and reads one or more markdown inputs, runs the hierarchical extraction pipeline, and writes structured outputs.

The CLI of `deet/main_hierarchical.py` has three subcommands: `parse_pdfs`, `predict_single_study`, and `predict_batch`. There are two equivalent ways to run each of them:

- **Terminal / CLI**: run as a module with `python -m` from the repository root (with your venv activated).
- **Python script**: paste the snippet into a `.py` file, or use `main_hierarchical_demo.py` in the repository root, which contains a runnable version of every command below with editable config dicts.

Both columns below do exactly the same thing, so pick whichever fits your workflow.

### Converting source documents to markdown

`parse_pdfs` converts every supported, non-markdown file directly inside a folder (non-recursive) into a sibling `.md` file, using the parsers defined in `deet/processors/parser.py` (pdf, epub, html, xml). If a `.md` file with the target name already exists, that file is skipped. Parse failures are logged and skipped so the rest of the folder is still processed.

Note: pandoc is required for non-PDF files. 

#### Option 1)  **Terminal (CLI)**

~~~sh
python -m deet.main_hierarchical parse_pdfs "misc\hierarchical_mvp\input\batch_pdfs"
~~~

#### Option 2)  **Python script**

~~~python
from deet.main_hierarchical import run_parse_pdfs, setup_console_logging

setup_console_logging()
run_parse_pdfs("misc/hierarchical_mvp/input/batch_pdfs")
~~~

### Predicting a single study

Example `demo_single_config.json`:

~~~json
{
	"study_type": "RCT",
	"llm_model": "anthropic/claude-sonnet-5",
	"max_tokens": 60000,
	"dspy_cache": false,
	"input_paths": [
		"misc/hierarchical_mvp/input/batch_pdfs/mira_rct.md"
	],
	"output_parent_dir": "misc/hierarchical_mvp/output/mira_rct",
	"export_csv": false,
	"export_xlsx": true,
	"export_json": false
}
~~~

You just need to edit the in/output paths and increase max_tokens if needed. If you have provided AZURE_API_BASE_ANTHROPIC for Anthropic, you can use any deployed Anthropic model as `"llm_model"` parameter. 

Example Anthropic model parameters for this file are: `"anthropic/claude-sonnet-5"`, `"anthropic/claude-opus-5"`

With Azure OpenAI credentials, example model names are `"azure/gpt-5.6-luna"`
and `"azure/gpt-5.6-terra"`. With standard OpenAI credentials, use an
`openai/` model name such as `"openai/gpt-5"`.

Run from the repository root:

#### Option 1)  **Terminal (CLI)**

Save the JSON above to `misc/hierarchical_mvp/configs/demo_single_config.json`, then:

~~~sh
python -m deet.main_hierarchical predict_single_study "misc\hierarchical_mvp\configs\demo_single_config.json"
~~~

#### Option 2)  **Python script**

~~~python
import json
from pathlib import Path

from deet.main_hierarchical import run_predict_single_study, setup_console_logging

setup_console_logging()

single_study_config = {
	"study_type": "RCT",
	"llm_model": "anthropic/claude-sonnet-5",
	"max_tokens": 60000,
	"dspy_cache": False,
	"input_paths": ["misc/hierarchical_mvp/input/batch_pdfs/mira_rct.md"],
	"output_parent_dir": "misc/hierarchical_mvp/output/mira_rct",
	"export_csv": False,
	"export_xlsx": True,
	"export_json": False,
}

config_path = Path("misc/hierarchical_mvp/configs/demo_single_config.json")
config_path.write_text(json.dumps(single_study_config, indent=2), encoding="utf-8")

run_predict_single_study(str(config_path))
~~~

Output summary:

- If `"export_json": True`: A timestamped JSON file is written to output_parent_dir, for example: mira_rct_YYYYMMDD_HHMMSS.json
- If `"export_csv": True`: CSV files are written under output_parent_dir/<study_name>/:
	- study_YYYYMMDD_HHMMSS.csv
	- interventions_YYYYMMDD_HHMMSS.csv
	- outcomes_YYYYMMDD_HHMMSS.csv
- If `"export_xlsx": True`: Excel (.xlsx) single file under output_parent_dir/<study_name> in the format study_YYYYMMDD_HHMMSS_<model>.xlsx.
- The all outputs will contain study characteristics, intervention arms, and extracted outcomes.

### Predicting a batch of studies

`predict_batch` runs the same extraction pipeline for every markdown file found directly inside `input_folder` (non-recursive), treating each file as its own study. It uses a `batch_config.json` built exactly like `hierarchical_config.json`, except `input_paths` is replaced with a single `input_folder` path:

~~~json
{
	"study_type": "RCT",
	"llm_model": "anthropic/claude-sonnet-5",
	"max_tokens": 60000,
	"dspy_cache": false,
	"input_folder": "misc/hierarchical_mvp/input/batch_pdfs",
	"output_parent_dir": "misc/hierarchical_mvp/output/batch_demo",
	"export_csv": false,
	"export_xlsx": true,
	"export_json": false
}
~~~


#### Option 1)  **Terminal (CLI)**

Save the JSON above to `misc/hierarchical_mvp/configs/demo_batch_config.json`, then:

~~~sh
python -m deet.main_hierarchical predict_batch "misc\hierarchical_mvp\configs\demo_batch_config.json"
~~~

#### Option 2)  **Python script**

~~~python
import json
from pathlib import Path

from deet.main_hierarchical import run_predict_batch, setup_console_logging

setup_console_logging()

batch_config = {
	"study_type": "RCT",
	"llm_model": "anthropic/claude-sonnet-5",
	"max_tokens": 60000,
	"dspy_cache": False,
	"input_folder": "misc/hierarchical_mvp/input/batch_pdfs",
	"output_parent_dir": "misc/hierarchical_mvp/output/batch_demo",
	"export_csv": False,
	"export_xlsx": True,
	"export_json": False,
}

config_path = Path("misc/hierarchical_mvp/configs/demo_batch_config.json")
config_path.write_text(json.dumps(batch_config, indent=2), encoding="utf-8")

run_predict_batch(str(config_path))
~~~

Unlike `predict_single_study`, outputs are written directly into `output_parent_dir` (no per-study subfolder), with the source markdown filename embedded in each output filename, for example: `misc/hierarchical_mvp/input/projectX/Abdullah_2005.md` produces `study_Abdullah_2005_YYYYMMDD_HHMMSS_<model>.xlsx` (and equivalently named `.csv`/`.json` files) in `output_parent_dir`. If processing one file fails, it is logged and the batch continues with the remaining files.

## Customisable hierarchical study extraction (dynamic)

This section describes an experimental dynamic hierarchical extraction workflow where the extraction schema is built from a prompt CSV at runtime. This has not been 100% tested and is a more rough draft, but it does work to edit/add extraction fields within classes in an existing study type schema.

Step 1: create the prompt CSV from a specified model schema. This is a useful first step because it gives the user something to edit and play with. Note: `--study-type RCT` is an optional parameter and default, but you can use any of the existing study_type values to export the csv, e.g. `--study-type ClimateCarbonPricing`. 

~~~sh
python -m deet.custom_hierarchical write_hierarchical_prompts_csv --study-type RCT
~~~

This generates a CSV with columns class, attribute, prompt, and datatype. You can review and edit prompts in this file before extraction.

For `RCT` and `PrognosticStudy`, outcomes use an `OutcomeTimePoint` object
containing `time_point_category`, `time_point_detail`, `time_point_value`, and
`time_point_unit`. The value is text, so reported ranges such as `6-12` are
preserved. The unit is a plain string, with the prompt suggesting `seconds`,
`minutes`, `hours`, `days`, `weeks`, `months`, `years`, or `NR`. Users can change
these options by editing the prompt CSV without changing the datatype.
Unreported values and units should use `NR`.
JSON exports retain the nested object; CSV and XLSX exports expose the value and
unit as separate columns. Static exports retain `outcome_time_point` (category)
and `outcome_time_point_detail`; custom exports flatten these as
`time_point_category` and `time_point_detail`.

Regenerate existing RCT and prognostic prompt CSVs to include the new fields.
The generated unit datatype is `str`. Prognostic JSON now uses the structured object rather
than a free-text `outcome_time_point` string; existing JSON inputs must be
updated to that structure.

`deet/custom_hierarchical.py` also exposes `parse_pdfs`, `predict_single_study`, and `predict_batch` subcommands, mirroring `main_hierarchical.py` but taking the prompt CSV as an additional argument.

Step 2: run dynamic extraction using both the config file and the prompt CSV.

You can use the python script `custom_hierarchical_demo.py` from the root folder and hard-code paths, or use the following method below:

~~~sh
python -m deet.custom_hierarchical predict_single_study "misc\hierarchical_mvp\configs\hierarchical_prompts.csv" "misc\hierarchical_mvp\configs\demo_single_config.json"
~~~

How it works:

- The JSON config controls study_type, input_paths, output_parent_dir, max_tokens, and dspy_cache.
- The prompt CSV controls the runtime schema and field-level prompts used to build dynamic Pydantic models.
- Output files are written to output_parent_dir as timestamped outputs for study, interventions, and outcomes.

To run the same dynamic schema across every markdown file in a folder, use `predict_batch` with a `batch_config.json` (same shape as above, but `input_paths` replaced by `input_folder`):

~~~sh
python -m deet.custom_hierarchical predict_batch "misc\hierarchical_mvp\configs\hierarchical_prompts.csv" "misc\hierarchical_mvp\configs\demo_batch_config.json"
~~~

As with `main_hierarchical.py`'s `predict_batch`, outputs are written directly into `output_parent_dir` with the source markdown filename embedded in each output filename.



## tl, dr

A key innovation of the [Destiny project](https://destiny-evidence.github.io/website/) is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process for data extraction and evaluation tasks. **`data-extraction-evaluation-toolkit`**; or **`deet`** is conceived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more processing-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

Currently, the app covers the following tools:

- **Document parsing** (from a range of formats; typically into `markdown`)
- **Gold standard data ingestion and standardisation** (currently only `eppi.json` datasets are supported out of the box, for other datasets, use the data models in `data_models/base.py` to ingest your gold standard references.)
- **LLM-powered data extraction**
- **Orchetration of tools into `Pipeline`s** (these tools can be existing `DEET` modules, custom python functions, or scripts (`R`, `python`, `bash` currently suppported.))
- **Linking of gold standard references & pdf-derived parsed documents**
- **A fully-fledged cli for typical `deet` tasks**
- **Comparison & evaluation of LLM vs human annotations**

Our roadmap for future development contains:

- **A framework for repeatable pipeline runs with slight modifications for comparison**
- **Support for prompt versioning tool**

## Quickstart

### To use the `deet` CLI

```sh
uv tool install git+https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git`
deet --help
```

### To use `deet` as a package

```sh
uv add git+https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git`
```

## Using `deet`

The `data-extraction-evaluation-toolkit` (`deet`) contains mutliple modules which can be leveraged alone, or orchestrated together to form a `Pipeline`. The goal of `DEET` is to be modular and extensible, allowing users to customise a specific pipeline or workflow to their needs.

Typical pipelines can be run using the CLI app `deet --help`

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please see the [contributing guidelines](CONTRIBUTING.md) for details on how best to contribute.

A few important steps when contributing:

```sh
pre-commit install
uv run pre-commit install --hook-type commit-msg
```

This will force you to use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

## Commit message format

All commits must use [**conventional commits**](conventionalcommits.org). The pre-commit hook will reject any commit that doesn't.

| prefix | example | version effect |
|---|---|---|
| `fix:` | `fix: handle null input` | patch: `0.1.0 -> 0.1.1` |
| `feat:` | `feat: add login page` | minor: `0.1.0 -> 0.2.0` |
| `feat!:` or `BREAKING CHANGE:` footer | `feat!: remove legacy api` | major: `0.1.0 -> 1.0.0` |
| `chore:`, `docs:`, `ci:`, `test:` | `chore: update deps` | no bump |

## Tests

### Programme outcomes and indicator frameworks

Both the predefined and custom CSV-schema pipelines support
`"study_type": "Programme"`. Programme extraction runs in three passes:

1. Extract programme characteristics and interventions (including intended results).
2. Extract `Quantitative_Outcome` records with numerical evidence.
3. Extract `Qualitative_Outcome` records, passing the quantitative extraction,
   programme characteristics, interventions and original report into this call.

Both outcome models have `results` and `outcome_type` fields. The type is fixed
to `quantitative` or `qualitative` respectively. Qualitative results include
first-hand experiences from interviews, focus groups, free-text responses and
participant or beneficiary feedback, as well as supported narrative findings.
They must not simply restate numbers from the quantitative pass. A topic can
appear in both lists when distinct evidence supports each type, and qualitative
findings do not require a corresponding quantitative outcome.

JSON has separate `quantitative_outcomes` and `qualitative_outcomes` lists.
CSV and Excel combine both lists (quantitative rows first) into one `outcomes`
table/sheet in the same workbook as `programme` and `interventions`.
`outcome_type` identifies each row. Qualitative records do not contain
`baseline_data`; their baseline cells in the shared sheet are blank. Empty
outcome groups are supported, including reports without either type of result.

Outcome extraction includes, among other fields:

- `baseline_data` (quantitative outcomes only): reported quantitative or qualitative conditions for the
  outcome before the intervention, including available values, dates and context.
  Baselines are kept separate from targets and post-intervention results, are
  never inferred from later changes. The extraction uses `NR` if no baseline is
  reported. This field immediately precedes `results` in the quantitative schema
  and shared outcomes export.
- `SDG_categories`: all fitting SDGs as a semicolon-separated list of exact
  labels from the 17 SDGs supplied in the field description, or blank if none
  applies. Assignments can be inferred from reported outcome evidence; explicit
  SDG references are not required. Invalid labels and duplicate occurrences are
  removed with logged and printed warnings; valid labels and outcomes are retained.
- `outcome_category_classification`: a semicolon-separated string of
  `category_id: category_statement` entries (indicator ID and full indicator
  statement for the a framework, e.g UNICEF strategic plan). Unmatched outcomes are retained
  with an empty classification.

A sample, and current default framework is
[`unicef_outcome_indicators.csv`](deet/hierarchical_mvp/unicef_outcome_indicators.csv).
It contains the 54 indicators supplied for this extraction task, across five
goal areas.

To change frameworks, copy the CSV and edit its rows in a text editor or
spreadsheet application, saving it as UTF-8 comma-separated text. Quote cells
containing commas, quotes or line breaks using standard CSV escaping (spreadsheet
applications do this automatically). Keep these organisation-neutral headers
five headers in this order:

1. `category_id`: stable identifier, such as a UNICEF indicator ID.
2. `category_group`: broader grouping, such as a UNICEF goal area.
3. `category_statement`: full category definition or indicator statement.
4. `inclusion_criteria`: evidence that qualifies for this category.
5. `exclusion_criteria`: near-misses that should not be classified here.

All cells must be non-empty and IDs unique. IDs are strings (so `1.10` is not
`1.1`); preserve them as text when editing spreadsheets. IDs and statements must
not contain semicolons, because semicolons delimit output entries. Goal areas,
IDs, statements and scope guidance can all be replaced for another organisation.

Add this optional key to either single-study or batch JSON configs, for **either
pipeline**:

```json
{
  "study_type": "Programme",
  "classification_framework_path": "my_framework.csv"
}
```

This snippet shows only the relevant keys; retain the other required config
fields. The framework path is relative to the JSON config's directory, or can be
absolute.

For custom extraction, regenerate the Programme prompt CSV with
`write_hierarchical_prompts_csv(study_type="Programme", csv_outpath=...)`
to replace the former `Outcome` class with `Quantitative_Outcome` and
`Qualitative_Outcome`, each using `results` rather than the former separate
quantitative/qualitative result columns. Update the Programme rows to use
`quantitative_outcomes` and `qualitative_outcomes` instead of `outcomes`.
Old Programme prompt CSVs are rejected with a migration message rather than
silently losing data. Prompts for both outcome types can then be edited separately.
The framework is a separate input: it is not embedded in that CSV or in the
outcome models. Both pipelines pass it into both outcome-extraction stages and
validate that returned classification entries exactly match its IDs and
statements, with no duplicates. Invalid entries and duplicate occurrences are
removed with logged and printed warnings rather than failing extraction. Valid
entries retain their order; if none remain, the field is blank. This checks label
validity, not semantic accuracy. Malformed framework files still raise errors.

#### Opt-in OpenAI prompt caching

Set `"study_type": "ProgrammeWithCaching"` and use an `openai/` model to
select the separate cache-friendly pipeline. The original `"Programme"` pipeline
and models are unchanged. Both predefined and custom CSV-schema extraction support
this option, using the same Programme prompt CSV fields, classification framework,
three extraction stages and JSON/CSV/Excel output layout.

The new pipeline uses one fixed signature and response schema for every stage.
Instructions and schemas are identical across calls; the report and classification
framework precede the changing stage selector and previous extraction results.
This keeps the entire report in the shared prompt prefix, including with DSPy's
JSON adapter and native structured-output schema. Moving instructions into model
docstrings alone would not achieve this: DSPy renders those schemas before the
report as well.

The shared response envelope contains all four Programme fields. Each stage fills
only its own fields, using null metadata and empty lists for unused fields.
Inactive results are rejected, and numerical classifications are cleaned before
being supplied to the qualitative stage. This adds fixed instruction/schema
overhead but avoids repeating earlier results in responses.

Provider caching is separate from `dspy_cache`, which controls DSPy's local
whole-response cache. Supported OpenAI models can cache matching prompt prefixes
automatically; keep the same model, adapter, predictor instructions and few-shot
examples across stages. Model-specific minimum prefix sizes, cache retention,
routing and request settings still determine actual hits. Changing adapter modes
creates different prefixes even though each mode's schema is stable across stages.
No live cache hit is guaranteed by prompt layout alone. With `dspy_cache` disabled,
inspect `cached_input_tokens` in the existing extraction usage logs to verify
provider reuse on your deployment. See OpenAI's
[prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

Classification may infer a match from reported evidence using the scope rules.

### Running tests

Tests are written using `pytest`. You can run the tests locally using

```shell
pytest
```

Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`. Integration tests are also run for pushes/PRs into `main` (Note: these will take approx 1-2h to complete, so consider a cup of coffee while you wait).

## Adding documentation

Please add to the docs whenever you feel it would be useful. The docs are built using [mkdocs](https://www.mkdocs.org/) and mix automatically-generated API documentation with more general documentation. An automatically generated html static site is built from the `docs/` directory, and the API documentation is generated from docstrings in the code.

To add your own documentation, add markdown files to the `docs/` directory _and_ add these to the `nav` block in `mkdocs.yml`. To add API documentation, add docstrings to the code and ensure that the relevant modules are included in the `nav` block in `mkdocs.yml`.

To build the docs locally, make sure you have the docs dependencies installed by running

```sh
uv sync --all-extras --all-groups
```

which will install the documentation dependencies alongside all other dependencies, including developer dependencies. Alternatively,

```sh
uv sync --group docs
```

will install _only_ the documentation dependencies, but may uninstall other optional dependencies you have installed.

Then, from the root of the repository, run `mkdocs serve --strict` from the root of the repository and open the link that is printed to the terminal. The documentation website is currently automatically built and deployed to GitHub Pages on pushes to the `main` branch, and uses the `gh-pages` branch to serve the docs.

The documentation website is available at [https://destiny-evidence.github.io/deet](https://destiny-evidence.github.io/deet).

## Acknowledgements

We acknowledge with thanks funding from the following funders and projects:

- **Wellcome Trust**
- **Education Endowment Foundation**
- **Economic and Social Research Council (ESRC)**
