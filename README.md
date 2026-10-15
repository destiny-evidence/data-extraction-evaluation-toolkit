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
Suggested measure names are prompt guidance, not enums: edit the
`ReportedStatistic` rows in generated prompt CSVs to customize them.
`basis` distinguishes endpoint values from change-from-baseline statistics;
`details` records confidence levels, the corresponding summary measure,
adjustment, or other reported context. Preserve values and interval bounds as
reported, without converting or calculating missing statistics. Use an empty
list when no statistic is reported. A between-arm mean difference is not an
arm-level average; preserve it in `effect_estimates` (or Cochrane `footnotes`),
not as a group's summary statistic.

JSON preserves these lists as objects. CSV/XLSX encode each list as JSON in its
own cell without duplicating the arm's outcome row. Cochrane exports add the
full statistic-list columns alongside the RevMan columns. They populate
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
