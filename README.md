# data-extraction-evaluation-toolkit

The Data Extraction and Evaluation Toolkit (DEET) is a suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

[Docs](https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/)

## Hierarchical RCT extraction (pre-defined schema)

DEET includes an experimental hierarchical extraction workflow that uses a pre-defined schema for RCT so far. This workflow is config-driven and reads one or more markdown inputs, runs the hierarchical extraction pipeline, and writes structured outputs.

Example hierarchical_config.json:

~~~json
{
	"study_type": "RCT",
	"max_tokens": 30000,
	"dspy_cache": false,
	"input_paths": [
		"misc/input/mira_rct/main.md"
	],
	"output_parent_dir": "misc/output/mira_rct"
}
~~~

Run from the repository root:

~~~sh
python deet/main_hierarchical.py misc/input/hierarchical_config.json
~~~

Output summary:

- A timestamped JSON file is written to output_parent_dir, for example: mira_rct_YYYYMMDD_HHMMSS.json
- CSV files are written under output_parent_dir/<study_name>/:
	- study_YYYYMMDD_HHMMSS.csv
	- interventions_YYYYMMDD_HHMMSS.csv
	- outcomes_YYYYMMDD_HHMMSS.csv
- The JSON and CSV outputs contain study characteristics, intervention arms, and extracted outcomes.

## Customisable hierarchcal RCT extraction (dynamic)

DEET also provides a dynamic hierarchical RCT workflow where the extraction schema is built from a prompt CSV at runtime.

Step 1: create the prompt CSV from the current RCT model schema.

~~~sh
python -m deet.custom_hierarchical write_hierarchical_prompts_csv --study-type RCT
~~~

This generates a CSV with columns class, attribute, prompt, and datatype. You can review and edit prompts in this file before extraction.

Step 2: run dynamic extraction using both the config file and the prompt CSV.

~~~sh
python -m deet.custom_hierarchical custom_extract misc/hierarchical_mvp/configs/hierarchical_prompts.csv misc/input/hierarchical_config.json
~~~

How it works:

- The JSON config controls study_type, input_paths, output_parent_dir, max_tokens, and dspy_cache.
- The prompt CSV controls the runtime schema and field-level prompts used to build dynamic Pydantic models.
- Output files are written to output_parent_dir as timestamped JSON plus CSV outputs for study, interventions, and outcomes.

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
