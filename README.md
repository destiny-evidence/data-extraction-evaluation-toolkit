# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

A key innovation the Destiny project seeks to deliver is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process for data extraction and evaluation tasks. `data-extraction-evaluation-toolkit` is conceived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more processing-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

## Installation

### Installing `pandoc`

This software depends on [`pandoc`](https://pandoc.org/), a widespread open source file/document conversion utility. In this codebase, this is implemented using `pypandoc`, but this depends on `pandoc` being installed at a system level.

```shell
brew install pandoc # mac
apt install pandoc # ubuntu/debian
choco install pandoc # windows
```

### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
uv sync
```

### Activate your environment

```sh
source .venv/bin/activate
```

### Installing pre-commit hooks

Install `pre-commit` locally (in your activated `venv`) to aid code consistency (if you're looking to contribute).

```sh
pre-commit install
```

## Using `DEET`

The `data-extraction-evaluation-toolkit` (`DEET`) contains mutliple modules which can be leveraged alone, or orchestrated together to form a `Pipeline`. The goal of `DEET` is to be modular and extensible, allowing users to customise a specific pipeline or workflow to their needs.

Currently, the app covers the following tools:

- **Document parsing** (from a range of formats; typically into `markdown`)
- **Gold standard data ingestion and standardisation** (currently only `EPPI-json` datasets are supported)
- **LLM-powered data extraction**
- **Orchetration of tools into `Pipeline`s** (these tools can be existing `DEET` modules, custom python functions, or scripts (`R`, `python`, `bash` currently suppported.))

Our roadmap for future development contains:

- **Comparison & evaluation of LLM vs human annotations**
- **Support for LLM prompts and SQL code**

## Document parsing

The first time you run anything from `deet/parser.py`, you will likely have to wait for a considerable (5-15 minutes) amount of time, as dependencies will collect and install. These dependencies include machine learning libraries and pre-trained models.

## Data Processing

### Annotation Converter

The annotation converter can be used to convert raw EPPI-Reviewer data into structured format.

**Usage:**

```bash
uv run python deet/scripts/annotation_converter_cli.py <input_file> <output_dir>
```

**Example:**

```bash
uv run python deet/scripts/annotation_converter_cli.py deet/annotations/raw/eppi/sample_eppi.json output/processed
```

This creates an organized directory structure:

```text
output/processed/eppi/{filename_without_extension}/
├── attributes.json
├── documents.json
├── annotated_documents.json
└── attribute_id_to_label_mapping.json
```

For example, processing `sample_eppi.json` creates `output/processed/eppi/sample_eppi/` with the JSON files inside.

## LLM-powered data extraction

## Orchastration into `Pipeline`s

## Projects

### Setup

Projects offer a way to organise and manage data and configuration files for a data extraction task.

The suggested way to make a project is in a separate directory, into which you install deet

From a terminal inside this directory you can do this with the following:

```shell
uv init
uv add git@github.com:destiny-evidence/data-extraction-evaluation-toolkit.git@project
```

After doing this you can create a project by providing a path to the project folder and a path to the EPPIJson data you want to add to the project with the command `DEET-create-project`

Assuming you want to create the project in the current directory using a file in that directory called `data.json`, this will look like

```shell
DEET-create-project . data.json
```

### Batches and runs

We don't run our data extraction pipeline on all documents inside a project, instead we create batches of documents, and run our pipeline on those documents ([further info](https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data)).

We can create a batch with `n` documents sampled at random with the `DEET-new-batch n` command. For example, `DEET-new-batch 5` will create a batch of 5 records.

Each time we run our pipeline on that batch of documents, we will save what we sent to the LLM, what came out, our configuration options, and some evaluation of the results in a new run folder within the batch folder. Configuration options can be changed by editing the `run-settings.yaml` file that was created when we set up the project. We may want to try a few different settings, editing and saving this file, before running

```shell
DEET-batch-pipeline
```

which will run the pipeline with the settings currently defined in the file.

### Creating and using a prompt definition file

Prompts for the individual attributes will be taken from the EPPIJson file, if they are specified there. An alternative way to do this is using a prompt definition file.

We can create this file in the `prompts` folder of our project using

```shell
DEET-write-csv-template
```

Now to change the prompts, we can copy this file, give it a new name, edit it, save it, and point to this new file in the `prompt_csv_path` option of our `run-settings.yaml` file. That way we can keep track of all the versions of our prompts, and which ones were used in which run.

### Metrics to compare runs

We run multiple runs with different prompts and configuration options in order to find out which of these works the best. We can assess this by comparing the `metrics.json` file that is created in each run after each successful pipeline run.

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please check existing issues, and perhaps pick something from there? Please create a new branch off `development`.
Once you're ready for your code to be reviewed, submit a PR into `development.

## Tests

Tests are written using `pytest`. You can run the tests locally using

```shell
pytest
```

Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`. Integration tests are also run for pushes/PRs into `main` (Note: these will take approx 1-2h to complete, so consider a cup of coffee while you wait).
