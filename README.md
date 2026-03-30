# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

[Look here if you just want help running deet via the CLI](#using-deet-via-the-cli).

A key innovation of the [Destiny project](https://destiny-evidence.github.io/website/) is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process for data extraction and evaluation tasks. **`data-extraction-evaluation-toolkit`**; or **`deet`** is conceived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more processing-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

## Installation

### Installing `pandoc`

This software depends on [`pandoc`](https://pandoc.org/), a widespread open source file/document conversion utility. In this codebase, this is implemented using `pypandoc`, but this depends on `pandoc` being installed at a system level.

```shell
brew install pandoc # mac
apt install pandoc # ubuntu/debian
choco install pandoc # windows
```

Alternative installation instructions for Windows users are [shown further below](#pandoc)

### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
git clone git@github.com:destiny-evidence/data-extraction-evaluation-toolkit.git # SSH
# or
git clone https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git # HTTPS
cd data-extraction-evaluation-toolkit
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

## Using `deet`

The `data-extraction-evaluation-toolkit` (`deet`) contains mutliple modules which can be leveraged alone, or orchestrated together to form a `Pipeline`. The goal of `DEET` is to be modular and extensible, allowing users to customise a specific pipeline or workflow to their needs.

Currently, the app covers the following tools:

- **Document parsing** (from a range of formats; typically into `markdown`)
- **Gold standard data ingestion and standardisation** (currently only `eppi.json` datasets are supported out of the box, for other datasets, use the data models in `data_models/base.py` to ingest your gold standard references.)
- **LLM-powered data extraction**
- **Orchetration of tools into `Pipeline`s** (these tools can be existing `DEET` modules, custom python functions, or scripts (`R`, `python`, `bash` currently suppported.))

Our roadmap for future development contains:

- **Linking of gold standard references & pdf-derived parsed documents**
- **A fully-fledged cli for typical `deet` tasks**
- **A framework for repeatable pipeline runs with slight modifications for comparison**
- **Comparison & evaluation of LLM vs human annotations**
- **Support for prompt versioning tool**

### Using `deet` via the CLI

The deet command line interface (CLI) allows you to use deet without writing any python code.

#### Getting started

Once you have installed DEET, and activated your environment, (see [installation](#installation)), you should be able to run the CLI.

To test this run

```bash
deet --help
```

You should see a list of commands. To get more information on any of these (for example extract-data), run

```bash
deet extract-data --help
```

You can call these commands from anywhere where your virtual environment with deet installed is activated. It's a good idea to navigate into a new directory for each data extraction project, where you can store the outputs of running deet for your project.

To navigate to another directory use the `cd` command. `cd ..` will move up a level, cd `deet-project` will move into the `deet-project` folder, assuming it exists in the current directory. Type `ls` to see the files and folders in your current directory.

#### Importing and linking data

Most commands start by importing gold standard data, and require that you specify the location of a file that contains:

- bibliographic records of documents,
- the attributes that should be extracted from them, and
- (optionally) the gold standard annotations of those documents made by humans

DEET currently supports importing this data from an EPPIJson file.
The following examples assume you have an EPPIJson file called `references.json` in your current working directory.

If you want to extract data from the full text of documents, you will need to link your documents to those full texts. Let's assume you have a folder of pdfs in the `pdfs` directory.

The first step to link these together is to create a "link map" file which maps each reference to the filename of it's corresponding full text.

Create this by running

```bash
deet init-linkage-mapping-file references.json
```

Then edit this file, making sure to carefully add the filename of each document in the `file_path` column.

You can then link the documents using the link map as follows.

```bash
deet link-documents-fulltexts --link-map-path link_map.csv references.json
```

Linking will parse the pdfs, and save the contents (along with the bibliographic records to wherever is specified in the option `--output-path`. This defaults to `linked_documents`

#### Setting up your prompts

When you extract data, deet will try to extract attributes using prompts specified in the EPPIJson file.

If you want to edit the prompts used for data extraction, and give further details on the attributes, you can create a prompt CSV, by running `deet init-prompt-csv`. This CSV will contain a row for each attribute: enter or amend the prompt column to set the prompt that will be passed to the LLM for that attribute. Delete rows if you do not want to extract data for that attribute.

#### Extracting data from parsed/linked data

Use the command `deet extract-data` to extract data from imported documents.

Once again, you will need to specify an EPPIJson file you want to import from,  and you may want to specify the way you want to fill in prompts:

`deet extract-data --prompt-population file --csv-path prompt_definitions.csv references.json`

To set further configuration options, you can supply a path to a configuration file with the option `--config-path`. To create a template for this file detailing configurable options, run

```bash
deet export-config-template
```

You can edit this file to change the configuration options for your extract-data pipeline.

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

Then, from the root of the repository, run `mkdocs build --strict` from the root of the repository and open `site/index.html` in a browser. The documentation website is currently automatically built and deployed to GitHub Pages on pushes to the `main` branch, and uses the `gh-pages` branch to serve the docs.

The documentation website is available at [https://destiny-evidence.github.io/deet](https://destiny-evidence.github.io/deet).

## Platform-specific installation instructions

### PANDOC

Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`.
