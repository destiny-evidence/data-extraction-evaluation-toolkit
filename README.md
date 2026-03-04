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

You should see a list of commands. To get more information on any of these (for example import-data), run

```bash
deet import-data --help
```

You can call these commands from anywhere where your virtual environment with deet installed is activated. It's a good idea to navigate into a new directory for each data extraction project, where you can store the outputs of running deet for your project.

To navigate to another directory use the `cd` command. `cd ..` will move up a level, cd `deet-project` will move into the `deet-project` folder, assuming it exists in the current directory. Type `ls` to see the files and folders in your current directory.

#### Importing and linking data

The first step to using `deet` is importing data. Currently, EPPIJson is the only supported external data format.

```bash
deet import-data --help
```

shows the arguments you need to import data, along with their defaults. You will see that running this command defaults to import deet-structured data from the current directory.

You could put the EPPIJson file you want to import inside your current directory. Assuming there is a file called `references.json` in your current directory, you could import it by running

```bash
deet import-data --gs-data-path references.json --gs-data-format eppi_json
```

This will read the EPPIJson data and write it into the current directory in deet format. Running `deet import-data` again will read this data back in.

Once you have imported data, you may want to link the data contained in your EPPIJson to a set of pdfs.

To do this, you can run

```bash
deet link-documents
```

If you want to, you can enter the same arguments as you did to import-data to specify the same EPPIJson file, otherwise it will default to reading the deet-structured data written by importing.

Deet will look for pdfs in the folder specified by --pdf-dir, this defaults to `pdfs`.

Deet will try to link documents to pdfs using IDs, authoryear, ..., but if you want to specify a file containing a map between your EPPIJson document_ids and the filenames of your pdfs, you can create a template for this with the `deet create-link-map` command. Once you have filled the template by specifying the file name for each document, you can link documents by pointing to this file

```bash
deet link-documents --link-map-path link_map.csv
```

Linking will parse the pdfs, and save the contents (along with the bibliographic records to wherever is specified in the option `--output-path`. This defaults to `linked_documents`

#### Extracting data from parsed/linked data

Once you have imported data (and linked this data to pdfs if you are doing full text data extraction), you are ready to extract data from them.

Use the command `deet data-extraction` to extract data from imported documents.

You can set `--gs-data-path` and `--gs-data-format` to specify the EPPIJson you want to import from, or leave these blank to load data from a previous import.

By default, data-extraction will try to extract attributes using prompts specified in the EPPIJson file.

If you want to edit the prompts used for data extraction, you can do this by setting the `--prompt-population` option to `cli`, to fill in prompts in the command line, or by setting `--prompt-population` to `file`, and pointing to a csv detailing a prompt for each attribute with `--csv-path`. To create a template for this csv, run `deet write-prompt-csv`.

To set further configuration options, you can supply a path to a configuration file with the option `--config-path`. To create a template for this file detailing configurable options, run

```bash
deet export-default-config
```

You can edit this file to change the configuration options for your data-extraction pipeline.

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please see the [contributing guidelines](CONTRIBUTING.md) for details on how best to contribute.

## Tests

Tests are written using `pytest`. You can run the tests locally using

```shell
pytest
```

Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`.
