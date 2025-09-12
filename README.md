# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

A key innovation the Destiny project seeks to deliver is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process, through leveraging LLMs (Large Language Models) for extraction, benchmarking and classification of such data extraction tasks. `data-extraction-evaluation-toolkit` is conveived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more LLM-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

In addition to offering an end-to-end workflow for data extraction, this software also provides tools for tweaking, benchmarking and configuring the LLM backend for a given data extraction workflow. Given the non-deterministic way LLMs produce output to user prompts, it may prove useful to modify small sections of said prompts, and record the difference in output.

## Installation

### Installing pandoc

This software depends on [pandoc](https://pandoc.org/), a widespread open source file/document conversion utility. In this codebase, this is implemented using `pypandoc`, but this depends on `pandoc` being installed at a system level.

```shell
brew install pandoc # mac
apt install pandoc # ubuntu/debian
choco install pandoc # windows
```

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

### Using the toolkit

The first time you run anything from `app/parser.py`, you will likely have to wait for a considerable (5-15 minutes) amount of time, as dependencies will collect and install. These dependencies include machine learning libraries and pre-trained models.

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
