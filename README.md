# data-extraction-evaluation-toolkit

The Data Extraction Evaluation Toolkit (DEET) is an open-source toolkit designed to support the development and evaluation of AI-assisted data extraction for evidence synthesis.

Extracting structured information from research studies is one of the most time-consuming stages of the evidence synthesis process. While large language models (LLMs) and other AI methods have the potential to reduce this workload, inaccurate extraction can introduce errors into reviews and ultimately affect evidence used for research and policy decisions. Robust evaluation is therefore essential to ensure that AI tools are reliable, transparent, and fit for purpose.

DEET provides an end-to-end, modular workflow for configuring, running and evaluating automated data extraction methods. It enables users to compare AI-generated outputs against human-created reference (gold standard) datasets, helping to assess the accuracy, consistency and performance of different approaches. Designed around open and FAIR (Findable, Accessible, Interoperable and Reusable) principles, DEET supports reproducible evaluation through reusable code, APIs and standardised data models.

DEET is intended for researchers, evidence synthesis teams, developers and organisations interested in evaluating or developing AI-enhanced evidence synthesis workflows. Users can expect structured extraction outputs, performance metrics and transparent evaluation results that support informed decisions about the use of AI in evidence synthesis.

Within DESTINY, DEET will be used to evaluate an AI-powered data extraction workflow for intervention studies. Researchers will compare data extracted automatically by a large language model with manually extracted reference data, identify where the AI performs well or makes errors, and use these results to improve extraction methods before they are incorporated into evidence synthesis workflows.

More detailed documentation can be found here [Docs](https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/)

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
