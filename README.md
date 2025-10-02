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

### Using the toolkit

The first time you run anything from `app/parser.py`, you will likely have to wait for a considerable (5-15 minutes) amount of time, as dependencies will collect and install. These dependencies include machine learning libraries and pre-trained models.

## Data Processing

### Annotation Converter

Before running LLM evaluation, you need to process raw EPPI-Reviewer JSON annotations into structured format. The annotation converter script is available at `app/scripts/annotation_converter.py` and can be used to convert raw EPPI-Reviewer data into the structured format needed for LLM evaluation.

**Usage:**

```bash
uv run python app/scripts/annotation_converter.py <input_file> <output_dir>
```

**Example:**

```bash
uv run python app/scripts/annotation_converter.py input_path output_path
```

This creates:

- `annotated_documents.json` - Documents with their annotations
- `attributes.json` - All available attributes
- `documents.json` - Document metadata
- `attribute_id_to_label_mapping.json` - Attribute ID to label mapping

## LLM Evaluation

### Quick Start

1. **Set up your LLM provider** in `.env`:

   ```bash
   # Copy example environment file
   cp env.example .env

   # Edit .env with your API keys
   # For Azure OpenAI:
   LLM_PROVIDER=azure
   AZURE_API_KEY=your-azure-api-key
   AZURE_API_BASE=https://your-resource.openai.azure.com/
   AZURE_DEPLOYMENT=your-deployment-name

   # For OpenAI:
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-openai-api-key
   ```

2. **Run LLM evaluation**:

   ```bash
   uv run python app/scripts/simple_llm_eval.py
   ```

This will:

- Load the first document from `app/annotations/processed/eppi/annotated_documents.json`
- Evaluate it against the first 2 attributes
- Show detailed logs of the LLM request/response
- Save results to `simple_evaluation_results.json`

### Understanding the Output

The LLM evaluation produces structured results:

```json
{
  "answers": [
    {
      "attribute_name": "Arm name",
      "Answer": "False",
      "Reasoning": "The document does not specify any names or identifiers for the arms of the study...",
      "Citation": "Participants in both conditions received an 8-week telephone-delivered..."
    }
  ]
}
```

- **Answer**: "True" or "False" - whether the attribute is present
- **Reasoning**: Detailed explanation of the decision
- **Citation**: Supporting text from the document (if available)

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
