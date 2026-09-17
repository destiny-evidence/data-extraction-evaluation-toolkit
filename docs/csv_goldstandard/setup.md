# CSV converter

These instructions can be followed when the gold standard labeled data is in CSV format.

## Expected input data

Each row in the CSV represents one document, with columns holding document metadata,
bibliographic details, and human-labeled annotations. The header row must include at
least:

```
document_id,name,abstract,mortality
```

## Column categories

Every column in the CSV falls into one of three categories.

### Required metadata fields

`document_id` and `name` must always be present.

- `document_id` — a unique identifier for the document.
- `name` — the document's name. This can be anything, such as the paper title.

### Reference fields

Columns that map to bibliographic metadata fields defined by `destiny_sdk.enhancements`.
By default, a column is auto-assigned as a reference field whenever its name matches
one of the allowed reference field keys below — for example, an `abstract` column is
automatically treated as the document's abstract.

Allowed reference field keys:

- `abstract`
- `authorship`
- `cited_by_count`
- `created_date`
- `updated_date`
- `publication_date`
- `publication_year`
- `publisher`
- `title`
<!-- - `pagination.volume`
- `pagination.issue`
- `pagination.first_page`
- `pagination.last_page`
- `publication_venue.display_name`
- `publication_venue.issn`
- `publication_venue.venue_type`
- `publication_venue.issn_l`
- `publication_venue.host_organization_name` -->

### Attribute fields

Any remaining columns are treated as gold standard annotations: labels that human
annotators have assigned to each document (with disagreements resolved) to build a
gold standard dataset. This dataset is later used to evaluate how well LLMs or ML
models can classify or extract the same information.

In the example above, `mortality` is an attribute field. Its type is automatically
inferred from the values in the column (allowed types are bool, int, float, and str).
Here, `mortality` is inferred as a boolean, where `True` means annotators determined
the study measures or analyzes mortality as an outcome, and `False` means it does not.

!!! note "Key rule for the `mortality` example"
    A study must explicitly measure or analyze mortality as an outcome to be labeled
    `TRUE`. High lethality of the underlying condition alone is not sufficient.

## Example data

{{ read_csv('examples/csv_goldstandard/gold_standard.csv') }}

## Using this format to set up a project

### 1. Initialize the project

Once your CSV is ready, initialize a project with `--format generic_csv`:

```sh
deet project init --data gold_standard.csv --format generic_csv
```

Or, in python:

```python
from pathlib import Path

from deet.data_models.project import DeetProject
from deet.processors.converter_register import SupportedImportFormat

project = DeetProject(
    name="my project",
    gold_standard_data_path=Path("gold_standard.csv"),
    gold_standard_data_format=SupportedImportFormat.GENERIC_CSV,
)
project.setup()
```

Either approach parses your CSV and builds the following structure in your project
directory:

```text
my-project/
├── project.yaml                    # saved project configuration
├── default_extraction_config.yaml  # default data extraction settings
├── link_map.csv                    # maps documents to pdf files
├── linked_documents/               # populated once you run `deet project link`
├── prompts/
│   └── prompt_definitions.csv      # one row per attribute column - edit this next
└── data-extraction-experiments/    # populated once you run an evaluation
```

`prompts/prompt_definitions.csv` has one row for every attribute column found in your
CSV. In our example, that's a single row for `mortality`, with the `prompt` column left
blank for you to fill in.

### 2. Write your prompts

Edit the `prompt` column for each attribute you want an LLM to extract. This is the
instruction the model will be given for that document, so it should describe the same
rule your human annotators used, e.g.:

{{ read_csv('examples/csv_goldstandard/prompts/prompt_definitions.csv') }}

Leave the `prompt` column blank for any attribute you don't want extracted. See
[writing and editing prompts](../setup/quickstart.md#writing-and-editing-prompts) for
more details, including how to override the inferred `output_data_type`.

### 3. Run the evaluation

```sh
deet experiments evaluate
```

This runs an LLM over each document using your prompts, compares its output against
the gold standard labels from your CSV (e.g. `mortality`), and saves the results —
including comparison and accuracy metrics — to a timestamped folder under
`data-extraction-experiments/`.

Or, in python:

```python
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.project import DeetProject
from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
from deet.extractors.cli_helpers import init_extraction_run, prepare_documents
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor

project = DeetProject.load()

processed_annotation_data = project.process_data()
processed_annotation_data.populate_custom_prompts(
    method=CustomPromptPopulationMethod.FILE,
    filepath=project.prompt_csv_path,
)

config = DataExtractionConfig()  # or configure options here
experiment_artefacts = init_extraction_run(
    project.experiments_dir, run_name="mortality-eval"
)

documents = prepare_documents(
    processed_annotation_data.documents,
    config,
    linked_document_path=project.linked_documents_path,
    pdf_dir=project.pdf_dir,
    link_map_path=project.link_map_path,
)

data_extractor = LLMDataExtractor(config=config)
run_output = data_extractor.extract_from_documents(
    attributes=processed_annotation_data.attributes,
    documents=documents,
    context_type=config.default_context_type,
    output_file=experiment_artefacts.llm_annotations,
    show_progress=True,
)

# Compare the LLM's output against your gold standard `mortality` labels
evaluator = GoldStandardLLMEvaluator(
    gold_standard_annotated_documents=processed_annotation_data.annotated_documents,
    llm_annotated_documents=run_output.annotated_documents,
    attributes=processed_annotation_data.attributes,
    extraction_run_id=experiment_artefacts.run_id,
)
evaluator.evaluate_llm_annotations()
evaluator.write_metrics_to_csv(experiment_artefacts.metrics)
evaluator.export_llm_comparison(experiment_artefacts.comparison)
evaluator.display_metrics()
```

See the [quickstart guide](../setup/quickstart.md) for more on linking PDFs,
configuring extraction runs, and interpreting the results.
