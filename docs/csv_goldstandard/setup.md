# CSV converter

These instructions can be followed when the gold standard labeled data is in CSV format.

## Expected input data

Each row in the CSV represents one document, with columns holding document metadata,
bibliographic details, and human-labeled annotations. The header row must include at
least the columns `document_id`, and `name`. The following columns would additionally provide
an abstract, and an attribute column for an attribute named "mortality".

```csv
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

??? note "Allowed reference field keys:"

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

Once your CSV is ready, initialize a project and select the format `generic_csv`

```sh
deet project init
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

See the [quickstart guide](../setup/quickstart.md) for more on
customizing prompts, linking PDFs,
configuring extraction runs, and interpreting the results.
