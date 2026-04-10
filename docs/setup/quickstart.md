<!-- markdownlint-disable MD033 -->

# deet tutorial

This guide will help you to run first data extraction experiments using either the cli or python

## Setting up a project

A `DeetProject` is a workspace for a data extraction task for a specific dataset.
Each project should have its own directory on your machine.
This is where we will store configuration options and the results of your data extraction experiments.

### Initialising

<div class="grid cards" markdown>

- **CLI**

    ---

    To set up a project using the CLI, run `deet project init` from the directory
    where you would like to store your project.
    This will interactively collect the information required to set your project up,
    and also prompt you to enter credentials for making API calls to LLMs.

    ```sh
    mkdir new-project
    cd new-project
    deet project init
    ```

    !!! example "Result (Terminal)"
        ![Type: GIF of CLI Wizard](../assets/images/project_init.gif)

- **Python**

    ---

    To set up a project in python, simply instantiate a DeetProject object,
    and then call `DeetProject.setup()`

    ```python
    from deet.data_models.project import DeetProject
    from deet.processors.converter_register import SupportedImportFormat
    from pathlib import Path

    project = DeetProject(
        name="my cool new project",
        gold_standard_data_path=Path("<path_to_your_data>"),
        gold_standard_data_format=SupportedImportFormat.EPPI_JSON, # Replace this if you are using another import format
        pdf_dir=Path("<path_to_your_pdf_dir>")
    )
    project.setup()
    ```

    You should create a `.env` file yourself to store necessary API keys (see [settings](../reference/api.md#deet.settings))

</div>

### Linking documents to pdfs

If you want to extract data from the full texts of your documents, you will need to edit the file `link_map.csv` created in your project directory by setting up deet, to point each document to the file that contains its pdf.

{{ read_csv('examples/quickstart/link_map.csv') }}

After you have edited this file, you can link the documents

<div class="grid cards" markdown>

- **CLI**

    ---

    In the CLI, you can do this by running

    ```sh
    deet project link
    ```

- **Python**

    ---

    To do this in python, use the DocumentReferenceLinker. You can also choose other strategies to link documents and pdfs (see [deet.processors.linker](../reference/api.md#deet.processors.linker))

    ```python
    from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy

    processed_annotation_data = project.process_data()

    linker = DocumentReferenceLinker(
        references=processed_annotation_data.documents,
        document_base_dir=project.pdf_dir,
        document_reference_mapping=project.link_map_path,
        linking_strategies=[LinkingStrategy.MAPPING_FILE],
    )
    linked_documents = linker.link_many_references_parsed_documents()
    ```

</div>

## Extracting data

### Writing and editing prompts

Setting up a project creates a file called `prompts/prompt_definitions.csv` with a row for each of the attributes you can extract from your data.
Edit this file, creating a prompt in the `prompt` column.
Leave the `prompt` column blank for any attribute you do not wish to extract.
You can also edit the `output_data_type` column (see [deet.data_models.base.AttributeType](../reference/api.md#deet.data_models.base.AttributeType)) if the automatically parsed data type is incorrect.

{{ read_csv('examples/quickstart/prompts/prompt_definitions.csv') }}

### Running an extraction experiment

Now that you've defined your prompts, you are ready to extract data from your documents

<div class="grid cards" markdown>

- **CLI**

    ---

    In the CLI, you can do this by running

    ```sh
    deet run extract
    ```

- **Python**

    ---

    To do this in python, use the LLMDataExtractor.
    You can use a DataExtractionConfig object to set configuration options

    ```python
    from deet.extractors.llm_data_extractor import LLMDataExtractor, DataExtractionConfig
    from deet.data_models.enums import CustomPromptPopulationMethod
    from deet.extractors.cli_helpers import (
        init_extraction_run,
        load_config_from_context,
        prepare_documents,
    )

    config = DataExtractionConfig(
        # configure options here, or leave blank to use defaults
    )

    processed_annotation_data = project.process_data()

    # Populate your custom prompts
    processed_annotation_data.populate_custom_prompts(
        method=CustomPromptPopulationMethod.FILE,
        filepath=project.prompt_csv_path
    )
    ```

</div>
