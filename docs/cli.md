# The command line interface (CLI)

The deet command line interface (CLI) allows you to use deet without writing any python code.
Instead, you enter commands directly into the command line.

## Getting started

Once you have installed DEET (see [installation](installation.md)), you should be able to run the CLI.

To test this run

```sh
deet --help
```

You should see a list of commands. To get more information on any of these (for example extract-data), run

```sh
deet extract-data --help
```

If you installed deet with `uv tool install`, you can call these commands from any directory.

## Data extraction project folders

We recommend that each data extraction project lives in its own directory.
You should therefore run deet from within a project directory.

??? "Navigation in the command line"

    To navigate to another directory in the command line, use the `cd` command. `cd ..` will move up a level, `cd deet-project` will move into the `deet-project` folder, assuming it exists in the current directory. Type `ls` to see the files and folders in your current directory. `mkdir deet-project` will create a new directory called deet-project.

## Settings

`deet` loads settings from a file in your current directory called `.env`.
This is used to store private information like your API key.
To set up deet to use LLMs via azure, you will need to enter your information into a file called .env.
Your file should look like this (replace the placeholders with your information):

```sh
AZURE_API_KEY="your-azure-api-key-here"
AZURE_API_BASE=https://your-resource.openai.azure.com/
```

## Gold Standard Data

Most commands start by importing gold standard data, and require that you specify the location of a file that contains:

- bibliographic records of documents,
- the attributes that should be extracted from them, and
- the gold standard annotations of those documents made by humans

DEET currently supports importing this data from an EPPIJson file.
The following examples assume you have an EPPIJson file called `references.json` in your current working directory.
Replace `references.json` with a path to your gold-standard data file (this doesn't have to be in your current directory).

## Linking gold standard data to pdfs

If you want to extract data from the full text of documents, you will need to link your documents to those full texts. Let's assume you have a folder of pdfs in the `pdfs` directory.

The first step to link these together is to create a "link map" file which maps each reference to the filename of it's corresponding full text.

Create this by running

```sh
deet init-linkage-mapping-file references.json
```

Then edit this file, making sure to carefully add the filename of each document in the `file_path` column.

??? "Editing a csv file"

    You can of course edit the link map csv in a spreadsheet program like microsoft excel.

    If, however, you want to stay in the command line, you could use a command line tabular data program like [VisiData](https://www.visidata.org/)

    Alternatively, if you are working in VS Code, we recommend you install an extension for working with CSV files, such as [CSV](https://marketplace.visualstudio.com/items?itemName=ReprEng.csv)

You can then link the documents using the link map as follows.

```sh
deet link-documents-fulltexts --link-map-path link_map.csv references.json
```

If your pdfs live somewhere other than `pdfs`, you can specify this with the `--pdf-dir` option.

Linking will parse the pdfs, and save the contents (along with the bibliographic records to wherever is specified in the option `--output-path`. This defaults to `linked_documents`

## Setting up your prompts

When you extract data, deet will try to extract attributes, using prompts, derived from the EPPIJson file.

If you want to edit the prompts used for data extraction, and give further details on the attributes, you can create a prompt CSV, by running

```sh
deet init-prompt-csv
```

This CSV will contain a row for each attribute: enter or amend the prompt column to set the prompt that will be passed to the LLM for that attribute. Delete rows if you do not want to extract data for that attribute.

## Extracting data from parsed/linked data

Use the command `deet extract-data` to extract data from imported documents.

Once again, you will need to specify an EPPIJson file you want to import from,  and you may want to specify the way you want to fill in prompts:

```sh
deet extract-data --prompt-population file --csv-path prompt_definitions.csv references.json
```

### Data extraction configuration

To set further configuration options, you can supply a path to a configuration file with the option `--config-path`. To create a template for this file detailing configurable options, run

```sh
deet export-config-template
```

You can edit this file to change the configuration options for your extract-data pipeline.

#### LLM Model and provider

By default, data extraction runs using gpt-4o-mini through Azure.

If you want to run models locally with [ollama](https://ollama.com/), you can set the `provider` option to `ollama`,
and the `model` option to any model you have running.

#### Context type

By default, data extraction will attempt to extract data from the full text of a record.
If, however, you only wish to use the abstract, you can set `default_context_type` from `full_document` to `abstract_only`.

#### Further configuration options

A full list of configurable options is available in the [API documentation](api.md/#deet.extractors.llm_data_extractor.DataExtractionConfig).
