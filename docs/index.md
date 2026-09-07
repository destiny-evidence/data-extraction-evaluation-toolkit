<!-- markdownlint-disable MD041 -->

# deet: the Data Extraction & Evaluation Toolkit

## What is `deet`

`deet` is a python framework to build reproducible and well-evaluated data extraction pipelines using LLMs.

It is designed for use in an evidence synthesis context, where the documents from which data is extracted are scientific papers.
In this context, [data extraction](concepts/data-extraction.md), is conceived in a broad sense, including:

1. the categorisation of documents as relevant or not relevant according to pre-defined inclusion/exclusion criteria (**screening**);
2. the categorisation of documents according to a predefined taxonomy of classes (**coding**);
3. the extraction of numeric or string information from documents, such as outcomes, effect sizes, or participant characteristics (**data extraction**);
4. the evaluation of the research quality of studies (**critical appraisal**).

`deet` is designed for data extraction for evidence synthesis in the narrower sense of **3.**,
although in principle it can be used to extract any type of data from any type of document.

## Using deet

Standard data extraction pipelines can be executed by using `deet` as a CLI tool.
This doesn't require any familiarity with python.

For more flexibility, you can use `deet` as a python package, or contribute to
deet by extending it.

To get started with any mode of using `deet`, follow the [tutorial](setup/installation.md)
