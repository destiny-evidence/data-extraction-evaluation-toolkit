<!-- markdownlint-disable MD041 -->

# deet: the Data Extraction & Evaluation Toolkit

## What is `deet`

The Data Extraction Evaluation Toolkit (DEET) is an open-source toolkit designed to support the development and evaluation of AI-assisted data extraction for evidence synthesis.

Extracting structured information from research studies is one of the most time-consuming stages of the evidence synthesis process. While large language models (LLMs) and other AI methods have the potential to reduce this workload, inaccurate extraction can introduce errors into reviews and ultimately affect evidence used for research and policy decisions. Robust evaluation is therefore essential to ensure that AI tools are reliable, transparent, and fit for purpose.

DEET provides an end-to-end, modular workflow for configuring, running and evaluating automated data extraction methods. It enables users to compare AI-generated outputs against human-created reference (gold standard) datasets, helping to assess the accuracy, consistency and performance of different approaches. Designed around open and FAIR (Findable, Accessible, Interoperable and Reusable) principles, DEET supports reproducible evaluation through reusable code, APIs and standardised data models.

DEET is intended for researchers, evidence synthesis teams, developers and organisations interested in evaluating or developing AI-enhanced evidence synthesis workflows. Users can expect structured extraction outputs, performance metrics and transparent evaluation results that support informed decisions about the use of AI in evidence synthesis.

## Example of how DEET is used
Within DESTINY, DEET will be used to evaluate an AI-powered data extraction workflow for intervention studies. Researchers will compare data extracted automatically by a large language model with manually extracted reference data, identify where the AI performs well or makes errors, and use these results to improve extraction methods before they are incorporated into evidence synthesis workflows.

## Using deet

Standard data extraction pipelines can be executed by using `deet` as a CLI tool.
This doesn't require any familiarity with python.

For more flexibility, you can use `deet` as a python package, or contribute to
deet by extending it.

To get started with any mode of using `deet`, follow the [tutorial](setup/installation.md)
