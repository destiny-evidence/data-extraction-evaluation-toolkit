# Data Extraction Experiment

To find out how well we are able to extract data from documents using LLMs,
we conduct <term: data extraction experiment>s.

## Configuring experiments

The success of any experiment will depend on the model chosen,
how this is configured, and how prompts are defined to explain the data extraction
task to the LLM. We use the term <term: experiment configuration> to the combination of all of these variables.

## Experiment artefacts

Each time we run an experiment (using `deet experiments extract`), we store the
results of the experiment, as well as all of the configuration options required
to reproduce it, as experiment artefacts. These are defined in
[`deet.data_models.project.ExperimentArtefacts`](../reference/api.md#deet.data_models.project.ExperimentArtefacts),
and comprise

- The exact prompts used in the experiment (`ExperimentArtefacts.prompts_snapshot`)
- All configurable options (such as the model, the temperature setting, etc.) (`ExperimentArtefacts.config_snapshot`)
- A csv containing metrics for each attribute and each document (see [evaluation](evaluation.md)) (`ExperimentArtefacts.metrics`)
- A csv showing the gold-standard and llm-generated data extraction results for each document and attribute side by side (`ExperimentArtefacts.comparison`)
- The annotated documents (containing the data extracted by the LLM) (`ExperimentArtefacts.llm_annotations`)

## Comparing experiments

By running multiple experiments, each time varying the configuration, we
are able to compare different <term: experiment configuration>s.

For example, we might try the same data extraction task with an open model
as well as with a closed model,
or with a small model, as well as with a large model, in order to assess whether
the same performance cannot be achieved in a more reproducible fashion, or
more efficiently.

Likewise, if our initial experiments are not entirely successful,
we may want to explore a number of different [prompting strategies](prompting-strategies.md),
or to iteratively adjust the wording of individual prompts,
based on our observations of how LLMs fail to produce extraction results that align with the gold standard.
For this, `ExperimentArtefacts.comparison` can be used.

In each case, the metrics produced by [evaluating](evaluation.md) allow us to assess whether a given experiment has performed the data extraction task better or worse than another comparable experiment.
