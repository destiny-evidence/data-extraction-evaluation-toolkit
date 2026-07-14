# Evaluation

## Comparing human and AI-extracted data

To evaluate an [experiment](experiment.md), deet compares the annotations produced by an LLM, with the gold-standard[^1] human-annotations provided, and measures the extent of the agreement using evaluation metrics.

## Good evaluation practice

Deet is designed to encourage good evaluation practices.

This means separating data used to develop a pipeline (e.g. by engineering prompts or selected best-performing models) from data used to evaluate a pipeline.

### Evaluation strategies

To do this, deet implements evaluation strategies to manage how data is used for evaluation, in order to maximise efficiency while avoiding leaking.

Evaluation strategies are in general managed by running

```sh
deet experiments splits
```

which will trigger an interactive wizard that prompts you with options managing your evaluation.

#### The null evaluation strategy

By default, the null evaluation strategy is selected for new projects. With the null evaluation strategy, data extraction with deet is run on all documents within a [project](project.md). This is appropriate if you have separated your evaluation data manually into a separate "evaluation" project, or if you just want to test something informally. In this case running `deet experiments splits` will simply print a message informing you that the current strategy has no further options.

#### The dev-val-test strategy

The dev-val-test strategy explicitly handles the separation of data used for training and evaluation, by creating three splits of the data ([more info](https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data)). Assigning documents to those splits is managed by running

```sh
deet experiments splits
```

The first time this is run, the only available option will be to assign documents to the development set. These documents will be used to iteratively try out and improve upon prompts and other experiment configuration options. Each time you run `deet experiments evaluate`, you will see performance metrics for the chosen experiment configuration.

Once you have reached an experiment configuration which achieves performance you are happy with, you can run `deet experiments splits` again to re-enter the wizard. You will now see an option to "Move to validation". Selecting this will prompt you to select a number of documents to be used for validation, that is, for a first out-of-sample test of your chosen configuration.

If you choose to move to validation, you will be prompted to choose the model configuration you wish to validate. The prompts and other settings from this model will then be used to extract data from the documents in the newly created validation set, and metrics will be reported.

Users are then prompted to choose between:

- Accepting the configuration and doing a final test of the pipeline on all remaining documents; and
- Rejecting the configuration (because performance was dissappointing), and moving back to prompt development.

If the former option is selected, the project is finished, and the results are to be seen as the final evaluation scores.

If the latter option is selected, documents from the validation set are passed to the development set, and you are invited to continue iterating prompts.

[^1]: We refer to human-annotated data as "gold-standard" data, recognising that human-annotated errors also contain errors.
