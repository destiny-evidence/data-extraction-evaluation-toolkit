# DeetProject: a data extraction project

A `deet` project [deet.data_models.project.DeetProject](../reference/api.md#deet.data_models.project.DeetProject)
defines a standardised structure to develop and evaluate automated pipelines for a data extraction task. Since automated data extraction is not expected to work perfectly straight away, a project provides the ability to *experiment* with different prompts and configuration options, without needing to repeat processes or configuration options that stay the same across your experiments.

A project should therefore describe and document your attempts to extract data for a single data extraction task, and a single dataset. Each time you try to extract data, this will be recorded as [data extraction experiment](../concepts/experiment.md)

When you create a project, you must give it a name, define the dataset you are using, and describe the format of that dataset. If you are extracting data from pdfs, you must provide a directory that contains those pdfs. Creating a project (either by running `deet project init`, or by calling `DeetProject.setup()`) will store these configuration options in a project configuration file in your project directory: `project.yaml`.

If you need to alter any of this information (e.g. because you want to rename your project, or because the path to your data has changed, you can edit this file directly). Your project file should look like this

```yaml
--8<-- "examples/quickstart/project.yaml"
```
