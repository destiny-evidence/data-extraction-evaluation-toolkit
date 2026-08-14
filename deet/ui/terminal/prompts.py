"""Re-usable inquirerpy prompts."""

from collections.abc import Mapping, Sequence

from InquirerPy import inquirer

from deet.data_models.project import DeetProject, ExperimentArtefacts


def select_experiment(project: DeetProject) -> ExperimentArtefacts:
    """Select an experiment from a project's completed experiments."""
    choices = [
        {"name": exp.run_id, "value": exp} for exp in project.completed_experiments
    ]
    selected_experiment: ExperimentArtefacts = inquirer.select(
        message="Select the experiment configuration to validate:", choices=choices
    ).execute()
    return selected_experiment


def select_from_list[ItemT: Mapping[str, object]](
    items: Sequence[ItemT],
    item_key: str,
    selected_value: str | None = None,
    *,
    display_key: str = "description",
    prompt_message: str = "Select an option",
) -> ItemT:
    """
    Select an item from a list by key, either by prompt or direct value.

    Args:
        items: List of dicts to select from.

    """
    if selected_value is None:
        choices = [
            {"name": item[display_key], "value": item[item_key]} for item in items
        ]
        selected_value = inquirer.select(
            message=prompt_message, choices=choices
        ).execute()
    try:
        return next(item for item in items if item[item_key] == selected_value)
    except StopIteration as e:
        key_not_found = f"'{selected_value}' not found"
        raise KeyError(key_not_found) from e
