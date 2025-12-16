"""FastHTML App definition."""

import json
from collections.abc import Generator
from itertools import groupby
from multiprocessing import Process
from pathlib import Path
from typing import Literal

import fasthtml.common as fh
import monsterui.all as mui
import yaml
from fh_pydantic_form import PydanticForm
from starlette.requests import Request

from deet.data_models.project import AttributeMetric, DeetProject
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.scripts.batch_pipeline import batch_pipeline

hdrs = mui.Theme.blue.headers()

app, rt = fh.fast_app(hdrs=hdrs)

p = DeetProject(path=".")


def get_metrics(run_path: Path) -> Generator[fh.FT]:
    """Return a table of metrics for a given pipeline run."""
    metric_path = run_path.joinpath("metrics.json")
    metrics = [AttributeMetric.model_validate(x) for x in json.load(metric_path.open())]
    metrics_names = ["Attribute", *list({x.metric for x in metrics})]
    yield fh.Tr([fh.Th(x) for x in metrics_names])
    for k, g in groupby(metrics, lambda x: x.attribute):
        yield fh.Tr(fh.Td(k.attribute_label), *[fh.Td(x.value) for x in g])


def sidebar() -> fh.FT:
    """return a sidebar."""
    contents = [
        ("Project home", "/"),
        ("Documents", "documents"),
        ("Attributes", "attributes"),
        ("Batches", "batches"),
        ("Pipeline runs", "list_runs"),
    ]
    return mui.NavContainer(
        *[fh.Li(fh.A(x[0], href=x[1])) for x in contents], cls=mui.NavT.primary
    )


def heading() -> fh.FT:
    """Return a heading."""
    return fh.Div(cls="space-y-5")(
        mui.H2("DEET UI"), mui.Subtitle("Manage A DEET project"), mui.DividerSplit()
    )


def layout(content: fh.FT) -> fh.FT:
    """Return a basic layout with header, sidebar and content."""
    return fh.Div(
        fh.Title("DEET Project"),
        mui.Container(
            heading(), fh.Div(cls="flex gap-x-12")(sidebar(), fh.Div(content))
        ),
    )


@rt()
def documents(current_page: int = 0) -> fh.FT:
    """Show a table of project documents."""
    docs = list(p.read_annotated_documents())
    page_size = 5
    paginated_docs = docs[current_page * page_size : (current_page + 1) * page_size]

    def footer() -> fh.FT:
        """Return a footer for a table of documents."""
        total_pages = (len(docs) + page_size - 1) // page_size
        links = [
            ("chevrons-left", 0),
            ("chevron-left", max(0, current_page - 1)),
            ("chevron-right", current_page + 1),
            ("chevrons-right", total_pages),
        ]
        return mui.DivFullySpaced(
            fh.Div(),
            mui.DivLAligned(
                mui.DivCentered(
                    f"Page {current_page + 1} of {total_pages}", cls=mui.TextT.sm
                ),
                mui.DivLAligned(
                    *[
                        fh.A(
                            mui.UkIconLink(icon=i, button=True),
                            href=documents.to(current_page=dest),
                        )
                        for i, dest in links
                    ]
                ),
            ),
        )

    content = fh.Div(
        mui.TableFromDicts(
            ["name", "context", "document_id"],
            body_data=[d.model_dump() for d in paginated_docs],
            sortable=True,
            cls=(mui.TableT.responsive, mui.TableT.sm, mui.TableT.divider),
        ),
        footer(),
    )
    return layout(content)


@rt()
def attributes() -> fh.FT:
    """Show a table of project attributes."""
    content = fh.Div(
        mui.TableFromDicts(
            ["attribute_label", "prompt", "output_data_type"],
            body_data=[x.model_dump() for x in p.read_attributes()],
            sortable=True,
            cls=(mui.TableT.responsive, mui.TableT.sm, mui.TableT.divider),
        ),  # footer()
    )
    # content = Ul(*attribute_list(p.read_attributes()))
    return layout(content)


@rt()
def batches() -> fh.FT:
    """Show a table of batches."""
    data = [
        {
            "name": b.parts[1],
            "number of documents": len(json.load(b.joinpath("batch_ids.json").open())),
            "runs": len([x for x in b.iterdir() if "run_" in x.name]),
        }
        for b in p.batches
    ]
    content = fh.Div(
        mui.TableFromDicts(
            ["name", "number of documents", "runs"],
            body_data=data,
            sortable=True,
            cls=(mui.TableT.responsive, mui.TableT.sm, mui.TableT.divider),
        ),  # footer()
    )

    return layout(content)


@rt()
def fill_config_modal(run_path: str) -> fh.FT:
    """Fill a modal with a view of the config for the run described in run_path."""
    cpath = Path(run_path).joinpath("run_settings.json")
    run_settings = json.load(cpath.open())
    return fh.Div(
        fh.H3(f"{run_path}", cls="text-lg font-bold"),
        fh.Div(*[fh.P(f"{k}: {v}") for k, v in run_settings.items()]),
        mui.ModalCloseButton("Close"),
    )


@rt()
def fill_metric_modal(run_path: str) -> fh.FT:
    """Fill a modal with a view of the metrics for the run described in run_path."""
    return fh.Div(
        mui.H3(f"{run_path}", cls="text-lg font-bold"),
        mui.Table(get_metrics(Path(run_path))),
        mui.ModalCloseButton("Close"),
    )


@rt()
def fill_run_form() -> fh.FT:
    """Fill a pipeline config form with the contents of run-settings.yaml."""
    run_config = yaml.safe_load(p.p.joinpath("run-settings.yaml").open())

    run_form_renderer = PydanticForm(
        "run_form",
        DataExtractionConfig,
        exclude_fields=["selected_attribute_ids"],
        initial_values=run_config,
    )

    return mui.Form(
        run_form_renderer.render_inputs(),
        fh.Div(
            mui.Button("Submit", type="submit", cls=mui.ButtonT.primary),
            cls="mt-4 flex items-center gap-2",
        ),
        hx_post="/submit_run",
        # hx_target="#result",
    )


@rt("/submit_run")
async def post_submit_run(req: Request) -> str:
    """Process form to submit a pipeline run. Run this in the background."""
    run_form_renderer = PydanticForm(
        "run_form",
        DataExtractionConfig,
    )
    config: DataExtractionConfig = await run_form_renderer.model_validate_request(req)

    with p.p.joinpath("run-settings.yaml").open("w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    proc = Process(target=batch_pipeline)
    # you have to set daemon true to not have to wait for the process to join
    proc.daemon = True
    proc.start()

    return "Submitted a job. It's probably running. Please refresh the page"


def run_detail_button(
    run_path: str, run_detail: Literal["metrics", "config", "form", "log"]
) -> fh.FT:
    """Generate a button to get details about a run of type `run_detail`."""
    match run_detail:
        case "metrics":
            get_path = fill_metric_modal.to(run_path=run_path)
        case "config":
            get_path = fill_config_modal.to(run_path=run_path)
        case "form":
            get_path = ""
        case "log":
            get_path = ""

    return mui.Td(
        fh.A(f"Show {run_detail}"),
        hx_get=get_path,
        hx_target="#run-modal-content",
        hx_swap="innerHTML",
        data_uk_toggle="target: #run-modal",
    )


@rt()
def list_runs(batch: str | None = None) -> fh.FT:
    """Return a page with a table of pipeline runs."""

    def cell_render(col: str, val: str) -> fh.FT:
        match col:
            case "config":
                return run_detail_button(val, "config")
            case "metrics":
                return run_detail_button(val, "metrics")
            case _:
                return mui.Td(val)

    if batch is not None:
        batch_path = Path(batch)
        runs = [x for x in batch_path.iterdir() if "run_" in x.name]
    else:
        runs = list(p.all_runs())

    runs.sort()

    run_data = [
        {"batch": run.parts[1], "run": run.parts[2], "config": run, "metrics": run}
        for run in runs
    ]

    table = mui.TableFromDicts(
        ["batch", "run", "config", "metrics"],
        body_data=run_data,
        body_cell_render=cell_render,
        sortable=True,
        cls=(mui.TableT.responsive, mui.TableT.sm, mui.TableT.divider),
    )

    run_table_controls = mui.Button(
        "Create Pipeline run",
        cls=(mui.ButtonT.primary, mui.TextPresets.bold_sm),
        data_uk_toggle="target: #run-modal",
        hx_get=fill_run_form.to(),
        hx_target="#run-modal-content",
        # hx_swap="innerHTML",
    )

    run_ui = fh.Div(
        mui.DivFullySpaced(mui.DivLAligned(run_table_controls), cls="mt-8"),
        table,
        # footer()
    )

    run_modal = mui.Modal(fh.Div(id="run-modal-content"), id="run-modal")

    content = mui.Container(run_ui, run_modal)
    return layout(content)


@rt()
def index() -> fh.FT:
    """Return the homepage."""
    content = fh.Div(fh.P("Home"))

    return layout(content)
