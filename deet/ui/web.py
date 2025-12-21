"""FastHTML App definition."""

import csv
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
from starlette.responses import RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

from deet.data_models.project import AttributeMetric, DeetProject
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.scripts.batch_pipeline import batch_pipeline
from deet.scripts.create_project import create_project
from deet.scripts.new_batch import new_batch

hdrs = mui.Theme.blue.headers()

app, rt = fh.fast_app(hdrs=hdrs)

p = DeetProject(path=".")


def get_metrics(run_path: Path) -> Generator[fh.FT]:
    """Return a table of metrics for a given pipeline run."""
    metric_path = run_path.joinpath("metrics.json")
    metrics = [AttributeMetric.model_validate(x) for x in json.load(metric_path.open())]
    metrics_names = ["Attribute", *list({x.metric for x in metrics})]
    yield fh.Tr(*[fh.Th(x) for x in metrics_names])
    for k, g in groupby(metrics, lambda x: x.attribute):
        yield fh.Tr(fh.Td(k.attribute_label), *[fh.Td(x.value) for x in g])


def sidebar() -> fh.FT:
    """return a sidebar."""
    contents = [
        ("Project home", "/"),
        ("Documents", "documents"),
        ("Attributes", "attributes"),
        ("Prompts", "prompts"),
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
def prompts() -> fh.FT:
    """Show list of prompt files."""
    prompt_files = [x for x in p.prompt_folder.iterdir() if ".csv" in x.name]
    prompt_links = [
        fh.Li(fh.A(x.name, href=prompt_definitions.to(file=str(x))))
        for x in prompt_files
    ]

    content = mui.DivVStacked(mui.H3("Prompt manager"), fh.Ul(*prompt_links))
    return layout(content)


@rt()
async def update_prompt_definitions(req: Request, file: str) -> RedirectResponse:
    """Write the updated prompt definitions to a new file."""
    form = await req.form()

    fpath = Path(file)
    if "_" in fpath.stem:
        parts = fpath.stem.split("_")
        idx = parts[-1]
        if idx.isdigit():
            int_idx = int(idx)
            idx = str(int_idx + 1)
        else:
            idx = f"{idx}_1"
        parts[-1] = idx
        stem = "_".join(parts)
        new_path = fpath.parent.joinpath(f"{stem}.csv")
    else:
        new_path = fpath.parent.joinpath(f"{fpath.stem}_1.csv")

    with fpath.open() as input_file:  # noqa: ASYNC230
        reader = csv.DictReader(input_file)
        field_names = reader.fieldnames
        if field_names is None:
            raise ValueError
        data = list(reader)

    with new_path.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        for k, v in form.items():
            rows = filter(lambda x: x["attribute_id"] == k, data)
            for row in rows:
                row["prompt"] = v
                writer.writerow(row)

    return RedirectResponse(
        url=prompt_definitions.to(file=str(new_path), update=True),
        status_code=HTTP_303_SEE_OTHER,
    )


@rt()
def prompt_definitions(file: str, *, update: bool = False) -> fh.FT:
    """Return Interface to edit prompt definitions."""
    if update:
        title = mui.H3(f"Successfully updated prompts and saved as {file}")
    else:
        title = mui.H3(f"Manage prompts in {file}")

    subtitle = fh.P("Edit prompts and click to save a new version")
    with Path(file).open() as csvfile:
        data = list(csv.DictReader(csvfile))

    table_data = [
        fh.Tr(
            mui.Td(mui.Input(value=x["prompt"], name=x["attribute_id"])),
            mui.Td(x["attribute_id"]),
            mui.Td(x["attribute_label"]),
        )
        for x in data
    ]

    table = mui.Table(
        fh.Tr(*[mui.Th(x) for x in ["prompt", "attribute_id", "attribute_label"]]),
        *table_data,
    )

    form = mui.Form(
        table,
        fh.Div(
            mui.Button(
                "Submit",
                type="submit",
                cls=mui.ButtonT.primary,
                hx_on="htmx:afterRequest: console.log('R')",
            ),
            cls="mt-4 flex items-center gap-2",
        ),
        action=update_prompt_definitions.to(file=file),
        method="POST",
    )

    content = mui.DivVStacked(title, subtitle, form)
    return layout(content)


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
async def submit_batch(req: Request) -> fh.FT:
    """Process form to create a new batch."""
    form = await req.form()
    batch_size: float = float(form["batch_size"])
    new_batch(batch_size)
    return get_batch_table()


def get_batch_table() -> fh.FT:
    """Get a table with a row for each batch."""
    data = [
        {
            "name": b.parts[1],
            "number of documents": len(json.load(b.joinpath("batch_ids.json").open())),
            "runs": len([x for x in b.iterdir() if "run_" in x.name]),
        }
        for b in p.batches
    ]
    return mui.TableFromDicts(
        ["name", "number of documents", "runs"],
        body_data=data,
        sortable=True,
        cls=(mui.TableT.responsive, mui.TableT.sm, mui.TableT.divider),
    )


@rt()
def batches() -> fh.FT:
    """Show a table of batches."""
    batch_table = get_batch_table()

    batch_modal = mui.Modal(
        fh.Div(
            mui.Form(
                mui.LabelInput(
                    "Batch size", type="number", step="0.01", id="batch_size"
                ),
                fh.P(
                    "enter either an integer or a decimal for a fraction of the"
                    "size of the original dataset",
                    cls=mui.TextPresets.muted_sm,
                ),
                fh.Div(
                    mui.Button(
                        "Submit",
                        type="submit",
                        cls=mui.ButtonT.primary,
                        hx_on="htmx:afterRequest: console.log('R')",
                    ),
                    cls="mt-4 flex items-center gap-2",
                ),
                hx_post="/submit_batch",
                hx_target="#batch-table",
            ),
            id="batch-modal-content",
        ),
        id="batch-modal",
    )

    batch_table_controls = mui.Button(
        "Create new batch",
        cls=(mui.ButtonT.primary, mui.TextPresets.bold_sm),
        data_uk_toggle="target: #batch-modal",
    )

    batch_ui = fh.Div(
        mui.DivFullySpaced(mui.DivLAligned(batch_table_controls), cls="mt-8"),
        fh.Div(
            batch_table,
            id="batch-table",
            hx_on__after_swap="document.getElementById('batch-modal').classList.remove('uk-open')",
        ),
        # footer()
    )
    content = fh.Container(batch_ui, batch_modal)

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
        mui.Table(*get_metrics(Path(run_path))),
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
async def upload_project_data(file: fh.UploadFile) -> fh.FT:
    """Upload and parse EPPI json data."""
    filebuffer = await file.read()
    filepath = p.p.joinpath("raw_data.json")
    filepath.write_bytes(filebuffer)
    create_project("", str(filepath))

    return fh.P("Finished uploading project data, created a project")


@rt()
def index() -> fh.FT:
    """Return the homepage."""
    if not p.folders_exist():
        content = fh.Div(
            mui.Form(
                mui.Input(type="file", name="file"),
                mui.Button("Upload", type="submit", cls="secondary"),
                hx_post="/upload_project_data",
                hx_target="#project-home",
            ),
            id="project-home",
        )
    else:
        content = fh.Div(fh.P("Home"))

    return layout(content)
