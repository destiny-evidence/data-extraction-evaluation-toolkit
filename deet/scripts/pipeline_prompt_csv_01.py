"""
An example of a pipeline script where prompts are supplied
by the user, asynchronously.

This is a two-part pipeline.
This script is part 01, where the data is ingested, and
then the attributes are exported into csv format for
the user to edit prompts.

This populated csv file then needs to be supplied in
`pipeline_prompt_csv_02.py` in order to use the prompts
in the data extraction stage.
"""

import argparse
from pathlib import Path

from deet.data_models.pipeline import Pipeline, jobify, stage_from_job
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter

converter = EppiAnnotationConverter()


def ingest_gold_standard_export_csv_func(
    eppi_json_path: Path, output_dir: Path, csv_path: Path
) -> None:
    """Convert EPPI JSON to DEET data models."""
    out = converter.process_annotation_file(eppi_json_path)

    # export attributes to csv for prompt editing.
    if csv_path.parent == Path("."):  # noqa: PTH201
        csv_path = output_dir / csv_path
    out.export_attributes_csv_file(filepath=csv_path)
    converter.save_processed_data(processed_data=out, output_dir=output_dir)


def main() -> None:
    """Run main part of script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--eppi_json_path", help="path to eppi json", type=Path, required=True
    )

    parser.add_argument(
        "-c",
        "--csv_path",
        help="path you want to write prompt editing csv to.",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    eppi_json_dir = str(Path(args.eppi_json_path).name).split(".")[:-1][0]
    eppi_out_path = (
        Path(args.eppi_json_path).parent / "tmp_parsed_eppi" / eppi_json_dir / "eppi"
    )

    ingest_gs_csv_stage = stage_from_job(
        jobify(
            name="ingest_gs_csv",
            func_kwargs={
                "eppi_json_path": args.eppi_json_path,
                "output_dir": eppi_out_path,
                "csv_path": args.csv_path,
            },
        )(ingest_gold_standard_export_csv_func)
    )

    my_beautiful_pipeline = Pipeline(
        name="test_pipeline",
        stages=[ingest_gs_csv_stage],
    )

    my_beautiful_pipeline.run()


if __name__ == "__main__":
    main()
