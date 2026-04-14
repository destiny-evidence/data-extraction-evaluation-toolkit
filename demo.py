from pathlib import Path

from deet.data_models.enums import CustomPromptPopulationMethod
from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.cli import (
    DEFAULT_CONFIG_PATH,
    export_config_template,
    extract_data,
    init_linkage_mapping_file,
    init_prompt_csv,
    link_documents_fulltexts,
)


def main() -> None:
    # Adjust this if you place the script somewhere else.
    repo_root = Path(__file__).resolve().parent

    # Paths for this demo
    gs_path = repo_root / "misc" / "ailbhe2" / "HPV_demo_subset.json"
    link_map = repo_root / "misc" / "ailbhe2" / "link_map.csv"
    pdf_dir = repo_root / "misc" / "ailbhe2" / "pdfs"  # must exist for linking
    linked_docs_dir = repo_root / "misc" / "ailbhe2" / "linked_documents"
    prompts_csv = repo_root / "misc" / "ailbhe2" / "prompt_definitions.csv"
    out_dir = repo_root / "misc" / "ailbhe2" / "data-extraction-experiments"

    # Use EPPI JSON import format
    fmt = SupportedImportFormat.EPPI_JSON

    # 1. Ensure we have a default config
    if not DEFAULT_CONFIG_PATH.exists():
        export_config_template(output_path=DEFAULT_CONFIG_PATH)

    # 2. Create linkage CSV (only if it doesn't already exist)
    if not link_map.exists():
        init_linkage_mapping_file(
            gs_data_path=gs_path,
            gs_data_format=fmt,
            link_map_path=link_map,
        )
        print(f"Created linkage mapping CSV at: {link_map}")
        print("Re-run this script to continue with linking and extraction.")
        return
    print(f"Linkage mapping file already exists at: {link_map}, skipping creation.")

    # # 3. Link documents to full texts (requires PDFs in pdf_dir)
    link_documents_fulltexts(
        gs_data_path=gs_path,
        link_map_path=link_map,
        gs_data_format=fmt,
        pdf_dir=pdf_dir,
        output_path=linked_docs_dir,
    )
    print(f"Linked documents saved to: {linked_docs_dir}")

    # # 4. Create prompt CSV for this dataset (only if it doesn't already exist)
    if not prompts_csv.exists():
        init_prompt_csv(
            gs_data_path=gs_path,
            gs_data_format=fmt,
            csv_path=prompts_csv,
        )
        print(f"Created prompt definitions CSV at: {prompts_csv}")
        print("Re-run this script to continue with extraction.")
        return
    print(
        f"Prompt definitions CSV already exists at: {prompts_csv}, skipping creation."
    )

    # # 5. Run extraction using prompts from the CSV
    extract_data(
        gs_data_path=gs_path,
        config_path=DEFAULT_CONFIG_PATH,
        gs_data_format=fmt,
        prompt_population=CustomPromptPopulationMethod.FILE,
        csv_path=prompts_csv,
        linked_document_path=linked_docs_dir,
        link_map_path=link_map,
        pdf_dir=pdf_dir,
        out_dir=out_dir,
        run_name="effect_demo",
        custom_evaluation_metrics=None,
    )
    print(f"Extraction results saved to: {out_dir}")


if __name__ == "__main__":
    main()
