"""Interactive demo for custom_hierarchical.py: parse PDFs, then run single-study and batch extraction.

"""


from __future__ import annotations

import json
from pathlib import Path

from deet.custom_hierarchical import (
    run_dynamic_batch_extraction_from_csv_schema,
    run_dynamic_extraction_from_csv_schema,
    write_hierarchical_prompts_csv,
)
from deet.processors.parser import parse_folder_to_markdown



#################################
input_folder = "misc/hierarchical_mvp/input/batch_pdfs"

created = parse_folder_to_markdown(input_folder)
print(f"Created {len(created)} markdown file(s).")

#########################################################
csv_path = Path("misc/hierarchical_mvp/configs/hierarchical_prompts_RCT.csv")
# Export the current extraction schema (study_type="RCT") to csv_path for editing.
write_hierarchical_prompts_csv(study_type="RCT", csv_outpath=csv_path)

# #############################################################
single_study_config = {
    "study_type": "RCT",
    "llm_model": "azure/gpt-5.6-luna",
    "max_tokens": 30000,
    "dspy_cache": False,
    "input_paths": ["misc/hierarchical_mvp/input/highqual_RCT/main.md"],
    "output_parent_dir": "misc/hierarchical_mvp/output/mira_rct",
    "export_csv": False,
    "export_xlsx": True,
    "export_json": False
}

single_study_config_path = Path("misc/hierarchical_mvp/configs/demo_single_config_RCT.json")
single_study_config_path.write_text(
    json.dumps(single_study_config, indent=2), encoding="utf-8"
)


###################
output_path = run_dynamic_extraction_from_csv_schema(
    csv_path=csv_path,
    config_path=single_study_config_path,
)
print(f"Extraction complete. Output saved to: {output_path}")

# # ##########################################################################################Batch config 
batch_config = {
    "study_type": "RCT",
    "llm_model": "azure/gpt-5.6-luna",
    "max_tokens": 30000,
    "dspy_cache": False,
    "input_folder": "misc/hierarchical_mvp/input/batch_pdfs",
    "output_parent_dir": "misc/hierarchical_mvp/output/batch_demo",
    "export_csv": False,
    "export_xlsx": True,
    "export_json": False
}

batch_config_path = Path("misc/hierarchical_mvp/configs/demo_batch_config_RCTS.json")
batch_config_path.write_text(json.dumps(batch_config, indent=2), encoding="utf-8")


output_paths = run_dynamic_batch_extraction_from_csv_schema(
    csv_path=csv_path,
    batch_config_path=batch_config_path,
)
for output_path in output_paths:
    print(output_path)
