"""Interactive demo for main_hierarchical.py: parse PDFs, then run single-study and batch extraction.

"""

from __future__ import annotations

import json
from pathlib import Path

from deet.main_hierarchical import (
    run_parse_pdfs,
    run_predict_batch,
    run_predict_single_study,
    setup_console_logging,
)

setup_console_logging()

######################################################PDF parsing (optional)
input_folder = "misc/hierarchical_mvp/input/batch_pdfs"
run_parse_pdfs(input_folder)

###FOR CLI usage, activate the venv and run:
## python -m deet.main_hierarchical parse_pdfs "misc\hierarchical_mvp\input\batch_pdfs"

#######################################################Single-study extraction
# single_study_config = {
#     "study_type": "RCT",
#     "llm_model": "anthropic/claude-sonnet-4-5",
#     "max_tokens": 60000,
#     "dspy_cache": False,
#     "input_paths": ["misc/hierarchical_mvp/input/batch_pdfs/mira_rct.md"],
#     "output_parent_dir": "misc/hierarchical_mvp/output/mira_rct",
#     "export_csv": False,
#     "export_xlsx": True,
#     "export_json": False,
# }

# single_study_config_path = Path("misc/hierarchical_mvp/configs/demo_single_config.json")
# single_study_config_path.write_text(
#     json.dumps(single_study_config, indent=2), encoding="utf-8"
# )

# run_predict_single_study(str(single_study_config_path))

###FOR CLI usage, activate the venv, make sure the config file is present, and run:
## python -m deet.main_hierarchical predict_single_study "misc\hierarchical_mvp\configs\demo_single_config.json"

# ########################################################Batch extraction
batch_config = {
    "study_type": "RCT",
    "llm_model": "anthropic/claude-sonnet-4-5",
    "max_tokens": 60000,
    "dspy_cache": False,
    "input_folder": "misc/hierarchical_mvp/input/batch_pdfs",
    "output_parent_dir": "misc/hierarchical_mvp/output/batch_demo",
    "export_csv": False,
    "export_xlsx": True,
    "export_json": False,
}

batch_config_path = Path("misc/hierarchical_mvp/configs/demo_batch_config.json")
batch_config_path.write_text(json.dumps(batch_config, indent=2), encoding="utf-8")

run_predict_batch(str(batch_config_path))
