"""Demo script: export EPPI gold data, build the reference mapping, and evaluate intervention extractions."""

from __future__ import annotations

from pathlib import Path

from deet.hierarchical_mvp.evaluate_hierarchical import (
    evaluate_interventions,
    summarize_evaluation,
)
from deet.hierarchical_mvp.evaluation_helpers_hierarchical import (
    export_from_eppi,
    generate_reference_mapping_template,
)

#################################
json_path = Path("misc/hierarchical_mvp/input/galenos/gold/GALENOS_LSR1_58studies.json")
gold_xlsx_path = Path("misc/hierarchical_mvp/input/galenos/gold/GALENOS_LSR1_eppi_export.xlsx")
mapping_csv_path = Path("misc/hierarchical_mvp/input/galenos/gold/reference_mapping_demo.csv")
interventions_eval_csv_path = Path(
    "misc/hierarchical_mvp/output/galenos/interventions_evaluation.csv"
)

# 1. Export the EPPI gold standard JSON into the Arms/Outcomes/Timepoints workbook.
export_from_eppi(json_path, gold_xlsx_path)
print(f"EPPI export written to {gold_xlsx_path}")

# 2. Generate the reference -> extraction-xlsx mapping template, if not already filled in.
if not mapping_csv_path.exists():
    generate_reference_mapping_template(gold_xlsx_path, mapping_csv_path)
    print(
        f"Mapping template written to {mapping_csv_path}. Fill in the "
        "'extraction_xlsx_path' column for the references you want to evaluate, "
        "then re-run this script."
    )
else:
    # 3. Evaluate predicted interventions against the gold standard (LLM-as-judge matching).
    evaluate_interventions(mapping_csv_path, gold_xlsx_path, interventions_eval_csv_path)
    print(f"Interventions evaluation written to {interventions_eval_csv_path}")

    # 4. Print summary scores.
    scores = summarize_evaluation(interventions_eval_csv_path)
    print(
        f"TP={scores['TP']} FP={scores['FP']} FN={scores['FN']} TN={scores['TN']} | "
        f"precision={scores['precision']:.3f} recall={scores['recall']:.3f} f1={scores['f1']:.3f}"
    )
