# Glossary

term:Ground Truth
: The "ground truth" data manually extracted by human experts. Used as the benchmark against which to evaluate AI-extracted data.

term:experiment configuration
: All of the variables that can be configured to affect how well LLMs perform a data extraction task. Includes the prompts used.

term:data extraction experiment
: An instance of an automated data extraction pipeline, used to evaluate a specific experiment configuration.

term:attribute presence
: Whether a gold-standard annotation for a given attribute exists on a document. In the comparison CSV this appears as `attribute_presence` (`"True"` or `"False"`). Absence does not always mean the gold value used for scoring is empty: missing annotations may still be represented with a type-specific default when metrics are computed.

term:citation page
: Page number(s) parsed from EPPI citation markup in gold full-text details (comparison CSV column `citation_page`). Empty when markup is missing or not EPPI-sourced.

term:citation highlight
: Highlight / quoted text extracted from EPPI citation markup after parsing and cleaning (comparison CSV column `citation_highlight_text`).

term:verbatim fuzzy match
: A 0–100 score for how well a support snippet is grounded in the document text used for extraction. For gold rows, the snippet is `human_additional_text` (the human annotator's supporting / verbatim text on the gold annotation). For LLM rows, the snippet is `llm_verbatim_text` (the model's `additional_text`). Scores are written to `human_verbatim_fuzzy_match_pct` and `llm_verbatim_fuzzy_match_pct` in the comparison CSV. Empty snippets score `0`.

term:good source instance
: For STRING / INTEGER / FLOAT evaluation, a scored document-attribute instance where the gold value is found in parsed `context` (the text the LLM read). Counted in `n_good_source_instances`.

term:gold value in citation
: For STRING / INTEGER / FLOAT evaluation, whether the gold value is found in citation text (`additional_text` and/or `item_attribute_full_text_details`). Exported per row as `gold_value_in_citation` in `goldstandard_llm_comparison.csv`.

term:match status
: Row-level outcome label in `goldstandard_llm_comparison.csv` for STRING / INTEGER / FLOAT (`match_status` column). Values: `exact_match` (gold equals LLM), `near_match` (STRING only: normalised Levenshtein similarity at or above the edit-distance threshold), `missing_prediction` (no LLM value), `extraction_error_bad_source` (mismatch and gold not in parsed context), `extraction_error_good_source` (mismatch and gold is in parsed context). Empty for BOOL rows.
