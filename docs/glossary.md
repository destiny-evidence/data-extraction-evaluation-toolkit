# Glossary

term:Ground Truth
: The "ground truth" data manually extracted by human experts. Used as the benchmark against which to evaluate AI-extracted data.

term:experiment configuration
: All of the variables that can be configured to affect how well LLMs perform a data extraction task. Includes the prompts used.

term:data extraction experiment
: An instance of an automated data extraction pipeline, used to evaluate a specific experiment configuration

term:attribute presence
: Whether a gold-standard annotation for a given attribute exists on a document. In the comparison CSV this appears as `attribute_presence` (`"True"` or `"False"`). Absence does not always mean the gold value used for scoring is empty: missing annotations may still be represented with a type-specific default when metrics are computed.

term:citation page
: Page number(s) parsed from EPPI citation markup in gold full-text details (comparison CSV column `citation_page`). Empty when markup is missing or not EPPI-sourced.

term:citation highlight
: Highlight / quoted text extracted from EPPI citation markup after parsing and cleaning (comparison CSV column `citation_highlight_text`).

term:verbatim fuzzy match
: A 0–100 score for how well a support snippet is grounded in the document text used for extraction. For gold rows, the snippet is `human_additional_text` (the human annotator's supporting / verbatim text on the gold annotation, e.g. from EPPI reviwer). For LLM rows, the snippet is `llm_verbatim_text` (the model's `additional_text`). Scores are written to `human_verbatim_fuzzy_match_pct` and `llm_verbatim_fuzzy_match_pct` in the comparison CSV. Empty snippets score `0`. *Example:* if the human additional text appears almost unchanged in the document's markdown, the score is near `100`; if the text only partly overlaps the document wording, the score is lower; if it is missing from the markdown or there is no supporting text, the score is near `0`.
