# String (and other) output support – branch notes

Notes for implementing string/other output support on a **fresh branch based off `development`**, without carrying over other `string_output` branch changes (e.g. CSV/LLM script refactors).

---

## Where it’s used: two-part CSV pipeline

The **`output_data_type`-from-CSV** behaviour is for the two-part pipeline scripts:

- **`deet/scripts/pipeline_prompt_csv_01.py`**
  Ingests EPPI JSON and **exports** attributes to a CSV via `ProcessedAnnotationData.export_attributes_csv_file()`. The CSV includes columns such as `prompt`, `attribute_id`, `attribute_label`, and (from `Attribute.model_dump()`) **`output_data_type`**. Users edit this CSV (prompts and, if desired, `output_data_type`).

- **`deet/scripts/pipeline_prompt_csv_02.py`**
  Ingests EPPI JSON again and **imports** that CSV via `out.populate_custom_prompts(method="file", filepath=csv_path)`, which calls `ProcessedAnnotationData._import_prompts_csv_file()`. That method populates prompts and, with the string-output changes, **also updates each attribute’s `output_data_type`** from the CSV column when present.

So: the CSV that links part 01 and part 02 is where `output_data_type` can be set per attribute (e.g. `string`, `bool`, `integer`, …) for the extraction run in part 02.

---

## Changes to port (from commit 6badec1 and related)

Focus only on “output = string” (and other types). Key edits:

1. **`deet/data_models/eppi.py`**
   - In **`_import_prompts_csv_file`**: after `populate_prompt_from_dict(row, ...)`, if the row has a non-empty **`output_data_type`** column, parse it (strip, lower), map via `{"string": AttributeType.STRING, "integer": AttributeType.INTEGER, "float": AttributeType.FLOAT, "bool": AttributeType.BOOL, "list": AttributeType.LIST, "dict": AttributeType.DICT}` and set `matching_attribute.output_data_type` accordingly.

2. **`deet/processors/eppi_annotation_converter.py`**
   - When building the attribute dict from EPPI: the line that forced **`"output_data_type": AttributeType.BOOL`** was commented out so the converter doesn’t override; attribute type can then come from `EppiAttribute` default or from the CSV in part 02.
   - When building gold-standard annotations: the line **`output_data=bool(output_data)`** was commented out so human annotations can keep string (or other) values instead of being forced to bool.

3. **Optional / unrelated to “output = string”** (from same commit, can adopt or leave):
   - `Attribute.populate_prompt_from_dict(..., overwrite: bool = False)`
   - `_import_prompts_csv_file(..., overwrite: bool = False)`
   - `llm_data_extractor.py`: `read_text(encoding="utf-8")` for system prompt

These don’t affect string/other output; include only if you want the same defaults and encoding behaviour.

---

## CSV format (already in place on development)

The attributes CSV used between part 01 and part 02 already has an **`output_data_type`** column (from `Attribute.write_to_csv` / `model_dump()`). Example from `misc/eppi/prompts.csv`: `prompt,question_target,output_data_type,attribute_id,attribute_label,...` with values like `bool`. For string extraction, the user can change that column to `string` (or other allowed values) for the relevant rows; part 02 will then set those attributes’ `output_data_type` when importing the CSV.

---

## Suggested workflow

1. Branch from **`development`**: e.g. `git checkout development && git pull && git checkout -b feature/string-output`.
2. Apply only the changes above (eppi.py `_import_prompts_csv_file` + eppi_annotation_converter.py attribute/annotation handling). Omit other `string_output`-branch refactors (directory-loop removal, compare_with_gold_standard, etc.) unless you want them.
3. Run the two-part pipeline (01 → edit CSV → 02) and confirm that attributes with `output_data_type` set to `string` in the CSV are used as string in extraction and in gold-standard annotation loading.

---

## Future improvements (not in 6badec1)

- **Coercion**: Add `AttributeType.coerce(value)` and use it in the LLM extractor when building `GoldStandardAnnotation` so that LLM output (e.g. `"True"` string) is normalized to the attribute’s type (e.g. bool).
- **Validator**: In `GoldStandardAnnotation.ensure_correct_type`, support both `Attribute` and dict-shaped `data["attribute"]` when reading `output_data_type`.
- **Prompts**: Update default prompt wording so it’s clear that the answer type can be boolean or string (or other) depending on the attribute.
