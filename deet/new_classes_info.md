I need a new study_type implemented for data extraction, called `<StudyTypeName>`.

It needs a `<StudyTypeName>model.py` and `<StudyTypeName>extraction.py` in deet/hierarchical_mvp/,
and must be wired into both main_hierarchical.py and custom_hierarchical.py, with the same
export functionality as other study types.

## Shape
- Number of top-level entity lists (e.g. interventions/arms): `<N>`
- Number of distinct outcome list types (e.g. RCT has 3: dichotomous/continuous/other;
  Prognostic has 2: hazard_ratio/other; ClimateCarbonPricing has 1: effect_outcomes): `<N>`
- Closest existing analog to mirror structurally (RCT-style / Prognostic-style /
  ClimateCarbonPricing-style / none — new shape): `<answer>`

## Classes
For each class, give: class name, and a table of fields as
  field_name | type (str/int/float/bool/list[...]) | description (include valid choice-list
  options inline in the description text, exactly as they should appear to the LLM)

### <ClassName1> (e.g. Study_Characteristics)
| field_name | type | description |
|---|---|---|
| ... | ... | ... |

### <ClassName2> (e.g. Intervention / entity-group class)
| field_name | type | description |
|---|---|---|

### <ClassName3...N> (one per outcome type, if more than one)
| field_name | type | description |
|---|---|---|

## Study composition
- study_characteristics: <ClassName1>
- <entity_list_field_name>: list[<ClassName2>]
- <outcome_list_field_name(s)>: list[<ClassNameN>] (one field per outcome type)

## Notes
- Flag any field name that isn't a valid Python identifier (spaces, special chars) so I know
  to rename it — tell me what to rename it to.
- Flag any class/field names that intentionally reuse names from existing study types
  (e.g. "Study", "Intervention", "Study_Characteristics") — this is safe since each CSV
  schema only ever describes one study type at a time, but I should know it's intentional.
- Anything else non-standard (e.g. no intervention concept at all, nested sub-lists, etc.)