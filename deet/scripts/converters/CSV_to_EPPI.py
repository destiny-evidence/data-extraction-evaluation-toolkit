import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Config:
    csv_path: Path
    out_json_path: Path
    write_md: bool
    md_dir: Optional[Path]
    delimit_str: str
    nlreplace: str
    ignore_columns: List[str]
    meta_variable_dict: Dict[str, str]
    categorical_columns: List[str]


def _resolve_path(p: str, base_dir: Path) -> Path:
    """Resolve paths relative to the config file directory (if not absolute)."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        fnf_msg=f"Config file not found: {config_path}"
        raise FileNotFoundError(fnf_msg)

    base_dir = config_path.parent.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    # Required keys
    csv_path = _resolve_path(raw["csv_path"], base_dir)
    out_json_path = _resolve_path(raw["out_json_path"], base_dir)

    # Optional keys with defaults
    write_md = bool(raw.get("write_md", True))
    md_dir_raw = raw.get("md_dir", None)
    md_dir = _resolve_path(md_dir_raw, base_dir) if (write_md and md_dir_raw) else None

    delimit_str = str(raw.get("delimit_str", ","))
    nlreplace = str(raw.get("nlreplace", ";"))
    ignore_columns = list(raw.get("ignore_columns", []))
    meta_variable_dict = dict(raw.get("meta_variable_dict", {}))
    categorical_columns = list(raw.get("categorical_columns", []))
    # Basic validation
    if "ItemId" not in meta_variable_dict:
        id_msg="meta_variable_dict must include an 'ItemId' mapping."
        raise ValueError(id_msg)
    if not isinstance(ignore_columns, list):
        col_msg="ignore_columns must be a list of column names."
        raise TypeError(col_msg)
    if not isinstance(meta_variable_dict, dict):
        meta_msg="meta_variable_dict must be an object/dict."
        raise TypeError(meta_msg)

    return Config(
        csv_path=csv_path,
        out_json_path=out_json_path,
        write_md=write_md,
        md_dir=md_dir,
        delimit_str=delimit_str,
        nlreplace=nlreplace,
        ignore_columns=ignore_columns,
        meta_variable_dict=meta_variable_dict,
        categorical_columns=categorical_columns

    )


def get_metadata(row, i: int, meta_variable_dict: Dict[str, str], nlreplace: str) -> Dict[str, Any]:
    """
    Formatting metadata for json.
    :param row: 
    :param i: 
    :param meta_variable_dict: 
    :param nlreplace: 
    :return: 
    """
    mycols = row.keys()
    refdict: Dict[str, Any] = {"Codes": [], "Outcomes": []}

    try:
        unique_ID_int = int(row[meta_variable_dict["ItemId"]])
        refdict["ItemId"] = unique_ID_int
    except Exception:
        #print("Unable to retrieve int ID, using index instead")
        refdict["ItemId"] = i

    # iterate through all required fields and fill them if possible from the row data
    for cvar in meta_variable_dict.keys():
        if cvar == "ItemId":
            continue

        mapped_col = meta_variable_dict.get(cvar, "")
        if mapped_col and mapped_col in mycols:
            mv = str(row[mapped_col]).replace("\n", nlreplace).strip()
            if mv.endswith(nlreplace):
                mv = mv[:-1]
            refdict[cvar] = mv
        else:
            refdict[cvar] = ""
    try:
        refdict["Year"] = str(int(refdict["Year"].replace(".0", "")))
        if refdict["TypeName"]=="":
            refdict["TypeName"]= "Journal, Article"
    except:
        pass

    return refdict


def add_md(refdict: Dict[str, Any], md_dir: Path) -> None:
    """
    Add markdown output.
    :param refdict: 
    :param md_dir: 
    :return: 
    """
    md_dir.mkdir(parents=True, exist_ok=True)
    out_path = md_dir / f"{refdict['ItemId']}.md"
    with out_path.open("w", encoding="utf-8") as f:
        if refdict.get("Title"):
            f.write(refdict["Title"])
        if refdict.get("Abstract"):
            f.write(refdict["Abstract"])


def custom_json(
    csv_path: Path,
    out_json_path: Path,
    write_md: bool,
    md_dir: Optional[Path],
    meta_variable_dict: Dict[str, str],
    ignore_columns: List[str],
    delimit_str: str,
    nlreplace: str,
    categorical_data: List[str]
) -> None:
    """
    Create cusom json file.
    :param csv_path: 
    :param out_json_path: 
    :param write_md: 
    :param md_dir: 
    :param meta_variable_dict: 
    :param ignore_columns: 
    :param delimit_str: 
    :param nlreplace: 
    :param categorical_data: 
    :return: 
    """
    df = pd.read_csv(csv_path, encoding="utf-8").fillna("")

    reflist: List[Dict[str, Any]] = []
    attributeslist: List[Dict[str, Any]] = []

    remove_set = set(ignore_columns) | set(meta_variable_dict.values())
    mapping_vars = [x for x in df.columns if x not in remove_set]

    #print(f"Transferring data from the following columns into JSON attributes: {mapping_vars}")
    # Collect all codes and assign new ids as discovered
    attributes: Dict[str, Dict[str, int]] = {k: {} for k in mapping_vars}
    cnt = 1  # attribute ID counter (for leaf attribute values)

    # Pre-assign attribute IDs for non-categorical columns
    # Each non-categorical column gets ONE attribute ID
    for mvar in mapping_vars:
        if mvar not in categorical_data:
            # Use the column name itself as the key for the single attribute
            attributes[mvar][mvar] = cnt
            cnt += 1

    for i, row in df.iterrows():
        myattributes: List[int] = []
        attr_names: List[str]=[]
        my_codelist=[]
        for mvar in mapping_vars:#any binary or string where there is a tickbox

            ###Code for adding non-categorical attributes here
            if mvar not in categorical_data:
                # Get the cell value and convert to string
                value = str(row[mvar]).strip()

                # Check if value is "1" or longer than 1 character
                if value and (value == "1" or len(value) > 1):
                    # Add the attribute ID for this column
                    attr_id = attributes[mvar][mvar]
                    myattributes.append(attr_id)
                    # Store the actual cell value as AdditionalText
                    attr_names.append(value)

        for mvar in categorical_data:#now do the categorical ones
            thisvar = str(row[mvar])
            thiscodes = [s.strip() for s in thisvar.split(delimit_str)]
            thiscodes = [c for c in thiscodes if c]  # remove empty

            for c in thiscodes:
                attr_names.append(c)

                if c not in attributes[mvar]:
                    attributes[mvar][c] = cnt
                    myattributes.append(cnt)
                    cnt += 1
                else:
                    myattributes.append(attributes[mvar][c])

        refdict = get_metadata(row, i, meta_variable_dict, nlreplace=nlreplace)

        if write_md:
            if md_dir is None:
                ve_msg = "write_md is true but md_dir was not provided in config."
                raise ValueError(ve_msg)
            add_md(refdict, md_dir)

        # Reformat reference-level attribute list
        codelist = [{"AttributeId": att,
                     "AdditionalText": attr_names[my_index],
                     "ItemAttributeFullTextDetails": [],
                     "ArmId":0,
                    "ArmTitle":""
                     } for my_index, att in enumerate(myattributes)]

        refdict["Codes"] = codelist
        reflist.append(refdict)

    # Reformat global-level attributes list
    for key, value in attributes.items():
        thisattributes = [{"AttributeId": v,
                           "AttributeName": k,
                           "AttributeType":"Selectable (show checkbox)",
                           "AttributeDescription": "",
                           "ExtURL": "",
                           "ExtType": "",
                           "OriginalAttributeID": 0,
                           "AttributeSetId": 0,
                           "AttributeTypeId": 2,
                           } for k, v in value.items()]
        adict = {
            "AttributeId": cnt,
            "AttributeName": key,
            "AttributeType": "Not selectable (no checkbox)",
            "AttributeDescription": "",
            "ExtURL": "",
            "ExtType": "",
            "OriginalAttributeID": 0,
            "AttributeSetId": 0,
            "AttributeTypeId": 1,
            "Attributes": {"AttributesList": thisattributes},
        }
        attributeslist.append(adict)
        cnt += 1  # parent-level attributes also need ids

    final_json = {
        "CodeSets": [
            {
                "SetName": "Mapping tool",
                "ReviewSetId": 1,
                "SetId": 1,
                "SetType": {
                    "SetTypeName": "Standard",
                    "SetTypeDescription": "The Standard codeset type is used for regular coding such as keywording or data-extraction. This codeset type can contain multiple levels of child codes but cannot contain the special code types \"Include\" and \"Exclude\"."
                },
                "Attributes": {"AttributesList": attributeslist},
            }
        ],
        "References": reflist,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    # print(f"Wrote JSON: {out_json_path}")
    # if write_md and md_dir:
    #     print(f"Wrote MD files to: {md_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CSV into EPPI-style JSON with coded attributes.")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    custom_json(
        csv_path=cfg.csv_path,
        out_json_path=cfg.out_json_path,
        write_md=cfg.write_md,
        md_dir=cfg.md_dir,
        meta_variable_dict=cfg.meta_variable_dict,
        ignore_columns=cfg.ignore_columns,
        delimit_str=cfg.delimit_str,
        nlreplace=cfg.nlreplace,
        categorical_data=cfg.categorical_columns
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
