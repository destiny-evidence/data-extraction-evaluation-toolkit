import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Config:
    csv_path: Path
    out_json_path: Path
    write_md: bool
    md_dir: Path | None
    delimit_str: str
    nlreplace: str
    ignore_columns: list[str]
    meta_variable_dict: dict[str, str]


def _resolve_path(p: str, base_dir: Path) -> Path:
    """Resolve paths relative to the config file directory (if not absolute)."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

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

    # Basic validation
    if "ItemId" not in meta_variable_dict:
        raise ValueError("meta_variable_dict must include an 'ItemId' mapping.")
    if not isinstance(ignore_columns, list):
        raise TypeError("ignore_columns must be a list of column names.")
    if not isinstance(meta_variable_dict, dict):
        raise TypeError("meta_variable_dict must be an object/dict.")

    return Config(
        csv_path=csv_path,
        out_json_path=out_json_path,
        write_md=write_md,
        md_dir=md_dir,
        delimit_str=delimit_str,
        nlreplace=nlreplace,
        ignore_columns=ignore_columns,
        meta_variable_dict=meta_variable_dict,
    )


def get_metadata(
    row, i: int, meta_variable_dict: dict[str, str], nlreplace: str
) -> dict[str, Any]:
    mycols = row.keys()
    refdict: dict[str, Any] = {"Codes": [], "Outcomes": []}

    try:
        unique_ID_int = int(row[meta_variable_dict["ItemId"]])
        refdict["ItemId"] = unique_ID_int
    except Exception:
        print("Unable to retrieve int ID, using index instead")
        refdict["ItemId"] = i

    # iterate through all required fields and fill them if possible from the row data
    for cvar in meta_variable_dict:
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

    return refdict


def add_md(refdict: dict[str, Any], md_dir: Path) -> None:
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
    md_dir: Path | None,
    meta_variable_dict: dict[str, str],
    ignore_columns: list[str],
    delimit_str: str,
    nlreplace: str,
) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8").fillna("")

    reflist: list[dict[str, Any]] = []
    attributeslist: list[dict[str, Any]] = []

    remove_set = set(ignore_columns) | set(meta_variable_dict.values())
    mapping_vars = [x for x in df.columns if x not in remove_set]

    print(
        f"Transferring data from the following columns into JSON attributes: {mapping_vars}"
    )

    # Collect all codes and assign new ids as discovered
    attributes: dict[str, dict[str, int]] = {k: {} for k in mapping_vars}
    cnt = 1  # attribute ID counter (for leaf attribute values)

    for i, row in df.iterrows():
        myattributes: list[int] = []

        for mvar in mapping_vars:
            thisvar = str(row[mvar])
            thiscodes = [s.strip() for s in thisvar.split(delimit_str)]
            thiscodes = [c for c in thiscodes if c]  # remove empty

            for c in thiscodes:
                if c not in attributes[mvar]:
                    attributes[mvar][c] = cnt
                    myattributes.append(cnt)
                    cnt += 1
                else:
                    myattributes.append(attributes[mvar][c])

        refdict = get_metadata(row, i, meta_variable_dict, nlreplace=nlreplace)

        if write_md:
            if md_dir is None:
                raise ValueError(
                    "write_md is true but md_dir was not provided in config."
                )
            add_md(refdict, md_dir)

        # Reformat reference-level attribute list
        codelist = [
            {"AttributeId": att, "ItemAttributeFullTextDetails": []}
            for att in myattributes
        ]
        refdict["Codes"] = codelist
        reflist.append(refdict)

    # Reformat global-level attributes list
    for key, value in attributes.items():
        thisattributes = [
            {"AttributeId": v, "AttributeName": k} for k, v in value.items()
        ]
        adict = {
            "AttributeId": cnt,
            "AttributeName": key,
            "Attributes": {"AttributesList": thisattributes},
        }
        attributeslist.append(adict)
        cnt += 1  # parent-level attributes also need ids

    final_json = {
        "CodeSets": [
            {
                "SetName": "Mapping tool",
                "Attributes": {"AttributesList": attributeslist},
            }
        ],
        "References": reflist,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    print(f"Wrote JSON: {out_json_path}")
    if write_md and md_dir:
        print(f"Wrote MD files to: {md_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CSV into EPPI-style JSON with coded attributes."
    )
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
