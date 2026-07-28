import pandas as pd
import json
from pathlib import Path
from rich import print

def main():
    df = pd.read_csv("taxonomy_resolved.csv")
    df = df.rename(columns={"item_id": "document_id", "title": "name"})

    with Path("taxonomy_nacsos_mapping.json").open() as f:
        mapping = json.load(f)

        keys = [m["col_pipe"] for m in mapping if m["col_pipe"] in df.columns]

    keep_cols = ["document_id","name","text"]
    keep_cols.extend(keys)
    print(keep_cols)

    df = df[keep_cols]
    df[keys] = df[keys].fillna(0)
    df.to_csv("taxonomy_resolved_clean.csv", index=False)


if __name__=="__main__":
    main()
