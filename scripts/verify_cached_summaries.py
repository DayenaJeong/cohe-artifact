#!/usr/bin/env python3
"""Verify cached aggregate paper-table summaries.

This script checks that every artifact file listed in
cached_summaries/paper_table_index.csv exists and can be parsed as CSV.
It writes a compact Markdown inventory under outputs/.
"""

import json
from pathlib import Path
import sys

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    index_path = root / "cached_summaries" / "paper_table_index.csv"
    out_path = root / "outputs" / "cached_summary_inventory.md"

    if not index_path.exists():
        raise SystemExit(f"Missing index file: {index_path}")

    index = pd.read_csv(index_path)
    required = {"paper_table", "artifact_file", "script_to_verify", "description", "reproduction_scope"}
    missing_columns = required - set(index.columns)
    if missing_columns:
        raise SystemExit(f"Index is missing columns: {sorted(missing_columns)}")

    rows = []
    errors = []
    for record in index.to_dict(orient="records"):
        artifact_rel = str(record["artifact_file"])
        artifact_path = root / artifact_rel
        if not artifact_path.exists():
            errors.append(f"Missing listed artifact_file: {artifact_rel}")
            continue
        if artifact_path.suffix == ".json":
            try:
                payload = json.loads(artifact_path.read_text())
            except Exception as exc:  # pragma: no cover - defensive CLI path
                errors.append(f"Malformed JSON {artifact_rel}: {exc}")
                continue
            if not payload:
                errors.append(f"JSON is empty: {artifact_rel}")
                continue
            rows.append(
                {
                    "paper_table": record["paper_table"],
                    "artifact_file": artifact_rel,
                    "rows": 1,
                    "columns": ", ".join(str(k) for k in payload.keys()),
                }
            )
        else:
            try:
                df = pd.read_csv(artifact_path)
            except Exception as exc:  # pragma: no cover - defensive CLI path
                errors.append(f"Malformed CSV {artifact_rel}: {exc}")
                continue
            if df.empty:
                errors.append(f"CSV has no rows: {artifact_rel}")
                continue
            rows.append(
                {
                    "paper_table": record["paper_table"],
                    "artifact_file": artifact_rel,
                    "rows": len(df),
                    "columns": ", ".join(str(c) for c in df.columns),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Cached Summary Inventory", ""]
    lines.append("| Paper table | Artifact file | Rows | Columns |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['paper_table']} | `{row['artifact_file']}` | {row['rows']} | {row['columns']} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {err}" for err in errors)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Verified {len(rows)} cached summary files.")
    print(f"Wrote {out_path.relative_to(root)}")
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
