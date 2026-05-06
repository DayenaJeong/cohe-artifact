#!/usr/bin/env python3
"""Render cached aggregate summaries as simple Markdown tables."""

from pathlib import Path
import json

import pandas as pd


def markdown_table(df):
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.fillna("").itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt(cells):
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    lines = [fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    index_path = root / "cached_summaries" / "paper_table_index.csv"
    out_path = root / "outputs" / "rendered_cached_tables.md"

    index = pd.read_csv(index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Rendered Cached Aggregate Tables",
        "",
        "These Markdown tables are rendered from anonymized aggregate CSV summaries.",
        "",
    ]
    for record in index.to_dict(orient="records"):
        artifact_rel = str(record["artifact_file"])
        artifact_path = root / artifact_rel
        if not artifact_path.exists():
            continue
        lines.extend([f"## {record['paper_table']}: {record['description']}", "", f"Source: `{artifact_rel}`", ""])
        if artifact_path.suffix == ".json":
            payload = json.loads(artifact_path.read_text())
            lines.extend(["```json", json.dumps(payload, indent=2), "```", ""])
        else:
            df = pd.read_csv(artifact_path)
            lines.extend([markdown_table(df), ""])

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
