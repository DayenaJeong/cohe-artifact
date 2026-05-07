#!/usr/bin/env python3
"""Verify released derived scalar arrays.

The check is intentionally conservative: it verifies row counts, declared
columns, duplicate IDs, and absence of local paths or identifying strings in
released tabular/text files.
"""

import csv
import json
from pathlib import Path
import re
import sys


FORBIDDEN_TERMS = [
    "/ho" + "me/",
    "/mnt/" + "data",
    "pos" + "eidon",
    "Di" + "ana",
    "Day" + "ena",
    "Je" + "ong",
    "Sung" + "lok",
    "Ch" + "oi",
    "KI" + "ST",
    "Seoul" + "Tech",
    "github" + ".com/",
]

FORBIDDEN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (re.escape(term) for term in FORBIDDEN_TERMS)
]


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def scan_text(path):
    text = path.read_text(errors="ignore")
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            fail(f"Forbidden local/private string in {path}: {pattern.pattern}")


def check_ids(rows, path, expected_n):
    if not rows:
        fail(f"Empty CSV: {path}")
    if "sample_id" not in rows[0]:
        fail(f"Missing sample_id column: {path}")
    ids = [row["sample_id"] for row in rows]
    if any(not value.strip() for value in ids):
        fail(f"Blank sample_id in {path}")
    if len(set(ids)) != len(ids):
        fail(f"Duplicate sample_id values in {path}")
    if len(rows) != expected_n:
        fail(f"Row count mismatch for {path}: {len(rows)} != {expected_n}")


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "derived_scalar_arrays"
    manifest_path = folder / "manifest.json"
    schema_path = folder / "schema.json"
    if not manifest_path.exists():
        fail(f"Missing {manifest_path.relative_to(root)}")
    if not schema_path.exists():
        fail(f"Missing {schema_path.relative_to(root)}")

    manifest = json.loads(manifest_path.read_text())
    schema = json.loads(schema_path.read_text())
    expected_n = int(manifest["row_count"])
    declared_columns = list(schema["required_columns"])

    for rel in [
        "derived_scalar_arrays/README.md",
        "derived_scalar_arrays/manifest.json",
        "derived_scalar_arrays/schema.json",
    ]:
        scan_text(root / rel)

    wide_path = root / schema["primary_file"]
    if not wide_path.exists():
        fail(f"Missing primary scalar CSV: {wide_path.relative_to(root)}")
    scan_text(wide_path)
    wide_rows = read_csv(wide_path)
    check_ids(wide_rows, wide_path.relative_to(root), expected_n)
    missing = [col for col in declared_columns if col not in wide_rows[0]]
    if missing:
        fail(f"Primary CSV missing declared columns: {missing}")

    for rel in schema["convenience_projection_files"]:
        path = folder / rel
        if not path.exists():
            fail(f"Missing projection file: {path.relative_to(root)}")
        scan_text(path)
        rows = read_csv(path)
        check_ids(rows, path.relative_to(root), expected_n)
        columns = set(rows[0])
        if path.name == "cifar100_split_ids.csv":
            required = {"sample_id", "split"}
            values = {row["split"] for row in rows}
            if values - {"train", "test"}:
                fail(f"Unexpected split values in {path.relative_to(root)}: {sorted(values)}")
        elif "proxy" in path.name:
            required = {"sample_id", "proxy_name", "proxy_score"}
        else:
            required = {"sample_id", "target_name", "target_score"}
        if not required <= columns:
            fail(f"{path.relative_to(root)} missing columns: {sorted(required - columns)}")

    print(f"Verified derived scalar arrays: {expected_n} rows, {len(declared_columns)} wide columns.")


if __name__ == "__main__":
    main()
