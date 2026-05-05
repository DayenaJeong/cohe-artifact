#!/usr/bin/env python3
"""Generate a conservative Markdown COHE claim card from stage summaries."""

import argparse
import csv
from pathlib import Path


def read_first(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1")
    parser.add_argument("--stage2")
    parser.add_argument("--stage3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    s1 = read_first(args.stage1)
    s2 = read_first(args.stage2)
    stage3_exists = bool(args.stage3 and Path(args.stage3).exists())

    r2 = float(s2.get("heldout_r2", "nan")) if s2 else float("nan")
    if not stage3_exists:
        tier = "alignment_or_predictive_validity_only"
        action = "Do not describe as operational transfer until a downstream action protocol is evaluated."
    elif r2 <= 0.005:
        tier = "no_reliable_transfer_or_weak_surrogate_evidence"
        action = "Use conservative wording and check transfer diagnostics before deployment claims."
    else:
        tier = "requires_user_review"
        action = "Inspect dependence, predictive validity, transfer deltas, and diagnostics before assigning a final tier."

    text = f"""# COHE Claim Card

## Stage 1: Dependence

- n: {s1.get('n', 'not reported')}
- Pearson r: {s1.get('pearson_r', 'not reported')}
- Spearman rho: {s1.get('spearman_rho', 'not reported')}
- Nonlinear check: {s1.get('nonlinear_check', 'not reported')}

## Stage 2: Prediction

- Held-out R2: {s2.get('heldout_r2', 'not reported')}
- Model: {s2.get('model', 'not reported')}

## Stage 3: Transfer

- Transfer summary supplied: {stage3_exists}

## Conservative Draft Tier

- Draft tier: {tier}
- Recommended action: {action}

Final claim-tier assignment should be checked by the user against the COHE reporting checklist and the intended downstream claim.
"""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
