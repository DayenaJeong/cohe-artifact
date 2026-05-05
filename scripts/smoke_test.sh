#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/smoke_test"
PY="${PYTHON:-python3}"

mkdir -p "$OUT"

"$PY" "$ROOT/scripts/run_stage1_dependence.py" \
  --proxy "$ROOT/examples/toy_proxy_scores.csv" \
  --target "$ROOT/examples/toy_target_scores.csv" \
  --out "$OUT/stage1_dependence.csv"

"$PY" "$ROOT/scripts/run_stage2_prediction.py" \
  --proxy "$ROOT/examples/toy_proxy_scores.csv" \
  --target "$ROOT/examples/toy_target_scores.csv" \
  --splits "$ROOT/examples/toy_split_ids.csv" \
  --out "$OUT/stage2_prediction.csv"

"$PY" "$ROOT/scripts/run_stage3_transfer_summary.py" \
  --transfer "$ROOT/examples/toy_transfer_results.csv" \
  --out "$OUT/stage3_transfer.csv"

"$PY" "$ROOT/scripts/generate_claim_card.py" \
  --stage1 "$OUT/stage1_dependence.csv" \
  --stage2 "$OUT/stage2_prediction.csv" \
  --stage3 "$OUT/stage3_transfer.csv" \
  --out "$OUT/cohe_claim_card.md"

"$PY" "$ROOT/scripts/verify_cached_summaries.py"
"$PY" "$ROOT/scripts/render_cached_tables.py"

test -s "$OUT/stage1_dependence.csv"
test -s "$OUT/stage2_prediction.csv"
test -s "$OUT/stage3_transfer.csv"
test -s "$OUT/cohe_claim_card.md"
test -s "$ROOT/outputs/cached_summary_inventory.md"
test -s "$ROOT/outputs/rendered_cached_tables.md"

echo "COHE smoke test completed: $OUT"
