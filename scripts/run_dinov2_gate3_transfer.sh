#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/dinov2_gate3_transfer}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/user_inputs/cifar100_data_root}"
PROXY_CSV="${PROXY_CSV:-${ROOT_DIR}/user_inputs/cifar100_dinov2_proxy.csv}"
FIRST_LEARNING="${FIRST_LEARNING:-${ROOT_DIR}/user_inputs/first_learning_epoch.npy}"
RANDOM_BASELINE="${RANDOM_BASELINE:-${ROOT_DIR}/user_inputs/cifar100_random_seed_level_results.csv}"

SEEDS="${SEEDS:-0 1 2}"
BUDGETS="${BUDGETS:-0.1}"
METHODS="${METHODS:-DINOv2-Isolation-Hard DINOv2-Isolation-Easy}"
EPOCHS="${EPOCHS:-200}"
EVAL_EPOCHS="${EVAL_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPUS="${GPUS:-1 2 3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

mkdir -p "${OUT_ROOT}/logs"

if [[ ! -f "${PROXY_CSV}" || ! -f "${FIRST_LEARNING}" || ! -f "${RANDOM_BASELINE}" ]]; then
  cat <<EOF
DINOv2 Gate 3 transfer requires user-supplied upstream assets:
  PROXY_CSV=${PROXY_CSV}
  FIRST_LEARNING=${FIRST_LEARNING}
  RANDOM_BASELINE=${RANDOM_BASELINE}
It also requires CIFAR-100 under DATA_ROOT and PyTorch/torchvision.
Use cached_summaries/dinov2_gate3_*.csv for lightweight verification.
EOF
  exit 2
fi

run_one() {
  local seed="$1"
  local budget="$2"
  local method="$3"
  local gpu="$4"
  local safe_method="${method//[^A-Za-z0-9]/}"
  local safe_budget="${budget/./p}"
  local log_path="${OUT_ROOT}/logs/seed${seed}_budget${safe_budget}_${safe_method}.log"

  echo "[DINOv2 Gate3] seed=${seed} budget=${budget} method=${method} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/run_dinov2_gate3_transfer.py" \
    --proxy-csv "${PROXY_CSV}" \
    --first-learning "${FIRST_LEARNING}" \
    --out-dir "${OUT_ROOT}" \
    --seeds "${seed}" \
    --budgets "${budget}" \
    --methods "${method}" \
    --epochs "${EPOCHS}" \
    --eval-epochs "${EVAL_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --data-root "${DATA_ROOT}" \
    --device cuda > "${log_path}" 2>&1
}

gpu_list=(${GPUS})
job_idx=0
running=0

for seed in ${SEEDS}; do
  for budget in ${BUDGETS}; do
    for method in ${METHODS}; do
      gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"
      run_one "${seed}" "${budget}" "${method}" "${gpu}" &
      job_idx=$((job_idx + 1))
      running=$((running + 1))
      if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
        wait -n
        running=$((running - 1))
      fi
    done
  done
done

wait

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_dinov2_gate3_transfer.py" \
  --result-root "${OUT_ROOT}" \
  --random-baseline "${RANDOM_BASELINE}" \
  --proxy-csv "${PROXY_CSV}" \
  --first-learning "${FIRST_LEARNING}" \
  --data-root "${DATA_ROOT}"

echo "[DINOv2 Gate3] completed under ${OUT_ROOT}"
