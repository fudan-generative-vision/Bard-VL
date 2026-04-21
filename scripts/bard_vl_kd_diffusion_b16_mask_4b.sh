#!/bin/bash

set -euo pipefail

source /path/to/miniconda3/bin/activate # modify
conda activate bard-vl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
root_path="${REPO_ROOT}"

export PYTHONPATH="${root_path}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29620}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_MACHINES=${WORLD_SIZE}
TOTAL_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))

POSITIONAL_ARGS=("$@")
if [ -z "${NODE_RANK:-}" ]; then
  if [ "${#POSITIONAL_ARGS[@]}" -gt 0 ] && [[ "${POSITIONAL_ARGS[0]}" =~ ^[0-9]+$ ]]; then
    NODE_RANK="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
  else
    NODE_RANK=0
  fi
fi

cuda_visible_devices() {
  local num_gpus=$1
  local devices=()
  local idx
  for ((idx = 0; idx < num_gpus; idx++)); do
    devices+=("${idx}")
  done
  local joined
  IFS=,
  joined="${devices[*]}"
  unset IFS
  printf '%s\n' "${joined}"
}

count_visible_devices() {
  local visible_devices=$1
  awk -F',' 'BEGIN { count = 0 } { for (i = 1; i <= NF; i++) if ($i != "") count++ } END { print count }' <<< "${visible_devices}"
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(cuda_visible_devices "${NUM_GPUS_PER_NODE}")}
VISIBLE_GPU_COUNT=$(count_visible_devices "${CUDA_VISIBLE_DEVICES}")
if [ "${VISIBLE_GPU_COUNT}" -ne "${NUM_GPUS_PER_NODE}" ]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${VISIBLE_GPU_COUNT} GPU(s), but NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}."
  echo "[ERROR] accelerate will still launch ${NUM_GPUS_PER_NODE} process(es) per node."
  exit 1
fi

CONFIG="${CONFIG:-examples/distillation/bard_vl_kd_diffusion_b16_mask_4b.yaml}"
PROJECT="${PROJECT:-exps/bard_vl_kd_diffusion_b16_mask_4b}"
DEEPSPEED_CONFIG_FILE="${DEEPSPEED_CONFIG_FILE:-accelerate_configs/deepspeed_zero1.json}"
if [[ "${DEEPSPEED_CONFIG_FILE}" != /* ]]; then
  DEEPSPEED_CONFIG_FILE="${root_path}/${DEEPSPEED_CONFIG_FILE}"
fi
TRAIN_LOG="${TRAIN_LOG:-${PROJECT}/train.log}"

EXTRA_ARGS=("${POSITIONAL_ARGS[@]}")

mkdir -p "${PROJECT}"
mkdir -p "$(dirname "${TRAIN_LOG}")"

RUNTIME_MIXED_PRECISION="${RUNTIME_MIXED_PRECISION:-bf16}"

cleanup_startup() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill "${TRAIN_PID}" >/dev/null 2>&1 || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup_startup EXIT INT TERM HUP

TRAIN_CMD=(
  accelerate launch
  --use_deepspeed
  --deepspeed_config_file "${DEEPSPEED_CONFIG_FILE}"
  --mixed_precision "${RUNTIME_MIXED_PRECISION}"
  --main_process_port "${MASTER_PORT}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --same_network
  --rdzv_backend static
  --num_machines "${NUM_MACHINES}"
  --num_processes "${TOTAL_GPUS}"
  train/kd_diffusion.py
  "config=${CONFIG}"
  "experiment.project=${PROJECT}"
  "${EXTRA_ARGS[@]}"
)

echo "[CONFIG] ${CONFIG}"
echo "[DEEPSPEED CONFIG] ${DEEPSPEED_CONFIG_FILE}"
echo "[NODES] ${NUM_MACHINES} | [GPUS/NODE] ${NUM_GPUS_PER_NODE} | [TOTAL GPUS] ${TOTAL_GPUS}"
echo "[MASTER] ${MASTER_ADDR}:${MASTER_PORT} | [NODE_RANK] ${NODE_RANK}"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES}"
echo "[MIXED_PRECISION] ${RUNTIME_MIXED_PRECISION}"

nohup "${TRAIN_CMD[@]}" > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!

sleep 5
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
  echo "training process died, tailing log:"
  tail -n 200 "${TRAIN_LOG}" || true
  exit 1
fi

trap - EXIT INT TERM HUP

echo "train pid: ${TRAIN_PID}"
echo "train log: ${TRAIN_LOG}"
