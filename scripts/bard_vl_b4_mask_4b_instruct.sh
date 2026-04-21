#!/bin/bash

source /path/to/miniconda3/bin/activate # modify
conda activate bard-vl

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=${MASTER_PORT:-29900}
NODE_RANK=${1:-0}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_FILE="examples/vlm_finetune/bard_vl/bard_vl_b4_mask_4b_instruct.yaml"
# LOG_FILE="logs/b4_mask_4b_instruct_${NODE_RANK}_${TIMESTAMP}.log"
LOG_FILE="logs/bard_vl_b4_mask_4b_instruct.log"
mkdir -p "$(dirname "$LOG_FILE")"

nohup torchrun --nproc-per-node=8 \
  --nnodes=1 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  examples/vlm_finetune/finetune.py \
  --config $CONFIG_FILE > $LOG_FILE 2>&1 &
