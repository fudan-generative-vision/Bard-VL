#!/bin/bash
export HF_HOME="datasets"

source /path/to/miniconda3/bin/activate # modify
conda activate bard-vl

export PYTHONPATH=/path/to/Bard-VL # modify
export HF_TOKEN="your huggingface token" # modify
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_READ_TIMEOUT=300

tasks="mmmu_val,mmmu_pro_standard,mme,realworldqa,mmstar,mmstar_oc,ai2d,chartqa"

for task in ${tasks//,/ }; do
    MODEL="../../pretrained_models/Bard-VL-B4-Mask-4B-Instruct"
    MODEL_NAME=$(basename $(dirname "$MODEL"))
    STEP_NAME=$(basename "$MODEL")
    BLOCK_SIZE=4
    CONFIDENCE=1.0
    STRATEGY="low_confidence_dynamic"
    OUTPUT_DIR="eval_results/$task"
    mkdir -p "$OUTPUT_DIR"

    LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_${STEP_NAME}_${CONFIDENCE}.log"

    accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
        --model bard_vl \
        --model_args="pretrained=$MODEL,max_pixels=4194304,block_size=$BLOCK_SIZE,attn_implementation=sdpa,remasking_strategy=$STRATEGY,confidence_threshold=$CONFIDENCE,interleave_visuals=False" \
        --tasks "$task" \
        --batch_size 1 \
        --output_path "$OUTPUT_DIR" \
        --verbosity=DEBUG 2>&1 | tee "$LOG_FILE"
done
