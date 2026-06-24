#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:?Please set MS_SWIFT_ROOT to your patched ms-swift checkout}"
SFT_ENTRY="$MS_SWIFT_ROOT/swift/cli/sft.py"
DATASET_PATH="$REPO_ROOT/data/train/train_infonce_hardneg_10k.json"
OUTPUT_DIR="$SCRIPT_DIR/output_gme"

export PYTHONPATH="$MS_SWIFT_ROOT:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=640000 \
USE_HF=1 \
NPROC_PER_NODE=4 \
python "$SFT_ENTRY" \
    --model Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
    --tuner_type lora \
    --dataset "$DATASET_PATH" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy steps \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR" \
    --lazy_tokenize true \
    --warmup_ratio 0.05 \
    --learning_rate 5e-5 \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --task_type embedding \
    --loss_type infonce_ranking \
    --dataloader_drop_last true
