#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:?Please set MS_SWIFT_ROOT to your patched ms-swift checkout}"
SFT_ENTRY="$MS_SWIFT_ROOT/swift/cli/sft.py"
DATASET_PATH="$REPO_ROOT/data/train/train_infonce_hardneg_10k.json"
OUTPUT_DIR="$SCRIPT_DIR/output_qwen"

export PYTHONPATH="$MS_SWIFT_ROOT:${PYTHONPATH:-}"

echo "infonce_rank_only_hardneg"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=4 \
MASTER_PORT=29503 \
python "$SFT_ENTRY" \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --model_type qwen3_vl_emb \
    --task_type embedding \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 5e-5 \
    --target_modules all-linear \
    --dataset "$DATASET_PATH" \
    --attn_impl flash_attn \
    --padding_free true \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --eval_strategy steps \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --num_train_epochs 1 \
    --max_length 8192 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero2 \
    --loss_type infonce_ranking
