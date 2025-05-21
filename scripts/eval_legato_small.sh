#!/bin/bash
OUTPUT_DIR=outputs/legato-small-lr-3e-4
PORT=31400

# # DDP EVALUTE BEST CKPT
# OMP_NUM_THREADS=16 WANDB_PROJECT=legato PYTHONPATH=. \
# accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
#     scripts/train.py configs/legato-small-eval.json 

# DDP EVALUTE ALL CKPTS
OMP_NUM_THREADS=16 WANDB_PROJECT=legato PYTHONPATH=. \
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --pretrained_model $OUTPUT_DIR \
    --restore_callback_states_from_checkpoint \
    --dataset_path datasets/PDMX-Synth \
    --mini_val_file datasets/mini_val.json \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    --do_eval --bf16_full_eval \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 2 \
    --predict_with_generate \
    --generation_max_length 2048 --generation_num_beams 3 \
    --report_to wandb \
    --run_name legato-small-eval-allckpts \
    --log_level "info" 