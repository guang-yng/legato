#!/bin/bash
export WANDB_PROJECT=legato
export OMP_NUM_THREADS=16
export PYTHONPATH=.
OUTPUT_DIR=outputs/legato-small-lr-3e-4
PORT=31400

# DDP EVALUTE BEST CKPT
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py configs/legato-small-eval.json 

# DDP EVALUTE HUGGINGFACE CKPT
CKPT_NAME=guangyangmusic/legato-small # You can change this to any checkpoint you want to evaluate (either local or on huggingface)
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato-small \
    --pretrained_model $CKPT_NAME \
    --restore_callback_states_from_checkpoint \
    --dataset_path datasets/PDMX-Synth \
    --mini_val_file datasets/mini_val.json \
    --output_dir outputs/legato-small-eval \
    --remove_unused_columns False \
    --do_eval --bf16_full_eval \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 2 \
    --predict_with_generate \
    --generation_max_length 2048 --generation_num_beams 3 \
    --report_to wandb \
    --run_name legato-small-eval \
    --log_level "info" 

# DDP EVALUTE SPECIFIC CKPT
CKPT_NAME=$OUTPUT_DIR/checkpoint-20000 # You can change this to any checkpoint you want to evaluate (either local or on huggingface)
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato-small \
    --pretrained_model $CKPT_NAME \
    --restore_callback_states_from_checkpoint \
    --dataset_path datasets/PDMX-Synth \
    --mini_val_file datasets/mini_val.json \
    --output_dir outputs/legato-small-lr-3e-4-eval-20k \
    --remove_unused_columns False \
    --do_eval --bf16_full_eval \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 2 \
    --predict_with_generate \
    --generation_max_length 2048 --generation_num_beams 3 \
    --report_to wandb \
    --run_name legato-small-eval-ckpt-20k \
    --log_level "info" 


# DDP EVALUTE ALL CKPTS
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato-small \
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