#!/bin/bash
export WANDB_PROJECT=legato
export OMP_NUM_THREADS=16
export PYTHONPATH=.
OUTPUT_DIR=outputs/legato-lr-3e-4
PORT=31400

# DDP EVALUTE BEST CKPT
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py configs/legato-eval.json 

# DDP EVALUTE HUGGINGFACE CKPT
CKPT_NAME=guangyangmusic/legato # You can change this to any checkpoint you want to evaluate (either local or on huggingface)
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato \
    --pretrained_model $CKPT_NAME \
    --restore_callback_states_from_checkpoint \
    --dataset_path datasets/PDMX-Synth \
    --mini_val_file datasets/mini_val.json \
    --output_dir outputs/legato-eval \
    --remove_unused_columns False \
    --do_eval --bf16_full_eval \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 2 \
    --predict_with_generate \
    --generation_max_length 2048 --generation_num_beams 3 \
    --report_to wandb \
    --run_name legato-eval \
    --log_level "info" 

# DDP EVALUTE SPECIFIC CKPT
CKPT_NAME=$OUTPUT_DIR/checkpoint-20000 # You can change this to any checkpoint you want to evaluate (either local or on huggingface)
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato \
    --pretrained_model $CKPT_NAME \
    --restore_callback_states_from_checkpoint \
    --dataset_path datasets/PDMX-Synth \
    --mini_val_file datasets/mini_val.json \
    --output_dir outputs/legato-lr-3e-4-eval-20k \
    --remove_unused_columns False \
    --do_eval --bf16_full_eval \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 2 \
    --predict_with_generate \
    --generation_max_length 2048 --generation_num_beams 3 \
    --report_to wandb \
    --run_name legato-eval-ckpt-20k \
    --log_level "info" 


# DDP EVALUTE ALL CKPTS
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py --model_config guangyangmusic/legato \
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
    --run_name legato-eval-allckpts \
    --log_level "info" 