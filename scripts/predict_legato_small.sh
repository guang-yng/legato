#!/bin/bash
export WANDB_PROJECT=legato
export OMP_NUM_THREADS=16
export PYTHONPATH=.
OUTPUT_DIR=outputs/legato-small-lr-3e-4
PORT=31400

# DDP PREDICT BEST CKPT
accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
    scripts/train.py configs/legato-small-predict.json 
