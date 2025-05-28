#!/bin/bash

PORT=31400
# DEEPSPEED ZERO2 TRAINING
OMP_NUM_THREADS=16 WANDB_PROJECT=legato PYTHONPATH=. \
accelerate launch --config_file configs/zero2.yaml --main_process_port $PORT \
    scripts/train.py configs/legato.json 