# LEGATO: Large-Scale End-to-End Generalizable Appaorach to Typeset OMR


Official codebase for paper "LEGATO: Large-Scale End-to-End Generalizable Appaorach to Typeset OMR".

## Environment

Create a new conda environment and activate:
```sh
conda create -n legato python=3.12
conda activate legato
```

Install dependencies:
```sh
pip install -r requirements.txt
```

Note: `requirements.txt` is only tested with CUDA 12.4.

## Model Training and Validataion

We use `legato-small` as example for the following section. However, training small model is not efficient due to the large frozen vision encoder. The small model only serves as a comparison with previous methods.

Please refer to [Model Checkpoints](#model-checkpoints) section for all pre-trained legato checkpoints.

### Training

To train a legato model, simply use the scripts under `scripts/` folder.
For example, train a legato-small model with the following scripts:
```sh
bash scripts/train_legato_small.sh
```

To customize the training hyperparameters, either modify the config file `configs/legato-small.json` or directly set all arguments in the command line like the following:
```sh
PYTHONPATH=. accelerate launch --config_file configs/zero2.yaml \
    scripts/train.py \
    --model_config guangyangmusic/legato-small \
    --dataset_path datasets/PDMX-Synth \
    --output_dir outputs/legato-small \
    --remove_unused_columns False \
    --do_train --do_eval \
    --metric_for_best_model eval_SER --greater_is_better False \
    --save_steps 5000 --eval_steps 5000 \
    --num_train_epochs 10 --learning_rate 3e-4 --per_device_train_batch_size 2 
```
Please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) for all availabel arguments.

We use DeepSpeed ZeRO-2 as our default parallel training setting. 
You can also customize by selecting a YAML config file under `configs/` or use your own accelerate config setting.

### Validation

We provide a script to simply validate the best checkpoint:
```sh
# DDP EVALUTE BEST CKPT
PYTHONPATH=. \
accelerate launch --config_file configs/inference.yaml \
    scripts/train.py configs/legato-small-eval.json 
```

Also, you can validate all checkpoints, a specific local checkpoint or a checkpoint on HuggingFace. These scripts are all included in `scripts/eval_legato_small.sh`.


### Model Checkpoints

We release our checkpoints on HuggingFace:

| Model Name    | Link |
| -------- | ------- |
| legato-small |  [guangyangmusic/legato-small](https://huggingface.co/guangyangmusic/legato-small)  |
