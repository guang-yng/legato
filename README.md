# LEGATO: Large-Scale End-to-End Generalizable Approach to Typeset OMR

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-%23ee4c2c?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Official repository for the paper:  
**["LEGATO: Large-Scale End-to-End Generalizable Approach to Typeset OMR"](https://arxiv.org/abs/2506.19065)**

## 🛠️ Setup Instructions

### 1. Create Environment

```bash
conda create -n legato python=3.12
conda activate legato
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Tested with CUDA 12.4.

## 📦 Pretrained Checkpoints

We release two LEGATO checkpoints on HuggingFace:

|Model|Link|
|-|-|
|legato-small|[guangyangmusic/legato-small](https://huggingface.co/guangyangmusic/legato-small)|
|legato|[guangyangmusic/legato](https://huggingface.co/guangyangmusic/legato)|

> 🔹 Recommended: Use `legato` (full model). The small variant is mainly for baseline comparisons and is less efficient.

## 🔍 Inference

### Run Inference on a Single Image

```bash
PYTHONPATH=. python scripts/inference.py \
    --model_path guangyangmusic/legato \
    --image_path path/to/image.png
```

### Batch Inference on a Folder

```bash
PYTHONPATH=. python scripts/inference.py \
    --model_path guangyangmusic/legato \
    --image_path path/to/image_folder/
```

> 🖼️ Image folder should contain only .jpg, .jpeg, or .png files.

### Inference from datasets.Dataset

Set `image_path` to the folder containing the dataset. Ensure the dataset has a column named image with score images.

### Half Precision Inference

Use `--fp16` flag to enable half-precision inference. This reduces memory usage but may impact performance.

## 🎯 Training & Validation

We use `legato` in examples below. Refer to the [Checkpoints](#-pretrained-checkpoints) section for pretrained models.

> The commands in this section have only been tested on a single node with multiple GPUs.

### 🔥 Training

Use the provided script:

```bash
bash scripts/train_legato.sh
```

Or customize:

```bash
PYTHONPATH=. accelerate launch --config_file configs/zero2.yaml \
    scripts/train.py \
    --model_config guangyangmusic/legato \
    --dataset_path datasets/PDMX-Synth \
    --output_dir outputs/legato \
    --remove_unused_columns False \
    --do_train --do_eval \
    --metric_for_best_model eval_SER --greater_is_better False \
    --save_steps 5000 --eval_steps 5000 \
    --num_train_epochs 10 \
    --learning_rate 3e-4 \
    --per_device_train_batch_size 2
```

Refer to [TrainingArguments docs](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) for more options.

LEGATO uses **DeepSpeed ZeRO-2** by default. You can modify or provide your own Accelerate config (`configs/*.yaml`).

### ✅ Validation

Validate the best checkpoint:

```bash
PYTHONPATH=. accelerate launch --config_file configs/inference.yaml \
    scripts/train.py configs/legato-eval.json
```

You can also:
- Evaluate specific local checkpoints
- Evaluate from HuggingFace
- Evaluate all checkpoints

See `scripts/eval_legato.sh` for examples.

### 🔮 Prediction

To predict using multiple GPUs:

```
accelerate launch --config_file configs/inference.yaml \
    scripts/train.py configs/legato-predict.json
```

> 🔄 Output saved as test_predictions.json in the output directory.

> 📉 If transcription is present in the dataset, error metrics will be computed automatically.

## 🔁 MusicXML Conversion & Evaluation

### 📏 ABC Error Rate Evaluation

To evaluate the ABC transcription accuracy of your model predictions, use the provided script:
```bash
PYTHONPATH=. python scripts/compute_ER.py \
    --prediction_file path/to/test_predictions.json \
    --ground_truth datasets/PDMX-Synth
```

### 🎼 ABC to MusicXML Conversion

Convert ABC predictions to MusicXML using:

```bash
DISPLAY=:0 python utils/convert.py --input_file xxx_abc.json
```

Requirements:
- MuseScore executable at `software/mscore`
- GUI-enabled environment (`DISPLAY=:0`)
- Depends on `utils/abc2xml.py`

### 🌲 TEDn Evaluation

Compute Tree Edit Distance with `<note>` flattening (TEDn):

```bash
PYTHONPATH=. python scripts/compute_TEDn.py \
    --prediction_file xxx_xml.json \
    --ground_truth path/to/dataset \
    --num_workers 4
```

Dataset must contain a `musicxml` column.

> Parts of `utils/TEDn_eval` are adapted from [OLiMPiC](https://github.com/ufal/olimpic-icdar24), licensed under the MIT License.

## 📄 Citation

```
@misc{yang2025legatolargescaleendtoendgeneralizable,
      title={LEGATO: Large-scale End-to-end Generalizable Approach to Typeset OMR}, 
      author={Guang Yang and Victoria Ebert and Nazif Tamer and Luiza Pozzobon and Noah A. Smith},
      year={2025},
      eprint={2506.19065},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.19065}, 
}
```