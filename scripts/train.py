import sys
import torch
import json
import os
import logging
import numpy as np
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainingArguments,
    TrainerState,
    HfArgumentParser, 
    AutoConfig,
    AutoProcessor,
    AutoModel,
    set_seed,
)
from transformers.trainer import TRAINER_STATE_NAME
from accelerate.logging import get_logger
from legato.config import DataArguments, ModelArguments
from legato.models import LegatoConfig, LegatoModel 
from legato.trainer import LegatoTrainer
from legato.metrics import compute_error_rates


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, data_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    logging.basicConfig(level=training_args.log_level.upper(), format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    logger = get_logger(__name__)

    torch.set_float32_matmul_precision('high')
    if training_args.torch_compile:
        # Increase the cache size limit to avoid recompilation
        # Large RAM usage may occur if the cache size is too large
        torch._dynamo.config.cache_size_limit = 256

    #### Load dataset
    dataset = load_from_disk(data_args.dataset_path)
    for split, mini_file in [("val", data_args.mini_val_file), ("test", data_args.mini_test_file)]:
        if mini_file:
            logger.info(f"Using mini {split} set: {mini_file}")
            with open(mini_file, "r") as f:
                filenames = json.load(f)
            dataset[split] = dataset[split].select(
                [dataset[split]['filename'].index(filename) for filename in filenames]
            )

    if data_args.dummy_data:
        logger.info("Using dummy data (32 items) for debugging only...")
        dataset['train'] = dataset['train'].select(range(32))
        dataset['val'] = dataset['val'].select(range(32))
        dataset['test'] = dataset['test'].select(range(32))

    #### Load model and tokenizer
    set_seed(training_args.seed)

    if model_args.pretrained_model:
        model = AutoModel.from_pretrained(model_args.pretrained_model)
    else:
        config = AutoConfig.from_pretrained(model_args.model_config)
        model = LegatoModel(config)

    processor = AutoProcessor.from_pretrained(model_args.model_config)
    tokenizer = processor.tokenizer

    def get_metric_target(examples):
        return {
            'label_ids': processor(text=examples['transcription'], add_special_tokens=False, verbose=False)['input_ids'],
        }

    if not training_args.do_predict:
        metric_targets = dataset['val'].map(
            get_metric_target, 
            remove_columns=dataset['val'].column_names,
            num_proc=training_args.dataloader_num_workers, 
            batched=True
        ).to_dict()
    else:
        metric_targets = dataset['test'].map(
            get_metric_target, 
            remove_columns=dataset['test'].column_names,
            num_proc=training_args.dataloader_num_workers, 
            batched=True
        ).to_dict()

    # We don't predict image tokens or padding tokens
    tokens_to_mask = torch.tensor([processor.image_token_id, tokenizer.pad_token_id])

    def collate_fn(examples):
        outputs = processor(
            images=[example['image'] for example in examples],
            text=[example['transcription'] for example in examples],
            return_num_tiles=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt',
        ) # pad to max length to reduce torch compilation overhead
        gen_outputs = processor(
            num_tiles=outputs.pop('num_tiles'), # Reuse num_tiles to save computation
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        outputs.update({
            f'gen_{k}': outputs[k] if k not in gen_outputs else gen_outputs[k]
            for k in outputs
        })
        outputs['labels'] = outputs['input_ids'].clone().masked_fill(
            torch.isin(outputs['input_ids'], tokens_to_mask), -100
        ) # We don't predict image tokens or padding tokens
        return outputs

    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]
    def remove_special_tokens(array):
        masks = np.isin(array, special_tokens, invert=True)
        return [a[mask] for a, mask in zip(array, masks)]

    def metric_fn(p):
        preds = remove_special_tokens(p.predictions)
        results = [compute_error_rates(
            tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds
        )] if training_args.process_index == 0 else [None]
        dist.broadcast_object_list(results, src=0)
        return results[0]

    trainer = LegatoTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=metric_fn
    )

    def _unwrap_and_save_model(trainer, output_dir):
        if trainer.is_world_process_zero():
            logger.info("Unwrapping the model...")
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            logger.info("Model unwrapped.")
            unwrapped_model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"Model and Processor saved. to {output_dir}")

    #### Train
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    #### Evaluate
    if training_args.do_eval:
        if not training_args.do_train and not model_args.pretrained_model: # if no training is done and no pretrained model is provided, evaluate all checkpoints
            ckpts = [ckpt for ckpt in os.listdir(training_args.output_dir) if ckpt.startswith("checkpoint")]
            assert len(ckpts) > 0, f"No checkpoints found in {training_args.output_dir}"
            best_ckpt, best_result = None, None
            for ckpt in sorted(ckpts):
                logger.info(f"Evaluating checkpoint {ckpt}...")
                trainer._load_from_checkpoint(os.path.join(training_args.output_dir, ckpt))
                trainer.state = TrainerState.load_from_json(os.path.join(training_args.output_dir, ckpt, TRAINER_STATE_NAME))
                trainer.state.init_training_references(trainer, trainer.state.max_steps, trainer.state.num_train_epochs, None)
                trainer._load_callback_state()
                result = trainer.evaluate()
                if best_result is None or result['eval_SER'] < best_result['eval_SER']:
                    best_result, best_ckpt = result, ckpt
                trainer.log_metrics("eval", result)

            logger.info(f"Best checkpoint: {best_ckpt}")
            trainer._load_from_checkpoint(os.path.join(training_args.output_dir, best_ckpt))

        else:
            best_result = trainer.evaluate()

        _unwrap_and_save_model(trainer, training_args.output_dir)
        final_val_results = {k.replace("eval_", "eval_best_"): v for k, v in best_result.items() if k.startswith("eval_")}
        trainer.log_metrics("best eval", final_val_results)
        trainer.log(final_val_results)

    #### Predict
    if training_args.do_predict:
        pass
    

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup distributed process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

