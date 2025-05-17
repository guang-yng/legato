import fire
import torch
import json
import numpy as np
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser, 
    AutoConfig,
    AutoProcessor,
    AutoModel,
    set_seed,
)
import logging
from accelerate.logging import get_logger
from legato.config import DataArguments, ModelArguments
from legato.models import LegatoModel 
from legato.trainer import OMRTrainer
from legato.metrics import compute_error_rates

#logger = get_logger(__name__)

def main(config_path: str):
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_json_file(json_file=config_path)
    set_seed(training_args.seed)

    logging.basicConfig(level=training_args.log_level.upper(), format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    logger = get_logger(__name__)

    torch.set_float32_matmul_precision('high')

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
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model)
    else:
        config = AutoConfig.from_pretrained(model_args.model_config)
        processor = AutoProcessor.from_pretrained(model_args.model_config)
        model = LegatoModel(config)

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
        outputs.update({f'gen_{k}': v for k, v in gen_outputs.items()})
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

    trainer = OMRTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=metric_fn
    )

    #### Train
    if training_args.do_train:
        trainer.train()
        result = trainer.evaluate()
        if trainer.is_world_process_zero():
            logger.log("Unwrapping the model...")
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            logger.log("Model unwrapped.")
            unwrapped_model.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
            logger.log(f"Model and Processor saved. to {training_args.output_dir}")
            final_val_results = {k.replace("eval_", "eval_best_"): v for k, v in result.items() if k.startswith("eval_")}
            trainer.log_metrics("best eval", final_val_results)
            trainer.log(final_val_results)


    #### Evaluate
    if training_args.do_eval and not training_args.do_train and not training_args.do_predict:
        pass

    #### Predict
    if training_args.do_predict:
        pass
    

if __name__ == "__main__":
    try:
        fire.Fire(main)
    finally:
        # Cleanup distributed process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

