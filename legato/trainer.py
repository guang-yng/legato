import contextlib
import torch
import os
import warnings
import inspect
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from typing import Dict, Union, Optional, List, Any, Tuple
from transformers import Seq2SeqTrainer 
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module

from transformers.trainer import *


class LegatoTrainer(Seq2SeqTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Skip the vision model parameters when saving the model.
        """
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = {
            k : v for k, v in state_dict.items() if not k.startswith('vision_model.')
        }
        super()._save(output_dir, state_dict)

    def _save_optimizer_and_scheduler(self, output_dir):
        if not self.is_deepspeed_enabled:
            super()._save_optimizer_and_scheduler(output_dir)
            return

        # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        # config `stage3_gather_16bit_weights_on_model_save` is True
        accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
            inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
        )
        if accept_exclude_frozen_parameters: # Remove peft checking so that always exclude frozen parameters
            self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
        else:
            self.model_wrapped.save_checkpoint(output_dir)

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_xla_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

    def _load_best_model(self):
        if self.is_deepspeed_enabled:
            logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
            deepspeed_load_checkpoint(
                self.model_wrapped,
                self.state.best_model_checkpoint,
                load_module_strict=False, # The checkpoint is not saved with strict mode
            )
        else:
            super()._load_best_model()

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        inputs = {k: v for k, v in inputs.items() if not k.startswith('gen_')}
        return super().training_step(model, inputs, num_items_in_batch)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = {k.replace('gen_', '') : v for k, v in inputs.items() if k.startswith('gen_')}
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            length = generation_inputs["input_ids"].shape[1]
            # if hasattr(model, "generate"):
            #     generated_tokens = model.generate(**generation_inputs, **gen_kwargs)
            # else:
            original_model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            generated_tokens = original_model.generate(**generation_inputs, **gen_kwargs)
            generated_tokens = generated_tokens[:, length-1:]

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        inputs = {k: v for k, v in inputs.items() if not k.startswith('gen_')}

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels
