import os
import torch
from typing import Optional, Union, List, Tuple
from transformers.utils import logging
from transformers import MllamaForConditionalGeneration, MllamaVisionModel
from .configuration_legato import LegatoConfig

logger = logging.get_logger(__name__)

class LegatoModel(MllamaForConditionalGeneration):
    """
    This class extends the MllamaForConditionalGeneration model to include a reference to an encoder model.
    """
    config_class = LegatoConfig
    def __init__(
        self, 
        config : LegatoConfig, 
        load_pretrained_encoder: bool = True
    ):
        super().__init__(config)
        encoder_ref = getattr(config, 'encoder_pretrained_model_name_or_path', None)
        if encoder_ref is not None:
            if load_pretrained_encoder:
                logger.info(f"Loading vision encoder from {encoder_ref}")
                self.vision_model = MllamaVisionModel.from_pretrained(encoder_ref)
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            else:
                self.vision_model = None # Remove vision model and load it later
        elif load_pretrained_encoder:
            raise ValueError(
                "The configuration does not specify 'encoder_pretrained_model_name_or_path'. "
                "Set load_pretrained_encoder to False to skip loading the encoder."
            )

    @classmethod
    def from_pretrained(cls, 
        pretrained_model_name_or_path, 
        *model_args, 
        **kwargs
    ):
        # Load the model configuration and weights
        if "load_pretrained_encoder" in kwargs:
            load_pretrained_encoder = kwargs.pop("load_pretrained_encoder")
            if not load_pretrained_encoder:
                raise ValueError(
                    "'load_pretrained_encoder=False' is contradicted with from_pretrained. "
                )
        model = super().from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            load_pretrained_encoder=False, # Load the vision encoder later
            **kwargs
        )

        # Check if the encoder is already loaded from the checkpoint
        if model.vision_model is None:
            # Retrieve the encoder reference from the configuration
            encoder_ref = getattr(model.config, 'encoder_pretrained_model_name_or_path', None)
            if encoder_ref is None:
                raise ValueError("The configuration does not specify 'encoder_pretrained_model_name_or_path'.")

            # Load the encoder from the specified reference
            logger.info(f"Loading vision encoder from {encoder_ref}")
            if 'config' in kwargs:
                kwargs.pop('config')
            model.vision_model = MllamaVisionModel.from_pretrained(encoder_ref, *model_args, **kwargs)
            model.config.vision_config = model.vision_model.config
            for param in model.vision_model.parameters():
                param.requires_grad = False
        else:
            model.config.encoder_pretrained_model_name_or_path = model.vision_model.config.name_or_path

        return model

    def save_pretrained(
        self, 
        save_directory: str | os.PathLike, 
        state_dict: dict = None, 
        **kwargs
    ) -> None:
        if state_dict is None:
            state_dict = {
                name : 
                    param for name, param in self.state_dict().items() 
                    if not name.startswith("vision_model.")
            }
        super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )
        # Unsqueeze scalar loss for gathering
        # This is needed for distributed training
        if isinstance(outputs, Tuple):
            if outputs[0] is not None:
                outputs = (outputs[0].unsqueeze(0),) + outputs[1:]
        else:
            if outputs.loss is not None:
                outputs.loss = outputs.loss.unsqueeze(0)
        return outputs