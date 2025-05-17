from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig, MllamaTextConfig

class LegatoConfig(MllamaConfig):
    r"""
    This is the configuration class to store the configuration of a Legato model.
    """

    model_type = "legato"
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        encoder_pretrained_model_name_or_path : str = "meta-llama/Llama-3.2-11B-Vision",
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MllamaVisionConfig()

        if text_config is None:
            text_config = MllamaTextConfig()

        self.encoder_pretrained_model_name_or_path = encoder_pretrained_model_name_or_path

        super().__init__(vision_config, text_config, **kwargs)


__all__ = ["LegatoConfig"]
