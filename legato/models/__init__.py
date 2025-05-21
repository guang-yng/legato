from transformers import AutoImageProcessor, AutoProcessor, AutoConfig, AutoModel
from .processing_legato import LegatoProcessor
from .image_processing_legato import LegatoImageProcessor
from .modeling_legato import LegatoModel
from .configuration_legato import LegatoConfig

AutoProcessor.register("LegatoProcessor", LegatoProcessor)
AutoImageProcessor.register("LegatoImageProcessor", slow_image_processor_class=LegatoImageProcessor)
AutoConfig.register("legato", LegatoConfig)
AutoModel._model_mapping.register(LegatoConfig, LegatoModel)