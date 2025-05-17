from typing import Optional, Union, List
from transformers import MllamaProcessor
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import (
    TextInput,
    PreTokenizedInput
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import Unpack, ImagesKwargs, ProcessingKwargs

from transformers.models.mllama.processing_mllama import (
    build_string_from_input,
    make_nested_list_of_images,
    get_cross_attention_token_mask,
    convert_sparse_cross_attention_mask_to_dense
)

from PIL import Image
    

class LegatoImagesKwargs(ImagesKwargs, total=False):
    max_image_tiles: Optional[int]


class LegatoProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: LegatoImagesKwargs

    _defaults = {
        "image_kwargs": {
            "max_image_tiles": 4,
        },
    }


class LegatoProcessor(MllamaProcessor):
    image_processor_class = "LegatoImageProcessor"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        num_tiles: Optional[List[List[int]]] = None,
        return_num_tiles: bool = False,
        **kwargs: Unpack[LegatoProcessorKwargs],
    ) -> BatchFeature:
        """
        """
        if text is None and images is None and num_tiles is None:
            raise ValueError("You must specify either text or images or num_tiles.")

        output_kwargs = self._merge_kwargs(
            LegatoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        images_kwargs = output_kwargs["images_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            
            if any(t.count(self.image_token) > 0 for t in text):
                raise ValueError("Text input should not contain image tokens.")

        if num_tiles is not None:
            assert images is None, "num_tiles should be used without images only."

        if images is not None and text is not None:
            if len(images) != len(text):
                raise ValueError("The number of images and text inputs must match.")
        if num_tiles is not None and text is not None:
            if len(num_tiles) != len(text):
                raise ValueError("The number of num_tiles and text inputs must match.")

        num_inputs = len(images) if images is not None else len(text) if text is not None else len(num_tiles)
        data = {}

        if images is not None:
            image_features = self.image_processor(images, **images_kwargs)
            num_tiles = image_features.pop("num_tiles")
            data.update(image_features)

        n_images_in_images = [0] * num_inputs
        if num_tiles is not None:
            n_images_in_images = [len(num_tile) for num_tile in num_tiles]

        if text is not None:
            text = [
                ((self.image_token * n_images + self.bos_token) if n_images > 0 else "")
                + text_item 
                for n_images, text_item in zip(n_images_in_images, text)
            ]
            _ = text_kwargs.pop("padding_side", None)  # hack until padding-side is an accepted kwarg by tokenizers
            encoding = self.tokenizer(text, **text_kwargs)
        else:
            prompt = [
                ((self.bos_token + self.image_token * n_images)  if n_images > 0 else "")
                + self.bos_token 
                for n_images in n_images_in_images
            ]
            _ = text_kwargs.pop("add_special_tokens", None)
            encoding = self.tokenizer(prompt, add_special_tokens=False, **text_kwargs)

        data.update(encoding)

        # Create cross attention mask
        if images is not None or num_tiles is not None:
            cross_attention_token_mask = [
                get_cross_attention_token_mask(token_ids, self.image_token_id) for token_ids in encoding["input_ids"]
            ]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=self.image_processor.max_image_tiles,
                length=max(len(input_ids) for input_ids in encoding["input_ids"]),
            )
            data["cross_attention_mask"] = cross_attention_mask

        return_tensors = common_kwargs.pop("return_tensors", None)
        batch_feature = BatchFeature(data=data, tensor_type=return_tensors)

        if return_num_tiles:
            batch_feature["num_tiles"] = num_tiles
        return batch_feature