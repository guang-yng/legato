from typing import Optional
from transformers import MllamaImageProcessor
from transformers.image_utils import ImageInput

def chunk_image(img):
    chunksize = img.width
    if img.height <= chunksize*4:
        return [img]
    imgs = []
    for i in range(0, img.height, chunksize*3):
        imgs.append(img.crop((0, i, img.width, min(i+chunksize*4, img.height))))
        if img.height <= i+chunksize*4:
            break
    return imgs

class LegatoImageProcessor(MllamaImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        **kwargs
    ):
        chunked_images = [chunk_image(image) for image in images]
        output = super().preprocess(
                images=chunked_images,
                **kwargs
            )
        assert all(len(chunk) == len(tiles) for chunk, tiles in zip(chunked_images, output['num_tiles']))
        return output