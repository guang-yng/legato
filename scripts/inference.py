import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from PIL import Image
from legato.models import LegatoModel
from transformers import AutoProcessor, GenerationConfig

def remove_special_tokens(arrays, special_tokens):
    outputs = []
    for array in arrays:
        outputs.append([tok for tok in array if tok not in special_tokens])
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Legato model. Output to standard output.")
    parser.add_argument("--model_path", type=str, default="guangyangmusic/legato", help="Path to the trained model")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor (tokenizer)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image or directory containing images for inference")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for generation")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 precision for inference")

    args = parser.parse_args()

    if args.processor_path is None:
        args.processor_path = args.model_path

    # Load the model and processor
    model = LegatoModel.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.processor_path)
    generation_config = GenerationConfig(max_length=2048, num_beams=args.beam_size, repetition_penalty=1.1)

    args.image_path = os.path.abspath(args.image_path)

    # Load the image and process it
    if os.path.isdir(args.image_path):
        if all(img.endswith(('.png', '.jpg', '.jpeg')) for img in os.listdir(args.image_path)):
            imgs = []
            for img_path in os.listdir(args.image_path):
                imgs.append(Image.open(os.path.join(args.image_path, img_path)).convert("RGB"))
        else:
            dataset = load_from_disk(args.image_path)
            imgs = dataset['image']
    else:
        imgs = [Image.open(args.image_path).convert("RGB")]

    model = model.to(device=args.device)
    if args.fp16:
        model = model.half()

    output_tokens = []
    for i in tqdm(range(0, len(imgs), args.batch_size), desc="Predicting..."):
        batch_imgs = imgs[i:min(i + args.batch_size, len(imgs))]
        inputs = processor(
            images=batch_imgs,
            truncation=True,
            return_tensors='pt'
        )

        # Move inputs to the specified device
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        # Generate the ABC notation
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config, use_model_defaults=False)

        output_tokens.extend(outputs.tolist())

    abc_outputs = processor.batch_decode(output_tokens, skip_special_tokens=True)

    special_tokens = processor.tokenizer.all_special_ids 
    preds = remove_special_tokens(output_tokens, special_tokens)

    if not os.path.isdir(args.image_path):
        print(abc_outputs[0])

    if args.output_path is None:
        args.output_path = os.path.dirname(args.image_path) 

    if os.path.isdir(args.output_path):
        if os.path.isdir(args.image_path):
            for idx, (abc, pred) in enumerate(zip(abc_outputs, preds)):
                output_file = os.path.join(
                    args.output_path,
                    f"image_{idx:04}_{args.model_path.replace('/', '_')}_abc.json",
                )
                with open(output_file, "w") as f:
                    json.dump({'abc_transcription': [abc], 'tokens': [pred]}, f)
            print("Inference completed. Outputs saved to directory:", args.output_path)
            exit()

        output_file = os.path.join(
            args.output_path,
            f"{os.path.basename(args.image_path).split('.')[0]}_{args.model_path.replace('/', '_')}_abc.json",
        )
    else:
        output_file = args.output_path
    with open(output_file, "w") as f:
        json.dump({'abc_transcription': abc_outputs, 'tokens': preds}, f)

    print("Inference completed. Output saved to:", output_file)
