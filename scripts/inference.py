import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from legato.models import LegatoModel
from transformers import AutoProcessor

def remove_special_tokens(arrays, tokenizer):
    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]
    outputs = []
    for array in arrays:
        outputs.append([tok for tok in array if tok not in special_tokens])
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Legato model. Output to standard output.")
    parser.add_argument("--model_path", type=str, default="guangyangmusic/legato", help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image or directory containing images for inference")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images")

    args = parser.parse_args()

    # Load the model and processor
    model = LegatoModel.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load the image and process it
    if os.path.isdir(args.image_path):
        imgs = []
        for img_path in os.listdir(args.image_path):
            imgs.append(Image.open(os.path.join(args.image_path, img_path)).convert("RGB"))
    else:
        imgs = [Image.open(args.image_path).convert("RGB")]

    model = model.to(args.device)

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
            outputs = model.generate(**inputs)

        output_tokens.extend(outputs.tolist())

    abc_outputs = processor.batch_decode(output_tokens, skip_special_tokens=True)
    preds = remove_special_tokens(output_tokens, processor.tokenizer)

    if not os.path.isdir(args.image_path):
        print(abc_outputs[0])

    if args.output_path is None:
        args.output_path = os.path.dirname(args.image_path) 
    with open(os.path.join(args.output_path, f"{os.path.basename(args.image_path)}.json"), "w") as f:
        json.dump({'abc_transcription': abc_outputs, 'tokens': preds}, f)

    print("Inference completed. Output saved to:", os.path.join(args.output_path, f"{os.path.basename(args.image_path)}_{args.model_path.replace('/', '_')}.json"))
