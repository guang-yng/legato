import torch
import argparse
from PIL import Image
from legato.models import LegatoModel
from transformers import AutoProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Legato model. Output to standard output.")
    parser.add_argument("--model_path", type=str, default="guangyangmusic/legato-small", help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image for inference")

    args = parser.parse_args()

    # Load the model and processor
    model = LegatoModel.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load the image and process it
    img = Image.open(args.image_path).convert("RGB")
    inputs = processor(
        images=img,
        truncation=True,
        return_tensors='pt'
    )

    # Move model and inputs to the specified device
    model = model.to(args.device)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Generate the ABC notation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    abc_content = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print(abc_content)
