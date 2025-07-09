import json
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from legato.metrics.error_rates import compute_error_rates
from argparse import ArgumentParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute ABC Error Rates")
    parser.add_argument("--prediction_file", type=str, help="Path to the XML prediction JSON file")
    parser.add_argument("--ground_truth", type=str, help="Path to the dataset")
    parser.add_argument("--tokenizer_path", type=str, default="guangyangmusic/legato", help="Path to the tokenizer, used to process ground truth in dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")

    args = parser.parse_args()

    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)

    ds = load_from_disk(args.ground_truth)['test']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    label_ids = tokenizer(ds['transcription'], padding=False, truncation=False, add_special_tokens=False).input_ids

    metrics = compute_error_rates(tokenizer, args.num_workers, label_ids, predictions['tokens'])

    print(metrics)

