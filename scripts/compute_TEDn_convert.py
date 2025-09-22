import json
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tedn_file", type=str, help="Path to the TEDn score JSON file")
    parser.add_argument("--fail_mask", type=str, help="Path to SMT++ fail mask JSON file")
    args = parser.parse_args()

    with open(args.tedn_file, "r") as f:
        tedn_scores = json.load(f)

    with open(args.fail_mask, "r") as f:
        fail_mask = json.load(f)
    
    # Revert the boolean mask (invert all values)
    fail_mask = [not x for x in fail_mask]

    tedn_convert = np.dot(tedn_scores['all_TEDn'], fail_mask)
    tedn_convert = tedn_convert / np.sum(fail_mask)

    print(f"TEDn: {tedn_scores['average_TEDn']}")
    print(f"TEDn convert: {tedn_convert}")
