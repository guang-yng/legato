import json
import os
import time
import subprocess
from datasets import load_from_disk
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", type=str, help="Path to the XML prediction JSON file")
    parser.add_argument("--ground_truth", type=str, help="Path to the dataset or ground truth file")

    args = parser.parse_args()
    
    if os.path.basename(args.prediction_file).split('.')[0].split('_')[-1] != 'xml':
        print("[WARNING] The prediction file name is not ended with `xml`. TEDn computation may fail.")

    with open(args.prediction_file, "r") as f:
        pred_xmls = json.load(f)

    if args.ground_truth.endswith('.json'):
        with open(args.ground_truth, "r") as f:
            gold_xmls = json.load(f)
    else:
        ds = load_from_disk(args.ground_truth)
        gold_xmls = [txt for txt in ds['musicxml']]
        if args.ground_truth.endswith('/'):
            args.ground_truth = args.ground_truth[:-1]

    folder_name = args.ground_truth.replace('.json', '') + '_preds'
    os.makedirs(folder_name, exist_ok=True)
    
    gt_folder = os.path.join(folder_name, 'gt')
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder, exist_ok=True)
        for i, gold_xml in enumerate(gold_xmls):
            with open(os.path.join(gt_folder, f'{i}.xml'), 'w') as f:
                f.write(gold_xml)
    
    pred_folder = os.path.join(folder_name, os.path.basename(args.prediction_file).replace('.json', ''))
    assert not os.path.exists(pred_folder), f"Prediction folder {pred_folder} already exists"
    os.makedirs(pred_folder)
    for i, pred_xml in enumerate(pred_xmls):
        with open(os.path.join(pred_folder, f'{i}.xml'), 'w') as f:
            f.write(pred_xml)

    print(f"Saved prediction and ground truth to {folder_name}")
    
    output_folder = os.path.join(pred_folder, f"output")
    
    print("Running musicdiff evaluation...")
    start_time = time.time()
    
    try:
        # Run musicdiff evaluation using subprocess
        os.makedirs(output_folder, exist_ok=True)
        
        cmd = [
            "python", "-m", "musicdiff", "--ml_training_evaluation",
            "--ground_truth_folder", gt_folder,
            "--predicted_folder", pred_folder,
            "--output_folder", output_folder
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Evaluation completed successfully in {duration:.2f} seconds")
        print(f"Results saved to: {output_folder}")
        if result.stdout:
            print(f"Output: {result.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {str(e)}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")