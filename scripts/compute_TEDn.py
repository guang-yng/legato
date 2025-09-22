import json
import concurrent.futures
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from datasets import load_from_disk
from argparse import ArgumentParser
from utils.TEDn_eval.evaluation.TEDn_xml_xml import TEDn_xml_xml

def cut_xml(xml, limit=6000):
    root = ET.fromstring(xml)
    num_nodes = sum(1 for _ in root.iter())
    initial_num_nodes = num_nodes

    while num_nodes > limit:
        parts = root.findall('part')
        for part in parts:
            part.remove(part[-1])
        num_nodes = sum(1 for _ in root.iter())

    return ET.tostring(root, encoding='unicode'), initial_num_nodes
        
def compute_score(input_data):
    idx, pred_xml, gold_xml = input_data
    pred_xml, pred_init_num_nodes = cut_xml(pred_xml, 6000)
    gold_xml, gold_init_num_nodes = cut_xml(gold_xml, 6000)
    if pred_init_num_nodes > 6000:
        print(f"Index {idx}: Pred XML initial nodes {pred_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(pred_xml).iter())}")
    if gold_init_num_nodes > 6000:
        print(f"Index {idx}: Gold XML initial nodes {gold_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(gold_xml).iter())}")
    score = TEDn_xml_xml(pred_xml, gold_xml, flavor='lmx')
    return idx, score.edit_cost / score.gold_cost * 100

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", type=str, help="Path to the XML prediction JSON file")
    parser.add_argument("--ground_truth", type=str, help="Path to the dataset or ground truth file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")

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

    # Submit all tasks first
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        tuples = list(zip(range(len(pred_xmls)), pred_xmls, gold_xmls))
        futures = [executor.submit(compute_score, data) for data in tuples]

        TED_scores = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing TED scores..."):
            TED_scores.append(future.result())

    TED_scores.sort(key=lambda x: x[0])
    TED_scores = [x[1] for x in TED_scores]

    scores = {"average_TEDn":  sum(TED_scores) / len(TED_scores), "all_TEDn": TED_scores}
    print(f"Average TED score: {scores['average_TEDn']}")

    output_file = args.prediction_file.replace(".json", "_ted_scores.json")
    print(f"Saving TED scores to {output_file}")
    with open(output_file, "w") as f:
        json.dump(scores, f)
    print("Done.")
