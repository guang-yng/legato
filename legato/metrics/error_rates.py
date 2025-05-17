from typing import Dict
from transformers import PreTrainedTokenizer
#from nltk import edit_distance
from Levenshtein import distance
from tqdm import tqdm
import multiprocessing
import numpy as np

def _edit_distance(args):
    return distance(*args)

def error_rate(preds, refs, desc="error rate", num_workers=8):
    with multiprocessing.Pool(num_workers) as pool:
        dists = list(tqdm(
            pool.imap_unordered(_edit_distance, zip(preds, refs)),
            desc=f"computing {desc}...",
            total=len(preds)
        ))
    return sum(dists) / sum(len(r) for r in refs) * 100

def compute_error_rates(
    tokenizer: PreTrainedTokenizer, 
    num_workers: int, 
    label_ids, 
    preds,
) -> Dict[str, float]:

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    refs_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    SER = error_rate(preds, label_ids, "SER", num_workers)
    CER = error_rate(preds_text, refs_text, "CER", num_workers)
    LER = error_rate([p.split('\n') for p in preds_text], [r.split('\n') for r in refs_text], "LER", num_workers)
    
    metrics = {"LER": LER, "CER": CER, "SER": SER}

    return metrics