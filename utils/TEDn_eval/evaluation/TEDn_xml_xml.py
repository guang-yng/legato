# Adapted from https://github.com/ufal/olimpic-icdar24 under the MIT License
# Copyright (c) 2024 Jiří Mayer
# SPDX-License-Identifier: MIT
from .TEDn import TEDn, TEDnResult
from ..symbolic.Pruner import Pruner
from ..symbolic.actual_durations_to_fractional import actual_durations_to_fractional
from ..symbolic.debug_compare import compare_parts
import xml.etree.ElementTree as ET
from typing import TextIO, Optional, Literal


def TEDn_xml_xml(
    predicted_xml: str,
    gold_musicxml: str,
    flavor: Literal["full", "lmx"],
    debug=False,
    canonicalize=True,
    errout: Optional[TextIO] = None
) -> TEDnResult:
    """
    Provides access to the TEDn metric with a nice string-based interface.

    Apart from having nice API, it also performs useful processing:
    - Delinearizes LMX back to MusicXML
    - Allows for "LMX-only" subset of MusicXML to be evaluated, see flavor="lmx"
    - Converts <duration> elements to fractional representation
        (number of quarter notes in fractions.Fraction stringified),
        which is necessary for proper <forward>, <backup> evaluation

    :param str predicted_lmx: The LMX string that was predicted by your img2seq model
    :param str gold_musicxml: The gold XML data loaded from the .musicxml annotation file
    :param str flavor: Use 'full' to do a regular TEDn computation, or 'lmx'
        to prune the gold target down to the set of musical concepts covered by LMX,
        for example removing <direction>, <harmony>, or <barline> elements.
    :param bool debug: Prints when a strict XML string comparison fails to get
        some idea about the places the model makes mistakes.
    :param bool canonicalize_gold: Run XML canonicalization on the gold string.
        Not necessary, but recommended. It primarily strips away whitespace
        (but TEDn ignores whitespace anyway).
    :param Optional[TextIO] errout: Delinearizer soft and hard errors are sent here.
    """

    assert flavor in {"full", "lmx"}

    # preprocess gold XML to remove whitespace
    # (not necessary after the TEDn .strip() bugfix, but present just in case...)
    if canonicalize:
        gold_musicxml = ET.canonicalize(
            gold_musicxml,
            strip_text=True
        )
        predicted_xml = ET.canonicalize(
            predicted_xml,
            strip_text=True
        )

    # prepare gold data
    gold_score = ET.fromstring(gold_musicxml)
    assert gold_score.tag == "score-partwise"
    gold_parts = gold_score.findall("part")
    to_remove = []
    for gold_child in gold_score:
        if gold_child.tag != "part":
            to_remove.append(gold_child)
    for child in to_remove:
        gold_score.remove(child)

    pred_score = ET.fromstring(predicted_xml)
    assert pred_score.tag == "score-partwise"
    pred_parts = pred_score.findall("part")[:2]
    cnt = 0
    to_remove = []
    for pred_child in pred_score:
        #print(pred_child.tag)
        if pred_child.tag != "part" or cnt >= 2:
            #print('remove', pred_child.tag)
            to_remove.append(pred_child)
        else:
            #print('keep', pred_child.tag)
            cnt += 1
    for child in to_remove:
        pred_score.remove(child)
    #print(len(pred_score))

    if len(gold_parts) < len(pred_parts):
        pred_parts = pred_parts[:len(gold_parts)]
        #breakpoint()
    for gold_part in gold_parts:
        actual_durations_to_fractional(gold_part) # evaluate in fractional durations
    for pred_part in pred_parts:
        actual_durations_to_fractional(pred_part)

    # prune down to the elements that we actually predict
    # (otherwise TEDn penalizes missing <direction> and various ornaments)
    if flavor == "lmx":
        pruner = Pruner(
            
            # these are acutally also ignored by TEDn
            prune_durations=False, # MUST BE FALSE! Is used in backups and forwards
            prune_measure_attributes=False,
            prune_prints=True,
            prune_slur_numbering=True,

            # these measure elements are not encoded in LMX, prune them
            prune_directions=True,
            prune_barlines=True,
            prune_harmony=True,
            
        )
        for predicted_part in pred_parts:
            pruner.process_part(predicted_part)
        for gold_part in gold_parts:
            pruner.process_part(gold_part)
            
    if debug:
        for gold_part, predicted_part in zip(gold_parts, pred_parts):
            compare_parts(expected=gold_part, given=predicted_part)

    # print(f"prediction nodes: {sum(1 for _ in pred_score.iter())}, gold nodes: {sum(1 for _ in gold_score.iter())}")
    if len(pred_parts) == len(gold_parts):
        # print("Do partwise comparison.")
        TEDn_scores = [TEDn(pred_part, gold_part) for pred_part, gold_part in zip(pred_parts, gold_parts)]
        return TEDnResult(
            sum(score.gold_cost for score in TEDn_scores),
            sum(score.edit_cost for score in TEDn_scores),
            sum(score.evaluation_time_seconds for score in TEDn_scores)
        )

    return TEDn(pred_score, gold_score)
    # return TEDnResult(1, 1, 1) # debugging
