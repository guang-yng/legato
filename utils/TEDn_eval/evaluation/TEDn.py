import xml.etree.ElementTree as ET
import zss
import time
import Levenshtein
from typing import List, Tuple
import copy


# Modifications, bugfixes, and notes regarding the source code:
# 1. Moved from lxml to xml (to keep one xml library in the whole repository).
# 2. Removed "decoding" logic, we only need to "encode" (in note flattening).
# 3. The TEDn function now works with both <part> elements or <score-partwise> elements.
# 4. The PitchEncoder used "alteration" instead of "alter" as the element tag.
#       This was probably a typo so we fixed it.
# 5. Added explicit "none" stem as "N" in the coding (does occur in the data).
# 6. The coding currently ignores XML attributes, which may be problematic
#       for measure-rest durations and for no-print invisible objects.
#       But these are niche edgecases so we did not introduce them.
# 7. The original coding excludes tuplets as a to-be-done in the future.
#       We kept tuplets not implemented, since their current representation
#       via <time-modification> seems reasonable given the number possible
#       tuplets in the corpus.
# 8. Extended the note <type> encoding to cover all possible types.
# 9. Missing <pitch> is encoded as "~", which does occur in the data occasionally.
# 10. Duration element was filtered out globally, but this is not ideal,
#       since it's importnat for <forward> and <backup> elements. It is true
#       that for <note> elements in 99.9% cases the duration can be resolved
#       from <type> and <time-modification>. So we modified to code to only ignore
#       the <duration> element inside <note> element.
# 11. Filtered out some additional sound-related and metadata-related elements.
# 12. Added more pitch encoding characters, since the corpus required it.
# 13. Added .strip() for Xml4ZSS_Levenshtein in text comparison (was forgotten).


def TEDn(predicted_element: ET.Element, gold_element: ET.Element) -> "TEDnResult":
    """
    Provide two <part> elements or <score-partwise> elements to compute
    the edit cost via the TEDn edit distance from the paper:

    FURTHER STEPS TOWARDS A STANDARD TESTBED FOR OPTICAL MUSIC RECOGNITION
    Hajič, Novotný, Pecina, Pokorný
    https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/07/289_Paper.pdf

    The code is based on:
    https://github.com/ufal/omreval/blob/master/evaluations/code/omreval/omreval/treedist_eval.py
    """
    assert gold_element.tag in ["part", "score-partwise"], "Unsupported input element type"
    assert gold_element.tag == predicted_element.tag, "Both arguments must be of the same element type"
    
    start_time = time.time()

    # what metric to use (hardcode the TEDn metric)
    metric_class = Xml4ZSS_Levenshtein

    # for the TEDn metric, we need to encode (semi-flatten) notes
    if metric_class is Xml4ZSS_Levenshtein:
        coder = NoteContentCoder()
        gold_element = encode_notes(copy.deepcopy(gold_element), coder)
        predicted_element = encode_notes(copy.deepcopy(predicted_element), coder)

    # Argument order: "How much does it cost to turn prediction into the true tree?"
    edit_cost = zss.distance(
        predicted_element, gold_element,
        get_children=metric_class.get_children,
        update_cost=metric_class.update,
        insert_cost=metric_class.insert,
        remove_cost=metric_class.remove
    )

    # the cost to create the gold tree from one-node tree
    # (used for error normalization)
    # (this computation is fast, O(N) compared to the previous one O(N^2))
    gold_cost = zss.distance(
        ET.Element(predicted_element.tag), gold_element,
        get_children=metric_class.get_children,
        update_cost=metric_class.update,
        insert_cost=metric_class.insert,
        remove_cost=metric_class.remove
    )
    
    end_time = time.time()

    return TEDnResult(
        gold_cost=gold_cost,
        edit_cost=edit_cost,
        evaluation_time_seconds=(end_time - start_time)
    )


class TEDnResult:
    def __init__(self,
        gold_cost: int,
        edit_cost: int,
        evaluation_time_seconds: float
    ):
        self.gold_cost = int(gold_cost)
        self.edit_cost = int(edit_cost)
        self.evaluation_time_seconds: float = evaluation_time_seconds
    
    @property
    def normalized_edit_cost(self) -> float:
        return float(self.edit_cost) / float(self.gold_cost)
    
    def __repr__(self) -> str:
        return (
            f"TEDnResult(" +
            f"gold_cost={self.gold_cost}, " +
            f"edit_cost={self.edit_cost}, " +
            f"evaluation_time_seconds={round(self.evaluation_time_seconds, 2)})"
        )


##################
# Metric classes #
##################

# Based on:
# https://github.com/ufal/omreval/blob/master/evaluations/code/omreval/omreval/zss_metrics.py


class ZSSMetricClass(object):
    """Base class for providing costs/metrics to the ZSS tree edit distance
    module."""
    @staticmethod
    def get_children(e):
        raise NotImplementedError()

    @staticmethod
    def update(e, f):
        raise NotImplementedError()

    @staticmethod
    def insert(e):
        raise NotImplementedError()

    @staticmethod
    def remove(e):
        raise NotImplementedError()


class Xml4ZSS(ZSSMetricClass):
    """A class that defines how edit operation costs should
    be computed from ``xml.etree.ElementTree.Element`` nodes.
    Pass to ``zss.distance()``"""
    @staticmethod
    def get_children(e: ET.Element) -> List[ET.Element]:
        return list(e)

    @staticmethod
    def update(e: ET.Element, f: ET.Element) -> int:
        tag_equal = False
        if e.tag == f.tag:
            tag_equal = True

        text_equal = False
        if e.text is None:
            if f.text is None:
                text_equal = True
        else:
            if f.text is not None:
                if e.text.strip() == f.text.strip():
                    text_equal = True

        if (not tag_equal) or (not text_equal):
            return 1
        else:
            return 0

    @staticmethod
    def insert(e: ET.Element) -> int:
        return 1

    @staticmethod
    def remove(e: ET.Element) -> int:
        return 1


class Xml4ZSS_Filtered(Xml4ZSS):
    """Filters out MusicXML entities irrelevant to evaluation."""
    filtered_out = {
        # https://www.w3.org/2021/06/musicxml40/musicxml-reference/element-tree/

        # ignored everywhere
        "*": {
            "footnote",
            "level",
        },

        # ignored in elements as direct children
        "score-partwise": {
            "work",
            "movement-number",
            "movement-title",
            "identification",
            "defaults",
            "credit",
            "part-list",
        },
        "measure": {
            "print",
            "sound",
            "listening",
        },
        "note": {
            "duration",
            "listen",
            "play",
            "tie", # sound, for notation is <tied>
        }
    }

    @staticmethod
    def get_children(e: ET.Element) -> List[ET.Element]:
        children = []
        for child in e:
            if child.tag in Xml4ZSS_Filtered.filtered_out["*"]:
                continue
            if e.tag in Xml4ZSS_Filtered.filtered_out:
                if child.tag in Xml4ZSS_Filtered.filtered_out[e.tag]:
                    continue
            children.append(child)
        return children


class Xml4ZSS_Levenshtein(Xml4ZSS_Filtered):
    """Tries to bypass some problems inherent to the edit
    distance on nodes, when single notes are rather complex
    subtrees and deleting a rest, which can be accomplished in
    a single keystroke, costs something like six deletions."""

    @staticmethod
    def update(e: ET.Element, f: ET.Element) -> int:
        tag_change_cost = 0
        if e.tag != f.tag:
            tag_change_cost = 1

        text_edit_cost = 0
        if (e.tag == 'note') or (f.tag == 'note'):
            if e.text is None:
                text_edit_cost = len(f.text)
            elif f.text is None:
                text_edit_cost = len(e.text)
            else:
                text_edit_cost: int = Levenshtein.distance(e.text, f.text)
        elif (e.text or "").strip() != (f.text or "").strip():
            text_edit_cost += 1

        return tag_change_cost + text_edit_cost

    @staticmethod
    def insert(e: ET.Element) -> int:
        if e.tag == 'note':
            return 1 + len(e.text)
        else:
            return 1

    @staticmethod
    def remove(e: ET.Element) -> int:
        return 1


###################
# Note flattening #
###################

# Based on:
# https://github.com/ufal/omreval/blob/master/evaluations/code/omreval/omreval/pitch_counter.py


class PitchCoder(object):
    """Encode a sequence of pitches as characters for Levenshtein distance
    computation. Assumes there are no more than 169 distinct pitches
    in a composition, which should hold for all music that doesn't have
    a ridiculous amount of enharmonic changes and/or double accidentals.

    Note that the PitchCoder should be consistent throughout a composition.

    Note: capital R is reserved for rests in note coding, so it's not a part
    of pitch encoding letters.
    """
    code_letters = '0123456789' + \
                   'abcdefghijklmnopqrstuvwxyz' + \
                   'ABCDEFGHIJKLMNOPQ' + 'STUVWXYZ' + \
                   '.,!?:;/\\|-_=+><[]{}()*&^%$#@~`' + \
                   "áčďéěíňóřšťúůýžäëïöüÿ" + \
                   "ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽÄËÏÖÜŸ" + \
                   "бгґдєйжилпфцчшщьюя" + \
                   "БГҐДЄЙЖИЛПФЦЧШЩЬЮЯ"

    def __init__(self):
        self.n_pitches = 0
        self.codes = dict()
        self.inverse_codes = dict()

    def pitch2pitch_index(self, pitch_element: ET.Element) -> Tuple[str, str, str]:
        """
        Converts a pitch element into a (step, alteration, octave) triplet
        that can be used to index the codes dict.
        """
        # https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/pitch/
        pitch_values = {'step': 'C',
                         'alter': '0',
                         'octave': '0'}
        for e in pitch_element:
            pitch_values[e.tag] = e.text
        pitch_index = tuple(pitch_values.values())
        return pitch_index

    def encode(self, pitch_element: ET.Element) -> str:
        """Converts the pitch element into a code letter."""
        pitch_index = self.pitch2pitch_index(pitch_element)

        if pitch_index not in self.codes:
            if self.n_pitches == len(PitchCoder.code_letters):
                raise ValueError('Too many distinct entities to encode, only {0}'
                                 ' codes available.'.format(len(PitchCoder.code_letters)))
            self.codes[pitch_index] = PitchCoder.code_letters[self.n_pitches]
            self.n_pitches += 1

        return self.codes[pitch_index]


class NoteContentCoder(object):
    """Encodes some content of a <note> MusicXML element as a string
    to be compared using Levenshtein distance.

    The following entities are encoded:

    * <pitch> (using the PitchCoder class, or 'R' for rests)
    * <voice> (assumes voices are single-digit)
    * <type> as a single digit: 0 for 128th or shorter, 1 for 64th, 2 for 32,
      3 for 16, 4 for 8th, 5 for 4th, 6 for half, 7 for whole
    * <stem> as U/D/N

    The <notation> and <direction> elements are left as elements.

    One position in the encoding should correspond roughly to one choice
    that needs to be made upon inserting the note.
    """
    note_type_table = {
        "1024th": "0", # added (was not in the original code)
        "512th": "0",
        "256th": "0",
        "128th": "0",
        "64th": "1",
        
        '32nd': "2", # the original code
        '16th': "3",
        'eighth': "4",
        'quarter': "5",
        'half': "6",
        'whole': "7",

        "breve": "8", # added (was not in the original code)
        "long": "9",
        "maxima": "9",
    }

    stem_type_table = {
        'up': 'U',
        'down': 'D',
        'none': 'N',
        None: '-',
    }

    REST_CODE = 'R'
    MISSING_PITCH_CODE = '~'

    ENCODES_TAGS = ['pitch', 'voice', 'type', 'stem']

    def __init__(self):
        self.pitch_coder = PitchCoder()

    def encode(self, note: ET.Element):
        is_rest = note.find("rest") is not None
        pitch_element = note.find("pitch")
        voice_element = note.find("voice")
        type_element = note.find("type")
        stem_element = note.find("stem")

        if is_rest:
            p: str = NoteContentCoder.REST_CODE
        elif pitch_element is not None:
            p: str = self.pitch_coder.encode(pitch_element)
        else:
            p: str = NoteContentCoder.MISSING_PITCH_CODE
        
        if voice_element is not None:
            v: str = voice_element.text
        else:
            v: str = "1"
        
        if type_element is not None:
            t: str = NoteContentCoder.note_type_table[type_element.text]
        else:
            t: str = NoteContentCoder.note_type_table["whole"]
        
        if stem_element is not None:
            s: str = NoteContentCoder.stem_type_table[stem_element.text]
        else:
            s: str = NoteContentCoder.stem_type_table[None]

        code = '{0}{1}{2}{3}'.format(p, v, t, s)
        return code


def encode_notes(root: ET.Element, coder: NoteContentCoder):
    """Uses the NoteContentCoder scheme to change <note> nodes."""
    for note in root.iter('note'):
        code = coder.encode(note)
        note.text = code
        to_remove = [
            e for e in note
            if e.tag in NoteContentCoder.ENCODES_TAGS
        ]
        for e in to_remove:
            note.remove(e)

    return root
