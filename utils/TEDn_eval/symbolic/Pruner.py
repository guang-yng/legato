# Adapted from https://github.com/ufal/olimpic-icdar24 under the MIT License
# Copyright (c) 2024 Jiří Mayer
# SPDX-License-Identifier: MIT
import xml.etree.ElementTree as ET


class Pruner:
    """Prunes unimportant attributes and elements from MusicXML so that
    XML equality comparison can be utilized during when comparing with
    delinearized XML; It ignores elements that TEDn ignores and attributes
    that are relevant to layout, not content."""

    def __init__(
        self,
        prune_durations=False,
        prune_measure_attributes=False,
        prune_prints=True,
        prune_directions=True,
        prune_barlines=True,
        prune_harmony=True,
        prune_slur_numbering=True,
    ):
        self.prune_durations = prune_durations
        self.prune_measure_attributes = prune_measure_attributes
        self.prune_slur_numbering = prune_slur_numbering

        self.measure_prune_tags = {"sound", "listening", "figured-bass", "bookmark", "grouping"}
        if prune_prints:
            self.measure_prune_tags.add("print")
        if prune_directions:
            self.measure_prune_tags.add("direction")
        if prune_barlines:
            self.measure_prune_tags.add("barline")
        if prune_harmony:
            self.measure_prune_tags.add("harmony")
        
        self.note_prune_tags = {"cue", "lyric", "instrument", "play", "listen", "notehead-text", "notehead"}
        if prune_durations:
            self.note_prune_tags.add("duration")

    def process_part(self, part: ET.Element):
        assert part.tag == "part"
        for measure in part:
            self.process_measure(measure)
    
    def process_measure(self, measure: ET.Element):
        assert measure.tag == "measure"

        if self.prune_measure_attributes:
            measure.attrib.pop("number", None)
            measure.attrib.pop("implicit", None)
            measure.attrib.pop("width", None)
        
        prune_children(measure, self.measure_prune_tags)

        for element in measure:
            if element.tag == "note":
                self.process_note(element)
            elif element.tag == "backup":
                self.process_backup(element)
            elif element.tag == "forward":
                self.process_forward(element)
            elif element.tag == "attributes":
                self.process_attributes(element)
    
    def process_forward(self, forward: ET.Element):
        if self.prune_durations:
            prune_children(forward, {"duration"})
    
    def process_backup(self, backup: ET.Element):
        if self.prune_durations:
            prune_children(backup, {"duration"})

    def process_attributes(self, attributes: ET.Element):
        prune_children(attributes, {"instruments", "staff-details",
        "measure-style", "directive", "part-symbol"})

        if self.prune_durations:
            divisions_element = attributes.find("divisions")
            if divisions_element is not None:
                attributes.remove(divisions_element)
        
        time_element = attributes.find("time")
        if time_element is not None:
            time_element.attrib.pop("symbol", None)
        
        for clef_element in attributes.findall("clef"):
            clef_element.attrib.pop("after-barline", None)

    def process_note(self, note: ET.Element):
        assert note.tag == "note"
        
        note.attrib.pop("default-x", None)
        note.attrib.pop("default-y", None)
        note.attrib.pop("dynamics", None)

        prune_children(note, self.note_prune_tags)

        rest_element = note.find("rest")
        if rest_element is not None:
            if len(rest_element) > 0:
                rest_element[:] = [] # remove children

        type_element = note.find("type")
        if type_element is not None:
            type_element.attrib.clear()

        accidental_element = note.find("accidental")
        if accidental_element is not None:
            accidental_element.attrib.clear()

        time_modification_element = note.find("time-modification")
        if time_modification_element is not None:
            allow_children(time_modification_element, {"actual-notes", "normal-notes"})

        notations_element = note.find("notations")
        if notations_element is not None:
            self.process_notations(notations_element)
            if len(notations_element) == 0:
                note.remove(notations_element)
    
    def process_notations(self, notations: ET.Element):
        allow_children(notations, {
            "tied", "slur", "tuplet", "ornaments", "articulations",
            "fermata", "arpeggiate"
        })
        for element in list(notations):
            if element.tag == "slur":
                element.attrib.pop("placement", None)
                element.attrib.pop("line-type", None)
                if self.prune_slur_numbering:
                    element.attrib.pop("number", None)
            elif element.tag == "tuplet":
                element.attrib.pop("bracket", None)
                element.attrib.pop("show-number", None)
                element[:] = [] # clear children
            elif element.tag == "fermata":
                element.attrib.clear()
            elif element.tag == "arpeggiate":
                element.attrib.clear()
            elif element.tag == "articulations":
                self.process_articulations(element)
                if len(element) == 0:
                    notations.remove(element)
            elif element.tag == "ornaments":
                self.process_ornaments(element)
                if len(element) == 0:
                    notations.remove(element)
    
    def process_articulations(self, articulations: ET.Element):
        allow_children(articulations, {
            "staccato", "accent", "strong-accent", "tenuto"
        })
        for element in articulations:
            element.clear()
    
    def process_ornaments(self, ornaments: ET.Element):
        allow_children(ornaments, {
            "tremolo", "trill-mark"
        })
        for element in list(ornaments):
            if element.tag == "trill-mark":
                element.clear()


def prune_children(element: ET.Element, tags: set):
    children_to_remove = [
        ch for ch in element
        if ch.tag in tags
    ]
    for ch in children_to_remove:
        element.remove(ch)


def allow_children(element: ET.Element, tags: set):
    children_to_remove = [
        ch for ch in element
        if ch.tag not in tags
    ]
    for ch in children_to_remove:
        element.remove(ch)
