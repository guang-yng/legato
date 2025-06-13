import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Optional


def actual_durations_to_fractional(part: ET.Element):
    assert part.tag == "part"
    
    if len(part) == 0:
        return
    
    current_divisions: Optional[int] = None
    
    def _visit_duration(duration_element: ET.Element):
        nonlocal current_divisions
        if duration_element is None:
            return
        
        assert current_divisions is not None

        d = Fraction(duration_element.text)
        d = d / current_divisions
        duration_element.text = str(d)
    
    def _visit_notelike(notelike_element: ET.Element):
        duration_element = notelike_element.find("duration")
        _visit_duration(duration_element)

    def _visit_attributes(attributes_element: ET.Element):
        nonlocal current_divisions
        divisions_element = attributes_element.find("divisions")
        if divisions_element is None:
            return
        current_divisions = int(divisions_element.text)
        attributes_element.remove(divisions_element)

    for measure in part:
        for element in measure:
            if element.tag in {"note", "forward", "backup"}:
                _visit_notelike(element)
            elif element.tag == "attributes":
                _visit_attributes(element)
