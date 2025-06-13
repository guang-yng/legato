import xml.etree.ElementTree as ET


def compare_parts(expected: ET.Element, given: ET.Element):
    measures_e = expected.findall("measure")
    measures_g = given.findall("measure")
    assert len(measures_e) == len(measures_g)
    for i in range(len(measures_e)):
        compare_measures(measures_e[i], measures_g[i])


def compare_measures(expected: ET.Element, given: ET.Element):
    measure_number = expected.get("number")

    if len(expected) != len(given):
        print("Non-matching measure contents!")
        compare_elements(measure_number, expected, given)
        return

    # everything
    for e, g in zip(expected, given):
        compare_elements(measure_number, e, g)


def compare_elements(measure_number: str, expected: ET.Element, given: ET.Element):
    e = ET.canonicalize(
        ET.tostring(expected),
        strip_text=True
    )
    g = ET.canonicalize(
        ET.tostring(given),
        strip_text=True
    )
    if e != g:
        print()
        print("MEASURE: ", measure_number)
        print("EXPECTED:", e)
        print("GIVEN:   ", g)