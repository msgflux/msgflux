import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, Optional

import defusedxml.ElementTree as defused_ET
from defusedxml import minidom

from msgflux.dotdict import dotdict

_type_converters = {
    "int": int,
    "float": float,
    "str": str,
    "bool": lambda x: x.lower() == "true",
    "dict": lambda x: x,
    "list": lambda x: x,
}


def apply_xml_tags(tag_id: str, content: str, output_id: Optional[str] = None) -> str:
    if output_id is None:
        output_id = tag_id
    return f"<{tag_id}>\n{content}\n</{output_id}>"


def _xml_to_typed_value(element: ET.Element) -> Any:
    """Convert an XML element to a Python value based on type."""
    dtype_attr = element.attrib.get(
        "dtype", "str"
    )  # Assumes "str" ​​if "dtype" is not specified

    if dtype_attr == "dict":
        result = {}  # Create a dictionary with your children
        for child in element:
            result[child.tag] = _xml_to_typed_value(child)
        return result
    elif dtype_attr == "list":
        return [
            _xml_to_typed_value(child) for child in element
        ]  # Create a list with the children
    elif dtype_attr in _type_converters:
        converter = _type_converters[dtype_attr]  # Converts text to the specified type
        return converter(element.text)
    else:
        raise ValueError(f"Unknown dtype: {dtype_attr}")


def xml_to_typed_dict(
    typed_xml: str, *, extract_root_values: Optional[bool] = True
) -> Dict[str, Any]:
    """Converts an XML into a typed dictionary, returning direct values for single tags.

    Args:
        typed_xml:
            Text content xml-based.
        extract_root_values:
            If True, extract the value from root.

    Returns:
        A dict with typed entities extracted

    ::: example
        typed_xml = '''
        <person dtype="dict">
            <name dtype="str">Prevost</name>
            <age dtype="int">69</age>
            <hobbies dtype="list">
                <hobby>Evangelize</hobby>
                <hobby>Defend the sick</hobby>
            </hobbies>
        </person>
        <message>God loves everyone, and evil will not prevail.</message>
        <good_pope dtype="bool">true</good_pope>
        '''
        print(xml_to_typed_dict(typed_xml))
    """
    # Add root in xml string
    typed_xml = apply_xml_tags("root", typed_xml)
    root = defused_ET.fromstring(typed_xml)
    tag_count = defaultdict(int)
    temp_result = defaultdict(list)

    for child in root:  # Counts how many times each tag appears and collects the values
        tag_count[child.tag] += 1
        value = _xml_to_typed_value(child)
        temp_result[child.tag].append(value)

    # Adjust the result: direct value if the tag appears once
    # list if it appears multiple times
    result = {}
    for tag, values in temp_result.items():
        if tag_count[tag] == 1:
            result[tag] = values[0]  # Return single value
        else:
            result[tag] = values  # Return list
    if extract_root_values:
        result = dict(next(iter(result.values())))
    result = dotdict(result)
    return result


def dict_to_typed_xml(data: Dict[str, Any]) -> str:
    """Converts a dictionary into a typed XML string
    without a root tag, formatted readably.
    """

    def build_element(name: str, value: Any) -> ET.Element:
        """Helper function to build an XML element from a key-value pair."""
        if isinstance(value, dict):
            elem = ET.Element(name, dtype="dict")
            for k, v in value.items():
                elem.append(build_element(k, v))
            return elem
        elif isinstance(value, list):
            elem = ET.Element(name, dtype="list")
            for item in value:
                elem.append(build_element("item", item))
            return elem
        else:
            type_str = type(value).__name__
            elem = ET.Element(name, dtype=type_str)
            elem.text = str(value)
            return elem

    # Generate a list of top-level elements
    root_elements = [build_element(key, value) for key, value in data.items()]

    # Format each element individually and collect the results
    pretty_xml_parts = []
    for elem in root_elements:
        # Convert the element to a string
        xml_str = ET.tostring(elem, encoding="unicode")
        # Parse and format it pretty
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")
        # Remove the XML declaration (<?xml ...>) and strip empty lines
        pretty_xml = "\n".join(
            line for line in pretty_xml.splitlines() if "<?xml" not in line
        )
        pretty_xml_parts.append(pretty_xml.strip())

    # Combine all parts with newlines
    return "\n".join(pretty_xml_parts)

def build_compact_field_xml(parent, name, schema, required_fields=None):
    dtype_map = {
        "object": "dict",
        "array": "list",
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool"
    }
    dtype = dtype_map.get(schema.get("type"), schema.get("type", "any"))
    required = str(name in (required_fields or [])).lower()

    field_el = ET.SubElement(parent, "field", {
        "name": name,
        "dtype": dtype,
        "required": required
    })
    if "description" in schema:
        field_el.set("description", schema["description"])

    if schema.get("type") == "object" and "properties" in schema:
        for sub_name, sub_schema in schema["properties"].items():
            build_compact_field_xml(
                field_el, sub_name, sub_schema, schema.get("required", [])
            )

    elif schema.get("type") == "array" and "items" in schema:
        build_compact_field_xml(
            field_el, f"{name}_item", schema["items"],
            schema.get("items", {}).get("required", [])
        )

def json_schema_to_xml_schema(json_schema: Dict[str, Any]) -> str:
    """Converts a JSON-schema object to XML-schema."""
    root_name = json_schema["json_schema"]["name"]
    root_schema = json_schema["json_schema"]["schema"]

    root_el = ET.Element(root_name)
    for prop_name, prop_schema in root_schema.get("properties", {}).items():
        build_compact_field_xml(
            root_el, prop_name, prop_schema, root_schema.get("required", [])
        )

    xml_str = ET.tostring(root_el, encoding="unicode")

    try:
        defused_ET.fromstring(xml_str)
        pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    except Exception: # Fallback
        pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

    return "\n".join(pretty_str.split("\n")[1:])  # remove XML head
