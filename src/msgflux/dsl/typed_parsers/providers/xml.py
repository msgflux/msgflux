import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, Optional

try:
    import defusedxml.ElementTree as defused_ET
    from defusedxml import minidom
except ImportError:
    defused_ET = None
    minidom = None

from msgflux.dsl.typed_parsers.base import BaseTypedParser
from msgflux.dsl.typed_parsers.registry import register_typed_parser
from msgflux.dotdict import dotdict
from msgflux.utils.xml import apply_xml_tags

_type_converters = {
    "int": int,
    "float": float,
    "str": str,
    "bool": lambda x: x.lower() == "true",
    "dict": lambda x: x,
    "list": lambda x: x,
}


TYPED_XML_TEMPLATE = """
{% if instructions %}{{ instructions }}{% endif %}

You SHOULD write your response in a structured manner using XML tags.
DO NOT write XML headers or add extra messages beyond the XML response.
You should then generate an XML specifying the dtype.
The available data types are: str (default if not specified), int, float, bool, dict and list.

Example of how you can write your response in XML:

<user_profile dtype="dict">
    <id dtype="int">1024</id>
    <username dtype="str">johndoe</username>
    <is_active dtype="bool">true</is_active>
    <account_balance dtype="float">2.75</account_balance>

    <preferences dtype="dict">
        <newsletter_subscribed dtype="bool">false</newsletter_subscribed>
        <theme>dark</theme>
    </preferences>

    <roles dtype="list">
        <role>admin</role>
        <role>editor</role>
    </roles>

    <login_history dtype="list">
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.100</ip_address>
            <successful dtype="bool">true</successful>
        </login_event>
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.0</ip_address>
            <successful dtype="bool">false</successful>
        </login_event>
    </login_history>
</user_profile>
""" # noqa: E501


@register_typed_parser
class TypedXMLParser(BaseTypedParser):
    typed_parser_type = "typed_xml"
    template = TYPED_XML_TEMPLATE

    @classmethod
    def _check_available(cls):
        if minidom is None and defused_ET is None:
            raise ImportError(
                "`defusedxml` client is not available. Install with "
                "`pip install msgflux[xml]`"
            )

    @classmethod
    def _xml_to_typed_value(cls, element: ET.Element) -> Any:
        """
        Convert an XML element to a Python value based on type.
        This is a helper method, marked as a classmethod.
        """
        dtype_attr = element.attrib.get("dtype", "str")

        if dtype_attr == "dict":
            result = {}
            for child in element:
                result[child.tag] = cls._xml_to_typed_value(child)
            return result
        elif dtype_attr == "list":
            return [cls._xml_to_typed_value(child) for child in element]
        elif dtype_attr in _type_converters:
            converter = _type_converters[dtype_attr]
            return converter(element.text)
        else:
            raise ValueError(f"Unknown dtype: {dtype_attr}")

    @classmethod
    def parse(
        cls, typed_xml: str, *, extract_root_values: Optional[bool] = True
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
        cls._check_available()

        typed_xml = apply_xml_tags("root", typed_xml)
        root = defused_ET.fromstring(typed_xml)
        tag_count = defaultdict(int)
        temp_result = defaultdict(list)

        for child in root:
            tag_count[child.tag] += 1
            value = cls._xml_to_typed_value(child)
            temp_result[child.tag].append(value)

        result = {}
        for tag, values in temp_result.items():
            if tag_count[tag] == 1:
                result[tag] = values[0]
            else:
                result[tag] = values
        
        if extract_root_values and len(result) == 1:
            result = next(iter(result.values()))
            if not isinstance(result, dict):
                 # handles case where root has a single, non-dict child
                 result = {list(root)[0].tag: result}
        
        result = dotdict(result)
        return result

    @classmethod
    def repr_from_dict(cls, data: Dict[str, Any]) -> str:
        """Converts a dictionary into a typed XML string
        without a root tag, formatted readably.
        """
        cls._check_available()

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

    @classmethod
    def _build_compact_field_xml(cls, parent, name, schema, required_fields=None):
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
                cls._build_compact_field_xml(
                    field_el, sub_name, sub_schema, schema.get("required", [])
                )

        elif schema.get("type") == "array" and "items" in schema:
            cls._build_compact_field_xml(
                field_el, f"{name}_item", schema["items"],
                schema.get("items", {}).get("required", [])
            )

    @classmethod
    def schema_from_response_format(cls, response_format: Dict[str, Any]) -> str:
        """Converts a response format object to XML-schema."""
        cls._check_available()

        root_name = response_format["json_schema"]["name"]
        root_schema = response_format["json_schema"]["schema"]

        root_el = ET.Element(root_name)
        for prop_name, prop_schema in root_schema.get("properties", {}).items():
            cls._build_compact_field_xml(
                root_el, prop_name, prop_schema, root_schema.get("required", [])
            )

        xml_str = ET.tostring(root_el, encoding="unicode")

        try:
            defused_ET.fromstring(xml_str)
            pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
        except Exception: # Fallback
            pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

        return "\n".join(pretty_str.split("\n")[1:]) # remove XML head
