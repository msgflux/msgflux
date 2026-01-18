from typing import Optional


def apply_xml_tags(tag_id: str, content: str, output_id: Optional[str] = None) -> str:
    if output_id is None:
        output_id = tag_id
    return f"<{tag_id}>{content}</{output_id}>"
