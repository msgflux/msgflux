import copy
import inspect
import re
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    Type,
    get_args,
    get_origin,
)

import msgspec

from msgflux.generation.control_flow import ToolFlowControl
from msgflux.logger import logger
from msgflux.utils.inspect import get_mime_type
from msgflux.utils.msgspec import msgspec_dumps
from msgflux.utils.validation import is_subclass_of


class ChatBlockMeta(type):
    def __call__(
        cls,
        role: str,
        content: str,
        media: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        role = role.lower()
        role_map = {
            "user": cls.user,
            "assist": cls.assist,
            "system": cls.system
        }
        if role not in role_map:
            raise ValueError(
                f"Invalid role `{role}`. Use {', '.join(role_map)}")
        if role == "user":
            return role_map[role](content, media)
        return role_map[role](content)


class ChatBlock(metaclass=ChatBlockMeta):    
    @classmethod
    def user(
        cls, 
        content: Union[str, List[Dict[str, Any]]],
        media: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if media is None:
            return {"role": "user", "content": content}
        content_list = []
        if content:
            content_list.append({"type": "text", "text": content})
        if isinstance(media, list):
            content_list.extend(media)
        else:
            content_list.append(media)
        return {"role": "user", "content": content_list}
    
    @classmethod
    def assist(cls, content: Any) -> Dict[str, str]:
        if not isinstance(content, str):
            content = msgspec_dumps(content)
        return {"role": "assistant", "content": content}
    
    @classmethod
    def system(cls, content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}
    
    @staticmethod
    def tool_call(id: str, name: str, arguments: str) -> Dict[str, str]:
        return {
            "id": id,            
            "type": "function",
            "function": {"name": name, "arguments": arguments}
        }
    
    @classmethod
    def assist_tool_calls(cls, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"role": "assistant", "tool_calls": tool_calls}

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }

    @staticmethod
    def text(text: str) ->  Dict[str, str]:
        return {"type": "text", "text": text}

    @staticmethod
    def image(
        url: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(url, list):
            return [{
                "type": "image_url",
                "image_url": {"url": u}
            } for u in url]
        return {
            "type": "image_url",
            "image_url": {"url": url}
        }

    @staticmethod
    def video(
        url: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(url, list):
            return [{
                "type": "video_url",
                "video_url": {"url": u}
            } for u in url]
        return {
            "type": "video_url",
            "video_url": {"url": url}
        }

    @staticmethod
    def audio(data: str, format: str) -> Dict[str, str]:
        return {
            "type": "input_audio",
            "input_audio": {"data": data, "format": format}
        }
    
    @staticmethod
    def file(filename: str, file_data: str) -> Dict[str, str]:
        return {
            "type": "file",
            "file": {"filename": filename, "file_data": file_data}
        }


class ChatML:
    """Manage messages in ChatML format."""

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        self.history = messages if messages is not None else []

    def add_user_message(
        self, 
        content: Union[str, Dict[str, Any]],
        media: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        """Adds a message with role `user`."""
        if isinstance(content, dict):
            self._add_message(content)
        self._add_message(ChatBlock.user(content, media))

    def add_assist_message(self, content: Union[str, Dict[str, Any]]):
        """Adds a message with role `assistant`."""
        if isinstance(content, dict):
            self._add_message(content)
        self._add_message(ChatBlock.assist(content))

    #def add_tool_message(self, content: Union[str, Dict[str, Any]]):
    #    """Adds a message with role `tool`."""
    #    self._add_message("tool", content) TODO

    def _add_message(self, message: Dict[str, Any]):
        """Internal method to add message to history."""
        self.history.append(message)

    def extend_history(self, messages):
        """Add a list of messages to the history."""
        return self.history.extend(messages)

    def get_messages(self):
        return self.history

    def clear(self):
        self.history = []


complex_arguments_schema = {
    "anyOf": [
        {
            "type": "object",
            # 'additionalProperties' agora define o schema para os VALORES
            "additionalProperties": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "number"}
                            ]
                        }
                    }
                ]
            }
        },
        {"type": "null"}
    ]
}

def response_format_from_msgspec_struct(
    struct_class: Type[msgspec.Struct],
) -> Dict[str, Any]:
    """Converts a msgspec.Struct to OpenAI's response_format format."""
    def _dereference_schema(schema_node: Any, definitions: Dict[str, Any]) -> Any:
        """Helper function to replace references '$ref'."""
        if isinstance(schema_node, dict):
            if '$ref' in schema_node:
                ref_name = schema_node['$ref'].split('/')[-1]
                return _dereference_schema(definitions[ref_name], definitions)
            else:
                return {key: _dereference_schema(value, definitions) for key, value in schema_node.items()}
        elif isinstance(schema_node, list):
            return [_dereference_schema(item, definitions) for item in schema_node]
        return schema_node

    def _add_additional_properties_false(schema_node: Any) -> None:
        """
        Recursively traverses the schema and adds
        'additionalProperties': False to all objects that have properties.
        Modifies the schema_node "in-place" (directly on the object).
        """        
        if isinstance(schema_node, dict):
            
            if schema_node.get("type") == "object":
                schema_node["additionalProperties"] = False
            for value in schema_node.values():
                _add_additional_properties_false(value)
        elif isinstance(schema_node, list):
            for item in schema_node:
                _add_additional_properties_false(item)

    def _ensure_all_properties_are_required(schema_node: Any) -> None:
        """It traverses the schema and, for each object, ensures that
        all of its properties are listed under 'required'.
        """        
        if isinstance(schema_node, dict):
            if schema_node.get("type") == "object" and "properties" in schema_node:
                all_property_keys = list(schema_node["properties"].keys())
                schema_node["required"] = sorted(all_property_keys)
            for value in schema_node.values():
                _ensure_all_properties_are_required(value)
        elif isinstance(schema_node, list):
            for item in schema_node:
                _ensure_all_properties_are_required(item)

    def _find_and_patch_property(
        schema_node: Any, prop_name: str, patch_schema: Dict[str, Any]
    ):
        """Recursively finds a property by name and replaces its schema."""
        if isinstance(schema_node, dict):
            if "properties" in schema_node and prop_name in schema_node["properties"]:
                schema_node["properties"][prop_name] = patch_schema

            # Continue the recursive search
            for value in schema_node.values():
                _find_and_patch_property(value, prop_name, patch_schema)
                
        elif isinstance(schema_node, list):
            for item in schema_node:
                _find_and_patch_property(item, prop_name, patch_schema)

    msgspec_schema = msgspec.json.schema(struct_class)
    definitions = msgspec_schema.get("$defs", {})
    root_ref = msgspec_schema.get("$ref")
    root_name = root_ref.split("/")[-1]
    root_definition = definitions.get(root_name)
    inlined_schema = _dereference_schema(root_definition, definitions)
    _add_additional_properties_false(inlined_schema)
    _ensure_all_properties_are_required(inlined_schema)

    if is_subclass_of(struct_class, ToolFlowControl):
        # Hack: LM providers NOT support `Any` type. ToolFlowControl needs receive
        # arguments, msgspec not render complex types as a long Union.
        # Force a complex type in `arguments`.
        _find_and_patch_property(inlined_schema, "arguments", complex_arguments_schema)

    inlined_schema.pop("title", None)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": struct_class.__name__.lower(),
            "schema": inlined_schema,
            "strict": True
        }
    }
    return response_format


# TODO tirar e deixar como um tutorial
def chatml_to_steps_format(
    model_state: List[Dict[str, Any]], response: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    steps = []
    pending_tool_calls = {}

    for message in model_state:
        if message["role"] == "user" and "content" in message:
            steps.append({"task": message["content"]})

        elif message["role"] == "assistant" and "content" in message:
            steps.append({"assistant": message["content"]})

        elif message.get("tool_calls"):
            # Iterates over all function calls in the `tool_calls` list
            for tool_call in message["tool_calls"]:
                fn_call_entry = {
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                    "results": None,  # To be updated when the answer is found
                }
                # Add each function call separately
                steps.append({"tool_call": fn_call_entry})
                pending_tool_calls[tool_call["id"]] = fn_call_entry

        elif message["role"] == "tool" and message.get("tool_call_id"):
            # Check if there is a corresponding function call pending
            tool_call_id = message["tool_call_id"]
            if tool_call_id in pending_tool_calls:
                # Update the result of the corresponding function call
                pending_tool_calls[tool_call_id]["result"] = message.get("content", "")

    if response:
        steps.append({"assistant": response})

    return steps


def clean_docstring(docstring: str) -> str:
    """Cleans the docstring by removing the Args section.

    Args:
        docstring: Complete docstring to clean

    Returns:
        Clean docstring without Args section
    """
    if not docstring:
        return ""

    # Remove the Args section and any text after it
    cleaned = re.sub(r"\s*Args:.*", "", docstring, flags=re.DOTALL).strip()

    return cleaned

def parse_docstring_args(docstring: str) -> Dict[str, str]:
    """Extracts parameter descriptions from the Args section of the docstring.

    Supports: 
        - name: 
            multi-line description... 
        - name: single-line description 
        - name (type): description

    Args:
        docstring: Complete docstring of the function/class

    Returns:
        Dictionary with parameter descriptions
    """
    # Remove identation
    docstring = inspect.cleandoc(docstring)

    if not docstring:
        return {}

    lines = docstring.splitlines()
    
    # find beginning of Args section:
    start = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*Args:\s*$', ln):
            start = i + 1
            break
        # also accepts "Args: text" on the same line (e.g. "Args: param: desc")
        if re.match(r"^\s*Args:\s+\S", ln):
            # takes the text after "Args:" and treats it as the first lines
            rest = ln.split("Args:", 1)[1].rstrip()
            lines[i] = rest
            start = i
            break

    if start is None:
        return {}

    param_descriptions: Dict[str, str] = {}
    current_param = None
    current_desc_lines = []

    # RegEx for parameter definition line:
    # capture: name, optional type, optional inline description
    param_def_re = re.compile(r"^\s{0,4}([A-Za-z_]\w*)(?:\s*\(([^)]*)\))?\s*:\s*(.*)$")

    # header_re identifies the next section (e.g., Returns:, Raises:, Examples:)
    # without indentation
    header_re = re.compile(r"^[A-Za-z][A-Za-z0-9 _]*:\s*$")

    for ln in lines[start:]:
        # if we find a header without indentation -> end of Args section
        if header_re.match(ln) and not ln.startswith(" "):
            break

        # try to match a parameter definition line
        m = param_def_re.match(ln)
        if m:
            # record previous parameter
            if current_param:
                param_descriptions[current_param] = " ".join(
                    p.strip() for p in current_desc_lines if p.strip()
                ).strip()

            current_param = m.group(1)
            inline_desc = m.group(3) or ""
            current_desc_lines = []
            if inline_desc:
                current_desc_lines.append(inline_desc.strip())
            # next lines (indented) will be part of the description
            continue

        # indented or continuation lines (start with space) are part of the description
        if current_param and (ln.startswith(" ") or ln.strip() == ""):
            # removes only 4 spaces of common indentation
            # (keeps relative sub-indentation)
            current_desc_lines.append(ln.strip())
            continue

    # save last param if exists
    if current_param:
        param_descriptions[current_param] = " ".join(
            p.strip() for p in current_desc_lines if p.strip()
        ).strip()

    return param_descriptions

def generate_json_schema(cls: type) -> Dict[str, Any]:
    """Generates a JSON schema for a class based on its characteristics.

    Args:
        cls:
            The class to generate the schema for

    Returns:
        JSON schema for the class
    """
    name = cls.get_module_name()
    description = cls.get_module_description()
    clean_description = clean_docstring(description)
    param_descriptions = parse_docstring_args(description)
    annotations = cls.get_module_annotations()

    properties = {}
    required = []

    for param, type_hint in annotations.items():
        if param == "return":
            continue

        prop_schema = {"type": "string"}  # Default as string

        # Check if enum is defined via typing.Literal
        if get_origin(type_hint) is Literal:
            prop_schema["enum"] = list(get_args(type_hint))

        # Add parameter description if available
        if param in param_descriptions:
            prop_schema["description"] = param_descriptions[param]

        # Mark as required unless Union (i.e., Optional) or default present in annotations structure
        if get_origin(type_hint) is not Union:
            required.append(param)

        properties[param] = prop_schema

    json_schema = {
        "name": name,
        "description": clean_description or f"Function for {name}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }

    return json_schema

def generate_tool_json_schema(cls: type) -> Dict[str, Any]:
    tool = generate_json_schema(cls)
    tool_json_schema = {"type": "function", "function": tool}
    return tool_json_schema

def adapt_messages_for_vllm_audio(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Adapts a list of messages from ChatML format, converting audio parts of type
    'input_audio' (OpenAI style) to type 'audio_url' with Data URI (vLLM style).

    Args:
        messages: The original list of messages.

    Returns:
        A new list of messages with the adapted audio parts.
        The original list is not modified.
    """
    adapted_messages = copy.deepcopy(messages)

    for message in adapted_messages:
        content = message.get("content")

        # Checks if the content is a list (indicating multimodality)
        if isinstance(content, list):
            processed_content = []
            for i, part in enumerate(content):
                # Check if the part is of type 'input_audio'
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    input_audio_data = part.get("input_audio")

                    # Check if internal data exists
                    if isinstance(input_audio_data, dict):
                        base64_data = input_audio_data.get("data")
                        audio_format = input_audio_data.get("format")

                        # If you have the base64 data and format, convert
                        if (
                            base64_data
                            and isinstance(base64_data, str)
                            and audio_format
                        ):
                            mime_type = get_mime_type(audio_format)
                            data_uri = f"data:{mime_type};base64,{base64_data}"

                            # Create the new structure of the audio part
                            vllm_audio_part = {
                                "type": "audio_url",
                                "audio_url": {"url": data_uri},
                            }
                            processed_content.append(vllm_audio_part)
                        else:
                            logger.warning(
                                "Skipping malformed `input_audio` part "
                                f"at index {i}: {part}"
                            )
                            processed_content.append(part)
                    else:
                        # Keep the original part if `input_audio` is not a dict
                        logger.warnning(
                            "Skipping malformed `input_audio` part "
                            f"(not a dict) at index {i}: {part}"
                        )
                        processed_content.append(part)

                else:
                    # Keep other parts (text, image, etc.) as is
                    processed_content.append(part)

            # Update the message content with the processed list
            message["content"] = processed_content
        # If the content is not a list (e.g. plain text), do nothing
    return adapted_messages
