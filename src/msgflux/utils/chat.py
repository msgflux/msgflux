import copy
import re
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_origin,
)

from jinja2 import Template

from msgflux.logger import logger
from msgflux.utils.inspect import get_mime_type
from msgflux.utils.xml import apply_xml_tags


class ChatML:
    """Manage messages in ChatML format."""

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        """Inicializa o gerenciador com um histórico opcional."""
        self.history = messages if messages is not None else []

    def add_user_message(self, content: Union[str, Dict[str, Any]]):
        """Adds a message with role `user`."""
        self._add_message("user", content)

    def add_assist_message(self, content: Union[str, Dict[str, Any]]):
        """Adds a message with role `assistant`."""
        self._add_message("assistant", content)

    def add_tool_message(self, content: Union[str, Dict[str, Any]]):
        """Adds a message with role `tool`."""
        self._add_message("tool", content)

    def _add_message(self, role: str, content: Union[str, Dict[str, Any]]):
        """Internal method to add message to history."""
        if role not in ["user", "assistant", "tool"]:
            raise ValueError(
                f"Role must be `user`, `assistant` or `tool` given `{role}`"
            )
        message = {"role": role, "content": content}
        self.history.append(message)

    def extend_history(self, messages):
        """Add a list of messages to the history."""
        return self.history.extend(messages)

    def get_messages(self):
        return self.history

    def clear(self):
        self.history = []


def format_examples(
    examples: List[Union[Tuple[str, str], Tuple[str, str, str]]],
) -> str:
    """Formats a list of examples into XML-style string format.

    Each example in the list should be a tuple containing input and output strings,
    with an optional title as the third element. The function generates sequential IDs
    for each example starting from 1.

    Args:
        examples: A list of tuples where each tuple contains:
            - Input string (required)
            - Output string (required)
            - Title string (optional)

    Returns:
        A formatted XML-style string containing all examples.

    Example:
        >>> examples = [
        ...     ("What is your name?", "My name is GPT-3.5.", "Introduction"),
        ...     ("What day is today?", "Today is Tuesday."),
        ... ]
        >>> print(format_examples(examples))
        <example id=1 title="Introduction">
        <input>What is your name?</input>
        <output>My name is GPT-3.5.</output>
        </example>

        <example id=2>
        <input>What day is today?</input>
        <output>Today is Tuesday.</output>
        </example>
    """
    result = []

    for i, example in enumerate(examples, start=1):
        if len(example) == 3:
            input_text, output_text, title = example
            result.append(f'<example id={i} title="{title}">')
        else:
            input_text, output_text = example
            result.append(f"<example id={i}>")

        result.append(apply_xml_tags("input", input_text))
        result.append(apply_xml_tags("output", output_text))
        result.append("</example>\n")

    return "\n".join(result)


def adapt_struct_schema_to_json_schema(
    original_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a Msgspec.Struct in Json Schema ChatCompletion-like."""

    def resolve_ref(ref: str, defs: Dict) -> Dict:
        """Resolves a reference `$ref` using the dictionary `$defs`."""
        ref_key = ref.split("/")[-1]
        return defs.get(ref_key, {})

    root_ref = original_schema.get("$ref", "")
    defs = original_schema.get("$defs", {})

    root_schema = resolve_ref(root_ref, defs)

    def deep_resolve_and_enforce_properties(schema: Dict) -> Dict:
        if "$ref" in schema:
            schema = resolve_ref(schema["$ref"], defs)

        # Enforce additionalProperties: false for all object types
        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        if "properties" in schema:
            schema["properties"] = {
                k: deep_resolve_and_enforce_properties(v)
                for k, v in schema["properties"].items()
            }

        if "items" in schema:
            schema["items"] = deep_resolve_and_enforce_properties(schema["items"])

        return schema

    resolved_schema = deep_resolve_and_enforce_properties(root_schema)

    adapted_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": root_schema.get("title", "response").lower(),
            "schema": {
                "type": resolved_schema.get("type", "object"),
                "properties": resolved_schema["properties"],
                "required": resolved_schema.get("required", []),
                "additionalProperties": False,
            },
            "strict": False,
        },
    }

    return adapted_schema


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

    Args:
        docstring: Complete docstring of the function/class

    Returns:
        Dictionary with parameter descriptions
    """
    if not docstring:
        return {}

    # Find the Args section
    args_match = re.search(
        r"Args:\s*(.*?)(?:\n\n|\n[A-Za-z]+:|\Z)", docstring, re.DOTALL
    )
    if not args_match:
        return {}

    # Extract parameter descriptions
    args_text = args_match.group(1).strip()
    param_descriptions = {}

    # Process line by line to avoid capturing descriptions of other parameters
    lines = args_text.split("\n")
    current_param = None
    current_desc = []

    for raw_line in lines:
        line = raw_line.strip()
        # Find a new parameter
        param_match = re.match(r"(\w+)\s*\((.*?)\):\s*(.+)", line)

        if param_match:
            # Save description of previous parameter if exists
            if current_param:
                param_descriptions[current_param] = " ".join(current_desc).strip()

            # Start new parameter
            current_param = param_match.group(1)
            current_desc = [param_match.group(3)]
        elif current_param and line:
            # Continue description of current parameter
            current_desc.append(line)

    # Save last description
    if current_param:
        param_descriptions[current_param] = " ".join(current_desc).strip()

    return param_descriptions


def generate_json_schema(cls: type) -> Dict[str, Any]:
    """Generates a JSON schema for a class based on its characteristics.

    Args:
        cls: The class to generate the schema for

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

        # Check if enum is defined
        if hasattr(type_hint, "__args__") and type_hint.__origin__ is Literal:
            prop_schema["enum"] = list(type_hint.__args__)

        # Add parameter description if available
        if param in param_descriptions:
            prop_schema["description"] = param_descriptions[param]

        # Mark as required
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


# TODO: needs improvement to write encoded json
# TODO virar jinja template
def get_react_tools_prompt_format(tool_schemas):
    template = Template("""
    You are a function calling AI model. You may call one or more functions
    to assist with the user query. Don't make assumptions about what values
    to plug into functions. Here are the available tools:


    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']
                        ['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}

    For each function call return a encoded json object with function name
    and arguments within <tool_call></tool_call> XML tags as follows:
    """)
    react_tools = template.render(tools=tool_schemas)
    return react_tools


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
                                "Warning: Skipping malformed 'input_audio' part "
                                "at index {i}: {part}"
                            )
                            processed_content.append(part)
                    else:
                        # Keep the original part if 'input_audio' is not a dict
                        logger.warnning(
                            "Warning: Skipping malformed 'input_audio' part "
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
