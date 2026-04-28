from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Mapping, Optional
from uuid import uuid4

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult
from msgflux.utils.chat import (
    ChatBlock,
    response_format_from_json_schema,
    schema_fragment_from_msgspec_type,
)
from msgflux.utils.msgspec import restore_transport_value

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses
    from msgflux.tools.definitions import ToolDefinitions

REACT_SYSTEM_MESSAGE = """
You are an Agent. In each episode, you will be given the task as input.
And you can see your past trajectory so far.

Your goal is to use one or more of the supplied tools to collect any necessary
information for producing the `final_answer`.

To do this, you will generate a `thought` containing your reasoning and plan.
Identify and define necessary `actions` by creating a list of action objects.
You MUST use the available tools when needed to achieve the objective.
Include the function `name` and `arguments` for each call.
Await the observations for the tool calls.
Analyze the results and repeat the thought-action cycle if necessary.
Once the objective is met using the tools, provide the `final_answer`.

Do NOT provide the `final_answer` before completing the required tool calls.
Optional fields may be omitted.
"""

REACT_TOOLS_TEMPLATE = """
{% set tool_choice = tool_choice or "auto" %}
You are a function calling AI model. You may call one or more functions
to assist with the user query. Don't make assumptions about what values
to plug into functions. Here are the available tools:

{%- set type_map = {"integer": "int", "number": "float", "string": "string", "boolean": "bool", "object": "object"} -%}
{%- macro render_type(spec) -%}
{%- set raw = spec.get('type', 'unknown') -%}
{%- set mapped = type_map.get(raw, raw) -%}
{%- if raw == 'array' and spec.get('items') -%}
list[{{ type_map.get(spec['items'].get('type', 'unknown'), spec['items'].get('type', 'unknown')) }}]
{%- elif raw == 'array' -%}
list
{%- else -%}
{{ mapped }}
{%- endif -%}
{%- endmacro -%}

{%- macro render_properties(properties, required, indent=0) -%}
{%- for arg, spec in properties.items() %}
{{ "  " * indent }}- {{ arg }} ({{ render_type(spec) }}
{%- if arg in required %}, required{% endif %})
{%- if spec.get('description') %}
{{ "  " * (indent + 1) }}{{ spec['description'] }}
{%- endif %}
{%- if spec.get('enum') %}
{{ "  " * (indent + 1) }}Options: {{ spec['enum'] | join(', ') }}
{%- endif %}
{%- if spec.get('type') == "object" and spec.get('properties') %}
{{ render_properties(spec['properties'], spec.get('required', []), indent + 1) }}
{%- elif spec.get('type') == "array" and spec.get('items') and
spec['items'].get('type') == "object" %}
{{ "  " * (indent + 1) }}Array items:
{{ render_properties(spec['items']['properties'], spec['items'].get('required', []), indent + 2) }}
{%- endif %}
{%- endfor %}
{%- endmacro %}

{%- for tool in tool_schemas %}
{%- set params = tool['function']['parameters'] %}
<tool>{{ tool['function']['name'] }}
{{ tool['function']['description'] }}
Parameters:
{{ render_properties(params['properties'], params.get('required', [])) }}
</tool>
{%- endfor %}

Tool choice: {{ tool_choice }}

Each action must include the function name and an `arguments` object containing
the tool parameters.
"""

ToolArguments = dict[str, Any]


class Action(Struct):
    """A tool call action with normalized tool arguments."""

    name: str
    arguments: Optional[ToolArguments] = None


class ReAct(Struct, ToolFlowControl):
    extract_reasoning: ClassVar[bool] = True
    reasoning_field: ClassVar[str] = "thought"

    thought: str
    actions: Optional[List[Action]] = None
    final_answer: Optional[str] = None

    @classmethod
    def build_provider_response_format(
        cls, tool_definitions: Optional["ToolDefinitions"] = None
    ) -> Optional[Dict[str, Any]]:
        """Build a dynamic OpenAI transport schema from the available tools."""
        action_variants = []
        tool_schemas = tool_definitions.schemas if tool_definitions else None
        for tool_schema in tool_schemas or []:
            function_schema = deepcopy(tool_schema["function"])
            parameters = function_schema.get("parameters", {})
            properties = deepcopy(parameters.get("properties", {}))
            required = list(parameters.get("required", []))
            action_variant = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [function_schema["name"]]},
                    **properties,
                },
                "required": ["name", *required],
                "additionalProperties": False,
            }
            action_variants.append(action_variant)

        action_items = None
        if len(action_variants) == 1:
            action_items = action_variants[0]
        elif action_variants:
            action_items = {"anyOf": action_variants}

        final_answer_type = Optional[str]
        for base in cls.__mro__:
            annotations = getattr(base, "__dict__", {}).get("__annotations__", {})
            if "final_answer" in annotations:
                final_answer_type = annotations["final_answer"]
                break

        schema = {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "actions": (
                    {
                        "anyOf": [
                            {"type": "array", "items": action_items},
                            {"type": "null"},
                        ]
                    }
                    if action_items is not None
                    else {"type": "null"}
                ),
                "final_answer": schema_fragment_from_msgspec_type(final_answer_type),
            },
            "required": ["thought", "actions", "final_answer"],
            "additionalProperties": False,
        }
        return response_format_from_json_schema(schema, cls.__name__.lower())

    @classmethod
    def normalize_provider_response(
        cls,
        raw_response: Mapping[str, Any],
        tool_definitions: Optional["ToolDefinitions"] = None,
    ) -> Mapping[str, Any]:
        """Normalize flattened action params to the logical Action(arguments=...)."""
        normalized = {
            "thought": raw_response.get("thought"),
            "actions": None,
            "final_answer": raw_response.get("final_answer"),
        }

        actions = raw_response.get("actions")
        if actions is None:
            return normalized

        normalized_actions = []
        tool_annotations = (
            tool_definitions.annotations if tool_definitions else None
        ) or {}
        for action in actions:
            name = action.get("name")
            argument_annotations = tool_annotations.get(name, {})
            arguments = {}
            for key, value in action.items():
                if key in {"name", "_id"}:
                    continue
                arguments[key] = restore_transport_value(
                    value,
                    argument_annotations.get(key, Any),
                )
            normalized_actions.append({"name": name, "arguments": arguments or None})

        normalized["actions"] = normalized_actions
        return normalized

    @classmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> ToolFlowResult:
        """Extract flow information from ReAct response."""
        final_answer = raw_response.get("final_answer")
        if final_answer is not None:
            raw_response.pop("actions", None)
            return ToolFlowResult(
                is_complete=True,
                tool_calls=None,
                reasoning=None,
                final_response=raw_response,
            )

        actions = raw_response.get("actions")
        if actions:
            tool_calls = []
            for act in actions:
                tool_id = str(uuid4())
                act["_id"] = tool_id
                args = act.get("arguments")
                if isinstance(args, Mapping):
                    args_dict = dict(args)
                else:
                    args_dict = args
                tool_calls.append((tool_id, act.get("name"), args_dict))

            return ToolFlowResult(
                is_complete=False,
                tool_calls=tool_calls,
                reasoning=raw_response.get("thought"),
                final_response=None,
            )

        raw_response.pop("actions", None)
        return ToolFlowResult(
            is_complete=True,
            tool_calls=None,
            reasoning=None,
            final_response=raw_response,
        )

    @classmethod
    def inject_results(
        cls, raw_response: Mapping[str, Any], tool_results: "ToolResponses"
    ) -> Mapping[str, Any]:
        """Inject tool results back into ReAct structure as observations."""
        actions = raw_response.get("actions") or []
        observations = []
        for act in actions:
            call = tool_results.get_by_id(act.get("_id"))
            if call is not None:
                observations.append(
                    {
                        "tool": act.get("name"),
                        "result": call.result or call.error,
                    }
                )
        raw_response["observations"] = observations
        return raw_response

    @classmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration."""
        step = {
            "thought": raw_response.get("thought"),
            "actions": [
                {"name": a.get("name"), "arguments": a.get("arguments")}
                for a in (raw_response.get("actions") or [])
            ],
            "observations": raw_response.get("observations", []),
        }
        if messages and messages[-1].get("role") == "assistant":
            last_react_msg = messages[-1].get("content")
            react_state = msgspec.json.decode(last_react_msg)
            react_state.append(step)
            messages[-1] = ChatBlock.assist(react_state)
        else:
            react_state = [step]
            messages.append(ChatBlock.assist(react_state))
        return messages


ReAct.system_message = REACT_SYSTEM_MESSAGE
ReAct.tools_template = REACT_TOOLS_TEMPLATE
