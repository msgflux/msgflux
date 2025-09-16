from typing import Any, List, Optional

from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl


REACT_SYSTEM_MESSAGE = """
You are an Agent. In each episode, you will be given the task as input.
And you can see your past trajectory so far.

Your goal is to use one or more of the supplied tools to collect any necessary information for producing the `final_response`.

To do this, you will generate a `thought` containing your reasoning and plan.
Identify and define necessary `actions` by creating a list of `toolcall` objects.
You MUST use the available tools when needed to achieve the objective.
Include the function `name` and `arguments` for each call.
Await the observations for the tool calls.
Analyze the results and repeat the thought-action cycle if necessary.
Once the objective is met using the tools, provide the `final_answer`.

Do NOT provide the `final_answer` before completing the required tool calls.
Optional fields may be omitted.
"""

REACT_TOOLS_TEMPLATE = """
You are a function calling AI model. You may call one or more functions
to assist with the user query. Don't make assumptions about what values
to plug into functions. Here are the available tools:

{%- macro render_properties(properties, indent=0) -%}
{%- for arg, spec in properties.items() %}
{{ "  " * indent }}- {{ arg }} ({{ spec.get('type', 'unknown') }}){%- if spec.get('description') %}: {{ spec['description'] }}{% endif %}
{%- if spec.get('enum') %}
{{ "  " * (indent + 1) }}Options: {{ spec['enum'] | join(', ') }}
{%- endif %}
{%- if spec.get('type') == "object" and spec.get('properties') %}
{{ render_properties(spec['properties'], indent + 1) }}
{%- elif spec.get('type') == "array" and spec.get('items') and spec['items'].get('type') == "object" %}
{{ "  " * (indent + 1) }}Array items:
{{ render_properties(spec['items']['properties'], indent + 2) }}
{%- endif %}
{%- endfor %}
{%- endmacro %}

{%- for tool in tools %}
<tool>{{ tool['function']['name'] }}
{{ tool['function']['description'] }}
{{ render_properties(tool['function']['parameters']['properties']) }}
</tool>
{%- endfor %}

For each function call return a encoded json object with function name
and arguments.
"""


class ToolCall(Struct):
    name: str
    arguments: Optional[Any]


class ReActStep(Struct):
    thought: str
    actions: List[ToolCall]


class ReAct(Struct, ToolFlowControl):
    current_step: Optional[ReActStep]
    final_answer: Optional[str]


ReAct.system_message = REACT_SYSTEM_MESSAGE
ReAct.tools_template = REACT_TOOLS_TEMPLATE
