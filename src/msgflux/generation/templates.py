class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    # TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE = """
{% if system_message or instructions or expected_output or examples or system_extra_message or tool_usage_guidance %}
<system_note>
{% if system_message %}<system_message>
{{ system_message }}
</system_message>
{% endif %}
{% if instructions %}<instructions>
{{ instructions }}
</instructions>
{% endif %}
{% if expected_output %}<expected_output>
{{ expected_output }}
</expected_output>
{% endif %}
{% if examples %}<examples>
{{ examples }}
</examples>
{% endif %}
{% if system_extra_message %}
{{ system_extra_message }}
{% endif %}
{% if current_date %}
The current date is: {{ current_date }}
{% endif %}
{% if tool_usage_guidance %}<tool_usage_guidance>
{% for tool in tool_usage_guidance %}<tool name="{{ tool.name }}">
{{ tool.guidance }}
</tool>
{% endfor %}</tool_usage_guidance>
{% endif %}
</system_note>
{% endif %}
"""  # noqa: E501


EXPECTED_OUTPUTS_TEMPLATE = """
{% if expected_inputs or expected_outputs %}
{% if expected_inputs %}
Your task inputs are:

{{ expected_inputs }}
{% endif %}

{% if expected_outputs %}
Your task outputs are:
{{ expected_outputs }}
Be consise in choosing your answers.
{% endif %}
{% endif %}
"""
