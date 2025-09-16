class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    # TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE = """
{% if system_message or instructions or expected_output or examples or team_members or system_extra_message %}
<developer_note>
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
</developer_note>
{% endif %}
""" # noqa: E501


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
""" # noqa: E501
