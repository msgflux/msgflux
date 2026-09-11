class PromptSpec:
    SYSTEM_PROMPT = "Instructions and stable context for the model"
    AGENT_SKILLS = "Available Agent Skills"
    # TASK_TEMPLATE = ""


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
