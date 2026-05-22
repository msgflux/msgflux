class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    AGENT_SKILLS = "Available Agent Skills"
    # TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE = """
{% if system_message or instructions or expected_output or examples or system_extra_message or agent_skills_enabled %}
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
{% if agent_skills_enabled %}<agent_skills>
Skills are reusable local instructions for specialized workflows. Use one when it matches the task. To load a skill, call `activate_skill` with its name before following its workflow. Loaded skill content is returned as a tool result message. Treat it as task-relevant instructions for this run. Resolve relative paths from the skill location.
{% if agent_skills %}
<available_skills>
{% for skill in agent_skills %}
<skill>
name: {{ skill.name }}
description: {{ skill.description }}
location: {{ skill.location }}
</skill>
{% endfor %}
</available_skills>
{% else %}
No skills are listed in this prompt.
{% endif %}
{% if agent_skill_search_enabled %}
Use `skill_search` to find relevant skills not listed above.
{% endif %}
</agent_skills>
{% endif %}
{% if current_date %}
The current date is: {{ current_date }}
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
