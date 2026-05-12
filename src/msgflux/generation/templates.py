class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    AGENT_SKILLS = "Available Agent Skills"
    # TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE = """
{% if system_message or instructions or expected_output or examples or system_extra_message or agent_skills or agent_skill_search_enabled %}
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
{% if agent_skills or agent_skill_search_enabled %}<agent_skills>
The following Agent Skills are available. Use them when the task matches a skill description. To activate a skill, call `activate_skill` with the skill name before following that skill's workflow. When a skill references relative paths, resolve them against the skill directory returned by `activate_skill`.
{% if agent_skills %}
<available_skills>
{% for skill in agent_skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{ skill.location }}</location>
  </skill>
{% endfor %}
</available_skills>
{% endif %}
{% if agent_skill_search_enabled %}
More skills are available. Use `skill_search` to find hidden discoverable skills.
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
