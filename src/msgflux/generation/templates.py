class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    # TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE = """
{% if system_message or instructions or expected_output or examples or team_members or system_extra_message %}
<developer_note>
{% if system_message %}{{ system_message }}
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


TYPED_XML_TEMPLATE = """
{% if instructions %}{{ instructions }}{% endif %}

You SHOULD write your response in a structured manner using XML tags.
DO NOT write XML headers or add extra messages beyond the XML response.
You should then generate an XML specifying the dtype.
The available data types are: str (default if not specified), int, float, bool, dict and list.

Example of how you can write your response in XML:

<user_profile dtype="dict">
    <id dtype="int">1024</id>
    <username dtype="str">johndoe</username>
    <is_active dtype="bool">true</is_active>
    <account_balance dtype="float">2.75</account_balance>

    <preferences dtype="dict">
        <newsletter_subscribed dtype="bool">false</newsletter_subscribed>
        <theme>dark</theme>
    </preferences>

    <roles dtype="list">
        <role>admin</role>
        <role>editor</role>
    </roles>

    <login_history dtype="list">
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.100</ip_address>
            <successful dtype="bool">true</successful>
        </login_event>
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.0</ip_address>
            <successful dtype="bool">false</successful>
        </login_event>
    </login_history>
</user_profile>

{% if json_schema %}
Here is a JSON-schema that you SHOULD use to guide you in generating your response using XML tags:
{{ json_schema }}
{% endif %}
""" # noqa: E501

SIGNATURE_DEFAULT_SYSTEM_MESSAGE = """
Your goal is to provide accurate, helpful, and well-reasoned responses.
Carefully analyze the user's request to fully understand the objective. Address all parts of the query.
Think step-by-step to formulate your answer. Where appropriate, briefly explain your reasoning process.
Structure your response clearly and concisely using dicts, lists, or other formatting.
Strive for factual accuracy.
Be helpful and informative, focusing on directly answering the user's prompt.
""".strip() # noqa: E501
