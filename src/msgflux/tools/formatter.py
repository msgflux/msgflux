from jinja2 import Template


def format_tools_from_schemas(tools_template: str, tool_schemas: str) -> str:
    template = Template(tools_template)
    react_tools = template.render(tools=tool_schemas)
    return react_tools
