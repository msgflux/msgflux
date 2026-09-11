"""Built-in extensions for optional system-prompt capabilities."""

from __future__ import annotations

from dataclasses import replace
from inspect import cleandoc
from typing import Any, Callable, List, Mapping, Union

from msgflux.core.examples import Example, ExampleCollection
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook, ModelContext
from msgflux.utils.msgspec import msgspec_dumps
from msgflux.utils.time import utc_current_date

__all__ = [
    "CurrentDateExtension",
    "FewShotExamplesExtension",
    "ToolUsageGuidanceExtension",
]


def _append_section(ctx: ModelContext, section: str) -> ModelContext:
    prompt = f"{ctx.system_prompt}\n\n{section}" if ctx.system_prompt else section
    return replace(ctx, system_prompt=prompt)


class FewShotExamplesExtension(AgentExtension):
    """Add stable few-shot examples to the model-facing system prompt."""

    def __init__(
        self,
        examples: Union[str, List[Union[Example, Mapping[str, Any]]]],
    ) -> None:
        super().__init__("few_shot_examples")
        if isinstance(examples, str):
            rendered = cleandoc(examples)
        elif isinstance(examples, list):
            rendered = ExampleCollection(examples).get_formatted(
                msgspec_dumps,
                msgspec_dumps,
            )
        else:
            raise TypeError(
                "`examples` must be a string or list of Example mappings, "
                f"given `{type(examples)}`"
            )
        if not rendered:
            raise ValueError("`examples` must not be empty")
        self.rendered_examples = rendered

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_examples),)

    def _add_examples(self, ctx: ModelContext) -> ModelContext:
        section = f"<examples>\n{self.rendered_examples}\n</examples>"
        return _append_section(ctx, section)


class CurrentDateExtension(AgentExtension):
    """Add the current UTC date to the rendered system prompt."""

    def __init__(self, date_factory: Callable[[], str] = utc_current_date) -> None:
        super().__init__("current_date")
        self.date_factory = date_factory

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_date),)

    def _add_date(self, ctx: ModelContext) -> ModelContext:
        return _append_section(ctx, f"The current date is: {self.date_factory()}")


class ToolUsageGuidanceExtension(AgentExtension):
    """Render guidance owned by tools in the active request catalog."""

    def __init__(self) -> None:
        super().__init__("tool_usage_guidance")

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_guidance),)

    def _add_guidance(self, ctx: ModelContext) -> ModelContext:
        guidance = self.agent.tool_library.get_tool_usage_guidance(ctx.tool_names)
        if not guidance:
            return ctx
        lines = ["<tool_usage_guidance>"]
        for tool in guidance:
            lines.extend(
                [
                    f'<tool name="{tool["name"]}">',
                    tool["guidance"],
                    "</tool>",
                ]
            )
        lines.append("</tool_usage_guidance>")
        return _append_section(ctx, "\n".join(lines))
