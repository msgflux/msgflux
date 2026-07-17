from rich.panel import Panel

from msgflux.vulcano import CommandResult, EventDraft, EventType

EXTENSION_NAME = "example"
EXTENSION_API_VERSION = 1


def setup(api):
    api.ui.set_status("example", "example extension loaded")
    api.ui.set_widget(
        "hint",
        ["Example extension", "Try /hello Ada"],
        placement="above_editor",
    )
    api.register_message_renderer(
        "greeting",
        lambda event, _context: Panel(
            str(event.payload["content"]),
            title="Extension renderer",
            border_style="green",
        ),
    )

    @api.command("hello", "Show where this extension was loaded from.")
    async def hello(args, ctx):
        name = args or "developer"
        source = ctx.api.source.kind
        generation = ctx.api.generation
        await ctx.send_message(
            "greeting",
            f"Hello, {name}. Loaded from {source} in generation {generation}.",
        )
        return CommandResult(
            events=(
                EventDraft(
                    EventType.COMMAND_OUTPUT,
                    {
                        "text": (
                            f"Hello, **{name}**. Loaded from `{source}` "
                            f"in generation `{generation}`."
                        )
                    },
                ),
            )
        )
