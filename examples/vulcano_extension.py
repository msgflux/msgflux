from msgflux.vulcano import CommandResult, EventDraft, EventType

EXTENSION_NAME = "example"
EXTENSION_API_VERSION = 1


def setup(api):
    @api.command("hello", "Show where this extension was loaded from.")
    def hello(args, ctx):
        name = args or "developer"
        source = ctx.api.source.kind
        generation = ctx.api.generation
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
