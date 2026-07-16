import pytest

from msgflux.vulcano import CommandContext, CommandResult
from msgflux.vulcano.commands import CommandRegistry, CommandSpec


def _handler(context, invocation):
    del context, invocation
    return CommandResult()


class _CommandApi:
    services = {}


def test_registry_resolves_alias_and_registration_can_be_removed():
    registry = CommandRegistry()
    command = CommandSpec(
        name="review",
        aliases=("inspect",),
        description="Review the working tree.",
        handler=_handler,
    )

    registration = registry._register(command)

    assert registry.resolve("review") is command
    assert registry.resolve("/inspect") is command
    assert "inspect" in registry

    registration.remove()

    assert "review" not in registry
    assert "inspect" not in registry


def test_registry_rejects_colliding_aliases():
    registry = CommandRegistry()
    registry._register(
        CommandSpec(
            name="review",
            aliases=("inspect",),
            description="Review the working tree.",
            handler=_handler,
        )
    )

    with pytest.raises(ValueError, match="inspect"):
        registry._register(
            CommandSpec(
                name="diagnose",
                aliases=("inspect",),
                description="Diagnose a failure.",
                handler=_handler,
            )
        )


def test_registry_parses_quoted_arguments():
    invocation = CommandRegistry().parse('/review "src/my file.py" --strict')

    assert invocation.name == "review"
    assert invocation.arguments == ("src/my file.py", "--strict")
    assert invocation.raw == '/review "src/my file.py" --strict'


@pytest.mark.asyncio
async def test_registry_accepts_async_extension_handler():
    registry = CommandRegistry()

    async def review(context, invocation):
        assert isinstance(context, CommandContext)
        assert invocation.arguments == ("src",)
        return CommandResult()

    registry._register(
        CommandSpec(
            name="review",
            description="Review a path.",
            handler=review,
        )
    )

    result = await registry.invoke(
        registry.parse("/review src"),
        CommandContext(commands=registry, api=_CommandApi()),
    )

    assert result == CommandResult()
