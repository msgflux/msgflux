"""Isolation capability negotiation, not an operating-system sandbox."""

from dataclasses import dataclass, field

_MECHANISMS = frozenset({"filesystem", "network", "process", "resource_limits"})


def _mechanisms(values):
    if isinstance(values, (str, bytes)):
        raise TypeError("Isolation mechanisms must be a collection")
    result = frozenset(values)
    if not result <= _MECHANISMS:
        raise ValueError("Unknown isolation mechanism")
    return result


@dataclass(frozen=True)
class SandboxRequirements:
    mechanisms: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, "mechanisms", _mechanisms(self.mechanisms))


@dataclass(frozen=True)
class SandboxCapabilities:
    """Trusted backend declarations; a successful check is not enforcement."""

    mechanisms: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, "mechanisms", _mechanisms(self.mechanisms))

    def require(self, requirements: SandboxRequirements) -> None:
        if not isinstance(requirements, SandboxRequirements):
            raise TypeError("Expected SandboxRequirements")
        missing = requirements.mechanisms - self.mechanisms
        if missing:
            raise PermissionError(
                f"Unsupported isolation mechanisms: {', '.join(sorted(missing))}"
            )
