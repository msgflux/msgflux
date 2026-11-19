"""Registry text survives snapshots and follows the lifetime of its registration."""

import msgspec
import pytest

from msgflux.nn.modules.tool.registry import ToolRegistry
from tests.nn.modules.test_tool_registry import make_definition
from tests.optim.test_tool_target import _agent, _lookup_tool


def test_registry_isolates_replacements_and_discards_removed_state():
    lookup = _lookup_tool()
    left = _agent(lookup).tool_library
    right = _agent(lookup).tool_library
    description = left.description_parameter("lookup")
    guidance = left.usage_guidance_parameter("lookup")
    guidance.copy_to_data("Use the invoice number.")
    left.registry.replace(
        msgspec.structs.replace(
            left.registry.get("lookup"), description="Get a receipt."
        )
    )
    assert left.description_parameter("lookup") is description
    assert description.data == "Get a receipt."
    assert right.description_parameter("lookup").data == lookup.__doc__
    assert right.usage_guidance_parameter("lookup").data == ""
    snapshot = left.registry.state_dict()
    left.remove("lookup")
    assert list(left.named_tool_text_parameters()) == []
    assert left.registry.state_dict() == {}
    left.add(lookup)
    assert left.description_parameter("lookup") is not description
    assert left.description_parameter("lookup").data == lookup.__doc__
    left.registry.load_state_dict(snapshot)
    assert left.description_parameter("lookup").data == "Get a receipt."
    assert left.usage_guidance_parameter("lookup").data == "Use the invoice number."
    left.clear()
    assert left.registry.state_dict() == {}


def test_distinct_qualified_names_survive_module_state_roundtrip():
    definitions = [
        msgspec.structs.replace(make_definition(), name=name)
        for name in ("ns.lookup", "ns%2Elookup")
    ]
    registry = ToolRegistry("tools", definitions)
    for name, field, param in registry.named_text_parameters():
        param.copy_to_data(f"{name}/{field}")
    restored = ToolRegistry("tools", definitions)
    restored.load_state_dict(registry.state_dict())
    for definition in restored.definitions():
        assert definition.description == f"{definition.name}/description"
        assert definition.usage_guidance == f"{definition.name}/usage_guidance"


def test_guidance_keeps_local_edits_when_defaults_are_enabled():
    lookup = _lookup_tool()
    lookup.default_usage_guidance = "Default guidance."
    local = _agent(lookup).tool_library
    plain = _agent(lookup).tool_library
    local.usage_guidance_parameter("lookup").copy_to_data("Tuned guidance.")
    local.apply_default_usage_guidance()
    plain.apply_default_usage_guidance()
    assert local.get_tool_definition("lookup").usage_guidance == "Tuned guidance."
    assert plain.get_tool_definition("lookup").usage_guidance == "Default guidance."
    assert not plain.usage_guidance_parameter("lookup").requires_grad
    with pytest.raises(ValueError):
        plain.usage_guidance_parameter("missing")
