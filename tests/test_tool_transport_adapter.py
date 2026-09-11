import inspect

import pytest

from msgflux.models.tool_adapters import ToolTransportAdapter
from msgflux.models.tool_adapters.openai_patch import OpenAIApplyPatchAdapter
from msgflux.models.tool_adapters.openai_shell import OpenAIShellAdapter


def test_adapter_requires_all_protocol_methods():
    methods = {
        "declaration",
        "supports",
        "validate_metadata",
        "decode",
        "render",
        "project_history",
        "interrupted",
    }
    assert ToolTransportAdapter.__abstractmethods__ == methods
    with pytest.raises(TypeError):
        ToolTransportAdapter()
    for missing in methods:
        implementation = type(
            "IncompleteAdapter",
            (ToolTransportAdapter,),
            {name: lambda self, *args, **kwargs: None for name in methods - {missing}},
        )
        with pytest.raises(TypeError, match=missing):
            implementation()


@pytest.mark.parametrize("adapter_type", [OpenAIShellAdapter, OpenAIApplyPatchAdapter])
def test_native_adapters_implement_stateless_contract(adapter_type):
    adapter = adapter_type()
    assert isinstance(adapter, ToolTransportAdapter)
    assert not inspect.isabstract(adapter_type)
    assert vars(adapter) == {}
    for name in ("provider", "api_mode", "codec", "kind", "item_type", "output_type"):
        assert isinstance(getattr(adapter, name), str) and getattr(adapter, name)
    assert type(adapter.version) is int and adapter.version > 0
