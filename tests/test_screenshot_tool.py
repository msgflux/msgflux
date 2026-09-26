import base64
from pathlib import Path

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import ExecutionScope, PermissionSet, execution_context
from msgflux.tools.builtin import ScreenshotTool


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "AAAAAElFTkSuQmCC"
)


@pytest.mark.asyncio
async def test_screenshot_captures_all_monitors_and_publishes_image(monkeypatch):
    captured = []

    def capture(path):
        captured.append(path)
        Path(path).write_bytes(PNG)

    monkeypatch.setattr(ScreenshotTool, "_capture_mss", staticmethod(capture))
    monkeypatch.setattr(ScreenshotTool, "_uses_wayland", staticmethod(lambda: False))
    library = ToolLibrary("desktop", [ScreenshotTool()])
    inbox = library.get_agent_inbox()
    scope = ExecutionScope(
        namespace="desktop",
        permissions=PermissionSet(grants={"desktop.capture"}),
    )

    with execution_context(scope=scope, agent_inbox=inbox):
        path = await library.arun("screenshot", {})

    assert len(captured) == 1
    assert Path(path).is_file()
    assert Path(path).read_bytes() == PNG
    notification = inbox.peek()[0]
    assert notification.metadata["tool"] == "screenshot"
    image_url = notification.metadata["content"][0]["image_url"]["url"]
    assert base64.b64decode(image_url.split(",", 1)[1]) == PNG
    assert (
        "desktop.capture"
        in library.get_tool_definition("screenshot").required_permissions
    )


@pytest.mark.asyncio
async def test_screenshot_requires_desktop_capture_permission(monkeypatch):
    def capture(path):
        pytest.fail(f"capture must not run without permission: {path}")

    monkeypatch.setattr(ScreenshotTool, "_capture_mss", staticmethod(capture))
    library = ToolLibrary("desktop", [ScreenshotTool()])
    scope = ExecutionScope(namespace="desktop", permissions=PermissionSet())

    with execution_context(scope=scope, agent_inbox=library.get_agent_inbox()):
        with pytest.raises(RuntimeError, match=r"desktop\.capture"):
            await library.arun("screenshot", {})
