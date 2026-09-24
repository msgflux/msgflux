"""Capture the host screen and publish it to the agent conversation."""

import asyncio
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from msgflux.data.types import Image
from msgflux.tools.config import tool_config
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.types import Hidden


@tool_config(
    runtime_inputs=["handle"],
    required_permissions=["desktop.capture"],
    retry=False,
)
class ScreenshotTool:
    """Capture the complete desktop, attach the image, and return its file path.

    Requires the optional ``screenshot`` dependency and a supported graphical
    session. Linux Wayland sessions use the XDG Desktop Portal.
    """

    name = "screenshot"
    display_name = "Screenshot"
    annotations = {"return": str}

    def __init__(self, *, max_image_bytes: int = 10_000_000):
        if type(max_image_bytes) is not int or max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be a positive integer")
        self.max_image_bytes = max_image_bytes

    def __call__(self, *, handle: Hidden[ToolLibraryHandle] = None) -> str:
        from msgflux.nn.functional import wait_for  # noqa: PLC0415

        return wait_for(self.acall, handle=handle)

    async def acall(self, *, handle: Hidden[ToolLibraryHandle] = None) -> str:
        if handle is None:
            raise RuntimeError("Screenshot publication requires an agent inbox")

        directory = None
        try:
            if self._uses_wayland():
                path = await self._capture_wayland()
            else:
                directory = Path(tempfile.mkdtemp(prefix="msgflux-screenshot-"))
                path = directory / "screenshot.png"
                await asyncio.to_thread(self._capture_mss, path)

            if path.stat().st_size > self.max_image_bytes:
                raise ValueError("Screenshot exceeds the configured byte limit")

            image = await Image(str(path)).acall()
            notification = handle.get_notification().message(
                [image],
                description="Desktop screenshot captured by this tool call.",
            )
            if notification is None:
                raise RuntimeError("Screenshot publication requires an agent inbox")
            return str(path)
        except Exception:
            if directory is not None:
                for child in directory.iterdir():
                    child.unlink(missing_ok=True)
                directory.rmdir()
            raise

    @staticmethod
    def _uses_wayland() -> bool:
        return os.name == "posix" and (
            os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
            or bool(os.environ.get("WAYLAND_DISPLAY"))
        )

    @staticmethod
    def _capture_mss(path: Path) -> None:
        try:
            from mss import MSS  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Install the screenshot extra with `pip install 'msgflux[screenshot]'`."
            ) from exc

        try:
            with MSS() as capture:
                capture.shot(mon=0, output=str(path))
        except Exception as exc:
            raise RuntimeError(f"Could not capture the desktop: {exc}") from exc

    async def _capture_wayland(self) -> Path:  # noqa: C901
        """Capture through XDG Desktop Portal and return its local file path."""
        try:
            from dbus_next import (  # noqa: PLC0415
                BusType,
                Message,
                MessageType,
                Variant,
            )
            from dbus_next.aio import MessageBus  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Install the screenshot extra with `pip install 'msgflux[screenshot]'`."
            ) from exc

        bus = await MessageBus(bus_type=BusType.SESSION).connect()
        response_future = asyncio.get_running_loop().create_future()
        responses = {}
        request_path = None

        def handle_message(message):
            if (
                message.message_type == MessageType.SIGNAL
                and message.interface == "org.freedesktop.portal.Request"
                and message.member == "Response"
            ):
                responses[message.path] = message.body
                if message.path == request_path and not response_future.done():
                    response_future.set_result(message.body)
            return False

        bus.add_message_handler(handle_message)
        try:
            match_reply = await bus.call(
                Message(
                    destination="org.freedesktop.DBus",
                    path="/org/freedesktop/DBus",
                    interface="org.freedesktop.DBus",
                    member="AddMatch",
                    signature="s",
                    body=[
                        "type='signal',interface='org.freedesktop.portal.Request',"
                        "member='Response'"
                    ],
                )
            )
            if match_reply.message_type == MessageType.ERROR:
                raise RuntimeError(match_reply.body[0])

            reply = await bus.call(
                Message(
                    destination="org.freedesktop.portal.Desktop",
                    path="/org/freedesktop/portal/desktop",
                    interface="org.freedesktop.portal.Screenshot",
                    member="Screenshot",
                    signature="sa{sv}",
                    body=["", {"interactive": Variant("b", True)}],
                )
            )
            if reply.message_type == MessageType.ERROR:
                raise RuntimeError(reply.body[0])
            request_path = reply.body[0]
            if request_path in responses:
                response_future.set_result(responses[request_path])

            try:
                response_code, results = await asyncio.wait_for(
                    response_future, timeout=60
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "Desktop screenshot approval timed out or was not completed."
                ) from exc
            if response_code != 0:
                raise RuntimeError(
                    "Desktop screenshot was cancelled or denied by the desktop."
                )

            uri_value = results.get("uri")
            uri = getattr(uri_value, "value", None)
            parsed = urlparse(uri) if isinstance(uri, str) else None
            if (
                parsed is None
                or parsed.scheme != "file"
                or parsed.netloc not in ("", "localhost")
            ):
                raise RuntimeError(
                    "Desktop portal did not return a local screenshot path."
                )
            path = Path(unquote(parsed.path))
            if not path.is_file():
                raise RuntimeError("Desktop portal screenshot file is unavailable.")
            return path
        finally:
            bus.remove_message_handler(handle_message)
            bus.disconnect()


__all__ = ["ScreenshotTool"]
