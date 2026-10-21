"""Failures to verify a trusted workspace resource."""

from typing import Literal

WorkspaceIdentityReason = Literal[
    "backend_changed",
    "resource_changed",
    "configuration_changed",
    "directory_changed",
    "directory_replaced",
    "root_replaced",
    "registry_identity_changed",
]


class WorkspaceIdentityError(PermissionError):
    """A known identity failure with a fixed public diagnostic.

    Chain underlying exceptions with ``raise ... from error`` to retain private
    details without including them in the message exposed by a service.
    """

    _MESSAGES = {
        "backend_changed": "Workspace backend changed",
        "resource_changed": "Workspace resource ID changed",
        "configuration_changed": "Workspace backend configuration changed",
        "directory_changed": "Workspace directory changed",
        "directory_replaced": "Workspace directory was replaced",
        "root_replaced": "Workspace root was replaced",
        "registry_identity_changed": "Workspace registry identity changed",
    }

    def __init__(self, reason: WorkspaceIdentityReason) -> None:
        if reason not in self._MESSAGES:
            raise ValueError("Unknown workspace identity failure reason")
        self.reason = reason
        super().__init__(self._MESSAGES[reason])
