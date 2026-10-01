"""Opt-in SQLite registry for stable workspace resource identities."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from threading import RLock
from uuid import uuid4

import msgspec

from msgflux.runtime.workspace.contracts import WorkspaceIdentity


class WorkspaceRegistryRecord(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Versioned immutable record of a resource and its validated local root."""

    schema_version: int
    identity: WorkspaceIdentity
    root: str
    root_device: int
    root_inode: int
    revision: int

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or not isinstance(self.identity, WorkspaceIdentity)
            or type(self.revision) is not int
            or self.revision < 1
        ):
            raise ValueError("Unsupported or invalid workspace registry record")
        if (
            not isinstance(self.root, str)
            or not os.path.isabs(self.root)
            or "\0" in self.root
        ):
            raise ValueError("Workspace registry root must be an absolute path")
        if (
            type(self.root_device) is not int
            or type(self.root_inode) is not int
            or self.root_device < 0
            or self.root_inode < 0
        ):
            raise ValueError("Workspace root fingerprint values must be non-negative")


class SQLiteWorkspaceRegistry:
    """Persist immutable workspace resource bindings outside workspace roots.

    Registration is transactional. Existing entries are only returned when every
    resource/configuration fingerprint matches; this class never adopts resources.
    """

    _SCHEMA = 1

    def __init__(self, path: str | Path):
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("Registry path must be non-empty")
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, self._SCHEMA):
                raise ValueError(f"Unsupported workspace registry schema: {version}")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS workspace_registry (
                    workspace_id TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    root TEXT NOT NULL,
                    root_device INTEGER NOT NULL,
                    root_inode INTEGER NOT NULL,
                    revision INTEGER NOT NULL
                )"""
            )
            self._conn.execute(f"PRAGMA user_version={self._SCHEMA}")
            self._conn.commit()
        except BaseException:
            self._conn.close()
            raise

    @staticmethod
    def _record(row) -> WorkspaceRegistryRecord:
        return WorkspaceRegistryRecord(
            schema_version=1,
            identity=WorkspaceIdentity(
                backend=row[0],
                resource_id=row[1],
                generation=row[2],
                config_revision=row[3],
            ),
            root=row[4],
            root_device=row[5],
            root_inode=row[6],
            revision=row[7],
        )

    @staticmethod
    def _select(workspace_id: str):
        return (
            """SELECT backend, resource_id, generation, config_revision,
        root, root_device, root_inode, revision FROM workspace_registry
        WHERE workspace_id=?""",
            (workspace_id,),
        )

    def register_or_verify(
        self,
        workspace_id: str,
        *,
        backend: str,
        resource_id: str,
        config_revision: str,
        root: str,
        root_device: int,
        root_inode: int,
    ) -> WorkspaceIdentity:
        """Register once or require an exact match with the prior registration."""
        self._validate_workspace_id(workspace_id, resource_id)
        candidate = WorkspaceRegistryRecord(
            schema_version=self._SCHEMA,
            identity=WorkspaceIdentity(
                backend=backend,
                resource_id=resource_id,
                generation=uuid4().hex,
                config_revision=config_revision,
            ),
            root=root,
            root_device=root_device,
            root_inode=root_inode,
            revision=1,
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                query, params = self._select(workspace_id)
                row = self._conn.execute(query, params).fetchone()
                if row is None:
                    record = candidate
                    self._conn.execute(
                        """INSERT INTO workspace_registry
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            workspace_id,
                            record.identity.backend,
                            record.identity.resource_id,
                            record.identity.generation,
                            record.identity.config_revision,
                            record.root,
                            record.root_device,
                            record.root_inode,
                            record.revision,
                        ),
                    )
                    self._conn.commit()
                    return record.identity
                record = self._record(row)
                expected = (
                    backend,
                    resource_id,
                    config_revision,
                    root,
                    root_device,
                    root_inode,
                )
                actual = (
                    record.identity.backend,
                    record.identity.resource_id,
                    record.identity.config_revision,
                    record.root,
                    record.root_device,
                    record.root_inode,
                )
                if actual != expected:
                    raise PermissionError(
                        "Workspace registry resource/configuration mismatch"
                    )
                self._conn.commit()
                return record.identity
            except BaseException:
                self._conn.rollback()
                raise

    def get(self, workspace_id: str) -> WorkspaceIdentity:
        """Return a registered identity, without registering missing resources."""
        return self.get_record(workspace_id).identity

    @staticmethod
    def _validate_workspace_id(workspace_id, resource_id):
        if (
            not isinstance(workspace_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", workspace_id)
            or resource_id != workspace_id
        ):
            raise ValueError("workspace_id and resource_id must match and be safe")

    def replace(
        self,
        workspace_id: str,
        *,
        expected_revision: int,
        backend: str,
        resource_id: str,
        config_revision: str,
        root: str,
        root_device: int,
        root_inode: int,
    ) -> WorkspaceRegistryRecord:
        """Replace registration using revision compare-and-swap, without revocation.

        The host must drain and close all old bindings before replacing a resource.
        This changes metadata; it does not stop commands or revoke live handles.
        """
        self._validate_workspace_id(workspace_id, resource_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        replacement = WorkspaceRegistryRecord(
            schema_version=self._SCHEMA,
            identity=WorkspaceIdentity(
                backend=backend,
                resource_id=resource_id,
                generation=uuid4().hex,
                config_revision=config_revision,
            ),
            root=root,
            root_device=root_device,
            root_inode=root_inode,
            revision=expected_revision + 1,
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                query, params = self._select(workspace_id)
                row = self._conn.execute(query, params).fetchone()
                if row is None:
                    raise FileNotFoundError("Workspace resource is not registered")
                previous = self._record(row)
                if previous.revision != expected_revision:
                    raise PermissionError("Workspace registry revision changed")
                cursor = self._conn.execute(
                    """UPDATE workspace_registry SET backend=?, resource_id=?,
                    generation=?, config_revision=?, root=?, root_device=?,
                    root_inode=?, revision=? WHERE workspace_id=? AND revision=?""",
                    (
                        replacement.identity.backend,
                        replacement.identity.resource_id,
                        replacement.identity.generation,
                        replacement.identity.config_revision,
                        replacement.root,
                        replacement.root_device,
                        replacement.root_inode,
                        replacement.revision,
                        workspace_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise PermissionError("Workspace registry revision changed")
                self._conn.commit()
                return replacement
            except BaseException:
                self._conn.rollback()
                raise

    def get_record(self, workspace_id: str) -> WorkspaceRegistryRecord:
        """Return the immutable versioned record for a registered resource."""
        with self._lock:
            query, params = self._select(workspace_id)
            row = self._conn.execute(query, params).fetchone()
        if row is None:
            raise FileNotFoundError("Workspace resource is not registered")
        return self._record(row)

    def verify(self, workspace_id: str, **expected) -> WorkspaceIdentity:
        """Require an existing exact resource fingerprint without adopting it."""
        record = self.get_record(workspace_id)
        wanted = tuple(
            expected[key]
            for key in (
                "backend",
                "resource_id",
                "config_revision",
                "root",
                "root_device",
                "root_inode",
            )
        )
        actual = (
            record.identity.backend,
            record.identity.resource_id,
            record.identity.config_revision,
            record.root,
            record.root_device,
            record.root_inode,
        )
        if actual != wanted:
            raise PermissionError("Workspace registry resource/configuration mismatch")
        return record.identity

    def close(self) -> None:
        with self._lock:
            self._conn.close()
