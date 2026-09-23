"""Logical identities for store instances without durable provider metadata."""

import os
from hashlib import blake2s
from uuid import uuid4

_PROCESS_NONCE = uuid4().hex


def process_routing_id(store: object) -> str:
    """Return a process-bound identity; durable providers must override it."""
    identity = f"{_PROCESS_NONCE}:{os.getpid()}:{id(store)}".encode()
    return f"process:{blake2s(identity, digest_size=16).hexdigest()}"
