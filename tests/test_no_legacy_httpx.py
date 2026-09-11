"""Prevent legacy HTTPX v1 from returning to the msgFlux runtime."""

import ast
import re
from pathlib import Path


def test_source_does_not_import_legacy_httpx():
    source_root = Path(__file__).parents[1] / "src" / "msgflux"
    legacy_imports = []

    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                legacy_imports.extend(
                    f"{path.relative_to(source_root)}:{node.lineno}"
                    for alias in node.names
                    if alias.name == "httpx"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "httpx":
                legacy_imports.append(f"{path.relative_to(source_root)}:{node.lineno}")

    assert legacy_imports == []


def test_project_does_not_declare_legacy_httpx_dependency():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert re.search(r'^\s*"httpx(?:\[|[<>=!~]|"|\s)', pyproject, re.MULTILINE) is None
