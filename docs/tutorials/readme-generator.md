# README Generator

Point the agent at a Python project folder, let it read the source files, analyze the structure, and generate a complete README — with installation instructions, usage examples, API reference, and a project overview.

## What You'll Build

```
project/
  ├── src/mylib/__init__.py
  ├── pyproject.toml
  └── ...
       │
       ▼
  FileReader ─── reads __init__.py, pyproject.toml, key source files
       │  code_context: str
       ▼
  ProjectAnalyzer ── Signature:
                       code_context →
                         project_name, description, key_modules,
                         public_api, dependencies, python_version
       │
       ▼
  ReadmeWriter ────── Signature:
                       analysis →
                         overview, installation, quickstart,
                         api_reference, configuration, contributing
       │
       ▼
  README.md assembled on disk
```

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Read the Project Files

A plain function that collects the most informative files from a project directory:

```python
import os
from pathlib import Path
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import List, Optional


def read_project(root: str, max_chars: int = 15_000) -> str:
    """Read key project files and return them as a single annotated string."""
    root_path = Path(root)
    priority_files = [
        "pyproject.toml", "setup.py", "setup.cfg",
        "README.md", "requirements.txt",
    ]
    source_extensions = {".py"}
    collected = []
    total = 0

    def add_file(path: Path):
        nonlocal total
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if total + len(content) > max_chars:
                content = content[: max_chars - total] + "\n... (truncated)"
            header = f"\n\n{'─' * 40}\n# File: {path.relative_to(root_path)}\n{'─' * 40}\n"
            collected.append(header + content)
            total += len(content)
        except OSError:
            pass

    # Priority files first
    for name in priority_files:
        p = root_path / name
        if p.exists():
            add_file(p)
        if total >= max_chars:
            break

    # Then Python source files (breadth-first, shallowest first)
    for p in sorted(root_path.rglob("*.py"), key=lambda x: len(x.parts)):
        if total >= max_chars:
            break
        if any(skip in str(p) for skip in (".venv", "__pycache__", ".git", "dist")):
            continue
        add_file(p)

    return "".join(collected)
```

---

## Step 2 — Analysis Signature

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class AnalyzeProject(Signature):
    """Analyze a Python project's code and configuration to understand its purpose and structure."""

    code_context: str = InputField(
        desc="Contents of key project files (pyproject.toml, source code, etc.)"
    )

    project_name: str = OutputField(desc="Project name from config or inferred from code")
    description: str = OutputField(
        desc="One paragraph describing what the project does and who it's for"
    )
    key_modules: List[str] = OutputField(
        desc="Main modules/packages and their purpose (one line each)"
    )
    public_api: List[dict] = OutputField(
        desc=(
            "Public functions, classes, and constants. "
            "Each: {'name': str, 'kind': 'function'|'class'|'constant', 'summary': str}"
        )
    )
    dependencies: List[str] = OutputField(
        desc="Required packages extracted from pyproject.toml or requirements.txt"
    )
    python_version: Optional[str] = OutputField(
        desc="Required Python version if specified"
    )
    install_command: str = OutputField(
        desc="Correct pip/uv install command for this project"
    )
```

---

## Step 3 — README Writer Signature

```python
class WriteReadme(Signature):
    """Generate a complete, developer-friendly README for a Python project."""

    project_name: str = InputField(desc="Project name")
    description: str = InputField(desc="Project description")
    key_modules: List[str] = InputField(desc="Module summaries")
    public_api: List[dict] = InputField(desc="Public API entries")
    dependencies: List[str] = InputField(desc="Required packages")
    python_version: Optional[str] = InputField(desc="Required Python version")
    install_command: str = InputField(desc="Install command")

    overview: str = OutputField(
        desc="Project overview section: what it is, why it exists, key features (Markdown)"
    )
    installation: str = OutputField(
        desc="Installation section with prerequisites and install command (Markdown)"
    )
    quickstart: str = OutputField(
        desc="Quickstart section with a minimal working example (Markdown + code block)"
    )
    api_reference: str = OutputField(
        desc="API reference section covering the main classes and functions (Markdown)"
    )
    contributing: str = OutputField(
        desc="Contributing section with development setup and PR guidelines (Markdown)"
    )
```

---

## Step 4 — Pipeline

```python
class ProjectAnalyzer(nn.Agent):
    model = model
    signature = AnalyzeProject
    config = {"verbose": True}


class ReadmeWriter(nn.Agent):
    model = model
    signature = WriteReadme


class ReadmeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = ProjectAnalyzer()
        self.writer   = ReadmeWriter()

    def forward(self, msg):
        # 1. Read project files if not already provided
        if not msg.get("code_context"):
            msg.code_context = read_project(msg.project_root)

        # 2. Analyze the project
        self.analyzer(msg)

        # 3. Write the README sections
        self.writer(msg)

        # 4. Assemble the final README
        msg.readme = self._assemble(msg)
        return msg

    @staticmethod
    def _assemble(msg) -> str:
        badge = (
            f"![Python {msg.python_version}]"
            f"(https://img.shields.io/badge/python-{msg.python_version}-blue)"
            if msg.get("python_version") else ""
        )
        return f"""# {msg.project_name}

{badge}

{msg.overview}

---

{msg.installation}

---

{msg.quickstart}

---

{msg.api_reference}

---

{msg.contributing}
"""
```

---

## Complete Example

```python
import os
from pathlib import Path
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import List, Optional


# ── File Reader ───────────────────────────────────────────────────────────────

def read_project(root: str, max_chars: int = 15_000) -> str:
    """Collect annotated contents of key project files."""
    root_path = Path(root)
    priority_files = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]
    collected = []
    total = 0

    def add_file(path: Path):
        nonlocal total
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if total + len(content) > max_chars:
                content = content[: max_chars - total] + "\n... (truncated)"
            header = f"\n\n# File: {path.relative_to(root_path)}\n"
            collected.append(header + content)
            total += len(content)
        except OSError:
            pass

    for name in priority_files:
        p = root_path / name
        if p.exists():
            add_file(p)
        if total >= max_chars:
            break

    for p in sorted(root_path.rglob("*.py"), key=lambda x: len(x.parts)):
        if total >= max_chars:
            break
        if any(skip in str(p) for skip in (".venv", "__pycache__", ".git", "dist")):
            continue
        add_file(p)

    return "".join(collected)


# ── Signatures ────────────────────────────────────────────────────────────────

class AnalyzeProject(Signature):
    """Analyze a Python project's code and configuration."""

    code_context: str = InputField(desc="Contents of key project files")
    project_name: str = OutputField(desc="Project name")
    description: str = OutputField(desc="One paragraph description")
    key_modules: List[str] = OutputField(desc="Main modules and their purpose")
    public_api: List[dict] = OutputField(
        desc="Public API: [{'name': str, 'kind': str, 'summary': str}, ...]"
    )
    dependencies: List[str] = OutputField(desc="Required packages")
    python_version: Optional[str] = OutputField(desc="Required Python version")
    install_command: str = OutputField(desc="pip/uv install command")


class WriteReadme(Signature):
    """Generate a complete README for a Python project."""

    project_name: str = InputField(desc="Project name")
    description: str = InputField(desc="Project description")
    key_modules: List[str] = InputField(desc="Module summaries")
    public_api: List[dict] = InputField(desc="Public API entries")
    dependencies: List[str] = InputField(desc="Required packages")
    python_version: Optional[str] = InputField(desc="Required Python version")
    install_command: str = InputField(desc="Install command")
    overview: str = OutputField(desc="Overview section (Markdown)")
    installation: str = OutputField(desc="Installation section (Markdown)")
    quickstart: str = OutputField(desc="Quickstart with code example (Markdown)")
    api_reference: str = OutputField(desc="API reference (Markdown)")
    contributing: str = OutputField(desc="Contributing guide (Markdown)")


# ── Agents ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class ProjectAnalyzer(nn.Agent):
    model = model
    signature = AnalyzeProject
    config = {"verbose": True}


class ReadmeWriter(nn.Agent):
    model = model
    signature = WriteReadme


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ReadmeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = ProjectAnalyzer()
        self.writer   = ReadmeWriter()

    def forward(self, msg):
        if not msg.get("code_context"):
            msg.code_context = read_project(msg.project_root)
        self.analyzer(msg)
        self.writer(msg)
        msg.readme = self._assemble(msg)
        return msg

    @staticmethod
    def _assemble(msg) -> str:
        badge = (
            f"![Python {msg.python_version}]"
            f"(https://img.shields.io/badge/python-{msg.python_version}-blue)"
            if msg.get("python_version") else ""
        )
        return f"""# {msg.project_name}

{badge}

{msg.overview}

---

{msg.installation}

---

{msg.quickstart}

---

{msg.api_reference}

---

{msg.contributing}
"""


# ── Run ───────────────────────────────────────────────────────────────────────

generator = ReadmeGenerator()

# Point at any Python project
msg = Message(project_root=".")
generator(msg)

# Print to console
print(msg.readme)

# Save to file
output_path = Path(msg.project_root) / "README_generated.md"
output_path.write_text(msg.readme, encoding="utf-8")
print(f"\nSaved to {output_path}")
```

---

## Async Version

```python
import asyncio


class ReadmeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = ProjectAnalyzer()
        self.writer   = ReadmeWriter()

    async def aforward(self, msg):
        if not msg.get("code_context"):
            msg.code_context = read_project(msg.project_root)
        await self.analyzer.acall(msg)
        await self.writer.acall(msg)
        msg.readme = ReadmeGenerator._assemble(msg)
        return msg


async def main():
    generator = ReadmeGenerator()
    msg = Message(project_root="/path/to/my/project")
    await generator.acall(msg)
    print(msg.readme[:2000])

asyncio.run(main())
```

---

## Batch Generation

Generate READMEs for multiple projects in parallel:

```python
import asyncio
import msgflux.nn.functional as F
from pathlib import Path


async def batch_generate(project_roots: list[str]):
    generator = ReadmeGenerator()
    messages = [Message(project_root=r) for r in project_roots]

    results = await F.ascatter_gather(
        [generator.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )

    for root, msg in zip(project_roots, results):
        out = Path(root) / "README_generated.md"
        out.write_text(msg.readme, encoding="utf-8")
        print(f"✓ {root} → {out}")


asyncio.run(batch_generate(["./proj_a", "./proj_b", "./proj_c"]))
```
