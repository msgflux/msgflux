# Code Generation and Debugging Agent

Build an agent that writes code for a task, executes it, reads the output or error, and self-corrects — using the ReAct pattern so every tool call and reasoning step is visible.

## What You'll Build

```
Task description
       │
       ▼
  CodeAgent (ReAct)
       │
       ├──► write_code(filename, code)   → saves to disk
       ├──► run_code(filename)           → stdout / stderr
       ├──► read_code(filename)          → current file contents
       └──► check_syntax(code)          → syntax errors before running
       │
  loop: reason → act → observe → reason ...
       │
       ▼
  Working code on disk + explanation on msg
```

`config = {"verbose": True}` prints every reasoning step and tool call so you can follow exactly what the agent is doing.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Sandboxed Execution Tools

The tools run code in a temporary directory. Each one has a clear docstring — the agent reads these to decide which tool to call.

```python
import subprocess
import tempfile
import ast
import os
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from msgflux.generation.reasoning import ReAct

# Temporary sandbox directory
SANDBOX = tempfile.mkdtemp(prefix="msgflux_sandbox_")


def write_code(filename: str, code: str) -> str:
    """Write Python code to a file in the sandbox.

    Args:
        filename: File name, e.g. 'solution.py'
        code:     Complete Python source code to write
    Returns:
        Confirmation message with the file path
    """
    path = os.path.join(SANDBOX, filename)
    with open(path, "w") as f:
        f.write(code)
    return f"Written to {path}"


def run_code(filename: str) -> str:
    """Execute a Python file in the sandbox and return its output or error.

    Args:
        filename: File to run (must have been written with write_code first)
    Returns:
        stdout on success, or stderr on error (prefixed with 'ERROR:')
    """
    path = os.path.join(SANDBOX, filename)
    if not os.path.exists(path):
        return f"ERROR: File '{filename}' not found. Use write_code first."

    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        return result.stdout.strip() or "(no output)"
    return f"ERROR:\n{result.stderr.strip()}"


def read_code(filename: str) -> str:
    """Read the current contents of a file in the sandbox.

    Args:
        filename: File to read
    Returns:
        Current source code
    """
    path = os.path.join(SANDBOX, filename)
    if not os.path.exists(path):
        return f"ERROR: File '{filename}' not found."
    with open(path) as f:
        return f.read()


def check_syntax(code: str) -> str:
    """Check Python code for syntax errors without executing it.

    Args:
        code: Python source code to validate
    Returns:
        'OK' if syntax is valid, or a description of the syntax error
    """
    try:
        ast.parse(code)
        return "OK"
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"
```

---

## Step 2 — Agent with ReAct

`generation_schema = ReAct` switches the agent to the Reasoning + Acting loop: it generates a thought, picks a tool, observes the result, then repeats until it decides it's done.

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CodeAgent(nn.Agent):
    """Writes, runs, and debugs Python code to solve programming tasks."""
    model = model

    system_message = """
    You are a Python expert. Given a programming task:
    1. Check syntax before running (use check_syntax).
    2. Write the code (use write_code).
    3. Run it (use run_code).
    4. If there are errors, read the current code (use read_code), fix it, and run again.
    5. Repeat until the output matches the expected result.
    6. Report what the code does and its final output.
    """

    tools            = [write_code, run_code, read_code, check_syntax]
    generation_schema = ReAct
    config           = {"verbose": True}
```

---

## Step 3 — Running the Agent

```python
agent = CodeAgent()

result = agent(
    "Write a Python function that finds all prime numbers up to N using "
    "the Sieve of Eratosthenes, then call it with N=50 and print the result."
)
print("\nFinal answer:", result)
```

With `verbose=True` you see every step:

```
[code_agent][tool_call] check_syntax: def sieve(n): ...
[code_agent][tool_responses] OK

[code_agent][tool_call] write_code: filename=solution.py, code=...
[code_agent][tool_responses] Written to /tmp/msgflux_sandbox_xxx/solution.py

[code_agent][tool_call] run_code: filename=solution.py
[code_agent][tool_responses] [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

Final answer: The sieve correctly returns all 15 primes up to 50.
```

---

## Complete Example

```python
import subprocess
import tempfile
import ast
import os
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message
from msgflux.generation.reasoning import ReAct


# ── Sandbox ───────────────────────────────────────────────────────────────────

SANDBOX = tempfile.mkdtemp(prefix="msgflux_sandbox_")


def write_code(filename: str, code: str) -> str:
    """Write Python code to a file in the sandbox."""
    path = os.path.join(SANDBOX, filename)
    with open(path, "w") as f:
        f.write(code)
    return f"Written to {path}"


def run_code(filename: str) -> str:
    """Execute a Python file and return stdout or stderr."""
    path = os.path.join(SANDBOX, filename)
    if not os.path.exists(path):
        return f"ERROR: File '{filename}' not found."
    result = subprocess.run(
        ["python", path], capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        return result.stdout.strip() or "(no output)"
    return f"ERROR:\n{result.stderr.strip()}"


def read_code(filename: str) -> str:
    """Read the current contents of a sandbox file."""
    path = os.path.join(SANDBOX, filename)
    if not os.path.exists(path):
        return f"ERROR: File '{filename}' not found."
    with open(path) as f:
        return f.read()


def check_syntax(code: str) -> str:
    """Check Python syntax without executing. Returns 'OK' or error description."""
    try:
        ast.parse(code)
        return "OK"
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


# ── Agent ─────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CodeAgent(nn.Agent):
    """Writes, runs, and debugs Python code to solve programming tasks."""
    model = model

    system_message = """
    You are a Python expert. Given a programming task:
    1. Check syntax before running (use check_syntax).
    2. Write the code (use write_code).
    3. Run it (use run_code).
    4. If there are errors, read the code, fix it, and run again.
    5. Repeat until correct.
    6. Report what the code does and its final output.
    """

    tools             = [write_code, run_code, read_code, check_syntax]
    generation_schema = ReAct
    config            = {"verbose": True}


# ── Tasks ─────────────────────────────────────────────────────────────────────

agent = CodeAgent()

tasks = [
    "Write a function that reverses a string without using slicing or reversed(). "
    "Test it with 'Hello, World!' and print the result.",

    "Implement a binary search function. Test it on a sorted list "
    "[1, 3, 5, 7, 9, 11, 13] searching for 7. Print the index found.",

    "Write a recursive Fibonacci function with memoization. "
    "Print fib(10) and fib(20).",
]

for task in tasks:
    print(f"\n{'=' * 60}")
    print(f"Task: {task[:80]}...")
    result = agent(task)
    print(f"\nResult: {result}")
```

---

## Async Version

```python
import asyncio


async def main():
    agent = CodeAgent()

    result = await agent.acall(
        "Write a class 'Stack' with push, pop, peek, and is_empty methods. "
        "Demo it by pushing 1, 2, 3, popping once, and printing the remaining stack."
    )
    print(result)


asyncio.run(main())
```

---

## Why ReAct?

| Single-shot generation | ReAct agent |
|---|---|
| Writes code once, can't verify it works | Writes → runs → reads errors → fixes |
| Syntax errors go undetected | `check_syntax` catches issues before execution |
| No recovery from runtime errors | Reads the error, adjusts, retries |
| Black box: you see only the final answer | `verbose=True` shows every thought and tool call |

The key insight: execution feedback closes the loop. The agent isn't just generating — it's observing the real output of its own code and correcting course.
