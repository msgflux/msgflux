import asyncio
import functools
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskError

__all__ = ["Inline"]


class Inline:
    """Parses and executes a mini domain-specific language (DSL) for
    defining workflow pipelines over modules.

    Supported Node Types:

    1. Module Node:
        Syntax:
            `"module_name"`
        Description:
            A single module that processes the message.

    !!! example

            `"a"`

    2. Parallel Node:
        Syntax:
            `"[module1, module2, ...]"`
        Description:
            Executes multiple modules in parallel. The same input message is
            passed to all modules and all mutations are applied in place.

    !!! example

            `"[feat_a, feat_b]"`

    3. Conditional Node:
        Syntax:
            `"{condition1 ? branch1, condition2 ? branch2, default}"`
        Description:
            Multi-branch conditional. Evaluates conditions in order and
            executes the first matching branch. The last item without `?`
            is the default (fallback) branch. Use `_` as an explicit
            wildcard default.

            Conditions follow the format `key_path operator value`,
            e.g., `output.agent == "xpto"`.
            Supports logic operators:
                * AND (&): condition1 & condition2
                * OR (||): condition1 || condition2
                * NOT (!): !(condition)
            None verification:
                * is None: user.name is None
                * is not None: user.name is not None

        !!! example

            `"{score > 0.9 ? premium, score > 0.5 ? standard, fallback}"`
            `"{cond ? a, b}"` — equivalent to if/else (backwards compatible).
            `"{cond1 ? a, _ ? b}"` — explicit wildcard default.

    4. While Loop Node:
        Syntax:
            `"@{condition}: actions;"`
        Description:
            Executes a block of actions repeatedly while the condition is true.
            The condition follows the same format as conditional nodes.
            Actions can be any valid DSL expression (sequential, parallel,
            conditional, nested while loops).

    !!! example

            `"@{counter < 10}: increment;"`
            `"@{active}: [monitor, logger] -> report;"`

    5. Arrow Separator (`->`):
        Description:
            Defines the sequence of operations in the pipeline.

    !!! example

            `"prep -> [feat_a, feat_b] -> combine"`
    """

    def __init__(
        self,
        expression: str,
        modules: Mapping[str, Callable],
        *,
        max_iterations: int = 1000,
    ):
        if not isinstance(expression, str):
            raise TypeError("`expression` must be a str")
        if not isinstance(modules, Mapping):
            raise TypeError("`modules` must be a `Mapping`")

        self.expression = expression
        self.modules = modules
        self.max_iterations = max_iterations  # Safety limit for while loops
        self.patterns = {
            "arrow": r"\s*->\s*",
            "parallel": r"\[(.*?)\]",
            "conditional": r"\{([^{}]+)\}",
            "while_loop": r"@\{(.*?)\}:\s*((?:[^;]|(?:->|\[.*?\]|\{.*?\}|@\{.*?\}:\s*.*?;))+);",  # noqa: E501
            "identifier": r"[a-zA-Z_][a-zA-Z0-9_]*",
            "comparison": r"([a-zA-Z0-9_.]+)\s*(==|!=|<=|>=|<|>|is not|is)\s*(.*)",
        }
        self.steps = self.parse(expression)

    @staticmethod
    def _validate_message(message: dotdict) -> None:
        if not isinstance(message, dotdict):
            raise TypeError("`message` must be an instance of `msgflux.dotdict`")

    def parse(self, expression: str) -> List[Dict[str, Any]]:
        """Parse DSL expression into a list of execution steps."""
        steps: List[Dict[str, Any]] = []
        remaining = expression.strip()

        while remaining:
            while_match = re.match(self.patterns["while_loop"], remaining)
            if while_match:
                condition = while_match.group(1).strip()
                actions = while_match.group(2).strip()
                steps.append(
                    {
                        "type": "while",
                        "condition": condition,
                        "actions": self.parse(actions),
                    }
                )

                remaining = remaining[while_match.end() :].strip()
                if remaining.startswith("->"):
                    remaining = remaining[2:].strip()
                continue

            arrow_split = re.split(self.patterns["arrow"], remaining, maxsplit=1)
            part = arrow_split[0].strip()
            remaining = arrow_split[1].strip() if len(arrow_split) > 1 else ""

            if not part:
                continue

            conditional_match = re.match(self.patterns["conditional"], part)
            if conditional_match:
                content = conditional_match.group(1)
                steps.append(self._parse_conditional_content(content))
                continue

            parallel_match = re.match(self.patterns["parallel"], part)
            if parallel_match:
                modules = [m.strip() for m in parallel_match.group(1).split(",")]
                steps.append({"type": "parallel", "modules": modules})
                continue

            if re.match(self.patterns["identifier"], part):
                steps.append({"type": "module", "module": part})
            else:
                raise ValueError(f"Invalid DSL syntax or unknown module: {part}")

        return steps

    def _parse_conditional_content(self, content: str) -> Dict[str, Any]:
        """Parse multi-branch conditional content into a step dict."""
        parts = [p.strip() for p in content.split(",")]
        branches: List[Dict[str, str]] = []
        default: List[str] = []

        for part in parts:
            if "?" in part:
                q_idx = part.index("?")
                condition = part[:q_idx].strip()
                module = part[q_idx + 1 :].strip()
                if condition == "_":
                    default = [module]
                else:
                    branches.append({"condition": condition, "module": module})
            else:
                default.append(part)

        return {"type": "conditional", "branches": branches, "default": default}

    def _tokenize_condition(self, condition: str) -> List[str]:
        condition = condition.strip()
        token_pattern = r"(\|\||&|!|\(|\)|[^&|!()]+)"  # noqa: S105
        tokens = re.findall(token_pattern, condition)
        return [token.strip() for token in tokens if token.strip()]

    def _parse_logical_expression(
        self, tokens: List[str], index: Optional[int] = 0
    ) -> Tuple:
        result, index = self._parse_or_expression(tokens, index)
        return result, index

    def _parse_or_expression(self, tokens: List[str], index: int) -> Tuple:
        left, index = self._parse_and_expression(tokens, index)

        while index < len(tokens) and tokens[index] == "||":
            index += 1
            right, index = self._parse_and_expression(tokens, index)
            left = ("OR", left, right)

        return left, index

    def _parse_and_expression(self, tokens: List[str], index: int) -> Tuple:
        left, index = self._parse_not_expression(tokens, index)

        while index < len(tokens) and tokens[index] == "&":
            index += 1
            right, index = self._parse_not_expression(tokens, index)
            left = ("AND", left, right)

        return left, index

    def _parse_not_expression(self, tokens: List[str], index: int) -> Tuple:
        if index < len(tokens) and tokens[index] == "!":
            index += 1
            expr, index = self._parse_primary_expression(tokens, index)
            return ("NOT", expr), index
        return self._parse_primary_expression(tokens, index)

    def _parse_primary_expression(self, tokens: List[str], index: int) -> Tuple:
        if index < len(tokens) and tokens[index] == "(":
            index += 1
            expr, index = self._parse_logical_expression(tokens, index)
            if index < len(tokens) and tokens[index] == ")":
                index += 1
                return expr, index
            raise ValueError("Missing closing parenthesis")
        if index < len(tokens):
            comparison_expr = tokens[index]
            index += 1
            return ("COMPARISON", comparison_expr), index
        raise ValueError("Expected comparison expression")

    def _evaluate_comparison(self, comparison_str: str, message: dotdict) -> bool:  # noqa: C901
        match = re.match(self.patterns["comparison"], comparison_str.strip())
        if not match:
            raise ValueError(
                f"Invalid condition format: {comparison_str}. "
                "Expected key_path operator value."
            )

        key_path, operator, expected_value_str = match.groups()
        actual_value = message.get(key_path, None)
        expected_value_str = expected_value_str.strip()

        if operator in ["is", "is not"]:
            if expected_value_str.lower() in ["none", "null"]:
                if operator == "is":
                    return actual_value is None
                return actual_value is not None
            raise ValueError(
                "`is` and `is not` operators only support `None` or `null` comparisons"
            )

        expected_value_str = expected_value_str.strip("'\"")

        try:
            if expected_value_str.lower() == "true":
                expected_value = True
            elif expected_value_str.lower() == "false":
                expected_value = False
            elif "." in expected_value_str:
                expected_value = float(expected_value_str)
            else:
                try:
                    expected_value = int(expected_value_str)
                except ValueError:
                    expected_value = expected_value_str

            if actual_value is None:
                return False

            if isinstance(expected_value, bool):
                if isinstance(actual_value, str):
                    actual_value = actual_value.lower() == "true"
                else:
                    actual_value = bool(actual_value)
            elif isinstance(expected_value, (int, float)):
                actual_value = type(expected_value)(actual_value)

        except (ValueError, TypeError):
            expected_value = expected_value_str
            actual_value = str(actual_value) if actual_value is not None else None

        if actual_value is None and operator not in ["==", "!="]:
            return False

        if operator == "==":
            return actual_value == expected_value
        if operator == "!=":
            return actual_value != expected_value
        if operator == "<":
            return actual_value < expected_value
        if operator == ">":
            return actual_value > expected_value
        if operator == "<=":
            return actual_value <= expected_value
        if operator == ">=":
            return actual_value >= expected_value
        raise ValueError(f"Unknown comparison operator: {operator}")

    def _evaluate_logical_tree(self, tree: tuple, message: dotdict) -> bool:
        if tree[0] == "COMPARISON":
            return self._evaluate_comparison(tree[1], message)
        if tree[0] == "AND":
            left_result = self._evaluate_logical_tree(tree[1], message)
            right_result = self._evaluate_logical_tree(tree[2], message)
            return left_result and right_result
        if tree[0] == "OR":
            left_result = self._evaluate_logical_tree(tree[1], message)
            right_result = self._evaluate_logical_tree(tree[2], message)
            return left_result or right_result
        if tree[0] == "NOT":
            expr_result = self._evaluate_logical_tree(tree[1], message)
            return not expr_result
        raise ValueError(f"Unknown logical operator: {tree[0]}")

    def _evaluate_condition(self, condition: str, message: dotdict) -> bool:
        tokens = self._tokenize_condition(condition)
        if not tokens:
            raise ValueError("Empty condition")

        tree, final_index = self._parse_logical_expression(tokens)
        if final_index != len(tokens):
            raise ValueError(f"Unexpected tokens after parsing: {tokens[final_index:]}")

        return self._evaluate_logical_tree(tree, message)

    def _get_module(self, module_name: str) -> Callable:
        module = self.modules.get(module_name)
        if module is None:
            raise ValueError(f"Module `{module_name}` not found.")
        return module

    def _validate_task_result(self, module_name: str, result: Any) -> None:
        if isinstance(result, TaskError):
            raise RuntimeError(
                f"Execution failed for `{module_name}`: {result.exception}"
            )

    def _validate_parallel_results(
        self, module_names: List[str], results: Tuple[Any, ...]
    ) -> None:
        failures = [
            (name, result)
            for name, result in zip(module_names, results)
            if isinstance(result, TaskError)
        ]
        if failures:
            failed = ", ".join(
                f"`{name}`: {task_error.exception}" for name, task_error in failures
            )
            raise RuntimeError(f"Parallel execution failed for: {failed}")

    def _call_module(self, module_name: str, message: dotdict) -> None:
        module = self._get_module(module_name)
        result = module(message)
        if asyncio.iscoroutine(result):
            raise RuntimeError(
                f"Module `{module_name}` returned a coroutine in sync execution. "
                "Use `acall` instead."
            )
        self._validate_task_result(module_name, result)

    async def _acall_module(self, module_name: str, message: dotdict) -> None:
        module = self._get_module(module_name)

        if hasattr(module, "acall"):
            result = await module.acall(message)
        elif asyncio.iscoroutinefunction(module):
            result = await module(message)
        else:
            result = module(message)

        self._validate_task_result(module_name, result)

    def _execute_while_loop(
        self, condition: str, actions: List[Dict[str, Any]], message: dotdict
    ) -> None:
        iterations = 0

        while self._evaluate_condition(condition, message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            self._execute_steps(actions, message)
            iterations += 1

    async def _aexecute_while_loop(
        self, condition: str, actions: List[Dict[str, Any]], message: dotdict
    ) -> None:
        iterations = 0

        while self._evaluate_condition(condition, message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            await self._aexecute_steps(actions, message)
            iterations += 1

    def _execute_steps(self, steps: List[Dict[str, Any]], message: dotdict) -> None:
        for step in steps:
            if step["type"] == "module":
                self._call_module(step["module"], message)

            elif step["type"] == "parallel":
                module_names = step["modules"]
                parallel_modules = [
                    functools.partial(self._call_module, module_name)
                    for module_name in module_names
                ]
                results = F.bcast_gather(parallel_modules, message)
                self._validate_parallel_results(module_names, results)

            elif step["type"] == "conditional":
                self._execute_conditional(step, message)

            elif step["type"] == "while":
                self._execute_while_loop(step["condition"], step["actions"], message)

    def _execute_conditional(self, step: Dict[str, Any], message: dotdict) -> None:
        for branch in step["branches"]:
            if self._evaluate_condition(branch["condition"], message):
                self._call_module(branch["module"], message)
                return
        for module_name in step["default"]:
            self._call_module(module_name, message)

    async def _aexecute_steps(
        self, steps: List[Dict[str, Any]], message: dotdict
    ) -> None:
        for step in steps:
            if step["type"] == "module":
                await self._acall_module(step["module"], message)

            elif step["type"] == "parallel":
                module_names = step["modules"]
                parallel_modules = [
                    functools.partial(self._acall_module, module_name)
                    for module_name in module_names
                ]
                args_list = [(message,)] * len(parallel_modules)
                results = await F.ascatter_gather(parallel_modules, args_list=args_list)
                self._validate_parallel_results(module_names, results)

            elif step["type"] == "conditional":
                await self._aexecute_conditional(step, message)

            elif step["type"] == "while":
                await self._aexecute_while_loop(
                    step["condition"], step["actions"], message
                )

    async def _aexecute_conditional(
        self, step: Dict[str, Any], message: dotdict
    ) -> None:
        for branch in step["branches"]:
            if self._evaluate_condition(branch["condition"], message):
                await self._acall_module(branch["module"], message)
                return
        for module_name in step["default"]:
            await self._acall_module(module_name, message)

    def __call__(self, message: dotdict) -> dotdict:
        self._validate_message(message)
        self._execute_steps(self.steps, message)
        return message

    async def acall(self, message: dotdict) -> dotdict:
        self._validate_message(message)
        await self._aexecute_steps(self.steps, message)
        return message

    # --- Visualization ----------------------------------------------------

    _MERMAID_THEME = (
        "%%{\n"
        "  init: {\n"
        "    'theme': 'base',\n"
        "    'themeVariables': {\n"
        "      'primaryColor': '#E9E7E7',\n"
        "      'primaryTextColor': '#000000',\n"
        "      'primaryBorderColor': '#C00000',\n"
        "      'lineColor': '#F8B229',\n"
        "      'secondaryColor': '#91939C',\n"
        "      'tertiaryColor': '#fff'\n"
        "    }\n"
        "  }\n"
        "}%%"
    )

    _MERMAID_STYLES = (
        "\n%% Styles\n"
        "classDef conditional fill:#f5e1ce,stroke:#333,stroke-width:2px;\n"
        "classDef condition fill:#f5e1ce,stroke:#333,stroke-width:2px;\n"
        "classDef loop fill:#fadc9b,stroke:#333,stroke-width:2px;"
    )

    def to_mermaid(self, title: str = "") -> str:
        """Generate a Mermaid flowchart from the parsed pipeline."""
        ctx: Dict[str, Any] = {"n": 0, "classes": []}
        header = f"---\ntitle: {self._md_title(title)}\n---\n" if title else ""
        lines = [f"{header}{self._MERMAID_THEME}", "flowchart TD"]
        self._md_render(self.steps, [], lines, ctx)
        lines.append(self._MERMAID_STYLES)
        for cls_line in ctx["classes"]:
            lines.append(cls_line)
        return "\n".join(lines)

    @staticmethod
    def _md_label(value: Any) -> str:
        return str(value).replace('"', "#quot;").replace("\n", "<br/>")

    @classmethod
    def _md_title(cls, value: str) -> str:
        return f'"{cls._md_label(value)}"'

    def _md_nid(self, ctx: Dict[str, Any]) -> str:
        nid = f"n{ctx['n']}"
        ctx["n"] += 1
        return nid

    def _md_render(
        self,
        steps: List[Dict[str, Any]],
        prev: List[str],
        lines: List[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        curr = prev
        for step in steps:
            stype = step["type"]
            if stype == "module":
                curr = self._md_module(step, curr, lines, ctx)
            elif stype == "parallel":
                curr = self._md_parallel(step, curr, lines, ctx)
            elif stype == "conditional":
                curr = self._md_conditional(step, curr, lines, ctx)
            elif stype == "while":
                curr = self._md_while(step, curr, lines, ctx)
        return curr

    def _md_module(
        self,
        step: Dict[str, Any],
        curr: List[str],
        lines: List[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        nid = self._md_nid(ctx)
        lines.append(f'  {nid}["{self._md_label(step["module"])}"]')
        for p in curr:
            lines.append(f"  {p} --> {nid}")
        return [nid]

    def _md_parallel(
        self,
        step: Dict[str, Any],
        curr: List[str],
        lines: List[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        exits: List[str] = []
        for mod in step["modules"]:
            nid = self._md_nid(ctx)
            lines.append(f'  {nid}["{self._md_label(mod)}"]')
            for p in curr:
                lines.append(f"  {p} --> {nid}")
            exits.append(nid)
        return exits

    def _md_conditional(
        self,
        step: Dict[str, Any],
        curr: List[str],
        lines: List[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        # Decision diamond
        cid = self._md_nid(ctx)
        lines.append(f'  {cid}{{"Decision"}}')
        ctx["classes"].append(f"class {cid} conditional;")
        for p in curr:
            lines.append(f"  {p} --> {cid}")
        branch_exits: List[str] = []
        # Each branch: Decision -> condition hexagon -> module rectangle
        for i, branch in enumerate(step["branches"]):
            keyword = "if" if i == 0 else "elif"
            cond_nid = self._md_nid(ctx)
            condition = self._md_label(branch["condition"])
            lines.append(f'  {cond_nid}{{{{"{keyword} {condition}"}}}}')
            ctx["classes"].append(f"class {cond_nid} condition;")
            lines.append(f"  {cid} --> {cond_nid}")
            mod_nid = self._md_nid(ctx)
            lines.append(f'  {mod_nid}["{self._md_label(branch["module"])}"]')
            lines.append(f"  {cond_nid} --> {mod_nid}")
            branch_exits.append(mod_nid)
        if step["default"]:
            else_nid = self._md_nid(ctx)
            lines.append(f'  {else_nid}{{{{"else"}}}}')
            ctx["classes"].append(f"class {else_nid} condition;")
            lines.append(f"  {cid} --> {else_nid}")
            default_exits = [else_nid]
            for mod in step["default"]:
                mod_nid = self._md_nid(ctx)
                lines.append(f'  {mod_nid}["{self._md_label(mod)}"]')
                for p in default_exits:
                    lines.append(f"  {p} --> {mod_nid}")
                default_exits = [mod_nid]
            branch_exits.extend(default_exits)
        else:
            branch_exits.append(cid)
        return branch_exits

    def _md_while(
        self,
        step: Dict[str, Any],
        curr: List[str],
        lines: List[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        cid = self._md_nid(ctx)
        lines.append(f'  {cid}(("{self._md_label(step["condition"])}"))')
        ctx["classes"].append(f"class {cid} loop;")
        for p in curr:
            lines.append(f"  {p} --> {cid}")
        body_exits = self._md_render(step["actions"], [cid], lines, ctx)
        for e in body_exits:
            lines.append(f"  {e} -.->|loop| {cid}")
        return [cid]
