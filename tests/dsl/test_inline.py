import pytest

from msgflux.core.dotdict import dotdict
from msgflux.dsl.inline import Inline


def inline(expression: str, modules, message: dotdict) -> dotdict:
    return Inline(expression, modules)(message)


async def ainline(expression: str, modules, message: dotdict) -> dotdict:
    return await Inline(expression, modules).acall(message)


@pytest.fixture
def modules():
    def prep(msg: dotdict) -> None:
        msg["output"] = {"agent": "xpto", "score": 10, "status": "success"}
        msg["counter"] = 0

    def increment(msg: dotdict) -> None:
        msg["counter"] = msg.get("counter", 0) + 1

    def feat_a(msg: dotdict) -> None:
        msg["feat_a"] = "result_a"

    def feat_b(msg: dotdict) -> None:
        msg["feat_b"] = "result_b"

    def final(msg: dotdict) -> None:
        msg["final"] = "done"

    return {
        "prep": prep,
        "increment": increment,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "final": final,
    }


@pytest.fixture
def async_modules():
    async def prep(msg: dotdict) -> None:
        msg["output"] = {"agent": "xpto", "score": 10, "status": "success"}
        msg["counter"] = 0

    async def increment(msg: dotdict) -> None:
        msg["counter"] = msg.get("counter", 0) + 1

    async def feat_a(msg: dotdict) -> None:
        msg["feat_a"] = "result_a"

    async def feat_b(msg: dotdict) -> None:
        msg["feat_b"] = "result_b"

    async def final(msg: dotdict) -> None:
        msg["final"] = "done"

    return {
        "prep": prep,
        "increment": increment,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "final": final,
    }


def test_inline_class_flow(modules):
    pipeline = Inline("prep -> final", modules)
    result = pipeline(dotdict())

    assert result["final"] == "done"
    assert "output" in result


@pytest.mark.asyncio
async def test_inline_class_async_flow(async_modules):
    pipeline = Inline("prep -> final", async_modules)
    result = await pipeline.acall(dotdict())

    assert result["final"] == "done"
    assert "output" in result


def test_simple_sequential_flow(modules):
    expression = "prep -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["final"] == "done"
    assert "output" in result


def test_parallel_flow(modules):
    expression = "prep -> [feat_a, feat_b] -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


def test_conditional_flow_true(modules):
    expression = "prep -> {output.agent == 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert result["final"] == "done"


def test_conditional_flow_false(modules):
    expression = "prep -> {output.agent == 'unknown'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


def test_while_loop(modules):
    expression = "prep -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 5
    assert result["final"] == "done"


def test_nested_while_loop(modules):
    expression = "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 3
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_simple_sequential_flow(async_modules):
    expression = "prep -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["final"] == "done"
    assert "output" in result


@pytest.mark.asyncio
async def test_async_parallel_flow(async_modules):
    expression = "prep -> [feat_a, feat_b] -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_conditional_flow_true(async_modules):
    expression = "prep -> {output.agent == 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_conditional_flow_false(async_modules):
    expression = "prep -> {output.agent == 'unknown'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_while_loop(async_modules):
    expression = "prep -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["counter"] == 5
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_nested_while_loop(async_modules):
    expression = "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["counter"] == 3
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["final"] == "done"


def test_conditional_with_and_operator(modules):
    """Test conditional with AND operator."""
    expression = (
        "prep -> {output.agent == 'xpto' & output.score >= 10?feat_a,feat_b} -> final"
    )
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_or_operator(modules):
    """Test conditional with OR operator."""
    expression = "prep -> {output.agent == 'unknown' || output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_not_operator(modules):
    """Test conditional with NOT operator."""
    expression = "prep -> {!(output.agent == 'unknown')?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_is_none(modules):
    """Test conditional with is None check."""

    def setup(msg: dotdict) -> dotdict:
        msg["value"] = None
        return msg

    modules_with_setup = {**modules, "setup": setup}
    expression = "setup -> {value is None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_is_not_none(modules):
    """Test conditional with is not None check."""

    def setup(msg: dotdict) -> dotdict:
        msg["value"] = "exists"
        return msg

    modules_with_setup = {**modules, "setup": setup}
    expression = "setup -> {value is not None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_with_comparison_operators(modules):
    """Test conditional with different comparison operators."""
    expression = "prep -> {output.score > 5?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_less_than(modules):
    """Test conditional with less than operator."""
    expression = "prep -> {output.score < 100?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_while_loop_with_nested_parallel(modules):
    """Test while loop with nested parallel execution."""
    expression = "prep -> @{counter < 2}: increment -> [feat_a, feat_b]; -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 2
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"


def test_inline_with_dotted_path(modules):
    """Test accessing nested attributes in conditions."""
    expression = "prep -> {output.status == 'success'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_parallel_with_multiple_modules(modules):
    """Test parallel execution with more than two modules."""

    def feat_c(msg: dotdict) -> dotdict:
        msg["feat_c"] = "result_c"
        return msg

    modules_extended = {**modules, "feat_c": feat_c}
    expression = "prep -> [feat_a, feat_b, feat_c] -> final"
    input_msg = dotdict()
    result = inline(expression, modules_extended, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["feat_c"] == "result_c"


@pytest.mark.asyncio
async def test_async_conditional_with_and_operator(async_modules):
    """Test async conditional with AND operator."""
    expression = (
        "prep -> {output.agent == 'xpto' & output.score >= 10?feat_a,feat_b} -> final"
    )
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_conditional_with_or_operator(async_modules):
    """Test async conditional with OR operator."""
    expression = "prep -> {output.agent == 'unknown' || output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_conditional_is_none(async_modules):
    """Test async conditional with is None check."""

    async def setup(msg: dotdict) -> dotdict:
        msg["value"] = None
        return msg

    modules_with_setup = {**async_modules, "setup": setup}
    expression = "setup -> {value is None?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_with_setup, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_parallel_with_multiple_modules(async_modules):
    """Test async parallel execution with multiple modules."""

    async def feat_c(msg: dotdict) -> dotdict:
        msg["feat_c"] = "result_c"
        return msg

    modules_extended = {**async_modules, "feat_c": feat_c}
    expression = "prep -> [feat_a, feat_b, feat_c] -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_extended, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["feat_b"] == "result_b"
    assert result["feat_c"] == "result_c"


def test_conditional_with_parentheses(modules):
    """Test conditional with parentheses for grouping."""
    expression = "prep -> {(output.agent == 'xpto') & (output.score >= 10)?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result


def test_conditional_complex_logic(modules):
    """Test conditional with complex logical expression."""
    expression = "prep -> {(output.agent == 'xpto' || output.score >= 100) & !(output.status == 'failed')?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_empty_branch_in_conditional(modules):
    """Test conditional with empty false branch."""
    expression = "prep -> {output.agent == 'xpto'?feat_a} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"
    assert result["final"] == "done"


def test_single_module_execution(modules):
    """Test executing a single module."""
    expression = "prep"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert "output" in result
    assert result["counter"] == 0


def test_sequential_multiple_steps(modules):
    """Test sequential execution with many steps."""
    expression = "prep -> increment -> increment -> increment -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["counter"] == 3
    assert result["final"] == "done"


def test_nested_conditionals(modules):
    """Test nested conditional expressions."""

    def check_score(msg: dotdict) -> dotdict:
        if msg["output"]["score"] >= 10:
            msg["high_score"] = True
        return msg

    modules_ext = {**modules, "check_score": check_score}
    expression = "prep -> {output.agent == 'xpto'?check_score,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_ext, input_msg)
    assert result.get("high_score") is True


def test_comparison_greater_equal(modules):
    """Test >= operator in conditional."""
    expression = "prep -> {output.score >= 10?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_comparison_less_equal(modules):
    """Test <= operator in conditional."""
    expression = "prep -> {output.score <= 100?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_while_loop_zero_iterations(modules):
    """Test while loop that doesn't execute (condition false from start)."""

    def set_high_counter(msg: dotdict) -> dotdict:
        msg["counter"] = 100
        return msg

    modules_ext = {**modules, "set_high": set_high_counter}
    expression = "set_high -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = inline(expression, modules_ext, input_msg)
    assert result["counter"] == 100  # No increment happened
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_conditional_with_parentheses(async_modules):
    """Test async conditional with parentheses."""
    expression = "prep -> {(output.agent == 'xpto') & (output.score >= 10)?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"


@pytest.mark.asyncio
async def test_async_empty_branch(async_modules):
    """Test async conditional with empty false branch."""
    expression = "prep -> {output.agent == 'xpto'?feat_a} -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["feat_a"] == "result_a"


@pytest.mark.asyncio
async def test_async_single_module(async_modules):
    """Test async single module execution."""
    expression = "prep"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert "output" in result


@pytest.mark.asyncio
async def test_async_sequential_multiple_steps(async_modules):
    """Test async sequential with multiple steps."""
    expression = "prep -> increment -> increment -> final"
    input_msg = dotdict()
    result = await ainline(expression, async_modules, input_msg)
    assert result["counter"] == 2


def test_parallel_task_error_raises_runtime_error():
    def ok(msg: dotdict) -> None:
        msg["ok"] = True

    def fail(msg: dotdict) -> None:
        raise ValueError("boom")

    pipeline = Inline("[ok, fail]", {"ok": ok, "fail": fail})

    with pytest.raises(RuntimeError, match="Parallel execution failed"):
        pipeline(dotdict())


@pytest.mark.asyncio
async def test_async_parallel_task_error_raises_runtime_error():
    async def ok(msg: dotdict) -> None:
        msg["ok"] = True

    async def fail(msg: dotdict) -> None:
        raise ValueError("boom")

    pipeline = Inline("[ok, fail]", {"ok": ok, "fail": fail})

    with pytest.raises(RuntimeError, match="Parallel execution failed"):
        await pipeline.acall(dotdict())


def test_conditional_missing_closing_parenthesis_raises_error(modules):
    """Test that missing closing parenthesis raises ValueError."""
    expression = "prep -> {(output.agent == 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    with pytest.raises(ValueError, match="Missing closing parenthesis"):
        inline(expression, modules, input_msg)


def test_conditional_invalid_format_raises_error(modules):
    """Test that invalid condition format raises ValueError."""
    expression = "prep -> {invalid_condition?feat_a,feat_b} -> final"
    input_msg = dotdict()
    with pytest.raises(ValueError, match="Invalid condition format"):
        inline(expression, modules, input_msg)


def test_conditional_is_with_non_none_raises_error(modules):
    """Test that 'is' operator with non-None value raises error."""
    expression = "prep -> {output.agent is 'xpto'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    with pytest.raises(ValueError, match="`is` and `is not` operators only support"):
        inline(expression, modules, input_msg)


def test_conditional_empty_condition_raises_error(modules):
    """Test that empty condition raises ValueError."""
    expression = "prep -> {?feat_a,feat_b} -> final"
    input_msg = dotdict()
    with pytest.raises(ValueError, match="Empty condition"):
        inline(expression, modules, input_msg)


def test_conditional_with_boolean_string_values(modules):
    """Test conditional with string boolean values."""

    def setup(msg: dotdict) -> dotdict:
        msg["is_active"] = "true"
        return msg

    modules_ext = {**modules, "setup": setup}
    expression = "setup -> {is_active == true?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_ext, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_with_boolean_false_string(modules):
    """Test conditional with string 'false' value."""

    def setup(msg: dotdict) -> dotdict:
        msg["is_active"] = "false"
        return msg

    modules_ext = {**modules, "setup": setup}
    expression = "setup -> {is_active == false?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_ext, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_with_float_comparison(modules):
    """Test conditional with float values."""

    def setup(msg: dotdict) -> dotdict:
        msg["price"] = 19.99
        return msg

    modules_ext = {**modules, "setup": setup}
    expression = "setup -> {price > 10.5?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules_ext, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_with_nonexistent_key_returns_false(modules):
    """Test that nonexistent key in condition evaluates to false."""
    expression = "prep -> {nonexistent.key == 'value'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_b"] == "result_b"
    assert "feat_a" not in result


def test_conditional_with_string_comparison(modules):
    """Test conditional with string comparison."""
    expression = "prep -> {output.status == 'success'?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_while_loop_max_iterations_safety(modules):
    """Test while loop respects max_iterations limit."""

    def never_stop(msg: dotdict) -> dotdict:
        msg["counter"] = msg.get("counter", 0) + 1
        return msg

    modules_ext = {**modules, "never_stop": never_stop}
    # This would be an infinite loop, but should raise RuntimeError at max_iterations
    expression = "prep -> @{counter < 10000}: never_stop; -> final"
    input_msg = dotdict()
    # Should raise RuntimeError due to max_iterations limit (default 1000)
    with pytest.raises(RuntimeError, match="While loop exceeded maximum iterations"):
        inline(expression, modules_ext, input_msg)


def test_parallel_empty_list_edge_case(modules):
    """Test parallel with minimal modules."""
    expression = "prep -> [feat_a] -> final"
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


def test_conditional_with_quoted_strings(modules):
    """Test conditional with both single and double quoted strings."""
    expression = 'prep -> {output.agent == "xpto"?feat_a,feat_b} -> final'
    input_msg = dotdict()
    result = inline(expression, modules, input_msg)
    assert result["feat_a"] == "result_a"


@pytest.mark.asyncio
async def test_async_conditional_invalid_format_raises_error(async_modules):
    """Test async invalid condition format raises ValueError."""
    expression = "prep -> {invalid?feat_a,feat_b} -> final"
    input_msg = dotdict()
    with pytest.raises(ValueError, match="Invalid condition format"):
        await ainline(expression, async_modules, input_msg)


@pytest.mark.asyncio
async def test_async_conditional_with_float(async_modules):
    """Test async conditional with float comparison."""

    async def setup(msg: dotdict) -> dotdict:
        msg["price"] = 25.5
        return msg

    modules_ext = {**async_modules, "setup": setup}
    expression = "setup -> {price > 20.0?feat_a,feat_b} -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_ext, input_msg)
    assert result["feat_a"] == "result_a"


@pytest.mark.asyncio
async def test_async_while_loop_zero_iterations(async_modules):
    """Test async while loop with zero iterations."""

    async def set_high(msg: dotdict) -> dotdict:
        msg["counter"] = 100
        return msg

    modules_ext = {**async_modules, "set_high": set_high}
    expression = "set_high -> @{counter < 5}: increment; -> final"
    input_msg = dotdict()
    result = await ainline(expression, modules_ext, input_msg)
    assert result["counter"] == 100


# ── Multi-branch conditional tests ──────────────────────────────────────────


def test_multibranch_first_match(modules):
    """First matching branch is executed."""
    expression = "prep -> {output.score > 9 ? feat_a, output.score > 5 ? feat_b, final}"
    result = inline(expression, modules, dotdict())
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert "final" not in result


def test_multibranch_second_match(modules):
    """Second branch matches when first doesn't."""
    expression = (
        "prep -> {output.score > 20 ? feat_a, output.score > 5 ? feat_b, final}"
    )
    result = inline(expression, modules, dotdict())
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
    assert "final" not in result


def test_multibranch_default(modules):
    """Default branch executes when no condition matches."""
    expression = (
        "prep -> {output.score > 100 ? feat_a, output.score > 50 ? feat_b, final}"
    )
    result = inline(expression, modules, dotdict())
    assert "feat_a" not in result
    assert "feat_b" not in result
    assert result["final"] == "done"


def test_multibranch_no_default(modules):
    """No module executes when no condition matches and no default."""
    expression = "prep -> {output.score > 100 ? feat_a, output.score > 50 ? feat_b}"
    result = inline(expression, modules, dotdict())
    assert "feat_a" not in result
    assert "feat_b" not in result


def test_multibranch_wildcard_default(modules):
    """Explicit _ wildcard acts as default."""
    expression = "prep -> {output.score > 100 ? feat_a, _ ? feat_b}"
    result = inline(expression, modules, dotdict())
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"


def test_multibranch_in_pipeline(modules):
    """Multi-branch conditional works inside a full pipeline."""
    expression = (
        "prep -> {output.score > 15 ? feat_a, output.score > 5 ? feat_b, final}"
        " -> increment"
    )
    result = inline(expression, modules, dotdict())
    assert result["feat_b"] == "result_b"
    assert result["counter"] == 1


def test_multibranch_with_logic_operators(modules):
    """Multi-branch with complex conditions."""
    expression = (
        "prep -> {"
        "output.score > 100 & output.agent == 'xpto' ? feat_a, "
        "output.score >= 10 || output.agent == 'unknown' ? feat_b, "
        "final"
        "}"
    )
    result = inline(expression, modules, dotdict())
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"


@pytest.mark.asyncio
async def test_async_multibranch_first_match(async_modules):
    """Async: first matching branch is executed."""
    expression = "prep -> {output.score > 9 ? feat_a, output.score > 5 ? feat_b, final}"
    result = await ainline(expression, async_modules, dotdict())
    assert result["feat_a"] == "result_a"
    assert "feat_b" not in result
    assert "final" not in result


@pytest.mark.asyncio
async def test_async_multibranch_default(async_modules):
    """Async: default branch executes when no condition matches."""
    expression = (
        "prep -> {output.score > 100 ? feat_a, output.score > 50 ? feat_b, final}"
    )
    result = await ainline(expression, async_modules, dotdict())
    assert "feat_a" not in result
    assert "feat_b" not in result
    assert result["final"] == "done"


@pytest.mark.asyncio
async def test_async_multibranch_no_default(async_modules):
    """Async: no module executes when nothing matches and no default."""
    expression = "prep -> {output.score > 100 ? feat_a, output.score > 50 ? feat_b}"
    result = await ainline(expression, async_modules, dotdict())
    assert "feat_a" not in result
    assert "feat_b" not in result


@pytest.mark.asyncio
async def test_async_multibranch_wildcard(async_modules):
    """Async: explicit _ wildcard acts as default."""
    expression = "prep -> {output.score > 100 ? feat_a, _ ? feat_b}"
    result = await ainline(expression, async_modules, dotdict())
    assert "feat_a" not in result
    assert result["feat_b"] == "result_b"
