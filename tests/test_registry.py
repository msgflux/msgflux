"""Tests for msgflux.core.registry.Registry."""

from msgflux.core.registry import Registry


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_fn(name: str):
    """Create a simple callable with a given __name__."""

    def fn():
        pass

    fn.__name__ = name
    return fn


class _ObjWithName:
    """Callable with a .name attribute."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self):
        pass


# ── bare decorator ───────────────────────────────────────────────────────────


def test_bare_decorator_uses_dunder_name():
    reg = Registry()
    fn = _make_fn("search")
    result = reg(fn)
    assert result is fn
    assert "search" in reg
    assert reg.to_list() == [fn]


def test_bare_decorator_prefers_dot_name():
    reg = Registry()
    obj = _ObjWithName("my_tool")
    result = reg(obj)
    assert result is obj
    assert "my_tool" in reg
    assert reg.get("my_tool") is obj


# ── decorator with explicit name (positional) ───────────────────────────────


def test_positional_name():
    reg = Registry()
    fn = _make_fn("original")

    @reg("custom")
    def wrapped():
        pass

    assert "custom" in reg
    assert "original" not in reg


# ── decorator with explicit name (keyword) ──────────────────────────────────


def test_keyword_name():
    reg = Registry()

    @reg(name="summarizer")
    def summarize():
        pass

    assert "summarizer" in reg
    assert reg.get("summarizer") is summarize


# ── list / items ─────────────────────────────────────────────────────────────


def test_to_list_returns_values():
    reg = Registry()
    a = _make_fn("a")
    b = _make_fn("b")
    reg(a)
    reg(b)
    assert reg.to_list() == [a, b]


def test_to_items_returns_dict():
    reg = Registry()
    a = _make_fn("alpha")
    b = _make_fn("beta")
    reg(a)
    reg(b)
    items = reg.to_items()
    assert items == {"alpha": a, "beta": b}


# ── len / iter / contains / repr ────────────────────────────────────────────


def test_len():
    reg = Registry()
    assert len(reg) == 0
    reg(_make_fn("x"))
    assert len(reg) == 1


def test_iter():
    reg = Registry()
    a = _make_fn("a")
    b = _make_fn("b")
    reg(a)
    reg(b)
    assert list(reg) == [a, b]


def test_repr():
    reg = Registry()
    reg(_make_fn("foo"))
    reg(_make_fn("bar"))
    assert repr(reg) == "Registry(['foo', 'bar'])"


# ── independence of instances ────────────────────────────────────────────────


def test_instances_are_independent():
    r1 = Registry()
    r2 = Registry()
    a = _make_fn("a")
    b = _make_fn("b")
    r1(a)
    r2(b)
    assert r1.to_list() == [a]
    assert r2.to_list() == [b]


# ── decorator on classes ─────────────────────────────────────────────────────


def test_register_class():
    reg = Registry()

    @reg(name="my_module")
    class MyModule:
        pass

    assert "my_module" in reg
    assert reg.get("my_module") is MyModule


def test_register_class_bare():
    reg = Registry()

    @reg
    class MyModule:
        pass

    assert "MyModule" in reg


# ── update ───────────────────────────────────────────────────────────────────


def test_update_merges_entries():
    r1 = Registry()
    r2 = Registry()
    a = _make_fn("a")
    b = _make_fn("b")
    r1(a)
    r2(b)
    r1.update(r2)
    assert r1.to_list() == [a, b]
    assert r2.to_list() == [b]  # source unchanged


def test_update_overwrites_on_collision():
    r1 = Registry()
    r2 = Registry()
    old = _make_fn("x")
    new = _make_fn("x")
    r1(old)
    r2(new)
    r1.update(r2)
    assert r1.get("x") is new


def test_update_returns_self():
    r1 = Registry()
    r2 = Registry()
    r3 = Registry()
    r1(_make_fn("a"))
    r2(_make_fn("b"))
    r3(_make_fn("c"))
    result = r1.update(r2).update(r3)
    assert result is r1
    assert len(r1) == 3


# ── pop / clear ──────────────────────────────────────────────────────────────


def test_pop_removes_and_returns():
    reg = Registry()
    fn = _make_fn("x")
    reg(fn)
    result = reg.pop("x")
    assert result is fn
    assert "x" not in reg
    assert len(reg) == 0


def test_pop_missing_raises():
    reg = Registry()
    try:
        reg.pop("missing")
        assert False, "should have raised"  # noqa: B011
    except KeyError:
        pass


def test_clear():
    reg = Registry()
    reg(_make_fn("a"))
    reg(_make_fn("b"))
    reg.clear()
    assert len(reg) == 0
    assert reg.to_list() == []
