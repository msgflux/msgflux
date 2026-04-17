"""Tests for msgflux.dotdict module."""

from dataclasses import asdict, dataclass

import msgspec
import pytest

from msgflux.core.dotdict import dotdict


class TestDotdictInitialization:
    """Test suite for dotdict initialization."""

    def test_init_empty(self):
        """Test initializing empty dotdict."""
        d = dotdict()
        assert len(d) == 0
        assert isinstance(d, dict)
        assert isinstance(d, dotdict)

    def test_init_with_dict(self):
        """Test initializing with dictionary."""
        data = {"name": "Alice", "age": 30}
        d = dotdict(data)

        assert d["name"] == "Alice"
        assert d["age"] == 30
        assert len(d) == 2

    def test_init_with_kwargs(self):
        """Test initializing with keyword arguments."""
        d = dotdict(name="Bob", age=25)

        assert d["name"] == "Bob"
        assert d["age"] == 25

    def test_init_with_dict_and_kwargs(self):
        """Test initializing with both dict and kwargs."""
        data = {"name": "Charlie"}
        d = dotdict(data, age=35, city="NYC")

        assert d["name"] == "Charlie"
        assert d["age"] == 35
        assert d["city"] == "NYC"

    def test_init_kwargs_override_dict(self):
        """Test that kwargs override dict values."""
        data = {"name": "Dave", "age": 40}
        d = dotdict(data, age=45)

        assert d["name"] == "Dave"
        assert d["age"] == 45

    def test_init_nested_dict(self):
        """Test initializing with nested dictionary."""
        data = {"user": {"name": "Eve", "profile": {"age": 28}}}
        d = dotdict(data)

        assert isinstance(d["user"], dotdict)
        assert isinstance(d["user"]["profile"], dotdict)
        assert d["user"]["profile"]["age"] == 28

    def test_init_with_list_of_dicts(self):
        """Test initializing with list containing dictionaries."""
        data = {"items": [{"name": "item1"}, {"name": "item2"}]}
        d = dotdict(data)

        assert isinstance(d["items"], list)
        assert isinstance(d["items"][0], dotdict)
        assert d["items"][0]["name"] == "item1"

    def test_init_with_iterable_of_pairs(self):
        """Test initializing dotdict from an iterable of key/value pairs."""
        d = dotdict((("name", "Alice"), ("age", 30)))

        assert d["name"] == "Alice"
        assert d["age"] == 30

    def test_dataclasses_asdict_accepts_nested_dotdict(self):
        """Test stdlib asdict can rebuild nested dotdict instances."""

        @dataclass
        class Payload:
            metadata: dotdict

        payload = Payload(
            metadata=dotdict(
                {
                    "participants": [
                        {"name": "Alice", "role": "speaker"},
                        {"name": "Bob", "role": "listener"},
                    ]
                }
            )
        )

        result = asdict(payload)

        assert result == {
            "metadata": {
                "participants": [
                    {"name": "Alice", "role": "speaker"},
                    {"name": "Bob", "role": "listener"},
                ]
            }
        }

    def test_init_frozen(self):
        """Test initializing frozen dotdict."""
        d = dotdict({"name": "Frank"}, frozen=True)

        assert d._frozen is True
        assert d["name"] == "Frank"


class TestDotNotationAccess:
    """Test suite for dot notation access."""

    def test_getattr_basic(self):
        """Test basic attribute access."""
        d = dotdict({"name": "Grace", "age": 32})

        assert d.name == "Grace"
        assert d.age == 32

    def test_getattr_nested(self):
        """Test nested attribute access."""
        d = dotdict({"user": {"profile": {"name": "Henry"}}})

        assert d.user.profile.name == "Henry"

    def test_getattr_nonexistent(self):
        """Test accessing non-existent attribute raises AttributeError."""
        d = dotdict({"name": "Iris"})

        with pytest.raises(AttributeError, match="has no attribute 'age'"):
            _ = d.age

    def test_setattr_basic(self):
        """Test basic attribute assignment."""
        d = dotdict()
        d.name = "Jack"
        d.age = 29

        assert d["name"] == "Jack"
        assert d["age"] == 29

    def test_setattr_nested(self):
        """Test nested attribute assignment."""
        d = dotdict({"user": {}})
        d.user.name = "Kate"

        assert d["user"]["name"] == "Kate"

    def test_setattr_frozen(self):
        """Test setting attribute on frozen dotdict raises error."""
        d = dotdict({"name": "Leo"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d.age = 30

    def test_setattr_private_allowed(self):
        """Test that private attributes (starting with _) can be set."""
        d = dotdict()
        d._internal = "value"

        assert d._internal == "value"


class TestBracketAccess:
    """Test suite for bracket notation access."""

    def test_getitem_basic(self):
        """Test basic bracket access."""
        d = dotdict({"name": "Mary", "age": 27})

        assert d["name"] == "Mary"
        assert d["age"] == 27

    def test_getitem_nonexistent(self):
        """Test accessing non-existent key raises KeyError."""
        d = dotdict({"name": "Nick"})

        with pytest.raises(KeyError):
            _ = d["age"]

    def test_setitem_basic(self):
        """Test basic bracket assignment."""
        d = dotdict()
        d["name"] = "Olivia"
        d["age"] = 26

        assert d["name"] == "Olivia"
        assert d["age"] == 26

    def test_setitem_frozen(self):
        """Test setting item on frozen dotdict raises error."""
        d = dotdict({"name": "Paul"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d["age"] = 31

    def test_setitem_wraps_dict(self):
        """Test that setting dict value wraps it as dotdict."""
        d = dotdict()
        d["user"] = {"name": "Quinn"}

        assert isinstance(d["user"], dotdict)
        assert d["user"]["name"] == "Quinn"


class TestDeletion:
    """Test suite for attribute/item deletion."""

    def test_delattr_basic(self):
        """Test basic attribute deletion."""
        d = dotdict({"name": "Rachel", "age": 33})
        del d.age

        assert "age" not in d
        assert d["name"] == "Rachel"

    def test_delattr_nonexistent(self):
        """Test deleting non-existent attribute raises AttributeError."""
        d = dotdict({"name": "Sam"})

        with pytest.raises(AttributeError, match="has no attribute"):
            del d.age

    def test_delattr_frozen(self):
        """Test deleting from frozen dotdict raises error."""
        d = dotdict({"name": "Tina", "age": 24}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot delete from frozen dotdict"):
            del d.age


class TestWrap:
    """Test suite for _wrap method."""

    def test_wrap_dict(self):
        """Test wrapping dictionary converts to dotdict."""
        d = dotdict()
        wrapped = d._wrap({"key": "value"})

        assert isinstance(wrapped, dotdict)
        assert wrapped["key"] == "value"

    def test_wrap_nested_dict(self):
        """Test wrapping nested dictionaries."""
        d = dotdict()
        data = {"level1": {"level2": {"value": 42}}}
        wrapped = d._wrap(data)

        assert isinstance(wrapped, dotdict)
        assert isinstance(wrapped["level1"], dotdict)
        assert isinstance(wrapped["level1"]["level2"], dotdict)

    def test_wrap_list_with_dicts(self):
        """Test wrapping list containing dictionaries."""
        d = dotdict()
        data = [{"name": "item1"}, {"name": "item2"}]
        wrapped = d._wrap(data)

        assert isinstance(wrapped, list)
        assert isinstance(wrapped[0], dotdict)
        assert isinstance(wrapped[1], dotdict)

    def test_wrap_primitive(self):
        """Test wrapping primitive values returns them unchanged."""
        d = dotdict()

        assert d._wrap("string") == "string"
        assert d._wrap(42) == 42
        assert d._wrap(3.14) == 3.14
        assert d._wrap(True) is True
        assert d._wrap(None) is None

    def test_wrap_respects_frozen(self):
        """Test that wrapped dicts inherit frozen status."""
        d = dotdict(frozen=True)
        wrapped = d._wrap({"key": "value"})

        assert isinstance(wrapped, dotdict)
        assert wrapped._frozen is True


class TestGetPath:
    """Test suite for get() method with paths."""

    def test_get_simple_path(self):
        """Test getting value with simple path."""
        d = dotdict({"name": "Uma"})

        assert d.get("name") == "Uma"

    def test_get_nested_path(self):
        """Test getting value with nested path."""
        d = dotdict({"user": {"profile": {"name": "Victor", "age": 35}}})

        assert d.get("user.profile.name") == "Victor"
        assert d.get("user.profile.age") == 35

    def test_get_path_with_list_index(self):
        """Test getting value from list using index in path."""
        d = dotdict({"items": [{"name": "first"}, {"name": "second"}]})

        assert d.get("items.0.name") == "first"
        assert d.get("items.1.name") == "second"

    def test_get_nonexistent_path_returns_default(self):
        """Test getting non-existent path returns default value."""
        d = dotdict({"user": {"name": "Wendy"}})

        assert d.get("user.age") is None
        assert d.get("user.age", 25) == 25
        assert d.get("nonexistent.path", "default") == "default"

    def test_get_invalid_index_returns_default(self):
        """Test getting invalid list index returns default."""
        d = dotdict({"items": [1, 2, 3]})

        assert d.get("items.10") is None
        assert d.get("items.10", "missing") == "missing"

    def test_get_non_numeric_index_returns_default(self):
        """Test getting non-numeric index on list returns default."""
        d = dotdict({"items": [1, 2, 3]})

        assert d.get("items.abc") is None

    def test_get_deep_nested_path(self):
        """Test getting deeply nested path."""
        d = dotdict({"a": {"b": {"c": {"d": {"value": "deep"}}}}})

        assert d.get("a.b.c.d.value") == "deep"


class TestSetPath:
    """Test suite for set() method with paths."""

    def test_set_simple_path(self):
        """Test setting value with simple path."""
        d = dotdict()
        d.set("name", "Xavier")

        assert d["name"] == "Xavier"

    def test_set_nested_path(self):
        """Test setting value with nested path creates intermediate dicts."""
        d = dotdict()
        d.set("user.profile.name", "Yara")

        assert d["user"]["profile"]["name"] == "Yara"
        assert isinstance(d["user"], dotdict)
        assert isinstance(d["user"]["profile"], dotdict)

    def test_set_path_with_list_index(self):
        """Test setting value in list using index in path."""
        d = dotdict({"items": [{}, {}]})
        d.set("items.0.name", "first")
        d.set("items.1.name", "second")

        assert d["items"][0]["name"] == "first"
        assert d["items"][1]["name"] == "second"

    def test_set_overwrites_existing_value(self):
        """Test setting path overwrites existing value."""
        d = dotdict({"user": {"name": "Zack"}})
        d.set("user.name", "Zoe")

        assert d["user"]["name"] == "Zoe"

    def test_set_creates_missing_intermediate_keys(self):
        """Test set creates all missing intermediate keys."""
        d = dotdict()
        d.set("a.b.c.d", "value")

        assert d["a"]["b"]["c"]["d"] == "value"

    def test_set_on_frozen_raises_error(self):
        """Test setting on frozen dotdict raises error."""
        d = dotdict(frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d.set("name", "Amy")

    def test_set_wraps_dict_values(self):
        """Test that set wraps dict values as dotdict."""
        d = dotdict()
        d.set("user", {"name": "Ben", "age": 40})

        assert isinstance(d["user"], dotdict)
        assert d["user"]["name"] == "Ben"

    def test_set_deep_nested_path(self):
        """Test setting deeply nested path."""
        d = dotdict()
        d.set("level1.level2.level3.level4.value", "deep")

        assert d.get("level1.level2.level3.level4.value") == "deep"


class TestUpdate:
    """Test suite for update() method."""

    def test_update_with_dict(self):
        """Test update with dictionary."""
        d = dotdict({"name": "Carl"})
        d.update({"age": 36, "city": "LA"})

        assert d["name"] == "Carl"
        assert d["age"] == 36
        assert d["city"] == "LA"

    def test_update_with_kwargs(self):
        """Test update with keyword arguments."""
        d = dotdict({"name": "Diana"})
        d.update(age=28, city="SF")

        assert d["age"] == 28
        assert d["city"] == "SF"

    def test_update_with_dict_and_kwargs(self):
        """Test update with both dict and kwargs."""
        d = dotdict({"name": "Evan"})
        d.update({"age": 32}, city="Boston")

        assert d["age"] == 32
        assert d["city"] == "Boston"

    def test_update_with_dotted_keys(self):
        """Test update with dotted keys creates nested structure."""
        d = dotdict()
        d.update({"user.name": "Fiona", "user.age": 29})

        assert d["user"]["name"] == "Fiona"
        assert d["user"]["age"] == 29

    def test_update_merges_nested_dotdict(self):
        """Test update merges with existing nested dotdict."""
        d = dotdict({"user": {"name": "George", "age": 45}})
        d.update({"user": {"city": "Miami"}})

        assert d["user"]["name"] == "George"
        assert d["user"]["age"] == 45
        assert d["user"]["city"] == "Miami"

    def test_update_overwrites_non_dict_value(self):
        """Test update overwrites non-dict values."""
        d = dotdict({"value": "old"})
        d.update({"value": "new"})

        assert d["value"] == "new"

    def test_update_on_frozen_raises_error(self):
        """Test update on frozen dotdict raises error."""
        d = dotdict({"name": "Helen"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen"):
            d.update({"age": 31})

    def test_update_too_many_args(self):
        """Test update with too many positional arguments raises TypeError."""
        d = dotdict()

        with pytest.raises(TypeError, match="expected at most 1 arguments"):
            d.update({"a": 1}, {"b": 2})

    def test_update_wraps_dict_values(self):
        """Test update wraps dict values as dotdict."""
        d = dotdict()
        d.update({"user": {"name": "Isla"}})

        assert isinstance(d["user"], dotdict)


class TestToDict:
    """Test suite for to_dict() method."""

    def test_to_dict_basic(self):
        """Test converting basic dotdict to regular dict."""
        d = dotdict({"name": "Jack", "age": 37})
        result = d.to_dict()

        assert isinstance(result, dict)
        assert not isinstance(result, dotdict)
        assert result == {"name": "Jack", "age": 37}

    def test_to_dict_nested(self):
        """Test converting nested dotdict to regular dict."""
        d = dotdict({"user": {"profile": {"name": "Karen"}}})
        result = d.to_dict()

        assert isinstance(result, dict)
        assert isinstance(result["user"], dict)
        assert not isinstance(result["user"], dotdict)
        assert isinstance(result["user"]["profile"], dict)
        assert result["user"]["profile"]["name"] == "Karen"

    def test_to_dict_with_list(self):
        """Test converting dotdict with lists to regular dict."""
        d = dotdict({"items": [{"name": "item1"}, {"name": "item2"}]})
        result = d.to_dict()

        assert isinstance(result["items"], list)
        assert isinstance(result["items"][0], dict)
        assert not isinstance(result["items"][0], dotdict)

    def test_to_dict_preserves_primitives(self):
        """Test to_dict preserves primitive values."""
        d = dotdict(
            {"string": "text", "number": 42, "float": 3.14, "bool": True, "none": None}
        )
        result = d.to_dict()

        assert result["string"] == "text"
        assert result["number"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None


class TestToJson:
    """Test suite for to_json() method."""

    def test_to_json_basic(self):
        """Test JSON serialization of basic dotdict."""
        d = dotdict({"name": "Larry", "age": 38})
        json_bytes = d.to_json()

        assert isinstance(json_bytes, bytes)
        decoded = msgspec.json.decode(json_bytes)
        assert decoded == {"name": "Larry", "age": 38}

    def test_to_json_nested(self):
        """Test JSON serialization of nested dotdict."""
        d = dotdict({"user": {"name": "Monica", "settings": {"theme": "dark"}}})
        json_bytes = d.to_json()
        decoded = msgspec.json.decode(json_bytes)

        assert decoded["user"]["name"] == "Monica"
        assert decoded["user"]["settings"]["theme"] == "dark"

    def test_to_json_with_list(self):
        """Test JSON serialization with lists."""
        d = dotdict({"items": [1, 2, 3], "objects": [{"key": "value"}]})
        json_bytes = d.to_json()
        decoded = msgspec.json.decode(json_bytes)

        assert decoded["items"] == [1, 2, 3]
        assert decoded["objects"][0]["key"] == "value"


class TestStringRepresentations:
    """Test suite for __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ output."""
        d = dotdict({"name": "Nancy", "age": 39})
        repr_str = repr(d)

        assert "dotdict" in repr_str
        assert "'name'" in repr_str
        assert "Nancy" in repr_str

    def test_str_basic(self):
        """Test __str__ output."""
        d = dotdict({"name": "Oscar", "age": 41})
        str_output = str(d)

        assert "name" in str_output
        assert "Oscar" in str_output
        assert "41" in str_output

    def test_repr_nested(self):
        """Test __repr__ with nested structure."""
        d = dotdict({"user": {"name": "Pam"}})
        repr_str = repr(d)

        assert "dotdict" in repr_str
        assert "user" in repr_str


class TestFrozenBehavior:
    """Test suite for frozen dotdict behavior."""

    def test_frozen_initialization(self):
        """Test frozen dotdict can be initialized."""
        d = dotdict({"name": "Quinn", "age": 34}, frozen=True)

        assert d["name"] == "Quinn"
        assert d["age"] == 34

    def test_frozen_prevents_setattr(self):
        """Test frozen prevents attribute assignment."""
        d = dotdict({"name": "Rita"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d.age = 30

    def test_frozen_prevents_setitem(self):
        """Test frozen prevents item assignment."""
        d = dotdict({"name": "Steve"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d["age"] = 35

    def test_frozen_prevents_delattr(self):
        """Test frozen prevents attribute deletion."""
        d = dotdict({"name": "Tara", "age": 42}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot delete from frozen dotdict"):
            del d.age

    def test_frozen_prevents_set(self):
        """Test frozen prevents set() method."""
        d = dotdict(frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d.set("name", "Uma")

    def test_frozen_prevents_update(self):
        """Test frozen prevents update() method."""
        d = dotdict({"name": "Vera"}, frozen=True)

        with pytest.raises(AttributeError, match="Cannot modify frozen"):
            d.update({"age": 28})

    def test_frozen_allows_reading(self):
        """Test frozen allows reading operations."""
        d = dotdict({"name": "Wade", "age": 36}, frozen=True)

        assert d.name == "Wade"
        assert d["age"] == 36
        assert d.get("name") == "Wade"

    def test_frozen_nested_dicts_are_frozen(self):
        """Test that nested dicts in frozen dotdict are also frozen."""
        d = dotdict({"user": {"name": "Xena"}}, frozen=True)

        assert d["user"]._frozen is True

        with pytest.raises(AttributeError, match="Cannot modify frozen dotdict"):
            d.user.age = 30


class TestEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_empty_path_get(self):
        """Test get with empty string path."""
        d = dotdict({"": "empty_key"})

        assert d.get("") == "empty_key"

    def test_empty_path_set(self):
        """Test set with empty string path."""
        d = dotdict()
        d.set("", "value")

        assert d[""] == "value"

    def test_nested_list_of_lists(self):
        """Test handling nested lists."""
        d = dotdict({"matrix": [[1, 2, 3], [4, 5, 6]]})

        assert d["matrix"][0][1] == 2
        assert d.get("matrix.0.1") == 2  # List indices in path work for multiple levels

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        d = dotdict({"key-with-dash": "value1", "key_with_underscore": "value2"})

        assert d["key-with-dash"] == "value1"
        assert d["key_with_underscore"] == "value2"

    def test_numeric_string_keys(self):
        """Test keys that are numeric strings."""
        d = dotdict({"123": "numeric_key"})

        assert d["123"] == "numeric_key"

    def test_overwrite_dict_with_primitive(self):
        """Test overwriting dict value with primitive."""
        d = dotdict({"user": {"name": "Yuri"}})
        d["user"] = "not_a_dict"

        assert d["user"] == "not_a_dict"
        assert not isinstance(d["user"], dotdict)

    def test_set_on_existing_list_index(self):
        """Test setting existing list index via set()."""
        d = dotdict({"items": [{"name": "old"}]})
        d.set("items.0.name", "new")

        assert d["items"][0]["name"] == "new"

    def test_large_nested_structure(self):
        """Test handling large nested structures."""
        d = dotdict()
        for i in range(10):
            d.set(f"level{i}.value", i)

        assert d.get("level5.value") == 5
        assert d.get("level9.value") == 9


class TestIterAndContains:
    """Test suite for __iter__ and __contains__."""

    def test_contains_existing_key(self):
        """Test that existing key is found with in operator."""
        d = dotdict({"name": "Alice", "age": 30})

        assert "name" in d
        assert "age" in d

    def test_contains_nonexistent_key(self):
        """Test that missing key is not found with in operator."""
        d = dotdict({"name": "Alice"})

        assert "missing" not in d

    def test_iter_yields_all_keys(self):
        """Test that iteration yields all keys."""
        d = dotdict({"a": 1, "b": 2, "c": 3})

        assert set(d) == {"a", "b", "c"}

    def test_iter_empty(self):
        """Test that empty dotdict yields no keys."""
        d = dotdict()

        assert list(d) == []

    def test_iter_nested_does_not_recurse(self):
        """Test that iteration only yields top-level keys."""
        d = dotdict({"user": {"name": "Bob"}, "score": 10})

        assert set(d) == {"user", "score"}


class TestHiddenKeys:
    """Test suite for hidden_keys behavior."""

    def test_hidden_key_get_returns_value(self):
        """Test that get() can read hidden keys when accessed directly."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert d.get("api_key") == "secret"

    def test_hidden_key_get_default_when_key_absent(self):
        """Test that get() returns default when hidden key does not exist in dict."""
        d = dotdict({"name": "Alice"}, hidden_keys=["api_key"])

        assert d.get("api_key") is None
        assert d.get("api_key", "fallback") == "fallback"

    def test_hidden_key_in_operator_returns_false(self):
        """Test that in operator returns False for hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert "api_key" not in d

    def test_visible_key_in_operator_unaffected(self):
        """Test that in operator still works normally for visible keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert "name" in d
        assert "missing" not in d

    def test_hidden_key_iter_excludes(self):
        """Test that iteration skips hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert list(d) == ["name"]

    def test_hidden_key_keys_excludes(self):
        """Test that keys() excludes hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert list(d.keys()) == ["name"]

    def test_hidden_key_values_excludes(self):
        """Test that values() excludes values of hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert list(d.values()) == ["Alice"]

    def test_hidden_key_items_excludes(self):
        """Test that items() excludes hidden key-value pairs."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert list(d.items()) == [("name", "Alice")]

    def test_hidden_key_to_dict_excludes(self):
        """Test that to_dict() excludes hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        result = d.to_dict()

        assert "api_key" not in result
        assert result == {"name": "Alice"}

    def test_hidden_key_to_json_excludes(self):
        """Test that to_json() excludes hidden keys."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        decoded = msgspec.json.decode(d.to_json())

        assert "api_key" not in decoded
        assert decoded == {"name": "Alice"}

    def test_hidden_key_repr_excludes(self):
        """Test that repr() excludes hidden keys and their values."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        r = repr(d)

        assert "api_key" not in r
        assert "secret" not in r
        assert "Alice" in r

    def test_hidden_key_str_excludes(self):
        """Test that str() excludes hidden keys and their values."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        s = str(d)

        assert "api_key" not in s
        assert "secret" not in s
        assert "Alice" in s

    def test_hidden_key_dot_access_works(self):
        """Test that dot notation can read hidden keys."""
        d = dotdict({"api_key": "secret"}, hidden_keys=["api_key"])

        assert d.api_key == "secret"

    def test_hidden_key_bracket_access_works(self):
        """Test that bracket notation can read hidden keys."""
        d = dotdict({"api_key": "secret"}, hidden_keys=["api_key"])

        assert d["api_key"] == "secret"

    def test_hidden_key_nested_path_via_get(self):
        """Test that get() traverses paths through hidden intermediate keys."""
        d = dotdict(
            {"secrets": {"token": "abc", "key": "xyz"}}, hidden_keys=["secrets"]
        )

        assert d.get("secrets.token") == "abc"
        assert d.get("secrets.key") == "xyz"

    def test_hidden_key_set_via_dot(self):
        """Test that dot notation can overwrite hidden keys."""
        d = dotdict({"api_key": "old"}, hidden_keys=["api_key"])

        d.api_key = "new"

        assert d.api_key == "new"

    def test_hidden_key_set_via_bracket(self):
        """Test that bracket notation can overwrite hidden keys."""
        d = dotdict({"api_key": "old"}, hidden_keys=["api_key"])

        d["api_key"] = "new"

        assert d["api_key"] == "new"

    def test_hidden_keys_multiple(self):
        """Test that multiple keys can be hidden simultaneously."""
        d = dotdict(
            {"api_key": "secret", "password": "pass123", "name": "Alice"},
            hidden_keys=["api_key", "password"],
        )

        assert "api_key" not in d
        assert "password" not in d
        assert "name" in d
        assert list(d.keys()) == ["name"]
        assert d.to_dict() == {"name": "Alice"}

    def test_hidden_key_len_includes_hidden(self):
        """Test that len() counts all stored keys, including hidden ones."""
        d = dotdict({"api_key": "secret", "name": "Alice"}, hidden_keys=["api_key"])

        assert len(d) == 2

    def test_hidden_key_propagates_to_nested_dicts(self):
        """Test that nested dicts created via wrap inherit hidden_keys."""
        d = dotdict(
            {"config": {"debug": True}},
            hidden_keys=["api_key"],
        )

        nested = d.config
        assert isinstance(nested, dotdict)
        assert "api_key" not in nested

    def test_visible_keys_fully_unaffected(self):
        """Test that hidden_keys does not change behaviour of visible keys."""
        d = dotdict(
            {"api_key": "secret", "name": "Alice", "age": 30},
            hidden_keys=["api_key"],
        )

        assert d.get("name") == "Alice"
        assert d.get("age") == 30
        assert "name" in d
        assert d.to_dict() == {"name": "Alice", "age": 30}
