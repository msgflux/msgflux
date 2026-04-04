"""Tests for msgflux.utils.common module."""

from typing import Any, Literal, Optional, Set, Tuple, Union

from msgflux.utils.common import has_format_placeholder, is_jinja_template, type_mapping


def test_type_mapping_contains_basic_types():
    """Test that type_mapping contains all expected basic types."""
    assert "str" in type_mapping
    assert "int" in type_mapping
    assert "float" in type_mapping
    assert "bool" in type_mapping
    assert "list" in type_mapping
    assert "dict" in type_mapping


def test_type_mapping_string_types():
    """Test string type mappings."""
    assert type_mapping["str"] is str
    assert type_mapping["string"] is str


def test_type_mapping_synonyms():
    """Test that synonym mappings point to same types."""
    assert type_mapping["str"] is type_mapping["string"]
    assert type_mapping["int"] is type_mapping["integer"]
    assert type_mapping["float"] is type_mapping["number"]
    assert type_mapping["bool"] is type_mapping["boolean"]
    assert type_mapping["list"] is type_mapping["array"]
    assert type_mapping["dict"] is type_mapping["object"]


class TestIsJinjaTemplate:
    """Tests for is_jinja_template."""

    def test_variable_syntax(self):
        assert is_jinja_template("Hello {{ name }}!") is True

    def test_block_syntax(self):
        assert is_jinja_template("{% if x %}yes{% endif %}") is True

    def test_comment_syntax(self):
        assert is_jinja_template("{# this is a comment #}") is True

    def test_signature_generated_template(self):
        assert is_jinja_template("<ticket>{{ ticket }}</ticket>") is True

    def test_python_format_placeholder(self):
        assert is_jinja_template("Hello {}!") is False

    def test_named_python_placeholder(self):
        assert is_jinja_template("Hello {name}!") is False

    def test_plain_text(self):
        assert is_jinja_template("Hello world") is False

    def test_empty_string(self):
        assert is_jinja_template("") is False


class TestHasFormatPlaceholder:
    """Tests for has_format_placeholder."""

    def test_empty_placeholder(self):
        assert has_format_placeholder("Hello {}!") is True

    def test_numbered_placeholder(self):
        assert has_format_placeholder("Hello {0}!") is True

    def test_numbered_placeholder_high_index(self):
        assert has_format_placeholder("{0} and {1}") is True

    def test_jinja_variable_only(self):
        assert has_format_placeholder("Hello {{ name }}!") is False

    def test_named_python_placeholder(self):
        assert has_format_placeholder("Hello {name}!") is False

    def test_plain_text(self):
        assert has_format_placeholder("Hello world") is False

    def test_empty_string(self):
        assert has_format_placeholder("") is False

    def test_mixed_jinja_and_format(self):
        assert has_format_placeholder("{% if x %}{{ x }}{% endif %}Task: {}") is True
