"""Public API regression tests for the v1 database integration removal."""

import importlib

import pytest

import msgflux as mf


@pytest.mark.parametrize("name", ["DB", "response_cache"])
def test_database_exports_are_removed(name):
    assert name not in mf.__all__
    assert name not in dir(mf)
    with pytest.raises(AttributeError):
        getattr(mf, name)


@pytest.mark.parametrize("module", ["msgflux.data.dbs", "msgflux.cache"])
def test_database_package_is_removed(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_supported_data_and_model_entrypoints_still_import():
    from msgflux.data.parsers import Parser
    from msgflux.data.retrievers import Retriever
    from msgflux.models import Model

    assert mf.Parser is Parser
    assert mf.Retriever is Retriever
    assert mf.Model is Model

    assert Parser.csv("csv").provider == "csv"
    assert Retriever.lexical("bm25").provider == "bm25"
    assert callable(Model.chat_completion)
