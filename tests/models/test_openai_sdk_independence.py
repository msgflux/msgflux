"""Regression tests for importing OpenAI models without the OpenAI SDK."""

from __future__ import annotations

import os
import subprocess
import sys


def test_openai_models_import_and_initialize_without_openai_sdk():
    script = r"""
import builtins

real_import = builtins.__import__

def reject_openai_sdk(name, *args, **kwargs):
    if name == "openai" or name.startswith("openai."):
        raise AssertionError(f"unexpected OpenAI SDK import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_openai_sdk

from msgflux.models.providers.openai import (
    OpenAIChatCompletion,
    OpenAIImageTextToImage,
    OpenAIModeration,
    OpenAISpeechToText,
    OpenAITextEmbedder,
    OpenAITextToImage,
    OpenAITextToSpeech,
)

models = (
    OpenAIChatCompletion(model_id="gpt-5.6-luna"),
    OpenAITextToSpeech(model_id="gpt-4o-mini-tts"),
    OpenAITextToImage(model_id="gpt-image-1-mini"),
    OpenAIImageTextToImage(model_id="gpt-image-1-mini"),
    OpenAISpeechToText(model_id="gpt-4o-mini-transcribe"),
    OpenAITextEmbedder(model_id="text-embedding-3-small"),
    OpenAIModeration(model_id="omni-moderation-latest"),
)

assert all(model.provider == "openai" for model in models)
"""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "test-key"

    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
