from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

import msgspec

from msgflux.models.chat_extensions import validate_chat_speed
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction

if TYPE_CHECKING:
    from msgflux.chat_messages import ChatMessages
    from msgflux.tools.catalog import ToolCatalogView


def validate_reasoning_effort(reasoning_effort: str | None) -> None:
    """Validate the provider-independent request-level effort value."""
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
    ):
        raise TypeError("`reasoning_effort` must be a non-empty string or None")


class ChatCompletionModel:
    model_type = "chat_completion"

    _encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _decoders: dict[type, msgspec.json.Decoder] = {}

    def supports_reasoning_effort(self) -> bool:
        """Return whether request-level reasoning effort can be configured."""
        return False

    def set_speed(self, speed: str | None) -> ChatCompletionModel:
        """Set a provider-neutral request speed when supported by the client."""
        validate_chat_speed(speed)
        if speed is not None:
            provider = getattr(self, "provider", self.__class__.__name__)
            warnings.warn(
                f"Provider `{provider}` does not support `speed={speed!r}`; the "
                "setting was ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def set_reasoning_effort(self, reasoning_effort: str | None) -> ChatCompletionModel:
        """Set request-level reasoning effort when supported by the client."""
        validate_reasoning_effort(reasoning_effort)
        if reasoning_effort is not None:
            provider = getattr(self, "provider", self.__class__.__name__)
            warnings.warn(
                f"Provider `{provider}` does not support request-level "
                "`reasoning_effort`; the setting was ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def supports_native_compaction(self) -> bool:
        """Return whether this provider/API exposes a native compact operation."""
        return False

    @property
    def context_capacity(self) -> int | None:
        """Return the known model context window, or None when unknown."""
        explicit = getattr(self, "context_length", None)
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        profile = getattr(self, "profile", None)
        limits = getattr(profile, "limits", None)
        capacity = getattr(limits, "context", None)
        return capacity if isinstance(capacity, int) and capacity > 0 else None

    def count_context_tokens(
        self,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> ContextTokenEstimate:
        """Estimate request input tokens when no exact provider counter exists."""
        payload: Any = messages
        materialize = getattr(messages, "materialize_context", None)
        if callable(materialize):
            payload = materialize(
                provider=getattr(self, "provider", None),
                api_mode=getattr(self, "api_mode", None),
            )
        to_items = getattr(messages, "to_items", None)
        if payload is not messages:
            to_items = getattr(payload, "to_items", None)
        if callable(to_items):
            payload = to_items()
        try:
            encoded = msgspec.json.encode(
                {
                    "system_prompt": system_prompt,
                    "messages": payload,
                    "tools": (
                        tool_catalog.portable_schemas() if tool_catalog else None
                    ),
                }
            )
        except (TypeError, ValueError):
            encoded = repr((system_prompt, payload)).encode()
        # A conservative provider-independent estimate. Exact counters should
        # override this method rather than teaching Agent about tokenizers.
        return ContextTokenEstimate(
            input_tokens=max(1, (len(encoded) + 2) // 3),
            source="heuristic",
        )

    async def acount_context_tokens(
        self,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> ContextTokenEstimate:
        """Async counterpart to :meth:`count_context_tokens`."""
        return self.count_context_tokens(
            messages,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    def compact_context(
        self,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        native: bool = True,
    ) -> ModelCompaction:
        """Create a portable complete-summary view using this model."""
        _ = native
        response = self(
            messages=messages,
            system_prompt=self._compaction_system_prompt(system_prompt),
            stream=False,
        )
        return ModelCompaction(
            format="messages",
            items=[self._summary_message(response)],
            provider=getattr(self, "provider", None),
            api_mode=getattr(self, "api_mode", None),
            model_id=getattr(self, "model_id", None),
            usage=self._compaction_usage(response),
        )

    async def acompact_context(
        self,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        native: bool = True,
    ) -> ModelCompaction:
        """Async portable compaction fallback."""
        _ = native
        response = await self.acall(
            messages=messages,
            system_prompt=self._compaction_system_prompt(system_prompt),
            stream=False,
        )
        return ModelCompaction(
            format="messages",
            items=[self._summary_message(response)],
            provider=getattr(self, "provider", None),
            api_mode=getattr(self, "api_mode", None),
            model_id=getattr(self, "model_id", None),
            usage=self._compaction_usage(response),
        )

    @staticmethod
    def _compaction_system_prompt(system_prompt: str | None) -> str:
        original = (
            f"\n\nOriginal agent instructions:\n{system_prompt}"
            if system_prompt
            else ""
        )
        return (
            "Create a complete, compact continuation state for the conversation. "
            "Preserve decisions, constraints, unresolved work, identifiers, tool "
            "results, and facts needed by a later model. Do not answer the latest "
            "request and do not add facts. Return only the continuation summary."
            f"{original}"
        )

    @staticmethod
    def _summary_message(response: Any) -> dict[str, str]:
        summary = getattr(response, "data", None)
        if not isinstance(summary, str) or not summary.strip():
            raise TypeError("Compaction model must return a non-empty text response")
        return {
            "role": "system",
            "content": (
                f"<conversation_summary>\n{summary.strip()}\n</conversation_summary>"
            ),
        }

    @staticmethod
    def _compaction_usage(response: Any) -> dict[str, Any] | None:
        metadata = getattr(response, "metadata", None)
        usage = getattr(metadata, "usage", None)
        if usage is None and isinstance(metadata, Mapping):
            usage = metadata.get("usage")
        return dict(usage) if isinstance(usage, Mapping) else None

    def _get_decoder(self, schema: type) -> msgspec.json.Decoder:
        """Return a cached Decoder for *schema*, creating it on first use."""
        decoder = self._decoders.get(schema)
        if decoder is None:
            decoder = msgspec.json.Decoder(schema)
            self._decoders[schema] = decoder
        return decoder

    def warmup_system_prompt(
        self,
        *,
        system_prompt: str | None,
        tool_catalog: ToolCatalogView | None = None,
    ):
        """Warm provider prompt/tool-schema caches without producing useful output."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prompt warmup."
        )

    async def awarmup_system_prompt(
        self,
        *,
        system_prompt: str | None,
        tool_catalog: ToolCatalogView | None = None,
    ):
        """Async prompt warmup counterpart."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prompt warmup."
        )


class BatchedChatCompletionModel:
    model_type = "batched_chat_completion"


class ModerationModel:
    model_type = "moderation"


# Classifiers


class AudioClassifierModel:
    model_type = "audio_classifier"


class ImageClassifierModel:
    model_type = "image_classifier"


class VideoClassifierModel:
    model_type = "video_classifier"


class TabularClassifierModel:
    model_type = "tabular_classifier"


class TextClassifierModel:
    model_type = "text_classifier"


class ZeroShotImageClassifierModel:
    model_type = "zero_shot_image_classifier"


class ZeroShotTextClassifierModel:
    model_type = "zero_shot_text_classifier"


# Embedders


class AudioEmbedderModel:
    model_type = "audio_embedder"


class ImageEmbedderModel:
    model_type = "image_embedder"


class TextEmbedderModel:
    model_type = "text_embedder"


# Audio Gen


class TextToSpeechModel:
    model_type = "text_to_speech"


class AudioToAudioModel:
    model_type = "audio_to_audio"


class TextToMusicModel:
    model_type = "text_to_music"


class VideoTextToAudioModel:
    model_type = "video_text_to_audio"


# Image Gen


class TextToImageModel:
    model_type = "text_to_image"


class ImageTextToImageModel:
    model_type = "image_text_to_image"


class ImageToImageModel:
    model_type = "image_to_image"


# Text Gen


class AudioTextToTextModel:
    model_type = "audio_text_to_text"


class SpeechToTextModel:
    model_type = "speech_to_text"


class ImageToTextModel:
    model_type = "image_to_text"


class OCRModel:
    model_type = "ocr"


class TextTranslationModel:
    model_type = "text_translation"


class VideoTextToTextModel:
    model_type = "video_text_to_text"


# Video Gen


class ImageTextToVideoModel:  # VideoGenModel
    model_type = "image_text_to_video"


class TextToVideoModel:  # VideoGenModel
    model_type = "text_to_video"


class VideoTextToVideoModel:  # VideoGenModel
    model_type = "video_text_to_video"


# 3D


class ImageTo3DModel:
    model_type = "image_text_to_3d"


class ImageTextTo3DModel:
    model_type = "image_text_to_3d"


class TextTo3DModel:
    model_type = "text_to_3d"


# Others


class AnyToAnyModel:
    model_type = "any_to_any"


class DepthEstimationModel:
    model_type = "depth_estimation"


class ImageSegmenterModel:
    model_type = "image_segmenter"


class MaskGenModel:
    model_type = "mask_gen"


class ObjectDetectorModel:
    model_type = "object_detector"


class VADModel:
    model_type = "vad"


class TabularRegressorModel:
    model_type = "tabular_regressor"


class TextRerankerModel:
    model_type = "text_reranker"
