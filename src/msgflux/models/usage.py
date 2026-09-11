"""Canonical token usage returned by language-model providers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from msgflux.core.dotdict import dotdict


class UsageCodec:
    """Normalize provider usage payloads while retaining their native shape."""

    input_token_fields = (
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
        "prompt_eval_count",
    )
    output_token_fields = (
        "output_tokens",
        "completion_tokens",
        "candidates_token_count",
        "eval_count",
    )
    total_token_fields = ("total_tokens", "total_token_count")
    input_detail_fields = (
        "input_tokens_details",
        "input_token_details",
        "prompt_tokens_details",
    )
    output_detail_fields = ("output_tokens_details", "completion_tokens_details")

    def normalize(self, usage: Any) -> dotdict | None:
        raw = self._to_mapping(usage)
        if raw is None:
            return None

        input_tokens = self._first_int(raw, self.input_token_fields)
        output_tokens = self._first_int(raw, self.output_token_fields)
        total_tokens = self._first_int(raw, self.total_token_fields)
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

        input_details = self._first_mapping(raw, self.input_detail_fields)
        output_details = self._first_mapping(raw, self.output_detail_fields)
        cached_tokens = self._detail_int(
            raw,
            input_details,
            (
                "cached_tokens",
                "cache_read_input_tokens",
                "cached_content_token_count",
            ),
        )

        return dotdict(
            {
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "total_tokens": total_tokens or 0,
                "duration_seconds": self._first_int(raw, ("seconds",)) or 0,
                "cache_hit_percentage": self._cache_hit_percentage(
                    input_tokens,
                    cached_tokens,
                ),
                "input_tokens_details": dotdict(
                    {
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": self._detail_int(
                            raw,
                            input_details,
                            ("cache_write_tokens", "cache_creation_input_tokens"),
                        ),
                        "audio_tokens": self._detail_int(
                            raw, input_details, ("audio_tokens",)
                        ),
                        "video_tokens": self._detail_int(
                            raw, input_details, ("video_tokens",)
                        ),
                        "image_tokens": self._detail_int(
                            raw, input_details, ("image_tokens",)
                        ),
                        "text_tokens": self._detail_int(
                            raw, input_details, ("text_tokens",)
                        ),
                    }
                ),
                "output_tokens_details": dotdict(
                    {
                        "reasoning_tokens": self._detail_int(
                            raw,
                            output_details,
                            ("reasoning_tokens", "thoughts_token_count"),
                        ),
                        "audio_tokens": self._detail_int(
                            raw, output_details, ("audio_tokens",)
                        ),
                        "image_tokens": self._detail_int(
                            raw, output_details, ("image_tokens",)
                        ),
                        "text_tokens": self._detail_int(
                            raw, output_details, ("text_tokens",)
                        ),
                        "accepted_prediction_tokens": self._detail_int(
                            raw, output_details, ("accepted_prediction_tokens",)
                        ),
                        "rejected_prediction_tokens": self._detail_int(
                            raw, output_details, ("rejected_prediction_tokens",)
                        ),
                    }
                ),
                "cost": raw.get("cost"),
                "raw": deepcopy(dict(raw)),
            }
        )

    @staticmethod
    def _to_mapping(value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        return value if isinstance(value, Mapping) else None

    @classmethod
    def _first_int(
        cls, mapping: Mapping[str, Any], fields: Iterable[str]
    ) -> int | None:
        for field in fields:
            value = mapping.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    @staticmethod
    def _first_mapping(
        mapping: Mapping[str, Any], fields: Iterable[str]
    ) -> Mapping[str, Any]:
        for field in fields:
            value = mapping.get(field)
            if isinstance(value, Mapping):
                return value
        return {}

    @classmethod
    def _detail_int(
        cls,
        raw: Mapping[str, Any],
        details: Mapping[str, Any],
        fields: Iterable[str],
    ) -> int:
        value = cls._first_int(details, fields)
        if value is None:
            value = cls._first_int(raw, fields)
        return value or 0

    @staticmethod
    def _cache_hit_percentage(
        input_tokens: int | None,
        cached_tokens: int,
    ) -> float | None:
        """Return the cached share of input tokens as a percentage."""
        if (
            input_tokens is None
            or input_tokens <= 0
            or cached_tokens < 0
            or cached_tokens > input_tokens
        ):
            return None
        return cached_tokens / input_tokens * 100


default_usage_codec = UsageCodec()
