from os import getenv
from typing import Any, Dict, Mapping, Optional

from msgflux.models.profiles import get_model_profile
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseMsgFlux:
    """Configurations to use msgFlux server models."""

    provider: str = "msgflux"

    def _get_base_url(self):
        base_url = getenv("MSGFLUX_BASE_URL", "http://127.0.0.1:8010/v1")
        if base_url is None:
            raise ValueError("Please set `MSGFLUX_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        return getenv("MSGFLUX_API_KEY", "msgflux-local")

    @property
    def profile(self):
        """Get model profile from registry.

        Returns:
            ModelProfile if found, None otherwise
        """
        return get_model_profile(self.model_id, provider_id=self.provider)


@register_model
class MsgFluxChatCompletion(_BaseMsgFlux, OpenAIChatCompletion):
    """msgFlux server Chat Completion."""

    def __init__(
        self,
        model_id: str,
        *,
        run_config: Optional[Mapping[str, Any]] = None,
        variables: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ):
        legacy_variables = kwargs.pop("vars", None)
        if variables is not None and legacy_variables is not None:
            raise ValueError("Use either `variables` or `vars`, not both.")

        selected_variables = variables if variables is not None else legacy_variables
        self.run_config = _merge_run_config(
            run_config,
            {"vars": selected_variables} if selected_variables is not None else None,
        )
        super().__init__(model_id=model_id, **kwargs)

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._adapt_params(params)
        extra_body = dict(params.get("extra_body") or {})
        request_run_config = extra_body.pop("run_config", None)
        run_config = _merge_run_config(self.run_config, request_run_config)

        if run_config:
            extra_body["run_config"] = run_config

        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)
        return params


def _merge_run_config(
    base: Optional[Mapping[str, Any]],
    update: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(update or {}).items():
        if key in {"vars", "kwargs"}:
            merged[key] = {
                **dict(merged.get(key) or {}),
                **dict(value or {}),
            }
        else:
            merged[key] = value
    return merged
