# ruff: noqa: A002

"""Agent coordination for optional, append-only context compaction."""

from __future__ import annotations

from typing import Any, Mapping

from msgflux._private.response_metadata import minimal_usage_metadata
from msgflux.chat_messages import ChatMessages
from msgflux.models.gateway import ModelGateway
from msgflux.nn.extensions.compaction import CONTEXT_COMPACTION_CAPABILITY
from msgflux.nn.hooks.events import BeforeCompaction
from msgflux.nn.modules.agent.context import _require_lifecycle_payload
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.events import EventType, emit_event


class AgentCompactionMixin:
    """Coordinate Model-owned compaction with Agent policy and durability."""

    def _maybe_compact_model_context(
        self,
        messages: Any,
        request_messages: Any,
        *,
        system_prompt: str | None,
        tool_catalog: Any,
        model_preference: str | None,
        vars: Mapping[str, Any],
        scope: ExecutionScope,
    ) -> bool:
        if (
            not isinstance(messages, ChatMessages)
            or not self.has_extension_capability(CONTEXT_COMPACTION_CAPABILITY)
            or not self.has_lifecycle_hooks("before_compaction")
        ):
            return False
        boundary = messages.latest_completed_turn_boundary()
        if boundary is None or self._boundary_is_already_compacted(messages, boundary):
            return False

        model = self.model
        estimate = self._count_context_tokens(
            model,
            request_messages,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
            model_preference=model_preference,
        )
        decision = self._run_lifecycle_hooks(
            "before_compaction",
            BeforeCompaction(
                scope=scope,
                vars=vars,
                messages=messages,
                estimated_input_tokens=estimate.input_tokens,
                estimate_source=estimate.source,
                context_capacity=getattr(model, "context_capacity", None),
                compacted_through_item_id=boundary["item_id"],
            ),
        )
        decision = _require_lifecycle_payload(
            "before_compaction", decision, BeforeCompaction
        )
        if decision.action != "compact":
            return False

        event_data = self._compaction_event_data(decision)
        emit_event(EventType.COMPACTION_START, event_data, scope=scope)
        try:
            compacted = self._compact_context(
                model,
                messages.through_item(decision.compacted_through_item_id),
                system_prompt=system_prompt,
                model_preference=model_preference,
            )
            compacted_usage = minimal_usage_metadata({"usage": compacted.usage})
            operation = messages.add_compaction(
                compacted_through_item_id=decision.compacted_through_item_id,
                views=[compacted.to_view()],
                reason=decision.reason,
                metadata=self._compaction_metadata(
                    compacted,
                    decision,
                    compacted_usage,
                ),
            )
            self._checkpoint_save(messages, vars, status="running")
        except Exception as error:
            emit_event(
                EventType.COMPACTION_END,
                {**event_data, "status": "failed", "error": str(error)},
                scope=scope,
            )
            raise
        emit_event(
            EventType.COMPACTION_END,
            self._compact_mapping(
                {
                    **event_data,
                    "status": "completed",
                    "format": compacted.format,
                    "model_id": compacted.model_id,
                    "operation_id": operation.get("item_id"),
                    "usage": compacted_usage,
                }
            ),
            scope=scope,
        )
        return True

    async def _amaybe_compact_model_context(
        self,
        messages: Any,
        request_messages: Any,
        *,
        system_prompt: str | None,
        tool_catalog: Any,
        model_preference: str | None,
        vars: Mapping[str, Any],
        scope: ExecutionScope,
    ) -> bool:
        if (
            not isinstance(messages, ChatMessages)
            or not self.has_extension_capability(CONTEXT_COMPACTION_CAPABILITY)
            or not self.has_lifecycle_hooks("before_compaction")
        ):
            return False
        boundary = messages.latest_completed_turn_boundary()
        if boundary is None or self._boundary_is_already_compacted(messages, boundary):
            return False

        model = self.model
        estimate = await await_with_abort(
            self._acount_context_tokens(
                model,
                request_messages,
                system_prompt=system_prompt,
                tool_catalog=tool_catalog,
                model_preference=model_preference,
            ),
            scope.abort_signal,
        )
        decision = await self._arun_lifecycle_hooks(
            "before_compaction",
            BeforeCompaction(
                scope=scope,
                vars=vars,
                messages=messages,
                estimated_input_tokens=estimate.input_tokens,
                estimate_source=estimate.source,
                context_capacity=getattr(model, "context_capacity", None),
                compacted_through_item_id=boundary["item_id"],
            ),
        )
        decision = _require_lifecycle_payload(
            "before_compaction", decision, BeforeCompaction
        )
        if decision.action != "compact":
            return False

        event_data = self._compaction_event_data(decision)
        emit_event(EventType.COMPACTION_START, event_data, scope=scope)
        try:
            compacted = await await_with_abort(
                self._acompact_context(
                    model,
                    messages.through_item(decision.compacted_through_item_id),
                    system_prompt=system_prompt,
                    model_preference=model_preference,
                ),
                scope.abort_signal,
            )
            compacted_usage = minimal_usage_metadata({"usage": compacted.usage})
            operation = messages.add_compaction(
                compacted_through_item_id=decision.compacted_through_item_id,
                views=[compacted.to_view()],
                reason=decision.reason,
                metadata=self._compaction_metadata(
                    compacted,
                    decision,
                    compacted_usage,
                ),
            )
            await self._acheckpoint_save(messages, vars, status="running")
        except Exception as error:
            emit_event(
                EventType.COMPACTION_END,
                {**event_data, "status": "failed", "error": str(error)},
                scope=scope,
            )
            raise
        emit_event(
            EventType.COMPACTION_END,
            self._compact_mapping(
                {
                    **event_data,
                    "status": "completed",
                    "format": compacted.format,
                    "model_id": compacted.model_id,
                    "operation_id": operation.get("item_id"),
                    "usage": compacted_usage,
                }
            ),
            scope=scope,
        )
        return True

    @staticmethod
    def _boundary_is_already_compacted(
        messages: ChatMessages,
        boundary: Mapping[str, Any],
    ) -> bool:
        latest = messages.latest_compaction()
        return latest is not None and latest.get(
            "compacted_through_item_id"
        ) == boundary.get("item_id")

    @staticmethod
    def _compaction_event_data(decision: BeforeCompaction) -> dict[str, Any]:
        return AgentCompactionMixin._compact_mapping(
            {
                "reason": decision.reason,
                "input_tokens_before": decision.estimated_input_tokens,
                "estimate_source": decision.estimate_source,
                "context_capacity": decision.context_capacity,
                "trigger_tokens": decision.trigger_tokens,
                "compacted_through_item_id": decision.compacted_through_item_id,
            }
        )

    @staticmethod
    def _compaction_metadata(compacted, decision, usage) -> dict[str, Any]:
        return AgentCompactionMixin._compact_mapping(
            {
                "provider": compacted.provider,
                "model_id": compacted.model_id,
                "api_mode": compacted.api_mode,
                "input_tokens_before": decision.estimated_input_tokens,
                "estimate_source": decision.estimate_source,
                "usage": usage,
            }
        )

    @staticmethod
    def _compact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _count_context_tokens(
        model,
        messages,
        *,
        system_prompt: str | None,
        tool_catalog: Any,
        model_preference: str | None,
    ):
        kwargs = {
            "messages": messages,
            "system_prompt": system_prompt,
            "tool_catalog": tool_catalog,
        }
        if isinstance(model, ModelGateway):
            kwargs["model_preference"] = model_preference
        return model.count_context_tokens(**kwargs)

    @staticmethod
    async def _acount_context_tokens(
        model,
        messages,
        *,
        system_prompt: str | None,
        tool_catalog: Any,
        model_preference: str | None,
    ):
        kwargs = {
            "messages": messages,
            "system_prompt": system_prompt,
            "tool_catalog": tool_catalog,
        }
        if isinstance(model, ModelGateway):
            kwargs["model_preference"] = model_preference
        return await model.acount_context_tokens(**kwargs)

    @staticmethod
    def _compact_context(
        model,
        messages,
        *,
        system_prompt: str | None,
        model_preference: str | None,
    ):
        kwargs = {"messages": messages, "system_prompt": system_prompt}
        if isinstance(model, ModelGateway):
            kwargs["model_preference"] = model_preference
        return model.compact_context(**kwargs)

    @staticmethod
    async def _acompact_context(
        model,
        messages,
        *,
        system_prompt: str | None,
        model_preference: str | None,
    ):
        kwargs = {"messages": messages, "system_prompt": system_prompt}
        if isinstance(model, ModelGateway):
            kwargs["model_preference"] = model_preference
        return await model.acompact_context(**kwargs)
