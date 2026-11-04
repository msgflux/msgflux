"""Execution-local coding policy for selectable model deployments."""

from __future__ import annotations

from dataclasses import replace

import msgspec

from msgflux.coding.config import CodingAgentOptions, resolve_coding_tools
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook
from msgflux.tools.builtin.workspace_tools import ReadFileTool


class CodingModelSelection(AgentExtension):
    """Pin configured deployment identities and project model-specific tools.

    Only public model/account references enter durable state. Live models and
    credentials remain owned by the host factory, outside checkpoints.
    """

    def __init__(
        self, options: CodingAgentOptions, models, *, accounts_root: str | None = None
    ):
        super().__init__("coding_models")
        self.options = options
        self.accounts_root = accounts_root
        self.models_by_name = dict(
            zip((item.model_name for item in options.models), models)
        )
        self.tool_names = frozenset((*options.tools_active, *options.tools_deferred))
        self.readers = {
            name: ReadFileTool(supports_vision=self._vision(model))
            for name, model in self.models_by_name.items()
        }

    @staticmethod
    def _vision(model):
        profile = getattr(model, "profile", None)
        modalities = getattr(profile, "modalities", None)
        return "image" in getattr(modalities, "input", ())

    def adapt_tools(self, tools):
        adapted = []
        for tool in tools:
            if isinstance(tool, ReadFileTool):
                reader = CodingGatewayReadTool(self, original=tool)
                reader.tool_config = dict(tool.tool_config)
                adapted.append(reader)
            else:
                adapted.append(tool)
        return adapted

    def hooks(self):
        return (
            Hook(event="transform_context", handler=self.bind),
            Hook(event="before_resume", handler=self.validate_resume),
            Hook(event="transform_tool_catalog", handler=self.project_tools),
        )

    def _selection(self, preference):
        name = preference or self.options.models[0].model_name
        if name not in self.models_by_name:
            raise ValueError(f"Unknown coding model alias {name!r}")
        return name

    def _binding(self, context):
        return {
            "model_name": self._selection(context.model_preference),
            "fallback": self.options.model_fallback,
            "accounts_root": self.accounts_root,
            "models": [
                {
                    "model_name": deployment.model_name,
                    "model": deployment.model,
                    "account_name": deployment.account_name
                    or context.scope.account_name,
                    "reasoning_effort": deployment.reasoning_effort
                    or self.options.reasoning_effort,
                }
                for deployment in self.options.models
                if self.options.model_fallback
                or deployment.model_name == self._selection(context.model_preference)
            ],
        }

    def _check_binding(self, context, *, resume=False):
        binding = self._binding(context)
        state = self.durable_state()
        saved = state.get("binding")
        if resume and saved is None:
            raise ValueError(
                "This checkpoint has no coding model binding; start a new run "
                "instead of changing a pending execution's model configuration"
            )
        if saved is not None and saved != binding:
            raise ValueError(
                "Coding model deployment or account configuration changed for "
                "this run; restore its configuration before resuming"
            )
        state["binding"] = binding
        return context

    def bind(self, context):
        return self._check_binding(context)

    def validate_resume(self, context):
        return self._check_binding(context, resume=True)

    def _selected_tools(self, name):
        deployment = next(
            item for item in self.options.models if item.model_name == name
        )
        selected = msgspec.structs.replace(
            self.options,
            model=deployment.model,
            models=(),
            model_fallback=False,
            tools_active=self.options.tools_active_selectors
            if self.options.tools_active_selectors is not None
            else self.options.tools_active,
            tools_deferred=self.options.tools_deferred_selectors
            if self.options.tools_deferred_selectors is not None
            else self.options.tools_deferred,
        )
        active, deferred = resolve_coding_tools(selected)
        return frozenset((*active, *deferred))

    def project_tools(self, context):
        name = self._selection(context.model_preference)
        allowed = (
            self.tool_names
            if self.options.model_fallback
            else self._selected_tools(name)
        )
        reader = self.reader(name)
        entries = tuple(
            msgspec.structs.replace(entry, usage_guidance=reader.default_usage_guidance)
            if entry.name == "read"
            else entry
            for entry in context.catalog.entries
            if entry.name not in self.tool_names or entry.name in allowed
        )
        return replace(
            context, catalog=msgspec.structs.replace(context.catalog, entries=entries)
        )

    def reader(self, name=None):
        if self.options.model_fallback:
            # A provider may fail after dispatch. Only publish image attachments
            # that every fallback deployment can accept.
            if not all(reader.supports_vision for reader in self.readers.values()):
                return next(
                    reader
                    for reader in self.readers.values()
                    if not reader.supports_vision
                )
        if name is None:
            binding = self.durable_state().get("binding")
            if (
                not isinstance(binding, dict)
                or binding.get("model_name") not in self.readers
            ):
                raise ValueError(
                    "Coding reads require a pinned model binding for this run"
                )
            name = binding["model_name"]
        return self.readers[name]


class CodingGatewayReadTool(ReadFileTool):
    """Use the selected deployment's reader without mutating a shared tool."""

    def __init__(
        self, selection: CodingModelSelection, *, original: ReadFileTool | None = None
    ):
        original = original or ReadFileTool()
        flags = {reader.supports_vision for reader in selection.readers.values()}
        super().__init__(
            supports_vision=any(flags),
            max_image_bytes=original.max_image_bytes,
            max_text_bytes=original.max_text_bytes,
        )
        self.selection = selection
        self._readers = {
            flag: ReadFileTool(
                supports_vision=flag,
                max_image_bytes=self.max_image_bytes,
                max_text_bytes=self.max_text_bytes,
            )
            for flag in flags
        }

    def _selected_reader(self):
        return self._readers[self.selection.reader().supports_vision]

    def _read_options(self, path, offset, limit):
        return self._selected_reader()._read_options(path, offset, limit)

    def _result(self, path, data, handle):
        return self._selected_reader()._result(path, data, handle)
