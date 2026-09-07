# ruff: noqa: A001, A002

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)

from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.data.types import Audio, File, Image, Video
from msgflux.runtime.context import (
    ExecutionScope,
)
from msgflux.tools.catalog import ToolCatalogEntry, ToolChoice
from msgflux.utils.chat import ChatBlock
from msgflux.utils.common import has_format_placeholder, is_jinja_template
from msgflux.utils.console import cprint

if TYPE_CHECKING:
    pass
from msgflux.nn.modules.agent.context import (
    _RESERVED_KWARGS,
    _UNSET,
    ToolFilter,
    ToolFilterValue,
)


class AgentInputMixin:
    """Task rendering, multimodal preparation, filtering, and input validation."""

    def _prepare_inputs(  # noqa: C901
        self,
        message: Optional[Union[str, Message, Mapping[str, Any]]] = None,
        *,
        start_turn: bool = True,
        **kwargs,
    ) -> Mapping[str, Any]:
        """Prepare model input in ChatML format and execution params."""
        # Extract reserved kwargs
        task = kwargs.pop("task", _UNSET)
        vars = kwargs.pop("vars", {})
        messages = kwargs.pop("messages", None)
        model_preference = kwargs.pop("model_preference", None)
        tool_filter = kwargs.pop("tool_filter", None)
        scope = kwargs.pop("scope", None)
        if scope is not None and not isinstance(scope, ExecutionScope):
            raise TypeError(
                f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
            )
        kwargs.pop("tool_call_id", None)

        # Get remaining kwargs (potential task inputs)
        remaining_kwargs = {
            k: v for k, v in kwargs.items() if k not in _RESERVED_KWARGS
        }

        is_message_envelope = isinstance(message, dotdict)
        is_direct_message = message is not None and not is_message_envelope

        if task is not _UNSET and remaining_kwargs:
            raise ValueError(
                f"Cannot pass both 'task' and named task arguments. "
                f"Received task={type(task).__name__} and "
                f"kwargs={list(remaining_kwargs.keys())}. "
                f"Use either agent(task=...) or agent(key1=value1, key2=value2)"
            )

        # Handle explicit task argument and named task arguments
        if task is not _UNSET:
            if is_direct_message:
                raise ValueError(
                    f"Cannot pass both 'message' and 'task'. "
                    f"Received message={type(message).__name__} and "
                    f"task={type(task).__name__}. "
                    f"Use either agent(message) or agent(task=...)"
                )
        elif remaining_kwargs:
            if is_direct_message:
                raise ValueError(
                    f"Cannot pass both 'message' argument and named task arguments. "
                    f"Received message={type(message).__name__} and "
                    f"kwargs={list(remaining_kwargs.keys())}. "
                    f"Use either agent(message) or agent(key1=value1, key2=value2)"
                )
            task = remaining_kwargs
            for key in remaining_kwargs:
                kwargs.pop(key)
        elif not is_message_envelope:
            task = message

        # Extract vars from Message if not provided
        if not vars and isinstance(message, dotdict) and self.vars is not None:
            vars = message.get(self.vars, {})

        # Extract messages from Message if not provided
        if (
            messages is None
            and isinstance(message, dotdict)
            and self.messages is not None
        ):
            messages = self._get_content_from_message(self.messages, message)

        validation_inputs = self._get_validation_inputs(message, task, vars)
        if validation_inputs is not None:
            self._validate_inputs(validation_inputs)

        (
            messages,
            effective_scope,
            effective_thread_id,
            effective_run_id,
        ) = self._prepare_messages_scope(messages=messages, scope=scope)
        # A brand-new run in an existing thread continues from the latest
        # checkpointed messages, but keeps vars from the current call.
        messages, vars, model_preference = self._continue_thread_from_checkpoint(
            messages=messages,
            vars=vars,
            model_preference=model_preference,
            thread_id=effective_thread_id,
            run_id=effective_run_id,
        )

        task_context = self._context_manager(message, vars=vars, **kwargs)
        content = self._render_task(message, task=task, vars=vars, **kwargs)

        if content is None and not messages:
            raise ValueError(
                "No task input provided. Expected one of:\n"
                "  - agent('your text')\n"
                "  - agent({'key': 'value'})\n"
                "  - agent(message=Message(...))\n"
                "  - agent(task=...)\n"
                "  - agent(param1=..., param2=...)"
            )

        if isinstance(messages, ChatMessages):
            messages.configure_thread(
                thread_id=effective_thread_id,
                namespace=self.get_module_name(),
            )
            if start_turn:
                self._start_chat_turn_if_needed(
                    messages=messages,
                    turn_id=effective_run_id,
                )
            if content is not None:
                user_item = {"role": "user", "content": content}
                if task_context:
                    user_item["metadata"] = {"task_context": task_context}
                messages.append(user_item)
        elif content is not None:
            user_item = ChatBlock.user(content)
            if task_context:
                user_item["metadata"] = {"task_context": task_context}
            chat_content = [user_item]
            if messages is None:
                messages = chat_content
            else:
                messages.extend(chat_content)

        if model_preference is None and isinstance(message, dotdict):
            model_preference = self.get_model_preference_from_message(message)

        # Runtime kwargs take precedence over message_fields mappings.
        if tool_filter is None and isinstance(message, dotdict):
            tool_filter = self._get_tool_filter_from_message(message)

        return {
            "messages": messages,
            "model_preference": model_preference,
            "tool_filter": tool_filter,
            "vars": vars,
            "scope": effective_scope,
        }

    async def _aprepare_inputs(  # noqa: C901
        self,
        message: Optional[Union[str, Message, Mapping[str, Any]]] = None,
        *,
        start_turn: bool = True,
        **kwargs,
    ) -> Mapping[str, Any]:
        """Async version of _prepare_inputs.
        Prepare model input in ChatML format and execution params.
        """
        # Extract reserved kwargs
        task = kwargs.pop("task", _UNSET)
        vars = kwargs.pop("vars", {})
        messages = kwargs.pop("messages", None)
        model_preference = kwargs.pop("model_preference", None)
        tool_filter = kwargs.pop("tool_filter", None)
        scope = kwargs.pop("scope", None)
        if scope is not None and not isinstance(scope, ExecutionScope):
            raise TypeError(
                f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
            )
        kwargs.pop("tool_call_id", None)

        # Get remaining kwargs (potential task inputs)
        remaining_kwargs = {
            k: v for k, v in kwargs.items() if k not in _RESERVED_KWARGS
        }

        is_message_envelope = isinstance(message, dotdict)
        is_direct_message = message is not None and not is_message_envelope

        if task is not _UNSET and remaining_kwargs:
            raise ValueError(
                f"Cannot pass both 'task' and named task arguments. "
                f"Received task={type(task).__name__} and "
                f"kwargs={list(remaining_kwargs.keys())}. "
                f"Use either agent(task=...) or agent(key1=value1, key2=value2)"
            )

        # Handle explicit task argument and named task arguments
        if task is not _UNSET:
            if is_direct_message:
                raise ValueError(
                    f"Cannot pass both 'message' and 'task'. "
                    f"Received message={type(message).__name__} and "
                    f"task={type(task).__name__}. "
                    f"Use either agent(message) or agent(task=...)"
                )
        elif remaining_kwargs:
            if is_direct_message:
                raise ValueError(
                    f"Cannot pass both 'message' argument and named task arguments. "
                    f"Received message={type(message).__name__} and "
                    f"kwargs={list(remaining_kwargs.keys())}. "
                    f"Use either agent(message) or agent(key1=value1, key2=value2)"
                )
            task = remaining_kwargs
            for key in remaining_kwargs:
                kwargs.pop(key)
        elif not is_message_envelope:
            task = message

        # Extract vars from Message if not provided
        if not vars and isinstance(message, dotdict) and self.vars is not None:
            vars = message.get(self.vars, {})

        # Extract messages from Message if not provided
        if (
            messages is None
            and isinstance(message, dotdict)
            and self.messages is not None
        ):
            messages = self._get_content_from_message(self.messages, message)

        validation_inputs = self._get_validation_inputs(message, task, vars)
        if validation_inputs is not None:
            self._validate_inputs(validation_inputs)

        (
            messages,
            effective_scope,
            effective_thread_id,
            effective_run_id,
        ) = self._prepare_messages_scope(messages=messages, scope=scope)
        # A brand-new run in an existing thread continues from the latest
        # checkpointed messages, but keeps vars from the current call.
        messages, vars, model_preference = await self._acontinue_thread_from_checkpoint(
            messages=messages,
            vars=vars,
            model_preference=model_preference,
            thread_id=effective_thread_id,
            run_id=effective_run_id,
        )

        task_context = self._context_manager(message, vars=vars, **kwargs)
        content = await self._arender_task(message, task=task, vars=vars, **kwargs)

        if content is None and not messages:
            raise ValueError(
                "No task input provided. Expected one of:\n"
                "  - agent('your text')\n"
                "  - agent({'key': 'value'})\n"
                "  - agent(message=Message(...))\n"
                "  - agent(task=...)\n"
                "  - agent(param1=..., param2=...)"
            )

        if isinstance(messages, ChatMessages):
            messages.configure_thread(
                thread_id=effective_thread_id,
                namespace=self.get_module_name(),
            )
            if start_turn:
                self._start_chat_turn_if_needed(
                    messages=messages,
                    turn_id=effective_run_id,
                )
            if content is not None:
                user_item = {"role": "user", "content": content}
                if task_context:
                    user_item["metadata"] = {"task_context": task_context}
                messages.append(user_item)
        elif content is not None:
            user_item = ChatBlock.user(content)
            if task_context:
                user_item["metadata"] = {"task_context": task_context}
            chat_content = [user_item]
            if messages is None:
                messages = chat_content
            else:
                messages.extend(chat_content)

        if model_preference is None and isinstance(message, dotdict):
            model_preference = self.get_model_preference_from_message(message)

        # Runtime kwargs take precedence over message_fields mappings.
        if tool_filter is None and isinstance(message, dotdict):
            tool_filter = self._get_tool_filter_from_message(message)

        return {
            "messages": messages,
            "model_preference": model_preference,
            "tool_filter": tool_filter,
            "vars": vars,
            "scope": effective_scope,
        }

    def _render_task(  # noqa: C901
        self,
        message: Union[str, Message, Mapping[str, Any]],
        vars: Mapping[str, Any],
        task: Any = _UNSET,
        **kwargs,
    ) -> Optional[Union[str, Mapping[str, Any]]]:
        if task is _UNSET:
            if isinstance(message, dotdict):
                task = self._extract_message_values(self.task, message)
            else:
                task = message

        if task is None and self.templates.get("task") is None:
            return None

        if self.templates.get("task"):
            if task:
                if isinstance(task, str):
                    task_template = self.templates["task"]
                    if is_jinja_template(task_template) and not has_format_placeholder(
                        task_template
                    ):
                        error_message = (
                            f"[{self.name}] task_template uses Jinja2 variables but "
                            "'task' was passed as a plain string. "
                            "Pass 'task' as a dict with the required variable names "
                            "or use message_fields to map from the message."
                        )
                        raise ValueError(error_message)
                    pre_task = self._format_task_template(vars)
                    task_content = self._format_template(task, pre_task)
                elif isinstance(task, Mapping):
                    task_data = dotdict(task)
                    task_data.update(vars)
                    task_content = self._format_task_template(task_data)
                else:
                    task_content = str(task)
            # It's possible to use `task_template` as the default task message
            # if no `task` is selected. This can be useful for multimodal
            # models that require a text message to be sent along with the data
            elif vars:
                task_content = self._format_task_template(vars)
            else:
                task_content = self.templates.get("task")
        else:
            task_content = task
            if isinstance(task_content, Mapping):
                raise ValueError(
                    "Dict task requires a 'task' template to be configured. "
                    "Pass a string task or configure templates['task']."
                )
            if task_content is not None and not isinstance(task_content, str):
                task_content = str(task_content)

        content = task_content.strip()

        multimodal_content = self._render_task_multimodal(message, **kwargs)
        if multimodal_content:
            multimodal_content.append(ChatBlock.text(content))
            return multimodal_content
        return content

    async def _arender_task(  # noqa: C901
        self,
        message: Union[str, Message, Mapping[str, Any]],
        vars: Mapping[str, Any],
        task: Any = _UNSET,
        **kwargs,
    ) -> Optional[Union[str, Mapping[str, Any]]]:
        """Async version of _render_task."""
        if task is _UNSET:
            if isinstance(message, dotdict):
                task = self._extract_message_values(self.task, message)
            else:
                task = message

        if task is None and self.templates.get("task") is None:
            return None

        if self.templates.get("task"):
            if task:
                if isinstance(task, str):
                    task_template = self.templates["task"]
                    if is_jinja_template(task_template) and not has_format_placeholder(
                        task_template
                    ):
                        error_message = (
                            f"[{self.name}] task_template uses Jinja2 variables but "
                            "'task' was passed as a plain string. "
                            "Pass 'task' as a dict with the required variable names "
                            "or use message_fields to map from the message."
                        )
                        raise ValueError(error_message)
                    pre_task = self._format_task_template(vars)
                    task_content = self._format_template(task, pre_task)
                elif isinstance(task, Mapping):
                    task_data = dotdict(task)
                    task_data.update(vars)
                    task_content = self._format_task_template(task_data)
                else:
                    task_content = str(task)
            # It's possible to use `task_template` as the default task message
            # if no `task` is selected. This can be useful for multimodal
            # models that require a text message to be sent along with the data
            elif vars:
                task_content = self._format_task_template(vars)
            else:
                task_content = self.templates.get("task")
        else:
            task_content = task
            if isinstance(task_content, Mapping):
                raise ValueError(
                    "Dict task requires a 'task' template to be configured. "
                    "Pass a string task or configure templates['task']."
                )
            if task_content is not None and not isinstance(task_content, str):
                task_content = str(task_content)

        content = task_content.strip()

        multimodal_content = await self._arender_task_multimodal(message, **kwargs)
        if multimodal_content:
            multimodal_content.append(ChatBlock.text(content))
            return multimodal_content
        return content

    def _context_manager(  # noqa: C901
        self,
        message: Union[str, Message, Mapping[str, Any]],
        vars: Mapping[str, Any],
        **kwargs,
    ) -> Optional[str]:
        """Mount context."""
        context_content = ""

        if self.context_cache:  # Fixed Context Cache
            context_content += self.context_cache

        context = None
        runtime_context = kwargs.pop("task_context", None)
        if runtime_context is not None:
            context = runtime_context
        elif isinstance(message, dotdict):
            context = self._extract_message_values(self.task_context, message)

        if context is not None:
            if self.templates.get("task_context"):
                if isinstance(context, Mapping):
                    context_vars = dict(context)
                    context_vars.update(vars)
                    msg_context = self._format_template(
                        context_vars, self.templates.get("task_context")
                    )
                else:
                    pre_msg_context = self._format_template(
                        vars, self.templates.get("task_context")
                    )
                    msg_context = self._format_template(context, pre_msg_context)
            elif isinstance(context, str):
                msg_context = context
            elif isinstance(context, list):
                msg_context = " ".join(str(v) for v in context if v is not None)
            elif isinstance(context, Mapping):
                msg_context = "\n".join(
                    f"{k}: {v if not isinstance(v, list) else ', '.join(v)}"
                    for k, v in context.items()
                )
            context_content += msg_context

        if context_content:
            if vars:
                context_content = self._format_template(vars, context_content)
            return context_content
        return None

    # --- Multimodal Inputs ---

    def _render_task_multimodal(
        self, message: Union[str, Message, Mapping[str, Any]], **kwargs
    ) -> Optional[List[Mapping[str, Any]]]:
        """Processes multimodal inputs (image, audio, video, file) via kwargs or
        message.
        Returns a list of multimodal content in ChatML format.
        """
        multimodal_paths = None
        task_multimodal = kwargs.get("task_multimodal", None)
        if task_multimodal is not None:
            multimodal_paths = task_multimodal
        elif isinstance(message, dotdict) and self.task_multimodal is not None:
            multimodal_paths = self._extract_message_values(
                self.task_multimodal, message
            )

        if multimodal_paths is None:
            return None

        content = []

        formatters = {
            "image": self._format_image_input,
            "audio": self._format_audio_input,
            "video": self._format_video_input,
            "file": self._format_file_input,
        }

        for media_type, formatter in formatters.items():
            media_sources = multimodal_paths.get(media_type, [])
            if not isinstance(media_sources, list):
                media_sources = [media_sources]
            for media_source in media_sources:
                if media_source is not None:
                    formatted_input = formatter(media_source)
                    if formatted_input:
                        content.append(formatted_input)

        return content

    async def _arender_task_multimodal(
        self, message: Union[str, Message, Mapping[str, Any]], **kwargs
    ) -> Optional[List[Mapping[str, Any]]]:
        """Async version of _render_task_multimodal.
        Processes multimodal inputs (image, audio, video, file) via kwargs or message.
        Returns a list of multimodal content in ChatML format.
        """
        multimodal_paths = None
        task_multimodal = kwargs.get("task_multimodal", None)
        if task_multimodal is not None:
            multimodal_paths = task_multimodal
        elif isinstance(message, dotdict) and self.task_multimodal is not None:
            multimodal_paths = self._extract_message_values(
                self.task_multimodal, message
            )

        if multimodal_paths is None:
            return None

        content = []

        formatters = {
            "image": self._aformat_image_input,
            "audio": self._aformat_audio_input,
            "video": self._aformat_video_input,
            "file": self._aformat_file_input,
        }

        for media_type, formatter in formatters.items():
            media_sources = multimodal_paths.get(media_type, [])
            if not isinstance(media_sources, list):
                media_sources = [media_sources]
            for media_source in media_sources:
                if media_source is not None:
                    formatted_input = await formatter(media_source)
                    if formatted_input:
                        content.append(formatted_input)

        return content

    def _format_image_input(self, image_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the image input for the model."""
        img = Image(image_source, **self.config.get("image_block_kwargs", {}))
        return img()

    def _format_video_input(self, video_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the video input for the model."""
        # URLs: don't force encode (keep URL), local files: encode
        is_url = video_source.startswith("http")
        vid = Video(
            video_source,
            force_encode=not is_url,
            **self.config.get("video_block_kwargs", {}),
        )
        return vid()

    def _format_audio_input(self, audio_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the audio input for the model."""
        aud = Audio(audio_source)
        return aud()

    def _format_file_input(self, file_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the file input for the model."""
        f = File(file_source)
        return f()

    async def _aformat_image_input(
        self, image_source: str
    ) -> Optional[Mapping[str, Any]]:
        """Async version of _format_image_input."""
        img = Image(image_source, **self.config.get("image_block_kwargs", {}))
        return await img.acall()

    async def _aformat_video_input(
        self, video_source: str
    ) -> Optional[Mapping[str, Any]]:
        """Async version of _format_video_input."""
        # URLs: don't force encode (keep URL), local files: encode
        is_url = video_source.startswith("http")
        vid = Video(
            video_source,
            force_encode=not is_url,
            **self.config.get("video_block_kwargs", {}),
        )
        return await vid.acall()

    async def _aformat_audio_input(
        self, audio_source: str
    ) -> Optional[Mapping[str, Any]]:
        """Async version of _format_audio_input."""
        aud = Audio(audio_source)
        return await aud.acall()

    async def _aformat_file_input(
        self, file_source: str
    ) -> Optional[Mapping[str, Any]]:
        """Async version of _format_file_input."""
        f = File(file_source)
        return await f.acall()

    # --- Debug ---

    def inspect_model_execution_params(
        self, message: Optional[Union[str, Mapping[str, Any], Message]] = None, **kwargs
    ) -> Mapping[str, Any]:
        """Debug model input parameters.

        Accepts the same arguments as forward() to inspect what would be sent to
        the model.
        """
        inputs = self._prepare_inputs(message, start_turn=False, **kwargs)
        model_execution_params = self._prepare_model_execution(
            prefilling=self.prefilling,
            drain_notifications=False,
            **inputs,
        )
        return model_execution_params

    # --- Tool Filtering ---

    def _apply_tool_filter(
        self,
        tool_specs: List[ToolCatalogEntry],
        tool_filter: ToolFilter,
    ) -> List[ToolCatalogEntry]:
        """Return only the logical tools allowed by the runtime filter."""
        if not isinstance(tool_filter, dict):
            raise ValueError(
                f"`tool_filter` must be a dict, given `{type(tool_filter)}`"
            )

        keys = set(tool_filter.keys())
        valid_keys = {"allow", "block"}

        if not keys:
            raise ValueError("`tool_filter` must contain 'allow' or 'block' key")

        if keys - valid_keys:
            raise ValueError(
                f"`tool_filter` contains invalid keys: {keys - valid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        if len(keys) > 1:
            raise ValueError(
                "`tool_filter` must contain only one key: 'allow' or 'block', "
                f"got both: {keys}"
            )

        if "allow" in tool_filter:
            allowed_tools = self._normalize_tool_filter_values(
                tool_filter["allow"], key="allow"
            )
            return [tool for tool in tool_specs if tool.name in allowed_tools]

        blocked_tools = self._normalize_tool_filter_values(
            tool_filter["block"], key="block"
        )
        if "*" in blocked_tools:
            return []
        return [tool for tool in tool_specs if tool.name not in blocked_tools]

    def _normalize_tool_filter_values(
        self,
        values: ToolFilterValue,
        *,
        key: str,
    ) -> set[str]:
        """Normalize string-or-list filter values into a validated set of names."""
        if isinstance(values, str):
            if not values:
                raise ValueError(
                    f"`tool_filter['{key}']` must be a non-empty string or list of strings"  # noqa: E501
                )
            return {values}

        if isinstance(values, list):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(
                    f"`tool_filter['{key}']` must contain only non-empty strings"
                )
            return set(values)

        raise ValueError(
            f"`tool_filter['{key}']` must be a string or list of strings, "
            f"given `{type(values)}`"
        )

    def _resolve_tool_choice(
        self,
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        tool_specs: Optional[List[ToolCatalogEntry]],
    ) -> ToolChoice:
        """Keep tool_choice aligned with the filtered tool set."""
        if not tool_specs:
            return ToolChoice(mode="none")

        tool_names = {tool.name for tool in tool_specs}
        adjusted_tool_choice = ToolChoice.coerce(tool_choice)
        if (
            adjusted_tool_choice.mode == "tool"
            and adjusted_tool_choice.name not in tool_names
        ):
            adjusted_tool_choice = ToolChoice()

        if adjusted_tool_choice != ToolChoice.coerce(tool_choice) and self.config.get(
            "verbose", False
        ):
            cprint(
                f"[{self.name}][tool_choice] Adjusted to `{adjusted_tool_choice}` "
                "after tool filtering",
                bc="y",
                ls="b",
            )

        return adjusted_tool_choice

    # --- Message State Helpers ---
