import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)

import msgspec

from msgflux.auto import AutoParams
from msgflux.core.examples import Example
from msgflux.core.message import Message
from msgflux.dsl.signature import (
    Signature,
)
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
    _GuardInterrupt,
)
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.extensions.base import (
    AgentExtension,
)
from msgflux.nn.extensions.feedback import DefaultToolFeedbackExtension
from msgflux.nn.extensions.limits import ToolTurnLimitExtension
from msgflux.nn.extensions.prompt import (
    FewShotExamplesExtension,
    ToolUsageGuidanceExtension,
)
from msgflux.nn.extensions.skills import SkillsExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import (
    BeforeResume,
    BeforeRun,
)
from msgflux.nn.modules.agent.approvals import AgentApprovalMixin
from msgflux.nn.modules.generator import Generator
from msgflux.nn.modules.module import Module
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
)
from msgflux.runtime.approvals.agent import (
    AgentApprovals,
    ApprovalReconciliationRequiredError,
)
from msgflux.runtime.context import (
    execution_context,
)
from msgflux.runtime.skills import SkillsConfig

if TYPE_CHECKING:
    from msgflux.data.stores import CheckpointStore
from msgflux.nn.modules.agent.compaction import AgentCompactionMixin
from msgflux.nn.modules.agent.configuration import AgentConfigurationMixin
from msgflux.nn.modules.agent.context import (
    _DEFAULT_AGENT_ANNOTATIONS,
    _RESERVED_KWARGS,
    _UNSET,
    _apply_before_resume,
    _prepare_agent_guard_input,
    _prepare_agent_guard_output,
)
from msgflux.nn.modules.agent.continuation import AgentContinuationMixin
from msgflux.nn.modules.agent.conversation import AgentConversationMixin
from msgflux.nn.modules.agent.inputs import AgentInputMixin
from msgflux.nn.modules.agent.lifecycle import AgentLifecycleMixin
from msgflux.nn.modules.agent.model_runtime import AgentModelRuntimeMixin


class Agent(
    AgentApprovalMixin,
    AgentLifecycleMixin,
    AgentContinuationMixin,
    AgentModelRuntimeMixin,
    AgentCompactionMixin,
    AgentInputMixin,
    AgentConversationMixin,
    AgentConfigurationMixin,
    Module,
    metaclass=AutoParams,
):
    """Agent is a Module type that uses language models to solve tasks.

    An Agent can perform actions in an environment using tools calls.
    For an Agent, a tool is any callable object.

    An Agent can handle multimodal inputs and outputs.
    """

    _event_source_type = "agent"
    _emit_nested_run_events = True

    # Configure AutoParams to use docstring as 'description' parameter
    _autoparams_use_docstring_for = "description"
    # Configure AutoParams to use class name as 'name' parameter
    _autoparams_use_classname_for = "name"

    _supported_outputs: List[str] = [
        "structured",
        "text_generation",
        "audio_generation",
        "audio_text_generation",
        "tool_responses",
    ]

    @property
    def tool_kind(self) -> str:
        return "agent"

    def __init__(  # noqa: C901
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelGateway, "Generator", str],
        *,
        system_prompt: Optional[str] = None,
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None,
        hooks: Optional[List["Hook"]] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        templates: Optional[Dict[str, str]] = None,
        context_cache: Optional[str] = None,
        prefilling: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        response_mode: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        skills: Optional[SkillsConfig] = None,
        extensions: Optional[
            Union[List[AgentExtension], Mapping[str, AgentExtension]]
        ] = None,
        mcp_servers: Optional[List[Mapping[str, Any]]] = None,
        signature: Optional[Union[str, Signature]] = None,
        description: Optional[str] = None,
        annotations: Optional[Mapping[str, type]] = None,
        checkpoint_store: Optional["CheckpointStore"] = None,
        agent_inbox: Optional[AgentInbox] = None,
        approvals: Optional[AgentApprovals] = None,
    ):
        """Initialize the Agent module.

        Args:
        name:
            Agent name in snake case format.
        model:
            Chat completion model client. Accepts a `ChatCompletionModel`,
            `ModelGateway`, `Generator`, or a shorthand string in the form
            ``"provider/model-id"`` (e.g. ``"openai/gpt-4.1-mini"``).
            When a string is provided, `Model.chat_completion` is called
            internally with no extra configuration.
        system_prompt:
            Instructions and stable context for the model. Runtime extensions
            may add request-specific sections without mutating this parameter.
        examples:
            Few-shot examples installed through ``FewShotExamplesExtension``.
        hooks:
            List of Hook instances (e.g. Guard) to register on the model.
            !!! example
                hooks=[Guard(validator=checker, on="pre", message="Blocked.")]
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task", "task_multimodal", "messages",
            "task_context", "model_preference", "vars"
            !!! example
                message_fields={
                    "task": "input.user",
                    "task_multimodal": {"audio": "audio.user"},
                    "messages": "messages.history",
                    "task_context": "context.data",
                    "model_preference": "model.preference",
                    "vars": "vars.data"
                }

            Field descriptions:
            - task: Field path for task input (str, dict, or tuple)
            - task_multimodal: Map datatype (image, video, audio, file)
              to field paths
            - messages: Field path for list of chats in ChatML format
            - task_context: Field path for task context (str or list of str)
            - model_preference: Field path for model preference (str, only valid
              with ModelGateway)
            - vars: Field path for inputs to templates and tools (str)
        config:
            Dictionary with configuration options.
            Valid keys: "verbose", "return_messages", "tool_choice",
            "stream", "image_block_kwargs", "video_block_kwargs",
            "reasoning_in_response", "validate_inputs"
            !!! example
                config={
                    "verbose": True,
                    "return_messages": False,
                    "tool_choice": "auto",
                    "stream": False,
                    "image_block_kwargs": {"detail": "high"},
                    "video_block_kwargs": {"format": "mp4"},
                    "validate_inputs": True,
                }

            Configuration options:
            - verbose: Print model output and tool calls to console (bool)
            - return_messages: Return dict with messages and response (bool)
            - tool_choice: Control tool selection ("auto", "required", or function name)
            - stream: Transmit response on-the-fly (bool)
            - image_block_kwargs: Dict of kwargs to pass to ChatBlock.image
              (e.g., {"detail": "high"})
            - video_block_kwargs: Dict of kwargs to pass to ChatBlock.video
              (e.g., {"format": "mp4"})
            - validate_inputs: Validate input types against the signature
              schema before calling the model (bool).
        templates:
            Dictionary mapping template types to Jinja template strings.
            Valid keys: "task", "response", "task_context"
            !!! example
                templates={
                    "task": "Who was {{person}}?",
                    "response": "{{final_answer}}",
                    "task_context": "Context: {{context}}"
                }

            Template descriptions:
            - task: Formats the task/prompt sent to the model
            - response: Formats the model's response
            - task_context: Formats task context (does NOT apply to context_cache)
        context_cache:
            A fixed context.
        prefilling:
            Forces an initial message from the model. From that message it
            will continue its response from there.
        generation_schema:
            Schema that defines how the output should be structured.
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        tools:
            A list of callable objects.
        skills:
            Compatibility alias that installs ``SkillsExtension(skills)``.
            Prefer passing a ``SkillsExtension`` through ``extensions``.
        extensions:
            Agent extensions installed at construction. Accepts a list, using
            each extension's name, or a mapping of explicit names to extensions.
        mcp_servers:
            List of MCP (Model Context Protocol) server configurations.
            Each config should contain:
            - name: Namespace for tools from this server
            - transport: "stdio" or "http"
            - For stdio: command, args, cwd, env
            - For http: base_url, headers
            - Optional: include_tools, exclude_tools, tool_config
            !!! example
                mcp_servers=[{
                    "name": "fs",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    "include_tools": ["read_file", "write_file"],
                    "tool_config": {
                        "read_file": {
                            "runtime_inputs": (
                                ContextBinding(
                                    source="vars",
                                    parameter="context",
                                    options={"key": "context"},
                                ),
                            )
                        }
                    }
                }]
        signature:
            A DSPy-based signature. A signature creates a task template,
            generation schema, system-prompt guidance and optional examples.
            Can be combined with standard generation_schemas like `ReAct` and
            `ChainOfThought`.
        description:
            The Agent description. It's useful when using an agent-as-tool.
        annotations
            Define the input and output annotations to use the agent-as-a-function.
        checkpoint_store:
            Store used to persist and resume agent execution snapshots. A store
            configured directly on the agent takes precedence over one inherited
            from `execution_context(...)`.
        approvals:
            Optional host-owned AgentApprovals policy for foreground tool calls.
            Requires atomic checkpoints and an explicit live principal. Pending
            batches pause before execution and resume under the same run identity.
        """
        if annotations is None:
            annotations = _DEFAULT_AGENT_ANNOTATIONS.copy()

        # Validate that signature and custom annotations are not both provided
        if signature is not None and annotations != _DEFAULT_AGENT_ANNOTATIONS:
            raise ValueError(
                "Cannot specify both 'signature' and custom 'annotations'. "
                "When using a signature, annotations are generated automatically "
                "from the signature inputs. Remove the 'annotations' parameter."
            )

        # Validate custom annotations don't use reserved kwargs
        if annotations != _DEFAULT_AGENT_ANNOTATIONS:
            input_names = {k for k in annotations if k != "return"}
            conflicts = input_names & _RESERVED_KWARGS
            if conflicts:
                raise ValueError(
                    f"Annotation input names {conflicts} conflict with reserved "
                    f"Agent kwargs. Reserved names: {_RESERVED_KWARGS}. "
                    f"Rename these inputs to avoid conflicts."
                )

        # Validate that signature and task template are not both provided
        if signature is not None and templates is not None and "task" in templates:
            raise ValueError(
                "Cannot specify both 'signature' and templates['task']. "
                "When using a signature, the task template is generated automatically "
                "from the signature inputs. Remove the 'task' key from templates."
            )

        super().__init__()
        self.set_name(name)
        self.set_description(description)

        # Only set annotations if signature is not provided
        # (signature will set annotations automatically in _set_signature)
        if signature is None:
            self.set_annotations(annotations)
        else:
            # Set default temporarily, will be overridden by _set_signature
            self.set_annotations(_DEFAULT_AGENT_ANNOTATIONS.copy())

        self._set_config(config)
        self.checkpoint_store = checkpoint_store
        if approvals is not None and not isinstance(approvals, AgentApprovals):
            raise TypeError("approvals must be AgentApprovals or None")
        self.approvals = approvals
        if agent_inbox is None:
            self.agent_inbox = AgentInbox(
                verbose=config.get("verbose", False) if config else False,
                owner=name,
                store=InMemoryAgentInboxStore(),
            )
        else:
            self.agent_inbox = agent_inbox
            if config and config.get("verbose", False):
                self.agent_inbox.set_verbose(True)

        stream = config.get("stream", False) if config else False

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if hooks and any(getattr(h, "on", None) == "post" for h in hooks):
                raise ValueError(
                    "Hooks with `on='post'` are not `stream=True` compatible"
                )

            if templates is not None and templates.get("response") is not None:
                raise ValueError(
                    "`templates['response']` is not `stream=True` compatible"
                )

        self._set_context_cache(context_cache)
        self._set_message_fields(message_fields)
        self._set_model(model)
        self._set_hooks(
            hooks,
            processors={
                "guard_pre": _prepare_agent_guard_input,
                "guard_post": _prepare_agent_guard_output,
            },
        )
        self._set_prefilling(prefilling)
        self._set_response_mode(response_mode)
        self._set_templates(templates)
        self._set_tools(tools, mcp_servers)

        if signature is not None:
            signature_params = {
                "signature": signature,
                "examples": examples,
                "system_prompt": system_prompt,
            }
            if generation_schema is not None:
                signature_params["generation_schema"] = generation_schema
            examples = self._set_signature(**signature_params)
        else:
            self._set_generation_schema(generation_schema)
            self._set_system_prompt(system_prompt)

        self._initialize_extensions()
        if isinstance(extensions, Mapping):
            configured_extension_names = set(extensions)
            configured_extension_names.update(
                extension.name for extension in extensions.values()
            )
        else:
            configured_extension_names = {
                extension.name for extension in extensions or ()
            }
        if examples is not None:
            if "few_shot_examples" in configured_extension_names:
                raise ValueError(
                    "`examples` cannot be combined with a `few_shot_examples` "
                    "extension."
                )
            self.register_extension(
                "few_shot_examples",
                FewShotExamplesExtension(examples),
            )
        if "tool_usage_guidance" not in configured_extension_names:
            self.register_extension(
                "tool_usage_guidance",
                ToolUsageGuidanceExtension(),
            )
        if "tool_feedback" not in configured_extension_names:
            self.register_extension(
                "tool_feedback",
                DefaultToolFeedbackExtension(),
            )
        if extensions:
            self._set_extensions(extensions)
        if skills is not None:
            if self.has_extension("skills"):
                raise ValueError(
                    "`skills` cannot be combined with a `skills` extension."
                )
            self.register_extension("skills", SkillsExtension(skills))

        max_tool_turns = self.config.get("max_tool_turns")
        if max_tool_turns is not None:
            if self.has_extension("tool_turn_limit"):
                raise ValueError(
                    "Use max_tool_turns or ToolTurnLimitExtension, not both"
                )
            self.register_extension(
                "tool_turn_limit", ToolTurnLimitExtension(max_tool_turns)
            )

    def forward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        *,
        approvals: Optional[AgentApprovals] = _UNSET,
        **kwargs: Any,
    ) -> Union[str, Mapping[str, None], ModelStreamResponse, Message]:
        """Execute the agent with the given message.

        Args:
            message: The input message, which can be:
                - str: Direct task input (used as task)
                - Message: Message object with fields mapped via message_fields.
                  Requires message_fields configuration, e.g.:
                  message_fields={"task": "input.user"}
                - dict: Task inputs as a dictionary
                - None: When using named task arguments (see below)
            approvals: Execution-local approval policy. Omitted uses the Agent
                default; None explicitly disables prompts for new batches.
                Pending approvals still require their host policy on resume.
            **kwargs: Can include:
                - Reserved kwargs (runtime overrides for message_fields):
                    - task_multimodal: Override multimodal inputs
                    - messages: Override chat messages (chat history)
                    - task_context: Override task context
                    - model_preference: Override model preference
                    - vars: Override template/tool variables
                    - tool_filter: Filter which tools are available to the model.
                      Must contain exactly one key: "allow" or "block".
                      Values can be a single tool name or a list of names.
                      - {"allow": ["tool1", "tool2"]}: Only these tools are available
                      - {"allow": "tool1"}: Only this tool is available
                      - {"block": ["tool3"]}: All tools except these are available
                      - {"block": "*"}: Disable all tools
                - Named task arguments: When message=None and a task template is
                  configured, any other kwargs are treated as task inputs.
                  Example: agent(name="Vilson", age=28)
                  This is useful when using agents as tools with typed annotations.

        Returns:
            Agent response (str, Message, or ModelStreamResponse depending on
            configuration)

        Raises:
            ValueError: If both message and named task arguments are provided,
                or if named arguments are used without a task template.

        Examples:
            >>> # String input
            >>> agent("What is the weather?")

            >>> # Dict input
            >>> agent({"city": "Natal"})

            >>> # Message input (requires message_fields configuration)
            >>> agent_with_message = Agent(
            ...     model=model,
            ...     message_fields={"task": "user.query"}
            ... )
            >>> msg = Message()
            >>> msg.set("user.query", "Hello")
            >>> agent_with_message(msg)

            >>> # Named arguments (requires task template)
            >>> agent = Agent(
            ...     model=model,
            ...     templates={"task": "Greet {{name}} who is {{age}} years old"},
            ... )
            >>> agent(name="Vilson", age=28)

            >>> # Filter tools - allow only specific tools
            >>> agent("query", tool_filter={"allow": ["search", "calculator"]})

            >>> # Filter tools - block specific tools
            >>> agent("query", tool_filter={"block": ["browser"]})
        """
        approvals = self._get_effective_approvals(approvals)
        requested_scope = self._get_requested_scope(kwargs)
        resumed = self._try_resume_from_checkpoint(
            kwargs.get("messages"),
            scope=requested_scope,
        )
        if resumed is not None:
            resume_event = self._run_lifecycle_hooks(
                "before_resume",
                BeforeResume(
                    scope=resumed["scope"],
                    messages=resumed["messages"],
                    model_preference=resumed.get("model_preference"),
                ),
            )
            inputs = _apply_before_resume(
                resumed,
                resume_event,
                vars=kwargs.get("vars", {}),
            )
        else:
            run_event = self._run_lifecycle_hooks(
                "before_run",
                BeforeRun(message=message, kwargs=dict(kwargs)),
            )
            inputs = self._prepare_inputs(
                run_event.message,
                **dict(run_event.kwargs),
            )

        self._update_agent_context(inputs)
        effective_checkpoint_store = self._get_effective_checkpoint_store()
        effective_task_store = self._get_effective_task_store()
        effective_inbox = self._get_scoped_agent_inbox(inputs.get("scope"))
        with (
            self._approval_context(approvals),
            execution_context(
                scope=inputs.get("scope"),
                checkpoint_store=effective_checkpoint_store,
                task_store=effective_task_store,
                agent_inbox=effective_inbox,
            ),
        ):
            try:
                try:
                    model_response = self._approval_replay() or self._execute_model(
                        prefilling=self.prefilling,
                        **inputs,
                    )
                except _GuardInterrupt as e:
                    model_response = self._guard_model_response(e.response)
                response = self._process_model_response(
                    message,
                    model_response,
                    **inputs,
                )
            except (AbortRequestedError, TaskInterruptRequestedError) as exc:
                self._settle_terminal_run(inputs, "interrupted", exc)
                self._raise_interrupted_from_abort(inputs, exc)
            except ApprovalReconciliationRequiredError:
                raise
            except TaskPauseRequestedError as exc:
                self._settle_terminal_run(inputs, "paused", exc)
                raise
            except Exception as exc:
                settled_error = self._settle_processing_error(inputs, exc)
                if settled_error is exc:
                    raise
                raise settled_error from exc
            return response

    async def aforward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        *,
        approvals: Optional[AgentApprovals] = _UNSET,
        **kwargs: Any,
    ) -> Union[str, Mapping[str, None], ModelStreamResponse, Message]:
        """Async version of forward."""
        approvals = self._get_effective_approvals(approvals)
        requested_scope = self._get_requested_scope(kwargs)
        resumed = await self._atry_resume_from_checkpoint(
            kwargs.get("messages"),
            scope=requested_scope,
        )
        if resumed is not None:
            resume_event = await self._arun_lifecycle_hooks(
                "before_resume",
                BeforeResume(
                    scope=resumed["scope"],
                    messages=resumed["messages"],
                    model_preference=resumed.get("model_preference"),
                ),
            )
            inputs = _apply_before_resume(
                resumed,
                resume_event,
                vars=kwargs.get("vars", {}),
            )
        else:
            run_event = await self._arun_lifecycle_hooks(
                "before_run",
                BeforeRun(message=message, kwargs=dict(kwargs)),
            )
            inputs = await self._aprepare_inputs(
                run_event.message,
                **dict(run_event.kwargs),
            )

        self._update_agent_context(inputs)
        effective_checkpoint_store = self._get_effective_checkpoint_store()
        effective_task_store = self._get_effective_task_store()
        effective_inbox = self._get_scoped_agent_inbox(inputs.get("scope"))
        with (
            self._approval_context(approvals),
            execution_context(
                scope=inputs.get("scope"),
                checkpoint_store=effective_checkpoint_store,
                task_store=effective_task_store,
                agent_inbox=effective_inbox,
            ),
        ):
            try:
                try:
                    model_response = self._approval_replay()
                    if model_response is None:
                        model_response = await self._aexecute_model(
                            prefilling=self.prefilling,
                            **inputs,
                        )
                except _GuardInterrupt as e:
                    model_response = self._guard_model_response(e.response)
                response = await self._aprocess_model_response(
                    message,
                    model_response,
                    **inputs,
                )
            except (AbortRequestedError, TaskInterruptRequestedError) as exc:
                await self._asettle_terminal_run(inputs, "interrupted", exc)
                self._raise_interrupted_from_abort(inputs, exc)
            except asyncio.CancelledError as exc:
                await self._asettle_terminal_run(inputs, "interrupted", exc)
                raise
            except ApprovalReconciliationRequiredError:
                raise
            except TaskPauseRequestedError as exc:
                await self._asettle_terminal_run(inputs, "paused", exc)
                raise
            except Exception as exc:
                settled_error = await self._asettle_processing_error(inputs, exc)
                if settled_error is exc:
                    raise
                raise settled_error from exc
            return response

    @staticmethod
    def _guard_model_response(response: str) -> ModelResponse:
        model_response = ModelResponse()
        model_response.set_response_type("text_generation")
        model_response.add(response)
        return model_response

    # --- Extensions ---
