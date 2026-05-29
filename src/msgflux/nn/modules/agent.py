from datetime import datetime, timezone
from inspect import cleandoc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from uuid import uuid4

import msgspec

from msgflux.agent_inbox import AgentInbox, AgentNotification
from msgflux.auto import AutoParams
from msgflux.chat_messages import ChatMessages
from msgflux.context import ExecutionScope, execution_context, get_execution_context
from msgflux.core.dotdict import dotdict
from msgflux.core.examples import Example, ExampleCollection
from msgflux.core.message import Message
from msgflux.data.types import Audio, File, Image, Video
from msgflux.dsl.signature import (
    Signature,
    SignatureFactory,
    generate_annotations_from_signature,
)
from msgflux.dsl.typed_parsers.registry import typed_parser_registry
from msgflux.exceptions import (
    TaskPauseRequestedError,
    TaskStopRequestedError,
    _GuardInterrupt,
)
from msgflux.generation.control_flow import ToolFlowControl
from msgflux.generation.templates import (
    EXPECTED_OUTPUTS_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    PromptSpec,
)
from msgflux.models import Model
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.functional import aspawn, await_for_event, spawn, wait_for_event
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.generator import Generator
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary, ToolResponses
from msgflux.nn.parameter import Parameter
from msgflux.runtime.skills import AgentSkillManager, SkillsConfig
from msgflux.tools.builtin import ActivateSkill, SkillSearch
from msgflux.tools.definitions import ToolDefinitions
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.common import has_format_placeholder, is_jinja_template
from msgflux.utils.console import cprint
from msgflux.utils.msgspec import StructFactory, is_optional_field, msgspec_dumps
from msgflux.utils.validation import is_subclass_of
from msgflux.utils.xml import apply_xml_tags

if TYPE_CHECKING:
    from msgflux.data.stores import CheckpointStore

# Reserved kwargs that should not be treated as task inputs
_RESERVED_KWARGS = {
    "task",
    "vars",
    "messages",
    "task_multimodal",
    "task_context",
    "model_preference",
    "tool_filter",
    "scope",
    "tool_call_id",
}

_UNSET = object()
_DEFAULT_AGENT_ANNOTATIONS = {"task": str, "return": str}
ToolFilterValue = Union[str, List[str]]
ToolFilter = Dict[str, ToolFilterValue]


def _prepare_agent_guard_input(model_execution_params):
    """Extract user content from ChatML messages for guard validation."""
    messages = model_execution_params.get("messages")
    if not messages:
        return model_execution_params
    last_message = messages[-1]
    if isinstance(last_message.get("content"), list):
        if last_message.get("content")[0]["type"] == "image_url":
            return [last_message]
        else:
            return last_message.get("content")[-1]
    else:
        return last_message.get("content")


def _prepare_agent_guard_output(model_response):
    """Convert model response to string for guard validation."""
    if isinstance(model_response, str):
        return model_response
    return str(model_response)


class Agent(Module, metaclass=AutoParams):
    """Agent is a Module type that uses language models to solve tasks.

    An Agent can perform actions in an environment using tools calls.
    For an Agent, a tool is any callable object.

    An Agent can handle multimodal inputs and outputs.
    """

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

    def __init__(  # noqa: C901
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelGateway, "Generator", str],
        *,
        system_message: Optional[str] = None,
        instructions: Optional[str] = None,
        expected_output: Optional[str] = None,
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None,
        system_extra_message: Optional[str] = None,
        hooks: Optional[List["Hook"]] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        templates: Optional[Dict[str, str]] = None,
        context_cache: Optional[str] = None,
        prefilling: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        typed_parser: Optional[str] = None,
        response_mode: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        skills: Optional[SkillsConfig] = None,
        mcp_servers: Optional[List[Mapping[str, Any]]] = None,
        signature: Optional[Union[str, Signature]] = None,
        description: Optional[str] = None,
        annotations: Optional[Mapping[str, type]] = None,
        checkpointer: Optional["CheckpointStore"] = None,
        agent_inbox: Optional[AgentInbox] = None,
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
        system_message:
            The Agent behaviour.
        instructions:
            What the Agent should do.
        expected_output:
            What the response should be like.
        examples:
            Examples of inputs, reasoning and outputs.
        system_extra_message:
            An extra message in system prompt.
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
            "stream", "image_block_kwargs", "video_block_kwargs", "include_date",
            "reasoning_in_response", "validate_inputs"
            !!! example
                config={
                    "verbose": True,
                    "return_messages": False,
                    "tool_choice": "auto",
                    "stream": False,
                    "image_block_kwargs": {"detail": "high"},
                    "video_block_kwargs": {"format": "mp4"},
                    "include_date": False,
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
            - include_date: Include current date with weekday in system prompt
              (bool). Format: "Weekday, Month DD, YYYY"
            - validate_inputs: Validate input types against the signature
              schema before calling the model (bool).
        templates:
            Dictionary mapping template types to Jinja template strings.
            Valid keys: "task", "response", "task_context", "system_prompt"
            !!! example
                templates={
                    "task": "Who was {{person}}?",
                    "response": "{{final_answer}}",
                    "task_context": "Context: {{context}}",
                    "system_prompt": "Custom system prompt: ..."
                }

            Template descriptions:
            - task: Formats the task/prompt sent to the model
            - response: Formats the model's response
            - task_context: Formats task context (does NOT apply to context_cache)
            - system_prompt: Overrides the default system prompt
              template. If not provided, uses SYSTEM_PROMPT_TEMPLATE.
              Available variables: system_message, instructions,
              expected_output, examples, system_extra_message,
              current_date (if include_date=True)
        context_cache:
            A fixed context.
        prefilling:
            Forces an initial message from the model. From that message it
            will continue its response from there.
        generation_schema:
            Schema that defines how the output should be structured.
        typed_parser:
            Converts the model raw output into a typed-dict. Supported parser:
            `typed_xml`.
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        tools:
            A list of callable objects.
        skills:
            Agent Skills config dict with `paths`, `catalog_limit`, `search_top_k`,
            `allow`, `block`, and `load`. Use `msgflux.default_skill_paths()` when
            you want common local paths.
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
                    "tool_config": {"read_file": {"inject_vars": ["context"]}}
                }]
        signature:
            A DSPy-based signature. A signature creates a task_template,
            a generation_schema, instructions and examples (both if passed).
            Can be combined with standard generation_schemas like `ReAct` and
            `ChainOfThought`. Can also be combined with `typed_parser`.
        description:
            The Agent description. It's useful when using an agent-as-tool.
        annotations
            Define the input and output annotations to use the agent-as-a-function.
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

        # Validate that signature and expected_output are not both provided
        if signature is not None and expected_output is not None:
            raise ValueError(
                "Cannot specify both 'signature' and 'expected_output'. "
                "When using a signature, expected_output is generated automatically "
                "from the signature outputs. Remove the 'expected_output' parameter."
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
        self.checkpointer = checkpointer
        if agent_inbox is None:
            self.agent_inbox = AgentInbox(
                verbose=config.get("verbose", False) if config else False,
                owner=name,
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

            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            if checkpointer is not None:
                raise ValueError("`checkpointer` is not `stream=True` compatible")

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
        self._set_system_extra_message(system_extra_message)

        self._set_response_mode(response_mode)
        self._set_templates(templates)
        self._set_skills(skills)
        self._set_tools(tools, mcp_servers)

        if signature is not None:
            signature_params = dotdict(
                signature=signature,
                examples=examples,
                instructions=instructions,
                system_message=system_message,
                typed_parser=typed_parser,
            )
            if generation_schema is not None:
                signature_params.generation_schema = generation_schema
            self._set_signature(**signature_params)
        else:
            self._set_typed_parser(typed_parser)
            self._set_examples(examples)
            self._set_generation_schema(generation_schema)
            self._set_expected_output(expected_output)
            self._set_instructions(instructions)
            self._set_system_message(system_message)

    def _get_requested_scope(
        self, kwargs: Mapping[str, Any]
    ) -> Optional[ExecutionScope]:
        scope = kwargs.get("scope")
        if scope is None:
            return None
        if not isinstance(scope, ExecutionScope):
            raise TypeError(
                f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
            )
        return scope

    def forward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
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
        requested_scope = self._get_requested_scope(kwargs)
        resumed = self._try_resume_from_checkpoint(
            kwargs.get("messages"),
            scope=requested_scope,
        )
        inputs = resumed or self._prepare_inputs(message, **kwargs)

        effective_checkpointer = self._get_effective_checkpointer()
        effective_inbox = self._get_effective_agent_inbox()
        if effective_inbox is not None:
            effective_inbox.bind_scope(
                inputs.get("scope"),
                namespace=self.get_module_name(),
            )
        with execution_context(
            scope=inputs.get("scope"),
            checkpoint_store=effective_checkpointer,
            agent_inbox=effective_inbox,
        ):
            try:
                model_response = self._execute_model(
                    prefilling=self.prefilling,
                    **inputs,
                )
            except _GuardInterrupt as e:
                return self._define_response_mode(e.response, message)
            except TaskStopRequestedError:
                self._checkpoint_save(
                    inputs.get("messages"), inputs.get("vars", {}), status="stopped"
                )
                raise
            except TaskPauseRequestedError:
                self._checkpoint_save(
                    inputs.get("messages"), inputs.get("vars", {}), status="paused"
                )
                raise
            except Exception:
                self._checkpoint_save_on_error(inputs)
                raise
            response = self._process_model_response(message, model_response, **inputs)
            return response

    async def aforward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        **kwargs: Any,
    ) -> Union[str, Mapping[str, None], ModelStreamResponse, Message]:
        """Async version of forward."""
        requested_scope = self._get_requested_scope(kwargs)
        resumed = await self._atry_resume_from_checkpoint(
            kwargs.get("messages"),
            scope=requested_scope,
        )
        inputs = resumed or await self._aprepare_inputs(message, **kwargs)

        effective_checkpointer = self._get_effective_checkpointer()
        effective_inbox = self._get_effective_agent_inbox()
        if effective_inbox is not None:
            effective_inbox.bind_scope(
                inputs.get("scope"),
                namespace=self.get_module_name(),
            )
        with execution_context(
            scope=inputs.get("scope"),
            checkpoint_store=effective_checkpointer,
            agent_inbox=effective_inbox,
        ):
            try:
                model_response = await self._aexecute_model(
                    prefilling=self.prefilling,
                    **inputs,
                )
            except _GuardInterrupt as e:
                return self._define_response_mode(e.response, message)
            except TaskStopRequestedError:
                await self._acheckpoint_save(
                    inputs.get("messages"),
                    inputs.get("vars", {}),
                    status="stopped",
                )
                raise
            except TaskPauseRequestedError:
                await self._acheckpoint_save(
                    inputs.get("messages"),
                    inputs.get("vars", {}),
                    status="paused",
                )
                raise
            except Exception:
                await self._acheckpoint_save_on_error(inputs)
                raise
            response = await self._aprocess_model_response(
                message,
                model_response,
                **inputs,
            )
            return response

    # --- Model Execution ---

    def _execute_model(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        session_id: Optional[str] = None,  # noqa: ARG002
        run_id: Optional[str] = None,  # noqa: ARG002
        scope: Optional[ExecutionScope] = None,  # noqa: ARG002
    ) -> Union[ModelResponse, ModelStreamResponse]:
        self._raise_if_background_task_stopped()
        model_execution_params = self._prepare_model_execution(
            messages=messages,
            prefilling=prefilling,
            model_preference=model_preference,
            vars=vars,
            tool_filter=tool_filter,
        )
        if self.config.get("verbose", False):
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        return self.generator(**model_execution_params)

    async def _aexecute_model(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        session_id: Optional[str] = None,  # noqa: ARG002
        run_id: Optional[str] = None,  # noqa: ARG002
        scope: Optional[ExecutionScope] = None,  # noqa: ARG002
    ) -> Union[ModelResponse, ModelStreamResponse]:
        self._raise_if_background_task_stopped()
        model_execution_params = self._prepare_model_execution(
            messages=messages,
            prefilling=prefilling,
            model_preference=model_preference,
            vars=vars,
            tool_filter=tool_filter,
        )
        if self.config.get("verbose", False):
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        return await self.generator.acall(**model_execution_params)

    def warmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
        background: bool = False,
    ):
        """Warm the provider cache for the rendered system prompt and tools.

        This bypasses task messages, chat history, checkpointers and response
        parsing. Warmup should only include stable prompt prefixes; dynamic user
        content would reduce cache hits for the real request.
        """
        if background:
            spawn(
                self._warmup_system_prompt,
                vars=vars,
                tool_filter=tool_filter,
                model_preference=model_preference,
            )
            return None
        return self._warmup_system_prompt(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )

    async def awarmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
        background: bool = False,
    ):
        """Async counterpart for warming the provider system prompt cache."""
        if background:
            await aspawn(
                self._awarmup_system_prompt,
                vars=vars,
                tool_filter=tool_filter,
                model_preference=model_preference,
            )
            return None
        return await self._awarmup_system_prompt(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )

    def _warmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ):
        params = self._prepare_warmup_execution(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )
        return self.model.warmup_system_prompt(**params)

    async def _awarmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ):
        params = self._prepare_warmup_execution(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )
        return await self.model.awarmup_system_prompt(**params)

    def _prepare_warmup_execution(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ) -> Mapping[str, Any]:
        model_execution_params = self._prepare_model_execution(
            messages=[],
            vars=vars or {},
            model_preference=model_preference,
            tool_filter=tool_filter,
        )
        return dotdict(
            system_prompt=model_execution_params.system_prompt,
            tool_definitions=model_execution_params.tool_definitions,
            **(
                {"model_preference": model_execution_params.model_preference}
                if model_preference
                else {}
            ),
        )

    def _prepare_model_execution(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        *,
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        drain_notifications: bool = True,
        session_id: Optional[str] = None,  # noqa: ARG002
        run_id: Optional[str] = None,  # noqa: ARG002
        scope: Optional[ExecutionScope] = None,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        model_messages = self._build_model_messages(
            messages,
            drain_notifications=drain_notifications,
        )

        tool_schemas = self.tool_library.get_tool_json_schemas()

        tool_choice = self.config.get("tool_choice")

        if tool_filter is not None and tool_schemas:
            tool_schemas = self._apply_tool_filter(tool_schemas, tool_filter)

        tool_choice = self._resolve_tool_choice(tool_choice, tool_schemas)
        tool_names = {
            schema["function"]["name"]
            for schema in tool_schemas or []
            if schema.get("function", {}).get("name")
        }
        system_prompt = self.get_system_prompt(vars, tool_names=tool_names)

        if not tool_schemas:
            tool_schemas = None

        tool_definitions = None

        if tool_schemas is not None:
            tool_definitions = ToolDefinitions(
                schemas=tool_schemas,
                annotations=self.tool_library.get_tool_annotations() or None,
                choice=tool_choice,
            )

        if is_subclass_of(self.generation_schema, ToolFlowControl) and tool_schemas:
            tools_template = self.generation_schema.tools_template
            inputs = {
                "tool_schemas": tool_definitions.schemas,
                "tool_choice": tool_definitions.choice,
            }
            flow_control_tools = self._format_template(inputs, tools_template)
            if system_prompt:
                system_prompt = flow_control_tools + "\n\n" + system_prompt
            else:
                system_prompt = flow_control_tools

        model_execution_params = dotdict(
            messages=model_messages,
            system_prompt=system_prompt or None,
            prefilling=prefilling,
            stream=self.config.get("stream", False),
            tool_definitions=tool_definitions,
            generation_schema=self.generation_schema,
            typed_parser=self.typed_parser,
        )

        if model_preference:
            model_execution_params.model_preference = model_preference

        return model_execution_params

    # --- Response Processing ---

    def _ensure_stream_response_ready(
        self, model_response: ModelStreamResponse
    ) -> None:
        if model_response.response_type is not None:
            return

        error = getattr(model_response, "error", None)
        if error is not None:
            raise RuntimeError(
                f"Model stream failed before producing a response: {error}"
            ) from error

        raise RuntimeError("Model stream ended before producing a response type.")

    def _process_model_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        session_id: Optional[str] = None,  # noqa: ARG002
        run_id: Optional[str] = None,  # noqa: ARG002
        scope: Optional[ExecutionScope] = None,  # noqa: ARG002
    ) -> Union[str, Mapping[str, Any], Message, ModelStreamResponse]:
        if isinstance(model_response, ModelStreamResponse):
            wait_for_event(model_response._response_type_event)
            self._ensure_stream_response_ready(model_response)

        if "tool_call" in model_response.response_type:
            model_response, messages = self._process_tool_call_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )
        elif is_subclass_of(self.generation_schema, ToolFlowControl):
            model_response, messages = self._process_tool_flow_control_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
            reasoning = model_response.reasoning
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"
            reasoning = None

        self._append_response_to_chat_messages(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
        )
        self._finalize_chat_turn(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
        )
        self._checkpoint_save(messages, vars, status="completed")

        if response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, response_type, messages, message, vars, reasoning
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    async def _aprocess_model_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        session_id: Optional[str] = None,  # noqa: ARG002
        run_id: Optional[str] = None,  # noqa: ARG002
        scope: Optional[ExecutionScope] = None,  # noqa: ARG002
    ) -> Union[str, Mapping[str, Any], Message, ModelStreamResponse]:
        if isinstance(model_response, ModelStreamResponse):
            await await_for_event(model_response._response_type_event)
            self._ensure_stream_response_ready(model_response)

        if "tool_call" in model_response.response_type:
            model_response, messages = await self._aprocess_tool_call_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )
        elif is_subclass_of(self.generation_schema, ToolFlowControl):
            (
                model_response,
                messages,
            ) = await self._aprocess_tool_flow_control_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
            reasoning = model_response.reasoning
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"
            reasoning = None

        self._append_response_to_chat_messages(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
        )
        self._finalize_chat_turn(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
        )
        await self._acheckpoint_save(messages, vars, status="completed")

        if response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, response_type, messages, message, vars, reasoning
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    # --- Tool Processing ---

    def _process_tool_flow_control_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Handle tool flow control responses using the ToolFlowControl interface."""
        max_tool_turns = self.config.get("max_tool_turns")
        completed_tool_turns = 0
        flow_control = self.generation_schema
        while True:
            raw_response = self._extract_raw_response(model_response)

            # Use ToolFlowControl interface via generation_schema
            flow_result = flow_control.extract_flow_result(raw_response)

            if flow_result.is_complete:
                if flow_result.final_response is not None:
                    model_response.data = flow_result.final_response
                return model_response, messages

            if self.config.get("verbose", False) and flow_result.reasoning:
                cprint(
                    f"[{self.name}][tool_calls_reasoning] {flow_result.reasoning}",
                    bc="br2",
                    ls="b",
                )

            if flow_result.tool_calls:
                if (
                    max_tool_turns is not None
                    and completed_tool_turns >= max_tool_turns
                ):
                    # Re-run once with tools disabled so the model can finalize.
                    tool_filter = self._block_all_tools(max_tool_turns)
                    model_response = self._execute_model(
                        messages=messages,
                        model_preference=model_preference,
                        vars=vars,
                        tool_filter=tool_filter,
                    )
                    continue

                tool_results = self._process_tool_call(
                    flow_result.tool_calls, message, messages, vars
                )
                completed_tool_turns += 1

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict().pop("return_directly")
                    tool_calls["reasoning"] = flow_result.reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, messages

                # Use interface to inject results
                raw_response = flow_control.inject_results(raw_response, tool_results)

                # Use interface to build history
                messages = flow_control.build_history(raw_response, messages)
                self._drain_inbox_into_messages(messages)
                self._checkpoint_save(messages, vars)

            model_response = self._execute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    async def _aprocess_tool_flow_control_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Async version of _process_tool_flow_control_response.
        Handle tool flow control responses using the ToolFlowControl interface.
        """
        max_tool_turns = self.config.get("max_tool_turns")
        completed_tool_turns = 0
        flow_control = self.generation_schema
        while True:
            raw_response = self._extract_raw_response(model_response)

            # Use ToolFlowControl interface via generation_schema (async)
            flow_result = await flow_control.aextract_flow_result(raw_response)

            if flow_result.is_complete:
                if flow_result.final_response is not None:
                    model_response.data = flow_result.final_response
                return model_response, messages

            if self.config.get("verbose", False) and flow_result.reasoning:
                cprint(
                    f"[{self.name}][tool_calls_reasoning] {flow_result.reasoning}",
                    bc="br2",
                    ls="b",
                )

            if flow_result.tool_calls:
                if (
                    max_tool_turns is not None
                    and completed_tool_turns >= max_tool_turns
                ):
                    # Re-run once with tools disabled so the model can finalize.
                    tool_filter = self._block_all_tools(max_tool_turns)
                    model_response = await self._aexecute_model(
                        messages=messages,
                        model_preference=model_preference,
                        vars=vars,
                        tool_filter=tool_filter,
                    )
                    continue

                tool_results = await self._aprocess_tool_call(
                    flow_result.tool_calls, message, messages, vars
                )
                completed_tool_turns += 1

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict().pop("return_directly")
                    tool_calls["reasoning"] = flow_result.reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, messages

                # Use interface to inject results (async version)
                raw_response = await flow_control.ainject_results(
                    raw_response, tool_results
                )

                # Use interface to build history (async version)
                messages = await flow_control.abuild_history(raw_response, messages)
                self._drain_inbox_into_messages(messages)
                await self._acheckpoint_save(messages, vars)

            model_response = await self._aexecute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    def _process_tool_call_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """ToolCall example:
        [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        max_tool_turns = self.config.get("max_tool_turns")
        completed_tool_turns = 0

        while True:
            if model_response.response_type == "tool_call":
                if (
                    max_tool_turns is not None
                    and completed_tool_turns >= max_tool_turns
                ):
                    # Re-run once with tools disabled so the model can finalize.
                    tool_filter = self._block_all_tools(max_tool_turns)
                    model_response = self._execute_model(
                        messages=messages,
                        model_preference=model_preference,
                        vars=vars,
                        tool_filter=tool_filter,
                    )
                    continue

                raw_response = model_response.data
                reasoning = model_response.reasoning

                if self.config.get("verbose", False):
                    if reasoning:
                        repr_str = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr_str, bc="br2", ls="b")

                tool_callings = raw_response.get_calls()
                tool_results = self._process_tool_call(
                    tool_callings, message, messages, vars
                )
                completed_tool_turns += 1

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict()
                    tool_calls.pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, messages

                id_results = {
                    call.id: call.result or call.error
                    for call in tool_results.tool_calls
                }
                raw_response.insert_results(id_results)
                tool_responses_message = raw_response.get_messages()
                messages.extend(tool_responses_message)
                self._drain_inbox_into_messages(messages)
                self._checkpoint_save(messages, vars)
            else:
                return model_response, messages

            model_response = self._execute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    async def _aprocess_tool_call_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Async version of _process_tool_call_response.
        ToolCall example: [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        max_tool_turns = self.config.get("max_tool_turns")
        completed_tool_turns = 0

        while True:
            if model_response.response_type == "tool_call":
                if (
                    max_tool_turns is not None
                    and completed_tool_turns >= max_tool_turns
                ):
                    # Re-run once with tools disabled so the model can finalize.
                    tool_filter = self._block_all_tools(max_tool_turns)
                    model_response = await self._aexecute_model(
                        messages=messages,
                        model_preference=model_preference,
                        vars=vars,
                        tool_filter=tool_filter,
                    )
                    continue

                raw_response = model_response.data
                reasoning = model_response.reasoning

                if self.config.get("verbose", False):
                    if reasoning:
                        repr_str = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr_str, bc="br2", ls="b")

                tool_callings = raw_response.get_calls()
                tool_results = await self._aprocess_tool_call(
                    tool_callings, message, messages, vars
                )
                completed_tool_turns += 1

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict()
                    tool_calls.pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, messages

                id_results = {
                    call.id: call.result or call.error
                    for call in tool_results.tool_calls
                }
                raw_response.insert_results(id_results)
                tool_responses_message = raw_response.get_messages()
                messages.extend(tool_responses_message)
                self._drain_inbox_into_messages(messages)
                await self._acheckpoint_save(messages, vars)
            else:
                return model_response, messages

            model_response = await self._aexecute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    def _process_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        if self.config.get("verbose", False):
            for call in tool_callings:
                repr_str = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr_str, bc="br2", ls="b")
        tool_results = self.tool_library(
            tool_callings=tool_callings,
            message=message,
            messages=messages,
            vars=vars,
        )
        if self.config.get("verbose", False):
            repr_str = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr_str += " return directly"
            cprint(repr_str, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr_str = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr_str, ls="b")
        return tool_results

    async def _aprocess_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        """Async version of _process_tool_call."""
        if self.config.get("verbose", False):
            for call in tool_callings:
                repr_str = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr_str, bc="br2", ls="b")
        tool_results = await self.tool_library.acall(
            tool_callings=tool_callings,
            message=message,
            messages=messages,
            vars=vars,
        )
        if self.config.get("verbose", False):
            repr_str = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr_str += " return directly"
            cprint(repr_str, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr_str = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr_str, ls="b")
        return tool_results

    def _apply_reasoning_in_response(self, raw_response, reasoning):
        if self.config.get("reasoning_in_response", False) and reasoning is not None:
            return dotdict(answer=raw_response, reasoning=reasoning)
        return raw_response

    def _prepare_response(
        self,
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        message: Union[str, Mapping[str, Any], Message],
        vars: Mapping[str, Any],
        reasoning: Optional[str] = None,
    ) -> Union[str, Mapping[str, Any], ModelStreamResponse]:
        formatted_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            raw_response = self._apply_reasoning_in_response(raw_response, reasoning)

            if response_type == "text_generation" or "structured" in response_type:
                if self.config.get("verbose", False):
                    if reasoning:
                        cprint(
                            f"[{self.name}][reasoning] {reasoning}", bc="br2", ls="b"
                        )
                    cprint(f"[{self.name}][response] {raw_response}", bc="y", ls="b")
                if self.templates.get("response"):
                    if isinstance(raw_response, str):
                        pre_response = self._format_response_template(vars)
                        formatted_response = self._format_template(
                            raw_response, pre_response
                        )
                    elif isinstance(raw_response, dict):
                        raw_response.update(vars)
                        formatted_response = self._format_response_template(
                            raw_response
                        )

        result = formatted_response or raw_response
        if self.config.get("return_messages", False):
            if response_type == "tool_responses":
                result.messages = messages
            else:
                result = dotdict(response=result, messages=messages)
        return self._define_response_mode(result, message)

    # --- Task Preparation ---

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

        effective_checkpointer = self._get_effective_checkpointer()
        should_use_chat_messages = (
            effective_checkpointer is not None
            or isinstance(messages, ChatMessages)
            or scope is not None
        )
        if should_use_chat_messages:
            messages = self._coerce_chat_messages(messages)

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

        effective_session_id = self._resolve_session_id(
            messages=messages,
            session_id=scope.session_id if scope is not None else None,
        )
        effective_run_id = self._resolve_run_id(
            messages=messages,
            run_id=scope.run_id if scope is not None else None,
        )
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            session_id=effective_session_id,
            namespace=self.get_module_name(),
            run_id=effective_run_id,
        )

        if isinstance(messages, ChatMessages):
            messages.configure_session(
                session_id=effective_session_id,
                namespace=self.get_module_name(),
            )
            if start_turn:
                self._start_chat_turn_if_needed(
                    messages=messages,
                    content=content,
                    task=task,
                    vars=vars,
                    turn_id=effective_run_id,
                    kwargs=kwargs,
                    message=message,
                )
            if content is not None:
                messages.add_user(content)
        elif content is not None:
            chat_content = [ChatBlock.user(content)]
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

        effective_checkpointer = self._get_effective_checkpointer()
        should_use_chat_messages = (
            effective_checkpointer is not None
            or isinstance(messages, ChatMessages)
            or scope is not None
        )
        if should_use_chat_messages:
            messages = self._coerce_chat_messages(messages)

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

        effective_session_id = self._resolve_session_id(
            messages=messages,
            session_id=scope.session_id if scope is not None else None,
        )
        effective_run_id = self._resolve_run_id(
            messages=messages,
            run_id=scope.run_id if scope is not None else None,
        )
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            session_id=effective_session_id,
            namespace=self.get_module_name(),
            run_id=effective_run_id,
        )

        if isinstance(messages, ChatMessages):
            messages.configure_session(
                session_id=effective_session_id,
                namespace=self.get_module_name(),
            )
            if start_turn:
                self._start_chat_turn_if_needed(
                    messages=messages,
                    content=content,
                    task=task,
                    vars=vars,
                    turn_id=effective_run_id,
                    kwargs=kwargs,
                    message=message,
                )
            if content is not None:
                messages.add_user(content)
        elif content is not None:
            chat_content = [ChatBlock.user(content)]
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
        content = ""

        context_content = self._context_manager(message, vars=vars, **kwargs)
        if context_content:
            content += context_content

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

        task_content = apply_xml_tags("task", task_content)
        content += task_content
        content = content.strip()  # Remove whitespace

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
        content = ""

        context_content = self._context_manager(message, vars=vars, **kwargs)
        if context_content:
            content += context_content

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

        task_content = apply_xml_tags("task", task_content)
        content += task_content
        content = content.strip()  # Remove whitespace

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
                    context.update(vars)
                    msg_context = self._format_template(
                        context, self.templates.get("task_context")
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
            elif isinstance(context, dict):
                msg_context = "\n".join(
                    f"{k}: {v if not isinstance(v, list) else ', '.join(v)}"
                    for k, v in context.items()
                )
            context_content += msg_context

        if context_content:
            if vars:
                context_content = self._format_template(vars, context_content)
            return apply_xml_tags("context", context_content) + "\n\n"
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
        tool_schemas: List[Dict[str, Any]],
        tool_filter: ToolFilter,
    ) -> List[Dict[str, Any]]:
        """Return only the tool schemas allowed by the runtime filter."""
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
            return [
                schema
                for schema in tool_schemas
                if schema.get("function", {}).get("name") in allowed_tools
            ]

        blocked_tools = self._normalize_tool_filter_values(
            tool_filter["block"], key="block"
        )
        if "*" in blocked_tools:
            return []
        return [
            schema
            for schema in tool_schemas
            if schema.get("function", {}).get("name") not in blocked_tools
        ]

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
        tool_schemas: Optional[List[Dict[str, Any]]],
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Keep tool_choice aligned with the filtered tool set."""
        if not tool_schemas:
            return None

        tool_names = {
            schema.get("function", {}).get("name")
            for schema in tool_schemas
            if schema.get("function", {}).get("name")
        }
        adjusted_tool_choice = tool_choice

        if isinstance(tool_choice, dict):
            choice_name = tool_choice.get("function", {}).get("name")
            if isinstance(choice_name, str) and choice_name not in tool_names:
                adjusted_tool_choice = "auto"
        elif (
            isinstance(tool_choice, str)
            and tool_choice not in {"auto", "required", "none"}
            and tool_choice not in tool_names
        ):
            adjusted_tool_choice = "auto"

        if adjusted_tool_choice != tool_choice and self.config.get("verbose", False):
            cprint(
                f"[{self.name}][tool_choice] Adjusted to `{adjusted_tool_choice}` "
                "after tool filtering",
                bc="y",
                ls="b",
            )

        return adjusted_tool_choice

    def _block_all_tools(self, max_tool_turns: int) -> ToolFilter:
        """Build the internal filter used for the final no-tools round."""
        if self.config.get("verbose", False):
            cprint(
                f"[{self.name}][max_tool_turns] Limit of {max_tool_turns} "
                "turns reached, blocking all tools",
                bc="y",
                ls="b",
            )
        return {"block": "*"}

    # --- Message State Helpers ---

    def _coerce_chat_messages(
        self,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]] = None,
    ) -> ChatMessages:
        if messages is None:
            return ChatMessages()
        if isinstance(messages, ChatMessages):
            return messages
        if isinstance(messages, list):
            return ChatMessages(messages)
        raise TypeError(
            "`messages` must be a `ChatMessages`, a list of mappings or None, "
            f"given `{type(messages)}`"
        )

    # --- Execution Context Resolution ---

    def _get_effective_checkpointer(self):
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is not None:
            return checkpointer
        return get_execution_context().get("checkpoint_store")

    def _get_effective_agent_inbox(self):
        inherited = get_execution_context().get("agent_inbox")
        if inherited is not None:
            return inherited
        return getattr(self, "agent_inbox", None)

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self.agent_inbox = agent_inbox
        self.tool_library.set_agent_inbox(agent_inbox)

    def _raise_if_background_task_stopped(self) -> None:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is None:
            return
        if task_handle.is_stop_requested():
            if self.config.get("verbose", False):
                cprint(
                    f"[{self.name}][task_stop] task_id={task_handle.task_id}",
                    bc="b",
                    ls="b",
                )
            raise TaskStopRequestedError(task_handle.task_id)

    def _handle_control_notifications(
        self,
        notifications: List[AgentNotification],
    ) -> List[AgentNotification]:
        remaining = []
        for notification in notifications:
            if notification.source != "control":
                remaining.append(notification)
                continue

            command = (notification.status or "").lower()
            reason = notification.hint or notification.metadata.get("reason")
            task_handle = get_execution_context().get("task_handle")
            task_id = getattr(task_handle, "task_id", None)

            if command in {"stop", "cancel"}:
                raise TaskStopRequestedError(
                    task_id or get_execution_context().get("run_id") or "unknown",
                    str(reason) if reason else None,
                )
            if command == "pause":
                raise TaskPauseRequestedError(
                    task_id if isinstance(task_id, str) else None,
                    str(reason) if reason else None,
                )

            remaining.append(notification)
        return remaining

    def _drain_inbox_into_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        drain_notifications: bool = True,
    ) -> bool:
        inbox = self._get_effective_agent_inbox()
        if inbox is None:
            return False

        notifications = inbox.drain() if drain_notifications else inbox.peek()
        notifications = self._handle_control_notifications(notifications)
        if not notifications:
            return False

        notification_message = inbox.render(notifications)
        self._persist_notification_message(messages, notification_message)
        return notification_message is not None

    # --- Inbox Delivery ---

    def _build_model_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        drain_notifications: bool = True,
    ) -> List[Mapping[str, Any]]:
        if isinstance(messages, ChatMessages):
            working_messages: Union[ChatMessages, List[Mapping[str, Any]]] = (
                messages if drain_notifications else messages.copy()
            )
        else:
            working_messages = messages if drain_notifications else list(messages)

        self._drain_inbox_into_messages(
            working_messages,
            drain_notifications=drain_notifications,
        )
        if isinstance(working_messages, ChatMessages):
            return working_messages.to_chatml()
        return list(working_messages)

    def _persist_notification_message(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        notification_message: Optional[Mapping[str, Any]],
    ) -> None:
        if notification_message is None:
            return
        if isinstance(messages, ChatMessages):
            if messages.get_active_turn_size() <= 2:
                messages.insert_before_active_turn(notification_message)
            else:
                messages.append(notification_message)
            return
        messages.append(notification_message)

    # --- Session And Run Resolution ---

    def _resolve_session_id(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        session_id: Optional[str],
    ) -> str:
        if isinstance(session_id, str) and session_id:
            return session_id
        if isinstance(messages, ChatMessages) and messages.session_id:
            return messages.session_id
        inherited = get_execution_context().get("session_id")
        if isinstance(inherited, str) and inherited:
            return inherited
        return f"sess_{uuid4().hex}"

    def _resolve_run_id(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        run_id: Optional[str],
    ) -> str:
        if isinstance(run_id, str) and run_id:
            return run_id
        if isinstance(messages, ChatMessages):
            active_turn = messages.get_active_turn()
            if active_turn and isinstance(active_turn.get("turn_id"), str):
                return active_turn["turn_id"]
        inherited = get_execution_context().get("run_id")
        if isinstance(inherited, str) and inherited:
            return inherited
        return f"run_{uuid4().hex[:8]}"

    # --- Chat Turn Tracking ---

    def _start_chat_turn_if_needed(
        self,
        *,
        messages: ChatMessages,
        content: Optional[Union[str, Mapping[str, Any], List[Mapping[str, Any]]]],  # noqa: ARG002
        task: Any,
        vars: Mapping[str, Any],
        turn_id: str,
        kwargs: Mapping[str, Any],
        message: Optional[Union[str, Message, Mapping[str, Any]]],
    ) -> None:
        active_turn = messages.get_active_turn()
        if active_turn is not None:
            if active_turn.get("turn_id") == turn_id:
                return
            messages.end_turn(status="interrupted")

        raw_task_inputs = task
        if raw_task_inputs is _UNSET:
            if isinstance(message, dotdict):
                raw_task_inputs = self._extract_message_values(self.task, message)
            else:
                raw_task_inputs = message

        raw_context_inputs = kwargs.get("task_context")
        if raw_context_inputs is None and isinstance(message, dotdict):
            raw_context_inputs = self._extract_message_values(
                self.task_context, message
            )

        messages.begin_turn(
            inputs=raw_task_inputs,
            context_inputs=raw_context_inputs,
            vars=vars,
            namespace=self.get_module_name(),
            turn_id=turn_id,
        )

    def _append_response_to_chat_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        metadata: Optional[Mapping[str, Any]],  # noqa: ARG002
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return
        if isinstance(raw_response, ModelStreamResponse):
            return
        if response_type != "text_generation" and "structured" not in response_type:
            return

        answer = None
        reasoning_content = None
        if isinstance(raw_response, str):
            answer = raw_response
        elif isinstance(raw_response, Mapping):
            answer = raw_response.get("answer")
            if answer is None and "answer" not in raw_response:
                answer = raw_response.get("text")
            reasoning_content = self._extract_reasoning_content(raw_response)
        elif raw_response is not None:
            answer = str(raw_response)

        if reasoning_content is not None or answer is not None:
            messages.add_assistant_response(
                content=answer,
                reasoning_content=reasoning_content,
            )

    # --- Response Extraction Helpers ---

    def _extract_reasoning_content(
        self,
        payload: Mapping[str, Any],
    ) -> Optional[str]:
        for field in ("reasoning_content", "reasoning_text", "think", "reasoning"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                return value
        return None

    def _finalize_chat_turn(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return

        status = "completed"
        assistant_output: Any = raw_response
        if isinstance(raw_response, ModelStreamResponse):
            status = "streaming"
            assistant_output = None
        elif response_type == "tool_responses":
            status = "tool_responses"

        messages.end_turn(
            assistant_output=assistant_output,
            response_type=response_type,
            response_metadata=metadata,
            status=status,
        )

    # --- Checkpoint Persistence ---

    def _checkpoint_save(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]], None],
        vars: Mapping[str, Any],
        status: str = "running",
    ) -> None:
        checkpointer = self._get_effective_checkpointer()
        if checkpointer is None or not isinstance(messages, ChatMessages):
            return

        turns = messages.turns
        if not turns:
            return

        session_id = messages.session_id or "default"
        run_id = turns[-1]["turn_id"]
        state = {
            "status": status,
            "messages": messages._to_state(),
            "vars": dict(vars) if vars else {},
            "metadata": {
                "namespace": self.get_module_name(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        checkpointer.save_state(self.get_module_name(), session_id, run_id, state)

    async def _acheckpoint_save(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]], None],
        vars: Mapping[str, Any],
        status: str = "running",
    ) -> None:
        checkpointer = self._get_effective_checkpointer()
        if checkpointer is None or not isinstance(messages, ChatMessages):
            return

        turns = messages.turns
        if not turns:
            return

        session_id = messages.session_id or "default"
        run_id = turns[-1]["turn_id"]
        state = {
            "status": status,
            "messages": messages._to_state(),
            "vars": dict(vars) if vars else {},
            "metadata": {
                "namespace": self.get_module_name(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        if hasattr(checkpointer, "asave_state"):
            await checkpointer.asave_state(
                self.get_module_name(),
                session_id,
                run_id,
                state,
            )
        else:
            checkpointer.save_state(self.get_module_name(), session_id, run_id, state)

    def _checkpoint_save_on_error(self, inputs: Mapping[str, Any]) -> None:
        if self._get_effective_checkpointer() is None:
            return
        messages = inputs.get("messages")
        vars = inputs.get("vars", {})
        self._checkpoint_save(messages, vars, status="failed")

    async def _acheckpoint_save_on_error(self, inputs: Mapping[str, Any]) -> None:
        if self._get_effective_checkpointer() is None:
            return
        messages = inputs.get("messages")
        vars = inputs.get("vars", {})
        await self._acheckpoint_save(messages, vars, status="failed")

    # --- Checkpoint Resume ---

    def _try_resume_from_checkpoint(
        self,
        messages_kwarg: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        *,
        scope: Optional[ExecutionScope] = None,
    ) -> Optional[Mapping[str, Any]]:
        checkpointer = self._get_effective_checkpointer()
        if checkpointer is None:
            return None
        run_id = scope.run_id if scope is not None else None
        if not isinstance(run_id, str) or not run_id:
            return None

        effective_session_id = self._resolve_session_id(
            messages=messages_kwarg,
            session_id=scope.session_id if scope is not None else None,
        )
        state = checkpointer.load_state(
            self.get_module_name(),
            effective_session_id,
            run_id,
        )
        if state is None:
            return None
        if state.get("status") in {"completed", "stopped"}:
            return None

        restored = ChatMessages()
        restored._hydrate_state(state.get("messages", {}))
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            session_id=effective_session_id,
            namespace=self.get_module_name(),
            run_id=run_id,
        )
        return {
            "messages": restored,
            "vars": state.get("vars", {}),
            "model_preference": state.get("model_preference"),
            "scope": effective_scope,
        }

    async def _atry_resume_from_checkpoint(
        self,
        messages_kwarg: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        *,
        scope: Optional[ExecutionScope] = None,
    ) -> Optional[Mapping[str, Any]]:
        checkpointer = self._get_effective_checkpointer()
        if checkpointer is None:
            return None
        run_id = scope.run_id if scope is not None else None
        if not isinstance(run_id, str) or not run_id:
            return None

        effective_session_id = self._resolve_session_id(
            messages=messages_kwarg,
            session_id=scope.session_id if scope is not None else None,
        )
        if hasattr(checkpointer, "aload_state"):
            state = await checkpointer.aload_state(
                self.get_module_name(),
                effective_session_id,
                run_id,
            )
        else:
            state = checkpointer.load_state(
                self.get_module_name(),
                effective_session_id,
                run_id,
            )

        if state is None:
            return None
        if state.get("status") in {"completed", "stopped"}:
            return None

        restored = ChatMessages()
        restored._hydrate_state(state.get("messages", {}))
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            session_id=effective_session_id,
            namespace=self.get_module_name(),
            run_id=run_id,
        )
        return {
            "messages": restored,
            "vars": state.get("vars", {}),
            "model_preference": state.get("model_preference"),
            "scope": effective_scope,
        }

    # --- Configuration ---

    def _set_task_context(self, task_context: Optional[Union[str, List[str]]] = None):
        if isinstance(task_context, (str, list)) or task_context is None:
            if isinstance(task_context, str) and task_context == "":
                raise ValueError(
                    f"`task_context` requires a string not emptygiven `{task_context}`"
                )
            if isinstance(task_context, list) and not task_context:
                raise ValueError(
                    f"`task_context` requires a list not emptygiven `{task_context}`"
                )
            self.register_buffer("task_context", task_context)
        else:
            raise TypeError(
                "`task_context` requires a string, list or None"
                f"given `{type(task_context)}`"
            )

    def _set_context_cache(self, context_cache: Optional[str] = None):
        if isinstance(context_cache, str) or context_cache is None:
            self.register_buffer("context_cache", context_cache)
        else:
            raise TypeError(
                "`context_cache` requires a string or None"
                f"given `{type(context_cache)}`"
            )

    def _set_prefilling(self, prefilling: Optional[str] = None):
        if isinstance(prefilling, str) or prefilling is None:
            self.register_buffer("prefilling", prefilling)
        else:
            raise TypeError(
                f"`prefilling` requires a string or Nonegiven `{type(prefilling)}`"
            )

    def _set_tools(
        self,
        tools: Optional[List[Callable]] = None,
        mcp_servers: Optional[List[Mapping[str, Any]]] = None,
    ):
        tools = list(tools or [])
        if self.agent_skill_manager.has_activatable_skills():
            tools.append(ActivateSkill(self.agent_skill_manager))
            if self.agent_skill_manager.has_searchable_skills():
                tools.append(SkillSearch(self.agent_skill_manager))
        self.tool_library = ToolLibrary(
            self.get_module_name(), tools, mcp_servers=mcp_servers
        )
        self.tool_library.set_agent_inbox(self.agent_inbox)

    def _set_skills(self, skills: Optional[SkillsConfig] = None):
        self.agent_skill_manager = AgentSkillManager(skills)

    def _set_generation_schema(
        self, generation_schema: Optional[msgspec.Struct] = None
    ):
        if generation_schema is None or is_subclass_of(
            generation_schema, msgspec.Struct
        ):
            self.register_buffer("generation_schema", generation_schema)
        else:
            raise TypeError(
                "`generation_schema` need be a `msgspec.Struct` or None "
                f"given `{type(generation_schema)}`"
            )

    def _set_model(
        self, model: Union[ChatCompletionModel, ModelGateway, "Generator", str]
    ):
        if isinstance(model, str):
            model = Model.chat_completion(model)
        if isinstance(model, Generator):
            self.generator = model
        else:
            if (
                not hasattr(model, "model_type")
                or model.model_type != "chat_completion"
            ):
                raise TypeError(
                    f"`model` must be a `chat_completion` model, given `{type(model)}`"
                )
            self.generator = Generator(model)

    @property
    def model(self):
        """Access underlying model for convenience."""
        return self.generator.model

    @model.setter
    def model(self, value: Union[ChatCompletionModel, ModelGateway, "Generator", str]):
        self._set_model(value)

    def _set_system_message(self, system_message: Optional[str] = None):
        if isinstance(system_message, str) or system_message is None:
            if isinstance(system_message, str):
                system_message = cleandoc(system_message)
            if (
                hasattr(self.generation_schema, "system_message")
                and self.generation_schema.system_message is not None
            ):
                if system_message is None:
                    system_message = self.generation_schema.system_message
                else:
                    system_message = (
                        self.generation_schema.system_message + system_message
                    )
            self.system_message = Parameter(system_message, PromptSpec.SYSTEM_MESSAGE)
        else:
            raise TypeError(
                "`system_message` requires a string or None "
                f"given `{type(system_message)}`"
            )

    def _set_instructions(self, instructions: Optional[str] = None):
        if isinstance(instructions, str) or instructions is None:
            if isinstance(instructions, str):
                instructions = cleandoc(instructions)
            typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
            if typed_parser_cls is not None:
                instructions = self._format_template(
                    {"instructions": instructions}, typed_parser_cls.template
                )
            self.instructions = Parameter(instructions, PromptSpec.INSTRUCTIONS)
        else:
            raise TypeError(
                f"`instructions` requires a string or None given `{type(instructions)}`"
            )

    def _set_expected_output(self, expected_output: Optional[str] = None):
        if isinstance(expected_output, str) or expected_output is None:  # TODO
            if isinstance(expected_output, str):
                expected_output = cleandoc(expected_output)
            expected_output_temp = ""
            if expected_output:
                expected_output_temp += expected_output
            typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
            if typed_parser_cls is not None:  # Schema as expected output
                response_format = response_format_from_msgspec_struct(
                    self.generation_schema
                )
                schema = typed_parser_cls.schema_from_response_format(response_format)
                content = {"expected_outputs": schema}
                rendered = self._format_template(content, EXPECTED_OUTPUTS_TEMPLATE)
                expected_output_temp += rendered
            self.expected_output = Parameter(
                expected_output_temp or None, PromptSpec.EXPECTED_OUTPUT
            )
        else:
            raise TypeError(
                "`expected_output` requires a string or None "
                f"given `{type(expected_output)}`"
            )

    def _set_examples(
        self,
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None,
    ):
        if isinstance(examples, (str, list)) or examples is None:
            if isinstance(examples, str):
                examples = cleandoc(examples)
            if isinstance(examples, list):
                typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
                collection = ExampleCollection(examples)
                if typed_parser_cls is not None:
                    serialize_func = typed_parser_cls.encode
                else:
                    serialize_func = msgspec_dumps
                examples = collection.get_formatted(serialize_func, serialize_func)
            self.examples = Parameter(examples, PromptSpec.EXAMPLES)
        else:
            raise TypeError(
                f"`examples` requires a List[Example] or None given `{type(examples)}`"
            )

    def _set_messages(self, messages: Optional[str] = None):
        if isinstance(messages, str) or messages is None:
            self.register_buffer("messages", messages)
        else:
            raise TypeError(
                f"`messages` requires a string or None given `{type(messages)}`"
            )

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        """Set agent configuration.

        Args:
            config:
                Dictionary with configuration options.
                Valid keys: "verbose", "return_messages", "tool_choice",
                "stream", "image_block_kwargs", "video_block_kwargs", "include_date"

        Raises:
            TypeError:
                If config is not a dict or None.
            ValueError:
                If invalid keys are provided.
        """
        # Define valid keys for Agent
        valid_keys = {
            "verbose",
            "return_messages",
            "tool_choice",
            "stream",
            "image_block_kwargs",
            "video_block_kwargs",
            "include_date",
            "reasoning_in_response",
            "max_tool_turns",
            "validate_inputs",
        }

        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        invalid_keys = set(config.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid config keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )

        self._validate_config_blocks(config)
        self._validate_config_limits(config)
        self._validate_config_flags(config)

        self.register_buffer("config", config.copy())

    def _validate_config_blocks(self, config: Dict[str, Any]):
        for key in ("image_block_kwargs", "video_block_kwargs"):
            if key in config and not isinstance(config[key], dict):
                raise TypeError(f"`{key}` must be a dict, given `{type(config[key])}`")

    def _validate_config_limits(self, config: Dict[str, Any]):
        if "max_tool_turns" in config:
            max_turns = config["max_tool_turns"]
            if not isinstance(max_turns, int) or max_turns < 1:
                raise ValueError(
                    f"`max_tool_turns` must be a positive integer, "
                    f"given `{config['max_tool_turns']}`"
                )

    def _validate_config_flags(self, config: Dict[str, Any]):
        if "validate_inputs" in config and not isinstance(
            config["validate_inputs"], bool
        ):
            raise TypeError(
                f"`validate_inputs` must be a bool, "
                f"given `{type(config['validate_inputs'])}`"
            )

    def _set_system_extra_message(self, system_extra_message: Optional[str] = None):
        if isinstance(system_extra_message, str) or system_extra_message is None:
            if isinstance(system_extra_message, str):
                system_extra_message = cleandoc(system_extra_message)
            self.register_buffer("system_extra_message", system_extra_message)
        else:
            raise TypeError(
                "`system_extra_message` requires a string or None "
                f"given `{type(system_extra_message)}`"
            )

    def _set_vars(self, vars: Optional[str] = None):
        if isinstance(vars, str) or vars is None:
            self.register_buffer("vars", vars)
        else:
            raise TypeError(f"`vars` requires a string or None given `{type(vars)}`")

    def _set_tool_filter(self, tool_filter: Optional[str] = None):
        if isinstance(tool_filter, str) or tool_filter is None:
            self.register_buffer("tool_filter", tool_filter)
        else:
            raise TypeError(
                f"`tool_filter` requires a string or None given `{type(tool_filter)}`"
            )

    def _set_message_fields(self, message_fields: Optional[Dict[str, Any]] = None):
        """Set message field mappings for Agent.

        Args:
            message_fields: Dictionary mapping field names to their values.
                Valid keys: "task", "task_multimodal", "messages",
                "task_context", "model_preference", "vars", "tool_filter"

        Raises:
            TypeError: If message_fields is not a dict or None
            ValueError: If invalid keys are provided
        """
        # Define valid keys for Agent class
        valid_keys = {
            "task",
            "task_multimodal",
            "messages",
            "task_context",
            "model_preference",
            "vars",
            "tool_filter",
        }

        if message_fields is None:
            # Set all fields to None
            self._set_task(None)
            self._set_task_multimodal(None)
            self._set_model_preference(None)
            self._set_task_context(None)
            self._set_messages(None)
            self._set_vars(None)
            self._set_tool_filter(None)
            return

        if not isinstance(message_fields, dict):
            raise TypeError(
                f"`message_fields` must be a dict or None, given "
                f"`{type(message_fields)}`"
            )

        # Validate keys
        invalid_keys = set(message_fields.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid message_fields keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        # Set each field using its setter, defaulting to None if not provided
        self._set_task(message_fields.get("task"))
        self._set_task_multimodal(message_fields.get("task_multimodal"))
        self._set_model_preference(message_fields.get("model_preference"))
        self._set_task_context(message_fields.get("task_context"))
        self._set_messages(message_fields.get("messages"))
        self._set_vars(message_fields.get("vars"))
        self._set_tool_filter(message_fields.get("tool_filter"))

    def _get_tool_filter_from_message(self, message: Message) -> Optional[ToolFilter]:
        """Read runtime tool filtering rules from a mapped Message field."""
        if isinstance(message, dotdict) and isinstance(self.tool_filter, str):
            return cast(Optional[ToolFilter], message.get(self.tool_filter))
        return None

    def _set_typed_parser(self, typed_parser: Optional[str] = None):
        if isinstance(typed_parser, str) or typed_parser is None:
            if (
                isinstance(typed_parser, str)
                and typed_parser not in typed_parser_registry
            ):
                raise ValueError(
                    f"`typed_parser` supports only `{typed_parser_registry.keys()}`"
                    f" given `{typed_parser}`"
                )
            self.register_buffer("typed_parser", typed_parser)
        else:
            raise TypeError(
                f"`typed_parser` requires a str given `{type(typed_parser)}`"
            )

    def _set_signature(
        self,
        *,
        signature: Optional[Union[str, Signature]] = None,
        examples: Optional[List[Example]] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        instructions: Optional[str] = None,
        system_message: Optional[str] = None,
        typed_parser: Optional[str] = None,
    ):
        if signature is not None:
            typed_parser_cls = typed_parser_registry.get(typed_parser, None)

            examples = examples or []
            output_descriptions = None
            signature_instructions = None

            if isinstance(signature, str):
                input_str_signature, output_str_signature = signature.split("->")
                inputs_info = StructFactory._parse_annotations(input_str_signature)
                outputs_info = StructFactory._parse_annotations(output_str_signature)
            elif issubclass(signature, Signature):
                output_str_signature = signature.get_str_signature().split("->")[-1]
                inputs_info = signature.get_inputs_info()
                outputs_info = signature.get_outputs_info()
                output_descriptions = signature.get_output_descriptions()
                signature_instructions = signature.get_instructions()
                signature_examples = SignatureFactory.get_examples_from_signature(
                    signature
                )
                if signature_examples:
                    examples.extend(signature_examples)
            else:
                raise TypeError(
                    "`signature` requires a string, `Signature` or None "
                    f"given `{type(signature)}`"
                )

            # Validate signature input names don't conflict with reserved kwargs
            input_names = {field.name for field in inputs_info}
            conflicts = input_names & _RESERVED_KWARGS
            if conflicts:
                raise ValueError(
                    f"Signature input names {conflicts} conflict with reserved "
                    f"Agent kwargs. Reserved names: {_RESERVED_KWARGS}. "
                    f"Rename these inputs to avoid conflicts."
                )

            # typed_parser
            self._set_typed_parser(typed_parser)

            # task template - add to templates dict, overriding if present
            task_template = SignatureFactory.get_task_template_from_signature(
                inputs_info
            )
            self.templates["task"] = task_template

            # instructions
            self._set_instructions(instructions or signature_instructions)

            # generation schema
            signature_output_struct = StructFactory.from_signature(
                output_str_signature, "Outputs", output_descriptions
            )
            fused_output_struct = None
            if generation_schema is not None:
                signature_as_type = cast(Type[msgspec.Struct], signature_output_struct)
                if is_optional_field(generation_schema, "final_answer"):
                    signature_as_type = Optional[signature_output_struct]  # type: ignore

                # Merge parent annotations with new final_answer annotation
                merged_annotations = {
                    **generation_schema.__annotations__,
                    "final_answer": signature_as_type,
                }

                fused_output_struct = type(
                    "Output",
                    (generation_schema,),
                    {"__annotations__": merged_annotations},
                )
            self._set_generation_schema(fused_output_struct or signature_output_struct)

            # system message
            self._set_system_message(system_message)

            # expected output
            expected_output = SignatureFactory.get_expected_output_from_signature(
                inputs_info, outputs_info, typed_parser_cls
            )
            self._set_expected_output(expected_output)

            # examples
            self._set_examples(examples)

            # Generate and set annotations from signature inputs
            generated_annotations = generate_annotations_from_signature(
                inputs_info, signature
            )
            self.set_annotations(generated_annotations)

            input_schema = SignatureFactory.get_input_schema_from_signature(
                inputs_info, signature
            )
            self._input_schema = input_schema
            if input_schema is not None:
                self._input_encoder = msgspec.json.Encoder()
                self._input_decoder = msgspec.json.Decoder(input_schema)
            else:
                self._input_encoder = None
                self._input_decoder = None

    def _get_validation_inputs(
        self,
        message: Optional[Union[str, Message, Mapping[str, Any]]],
        task: Any,
        vars: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        if not self.config.get("validate_inputs", False):
            return None
        if getattr(self, "_input_decoder", None) is None:
            return None

        if task is _UNSET:
            if isinstance(message, dotdict):
                task = self._extract_message_values(self.task, message)
            else:
                task = message

        if isinstance(task, Mapping):
            validation_inputs = dict(task)
            validation_inputs.update(vars)
            return validation_inputs
        if task is None and vars:
            return vars
        schema_fields = getattr(self._input_schema, "__struct_fields__", ())
        if len(schema_fields) == 1 and task is not None:
            validation_inputs = {schema_fields[0]: task}
            validation_inputs.update(vars)
            return validation_inputs
        return None

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        if not self.config.get("validate_inputs", False):
            return
        decoder = getattr(self, "_input_decoder", None)
        if decoder is None:
            return

        schema_fields = getattr(self._input_schema, "__struct_fields__", ())
        payload = {field: inputs[field] for field in schema_fields if field in inputs}

        try:
            decoder.decode(self._input_encoder.encode(payload))
        except (msgspec.ValidationError, msgspec.EncodeError, TypeError) as exc:
            raise ValueError(
                f"[{self.name}] Input validation failed: {exc}. "
                f"Expected schema: {self._input_schema.__struct_fields__}"
            ) from exc

    # --- System Prompt ---

    def get_system_prompt(
        self,
        vars: Optional[Mapping[str, Any]] = None,
        tool_names: Optional[set[str]] = None,
    ) -> str:
        """Render the system prompt using the Jinja template.
        Returns an empty string if no segments are provided.
        """
        template_inputs = dotdict(
            system_message=self.system_message.data,
            instructions=self.instructions.data,
            expected_output=self.expected_output.data,
            examples=self.examples.data,
            system_extra_message=self.system_extra_message,
            agent_skills=self.agent_skill_manager.catalog()
            if self.agent_skill_manager.has_activatable_skills()
            else None,
            loaded_agent_skills=self.agent_skill_manager.loaded_content()
            if self.agent_skill_manager.has_skills()
            else None,
            agent_skill_search_enabled=self.agent_skill_manager.has_searchable_skills()
            if self.agent_skill_manager.has_activatable_skills()
            else False,
            agent_skills_enabled=self.agent_skill_manager.has_skills(),
            agent_skill_activation_enabled=self.agent_skill_manager.has_activatable_skills(),
            tool_usage_guidance=self.tool_library.get_tool_usage_guidance(tool_names),
        )

        if self.config.get("include_date", False):
            now = datetime.now(tz=timezone.utc)
            # Format: "Monday, December 09, 2025"
            template_inputs.current_date = now.strftime("%A, %B %d, %Y")

        system_prompt = self._format_template(
            template_inputs, self.system_prompt_template
        )

        if vars:  # Runtime inputs to system template
            system_prompt = self._format_template(vars, system_prompt)
        return system_prompt

    @property
    def system_prompt_template(self) -> str:
        """Get the system prompt template.

        Returns the custom template if provided in templates dict,
        otherwise returns the default SYSTEM_PROMPT_TEMPLATE.
        """
        return self.templates.get("system_prompt", SYSTEM_PROMPT_TEMPLATE)
