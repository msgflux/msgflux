from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import (
    Any, Callable, List, Literal, Mapping, Optional, Union, Tuple, Type, cast
)

import msgspec

from msgflux.dotdict import dotdict
from msgflux.dsl.signature import Signature, SignatureFactory
from msgflux.dsl.typed_parsers.registry import typed_parser_registry
from msgflux.examples import Example, ExampleCollection
from msgflux.generation.control_flow import ToolFlowControl
from msgflux.generation.templates import (
    PromptSpec, EXPECTED_OUTPUTS_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
)
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary, ToolResponses
from msgflux.nn.parameter import Parameter
from msgflux.telemetry.span import instrument_agent_prepare_model_execution
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.console import cprint
from msgflux.utils.inspect import get_filename, get_mime_type
from msgflux.utils.msgspec import StructFactory, is_optional_field, msgspec_dumps
from msgflux.utils.validation import is_subclass_of
from msgflux.utils.xml import apply_xml_tags


class Agent(Module):
    """Agent is a Module type that uses language models to solve tasks.

    An Agent can perform actions in an environment using tools calls.
    For an Agent, a tool is any callable object.

    An Agent can handle multimodal inputs and outputs.
    """

    _supported_outputs: List[str] = [
        "reasoning_structured",
        "reasoning_text_generation",
        "structured",
        "text_generation",
        "audio_generation",
        "audio_text_generation",
        "tool_responses"
    ]

    def __init__(
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelGateway],
        *,
        system_message: Optional[str] = None,
        instructions: Optional[str] = None,
        expected_output: Optional[str] = None,
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None,
        system_extra_message: Optional[str] = None,
        include_date: Optional[bool] = False,
        stream: Optional[bool] = False,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
        task_inputs: Optional[Union[str, Mapping[str, str]]] = None,
        task_multimodal_inputs: Optional[Mapping[str, List[str]]] = None,
        task_messages: Optional[str] = None,
        task_template: Optional[str] = None,
        context_inputs: Optional[Union[str, List[str]]] = None,
        context_cache: Optional[str] = None,
        context_inputs_template: Optional[str] = None,
        model_preference: Optional[str] = None,
        prefilling: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        typed_parser: Optional[str] = None,
        response_mode: Optional[str] = "plain_response",
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        mcp_servers: Optional[List[Mapping[str, Any]]] = None,
        vars: Optional[str] = None,
        response_template: Optional[str] = None,
        fixed_messages: Optional[List[Mapping[str, Any]]] = None,
        signature: Optional[Union[str, Signature]] = None,
        return_model_state: Optional[bool] = False,
        verbose: Optional[bool] = False,
        description: Optional[str] = None,
        annotations: Optional[Mapping[str, type]] = None,
        image_detail: Optional[Literal["high", "low"]] = None,
    ):
        """Args:
        name:
            Agent name in snake case format.
        model:
            Chat Completation Model client.
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
        include_date:
            If True, include the current date in the system prompt.
        stream:
            If the response is transmitted on-fly.
        input_guardrail:
            Guardrail to input.
        output_guardrail:
            Guardrail to output.
        task_inputs:
            Field of the Message object that will be the input to the task.
        task_multimodal_inputs:
            Map datatype (image, video, audio, file) to field of the Message object.
            !!! example
                # single audio
                task_multimodal_inputs={"audio": "audio.user"}
                # multi image
                task_multimodal_inputs={"image": ["images.user", "image.mask"]}
                # single video
                task_multimodal_inputs={"video": "video.path"}
        task_messages:
            Field of the Message object that will be a list of chats in
            ChatML format.
        task_template:
            A Jinja template to format task.
        context_inputs:
            Field of the Message object that will be the context to the task.
        context_cache:
            A fixed context.
        context_inputs_template:
            A template to context inputs.
        model_preference:
            Field of the Message object that will be the model preference.
            This is only valid if the model is of type ModelGateway.
        prefilling:
            Forces an initial message from the model. From that message it
            will continue its response from there.
        generation_schema:
            Schema that defines how the output should be structured.
        typed_parser:
            Converts the model raw output into a typed-dict. Supported parser:
            `typed_xml`.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        vars:
            Field of the Message object that will be the inputs to templates and tools.
        tools:
            A list of callable objects.
        tool_choice:
            By default the model will determine when and how many tools to use.
            You can force specific behavior with the tool_choice parameter.
                1. auto:
                    Call zero, one, or multiple functions. tool_choice: "auto"
                2. required:
                    Call one or more functions. tool_choice: "required"
                3. Forced Function:
                    Call exactly one specific function. E.g. "add".
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
        response_template:
            A Jinja template to format response.
        fixed_messages:
            A fixed list of chats in ChatML format.
        signature:
            A DSPy-based signature. A signature creates a task_template,
            a generation_schema, instructions and examples (both if passed).
            Can be combined with standard generation_schemas like `ReAct` and
            `ChainOfThought`. Can also be combined with `typed_parser`.
        return_model_state:
            If True, returns a dictionary containing model_state and response.
        verbose:
            If True, prints the model output and tool calls and their responses
            to the console.
        description:
            The Agent description. It's useful when using an agent-as-a-tool.
        annotations
            Define the input and output annotations to use the agent-as-a-function.
        image_detail:
            Controls the detail level for image processing.
            "high" enables detailed patch analysis, "low" uses lower resolution.
        """
        if annotations is None:
            annotations = {"message": str, "return": str}
        super().__init__()

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if output_guardrail is not None:
                raise ValueError("`output_guardrail` is not `stream=True` compatible")

            if response_template is not None:
                raise ValueError("`response_template` is not `stream=True` compatible")

            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

        if signature is not None:
            signature_params = dotdict(
                signature=signature,
                examples=examples,
                instructions=instructions,
                system_message=system_message,
                typed_parser=typed_parser
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
            self._set_task_template(task_template)            

        self.set_name(name)
        self.set_description(description)
        self.set_annotations(annotations)
        self._set_context_cache(context_cache)
        self._set_context_inputs(context_inputs)
        self._set_context_inputs_template(context_inputs_template)
        self._set_fixed_messages(fixed_messages)
        self._set_input_guardrail(input_guardrail)
        self._set_output_guardrail(output_guardrail)
        self._set_task_messages(task_messages)
        self._set_model(model)
        self._set_model_preference(model_preference)
        self._set_prefilling(prefilling)
        self._set_system_extra_message(system_extra_message)
        self._set_include_date(include_date)
        self._set_response_mode(response_mode)
        self._set_return_model_state(return_model_state)
        self._set_stream(stream)
        self._set_response_template(response_template)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_task_inputs(task_inputs)
        self._set_vars(vars)
        self._set_tool_choice(tool_choice)
        self._set_tools(tools, mcp_servers)
        self._set_verbose(verbose)
        self._set_image_detail(image_detail)

    def forward(
        self, message: Optional[Union[str, Mapping[str, Any], Message]] = None, **kwargs
    ) -> Union[str, Mapping[str, None], ModelStreamResponse, Message]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(prefilling=self.prefilling, **inputs)
        response = self._process_model_response(message, model_response, **inputs)
        return response

    async def aforward(
        self, message: Optional[Union[str, Mapping[str, Any], Message]] = None, **kwargs
    ) -> Union[str, Mapping[str, None], ModelStreamResponse, Message]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = await self._aexecute_model(prefilling=self.prefilling, **inputs)
        response = await self._aprocess_model_response(message, model_response, **inputs)
        return response

    def _execute_model(
        self,
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(
            model_state=model_state, prefilling=prefilling,
            model_preference=model_preference, vars=vars
        )
        if self.input_guardrail:
            self._execute_input_guardrail(model_execution_params)
        if self.verbose:
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        model_response = self.model(**model_execution_params)
        return model_response

    async def _aexecute_model(
        self,
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(
            model_state=model_state, prefilling=prefilling,
            model_preference=model_preference, vars=vars
        )
        if self.input_guardrail:
            await self._aexecute_input_guardrail(model_execution_params)
        if self.verbose:
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        model_response = await self.model.acall(**model_execution_params)
        return model_response

    @instrument_agent_prepare_model_execution
    def _prepare_model_execution(
        self,
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],        
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> Mapping[str, Any]:
        # model_state, prefilling, model_preference, vars
        agent_state = []

        if self.fixed_messages:
            agent_state.extend(self.fixed_messages)

        agent_state.extend(model_state)

        system_prompt = self._get_system_prompt(vars)

        tool_schemas = self.tool_library.get_tool_json_schemas()
        if not tool_schemas:
            tool_schemas = None

        tool_choice = self.tool_choice

        if is_subclass_of(self.generation_schema, ToolFlowControl) and tool_schemas:
            tools_template = self.generation_schema.tools_template
            inputs = {"tool_schemas": tool_schemas, "tool_choice": tool_choice}
            flow_control_tools = self._format_template(inputs, tools_template)
            if system_prompt:
                system_prompt = flow_control_tools + "\n\n" + system_prompt
            else:
                system_prompt = flow_control_tools
            tool_schemas = None  # Disable tool_schemas to controlflow preference
            tool_choice = None  # Disable tool_choice to controlflow preference

        model_execution_params = dotdict(
            messages=agent_state,
            system_prompt=system_prompt or None,
            prefilling=prefilling,
            stream=self.stream,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
            generation_schema=self.generation_schema,
            typed_parser=self.typed_parser,
        )

        if model_preference:
            model_execution_params.model_preference = model_preference

        return model_execution_params

    def _prepare_input_guardrail_execution(
        self, model_execution_params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        messages = model_execution_params.get("messages")
        last_message = messages[-1]
        if isinstance(last_message.get("content"), list):
            if last_message.get("content")[0]["type"] == "image_url":
                data = [last_message]
            else: # audio, file
                data = last_message.get("content")[-1]  # text input
        else:
            data = last_message.get("content")
        guardrail_params = {"data": data}
        return guardrail_params

    def _process_model_response(
        self,
        message: Union[str, Mapping[str, str], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
    ) -> Union[str, Mapping[str, str], Message, ModelStreamResponse]:
        if "tool_call" in model_response.response_type:
            model_response, model_state = self._process_tool_call_response(
                model_response, model_state, vars, model_preference
            )
        elif is_subclass_of(self.generation_schema, ToolFlowControl):
            model_response, model_state = self._process_tool_flow_control_response(
                model_response, model_state, vars, model_preference
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"

        if response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, response_type, model_state, message, vars
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    async def _aprocess_model_response(
        self,
        message: Union[str, Mapping[str, str], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
    ) -> Union[str, Mapping[str, str], Message, ModelStreamResponse]:
        if "tool_call" in model_response.response_type:
            model_response, model_state = await self._aprocess_tool_call_response(
                model_response, model_state, vars, model_preference
            )
        elif is_subclass_of(self.generation_schema, ToolFlowControl):
            model_response, model_state = await self._aprocess_tool_flow_control_response(
                model_response, model_state, vars, model_preference
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"

        if response_type in self._supported_outputs:
            response = await self._aprepare_response(
                raw_response, response_type, model_state, message, vars
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    def _process_tool_flow_control_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Mapping[str, Any],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
    ) -> Tuple[Union[str, Mapping[str, Any], ModelStreamResponse], Mapping[str, Any]]:
        """Handle the fields returned by `ReAct`. If the fields are different,
        you must rewrite this function.
        """
        while True:
            raw_response = self._extract_raw_response(model_response)

            if getattr(raw_response, "final_answer", None):
                return model_response, model_state

            if getattr(raw_response, "current_step", None):
                step = raw_response.current_step
                actions = step.actions
                reasoning = step.thought

                if self.verbose:
                    repr = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                    cprint(repr, bc="br2", ls="b")

                for act in actions:
                    act.id = str(uuid4())  # Add tool_id

                tool_callings = [(act.id, act.name, act.arguments) for act in actions]
                tool_results = self._process_tool_call(tool_callings, model_state, vars)

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict().pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    # TODO converter tool calls em tool call msgs
                    return tool_responses, model_state

                for act in actions:  # Add results
                    result = tool_results.get_by_id(act.id).result
                    error = tool_results.get_by_id(act.id).error
                    act.result = result or error

                # Compact steps history
                if model_state and model_state[-1].get("role") == "assistant":
                    last_react_msg = model_state[-1].get("content")
                    react_state = msgspec.json.decode(last_react_msg)
                    react_state.append(raw_response)
                    model_state[-1] = ChatBlock.assist(react_state)
                else:
                    react_state = [raw_response]
                    model_state.append(ChatBlock.assist(react_state))

            model_response = self._execute_model(
                model_state=model_state,
                model_preference=model_preference,
                vars=vars
            )

    async def _aprocess_tool_flow_control_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Mapping[str, Any],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
    ) -> Tuple[Union[str, Mapping[str, Any], ModelStreamResponse], Mapping[str, Any]]:
        """Async version of _process_tool_flow_control_response.
        Handle the fields returned by `ReAct`. If the fields are different,
        you must rewrite this function.
        """
        while True:
            raw_response = self._extract_raw_response(model_response)

            if getattr(raw_response, "final_answer", None):
                return model_response, model_state

            if getattr(raw_response, "current_step", None):
                step = raw_response.current_step
                actions = step.actions
                reasoning = step.thought

                if self.verbose:
                    repr = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                    cprint(repr, bc="br2", ls="b")

                for act in actions:
                    act.id = str(uuid4())  # Add tool_id

                tool_callings = [(act.id, act.name, act.arguments) for act in actions]
                tool_results = await self._aprocess_tool_call(tool_callings, model_state, vars)

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict().pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    # TODO converter tool calls em tool call msgs
                    return tool_responses, model_state

                for act in actions:  # Add results
                    result = tool_results.get_by_id(act.id).result
                    error = tool_results.get_by_id(act.id).error
                    act.result = result or error

                # Compact steps history
                if model_state and model_state[-1].get("role") == "assistant":
                    last_react_msg = model_state[-1].get("content")
                    react_state = msgspec.json.decode(last_react_msg)
                    react_state.append(raw_response)
                    model_state[-1] = ChatBlock.assist(react_state)
                else:
                    react_state = [raw_response]
                    model_state.append(ChatBlock.assist(react_state))

            model_response = await self._aexecute_model(
                model_state=model_state,
                model_preference=model_preference,
                vars=vars
            )

    def _process_tool_call_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Mapping[str, Any],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None
    ) -> Tuple[Union[str, Mapping[str, Any], ModelStreamResponse], Mapping[str, Any]]:
        """ToolCall example: [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        while True:
            if model_response.response_type == "tool_call":
                raw_response = model_response.data
                reasoning = raw_response.reasoning

                if self.verbose:
                    if reasoning:
                        repr = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr, bc="br2", ls="b")

                tool_callings = raw_response.get_calls()
                tool_results = self._process_tool_call(tool_callings, model_state, vars)

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict()
                    tool_calls.pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, model_state

                id_results = {
                    call.id: call.result or call.error
                    for call in tool_results.tool_calls
                }
                raw_response.insert_results(id_results)
                tool_responses_message = raw_response.get_messages()
                model_state.extend(tool_responses_message)
            else:
                return model_response, model_state

            model_response = self._execute_model(
                model_state=model_state,
                model_preference=model_preference,
                vars=vars
            )

    async def _aprocess_tool_call_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Mapping[str, Any],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None
    ) -> Tuple[Union[str, Mapping[str, Any], ModelStreamResponse], Mapping[str, Any]]:
        """Async version of _process_tool_call_response.
        ToolCall example: [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        while True:
            if model_response.response_type == "tool_call":
                raw_response = model_response.data
                reasoning = raw_response.reasoning

                if self.verbose:
                    if reasoning:
                        repr = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr, bc="br2", ls="b")

                tool_callings = raw_response.get_calls()
                tool_results = await self._aprocess_tool_call(tool_callings, model_state, vars)

                if tool_results.return_directly:
                    tool_calls = tool_results.to_dict()
                    tool_calls.pop("return_directly")
                    tool_calls["reasoning"] = reasoning
                    tool_responses = dotdict(tool_responses=tool_calls)
                    return tool_responses, model_state

                id_results = {
                    call.id: call.result or call.error
                    for call in tool_results.tool_calls
                }
                raw_response.insert_results(id_results)
                tool_responses_message = raw_response.get_messages()
                model_state.extend(tool_responses_message)
            else:
                return model_response, model_state

            model_response = await self._aexecute_model(
                model_state=model_state,
                model_preference=model_preference,
                vars=vars
            )

    def _process_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        if self.verbose:
            for call in tool_callings:
                repr = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr, bc="br2", ls="b")
        tool_results = self.tool_library(
            tool_callings=tool_callings,
            model_state=model_state,
            vars=vars,
        )
        if self.verbose:
            repr = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr += " return directly"
            cprint(repr, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr, ls="b")
        return tool_results

    async def _aprocess_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        model_state: List[Mapping[str, Any]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        """Async version of _process_tool_call."""
        if self.verbose:
            for call in tool_callings:
                repr = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr, bc="br2", ls="b")
        tool_results = await self.tool_library.acall(
            tool_callings=tool_callings,
            model_state=model_state,
            vars=vars,
        )
        if self.verbose:
            repr = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr += " return directly"
            cprint(repr, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr, ls="b")
        return tool_results

    def _prepare_response(
        self,
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        model_state: List[Mapping[str, Any]],
        message: Union[str, Mapping[str, Any], Message],
        vars: Mapping[str, Any],
    ) -> Union[str, Mapping[str, Any], ModelStreamResponse]:
        formatted_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            if response_type == "text_generation" or "structured" in response_type:
                if self.verbose:
                    cprint(f"[{self.name}][response] {raw_response}", bc="y", ls="b")
                if self.output_guardrail:
                    self._execute_output_guardrail(raw_response)
                if self.response_template:
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

        response = formatted_response or raw_response
        if self.return_model_state:
            if response_type == "tool_responses":
                response.model_state = model_state
            else:
                response = dotdict(model_response=response, model_state=model_state)
        return self._define_response_mode(response, message)

    async def _aprepare_response(
        self,
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        model_state: List[Mapping[str, Any]],
        message: Union[str, Mapping[str, Any], Message],
        vars: Mapping[str, Any],
    ) -> Union[str, Mapping[str, Any], ModelStreamResponse]:
        """Async version of _prepare_response with async output guardrail support."""
        formatted_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            if response_type == "text_generation" or "structured" in response_type:
                if self.verbose:
                    cprint(f"[{self.name}][response] {raw_response}", bc="y", ls="b")
                if self.output_guardrail:
                    await self._aexecute_output_guardrail(raw_response)
                if self.response_template:
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

        response = formatted_response or raw_response
        if self.return_model_state:
            if response_type == "tool_responses":
                response.model_state = model_state
            else:
                response = dotdict(model_response=response, model_state=model_state)
        return self._define_response_mode(response, message)

    def _prepare_output_guardrail_execution(
        self, model_response: Union[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        if isinstance(model_response, str):
            data = model_response
        else:
            data = str(model_response)
        guardrail_params = {"data": data}
        return guardrail_params

    def _prepare_task(
        self, message: Union[str, Message, Mapping[str, str]], **kwargs
    ) -> Mapping[str, Any]:
        """Prepare model input in ChatML format and execution params."""
        vars = kwargs.pop("vars", {})
        if (
            not vars
            and isinstance(message, Message)
            and self.vars is not None
        ):
            vars = message.get(self.vars, {})

        task_messages = kwargs.pop("task_messages", None)
        if (
            task_messages is None
            and isinstance(message, Message)
            and self.vars is not None
        ):
            task_messages = self._get_content_from_message(self.task_messages, message)

        content = self._process_task_inputs(message, vars=vars, **kwargs)

        if content is None and task_messages is None:
            raise ValueError("No data was detected to make the model input")

        if content is not None:            
            chat_content = [ChatBlock.user(content)]
            if task_messages is None:
                model_state = chat_content
            else:
                task_messages.extend(chat_content)
                model_state = task_messages
        else:
            model_state = task_messages

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {
            "model_state": model_state,
            "model_preference": model_preference,
            "vars": vars,
        }

    def _process_task_inputs(
        self,
        message: Union[str, Message, Mapping[str, str]],
        vars: Mapping[str, Any],
        **kwargs
    ) -> Optional[Union[str, Mapping[str, Any]]]:
        content = ""

        context_content = self._context_manager(message, vars=vars, **kwargs)
        if context_content:
            content += context_content

        if isinstance(message, Message):
            task_inputs = self._extract_message_values(self.task_inputs, message)
        else:
            task_inputs = message

        if task_inputs is None and self.task_template is None:
            return None

        if self.task_template:
            if task_inputs:       
                if isinstance(task_inputs, str):
                    pre_task = self._format_task_template(vars)
                    task_content = self._format_template(task_inputs, pre_task)
                elif isinstance(task_inputs, dict):
                    task_inputs.update(vars)
                    task_content = self._format_task_template(task_inputs)
            # It's possible to use `task_template` as the default task message
            # if no `task_inputs` is selected. This can be useful for multimodal
            # models that require a text message to be sent along with the data
            else:
                if vars:
                    task_content = self._format_task_template(vars)
                else:
                    task_content = self.task_template
        else:
            task_content = task_inputs
            if isinstance(task_content, Mapping): # dict -> str
                task_content = "\n".join(f"{k}: {v}" for k, v in task_content.items())

        task_content = apply_xml_tags("task", task_content)
        content += task_content
        content = content.strip() # Remove whitespace

        multimodal_content = self._process_task_multimodal_inputs(message, **kwargs)
        if multimodal_content:            
            multimodal_content.append(ChatBlock.text(content))
            return multimodal_content
        return content

    def _context_manager( # noqa: C901
        self,
        message: Union[str, Message, Mapping[str, str]],
        vars: Mapping[str, Any],
        **kwargs
    ) -> Optional[str]:
        """Mount context."""
        context_content = ""

        if self.context_cache:  # Fixed Context Cache
            context_content += self.context_cache

        context_inputs = None
        runtime_context_inputs = kwargs.pop("context_inputs", None)
        if runtime_context_inputs is not None:
            context_inputs = runtime_context_inputs
        elif isinstance(message, Message):
            context_inputs = self._extract_message_values(self.context_inputs, message)

        if context_inputs is not None:
            if self.context_inputs_template:
                if isinstance(context_inputs, Mapping):                
                    context_inputs.update(vars)
                    msg_context = self._format_template(
                        context_inputs, self.context_inputs_template
                    )
                else:
                    pre_msg_context = self._format_template(
                        vars, self.context_inputs_template
                    )
                    msg_context = self._format_template(
                        context_inputs, pre_msg_context
                    )                                    
            elif isinstance(context_inputs, str):
                msg_context = context_inputs
            elif isinstance(context_inputs, list):
                msg_context = " ".join(
                    str(v) for v in context_inputs if v is not None
                )
            elif isinstance(context_inputs, dict):
                msg_context = "\n".join(
                    f"{k}: {v if not isinstance(v, list) else ', '.join(v)}"
                    for k, v in context_inputs.items()
                )
            context_content += msg_context

        if context_content:
            if vars:
                context_content = self._format_template(vars, context_content)
            return apply_xml_tags("context", context_content) + "\n\n"
        return None

    def _process_task_multimodal_inputs(
        self, message: Union[str, Message, Mapping[str, str]], **kwargs
    ) -> Optional[List[Mapping[str, Any]]]:
        """Processes multimodal inputs (image, audio, video, file) via kwargs or message.
        Returns a list of multimodal content in ChatML format.
        """
        multimodal_paths = None
        task_multimodal_inputs = kwargs.get("task_multimodal_inputs", None)
        if task_multimodal_inputs is not None:
            multimodal_paths = task_multimodal_inputs
        elif isinstance(message, Message) and self.task_multimodal_inputs is not None:
            multimodal_paths = self._extract_message_values(
                self.task_multimodal_inputs, message
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

    def _format_image_input(self, image_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the image input for the model."""
        encoded_image = self._prepare_data_uri(image_source, force_encode=False)

        if not encoded_image:
            return None

        if not encoded_image.startswith("http"):
            # Try to guess from the original source
            mime_type = get_mime_type(image_source)
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"  # Fallback
            encoded_image = f"data:{mime_type};base64,{encoded_image}"

        return ChatBlock.image(encoded_image, detail=self.image_detail)

    def _format_video_input(self, video_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the video input for the model."""
        # Check if it's a URL
        if video_source.startswith("http://") or video_source.startswith("https://"):
            return ChatBlock.video(video_source)

        # Otherwise, encode as base64
        encoded_video = self._prepare_data_uri(video_source, force_encode=True)

        if not encoded_video:
            return None

        # Get MIME type or use mp4 as fallback
        mime_type = get_mime_type(video_source)
        if not mime_type.startswith("video/"):
            mime_type = "video/mp4"  # Fallback

        video_data_uri = f"data:{mime_type};base64,{encoded_video}"

        return ChatBlock.video(video_data_uri)

    def _format_audio_input(self, audio_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the audio input for the model."""
        base64_audio = self._prepare_data_uri(audio_source, force_encode=True)

        if not base64_audio:
            return None

        audio_format_suffix = Path(audio_source).suffix.lstrip(".")
        mime_type = get_mime_type(audio_source)
        if not mime_type.startswith("audio/"):
            # If MIME type is not audio, use suffix or fallback
            audio_format_for_uri = (
                audio_format_suffix if audio_format_suffix else "mpeg"
            )  # fallback
            mime_type = f"audio/{audio_format_for_uri}"

        # Use suffix like 'format' if available, otherwise extract from mime type
        audio_format = (
            audio_format_suffix if audio_format_suffix else mime_type.split("/")[-1]
        )        

        return ChatBlock.audio(base64_audio, audio_format)

    def _format_file_input(self, file_source: str) -> Optional[Mapping[str, Any]]:
        """Formats the file input for the model."""
        base64_file = self._prepare_data_uri(file_source, force_encode=True)

        if not base64_file:
            return None

        filename = get_filename(file_source)
        mime_type = get_mime_type(file_source)

        if mime_type == "application/octet-stream" and filename.lower().endswith(
            ".pdf"
        ):
            mime_type = "application/pdf"

        file_data_uri = f"data:{mime_type};base64,{base64_file}"

        return ChatBlock.file(filename, file_data_uri)

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(
            prefilling=self.prefilling, **inputs
        )
        return model_execution_params

    def _set_context_inputs(
        self, context_inputs: Optional[Union[str, List[str]]] = None
    ):
        if isinstance(context_inputs, (str, list)) or context_inputs is None:
            if isinstance(context_inputs, str) and context_inputs == "":
                raise ValueError(
                    "`context_inputs` requires a string not empty"
                    f"given `{context_inputs}`"
                )
            if isinstance(context_inputs, list) and not context_inputs:
                raise ValueError(
                    "`context_inputs` requires a list not empty"
                    f"given `{context_inputs}`"
                )
            self.register_buffer("context_inputs", context_inputs)
        else:
            raise TypeError(
                "`context_inputs` requires a string, list or None"
                f"given `{type(context_inputs)}`"
            )

    def _set_context_cache(self, context_cache: Optional[str] = None):
        if isinstance(context_cache, str) or context_cache is None:
            self.register_buffer("context_cache", context_cache)
        else:
            raise TypeError(
                "`context_cache` requires a string or None"
                f"given `{type(context_cache)}`"
            )

    def _set_context_inputs_template(
        self, context_inputs_template: Optional[str] = None
    ):
        if isinstance(context_inputs_template, str) or context_inputs_template is None:
            self.register_buffer("context_inputs_template", context_inputs_template)
        else:
            raise TypeError(
                "`context_inputs_template` requires a string or None"
                f"given `{type(context_inputs_template)}`"
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
        mcp_servers: Optional[List[Mapping[str, Any]]] = None
    ):
        self.tool_library = ToolLibrary(
            self.get_module_name(),
            tools or [],
            mcp_servers=mcp_servers
        )

    def _set_fixed_messages(
        self, fixed_messages: Optional[List[Mapping[str, Any]]] = None
    ):
        if (
            isinstance(fixed_messages, list)
            and all(dict(obj) for obj in fixed_messages)
        ) or fixed_messages is None:
            self.register_buffer("fixed_messages", fixed_messages)
        else:
            raise TypeError(
                "`fixed_messages` need be a list of dict or None"
                f"given `{type(fixed_messages)}`"
            )

    def _set_generation_schema(
        self, generation_schema: Optional[msgspec.Struct] = None
    ):
        if (
            generation_schema is None 
            or
            is_subclass_of(generation_schema, msgspec.Struct)
        ):
            self.register_buffer("generation_schema", generation_schema)
        else:
            raise TypeError(
                "`generation_schema` need be a `msgspec.Struct` or None "
                f"given `{type(generation_schema)}`"
            )

    def _set_model(self, model: Union[ChatCompletionModel, ModelGateway]):
        if model.model_type == "chat_completion":
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` need be a `chat completion` model, given `{type(model)}`"
            )

    def _set_tool_choice(self, tool_choice: Optional[str] = None):
        if isinstance(tool_choice, str) or tool_choice is None:
            self.register_buffer("tool_choice", tool_choice)
        else:
            raise TypeError(
                f"`tool_choice` need be a str or None given `{type(tool_choice)}`"
            )

    def _set_system_message(self, system_message: Optional[str] = None):
        if isinstance(system_message, str) or system_message is None:
            if (
                hasattr(self.generation_schema, "system_message")
                and 
                self.generation_schema.system_message is not None
            ):
                if system_message is None:
                    system_message = self.generation_schema.system_message
                else:
                    system_message = self.generation_schema.system_message + system_message
            self.system_message = Parameter(system_message, PromptSpec.SYSTEM_MESSAGE)
        else:
            raise TypeError(
                "`system_message` requires a string or None "
                f"given `{type(system_message)}`"
            )

    def _set_include_date(self, include_date: Optional[bool] = False): # noqa: FBT001, FBT002
        if isinstance(include_date, bool):
            self.register_buffer("include_date", include_date)
        else:
            raise TypeError(
                f"`include_date` requires a bool given `{type(include_date)}`"
            )

    def _set_instructions(self, instructions: Optional[str] = None):
        if isinstance(instructions, str) or instructions is None:
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
        if isinstance(expected_output, str) or expected_output is None: # TODO
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
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None
    ):
        if isinstance(examples, (str, list)) or examples is None:
            if isinstance(examples, list):
                typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
                collection = ExampleCollection(examples)
                if typed_parser_cls is not None:
                    T = typed_parser_cls.repr_from_dict
                else:
                    T = msgspec_dumps
                examples = collection.get_formatted(T, T)
            self.examples = Parameter(examples, PromptSpec.EXAMPLES)
        else:
            raise TypeError(
                f"`examples` requires a List[Example] or None given `{type(examples)}`"
            )

    def _set_return_model_state(
        self, return_model_state: Optional[bool] = False  # noqa: FBT001, FBT002
    ):
        if isinstance(return_model_state, bool):
            self.register_buffer("return_model_state", return_model_state)
        else:
            raise TypeError(
                "`return_model_state` requires a bool "
                f"given `{type(return_model_state)}`"
            )

    def _set_task_messages(self, task_messages: Optional[str] = None):
        if isinstance(task_messages, str) or task_messages is None:
            self.register_buffer("task_messages", task_messages)
        else:
            raise TypeError(
                "`task_messages` requires a string or None "
                f"given `{type(task_messages)}`"
            )

    def _set_system_extra_message(self, system_extra_message: Optional[str] = None):
        if isinstance(system_extra_message, str) or system_extra_message is None:
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
            raise TypeError(
                "`vars` requires a string or None "
                f"given `{type(vars)}`"
            )

    def _set_typed_parser(self, typed_parser: Optional[str] = None):
        if isinstance(typed_parser, str) or typed_parser is None:
            if (
                isinstance(typed_parser, str)
                and 
                typed_parser not in typed_parser_registry
            ):
                raise ValueError(
                    f"`typed_parser` supports only `{typed_parser_registry.keys()}`"
                    f" given `{typed_parser}`"
                )
            self.register_buffer("typed_parser", typed_parser)
        else:
            raise TypeError(f"`typed_parser` requires a str given `{type(typed_parser)}`")

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

            # typed_parser
            self._set_typed_parser(typed_parser)

            # task template
            task_template = SignatureFactory.get_task_template_from_signature(
                inputs_info
            )
            self._set_task_template(task_template)

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
                    signature_as_type = Optional[signature_output_struct] # type: ignore
                class Output(generation_schema):  
                    final_answer: signature_as_type # type: ignore
                fused_output_struct = Output                
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

    def _set_image_detail(
        self, image_detail: Optional[Literal["high", "low"]] = None
    ):
        if image_detail in ("high", "low", None):
            self.register_buffer("image_detail", image_detail)
        else:
            raise ValueError(
                "`image_detail` must be 'high', 'low' or None "
                f"given `{image_detail}`"
            )

    def _get_system_prompt(
        self, vars: Optional[Mapping[str, Any]] = None
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
        )

        if self.include_date:
            now = datetime.now(tz=timezone.utc)
            template_inputs.current_date = now.strftime("%m/%d/%Y")

        system_prompt = self._format_template(template_inputs, SYSTEM_PROMPT_TEMPLATE)

        if vars:  # Runtime inputs to system template
            system_prompt = self._format_template(vars, system_prompt)
        return system_prompt
