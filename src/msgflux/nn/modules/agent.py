from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import msgspec

from msgflux.dotdict import dotdict
from msgflux.dsl.signature import (
    SIGNATURE_SYSTEM_MESSAGES,
    Signature,
    get_examples_from_signature,
    get_expected_output_from_signature,
    get_task_template_from_signature,
)
from msgflux.generation.reasoning.react import ReAct
from msgflux.generation.templates import (
    SIGNATURE_DEFAULT_SYSTEM_MESSAGE,
    SYSTEM_PROMPT_TEMPLATE,
    TYPED_XML_TEMPLATE,
    PromptSpec,
)
from msgflux.logger import logger
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.parameter import Parameter
from msgflux.telemetry.span import instrument_agent_prepare_model_execution
from msgflux.utils.chat import (
    adapt_struct_schema_to_json_schema,
    format_examples,
    get_react_tools_prompt_format,
)
from msgflux.utils.inspect import get_filename, get_mime_type
from msgflux.utils.msgspec import StructFactory
from msgflux.utils.tool import ToolFlowControl
from msgflux.utils.validation import is_subclass_of
from msgflux.utils.xml import apply_xml_tags

# it is possible to continue generating a model. Just resend to it what it
# wrote and then it will continue from there
# the system can change the response to the stream if x condition is met. nein


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
    ]

    def __init__(
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelGateway],
        *,
        system_message: Optional[str] = None,
        instructions: Optional[str] = None,
        expected_output: Optional[str] = None,
        examples: Optional[str] = None,
        system_extra_message: Optional[str] = None,
        include_date: Optional[bool] = False,
        stream: Optional[bool] = False,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
        task_inputs: Optional[Union[str, Dict[str, str]]] = None,
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None,
        task_messages: Optional[str] = None,
        task_template: Optional[str] = None,
        context_inputs: Optional[Union[str, List[str]]] = None,
        context_cache: Optional[str] = None,
        context_inputs_template: Optional[str] = None,
        model_preference: Optional[str] = None,
        prefilling: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        typed_xml: Optional[bool] = False,
        response_mode: Optional[str] = "plain_response",
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        injected_kwargs: Optional[str] = None,
        response_template: Optional[str] = None,
        fixed_messages: Optional[List[Dict[str, Any]]] = None,
        signature: Optional[Union[str, Signature]] = None,
        return_model_state: Optional[bool] = False,
        # verbose: Optional[bool] = False,
        description: Optional[str] = None,
        system_prompt_template: Optional[str] = SYSTEM_PROMPT_TEMPLATE,
        typed_xml_template: Optional[str] = TYPED_XML_TEMPLATE,
        _annotations: Optional[Dict[str, type]] = None,
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
            Examples of inputs, plans and outputs.
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
            Field of the Message object that will be the multimodal input
            to the task.
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
        typed_xml:
            Converts the model output, which should be typed-XML, into a typed-dict.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        injected_kwargs:
            Field of the Message object that will be the inputs to templates and tools.
        tools:
            A list of callable objects.
        tool_choice:
            By default the model will determine when and how many tools to use.
            You can force specific behavior with the tool_choice parameter.
                1. auto:
                    (Default) Call zero, one, or multiple functions. tool_choice: "auto"
                2. required:
                    Call one or more functions. tool_choice: "required"
                3. Forced Function:
                    Call exactly one specific function. E.g. "add".
        response_template:
            A Jinja template to format response.
        fixed_messages:
            A fixed list of chats in ChatML format.
        signature:
            A DSPy-based signature. A signature creates a task_template,
            a generation_schema, instructions and examples (both if passed).
            Can be combined with standard generation_schemas like `ReAct` and
            `ChainOfThought`. Can also be combined with `typed_xml`.
        return_model_state:
            If True, returns a dictionary containing model_state and response.
        description:
            The Agent description. It's useful when using an agent-as-a-tool.
        system_prompt_template:
            A Jinja template to format system prompt.
        typed_xml_template:
            A Jinja template to inject instructions to use xml to dict.
        _annotations
            Define the input and output annotations to use the agent-as-a-function.
        """
        if _annotations is None:
            _annotations = {"message": Union[str, Dict[str, str]], "return": str}
        super().__init__()

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if output_guardrail is not None:
                raise ValueError("`output_guardrail` is not `stream=True` compatible")

            if response_template is not None:
                raise ValueError("`response_template` is not `stream=True` compatible")

            if typed_xml is True:
                raise ValueError("`typed_xml=True` is not `stream=True` compatible")

        self._set_typed_xml_template(typed_xml_template)

        if signature is not None:
            signature_params = dotdict(
                {
                    "signature": signature,
                    "instructions": instructions,
                    "system_message": system_message,
                    "typed_xml": typed_xml,
                }
            )
            if generation_schema is not None:
                signature_params.generation_schema = generation_schema
            self._set_signature(**signature_params)
        else:
            self._set_examples(examples)
            self._set_expected_output(expected_output)
            self._set_generation_schema(generation_schema)
            self._set_instructions(instructions)
            self._set_system_message(system_message)
            self._set_task_template(task_template)
            self._set_typed_xml(typed_xml)

        self.set_name(name)
        self.set_description(description)
        self._set_annotations(_annotations)
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
        self._set_system_prompt_template(system_prompt_template)
        self._set_response_mode(response_mode)
        self._set_return_model_state(return_model_state)
        self._set_stream(stream)
        self._set_response_template(response_template)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_task_inputs(task_inputs)
        self._set_injected_kwargs(injected_kwargs)
        self._set_tool_choice(tool_choice)
        self._set_tools(tools)

    def forward(
        self, message: Union[str, Dict[str, Any], Message], **kwargs
    ) -> Union[str, Dict[str, None], ModelStreamResponse, Message]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(prefilling=self.prefilling, **inputs)
        response = self._process_model_response(message, model_response, **inputs)
        return response

    def _execute_model(
        self,
        model_state: List[Dict[str, Any]],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        if injected_kwargs is None:
            injected_kwargs = {}
        model_execution_params = self._prepare_model_execution(
            model_state, prefilling, model_preference, injected_kwargs
        )
        if self.input_guardrail:
            self._execute_input_guardrail(model_execution_params)
        model_response = self.model(**model_execution_params)
        return model_response

    @instrument_agent_prepare_model_execution
    def _prepare_model_execution(
        self,
        model_state: List[Dict[str, Any]],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if injected_kwargs is None:
            injected_kwargs = {}
        agent_state = []

        if self.fixed_messages:
            agent_state.extend(self.fixed_messages)

        agent_state.extend(model_state)

        system_prompt = self._get_system_prompt(injected_kwargs)

        tool_schemas = self.tool_library.get_tool_json_schemas()
        if not tool_schemas:
            tool_schemas = None

        if is_subclass_of(self.generation_schema, ReAct) and tool_schemas:
            react_tools = get_react_tools_prompt_format(tool_schemas)
            if system_prompt:  # TODO: template to react tools
                system_prompt += "\n\n" + react_tools
            else:
                system_prompt = react_tools
            tool_schemas = None  # Disable tool_schemas to react controlflow preference

        model_execution_params = dotdict(
            {
                "messages": agent_state,
                "system_prompt": system_prompt or None,
                "prefilling": prefilling,
                "stream": self.stream,
                "tool_schemas": tool_schemas,
                "tool_choice": self.tool_choice,
                "generation_schema": self.generation_schema,
                "typed_xml": self.typed_xml,
            }
        )

        if model_preference:
            model_execution_params.model_preference = model_preference

        return model_execution_params

    def _prepare_input_guardrail_execution(
        self, model_execution_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        model_state = model_execution_params.get("model_state")
        last_message = model_state[-1]
        if isinstance(last_message.get("content"), list):
            if last_message.get("content")[0]["type"] == "image_url":
                data = [last_message]
            else:  # audio, file
                data = last_message.get("content")[-1]  # text input
        else:
            data = last_message.get("content")
        guardrail_params = {"data": data}
        return guardrail_params

    def _process_model_response(
        self,
        message: Union[str, Dict[str, str], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: List[Dict[str, Any]],
        model_preference: Optional[str] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        if injected_kwargs is None:
            injected_kwargs = {}
        if "tool_call" in model_response.response_type:
            model_response, model_state = self._process_tool_call_response(
                model_response, model_state, model_preference, injected_kwargs
            )
        elif is_subclass_of(self.generation_schema, ToolFlowControl):
            model_response, model_state = self._process_tool_flow_control_response(
                model_response, model_state, model_preference, injected_kwargs
            )

        raw_response = self._extract_raw_response(model_response)

        response_type = model_response.response_type

        if response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, response_type, model_state, message
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    def _process_tool_flow_control_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Dict[str, Any],
        model_preference: Optional[str] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[str, Dict[str, Any], ModelStreamResponse], Dict[str, Any]]:
        """Handle the fields returned by `ReAct`. If the fields are different,
        you must rewrite this function.
        """
        if injected_kwargs is None:
            injected_kwargs = {}
        while True:
            raw_response = self._extract_raw_response(model_response)

            if raw_response.current_step:
                actions = raw_response.current_step.actions
                tool_callings = [(act.id, act.name, act.arguments) for act in actions]
                tool_execution_result = self._process_tool_call(
                    tool_callings, model_state, injected_kwargs
                )

                if tool_execution_result.return_directly:
                    return tool_execution_result.responses, model_state

                for act in actions:
                    act.result = tool_execution_result[[act.id]]

                if model_state[-1]["role"] == "assistant":
                    last_react_msg = model_state[-1]["content"]
                    react_state = msgspec.json.decode(last_react_msg)
                    react_state.append(raw_response)
                    react_state_encoded = msgspec.json.encode(react_state)
                    model_state[-1] = react_state_encoded
                else:
                    react_state = []
                    react_state.append(raw_response)
                    react_state_encoded = msgspec.json.encode(react_state)
                    model_state.append(
                        [{"role": "assistant", "content": react_state_encoded}]
                    )

            elif raw_response.final_answer:
                return model_response, model_state

            model_response = self._execute_model(
                model_state=model_state,
                model_preference=model_preference,
            )

    def _process_tool_call_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Optional[Dict[str, Any]],
        model_preference: Optional[str] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[str, Dict[str, Any], ModelStreamResponse], Dict[str, Any]]:
        """ToolCall example: [{'role': 'assistant', 'tool_calls': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        if injected_kwargs is None:
            injected_kwargs = {}
        while True:
            if model_response.response_type == "tool_call":
                raw_response = self._extract_raw_response(model_response)
                tool_callings = raw_response.get_calls()
                tool_execution_result = self._process_tool_call(
                    tool_callings, model_state, injected_kwargs
                )
                if tool_execution_result.return_directly:
                    return tool_execution_result.responses, model_state

                raw_response.insert_results(tool_execution_result.responses)
                tool_responses_message = raw_response.get_messages()
                model_state.extend(tool_responses_message)
            else:
                return model_response, model_state

            model_response = self._execute_model(
                model_state=model_state,
                model_preference=model_preference,
            )

    def _process_tool_call(
        self,
        tool_callings: Dict[str, Any],
        model_state: List[Dict[str, Any]],
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if injected_kwargs is None:
            injected_kwargs = {}
        tool_execution_result = self.tool_library(
            tool_callings=tool_callings,
            model_state=model_state,
            injected_kwargs=injected_kwargs,
        )
        return tool_execution_result

    def _prepare_response(
        self,
        raw_response: Union[str, Dict[str, Any], ModelStreamResponse],
        response_type: str,
        model_state: List[Dict[str, Any]],
        message: Union[str, Dict[str, Any], Message],
    ) -> Union[str, Dict[str, Any], ModelStreamResponse]:
        formated_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            if "text_generation" in response_type or "structured" in response_type:
                if self.output_guardrail:
                    self._execute_output_guardrail(raw_response)
                if self.response_template:
                    formated_response = self._format_response_template(raw_response)
        response = agent_response = formated_response or raw_response
        if self.return_model_state:
            response = dotdict({"response": agent_response, "model_state": model_state})
        return self._define_response_mode(response, message)

    def _prepare_output_guardrail_execution(
        self, model_response: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(model_response, str):
            data = model_response
        else:
            data = str(model_response)
        guardrail_params = {"data": data}
        return guardrail_params

    def _prepare_task(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Prepare model input in ChatML format and execution params."""
        injected_kwargs = kwargs.pop(
            "injected_kwargs", {}
        )  # Runtime params to templates
        if (
            not injected_kwargs
            and isinstance(message, Message)
            and self.injected_kwargs is not None
        ):
            injected_kwargs = message.get(self.injected_kwargs, {})
        if injected_kwargs:
            injected_kwargs = dotdict(injected_kwargs)

        task_messages = kwargs.pop("task_messages", None)
        if (
            task_messages is None
            and isinstance(message, Message)
            and self.injected_kwargs is not None
        ):
            task_messages = self._get_content_from_message(self.task_messages, message)

        content = self._process_task_inputs(
            message, injected_kwargs=injected_kwargs, **kwargs
        )

        if content is None and task_messages is None:
            raise ValueError("No data was detected to make the model input")

        if content is not None:
            chat_content = [{"role": "user", "content": content}]
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
            "injected_kwargs": injected_kwargs,
        }

    def _process_task_inputs(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Union[str, Dict[str, Any]]:
        content = ""

        context_content = self._context_manager(message, **kwargs)
        if context_content:
            content += context_content

        if isinstance(message, Message):
            task_inputs = self._extract_message_values(self.task_inputs, message)
        else:
            task_inputs = message

        if task_inputs is None and self.task_template is None:
            raise AttributeError(
                "When using a `Message` in `nn.Agent` it is necessary to "
                "have configured `task_inputs` or `task_template`"
            )

        if self.task_template:
            if task_inputs:
                task_content = self._format_task_template(task_inputs)
            # It's possible to use `task_template` as the default task message
            # if no `task_inputs` is selected. This can be useful for multimodal
            # models that require a text message to be sent along with the data
            else:
                task_content = self.task_template
        else:
            task_content = task_inputs

        if kwargs.get("injected_kwargs", None):
            task_content = self._format_template(
                kwargs["injected_kwargs"], task_content
            )

        task_content = apply_xml_tags("task", task_content)
        content += task_content
        content = content.strip()  # Remove whitespace

        multimodal_content = self._process_task_multimodal_inputs(message, **kwargs)
        if multimodal_content:
            multimodal_content.append({"type": "text", "text": content})
            return multimodal_content
        return content

    def _context_manager( # noqa: C901
        self, message: Union[str, Message, Dict[str, str]], **kwargs
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
                msg_context = self._format_template(
                    context_inputs, self.context_inputs_template
                )
            elif isinstance(context_inputs, str):
                msg_context = context_inputs
            elif isinstance(context_inputs, list):
                msg_context = " ".join(
                    str(v) for v in context_inputs if v is not None
                )
            elif isinstance(context_inputs, dict):
                msg_context = "\n\n".join(str(v) for v in context_inputs.values())
            context_content += "\n\n" + msg_context

        if context_content:
            if kwargs.get("injected_kwargs", None):
                context_content = self._format_template(
                    kwargs["injected_kwargs"], context_content
                )
            return apply_xml_tags("context", context_content)
        return None

    def _process_task_multimodal_inputs(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Processes multimodal inputs (image, audio, file) via kwargs or message.
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
            "file": self._format_file_input,
        }

        for media_type, formatter in formatters.items():
            media_sources = multimodal_paths.get(media_type, [])
            if not isinstance(media_sources, list):
                logger.warning(
                    f"Expected list for multimodal config key `{media_type}`, "
                    f"got `{type(media_sources)}`"
                )
                continue

            for media_source in media_sources:
                if media_source:
                    formatted_input = formatter(media_source)
                    if formatted_input:
                        content.append(formatted_input)

        return content

    def _format_image_input(self, image_source: str) -> Optional[Dict[str, Any]]:
        """Formats the image input for the model."""
        encoded_image = self._prepare_data_uri(image_source, force_encode=False)

        if not encoded_image:
            return None

        if not encoded_image.startswith("http"):
            mime_type = get_mime_type(
                image_source
            )  # Try to guess from the original source
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"  # Fallback
            encoded_image = f"data:{mime_type};base64,{encoded_image}"
        return {"type": "image_url", "image_url": {"url": encoded_image}}

    def _format_audio_input(self, audio_source: str) -> Optional[Dict[str, Any]]:
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
        format_key = (
            audio_format_suffix if audio_format_suffix else mime_type.split("/")[-1]
        )
        return {
            "type": "input_audio",
            "input_audio": {"data": base64_audio, "format": format_key},
        }

    def _format_file_input(self, file_source: str) -> Optional[Dict[str, Any]]:
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

        return {
            "type": "file",
            "file": {"filename": filename, "file_data": file_data_uri},
        }

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

    def _set_tools(self, tools: Optional[List[Callable]] = None):
        self.tool_library = ToolLibrary(self.get_module_name(), tools or [])

    def _set_fixed_messages(
        self, fixed_messages: Optional[List[Dict[str, Any]]] = None
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
            is_subclass_of(generation_schema, msgspec.Struct)
            or generation_schema is None
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
            if isinstance(tool_choice, str):
                if tool_choice not in ["auto", "required"]:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            self.register_buffer("tool_choice", tool_choice)
        else:
            raise TypeError(
                f"`tool_choice` need be a str or None given `{type(tool_choice)}`"
            )

    def _set_system_message(self, system_message: Optional[str] = None):
        if isinstance(system_message, str) or system_message is None:
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
            self.instructions = Parameter(instructions, PromptSpec.INSTRUCTIONS)
        else:
            raise TypeError(
                f"`instructions` requires a string or None given `{type(instructions)}`"
            )

    def _set_expected_output(self, expected_output: Optional[str] = None):
        if isinstance(expected_output, str) or expected_output is None:
            self.expected_output = Parameter(
                expected_output, PromptSpec.EXPECTED_OUTPUT
            )
        else:
            raise TypeError(
                "`expected_output` requires a string or None "
                f"given `{type(expected_output)}`"
            )

    def _set_examples(self, examples: Optional[str] = None):
        if isinstance(examples, str) or examples is None:
            self.examples = Parameter(examples, PromptSpec.EXAMPLES)
        else:
            raise TypeError(
                f"`examples` requires a string or None given `{type(examples)}`"
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

    def _set_system_prompt_template(self, system_prompt_template: Optional[str] = None):
        if isinstance(system_prompt_template, str) or system_prompt_template is None:
            self.register_buffer("system_prompt_template", system_prompt_template)
        else:
            raise TypeError(
                "`system_prompt_template` requires a string given "
                f"`{type(system_prompt_template)}`"
            )

    def _set_system_extra_message(self, system_extra_message: Optional[str] = None):
        if isinstance(system_extra_message, str) or system_extra_message is None:
            self.register_buffer("system_extra_message", system_extra_message)
        else:
            raise TypeError(
                "`system_extra_message` requires a string or None "
                f"given `{type(system_extra_message)}`"
            )

    def _set_injected_kwargs(self, injected_kwargs: Optional[str] = None):
        if isinstance(injected_kwargs, str) or injected_kwargs is None:
            self.register_buffer("injected_kwargs", injected_kwargs)
        else:
            raise TypeError(
                "`injected_kwargs` requires a string or None "
                f"given `{type(injected_kwargs)}`"
            )

    def _set_typed_xml_template(self, typed_xml_template: str):
        if isinstance(typed_xml_template, str):
            self.register_buffer("typed_xml_template", typed_xml_template)
        else:
            raise TypeError(
                "`typed_xml_template` requires a string "
                f"given `{type(typed_xml_template)}`"
            )

    def _set_typed_xml(self, typed_xml: Optional[bool] = False): # noqa: FBT001, FBT002
        if isinstance(typed_xml, bool):
            if typed_xml:
                json_schema = None
                if self.generation_schema:
                    schema = msgspec.json.schema(self.generation_schema)
                    json_schema = adapt_struct_schema_to_json_schema(schema)
                template_inputs = {
                    "instructions": self.instructions.data,
                    "json_schema": json_schema,
                }
                xml_instructions = self._format_template(
                    template_inputs, self.typed_xml_template
                )
                self._set_instructions(xml_instructions)
            self.register_buffer("typed_xml", typed_xml)
        else:
            raise TypeError(f"`typed_xml` requires a bool given `{type(typed_xml)}`")

    def _set_signature(
        self,
        *,
        signature: Optional[Union[str, Signature]] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        instructions: Optional[str] = None,
        system_message: Optional[str] = None,
        typed_xml: Optional[bool] = False,
    ):
        if signature is not None:
            examples = None

            # Get system message
            schema_system_message = SIGNATURE_SYSTEM_MESSAGES.get(
                generation_schema, SIGNATURE_DEFAULT_SYSTEM_MESSAGE
            )
            self._set_system_message(system_message or schema_system_message)

            if isinstance(signature, str):
                input_str_signature, output_str_signature = signature.split("->")
                inputs_desc = StructFactory._parse_annotations(input_str_signature)
                outputs_desc = StructFactory._parse_annotations(output_str_signature)

            elif issubclass(signature, Signature):
                # Get instructions
                instructions = signature.get_instructions()

                # Get examples from signature
                examples = get_examples_from_signature(signature)

                # Descriptions
                inputs_desc = signature.get_input_descriptions()
                outputs_desc = signature.get_output_descriptions()
                output_str_signature = signature.get_str_signature().split("->")[-1]

            else:
                raise TypeError(
                    "`signature` requires a string, `Signature` or None "
                    f"given `{type(signature)}`"
                )

            # Create task template
            task_template = get_task_template_from_signature(inputs_desc)
            self._set_task_template(task_template)

            # Set instructions
            self._set_instructions(instructions)

            # Create generation schema
            output_struct = StructFactory.from_signature(
                output_str_signature, "Outputs"
            )
            if generation_schema is not None:
                output_struct = generation_schema[output_struct]  # Insert as an TypeVar

                class Output(
                    output_struct, msgspec.Struct
                ):  # Convert typing._GenericAlias to Struct
                    pass

                output_struct = Output
            self._set_generation_schema(output_struct)

            # Create expected outputs
            expected_output = get_expected_output_from_signature(
                inputs_desc, outputs_desc
            )
            if typed_xml is False:
                expected_output += "\nWrite an encoded JSON."
            self._set_expected_output(expected_output)

            # Create examples
            if examples is not None:
                input_examples_dict, output_json_string = examples
                input_examples_string = self._format_task_template(
                    input_examples_dict, typed_xml
                )
                examples = format_examples(
                    [(input_examples_string, output_json_string)]
                )
            self._set_examples(examples)

            # Set xml output
            self._set_typed_xml(typed_xml)

    def _get_system_prompt(
        self, injected_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render the system prompt using the Jinja template.
        Returns an empty string if no segments are provided.
        """
        template_inputs = {
            "system_message": self.system_message.data,
            "instructions": self.instructions.data,
            "expected_output": self.expected_output.data,
            "examples": self.examples.data,
            "system_extra_message": self.system_extra_message,
        }

        if self.include_date:
            now = datetime.now(tz=timezone.utc)
            template_inputs["current_date"] = now.strftime("%m/%d/%Y")

        system_prompt = self._format_template(
            template_inputs, self.system_prompt_template
        )

        if injected_kwargs:  # Runtime inputs to system template
            system_prompt = self._format_template(injected_kwargs, system_prompt)
        return system_prompt
