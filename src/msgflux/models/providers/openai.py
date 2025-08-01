import base64
import tempfile
from contextlib import contextmanager
from os import getenv
from typing import Any, Dict, List, Literal, Optional, Union

import msgspec

try:
    import httpx
    import openai
    from openai import OpenAI
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
except Exception as e:
    raise ImportError(
        "`openai` client is not detected, please install"
        "using `pip install msgflux[openai]`"
    ) from e

from msgflux.dotdict import dotdict
from msgflux.exceptions import KeyExhaustedError
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.types import (
    ChatCompletionModel,
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextToImageModel,
    TextToSpeechModel,
)
from msgflux.nn import functional as F
from msgflux.utils.chat import adapt_struct_schema_to_json_schema
from msgflux.utils.encode import encode_data_to_bytes
from msgflux.utils.msgspec import struct_to_dict
from msgflux.utils.tenacity import model_retry
from msgflux.utils.xml import xml_to_typed_dict

OpenAIInstrumentor().instrument()

# support continuing generation by validating the reason


class _BaseOpenAI(BaseModel):
    provider: str = "openai"

    def _initialize(self):
        """Initialize the OpenAI client with empty API key."""
        self.current_key_index = 0
        max_retries = getenv("OPENAI_MAX_RETRIES", openai.DEFAULT_MAX_RETRIES)
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = OpenAI(
            **self.sampling_params,
            api_key="",
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )

    def _get_base_url(self):
        return None

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("OPENAI_API_KEY")
        if not keys:
            raise ValueError(
                "The OpenAI key is not available. Please set `OPENAI_API_KEY`"
            )
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

    def _set_next_api_key(self):
        """Set the next API key in the rotation."""
        if self.current_key_index >= len(self._api_key) - 1:
            raise KeyExhaustedError("All API keys have been exhausted")
        self.current_key_index += 1
        self.client.api_key = self._api_key[self.current_key_index]

    def _execute_with_retry(self, **kwargs):
        """Execute the model with the current API key and handle retries."""
        try:
            return self._execute(**kwargs)
        except (openai.RateLimitError, openai.APIError):
            # Try the next API key
            self._set_next_api_key()
            # Recursively try again with the new key
            return self._execute_with_retry(**kwargs)
        except Exception as e:
            # For other exceptions, we might want to retry with the same key
            raise e

    def _execute_model(self, **kwargs):
        """Main method to execute the model with automatic key rotation and retries."""
        # Set the initial API key
        self.client.api_key = self._api_key[self.current_key_index]

        try:
            return self._execute_with_retry(**kwargs)
        except KeyExhaustedError as e:
            # Reset the key index for future calls
            self.current_key_index = 0
            raise e


class OpenAIChatCompletion(_BaseOpenAI, ChatCompletionModel):
    """OpenAI Chat Completion."""

    def __init__(
        self,
        model_id: str,
        *,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = 8192,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        return_reasoning: Optional[bool] = False,
    ):
        """Args:
        model_id:
            Model ID in provider.
        modalities:
            Types of output you would like the model to generate.
            Can be: ["text"], ["audio"] or ["text", "audio"].
        audio:
            Audio configurations. Define voice and output format.
        max_tokens:
            An upper bound for the number of tokens that can be
            generated for a completion, including visible output
            tokens and reasoning tokens.
        reasoning_effort:
            Constrains effort on reasoning for reasoning models.
            Currently supported values are low, medium, and high.
            Reducing reasoning effort can result in faster responses
            and fewer tokens used on reasoning in a response.
            Can be: "low", "medium" or "high".
        temperature:
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and
            deterministic.
        top_p:
            An alternative to sampling with temperature, called nucleus
            sampling, where the model considers the results of the tokens
            with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        web_search_options:
            This tool searches the web for relevant results to use in a response.
            OpenAI-only.
        base_url:
            URL to model provider.
        return_reasoning:
            If the model returns the `reasoning` field it will be added
            along with the response.
        """
        if modalities is None:
            modalities = ["text"]
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "modalities": modalities,
            "web_search_options": web_search_options,
        }
        if audio:
            self.sampling_run_params["audio"] = audio
        if reasoning_effort is not None:
            self.sampling_run_params["reasoning_effort"] = reasoning_effort
        self.return_reasoning = return_reasoning
        self._initialize()
        self._get_api_key()

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider in "openai":
            params["max_completion_tokens"] = params.pop("max_tokens")
            tools = params.get("tools", None)
            if tools:  # OpenAI supports 'strict' mode to tools
                for tool in tools:
                    tool["function"]["strict"] = True
        return params

    def _execute(self, **kwargs):
        if kwargs.get("tool_schemas"):
            kwargs["parallel_tool_calls"] = True
        prefilling = kwargs.pop("prefilling")
        if prefilling:
            kwargs.get("messages").append({"role": "assistant", "content": prefilling})
        params = {**kwargs, **self.sampling_run_params}
        adapted_params = self._adapt_params(params)
        model_output = self.client.chat.completions.create(
            model=self.model_id,
            **adapted_params,
        )
        return model_output

    def _generate(self, **kwargs): # noqa: C901
        response = ModelResponse()
        metadata = dotdict()

        typed_xml = kwargs.pop("typed_xml")
        generation_schema = kwargs.pop("generation_schema")

        if generation_schema is not None and typed_xml is False:
            schema = msgspec.json.schema(generation_schema)
            json_schema = adapt_struct_schema_to_json_schema(schema)
            kwargs["response_format"] = json_schema

        model_output = self._execute_model(**kwargs)

        metadata.update({"usage": model_output.usage.to_dict()})

        choice = model_output.choices[0]

        prefix_response_type = ""
        reasoning_content = None
        if (
            self.return_reasoning is True
            and hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content is not None
        ):
            reasoning_content = choice.message.reasoning_content
            prefix_response_type = "reasoning_"

        if choice.message.annotations:  # Extra responses (e.g web search references)
            annotations_content = [
                item.model_dump() for item in choice.message.annotations
            ]
            metadata.annotations = annotations_content

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator(reasoning_content)
            response.set_response_type(f"{prefix_response_type}tool_call")
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                tool_id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif choice.message.content:
            if typed_xml is True:
                response.set_response_type(f"{prefix_response_type}structured")
                response_content = xml_to_typed_dict(choice.message.content)
                if generation_schema:  # Type validation
                    encoded_response_content = msgspec.json.encode(response_content)
                    msgspec.json.decode(
                        encoded_response_content, type=generation_schema
                    )
            elif generation_schema is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                struct = msgspec.json.decode(
                    choice.message.content, type=generation_schema
                )
                response_content = struct_to_dict(struct)
            else:
                response.set_response_type(
                    f"{prefix_response_type}text_generation"
                )
                if reasoning_content is not None:
                    response_content = dotdict({"answer": choice.message.content})
                else:
                    response_content = choice.message.content
        elif choice.message.audio:
            response_content = dotdict(
                {
                    "id": choice.message.audio.id,
                    "audio": base64.b64decode(choice.message.audio.data),
                }
            )
            if choice.message.audio.transcript:
                response.set_response_type("audio_text_generation")
                response_content.text = choice.message.audio.transcript
            else:
                response.set_response_type("audio_generation")

        if reasoning_content is not None:
            response_content.think = reasoning_content

        response.add(response_content)
        response.set_metadata(metadata)
        return response

    async def _stream_generate(self, **kwargs): # noqa: C901
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")

        model_output = self._execute_model(**kwargs)

        for chunk in model_output:
            if chunk.choices:
                if (
                    self.return_reasoning is True
                    and hasattr(chunk.choices[0].delta, "reasoning_content")
                    and chunk.choices[0].delta.reasoning_content is not None
                ):
                    if stream_response.response_type is None:
                        stream_response.set_response_type("reasoning_text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(chunk.choices[0].delta.reasoning_content)
                elif chunk.choices[0].delta.content:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(chunk.choices[0].delta.content)
                elif chunk.choices[0].delta.tool_calls:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("tool_call")
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    call_index = tool_call.index
                    tool_id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    aggregator.process(call_index, tool_id, name, arguments)
                elif (
                    hasattr(chunk.choices[0].delta, "annotations")
                    and chunk.choices[0].delta.annotations is not None
                ):
                    annotations_content = [
                        item.model_dump() for item in chunk.choices[0].delta.annotations
                    ]
                    metadata.annotations = annotations_content
            elif chunk.usage:
                metadata.update(chunk.usage.to_dict())

        if aggregator.tool_calls:
            stream_response.add(aggregator)
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    @model_retry
    def __call__(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_xml: Optional[bool] = False,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message
                it will continue its response from there.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use.
                You can force specific behavior with the tool_choice parameter.
                    1. auto:
                        (Default) Call zero, one, or multiple functions.
                    2. required:
                        Call one or more functions.
                    3. Forced Tool:
                        Call exactly one specific tool e.g:
                        {"type": "function", "function": {"name": "get_weather"}}
            typed_xml:
                Converts the model output, which should be typed-XML, into a typed-dict.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_xml=True` and `stream=True`.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if isinstance(system_prompt, str):
            messages.insert(0, {"role": "system", "content": system_prompt})

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "tool_choice": tool_choice,
            "tools": tool_schemas,
        }

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if typed_xml is True:
                raise ValueError("`typed_xml=True` is not `stream=True` compatible")

            stream_response = ModelStreamResponse()
            F.background_task(
                self._stream_generate,
                **generation_params,
                stream=stream,
                stream_response=stream_response,
                stream_options={"include_usage": True},
            )
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(
                **generation_params,
                typed_xml=typed_xml,
                generation_schema=generation_schema,
            )
            return response


class OpenAITextToSpeech(_BaseOpenAI, TextToSpeechModel):
    """OpenAI Text to Speech."""

    def __init__(
        self,
        model_id: str,
        voice: Optional[str] = "alloy",
        speed: Optional[float] = 1.0,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        voice:
            The voice to use when generating the audio.
        speed:
            the speed of the generated audio. Select a value
            from 0.25 to 4.0. 1.0 is the default.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "voice": voice,
            "speed": speed,
        }
        self._initialize()
        self._get_api_key()

    @contextmanager
    def _execute_with_retry(self, **kwargs):
        while True:
            try:
                with self._execute(**kwargs) as result:
                    yield result
                break
            except (openai.RateLimitError, openai.APIError):
                self._set_next_api_key()
            except Exception as e:
                raise e

    @contextmanager
    def _execute_model(self, **kwargs):
        self.client.api_key = self._api_key[self.current_key_index]
        try:
            with self._execute_with_retry(**kwargs) as result:
                yield result
        except KeyExhaustedError as e:
            self.current_key_index = 0
            raise e

    @contextmanager
    def _execute(self, **kwargs):
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        with self._execute_model(**kwargs) as model_output:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                model_output.stream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        with self._execute_model(**kwargs) as model_output:
            for chunk in model_output.iter_bytes(chunk_size=1024):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    @model_retry
    def __call__(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            The text to generate audio for.
        stream:
            Whether generation should be in streaming mode.
        prompt:
            Control the voice of your generated audio with additional instructions.
        response_format:
            The format to audio in.
        """
        params = dotdict({"input": data, "response_format": response_format})
        if prompt:
            params.instructions = prompt
        if stream:
            stream_response = ModelStreamResponse()
            params.stream_response = stream_response
            F.background_task(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response


class OpenAITextToImage(_BaseOpenAI, TextToImageModel):
    """OpenAI Image Generation."""

    def __init__(
        self,
        *,
        model_id: str,
        size: Optional[str] = "auto",
        quality: Optional[str] = "auto",
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
        moderation: Optional[Literal["auto", "low"]] = None,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        moderation:
            Control the content-moderation level for images generated.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "size": size,
            "quality": quality,
            "background": background,
        }
        if moderation:
            self.sampling_run_params["moderation"] = moderation
        self._initialize()
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.images.generate(
            model=self.model_id, **kwargs, **self.sampling_run_params
        )
        return model_output

    def _get_metadata(self, model_output):
        metadata = dotdict(
            {
                "usage": model_output.usage.to_dict(),
                "details": {
                    "size": model_output.size,
                    "quality": model_output.quality,
                    "output_format": model_output.output_format,
                    "background": model_output.background,
                },
            }
        )
        return metadata

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_generation")

        model_output = self._execute_model(**kwargs)

        metadata = self._get_metadata(model_output)

        images = []
        for item in model_output.data:
            if item.url:
                images.append(item.url)
            if item.b64_json:
                images.append(item.b64_json)

        if len(images) == 1:
            images = images[0]

        response.add(images)
        response.set_metadata(metadata)

        return response

    @model_retry
    def __call__(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(n=n, prompt=prompt)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = self._generate(**generation_params)
        return response


class OpenAIImageTextToImage(OpenAITextToImage, ImageTextToImageModel):
    """OpenAI Image Edit."""

    def _execute(self, **kwargs):
        model_output = self.client.images.edit(
            model=self.model_id, **kwargs, **self.sampling_run_params
        )
        return model_output

    def _prepare_inputs(self, image, mask):
        inputs = {}
        if isinstance(image, str):
            image = [image]
        inputs["image"] = [encode_data_to_bytes(item) for item in image]
        if mask:
            inputs["mask"] = encode_data_to_bytes(mask)
        return inputs

    @model_retry
    def __call__(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(prompt=prompt, n=n)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = self._generate(**generation_params, **inputs)
        return response


class OpenAISpeechToText(_BaseOpenAI, SpeechToTextModel):
    """OpenAI Speech to Text."""

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Optional[float] = 0.0,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        temperature:
            The sampling temperature, between 0 and 1.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"temperature": temperature}
        self._initialize()
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.audio.transcriptions.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("transcript")

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            if model_output.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in model_output.words
                ]
                transcript["words"] = words
            if model_output.segment:
                segments = [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in model_output.segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = self._execute_model(**kwargs)

        for event in model_output:
            chunk = event.transcript.text.delta
            if chunk:
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()
            elif event.transcript.text.done:
                stream_response.add(None)

        return stream_response

    @model_retry
    def __call__(
        self,
        data: Union[str, bytes],
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            Url, path, base64 to audio.
        stream:
            Whether generation should be in streaming mode.
        response_format:
            The format of the output, in one of these options:
            json, text, srt, verbose_json, or vtt.
        timestamp_granularities:
            The timestamp granularities to populate for this
            transcription. `response_format` must be set `verbose_json`
            to use timestamp granularities. Either or both of these
            options are supported: word, or segment. Note: There is no
            additional latency for segment timestamps, but generating
            word timestamps incurs additional latency.
        prompt:
            An optional text to guide the model's style or continue a
            previous audio segment. The prompt should match the audio language.
        language:
            The language of the input audio. Supplying the input language in
            ISO-639-1 (e.g. en) format will improve accuracy and latency.
        """
        if isinstance(data, str):
            data = encode_data_to_bytes(data)
        params = {
            "file": data,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
        }
        if stream:
            stream_response = ModelStreamResponse()
            params["stream_response"] = stream_response
            params["stream"] = stream
            F.background_task(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response


class OpenAITextEmbedder(_BaseOpenAI, TextEmbedderModel):
    """OpenAI Text Embedder."""

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self._initialize()
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.embeddings.create(
            model=self.model_id,
            **kwargs,
            **self.sampling_run_params,
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embedding = model_output.data[0].embedding
        metadata = dotdict({"usage": model_output.usage.to_dict()})
        response.add(embedding)
        response.set_metadata(metadata)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[str]],
    ):
        """Args:
        data:
            Input text to embed.
        """
        response = self._generate(input=data)
        return response


class OpenAIModeration(_BaseOpenAI, ModerationModel):
    """OpenAI Moderation."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self._initialize()
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.moderations.create(
            model=self.model_id,
            **kwargs,
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = self._execute_model(**kwargs)
        moderation = dotdict({"results": model_output.results[0].model_dump()})
        moderation.safe = not moderation.results.flagged
        response.add(moderation)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = self._generate(input=data)
        return response
