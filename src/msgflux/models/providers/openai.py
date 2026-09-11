import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import msgspec

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.models.cache import generate_cache_key
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.chat_context import OpenAIResponsesContextAdapter
from msgflux.models.chat_extensions import (
    ChatRequestContext,
    ChatSpeedExtension,
)
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.model_credentials import ModelCredentialResolver
from msgflux.models.multipart import (
    aprepare_multipart_file,
    prepare_multipart_data,
    prepare_multipart_file,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleHTTPModel,
    OpenAIResponsesAPI,
)
from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion as _OpenAICompatibleChatCompletion,
)
from msgflux.models.reasoning import (
    OpenAIReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.sse import aiter_sse_json, iter_sse_json
from msgflux.models.tool_adapters.openai_patch import OpenAIApplyPatchAdapter
from msgflux.models.tool_adapters.openai_shell import OpenAIShellAdapter
from msgflux.models.types import (
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextToImageModel,
    TextToSpeechModel,
)
from msgflux.models.usage import default_usage_codec


class OpenAIServiceTierExtension(ChatSpeedExtension):
    """Map canonical speed preferences to OpenAI service tiers."""

    def accepts(self, owner: Any, value: Any) -> bool:
        if value == "fast":
            return True
        if value != "ultrafast":
            return False
        model_id = owner.model_id.rsplit("/", maxsplit=1)[-1]
        return model_id in {"gpt-5.6", "gpt-5.6-sol"} or model_id.startswith(
            "gpt-5.6-sol-20"
        )

    def prepare_request(self, owner, request, context: ChatRequestContext):
        speed = owner.chat_settings.get("speed")
        if speed is None or context.operation == "token_count":
            return request
        params = dict(request.params)
        extra_body = dict(params.get("extra_body") or {})
        extra_body.pop("service_tier", None)
        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)
        params["service_tier"] = speed
        return msgspec.structs.replace(request, params=params)


@register_model
class OpenAIChatCompletion(_OpenAICompatibleChatCompletion):
    """OpenAI Chat Completions provider."""

    provider = "openai"
    chat_extensions = (OpenAIServiceTierExtension(),)
    native_tool_adapters = (OpenAIShellAdapter(), OpenAIApplyPatchAdapter())
    capabilities = ChatProviderCapabilities(
        default_api_mode="responses",
        api_modes=(
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
                reasoning_codec=OpenAIResponsesReasoningCodec(),
                reasoning_summary=True,
                encrypted_reasoning=True,
                request_reasoning_effort=True,
                context_adapter=OpenAIResponsesContextAdapter(),
                hosted_tool_search_model_families=(
                    "gpt-5.6",
                    "gpt-5.6-sol",
                    "gpt-5.6-terra",
                    "gpt-5.6-luna",
                ),
            ),
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=OpenAIReasoningCodec(),
        init_logprobs=True,
        prompt_cache_retention=True,
        uses_max_completion_tokens=True,
    )


@register_model
class OpenAITextToSpeech(OpenAICompatibleHTTPModel, TextToSpeechModel):
    """OpenAI Text to Speech."""

    endpoint = "/audio/speech"

    def __init__(
        self,
        model_id: str,
        voice: Optional[str] = "alloy",
        speed: Optional[float] = 1.0,
        stream_chunk_size: int = 1024,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        voice:
            The voice to use when generating the audio.
        speed:
            the speed of the generated audio. Select a value
            from 0.25 to 4.0. 1.0 is the default.
        stream_chunk_size:
            Number of bytes yielded per streaming audio chunk.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        if not isinstance(stream_chunk_size, int) or stream_chunk_size <= 0:
            raise ValueError("`stream_chunk_size` must be a positive integer")
        self.model_id = model_id
        self.stream_chunk_size = stream_chunk_size
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "voice": voice,
            "speed": speed,
        }
        self.retry = retry
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, **kwargs):
        return self.http_transport.stream(
            self,
            self.endpoint,
            json={"model": self.model_id, **kwargs, **self.sampling_run_params},
            iterate=lambda response: response.iter_bytes(
                chunk_size=self.stream_chunk_size
            ),
        )

    def _aexecute_model(self, **kwargs):
        return self.http_transport.astream(
            self,
            self.endpoint,
            json={"model": self.model_id, **kwargs, **self.sampling_run_params},
            iterate=lambda response: response.aiter_bytes(
                chunk_size=self.stream_chunk_size
            ),
        )

    def _generate(self, **kwargs):
        response = ModelResponse()

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                for chunk in self._execute_model(**kwargs):
                    temp_file.write(chunk)
        except BaseException:
            if temp_file_path is not None:
                Path(temp_file_path).unlink(missing_ok=True)
            raise

        response.set_response_type("audio_generation")
        response.add(temp_file_path)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                async for chunk in self._aexecute_model(**kwargs):
                    temp_file.write(chunk)
        except BaseException:
            if temp_file_path is not None:
                Path(temp_file_path).unlink(missing_ok=True)
            raise

        response.set_response_type("audio_generation")
        response.add(temp_file_path)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        try:
            for chunk in self._execute_model(**kwargs):
                stream_response.add(chunk)
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        try:
            async for chunk in self._aexecute_model(**kwargs):
                stream_response.add(chunk)
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

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
            stream_response = ModelStreamResponse(mode="sync")
            params.stream_response = stream_response
            F.detached(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
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
            stream_response = ModelStreamResponse(mode="async")
            params.stream_response = stream_response
            await F.adetached(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


class _OpenAIImageResponseMixin:
    """Decode image responses independently from their request transport."""

    usage_codec = default_usage_codec

    def _set_image_config(
        self,
        *,
        model_id: str,
        moderation: Optional[Literal["auto", "low"]],
        base_url: Optional[str],
        retry: Optional[Any],
    ) -> None:
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {}
        if moderation:
            self.sampling_run_params["moderation"] = moderation
        self.retry = retry

    @staticmethod
    def _response_field(model_output, name: str, default=None):
        if isinstance(model_output, dict):
            return model_output.get(name, default)
        return getattr(model_output, name, default)

    def _get_metadata(self, model_output):
        return dotdict(
            usage=self.usage_codec.normalize(
                self._response_field(model_output, "usage")
            ),
            details={
                "created": self._response_field(model_output, "created"),
                "size": self._response_field(model_output, "size"),
                "quality": self._response_field(model_output, "quality"),
                "output_format": self._response_field(model_output, "output_format"),
                "background": self._response_field(model_output, "background"),
            },
        )

    def _process_model_output(self, model_output) -> ModelResponse:
        response = ModelResponse()
        response.set_response_type("image_generation")

        images = []
        for item in self._response_field(model_output, "data", ()) or ():
            url = self._response_field(item, "url")
            b64_json = self._response_field(item, "b64_json")
            if url:
                images.append(url)
            if b64_json:
                images.append(b64_json)

        response.add(images[0] if len(images) == 1 else images)
        response.set_metadata(self._get_metadata(model_output))
        return response

    def _generate(self, **kwargs):
        return self._process_model_output(self._execute_model(**kwargs))

    async def _agenerate(self, **kwargs):
        return self._process_model_output(await self._aexecute_model(**kwargs))


@register_model
class OpenAITextToImage(
    _OpenAIImageResponseMixin,
    OpenAICompatibleHTTPModel,
    TextToImageModel,
):
    """OpenAI Image Generation."""

    endpoint = "/images/generations"

    def __init__(
        self,
        *,
        model_id: str,
        moderation: Optional[Literal["auto", "low"]] = None,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        moderation:
            Control the content-moderation level for images generated.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        self._set_image_config(
            model_id=model_id,
            moderation=moderation,
            base_url=base_url,
            retry=retry,
        )
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, **kwargs):
        return dotdict(
            self._request_json(
                self.endpoint,
                {**kwargs, **self.sampling_run_params},
            )
        )

    async def _aexecute_model(self, **kwargs):
        return dotdict(
            await self._arequest_json(
                self.endpoint,
                {**kwargs, **self.sampling_run_params},
            )
        )

    def __call__(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = self._generate(**generation_params)
        return response

    async def acall(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = await self._agenerate(**generation_params)
        return response


@register_model
class OpenAIImageTextToImage(
    _OpenAIImageResponseMixin,
    OpenAICompatibleHTTPModel,
    ImageTextToImageModel,
):
    """OpenAI Image Edit."""

    endpoint = "/images/edits"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        self._set_image_config(
            model_id=model_id,
            moderation=None,
            base_url=base_url,
            retry=retry,
        )
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, *, files, **kwargs):
        response = self.http_transport.request(
            self,
            self.endpoint,
            data=prepare_multipart_data({**kwargs, **self.sampling_run_params}),
            files=files,
        )
        return dotdict(response.json())

    async def _aexecute_model(self, *, files, **kwargs):
        response = await self.http_transport.arequest(
            self,
            self.endpoint,
            data=prepare_multipart_data({**kwargs, **self.sampling_run_params}),
            files=files,
        )
        return dotdict(response.json())

    def _prepare_files(self, image, mask):
        if isinstance(image, (str, bytes)):
            image = [image]
        files = [
            (
                "image[]",
                prepare_multipart_file(
                    item,
                    default_filename=f"image-{index}.png",
                ),
            )
            for index, item in enumerate(image)
        ]
        if mask:
            files.append(
                (
                    "mask",
                    prepare_multipart_file(mask, default_filename="mask.png"),
                )
            )
        return files

    async def _aprepare_files(self, image, mask):
        if isinstance(image, (str, bytes)):
            image = [image]
        files = [
            (
                "image[]",
                await aprepare_multipart_file(
                    item,
                    default_filename=f"image-{index}.png",
                ),
            )
            for index, item in enumerate(image)
        ]
        if mask:
            files.append(
                (
                    "mask",
                    await aprepare_multipart_file(mask, default_filename="mask.png"),
                )
            )
        return files

    def __call__(
        self,
        prompt: str,
        image: Union[bytes, str, List[Union[bytes, str]]],
        *,
        mask: Optional[Union[bytes, str]] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
        input_fidelity: Optional[Literal["high", "low"]] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "jpeg", "webp"]] = None,
        quality: Optional[Literal["standard", "low", "medium", "high", "auto"]] = None,
        size: Optional[str] = None,
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
        background:
            Background mode for the edited image.
        input_fidelity:
            How strongly the edit should preserve details from input images.
        output_compression:
            Compression level from 0 to 100 for JPEG or WebP output.
        output_format:
            Format of the generated image.
        quality:
            Quality of the generated image.
        size:
            Size of the generated image.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format
        for name, value in (
            ("background", background),
            ("input_fidelity", input_fidelity),
            ("output_compression", output_compression),
            ("output_format", output_format),
            ("quality", quality),
            ("size", size),
        ):
            if value is not None:
                generation_params[name] = value

        files = self._prepare_files(image, mask)
        response = self._generate(**generation_params, files=files)
        return response

    async def acall(
        self,
        prompt: str,
        image: Union[bytes, str, List[Union[bytes, str]]],
        *,
        mask: Optional[Union[bytes, str]] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
        input_fidelity: Optional[Literal["high", "low"]] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "jpeg", "webp"]] = None,
        quality: Optional[Literal["standard", "low", "medium", "high", "auto"]] = None,
        size: Optional[str] = None,
    ) -> ModelResponse:
        """Async version of __call__. Args:
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
        background:
            Background mode for the edited image.
        input_fidelity:
            How strongly the edit should preserve details from input images.
        output_compression:
            Compression level from 0 to 100 for JPEG or WebP output.
        output_format:
            Format of the generated image.
        quality:
            Quality of the generated image.
        size:
            Size of the generated image.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format
        for name, value in (
            ("background", background),
            ("input_fidelity", input_fidelity),
            ("output_compression", output_compression),
            ("output_format", output_format),
            ("quality", quality),
            ("size", size),
        ):
            if value is not None:
                generation_params[name] = value

        files = await self._aprepare_files(image, mask)
        response = await self._agenerate(**generation_params, files=files)
        return response


@register_model
class OpenAISpeechToText(OpenAICompatibleHTTPModel, SpeechToTextModel):
    """OpenAI Speech to Text."""

    endpoint = "/audio/transcriptions"

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Optional[float] = 0.0,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        temperature:
            The sampling temperature, between 0 and 1.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"temperature": temperature}
        self.retry = retry
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, *, file, response_format, **kwargs):
        response = self.http_transport.request(
            self,
            self.endpoint,
            data=prepare_multipart_data(
                {
                    **kwargs,
                    **self.sampling_run_params,
                    "response_format": response_format,
                }
            ),
            files=[("file", file)],
        )
        return self._decode_response(response, response_format)

    async def _aexecute_model(self, *, file, response_format, **kwargs):
        response = await self.http_transport.arequest(
            self,
            self.endpoint,
            data=prepare_multipart_data(
                {
                    **kwargs,
                    **self.sampling_run_params,
                    "response_format": response_format,
                }
            ),
            files=[("file", file)],
        )
        return self._decode_response(response, response_format)

    @staticmethod
    def _decode_response(response, response_format):
        if response_format in {"text", "srt", "vtt"}:
            return response.text
        return dotdict(response.json())

    def _process_transcript(self, model_output, response_format):
        response = ModelResponse()
        response.set_response_type("transcript")
        if isinstance(model_output, str):
            transcript = {"text": model_output}
            usage = None
            details = {"response_format": response_format}
        else:
            transcript = dict(model_output)
            raw_usage = transcript.pop("usage", None)
            usage = self.usage_codec.normalize(raw_usage)
            details = {
                "response_format": response_format,
                "language": transcript.get("language"),
                "duration": transcript.get("duration"),
            }
        response.add(transcript)
        response.set_metadata(dotdict(usage=usage, details=details))
        return response

    def _generate(self, *, response_format, **kwargs):
        return self._process_transcript(
            self._execute_model(response_format=response_format, **kwargs),
            response_format,
        )

    async def _agenerate(self, *, response_format, **kwargs):
        return self._process_transcript(
            await self._aexecute_model(response_format=response_format, **kwargs),
            response_format,
        )

    def _stream_model(self, *, file, **kwargs):
        return self.http_transport.stream(
            self,
            self.endpoint,
            data=prepare_multipart_data({**kwargs, **self.sampling_run_params}),
            files=[("file", file)],
            iterate=lambda response: iter_sse_json(response.iter_lines()),
        )

    def _astream_model(self, *, file, **kwargs):
        return self.http_transport.astream(
            self,
            self.endpoint,
            data=prepare_multipart_data({**kwargs, **self.sampling_run_params}),
            files=[("file", file)],
            iterate=lambda response: aiter_sse_json(response.aiter_lines()),
        )

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        try:
            for event in self._stream_model(**kwargs):
                if event.get("type") == "transcript.text.delta":
                    chunk = event.get("delta")
                    if chunk:
                        stream_response.add(chunk)
                elif event.get("type") == "transcript.text.done":
                    self._set_stream_metadata(
                        stream_response,
                        event,
                        response_format=kwargs.get("response_format"),
                    )
                    break
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

        return stream_response

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        try:
            async for event in self._astream_model(**kwargs):
                if event.get("type") == "transcript.text.delta":
                    chunk = event.get("delta")
                    if chunk:
                        stream_response.add(chunk)
                elif event.get("type") == "transcript.text.done":
                    self._set_stream_metadata(
                        stream_response,
                        event,
                        response_format=kwargs.get("response_format"),
                    )
                    break
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

        return stream_response

    def _set_stream_metadata(self, stream_response, event, *, response_format):
        stream_response.set_metadata(
            dotdict(
                usage=self.usage_codec.normalize(event.get("usage")),
                details={"response_format": response_format},
            )
        )

    def __call__(
        self,
        data: Union[bytes, str],
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt", "diarized_json"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        include: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        chunking_strategy: Optional[Union[str, Dict[str, Any]]] = None,
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
        include:
            Additional response fields, such as token log probabilities.
        keywords:
            Words or phrases that should guide supported transcription models.
        languages:
            Candidate input languages for supported transcription models.
        chunking_strategy:
            Server-side audio chunking strategy, such as `auto` or VAD options.
        """
        file = prepare_multipart_file(data, default_filename="audio.wav")
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
            "include": include,
            "keywords": keywords,
            "languages": languages,
            "chunking_strategy": chunking_strategy,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="sync")
            params["stream_response"] = stream_response
            params["stream"] = stream
            F.detached(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: Union[bytes, str],
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt", "diarized_json"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        include: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        chunking_strategy: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
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
        include:
            Additional response fields, such as token log probabilities.
        keywords:
            Words or phrases that should guide supported transcription models.
        languages:
            Candidate input languages for supported transcription models.
        chunking_strategy:
            Server-side audio chunking strategy, such as `auto` or VAD options.
        """
        file = await aprepare_multipart_file(data, default_filename="audio.wav")
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
            "include": include,
            "keywords": keywords,
            "languages": languages,
            "chunking_strategy": chunking_strategy,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="async")
            params["stream_response"] = stream_response
            params["stream"] = stream
            await F.adetached(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextEmbedder(OpenAICompatibleHTTPModel, TextEmbedderModel):
    """OpenAI Text Embedder."""

    batch_support: bool = True
    endpoint = "/embeddings"

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, **kwargs):
        params = {**kwargs, **self.sampling_run_params}
        return dotdict(
            self._request_json(
                self.endpoint,
                {name: value for name, value in params.items() if value is not None},
            )
        )

    async def _aexecute_model(self, **kwargs):
        params = {**kwargs, **self.sampling_run_params}
        return dotdict(
            await self._arequest_json(
                self.endpoint,
                {name: value for name, value in params.items() if value is not None},
            )
        )

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict(
            {"usage": self.usage_codec.normalize(model_output.get("usage"))}
        )
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = await self._aexecute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict(
            {"usage": self.usage_codec.normalize(model_output.get("usage"))}
        )
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ):
        """Args:
        data:
            Input text to embed.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[str]],
    ):
        """Async version of __call__. Args:
        data:
            Input text to embed.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response


@register_model
class OpenAIModeration(OpenAICompatibleHTTPModel, ModerationModel):
    """OpenAI Moderation."""

    endpoint = "/moderations"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
        http_transport: Optional[Union[HTTPTransport, type[HTTPTransport]]] = None,
        credential_resolver: Optional[ModelCredentialResolver] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        http_transport:
            Direct HTTP transport. Instances may inject custom sync and async clients.
        credential_resolver:
            Request-time authentication resolver.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        if http_transport is not None:
            self.http_transport = http_transport
        self._set_credential_resolver(credential_resolver)
        self._initialize()

    def _execute_model(self, **kwargs):
        return dotdict(self._request_json(self.endpoint, kwargs))

    async def _aexecute_model(self, **kwargs):
        return dotdict(await self._arequest_json(self.endpoint, kwargs))

    @staticmethod
    def _process_model_output(model_output: dotdict) -> dotdict:
        results = model_output.get("results") or []
        if not results:
            raise ValueError("OpenAI moderation response did not contain results")
        moderation = dotdict({"results": results[0]})
        moderation.safe = not moderation.results.flagged
        return moderation

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = self._execute_model(**kwargs)
        response.add(self._process_model_output(model_output))

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = await self._aexecute_model(**kwargs)
        response.add(self._process_model_output(model_output))

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[str], List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[str], List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Async version of __call__. Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response
