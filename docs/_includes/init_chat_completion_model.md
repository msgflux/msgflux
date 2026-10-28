!!! info "Setup your chat completion model (<a href='/dependency-management/#chat-completion'>check dependencies</a>)"

    === "OpenAI"

        Authenticate by setting the `OPENAI_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(OPENAI_API_KEY="...")
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        ```

    === "Gemini"

        Authenticate with `GEMINI_API_KEY`. Gemini uses Google's native
        Interactions API through the included HTTPX2 transport.

        ```python
        import msgflux as mf

        mf.set_envs(GEMINI_API_KEY="...")
        model = mf.Model.chat_completion(
            "gemini/gemini-3.6-flash", reasoning_effort="low"
        )
        ```

    === "Groq"

        Authenticate by setting the `GROQ_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(GROQ_API_KEY="...")
        model = mf.Model.chat_completion("groq/openai/gpt-oss-120b")
        ```

    === "Ollama"

        Install [Ollama](https://ollama.ai) and pull your model first:

        ```bash
        ollama pull gpt-oss:120b
        ```

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("ollama/gpt-oss:120b")
        ```

    === "OpenRouter"

        Authenticate by setting the `OPENROUTER_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(OPENROUTER_API_KEY="...")
        model = mf.Model.chat_completion("openrouter/anthropic/claude-opus-4-6")
        ```

    === "OpenCode Go"

        Authenticate with `OPENCODE_API_KEY`. Chat Completions is the default
        protocol; `api_mode="responses"` selects the OpenAI-compatible
        Responses endpoint and `api_mode="anthropic_messages"` selects the
        native Anthropic Messages backend. All three use Bearer authentication
        and the `opencode-go/model` namespace. `OPENCODE_GO_BASE_URL` overrides
        the OpenAI-compatible base URL
        (`https://opencode.ai/inference/go/openai/v1`). Google/Interactions is
        not supported.

        ```python
        import msgflux as mf

        mf.set_envs(OPENCODE_API_KEY="...")
        model = mf.Model.chat_completion("opencode-go/deepseek-v4.1-flash")
        responses_model = mf.Model.chat_completion(
            "opencode-go/gpt-6-luna", api_mode="responses"
        )
        anthropic_model = mf.Model.chat_completion(
            "opencode-go/claude-haiku-5-5",
            api_mode="anthropic_messages",
        )
        ```

    === "SambaNova"

        Authenticate by setting the `SAMBANOVA_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(SAMBANOVA_API_KEY="...")
        model = mf.Model.chat_completion("sambanova/openai/gpt-oss-120b")
        ```

    === "vLLM"

        Self-hosted with an OpenAI-compatible API:

        ```bash
        vllm serve openai/gpt-oss-120b
        ```

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "vllm/openai/gpt-oss-120b",
            base_url="http://localhost:8000/v1",
        )
        ```

    === "Baseten"

        Authenticate with `BASETEN_API_KEY`. The default OpenAI-compatible base
        URL is `https://inference.baseten.co/v1`; set `BASETEN_BASE_URL` to
        override it.

        ```python
        import msgflux as mf

        mf.set_envs(BASETEN_API_KEY="...")
        model = mf.Model.chat_completion("baseten/zai-org/GLM-5.2")
        ```

    === "NVIDIA"

        Authenticate by setting the `NVIDIA_API_KEY` env variable (free trial
        keys available at build.nvidia.com). Model availability and supported
        features depend on the NVIDIA deployment.

        ```python
        import msgflux as mf

        mf.set_envs(NVIDIA_API_KEY="...")
        model = mf.Model.chat_completion("nvidia/openai/gpt-oss-20b")
        ```

    === "Fireworks"

        Authenticate by setting the `FIREWORKS_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(FIREWORKS_API_KEY="...")
        model = mf.Model.chat_completion(
            "fireworks/accounts/fireworks/models/gpt-oss-120b"
        )
        ```

    === "Anthropic"

        Authenticate by setting the `ANTHROPIC_API_KEY` env variable. Uses
        the native Messages API (thinking blocks are replayed verbatim
        for multi-turn and tool flows).

        ```python
        import msgflux as mf

        mf.set_envs(ANTHROPIC_API_KEY="...")
        model = mf.Model.chat_completion(
            "anthropic/claude-haiku-5-5", reasoning_effort="low"
        )
        ```

    === "Other providers"

        msgFlux supports 12+ providers. Any provider with an OpenAI-compatible API works:

        ```python
        import msgflux as mf

        # Together AI
        model = mf.Model.chat_completion("together/openai/gpt-oss-120b")

        # Cerebras
        model = mf.Model.chat_completion("cerebras/openai/gpt-oss-120b")
        ```
