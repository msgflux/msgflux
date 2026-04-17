!!! info "Setup your chat completion model (<a href='/dependency-management/#chat-completion'>check dependencies</a>)"

    === "OpenAI"

        Authenticate by setting the `OPENAI_API_KEY` env variable.

        ```python
        import msgflux as mf

        mf.set_envs(OPENAI_API_KEY="...")
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
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

    === "Other providers"

        msgFlux supports 12+ providers. Any provider with an OpenAI-compatible API works:

        ```python
        import msgflux as mf

        # Together AI
        model = mf.Model.chat_completion("together/openai/gpt-oss-120b")

        # Cerebras
        model = mf.Model.chat_completion("cerebras/openai/gpt-oss-120b")
        ```
