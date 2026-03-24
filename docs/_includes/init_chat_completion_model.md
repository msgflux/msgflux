!!! info "Setup your chat completion model"

    ```bash
    pip install msgflux[openai]
    ```

    === "OpenAI"

        Authenticate by setting the `OPENAI_API_KEY` env variable or using `set_envs`.

        ```python
        mf.set_envs(OPENAI_API_KEY="...")
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        ```

    === "Groq"

        Authenticate by setting the `GROQ_API_KEY` env variable.

        ```python
        mf.set_envs(GROQ_API_KEY="...")
        model = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")
        ```

    === "Ollama"

        Install [Ollama](https://ollama.ai) and pull your model first:

        ```bash
        ollama pull llama3.2
        ```

        ```python
        model = mf.Model.chat_completion("ollama/llama3.2")
        ```

    === "OpenRouter"

        Authenticate by setting the `OPENROUTER_API_KEY` env variable.

        ```python
        mf.set_envs(OPENROUTER_API_KEY="...")
        model = mf.Model.chat_completion("openrouter/anthropic/claude-sonnet-4")
        ```

    === "SambaNova"

        Authenticate by setting the `SAMBANOVA_API_KEY` env variable.

        ```python
        mf.set_envs(SAMBANOVA_API_KEY="...")
        model = mf.Model.chat_completion("sambanova/Meta-Llama-3.1-8B-Instruct")
        ```

    === "vLLM"

        Self-hosted with an OpenAI-compatible API:

        ```bash
        vllm serve meta-llama/Llama-3.1-8B-Instruct
        ```

        ```python
        model = mf.Model.chat_completion(
            "vllm/meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        ```

    === "Other providers"

        msgFlux supports 12+ providers. Any provider with an OpenAI-compatible API works:

        ```python
        # Together AI
        model = mf.Model.chat_completion("together/meta-llama/Llama-3.3-70B-Instruct-Turbo")

        # Cerebras
        model = mf.Model.chat_completion("cerebras/llama-3.3-70b")
        ```
