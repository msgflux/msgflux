!!! info "Init Model ([`Check Dependencies`](../dependency-management.md#chatcompletion))"
    === "OpenAI"
        Configure your access key `OPENAI_API_KEY`
        ```python
        mf.set_envs(OPENAI_API_KEY="<>")
        model = mf.Model.chat_completion("openai/gpt-4.1-nano", temperature=0.7")
        ```
    === "OpenRouter"
        Configure your access key `OPENROUTER_API_KEY`    
        ```python
        mf.set_envs(OPENROUTER_API_KEY="<>")
        model = mf.Model.chat_completion("openrouter/deepseek/deepseek-r1-distill-qwen-7b")
        ```
    === "SambaNova"
        Configure your access key `SAMBANOVA_API_KEY`    
        ```python
        mf.set_envs(SAMBANOVA_API_KEY="<>")
        model = mf.Model.chat_completion("Llama-4-Maverick-17B-128E-Instruct")
        ```        
    === "vLLM"
        Install vLLM (optional)
        ```bash
        pip install uv # for fast package install
        uv pip install vllm --torch-backend=auto
        ```
        Start vLLM server
        ```python
        vllm serve Qwen/Qwen2.5-1.5B-Instruct 
        ```
        Configure your URL `VLLM_BASE_URL`
        ```python
        base_url = "http://localhost:8000/v1"
        # mf.set_envs(VLLM_BASE_URL=base_url) # or pass as a `base_url` param
        model = mf.Model.chat_completion("Qwen/Qwen2.5-1.5B-Instruct", base_url=base_url)
        ```
    === "More ▼"
        Configure your access key `OPENROUTER_API_KEY`    
        ```python
        mf.set_envs(OPENROUTER_API_KEY="<>")
        model = mf.Model.chat_completion("openrouter/deepseek/deepseek-r1-distill-qwen-7b")
        ```
        === "OI"
        Configure your access key `OPENROUTER_API_KEY`    
        ```python
        mf.set_envs(OPENROUTER_API_KEY="<>")
        model = mf.Model.chat_completion("openrouter/deepseek/deepseek-r1-distill-qwen-7b")
        ```
        === "SOM"
        Configure your access key `OPENROUTER_API_KEY`    
        ```python
        mf.set_envs(OPENROUTER_API_KEY="<>")
        model = mf.Model.chat_completion("openrouter/deepseek/deepseek-r1-distill-qwen-7b")
        ```
