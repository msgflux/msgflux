# How to Debug an Agent

Understanding what's happening inside your agent is essential for building reliable AI applications. When an agent produces unexpected results, you need visibility into the prompts being sent, the model's reasoning, and how tools are being called.

msgFlux provides several inspection mechanisms to help you debug and understand agent behavior:

- **Verbose Mode**: Real-time console output of model calls and tool executions
- **Inspect Model Execution**: View the exact parameters that will be passed to the LM
- **Return Messages**: Retur the interal agent's messages
- **State Dict**: Inspect the agent's internal buffers and parameters

???+ example

    === "Verbose Mode"

        Verbose mode will print the model call steps, tool calls and their return values ​​to the console.

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"verbose": True}

        agent = Assistant()
        response = agent("Can I help me?")        
        ```

        Expected Output:
        
        ```bash
        [Assistant][call_model]
        [Assistant][response] Of course! How can I assist you today?
        ```

    === "Inspect Model Execution"

        This inspection allows you to view the exact values ​​that will be passed to the LM call.

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        agent = Assistant()
        params = agent.inspect_model_execution_params("Hello")
        print(params)
        ```

        When tools are available, `params.tool_catalog` is the immutable
        `ToolCatalogView` prepared by the Agent before provider adaptation.
        Inspect the regular logical tools with:

        ```python
        names = [entry.name for entry in params.tool_catalog.tool_entries()]
        print(names)
        ```

        This view retains thread identity, loading state, stable references,
        and native bindings. The concrete Model provider compiles it to the
        selected wire protocol only when the request is executed.

        Expected Output:
        
        ```bash
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "system_prompt": None,
            "prefilling": None,
            "stream": False,
            "tool_catalog": None,
            "generation_schema": None
        }
        ```


    === "Return Messages"

        Another inspection possibility is to analyze the internal agent state (messages). In msgFlux this is called `messages`. Returning the `messages` allows you to continue an interaction in future calls.

        When the configuration `config={"return_messages": True}` is passed, the agent returns a dict containing the keys `response` and `messages`.

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"return_messages": True}

        agent = Assistant()
        response = agent("Hello")
        print(response.response)
        print(response.messages)
        ```

        Expected Output:
        
        ```bash
        Hello! How can I assist you today?
        [dotdict({
        'role': 'user'
        'content': 'Hello'
        })]
        ```

    === "State Dict"

        To inspect the agent's buffers and parameters, simply call its *.state_dict()* method.

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_prompt = """
            You are a helpful assistant. Treat the user well and return a
            correct, concise response.
            """

        agent = Assistant()
        print(agent.state_dict())
        ```

        The returned mapping includes the canonical `system_prompt`, registered
        buffers, model state, and child modules. Runtime prompt contributions
        from extensions are visible through `agent.get_system_prompt()`.
