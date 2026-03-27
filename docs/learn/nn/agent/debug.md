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
        # pip install msgflux[openai]
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
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        agent = Assistant()
        params = agent.inspect_model_execution_params("Hello")
        print(params)
        ```

        Expected Output:
        
        ```bash
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<task>Hello</task>"
                }
            ],
            "system_prompt": None,
            "prefilling": None,
            "stream": False,
            "tool_schemas": None,
            "tool_choice": None,
            "generation_schema": None,
            "typed_parser": None
        }
        ```


    === "Return Messages"

        Another inspection possibility is to analyze the internal agent state (messages). In msgFlux this is called `messages`. Returning the `messages` allows you to continue an interaction in future calls.

        When the configuration `config={"return_messages": True}` is passed, the agent returns a dict containing the keys `response` and `messages`.

        ```python
        # pip install msgflux[openai]
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
        'content': '<task>Hello</task>'
        })]
        ```

    === "State Dict"

        To inspect the agent's buffers and parameters, simply call its *.state_dict()* method.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message  = "You are a helpful assistant."
            instructions    = "Treat the user well."
            expected_output = "Correct responses."

        agent = Assistant()
        print(agent.state_dict())
        ```

        Expected Output:
        
        ```bash
        {
            "examples": None,
            "expected_output": "Correct responses.",
            "instructions": "Treat the user well.",
            "system_message": "You are a helpful assistant.",
            "name": "Assistant",
            "description": None,
            "config": {},
            "context_cache": None,
            "task": None,
            "task_multimodal": None,
            "model_preference": None,
            "task_context": None,
            "messages": None,
            "vars": None,
            "prefilling": None,
            "system_extra_message": None,
            "response_mode": None,
            "typed_parser": None,
            "generation_schema": None,
            "lm.model": {
                "msgflux_type": "model",
                "provider": "openai",
                "model_type": "chat_completion",
                "state": {
                    "model_id": "gpt-4.1-mini",
                    "context_length": None,
                    "reasoning_max_tokens": None,
                    "enable_cache": False,
                    "cache_size": 128,
                    "sampling_params": {
                        "base_url": None
                    },
                    "sampling_run_params": {
                        "max_tokens": None
                    },
                    "enable_thinking": None,
                    "parallel_tool_calls": true,
                    "reasoning_in_tool_call": true,
                    "validate_typed_parser_output": False,
                    "return_reasoning": False,
                    "verbose": False,
                    "current_key_index": 0
                }
            },
            "tool_library.name": "Assistant_tool_library",
            "tool_library.special_library": [],
            "tool_library.tool_configs": {},
            "tool_library.mcp_clients": {}
        }
        ```
