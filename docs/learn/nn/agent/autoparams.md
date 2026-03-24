# AutoParams

The **preferred** way to define agents is using class attributes:

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class SalesAgent(nn.Agent):
        """Persuasive sales agent focused on customer needs."""

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        system_message = "You are an expert sales professional."
        instructions = "Identify customer needs and present value propositions."
        expected_output = "Respond with three bullet points."
        config = {"verbose": True}

    agent = SalesAgent()
    print(agent.name)         # "SalesAgent"
    print(agent.description)  # "Persuasive sales agent focused on customer needs."
    ```

AutoParams captures the class name as the agent *name*. The class docstring then becomes the *description* of that agent, which is useful for using that **agent-as-tool**.
