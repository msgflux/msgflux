# System Prompt Components

The system prompt is composed of 7 components:

| Component | Description | Example |
|-----------|-------------|---------|
| **system_message** | Agent behavior and role | "You are an agent specialist in..." |
| **instructions** | What the agent should do | "You MUST respond to the user..." |
| **expected_output** | Format of the response | "Your answer must be concise..." |
| **examples** | Input/output examples | Examples of reasoning and outputs |
| **system_extra_message** | Additional system context | Extra instructions or constraints |
| **include_date** | Include current date | Adds "Weekday, Month DD, YYYY" |
| **tool_usage_guidance** | Tool-specific usage instructions | Guidance from exposed tools, such as when to call each tool |

All components are assembled using a **system prompt template**, that can be customized via `templates={"system": "..."}`. By default, the template concatenates all defined components in a structured format using XML tags.

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class BusinessAgent(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        system_message = """
        You are a business development assistant,
        focused on helping sales teams.
        """
        instructions = """
        When given a company description, identify potential needs,
        suggest an outreach strategy, and provide a value proposition.
        """
        expected_output = """
        Respond in three bullet points:
        - Identified Needs
        - Outreach Strategy
        - Value Proposition
        """
        system_extra_message = """
        Ensure recommendations align with ethical sales practices.
        """
        config = {"include_date": True, "verbose": True}

    agent = BusinessAgent()
    print(agent.get_system_prompt())
    ```

    Expected Output:

    ```bash
    <developer_note>
    <system_message>

        You are a business development assistant,
        focused on helping sales teams.
        
    </system_message>

    <instructions>

        When given a company description, identify potential needs,
        suggest an outreach strategy, and provide a value proposition.
        
    </instructions>

    <expected_output>

        Respond in three bullet points:
        - Identified Needs
        - Outreach Strategy
        - Value Proposition
        
    </expected_output>

        Ensure recommendations align with ethical sales practices.
        

    The current date is: Friday, February 20, 2026

    </developer_note>
    ```

## Examples

**[In-Context Learning (ICL)](https://arxiv.org/abs/2005.14165)** is a technique where language models learn to perform tasks by observing examples provided directly in the prompt, without any parameter updates. This allows models to generalize from just a few demonstrations.

**Few-Shot Learning** refers to providing a small number of input-output examples that guide the model's behavior. These examples act as implicit instructions, showing the model the expected format, reasoning style, and output structure.

Benefits of using examples:

- **Format consistency**: The model mimics the structure of your examples
- **Implicit instructions**: Complex behaviors can be demonstrated rather than explained
- **Reasoning patterns**: Show chain-of-thought reasoning for the model to follow
- **Domain adaptation**: Tailor responses to your specific use case

There are three ways to pass examples to an Agent:

???+ note "Few-shot Example Formats"

    === "String Examples"

        Simple text format:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = """
        Input: A startup offering AI tools for logistics.
        Output:
        - Needs: Supply chain optimization
        - Strategy: Highlight cost savings
        - Value: Reduce delays with predictive analytics

        Input: An e-commerce platform for handmade crafts.
        Output:
        - Needs: Market visibility
        - Strategy: Cross-promotion with eco marketplaces
        - Value: Global audience access for artisans
        """

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a business development assistant."
            instructions = "Identify needs and suggest strategies."
            expected_output = "Three bullet points: Needs, Strategy, Value"
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

        Expected Output:
        
        ```bash
        <developer_note>
        <system_message>
        You are a business development assistant.
        </system_message>

        <instructions>
        Identify needs and suggest strategies.
        </instructions>

        <expected_output>
        Three bullet points: Needs, Strategy, Value
        </expected_output>

        <examples>

        Input: A startup offering AI tools for logistics.
        Output:
        - Needs: Supply chain optimization
        - Strategy: Highlight cost savings
        - Value: Reduce delays with predictive analytics

        Input: An e-commerce platform for handmade crafts.
        Output:
        - Needs: Market visibility
        - Strategy: Cross-promotion with eco marketplaces
        - Value: Global audience access for artisans

        </examples>

        </developer_note>
        ```

    === "Example Class"

        Structured examples with metadata:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = [
            mf.Example(
                inputs="A fintech offering digital wallets.",
                labels={
                    "Needs": "Payment integration and trust",
                    "Strategy": "Position as secure and easy-to-use",
                    "Value": "Simplify digital payments"
                },
                reasoning="Small retailers need trust and ease.",
                title="Fintech Lead",
                topic="Sales"
            ),
            mf.Example(
                inputs="An e-commerce for handmade crafts.",
                labels={
                    "Needs": "Visibility and market reach",
                    "Strategy": "Partner with eco marketplaces",
                    "Value": "Global audience for artisans"
                },
                reasoning="Handmade crafts need visibility to scale."
            )
        ]

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

        Expected Output:
        
        ```bash
        <developer_note>

        <examples>
        <example id=1 title="Fintech Lead" topic="Sales">
        <input>A fintech offering digital wallets.</input>
        <reasoning>Small retailers need trust and ease.</reasoning>
        <output>{"Needs":"Payment integration and trust","Strategy":"Position as secure and easy-to-use","Value":"Simplify digital payments"}</output>
        </example>

        <example id=2>
        <input>An e-commerce for handmade crafts.</input>
        <reasoning>Handmade crafts need visibility to scale.</reasoning>
        <output>{"Needs":"Visibility and market reach","Strategy":"Partner with eco marketplaces","Value":"Global audience for artisans"}</output>
        </example>
        </examples>

        </developer_note>
        ```

    === "Dict Examples"

        Dict-based examples are converted to `Example` objects:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        examples = [
            {
                "inputs": "A startup offering AI tools for logistics.",
                "labels": {
                    "Needs": "Supply chain optimization",
                    "Strategy": "Highlight cost savings",
                    "Value": "Reduce delays with predictive analytics"
                }
            },
            {
                "inputs": "An e-commerce for handmade crafts.",
                "labels": {
                    "Needs": "Market visibility",
                    "Strategy": "Cross-promotion with eco marketplaces",
                    "Value": "Global audience access"
                }
            }
        ]

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            examples = examples

        agent = SalesAgent()
        print(agent.get_system_prompt())
        ```

        Expected Output:
        
        ```bash
        <developer_note>

        <examples>
        <example id=1>
        <input>A startup offering AI tools for logistics.</input>
        <output>{"Needs":"Supply chain optimization","Strategy":"Highlight cost savings","Value":"Reduce delays with predictive analytics"}</output>
        </example>

        <example id=2>
        <input>An e-commerce for handmade crafts.</input>
        <output>{"Needs":"Market visibility","Strategy":"Cross-promotion with eco marketplaces","Value":"Global audience access"}</output>
        </example>
        </examples>

        </developer_note>
        ```
