# Vars

`vars` is the **unified execution variable space** — a `dict`-like object that flows across the entire execution: templates, tools, and agents. It can be read, updated, and overwritten at any point during a run.

Three use cases drive its design:

- **Templates** — inject runtime context into prompts without hardcoding values in the agent definition
- **Tool output isolation** — a tool can write internal data back to `vars` instead of returning it to the model, giving you precise control over what the model sees
- **Security** — pass sensitive values (API keys, internal IDs, auth tokens) that tools need but the model must never be exposed to

???+ example "Examples"

    === "Templates"

        Render runtime values into the agent's Jinja2 templates. The model sees the rendered text — not the variable names:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class SupportAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = """
            You are assisting {{ customer_name }}.
            {% if is_premium %}This is a premium customer — prioritize their request.{% endif %}
            """
            config = {"verbose": True}

        agent = SupportAgent()

        vars = {"customer_name": "Clark Kent", "is_premium": True}

        response = agent("My invoice is wrong.", vars=vars)
        ```

    === "Tools"

        `inject_vars` accepts two forms:

        - **`True`** — all vars are passed as a single `vars` dict in `kwargs`
        - **`["field1", "field2"]`** — only the listed fields are injected as direct named arguments; raises an error if any field is missing

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # inject_vars=True — receives all vars as kwargs["vars"]
        @mf.tool_config(inject_vars=True)
        def get_discount_full(**kwargs) -> str:
            """Get the discount for the current customer."""
            vars = kwargs.get("vars")
            customer_name = vars.get("customer_name", "Guest")
            return f"{customer_name} has a 15% loyalty discount."


        # inject_vars=[...] — selected fields become direct named arguments
        @mf.tool_config(inject_vars=["customer_name"])
        def get_discount_selective(customer_name: str) -> str:
            """Get the discount for the current customer."""
            return f"{customer_name} has a 15% loyalty discount."


        class BusinessAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_discount_selective]
            config = {"verbose": True}

        agent = BusinessAgent()

        vars = {"customer_name": "Clark Kent"}

        response = agent("What discount do I have?", vars=vars)
        ```

        See [inject_vars](tools/config.md#inject_vars) for more details.

    === "Skills"

        Load agent skills dynamically from `vars`. Each skill has a `description` (shown in the prompt) and `content` (the full instructions). The model sees only the names and descriptions — it calls the `skill` tool to retrieve the content before acting.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        SKILLS = {
            "eli5": {
                "description": "Explain any concept as if the user is 5 years old",
                "content": (
                    "Use only simple words and short sentences. Build one analogy around something "
                    "a child already knows — toys, food, playgrounds. Never use jargon. "
                    "End with a single sentence the child could repeat to a friend."
                ),
            },
            "roast": {
                "description": "Give brutally honest, constructive criticism on a piece of writing",
                "content": (
                    "Be direct and unsparing. Call out weak arguments, vague language, "
                    "hollow transitions, and missed opportunities. Do not soften the feedback. "
                    "Close with exactly three concrete, actionable improvements."
                ),
            },
            "pitch": {
                "description": "Turn any idea into a compelling one-paragraph startup pitch",
                "content": (
                    "Open with the painful problem. Name the target customer in one phrase. "
                    "State the solution in one sentence. End with the unfair advantage — "
                    "why this team or insight wins. Stay under 80 words. No fluff."
                ),
            },
        }


        @mf.tool_config(inject_vars=["skills"])
        def skill(name: str, **kwargs) -> str:
            """Read the full instructions for a skill by name."""
            skills = kwargs["skills"]
            entry = skills.get(name)
            if not entry:
                return f"Skill '{name}' not found."
            return entry["content"]


        class TaskAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a sharp, versatile assistant."
            instructions = """
            You have the following skills:

            {% for name, meta in skills.items() %}
            - **{{ name }}**: {{ meta.description }}
            {% endfor %}

            When the task matches a skill, call `skill` with the skill name to load its full
            instructions before you respond. Never guess — always read the skill first.
            """
            tools = [skill]
            config = {"verbose": True}


        agent = TaskAgent()

        vars = {"skills": SKILLS}

        agent("Roast this intro: 'In today's fast-paced world, AI is more important than ever.'", vars=vars)
        ```

        The model never sees `content` — it only sees the skill list rendered in the prompt, then fetches the instructions on demand via the tool.

    === "Storing data"

        A tool can write data back into `vars` instead of returning it to the model. This is useful when a tool retrieves internal records that your application needs downstream, but that would only add noise to the model's context.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        ACCOUNTS = {
            "C-001": {"name": "Clark Kent", "plan": "Premium", "balance": 1250.00, "overdue": False},
            "C-002": {"name": "Lois Lane",  "plan": "Basic",   "balance":   45.00, "overdue": True},
        }

        @mf.tool_config(inject_vars=True)
        def load_account(**kwargs) -> str:
            """Load the customer's account details."""
            vars = kwargs.get("vars")
            account_id = vars.get("account_id")
            account = ACCOUNTS.get(account_id)
            if not account:
                return "Account not found."

            # Full record stored in vars — available to the application after the call
            vars["account"] = account

            # Model only sees a short confirmation
            return f"Account loaded. Plan: {account['plan']}. Balance: ${account['balance']:.2f}."

        class BillingAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [load_account]
            config = {"verbose": True}

        agent = BillingAgent()

        vars = {"account_id": "C-001"}

        agent("Is my account up to date?", vars=vars)

        # Full account data is now available here — the model never saw it
        print(vars["account"])
        # {'name': 'Clark Kent', 'plan': 'Premium', 'balance': 1250.0, 'overdue': False}
        ```

    === "Security"

        Some values must reach your tools but must never appear in the model's context — API keys, internal service tokens, customer IDs used for authenticated lookups. Pass them through `vars`: the model is unaware they exist.

        Using `inject_vars=["customer_id"]` makes the intent explicit: only that field is injected, directly as a named argument.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        ORDER_DB = {
            "C-001": [
                {"id": "ORD-100", "item": "Laptop",  "status": "Delivered"},
                {"id": "ORD-101", "item": "Monitor", "status": "Shipped"},
            ]
        }

        @mf.tool_config(inject_vars=["customer_id"])
        def list_orders(customer_id: str) -> str:
            """List the customer's recent orders."""
            orders = ORDER_DB.get(customer_id, [])

            if not orders:
                return "No orders found."

            lines = [f"- {o['item']} ({o['status']})" for o in orders]
            return "Recent orders:\n" + "\n".join(lines)

        class OrderAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [list_orders]
            config = {"verbose": True}

        agent = OrderAgent()

        # customer_id is injected securely — the model has no access to it
        vars = {"customer_id": "C-001"}

        agent("Show me my recent orders.", vars=vars)
        ```

        The model calls `list_orders()` with no arguments — it cannot infer, leak, or hallucinate the `customer_id`. Authentication happens entirely inside the tool.

    === "Message"

        Inside a pipeline, vars live on the `Message` object. Use `message_fields={"vars": "field_name"}` to tell the agent where to read them — no need to pass `vars=` explicitly on every call.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=["customer_name"])
        def get_discount(customer_name: str) -> str:
            """Get the discount for the current customer."""
            return f"{customer_name} has a 15% loyalty discount."

        class SupportAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "You are assisting {{ customer_name }}."
            tools = [get_discount]
            message_fields = {"task": "query", "vars": "variables"}  # (1)!
            response_mode = "answer"
            config = {"verbose": True}

        agent = SupportAgent()

        msg = mf.Message()
        msg.query     = "What discount do I have?"
        msg.variables = {"customer_name": "Clark Kent"}  # (2)!

        agent(msg)
        print(msg.answer)
        ```

        1. `"vars": "variables"` tells the agent to read vars from `msg.variables`. The same dict flows into both the Jinja2 template and the tool.
        2. Set the vars dict on any field — the name is arbitrary, as long as it matches `message_fields`.
