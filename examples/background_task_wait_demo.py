# /// script
# dependencies = []
# ///

import argparse
import time

import msgflux as mf
from msgflux import nn

mf.load_dotenv()


def build_agent(model_name: str) -> nn.Agent:
    model_kwargs = (
        {"reasoning_effort": "none"}
        if model_name.startswith("openai/gpt-5.6-luna")
        else {}
    )
    model = mf.Model.chat_completion(model_name, **model_kwargs)

    @mf.tool_config(background=True)
    def slow_square(x: int) -> int:
        """Compute the square of a number in the background."""
        time.sleep(0.4)
        return x * x

    return nn.Agent(
        name="wait_assistant",
        model=model,
        system_message="You are a precise assistant.",
        config={"verbose": True},
        instructions=(
            "When a background task result is required to answer the user, "
            "call task_wait(task_id=...) immediately after dispatch. "
            "Do not answer until you have the final result."
        ),
        tools=[slow_square],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    args = parser.parse_args()

    assistant = build_agent(args.model)
    response = assistant(
        "Use the slow_square tool to compute 12 squared. "
        "Wait for the task to finish, then answer with only the number."
    )

    print(f"model={args.model}")  # noqa: T201
    print(response)  # noqa: T201


if __name__ == "__main__":
    main()
