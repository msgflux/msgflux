# /// script
# dependencies = []
# ///

import argparse
import time

import msgflux as mf
import msgflux.nn as nn


mf.load_dotenv()


def build_agent(model_name: str) -> nn.Agent:
    model = mf.Model.chat_completion(model_name)

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
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    args = parser.parse_args()

    assistant = build_agent(args.model)
    response = assistant(
        "Use the slow_square tool to compute 12 squared. "
        "Wait for the task to finish, then answer with only the number."
    )

    print(f"model={args.model}")
    print(response)


if __name__ == "__main__":
    main()
