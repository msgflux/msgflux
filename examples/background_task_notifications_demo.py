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
    def slow_lookup(ticket_id: str) -> str:
        """Resolve a synthetic ticket value in the background."""
        time.sleep(0.4)
        return f"{ticket_id}: owner=platform, priority=high"

    return nn.Agent(
        name="notification_assistant",
        model=model,
        system_message="You are a precise assistant.",
        config={"verbose": True},
        instructions=(
            "If the user explicitly asks to only dispatch background work, "
            "call the background tool and stop after confirming the dispatch. "
            "On later turns, if you receive a completed task notification, "
            "call task_output(task_id=...) before answering."
        ),
        tools=[slow_lookup],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    args = parser.parse_args()

    assistant = build_agent(args.model)
    history = mf.ChatMessages()

    dispatch_response = assistant(
        "Start a background lookup for ticket MSGFLUX-42. "
        "Only dispatch it for now. Do not wait for the final value yet.",
        messages=history,
    )
    time.sleep(0.8)
    final_response = assistant(
        "Continue. If a task completed notification arrived, consume it and "
        "tell me the final ticket value.",
        messages=history,
    )

    print(f"model={args.model}")
    print("dispatch:", dispatch_response)
    print("final:", final_response)


if __name__ == "__main__":
    main()
