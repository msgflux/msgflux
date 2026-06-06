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

    @mf.tool_config(background=True, inject_notification=True)
    def slow_pipeline(ticket_id: str, notification) -> str:
        """Run a synthetic pipeline and publish status updates."""
        notification.update(
            "prepare",
            hint="Background work started.",
            metadata={"stage": "prepare"},
            dedupe_key=f"pipeline:{ticket_id}",
        )
        time.sleep(0.5)
        notification.update(
            "process",
            metadata={"stage": "process"},
            dedupe_key=f"pipeline:{ticket_id}",
        )
        time.sleep(0.8)
        return f"{ticket_id}: indexed 24 files and generated 3 summaries"

    return nn.Agent(
        name="status_updates_assistant",
        model=model,
        system_message="You are a precise assistant.",
        config={"verbose": True},
        instructions=(
            "If you receive a notification with source=tool_status, summarize the "
            "latest background status update briefly. "
            "If you receive a task notification with status=completed, call "
            "task_output(task_id=...) before answering."
        ),
        tools=[slow_pipeline],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    args = parser.parse_args()

    assistant = build_agent(args.model)
    history = mf.ChatMessages()

    dispatch_response = assistant(
        "Start a background pipeline for ticket MSGFLUX-77. "
        "Only dispatch it for now.",
        messages=history,
    )
    time.sleep(0.2)
    status_response = assistant(
        "Continue. If there is a status notification, tell me the current stage only.",
        messages=history,
    )
    time.sleep(1.4)
    final_response = assistant(
        "Continue again. If the task completed, consume the output and tell me the result.",
        messages=history,
    )

    print(f"model={args.model}")
    print("dispatch:", dispatch_response)
    print("status:", status_response)
    print("final:", final_response)


if __name__ == "__main__":
    main()
