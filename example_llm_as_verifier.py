"""Quick verbose example for `LLMAsVerifier`.

Run:
    uv run python example_llm_as_verifier.py
"""

# ruff: noqa: T201

import msgflux as mf
from msgflux.generation.verifiers import (
    LLMAsVerifier,
    ScoreScale,
    format_terminal_trajectory,
)

MODEL_NAME = "openai/gpt-4.1-mini"


def print_divider(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_verbose_outputs(result) -> None:
    raw_outputs = result.metadata.get("raw_outputs", [])
    for index, output in enumerate(raw_outputs, start=1):
        print_divider(f"Attempt {index}")
        print("Criterion:", output.get("criterion_name"))
        print("Repetition:", output.get("repetition"))
        print("\nPrompt:\n")
        print(output.get("prompt", ""))
        print("\nRaw response:\n")
        print(output.get("response_text", ""))
        print("\nEvidence:")
        for label, evidence in output.get("evidence", {}).items():
            print(f"  - {label}: {evidence}")


def run_pairwise_example() -> None:
    print_divider("Pairwise Example")
    verifier = LLMAsVerifier.answer_reranking(
        model=mf.Model.chat_completion(MODEL_NAME, temperature=0, max_tokens=220),
        score_scale=ScoreScale.letter(granularity=20),
        strict_logprobs=True,
        verbose=True,
    )

    result = verifier(
        task="What is the capital of France?",
        candidates={
            "correct": "Paris is the capital of France.",
            "wrong": "Lyon is the capital of France.",
        },
    )

    print("Verdict:", result.verdict)
    print("Winner:", result.winner)
    print("Scores:", result.scores)
    print_verbose_outputs(result)


def run_tournament_example() -> None:
    print_divider("Tournament Example")
    verifier = LLMAsVerifier.terminal_bench(
        model=mf.Model.chat_completion(MODEL_NAME, temperature=0, max_tokens=220),
        # This example is meant for manual inspection. Keep fallback parsing
        # enabled here so an occasional malformed score tag does not abort the run.
        strict_logprobs=False,
        verbose=True,
    )

    result = verifier.select_best(
        task=(
            "Install the binary to /usr/local/bin/tool and ensure running "
            "`tool --version` prints `tool 1.2.0`."
        ),
        candidates={
            "best": format_terminal_trajectory(
                summary="Installed the binary in the requested path and verified it.",
                metadata={"expected_output": "tool 1.2.0"},
                steps=[
                    {
                        "command": "cp ./tool /usr/local/bin/tool",
                        "exit_code": 0,
                    },
                    {
                        "command": "tool --version",
                        "output": "tool 1.2.0",
                        "exit_code": 0,
                    },
                ],
                final_answer="Installation complete.",
            ),
            "okay": format_terminal_trajectory(
                summary="Installed the binary and ran the version command.",
                metadata={"expected_output": "tool 1.2.0"},
                steps=[
                    {
                        "command": "cp ./tool /usr/local/bin/tool",
                        "exit_code": 0,
                    },
                    {
                        "command": "tool --version",
                        "output": "tool version 1.2.0",
                        "exit_code": 0,
                    },
                ],
                final_answer="Installation complete.",
            ),
            "wrong": format_terminal_trajectory(
                summary="Installed the binary in the wrong path.",
                metadata={"expected_output": "tool 1.2.0"},
                steps=[
                    {
                        "command": "cp ./tool /tmp/tool",
                        "exit_code": 0,
                    },
                    {
                        "command": "/workspace/tool --version",
                        "output": "tool 1.2.0",
                        "exit_code": 0,
                    },
                ],
                final_answer="Installation complete.",
            ),
        },
    )

    print("Winner:", result.winner)
    print("Ranking:", result.ranking)
    print("Wins:", result.wins)
    print("Average scores:", result.average_scores)

    for index, match in enumerate(result.metadata.get("raw_outputs", []), start=1):
        print_divider(f"Match {index}")
        print(
            "Candidates:",
            match.get("candidate_a_label"),
            "vs",
            match.get("candidate_b_label"),
        )
        for output_index, output in enumerate(match.get("outputs", []), start=1):
            print(f"\n  Output {output_index}:")
            print("  Criterion:", output.get("criterion_name"))
            print("  Repetition:", output.get("repetition"))
            print("\n  Prompt:\n")
            print(output.get("prompt", ""))
            print("\n  Raw response:\n")
            print(output.get("response_text", ""))


def main() -> None:
    mf.load_dotenv()
    run_pairwise_example()
    run_tournament_example()


if __name__ == "__main__":
    main()
