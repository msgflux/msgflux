"""Optimize two Agent instructions together and restore them in a new program."""

from pathlib import Path
from tempfile import TemporaryDirectory

from optimize_triage import build_agent, reflection

from msgflux import Example, nn
from msgflux.optim import GEPA, OptimizationArtifact


class TicketPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = build_agent()
        self.second = build_agent()

    def forward(self, first_ticket, second_ticket):
        return [self.first(first_ticket), self.second(second_ticket)]


def accuracy(example, prediction):
    hits = sum(a == b for a, b in zip(example.labels, prediction, strict=True))
    return hits / len(example.labels), {"expected": example.labels}


def main():
    train = Example(
        {"first_ticket": "Missing invoice", "second_ticket": "Reset password"},
        ["billing", "access"],
    )
    val = Example(
        {"first_ticket": "Invoice copy", "second_ticket": "Forgot password"},
        ["billing", "access"],
    )
    fit = GEPA(reflection, max_metric_calls=30).compile(
        TicketPair, trainset=[train], valset=[val], metric=accuracy
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "program.json"
        fit.save(path)
        restored = TicketPair()
        OptimizationArtifact.load(path).apply(restored)
        if restored("Another invoice", "New password") != ["billing", "access"]:
            raise RuntimeError("Restored program did not apply both instructions")
    return fit


if __name__ == "__main__":
    main()
