"""Checkpoint before network requests made by the coding host."""

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook


class CodingCheckpointExtension(AgentExtension):
    """Persist canonical context before asking the provider to generate.

    This is coding-host policy. Model-facing transforms and the UI transcript
    never become a second history store. Normal Agent checkpoint/revision and
    workspace-reference machinery remains responsible for the commit.
    """

    def __init__(self):
        super().__init__("coding_checkpoints")

    def hooks(self):
        return (Hook(event="transform_context", handler=self._checkpoint),)

    def _checkpoint(self, context):
        self.agent._checkpoint_save(context.messages, context.vars, status="running")
        return context
