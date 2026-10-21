"""Host-owned lifecycle for a selected group of Agent extensions."""

from msgflux.nn.extensions.discovery import AgentExtensionCatalog


class AgentExtensionInstallation:
    """Install fresh configured instances; explicitly release their resources."""

    def __init__(self, catalog: AgentExtensionCatalog, agent) -> None:
        self.catalog = catalog
        self.agent = agent
        self._handles = []
        self._failed = []

    def install(self, names) -> None:
        for name in names:
            extension = self.catalog.create(name)
            try:
                handle = self.agent.register_extension(name, extension)
            except BaseException:
                self._failed.append(extension)
                raise
            self._handles.append(handle)

    async def aclose(self) -> None:
        errors = []
        for extension in reversed(self._failed):
            try:
                await extension.aon_remove(self.agent)
            except Exception as error:
                errors.append(error)
        self._failed.clear()
        for handle in reversed(self._handles):
            try:
                await handle.aremove()
            except Exception as error:
                errors.append(error)
        self._handles.clear()
        if errors:
            raise ExceptionGroup("Agent extension cleanup failed", errors)
