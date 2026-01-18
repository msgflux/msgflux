class Core:
    """Core is a set of methods that are common among some classes in msgflux.

    This class provides basic functionality for state management, allowing instances
    to be serialized and deserialized. It is intended to be inherited by other classes
    that require these common features.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
