import importlib
import pathlib
import pkgutil
from types import ModuleType
from typing import Any, Union

_loaded_autoloads = set()


def validate_factory_path(path: str) -> None:
    """Validate an importable factory reference without importing its module."""
    module_name, separator, attribute = path.partition(":")
    if (
        not separator
        or not module_name
        or not attribute
        or any(character.isspace() for character in path)
    ):
        raise ValueError("factory must use the form 'module:attribute'")


def load_factory(path: str) -> Any:
    """Import a configured callable, including nested entry-point attributes."""
    validate_factory_path(path)
    module_name, _, attribute = path.partition(":")
    factory = importlib.import_module(module_name)
    for component in attribute.split("."):
        factory = getattr(factory, component)
    if not callable(factory):
        raise TypeError("configured factory attribute must be callable")
    return factory


def import_module_from_lib(_import: str, _from: str):
    """Import a module (function or class) from library."""
    try:
        modules = __import__(_from, fromlist=[_import])
        module = getattr(modules, _import)
        return module
    except ImportError as e:
        raise ImportError(f"Could not import module `{_import}`") from e
    except AttributeError as e:
        raise AttributeError(f"Module `{_from}` does not have class `{_import}`") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while importing: {e}") from e


def import_dependencies(dependencies: list[dict]) -> dict:
    """Import multiple dependencies from different libraries, with optional aliases.

    Args:
        dependencies:
            A list of dictionaries, each with keys:
                * 'from' (library name)
                * 'import' (module or function name, or '*' for whole library)
                * 'as' (optional alias for the module/library)

    Returns:
        A dict with the module names (or aliases) as keys and the imported modules
        as values.
    """
    for dependency in dependencies:
        lib_name = dependency["from"]
        module_name = dependency.get("import", "*")
        # Use alias if provided, otherwise use module_name
        alias = dependency.get("as", module_name)
        if module_name == "*":
            imported_module = __import__(lib_name)
            globals()[alias] = imported_module
        else:
            imported_module = import_module_from_lib(module_name, lib_name)
            globals()[alias] = imported_module
    return


def autoload_package(package: Union[str, ModuleType]):
    """Imports all modules from a package, just once.

    Args:
        package:
            Name of the package or module already imported.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)

    if package.__name__ in _loaded_autoloads:
        return

    package_path = pathlib.Path(package.__file__).parent

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{package.__name__}.{module_info.name}")

    _loaded_autoloads.add(package.__name__)


class AutoloadRegistry(dict[str, Any]):
    """Dictionary-like registry that loads its provider package on first read."""

    def __init__(self, provider_package: str):
        super().__init__()
        self._provider_package = provider_package

    def _ensure_loaded(self) -> None:
        autoload_package(self._provider_package)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().get(key, default)

    def items(self):
        self._ensure_loaded()
        return super().items()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()
