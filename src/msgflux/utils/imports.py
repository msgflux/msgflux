import importlib
import pathlib
import pkgutil
from types import ModuleType
from typing import Union

_loaded_autoloads = set()


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
