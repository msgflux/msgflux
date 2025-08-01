from copy import deepcopy
from typing import Any, Optional

from typing_extensions import Self


class Parameter:
    """Parameter is a prompt component in `nn.Module` that can be optimized.

    Parameters that have a very special property when used with `Module`s
    - when they're assigned as Module attributes they are automatically
    added to the list of its parameters, and will appear e.g. in
    `Module.parameters` iterator.

    Args:
        data:
            Prompt component content.
        spec:
            Prompt component specification.
        requires_grad:
            If the parameter requires "gradient".
    """

    grad: Optional[str] = None

    def __init__(self, data: str, spec: str, *, requires_grad: Optional[bool] = True):
        self.data = data
        self.spec = spec
        self.requires_grad = requires_grad

    def requires_grad_(self, *, requires_grad: bool) -> None:
        self.requires_grad = requires_grad

    def __hash__(self):
        return hash((self.data, self.spec))

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return False
        return self.data == other.data and self.spec == other.spec

    def __str__(self):
        return self.data

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, str):
            return self.data + other
        elif isinstance(other, Parameter):
            return self.data + other.data
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return other + self.data
        else:
            return NotImplemented

    def copy_to_data(self, data: Any):
        """Copy new data to self.data."""
        self.data = deepcopy(data)

    def clone(self) -> Self:
        return deepcopy(self)

    def copy_(self, src: "Parameter"):
        """Copies the elements from src into self tensor and returns self.

        Args:
            src:
                The source parameter to copy from
        """
        if src is not None:
            self.data = deepcopy(src.data)
