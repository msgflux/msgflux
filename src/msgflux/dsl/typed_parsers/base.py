from abc import ABC, abstractmethod


class BaseTypedParser(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "typed_parser_type"):
            raise TypeError(
                f"{cls.__name__} must define class ""attribute `typed_parser_type`"
            )
        if not hasattr(cls, "template"):
            raise TypeError(f"{cls.__name__} must define class attribute `template`")

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @abstractmethod
    def repr_from_dict(self):
        raise NotImplementedError

    @abstractmethod
    def schema_from_response_format(self):
        raise NotImplementedError
