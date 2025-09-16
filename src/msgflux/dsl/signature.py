from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from jinja2 import Template

from msgflux.dsl.typed_parsers.base import BaseTypedParser
from msgflux.generation.templates import EXPECTED_OUTPUTS_TEMPLATE
from msgflux.utils.xml import apply_xml_tags


SIGNATURE_DEFAULT_SYSTEM_MESSAGE = """
Your goal is to provide accurate, helpful, and well-reasoned responses.
Carefully analyze the user's request to fully understand the objective. Address all parts of the query.
Think step-by-step to formulate your answer. Where appropriate, briefly explain your reasoning process.
Structure your response clearly and concisely using dicts, lists, or other formatting.
Strive for factual accuracy.
Be helpful and informative, focusing on directly answering the user's prompt.
""".strip() # noqa: E501


@dataclass
class SignatureExamples:
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


@dataclass
class FieldInfo:
    name: str
    dtype: str
    desc: Optional[str] = None


class Image:
    """Represents an image input or output in a model signature.
    
    Can hold metadata in the future (e.g., format, resolution).
    """
    pass


class Audio:
    """Represents an audio input or output in a model signature.
    
    Can hold metadata in the future (e.g., sample rate, channels).
    """
    pass


class Field:
    def __init__(self, desc: Optional[str] = None):
        if isinstance(desc, str) or desc is None:
            self.desc = desc
        else:
            raise TypeError("`desc` must be a string or None")


class InputField(Field):
    """Represents an input field in a model signature.

    Args:
        desc:
            A description of the input field. Defaults to an empty string.
    """


class OutputField(Field):
    """Represents an output field in a model signature.

    Args:
        desc:
            A description of the output field. Defaults to an empty string.
    """


class _SignatureMeta(type):
    """Metaclass to process input and output fields in a model signature.

    This metaclass collects all `InputField` and `OutputField` instances
    defined in a class and stores them in `_inputs` and `_outputs`
    dictionaries, respectively.
    """

    def __new__(cls, name, bases, dct):
        inputs = {}
        outputs = {}

        for key, value in dct.items():
            if isinstance(value, InputField):
                inputs[key] = value
            elif isinstance(value, OutputField):
                outputs[key] = value

        dct["_inputs"] = inputs
        dct["_outputs"] = outputs
        return super().__new__(cls, name, bases, dct)


class Signature(metaclass=_SignatureMeta):
    """Base class for model signatures.

    This class provides functionality to define and inspect input and output fields
    of a model. It uses the `_SignatureMeta` metaclass to automatically collect
    `InputField` and `OutputField` instances.

    !!! example:
        ```python
        class CheckCitationFaithfulness(Signature):
            \"\"\"Verify that the text is based on the provided context.\"\"\"

            context: str = InputField(desc="Facts here are assumed to be true")
            text: str = InputField()
            faithfulness: bool = OutputField(ex="True")
            evidence: dict[str, list[str]] = OutputField(
                desc="Supporting evidence for claims"
            )

        # Get the class docstring
        print(CheckCitationFaithfulness.get_instructions())
        # Output:
        # "Verify that the text is based on the provided context."

        # Get the signature in string format
        print(CheckCitationFaithfulness.get_str_signature())
        # Output:
        # "context: str, text: str -> faithfulness: bool, evidence: dict[str, list[str]]"

        # Get input descriptions
        print(CheckCitationFaithfulness.get_input_descriptions())
        # Output:
        #[
        #   ('context', 'str', 'Facts here are assumed to be true', None),
        #   ('text', 'str', '', None)
        #]

        # Get output descriptions
        print(CheckCitationFaithfulness.get_output_descriptions())
        # Output:
        #[
        #   ('faithfulness', 'bool', '', "True"),
        #   ('evidence', 'dict[str, list[str]]', 'Supporting evidence for claims', None)
        #]
        ```
    """

    @classmethod
    def _dtype_to_str(cls, dtype_obj: Any) -> str:
        """Converts a type object to a readable string representation.

        Args:
            dtype_obj:
                The type object to convert.

        Returns:
            A string representation of the dtype.
        """
        origin = get_origin(dtype_obj)
        if origin is not None:
            args = get_args(dtype_obj)
            if origin is Union:
                # Checks if it is an Optional (Union with None)
                if len(args) == 2 and type(None) in args:
                    # Get the type that is not None
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    return f"Optional[{cls._dtype_to_str(non_none_type)}]"
                else:
                    return " | ".join(cls._dtype_to_str(arg) for arg in args)
            else:
                arg_strs = [cls._dtype_to_str(arg) for arg in args]
                return f"{origin.__name__}[{', '.join(arg_strs)}]"
        elif hasattr(dtype_obj, "__name__"):
            return dtype_obj.__name__
        else:
            return str(dtype_obj)

    @classmethod
    def _get_inputs(cls) -> Dict[str, str]:
        """Retrieves the input fields with their names and types.

        Returns:
            A dictionary mapping input field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._dtype_to_str(type_hints[key]) for key in cls._inputs}

    @classmethod
    def _get_outputs(cls) -> Dict[str, str]:
        """Retrieves the output fields with their names and types.

        Returns:
            A dictionary mapping output field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._dtype_to_str(type_hints[key]) for key in cls._outputs}

    @classmethod
    def get_str_signature(cls) -> str:
        """Returns the signature of the parameters in string format.

        Returns:
            A string representation of the input and output fields.
        """
        inputs = [f"{key}: {dtype}" for key, dtype in cls._get_inputs().items()]
        outputs = [f"{key}: {dtype}" for key, dtype in cls._get_outputs().items()]
        return ", ".join(inputs) + " -> " + ", ".join(outputs)

    # TODO: tem bugs no parser quando usa optional

    @classmethod
    def get_inputs_info(cls) -> List[FieldInfo]:
        """Returns a list of objects containing the input field name,
        dtype, and description.
        """
        inputs = cls._get_inputs()
        return [
            FieldInfo(key, dtype, cls._inputs[key].desc)
            for key, dtype in inputs.items()
        ]

    @classmethod
    def get_outputs_info(cls) -> List[FieldInfo]:
        """Returns a list of objects containing the output field name,
        dtype, and description.
        """
        outputs = cls._get_outputs()
        return [
            FieldInfo(key, dtype, cls._outputs[key].desc)
            for key, dtype in outputs.items()
        ]

    @classmethod
    def get_instructions(cls) -> Optional[str]:
        """Returns the class docstring.

        Returns:
            The docstring of the class, or `None` if no docstring is present.
        """
        return cls.__doc__.strip() if cls.__doc__ else None

    @classmethod
    def get_output_descriptions(cls) -> Optional[Dict[str, str]]:
        outputs = cls._get_outputs()
        descriptions = {
            key: cls._outputs[key].desc
            for key in outputs.keys()
            if cls._outputs[key].desc is not None
        }
        return descriptions if descriptions else None

class SignatureFactory:

    @classmethod
    def get_examples_from_signature(
        cls, signature_cls: Type[Signature],
    ) -> Optional[SignatureExamples]:
        if hasattr(signature_cls, "__examples__"):
            return signature_cls.__examples__

    @classmethod
    def get_expected_output_from_signature(
        cls,
        inputs_info: List[FieldInfo],
        outputs_info: Optional[List[FieldInfo]] = None,
        typed_parser_cls: Optional[Type[BaseTypedParser]] = None,        
    ) -> str:
        expected_inputs = ""
        for i, input_info in enumerate(inputs_info, 1):
            part = f"{i}. `{input_info.name}` ({input_info.dtype})"
            if input_info.desc:
                part += f": {input_info.desc}"
            expected_inputs += part + "\n"

        expected_outputs = ""
        if typed_parser_cls is None:
            # Expose ONLY the outputs for JSON-based. If fused with another Struct
            # (e.g ReAct) it will be passed as 'response_format' to the model client.
            # This removes duplication.
            expected_outputs += "Your task outputs are:" + "\n\n"
            for i, output_info in enumerate(outputs_info, 1):
                part = f"{i}. `{output_info.name}` ({output_info.dtype})"
                if output_info.desc:
                    part += f": {output_info.desc}"
                expected_outputs += part + "\n"
            expected_outputs += "\nWrite an encoded JSON."
        io = {"expected_inputs": expected_inputs, "expected_outputs": expected_outputs}
        template = Template(EXPECTED_OUTPUTS_TEMPLATE)
        rendered = template.render(io)
        expected_output = rendered.strip()
        return expected_output

    @classmethod
    def get_task_template_from_signature(
        cls, inputs_info: List[FieldInfo],
    ) -> str:
        task_template = ""
        for input_info in inputs_info:
            if input_info.dtype in ["Audio", "Image"]:
                part = apply_xml_tags(input_info.dtype, input_info.name)
            else:
                part = apply_xml_tags(input_info.name, f"{{{{ {input_info.name} }}}}")
            task_template += part + "\n"
        return task_template.strip()
