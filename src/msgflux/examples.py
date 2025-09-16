from typing import Any, List, Union, Mapping, Optional
from dataclasses import dataclass, field
from msgflux.utils.xml import apply_xml_tags


@dataclass
class Example:
    """Represents a single example for a LM.
    
    Args:
        inputs:
            Task inputs.
        labels:
            Expected model output.
        title:
            Example title.
        reasoning:
            Intermediate thinking/logic before generating output.
        topic:
            Example topic/category.
    """
    inputs: Union[str, Mapping[str, Any]]
    labels: Union[str, Mapping[str, Any]]
    reasoning: Optional[str] = None
    title: Optional[str] = None
    topic: Optional[str] = None
    _inputs_needs_transform: bool = field(init=False)
    _labels_needs_transform: bool = field(init=False)
    
    def __post_init__(self):
        self._inputs_needs_transform = isinstance(self.inputs, dict)
        self._labels_needs_transform = isinstance(self.labels, dict)
    
    def transform(
        self, 
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None
    ) -> "Example":
        """
        Retorna uma nova instância do Example com inputs/outputs transformados para string.
        
        Args:
            input_transformer: Função para transformar input dict em string
            output_transformer: Função para transformar output dict em string
            
        Returns:
            Nova instância de Example com valores em string
        """
        new_input = self.inputs
        new_output = self.labels
        
        if self._inputs_needs_transform and input_transformer:
            new_input = input_transformer(self.inputs)
            
        if self._labels_needs_transform and output_transformer:
            new_output = output_transformer(self.labels)
        
        return Example(
            inputs=new_input,
            labels=new_output,
            title=self.title,
            reasoning=self.reasoning,
            topic=self.topic
        )
    
    def to_xml(self, example_id: int) -> str:
        """Convert the example to XML format."""
        attributes = [f"id={example_id}"]
        
        if self.title:
            attributes.append(f'title="{self.title}"')
        if self.topic:
            attributes.append(f'topic="{self.topic}"')
            
        attr_str = " ".join(attributes)
        
        result = [f"<example {attr_str}>"]
        result.append(apply_xml_tags("input", str(self.inputs)))        
        if self.reasoning:
            result.append(apply_xml_tags("reasoning", self.reasoning))
        result.append(apply_xml_tags("output", str(self.labels)))
        result.append("</example>")
        
        return "\n".join(result)


class ExampleCollection:
    def __init__(
        self, examples: Optional[List[Union[Example, Mapping[str, Any]]]] = None
    ):
        self.examples = []
        if examples:
            for example in examples:
                if isinstance(example, Example):
                    self.examples.append(example)
                else:
                    self.add(**example)

    def _add(self, example: Example):
        self.examples.append(example)
    
    def add(
        self,
        inputs: Union[str, Mapping[str, Any]],
        labels: Union[str, Mapping[str, Any]],
        title: str = None,
        reasoning: str = None,
        topic: str = None
    ):
        """Add an example to the collection."""
        example = Example(
            inputs=inputs,
            labels=labels,
            title=title,
            reasoning=reasoning,
            topic=topic,
        )
        self._add(example)

    def get_examples(self) -> List[Example]:
        """Get all examples in the collection."""
        return self.examples

    def filter_by_topic(self, topic: str) -> List[Example]:
        """Filter examples by topic."""
        return [ex for ex in self.examples if ex.topic == topic]
    
    def get_examples_needing_transform(self) -> List[Example]:
        """Returns examples that need transformation."""
        return [
            ex for ex in self.examples 
            if ex._inputs_needs_transform or ex._labels_needs_transform
        ]
    
    def transform(
        self, 
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None
    ) -> "ExampleCollection":
        """Transforms all examples to strings."""
        transformed = [ex.transform(input_transformer, output_transformer) 
                      for ex in self.examples]
        return ExampleCollection(transformed)
    
    def get_formatted(
        self,
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None
    ) -> str:
        """ Converts the entire collection to XML format.

        Returns:
            Formatted examples using xml.
        """
        if not self.examples:
            return None
        
        collection = self.transform(input_transformer, output_transformer)

        xml_parts = []
        for i, example in enumerate(collection.get_examples(), start=1):
            xml_parts.append(example.to_xml(i))
        
        return "\n\n".join(xml_parts)
