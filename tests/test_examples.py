import pytest

from msgflux.core.examples import Example, ExampleCollection


@pytest.fixture
def simple_example():
    return Example(inputs="input", labels="label")


@pytest.fixture
def dict_example():
    return Example(inputs={"in_key": "in_val"}, labels={"out_key": "out_val"})


class TestExample:
    def test_post_init(self, simple_example, dict_example):
        assert not simple_example._inputs_needs_transform
        assert not simple_example._labels_needs_transform
        assert dict_example._inputs_needs_transform
        assert dict_example._labels_needs_transform

    def test_transform_no_transformer(self, simple_example, dict_example):
        assert simple_example.transform().inputs == "input"
        assert dict_example.transform().inputs == {"in_key": "in_val"}

    def test_transform_with_transformer(self, dict_example):
        input_transformer = lambda x: f"input: {x['in_key']}"
        output_transformer = lambda x: f"output: {x['out_key']}"
        transformed = dict_example.transform(input_transformer, output_transformer)
        assert transformed.inputs == "input: in_val"
        assert transformed.labels == "output: out_val"

    def test_to_xml(self, simple_example):
        xml = simple_example.to_xml(1)
        assert "<example id=1>" in xml
        assert "<input>input</input>" in xml
        assert "<output>label</output>" in xml
        assert "</example>" in xml

    def test_to_xml_with_all_fields(self):
        example = Example(
            inputs="input",
            labels="label",
            title="title",
            reasoning="reasoning",
            topic="topic",
        )
        xml = example.to_xml(1)
        assert 'title="title"' in xml
        assert 'topic="topic"' in xml
        assert "<reasoning>reasoning</reasoning>" in xml


@pytest.fixture
def simple_collection():
    return ExampleCollection(
        [
            Example(inputs="in1", labels="out1", topic="topic1"),
            Example(inputs="in2", labels="out2", topic="topic2"),
        ]
    )


@pytest.fixture
def dict_collection():
    return ExampleCollection(
        [
            {"inputs": {"in": 1}, "labels": {"out": 1}, "topic": "topic1"},
            {"inputs": {"in": 2}, "labels": {"out": 2}, "topic": "topic2"},
        ]
    )


class TestExampleCollection:
    def test_init_with_examples(self, simple_collection):
        assert len(simple_collection.get_examples()) == 2

    def test_init_with_dicts(self, dict_collection):
        assert len(dict_collection.get_examples()) == 2
        assert isinstance(dict_collection.get_examples()[0], Example)

    def test_add(self):
        collection = ExampleCollection()
        collection.add(inputs="in", labels="out")
        assert len(collection.get_examples()) == 1

    def test_filter_by_topic(self, simple_collection):
        filtered = simple_collection.filter_by_topic("topic1")
        assert len(filtered) == 1
        assert filtered[0].inputs == "in1"

    def test_get_examples_needing_transform(self, simple_collection, dict_collection):
        assert len(simple_collection.get_examples_needing_transform()) == 0
        assert len(dict_collection.get_examples_needing_transform()) == 2

    def test_transform(self, dict_collection):
        input_transformer = lambda x: str(x["in"])
        output_transformer = lambda x: str(x["out"])
        transformed = dict_collection.transform(input_transformer, output_transformer)
        assert transformed.get_examples()[0].inputs == "1"
        assert transformed.get_examples()[0].labels == "1"

    def test_get_formatted_empty(self):
        collection = ExampleCollection()
        assert collection.get_formatted() is None

    def test_get_formatted(self, simple_collection):
        formatted = simple_collection.get_formatted()
        assert "<example id=1" in formatted
        assert "<example id=2" in formatted
