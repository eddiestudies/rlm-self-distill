import polars as pl
import pytest

from self_distill.datasets.base import BaseDataset, DatasetItem, Split


class TestSplit:
    def test_split_values(self):
        assert Split.TRAIN.value == "train"
        assert Split.DEV.value == "dev"
        assert Split.TEST.value == "test"

    def test_split_from_string(self):
        assert Split("train") == Split.TRAIN
        assert Split("dev") == Split.DEV
        assert Split("test") == Split.TEST

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError):
            Split("invalid")


class TestDatasetItem:
    def test_dataset_item_creation(self):
        item = DatasetItem(question="What is 2+2?", answer="4")
        assert item.question == "What is 2+2?"
        assert item.answer == "4"

    def test_dataset_item_equality(self):
        item1 = DatasetItem(question="Q", answer="A")
        item2 = DatasetItem(question="Q", answer="A")
        assert item1 == item2


class ConcreteDataset(BaseDataset):
    """Concrete implementation for testing BaseDataset."""

    def __init__(self, split: Split | str, data: pl.DataFrame):
        super().__init__(split)
        self._test_data = data

    @property
    def question_column(self) -> str:
        return "question"

    @property
    def answer_column(self) -> str:
        return "answer"

    def load(self) -> pl.DataFrame:
        return self._test_data


class TestBaseDataset:
    @pytest.fixture
    def sample_data(self):
        return pl.DataFrame(
            {
                "question": ["Q1", "Q2", "Q3"],
                "answer": ["A1", "A2", "A3"],
            }
        )

    @pytest.fixture
    def dataset(self, sample_data):
        return ConcreteDataset(Split.TRAIN, sample_data)

    def test_init_with_split_enum(self, sample_data):
        ds = ConcreteDataset(Split.TRAIN, sample_data)
        assert ds.split == Split.TRAIN

    def test_init_with_split_string(self, sample_data):
        ds = ConcreteDataset("train", sample_data)
        assert ds.split == Split.TRAIN

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem(self, dataset):
        item = dataset[0]
        assert isinstance(item, DatasetItem)
        assert item.question == "Q1"
        assert item.answer == "A1"

    def test_getitem_last(self, dataset):
        item = dataset[2]
        assert item.question == "Q3"
        assert item.answer == "A3"

    def test_iter(self, dataset):
        items = list(dataset)
        assert len(items) == 3
        assert all(isinstance(item, DatasetItem) for item in items)
        assert items[0].question == "Q1"
        assert items[1].question == "Q2"
        assert items[2].question == "Q3"

    def test_data_property_lazy_loads(self, sample_data):
        ds = ConcreteDataset(Split.TRAIN, sample_data)
        assert ds._data is None
        _ = ds.data
        assert ds._data is not None

    def test_data_property_caches(self, sample_data):
        ds = ConcreteDataset(Split.TRAIN, sample_data)
        data1 = ds.data
        data2 = ds.data
        assert data1 is data2
