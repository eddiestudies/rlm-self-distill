from unittest.mock import patch

import polars as pl
import pytest

from self_distill.datasets import DATA, load_dataset
from self_distill.datasets.base import Split
from self_distill.datasets.cola import (
    COLA_URL,
    DEFAULT_DEV_RATIO,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    CoLADataset,
    CoLAItem,
)


@pytest.fixture
def mock_cola_data():
    return pl.DataFrame(
        {
            "text": [f"Sentence {i}" for i in range(100)],
            "label": [i % 2 for i in range(100)],  # Alternating 0/1
            "id": [f"rule_{i % 5}" for i in range(100)],  # 5 different rules
        }
    )


class TestCoLAItem:
    def test_cola_item_creation(self):
        item = CoLAItem(question="The cat sat.", answer="1", rule_id="rule_1")
        assert item.question == "The cat sat."
        assert item.answer == "1"
        assert item.rule_id == "rule_1"

    def test_cola_item_without_rule_id(self):
        item = CoLAItem(question="The cat sat.", answer="1")
        assert item.rule_id is None


class TestCoLADatasetConstants:
    def test_url_defined(self):
        assert COLA_URL == "hf://datasets/shivkumarganesh/CoLA/CoLA.csv"

    def test_default_ratios(self):
        assert DEFAULT_DEV_RATIO == 0.1
        assert DEFAULT_TEST_RATIO == 0.1

    def test_default_seed(self):
        assert DEFAULT_SEED == 42


class TestCoLADatasetInit:
    def test_init_with_split_enum(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN)
            assert ds.split == Split.TRAIN

    def test_init_with_split_string(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset("train")
            assert ds.split == Split.TRAIN

    def test_init_custom_ratios(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN, dev_ratio=0.15, test_ratio=0.15)
            assert ds.dev_ratio == 0.15
            assert ds.test_ratio == 0.15

    def test_init_include_rule_id(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN, include_rule_id=True)
            assert ds.include_rule_id is True


class TestCoLADatasetColumns:
    def test_question_column(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN)
            assert ds.question_column == "text"

    def test_answer_column(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN)
            assert ds.answer_column == "label"

    def test_rule_id_column(self):
        with patch("self_distill.datasets.cola.pl.read_csv"):
            ds = CoLADataset(Split.TRAIN)
            assert ds.rule_id_column == "id"


class TestCoLADatasetLoad:
    def test_load_test_split(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TEST, test_ratio=0.1)
            data = ds.load()

            mock_read.assert_called_once_with(COLA_URL)
            assert len(data) == 10  # 10% of 100

    def test_load_dev_split(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.DEV, dev_ratio=0.1, test_ratio=0.1)
            data = ds.load()

            assert len(data) == 10  # 10% of 100

    def test_load_train_split(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN, dev_ratio=0.1, test_ratio=0.1)
            data = ds.load()

            assert len(data) == 80  # 100 - 10% dev - 10% test

    def test_splits_are_disjoint(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data

            train_ds = CoLADataset(Split.TRAIN, dev_ratio=0.1, test_ratio=0.1, seed=42)
            dev_ds = CoLADataset(Split.DEV, dev_ratio=0.1, test_ratio=0.1, seed=42)
            test_ds = CoLADataset(Split.TEST, dev_ratio=0.1, test_ratio=0.1, seed=42)

            train_texts = set(train_ds.data["text"].to_list())
            dev_texts = set(dev_ds.data["text"].to_list())
            test_texts = set(test_ds.data["text"].to_list())

            # No overlap between any splits
            assert len(train_texts & dev_texts) == 0
            assert len(train_texts & test_texts) == 0
            assert len(dev_texts & test_texts) == 0

            # All data is accounted for
            assert len(train_texts) + len(dev_texts) + len(test_texts) == 100

    def test_deterministic_split_with_same_seed(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data

            ds1 = CoLADataset(Split.DEV, seed=42)
            ds2 = CoLADataset(Split.DEV, seed=42)

            assert ds1.data["text"].to_list() == ds2.data["text"].to_list()


class TestCoLADatasetIteration:
    def test_getitem_without_rule_id(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN, include_rule_id=False)

            item = ds[0]
            assert isinstance(item, CoLAItem)
            assert item.rule_id is None

    def test_getitem_with_rule_id(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN, include_rule_id=True)

            item = ds[0]
            assert isinstance(item, CoLAItem)
            assert item.rule_id is not None
            assert item.rule_id.startswith("rule_")

    def test_answer_is_string(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN)

            item = ds[0]
            assert isinstance(item.answer, str)
            assert item.answer in ("0", "1")


class TestCoLADatasetRuleFiltering:
    def test_get_rule_ids(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN, dev_ratio=0.0, test_ratio=0.0)

            rule_ids = ds.get_rule_ids()
            assert len(rule_ids) == 5
            assert all(r.startswith("rule_") for r in rule_ids)

    def test_filter_by_rule(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = CoLADataset(Split.TRAIN, dev_ratio=0.0, test_ratio=0.0)

            filtered = ds.filter_by_rule("rule_0")
            assert len(filtered) == 20  # 100 / 5 rules
            assert all(r == "rule_0" for r in filtered["id"].to_list())


class TestLoadDatasetFunctionCoLA:
    def test_load_cola_train(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = load_dataset(DATA.COLA, "train")

            assert isinstance(ds, CoLADataset)
            assert ds.split == Split.TRAIN

    def test_load_cola_with_kwargs(self, mock_cola_data):
        with patch("self_distill.datasets.cola.pl.read_csv") as mock_read:
            mock_read.return_value = mock_cola_data
            ds = load_dataset(DATA.COLA, "dev", include_rule_id=True, seed=123)

            assert ds.include_rule_id is True
            assert ds.seed == 123
