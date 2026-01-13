from unittest.mock import patch

import polars as pl
import pytest

from self_distill.datasets import DATA, load_dataset
from self_distill.datasets.base import DatasetItem, Split
from self_distill.datasets.gsm8k import (
    DEFAULT_DEV_RATIO,
    DEFAULT_SEED,
    GSM8K_BASE_URL,
    GSM8K_SPLITS,
    GSM8KDataset,
)


@pytest.fixture
def mock_train_data():
    return pl.DataFrame(
        {
            "question": [f"Question {i}" for i in range(100)],
            "answer": [f"Answer {i}" for i in range(100)],
        }
    )


@pytest.fixture
def mock_test_data():
    return pl.DataFrame(
        {
            "question": [f"Test Question {i}" for i in range(20)],
            "answer": [f"Test Answer {i}" for i in range(20)],
        }
    )


class TestGSM8KDatasetConstants:
    def test_splits_defined(self):
        assert "train" in GSM8K_SPLITS
        assert "test" in GSM8K_SPLITS

    def test_base_url(self):
        assert GSM8K_BASE_URL == "hf://datasets/openai/gsm8k/"

    def test_default_dev_ratio(self):
        assert DEFAULT_DEV_RATIO == 0.1

    def test_default_seed(self):
        assert DEFAULT_SEED == 42


class TestGSM8KDatasetInit:
    def test_init_with_split_enum(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset(Split.TRAIN)
            assert ds.split == Split.TRAIN

    def test_init_with_split_string(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset("train")
            assert ds.split == Split.TRAIN

    def test_init_custom_dev_ratio(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset(Split.DEV, dev_ratio=0.2)
            assert ds.dev_ratio == 0.2

    def test_init_custom_seed(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset(Split.TRAIN, seed=123)
            assert ds.seed == 123


class TestGSM8KDatasetColumns:
    def test_question_column(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset(Split.TRAIN)
            assert ds.question_column == "question"

    def test_answer_column(self):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet"):
            ds = GSM8KDataset(Split.TRAIN)
            assert ds.answer_column == "answer"


class TestGSM8KDatasetLoad:
    def test_load_test_split(self, mock_test_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_test_data
            ds = GSM8KDataset(Split.TEST)
            data = ds.load()

            mock_read.assert_called_once_with(GSM8K_BASE_URL + GSM8K_SPLITS["test"])
            assert len(data) == 20

    def test_load_train_split(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data
            ds = GSM8KDataset(Split.TRAIN, dev_ratio=0.1)
            data = ds.load()

            mock_read.assert_called_once_with(GSM8K_BASE_URL + GSM8K_SPLITS["train"])
            assert len(data) == 90  # 100 - 10% dev

    def test_load_dev_split(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data
            ds = GSM8KDataset(Split.DEV, dev_ratio=0.1)
            data = ds.load()

            mock_read.assert_called_once_with(GSM8K_BASE_URL + GSM8K_SPLITS["train"])
            assert len(data) == 10  # 10% of 100

    def test_train_and_dev_are_disjoint(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data

            train_ds = GSM8KDataset(Split.TRAIN, dev_ratio=0.1, seed=42)
            dev_ds = GSM8KDataset(Split.DEV, dev_ratio=0.1, seed=42)

            train_questions = set(train_ds.data["question"].to_list())
            dev_questions = set(dev_ds.data["question"].to_list())

            assert len(train_questions & dev_questions) == 0
            assert len(train_questions) + len(dev_questions) == 100

    def test_deterministic_split_with_same_seed(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data

            ds1 = GSM8KDataset(Split.DEV, seed=42)
            ds2 = GSM8KDataset(Split.DEV, seed=42)

            assert ds1.data["question"].to_list() == ds2.data["question"].to_list()

    def test_different_split_with_different_seed(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data

            ds1 = GSM8KDataset(Split.DEV, seed=42)
            ds2 = GSM8KDataset(Split.DEV, seed=123)

            assert ds1.data["question"].to_list() != ds2.data["question"].to_list()


class TestGSM8KDatasetIteration:
    def test_iteration(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data
            ds = GSM8KDataset(Split.TRAIN, dev_ratio=0.0)  # No dev split

            items = list(ds)
            assert len(items) == 100
            assert all(isinstance(item, DatasetItem) for item in items)

    def test_indexing(self, mock_test_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_test_data
            ds = GSM8KDataset(Split.TEST)

            item = ds[0]
            assert isinstance(item, DatasetItem)
            assert "Test Question" in item.question
            assert "Test Answer" in item.answer


class TestLoadDatasetFunction:
    def test_load_gsm8k_train(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data
            ds = load_dataset(DATA.GSM8K, "train")

            assert isinstance(ds, GSM8KDataset)
            assert ds.split == Split.TRAIN

    def test_load_gsm8k_with_kwargs(self, mock_train_data):
        with patch("self_distill.datasets.gsm8k.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_train_data
            ds = load_dataset(DATA.GSM8K, "dev", dev_ratio=0.2, seed=123)

            assert ds.dev_ratio == 0.2
            assert ds.seed == 123

    def test_load_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("invalid_dataset", "train")
