from unittest.mock import patch

import polars as pl
import pytest

from self_distill.datasets import DATA, load_dataset
from self_distill.datasets.base import DatasetItem, Split
from self_distill.datasets.sciq import (
    DEFAULT_DEV_RATIO,
    DEFAULT_SEED,
    SCIQ_BASE_URL,
    SCIQ_SPLITS,
    SciQDataset,
    SciQItem,
    SciQPhysicsDataset,
)


@pytest.fixture
def mock_sciq_data():
    return pl.DataFrame(
        {
            "question": [f"Science question {i}?" for i in range(100)],
            "correct_answer": [f"Answer {i}" for i in range(100)],
            "support": [f"Supporting explanation {i}" for i in range(100)],
            "distractor1": [f"Wrong answer A{i}" for i in range(100)],
            "distractor2": [f"Wrong answer B{i}" for i in range(100)],
            "distractor3": [f"Wrong answer C{i}" for i in range(100)],
        }
    )


@pytest.fixture
def mock_physics_data():
    """Data with some physics questions and some non-physics questions."""
    questions = [
        "What is the force required to accelerate a 5kg mass?",
        "What causes friction between surfaces?",
        "What is photosynthesis?",
        "How do cells divide?",
        "What is the velocity of light?",
        "What is Newton's first law?",
        "How do plants grow?",
        "What is kinetic energy?",
        "What causes earthquakes?",
        "What is electric current?",
    ]
    supports = [
        "Force equals mass times acceleration (F=ma).",
        "Friction is caused by surface interactions.",
        "Plants convert sunlight to energy.",
        "Cells divide through mitosis.",
        "Light travels at 3x10^8 m/s in vacuum.",
        "An object at rest stays at rest unless acted upon by a force.",
        "Plants need water and sunlight.",
        "Energy of motion is called kinetic energy.",
        "Tectonic plates cause earthquakes.",
        "Flow of electrons through a conductor.",
    ]
    return pl.DataFrame(
        {
            "question": questions,
            "correct_answer": [f"Answer {i}" for i in range(10)],
            "support": supports,
            "distractor1": [f"Wrong A{i}" for i in range(10)],
            "distractor2": [f"Wrong B{i}" for i in range(10)],
            "distractor3": [f"Wrong C{i}" for i in range(10)],
        }
    )


class TestSciQItem:
    def test_item_creation(self):
        item = SciQItem(
            question="What is gravity?",
            answer="A force",
            support="Gravity pulls objects together.",
            distractors=["A color", "A sound", "A smell"],
        )
        assert item.question == "What is gravity?"
        assert item.answer == "A force"
        assert item.support == "Gravity pulls objects together."
        assert len(item.distractors) == 3

    def test_item_without_support(self):
        item = SciQItem(question="Test?", answer="Yes")
        assert item.support == ""
        assert item.distractors is None


class TestSciQDatasetConstants:
    def test_base_url(self):
        assert "huggingface" in SCIQ_BASE_URL

    def test_splits_defined(self):
        assert "train" in SCIQ_SPLITS
        assert "validation" in SCIQ_SPLITS
        assert "test" in SCIQ_SPLITS

    def test_default_dev_ratio(self):
        assert DEFAULT_DEV_RATIO == 0.1

    def test_default_seed(self):
        assert DEFAULT_SEED == 42


class TestSciQDatasetInit:
    def test_init_with_split_enum(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(Split.TRAIN)
            assert ds.split == Split.TRAIN

    def test_init_with_split_string(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset("train")
            assert ds.split == Split.TRAIN

    def test_init_include_support_default(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(Split.TRAIN)
            assert ds.include_support is True

    def test_init_include_distractors_default(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(Split.TRAIN)
            assert ds.include_distractors is False

    def test_init_custom_options(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(
                Split.TRAIN,
                include_support=False,
                include_distractors=True,
                seed=123,
            )
            assert ds.include_support is False
            assert ds.include_distractors is True
            assert ds.seed == 123


class TestSciQDatasetColumns:
    def test_question_column(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(Split.TRAIN)
            assert ds.question_column == "question"

    def test_answer_column(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQDataset(Split.TRAIN)
            assert ds.answer_column == "correct_answer"


class TestSciQDatasetLoad:
    def test_load_train_split(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)
            data = ds.load()

            mock_read.assert_called_once_with(SCIQ_SPLITS["train"])
            assert len(data) == 100

    def test_load_dev_split(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.DEV)
            data = ds.load()

            mock_read.assert_called_once_with(SCIQ_SPLITS["validation"])

    def test_load_test_split(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TEST)
            data = ds.load()

            mock_read.assert_called_once_with(SCIQ_SPLITS["test"])


class TestSciQDatasetIteration:
    def test_iteration(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)

            items = list(ds)
            assert len(items) == 100
            assert all(isinstance(item, SciQItem) for item in items)

    def test_indexing(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)

            item = ds[0]
            assert isinstance(item, SciQItem)
            assert "Science question" in item.question
            assert "Answer" in item.answer

    def test_item_with_support(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN, include_support=True)

            item = ds[0]
            assert item.support != ""

    def test_item_without_support(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN, include_support=False)

            item = ds[0]
            assert item.support == ""

    def test_item_with_distractors(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN, include_distractors=True)

            item = ds[0]
            assert item.distractors is not None
            assert len(item.distractors) == 3

    def test_item_without_distractors(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN, include_distractors=False)

            item = ds[0]
            assert item.distractors is None


class TestSciQDatasetMethods:
    def test_get_formatted_question_without_choices(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)

            formatted = ds.get_formatted_question(0, include_choices=False)
            assert "Science question 0?" in formatted
            # Should not include choice labels
            assert "A." not in formatted

    def test_get_formatted_question_with_choices(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)

            formatted = ds.get_formatted_question(0, include_choices=True)
            assert "Science question 0?" in formatted
            # Should include choice labels
            assert "A." in formatted
            assert "B." in formatted
            assert "C." in formatted
            assert "D." in formatted

    def test_get_with_support(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = SciQDataset(Split.TRAIN)

            question, answer, support = ds.get_with_support(0)
            assert "Science question 0?" in question
            assert "Answer 0" in answer
            assert "Supporting explanation 0" in support


class TestSciQPhysicsDatasetInit:
    def test_init_inherits_from_sciq(self):
        with patch("self_distill.datasets.sciq.pl.read_parquet"):
            ds = SciQPhysicsDataset(Split.TRAIN)
            assert isinstance(ds, SciQDataset)

    def test_physics_keywords_defined(self):
        assert len(SciQPhysicsDataset.PHYSICS_KEYWORDS) > 0
        assert "force" in SciQPhysicsDataset.PHYSICS_KEYWORDS
        assert "energy" in SciQPhysicsDataset.PHYSICS_KEYWORDS
        assert "velocity" in SciQPhysicsDataset.PHYSICS_KEYWORDS


class TestSciQPhysicsDatasetLoad:
    def test_filters_physics_questions(self, mock_physics_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_physics_data
            ds = SciQPhysicsDataset(Split.TRAIN)
            data = ds.load()

            # Should filter out non-physics questions
            assert len(data) < len(mock_physics_data)
            assert len(data) > 0

    def test_keeps_physics_keywords_in_question(self, mock_physics_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_physics_data
            ds = SciQPhysicsDataset(Split.TRAIN)
            data = ds.load()

            # All kept questions should have physics keywords
            for row in data.iter_rows(named=True):
                question = row["question"].lower()
                support = row["support"].lower()
                combined = question + " " + support
                # At least one physics keyword should be present
                has_keyword = any(
                    kw in combined for kw in SciQPhysicsDataset.PHYSICS_KEYWORDS
                )
                assert has_keyword, f"No physics keyword found in: {question}"

    def test_iteration_returns_sciq_items(self, mock_physics_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_physics_data
            ds = SciQPhysicsDataset(Split.TRAIN)

            items = list(ds)
            assert len(items) > 0
            assert all(isinstance(item, SciQItem) for item in items)


class TestLoadDatasetFunctionSciQ:
    def test_load_sciq_train(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = load_dataset(DATA.SCIQ, "train")

            assert isinstance(ds, SciQDataset)
            assert ds.split == Split.TRAIN

    def test_load_sciq_with_kwargs(self, mock_sciq_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_sciq_data
            ds = load_dataset(
                DATA.SCIQ, "dev", include_support=False, include_distractors=True
            )

            assert ds.include_support is False
            assert ds.include_distractors is True

    def test_load_sciq_physics(self, mock_physics_data):
        with patch("self_distill.datasets.sciq.pl.read_parquet") as mock_read:
            mock_read.return_value = mock_physics_data
            ds = load_dataset(DATA.SCIQ_PHYSICS, "train")

            assert isinstance(ds, SciQPhysicsDataset)
            assert ds.split == Split.TRAIN
