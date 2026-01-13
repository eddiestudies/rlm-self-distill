import json

import polars as pl

from self_distill.datasets import DATA, load_dataset
from self_distill.datasets.base import DatasetItem, Split
from self_distill.datasets.pii import (
    DEFAULT_DEV_RATIO,
    DEFAULT_SEED,
    PIIDetectionDataset,
    PIIEntity,
    PIIMaskingDataset,
    PIITask,
    PII_TASKS,
    _create_pii_dataframe,
)


class TestPIIEntity:
    def test_entity_creation(self):
        entity = PIIEntity(text="john@example.com", pii_type="EMAIL", start=0, end=16)
        assert entity.text == "john@example.com"
        assert entity.pii_type == "EMAIL"
        assert entity.start == 0
        assert entity.end == 16


class TestPIITask:
    def test_task_creation(self):
        task = PIITask(
            text="Contact me at john@example.com",
            entities=[PIIEntity("john@example.com", "EMAIL", 14, 30)],
        )
        assert task.text == "Contact me at john@example.com"
        assert len(task.entities) == 1

    def test_has_pii_true(self):
        task = PIITask(
            text="Email: test@test.com",
            entities=[PIIEntity("test@test.com", "EMAIL", 7, 20)],
        )
        assert task.has_pii is True

    def test_has_pii_false(self):
        task = PIITask(text="No PII here.", entities=[])
        assert task.has_pii is False

    def test_get_masked_text_single_entity(self):
        task = PIITask(
            text="Email: test@test.com",
            entities=[PIIEntity("test@test.com", "EMAIL", 7, 20)],
        )
        assert task.get_masked_text() == "Email: [EMAIL]"

    def test_get_masked_text_multiple_entities(self):
        task = PIITask(
            text="Call 555-1234 or email test@test.com",
            entities=[
                PIIEntity("555-1234", "PHONE", 5, 13),
                PIIEntity("test@test.com", "EMAIL", 23, 36),
            ],
        )
        assert task.get_masked_text() == "Call [PHONE] or email [EMAIL]"

    def test_get_masked_text_no_entities(self):
        task = PIITask(text="No PII here.", entities=[])
        assert task.get_masked_text() == "No PII here."

    def test_get_entities_json(self):
        task = PIITask(
            text="Email: test@test.com",
            entities=[PIIEntity("test@test.com", "EMAIL", 7, 20)],
        )
        result = json.loads(task.get_entities_json())
        assert len(result) == 1
        assert result[0]["text"] == "test@test.com"
        assert result[0]["type"] == "EMAIL"
        assert result[0]["start"] == 7
        assert result[0]["end"] == 20

    def test_get_entities_json_empty(self):
        task = PIITask(text="No PII here.", entities=[])
        result = json.loads(task.get_entities_json())
        assert result == []


class TestPIITasksData:
    def test_pii_tasks_not_empty(self):
        assert len(PII_TASKS) > 0

    def test_pii_tasks_have_variety(self):
        has_pii = sum(1 for task in PII_TASKS if task.has_pii)
        no_pii = sum(1 for task in PII_TASKS if not task.has_pii)
        assert has_pii > 0
        assert no_pii > 0

    def test_pii_tasks_have_multiple_types(self):
        all_types = set()
        for task in PII_TASKS:
            for entity in task.entities:
                all_types.add(entity.pii_type)
        # Should have at least EMAIL, PHONE, SSN
        assert "EMAIL" in all_types
        assert "PHONE" in all_types
        assert "SSN" in all_types


class TestCreatePIIDataframe:
    def test_creates_dataframe(self):
        df = _create_pii_dataframe()
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self):
        df = _create_pii_dataframe()
        assert "text" in df.columns
        assert "entities_json" in df.columns
        assert "masked_text" in df.columns
        assert "has_pii" in df.columns

    def test_row_count_matches_tasks(self):
        df = _create_pii_dataframe()
        assert len(df) == len(PII_TASKS)


class TestPIIDetectionDatasetConstants:
    def test_default_dev_ratio(self):
        assert DEFAULT_DEV_RATIO == 0.1

    def test_default_seed(self):
        assert DEFAULT_SEED == 42


class TestPIIDetectionDatasetInit:
    def test_init_with_split_enum(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        assert ds.split == Split.TRAIN

    def test_init_with_split_string(self):
        ds = PIIDetectionDataset("train")
        assert ds.split == Split.TRAIN

    def test_init_custom_dev_ratio(self):
        ds = PIIDetectionDataset(Split.DEV, dev_ratio=0.2)
        assert ds.dev_ratio == 0.2

    def test_init_custom_seed(self):
        ds = PIIDetectionDataset(Split.TRAIN, seed=123)
        assert ds.seed == 123


class TestPIIDetectionDatasetColumns:
    def test_question_column(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        assert ds.question_column == "text"

    def test_answer_column(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        assert ds.answer_column == "entities_json"


class TestPIIDetectionDatasetLoad:
    def test_load_returns_dataframe(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        data = ds.load()
        assert isinstance(data, pl.DataFrame)

    def test_load_train_split(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        data = ds.load()
        assert len(data) > 0

    def test_load_dev_split(self):
        ds = PIIDetectionDataset(Split.DEV)
        data = ds.load()
        assert len(data) > 0

    def test_load_test_split(self):
        ds = PIIDetectionDataset(Split.TEST)
        data = ds.load()
        assert len(data) > 0

    def test_splits_cover_all_data(self):
        train = PIIDetectionDataset(Split.TRAIN, seed=42)
        dev = PIIDetectionDataset(Split.DEV, seed=42)
        test = PIIDetectionDataset(Split.TEST, seed=42)

        total = len(train.data) + len(dev.data) + len(test.data)
        assert total == len(PII_TASKS)

    def test_deterministic_split_with_same_seed(self):
        ds1 = PIIDetectionDataset(Split.DEV, seed=42)
        ds2 = PIIDetectionDataset(Split.DEV, seed=42)
        assert ds1.data["text"].to_list() == ds2.data["text"].to_list()


class TestPIIDetectionDatasetIteration:
    def test_iteration(self):
        ds = PIIDetectionDataset(Split.TRAIN)
        items = list(ds)
        assert len(items) > 0
        assert all(isinstance(item, DatasetItem) for item in items)

    def test_indexing(self):
        ds = PIIDetectionDataset(Split.TEST)
        item = ds[0]
        assert isinstance(item, DatasetItem)
        assert isinstance(item.question, str)
        assert isinstance(item.answer, str)

    def test_answer_is_json(self):
        ds = PIIDetectionDataset(Split.TEST)
        item = ds[0]
        # Should be valid JSON
        parsed = json.loads(item.answer)
        assert isinstance(parsed, list)


class TestPIIMaskingDatasetInit:
    def test_init_with_split_enum(self):
        ds = PIIMaskingDataset(Split.TRAIN)
        assert ds.split == Split.TRAIN

    def test_init_with_split_string(self):
        ds = PIIMaskingDataset("train")
        assert ds.split == Split.TRAIN

    def test_init_custom_dev_ratio(self):
        ds = PIIMaskingDataset(Split.DEV, dev_ratio=0.2)
        assert ds.dev_ratio == 0.2


class TestPIIMaskingDatasetColumns:
    def test_question_column(self):
        ds = PIIMaskingDataset(Split.TRAIN)
        assert ds.question_column == "text"

    def test_answer_column(self):
        ds = PIIMaskingDataset(Split.TRAIN)
        assert ds.answer_column == "masked_text"


class TestPIIMaskingDatasetLoad:
    def test_load_returns_dataframe(self):
        ds = PIIMaskingDataset(Split.TRAIN)
        data = ds.load()
        assert isinstance(data, pl.DataFrame)

    def test_load_all_splits(self):
        for split in [Split.TRAIN, Split.DEV, Split.TEST]:
            ds = PIIMaskingDataset(split)
            data = ds.load()
            assert len(data) > 0


class TestPIIMaskingDatasetIteration:
    def test_iteration(self):
        ds = PIIMaskingDataset(Split.TRAIN)
        items = list(ds)
        assert len(items) > 0

    def test_answer_contains_masks_or_original(self):
        ds = PIIMaskingDataset(Split.TEST)
        for item in ds:
            # Answer should either contain masked text or be same as question
            assert isinstance(item.answer, str)


class TestLoadDatasetFunctionPII:
    def test_load_pii_detection(self):
        ds = load_dataset(DATA.PII_DETECTION, "train")
        assert isinstance(ds, PIIDetectionDataset)
        assert ds.split == Split.TRAIN

    def test_load_pii_masking(self):
        ds = load_dataset(DATA.PII_MASKING, "train")
        assert isinstance(ds, PIIMaskingDataset)
        assert ds.split == Split.TRAIN

    def test_load_with_kwargs(self):
        ds = load_dataset(DATA.PII_DETECTION, "dev", seed=123)
        assert ds.seed == 123
