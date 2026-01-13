from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import polars as pl


class Split(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class DatasetItem:
    """A single question-answer pair."""

    question: str
    answer: str


class BaseDataset(ABC):
    """Base class for all datasets."""

    def __init__(self, split: Split | str):
        if isinstance(split, str):
            split = Split(split)
        self.split = split
        self._data: pl.DataFrame | None = None

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Load the dataset and return as a polars DataFrame."""
        pass

    @property
    def data(self) -> pl.DataFrame:
        """Lazy load and cache the dataset."""
        if self._data is None:
            self._data = self.load()
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.data.row(idx, named=True)
        return DatasetItem(
            question=row[self.question_column],
            answer=row[self.answer_column],
        )

    def __iter__(self) -> Iterator[DatasetItem]:
        for i in range(len(self)):
            yield self[i]

    @property
    @abstractmethod
    def question_column(self) -> str:
        """Column name for questions."""
        pass

    @property
    @abstractmethod
    def answer_column(self) -> str:
        """Column name for answers."""
        pass
