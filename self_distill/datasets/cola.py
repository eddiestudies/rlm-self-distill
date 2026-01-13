import polars as pl

from self_distill.datasets.base import BaseDataset, DatasetItem, Split

COLA_URL = "hf://datasets/shivkumarganesh/CoLA/CoLA.csv"

# Default dev/test split ratios (from full dataset)
DEFAULT_DEV_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42


class CoLAItem(DatasetItem):
    """A CoLA dataset item with grammaticality label and rule ID."""

    rule_id: str | None

    def __init__(self, question: str, answer: str, rule_id: str | None = None):
        self.question = question
        self.answer = answer
        self.rule_id = rule_id


class CoLADataset(BaseDataset):
    """
    CoLA (Corpus of Linguistic Acceptability) dataset.

    Contains sentences labeled for grammatical acceptability.
    - text: The sentence to evaluate
    - label: 1 for grammatically acceptable, 0 for unacceptable
    - id: The linguistic rule category

    Useful for training models on grammatical rules and rule-based systems.
    """

    def __init__(
        self,
        split: Split | str,
        dev_ratio: float = DEFAULT_DEV_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SEED,
        include_rule_id: bool = False,
    ):
        super().__init__(split)
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.include_rule_id = include_rule_id

    @property
    def question_column(self) -> str:
        return "text"

    @property
    def answer_column(self) -> str:
        return "label"

    @property
    def rule_id_column(self) -> str:
        return "id"

    def __getitem__(self, idx: int) -> CoLAItem:
        row = self.data.row(idx, named=True)
        return CoLAItem(
            question=row[self.question_column],
            answer=str(row[self.answer_column]),
            rule_id=row[self.rule_id_column] if self.include_rule_id else None,
        )

    def load(self) -> pl.DataFrame:
        """Load CoLA data for the specified split."""
        full_data = pl.read_csv(COLA_URL)

        # Add row index for splitting
        full_data = full_data.with_row_index("_idx")

        n_total = len(full_data)
        n_dev = int(n_total * self.dev_ratio)
        n_test = int(n_total * self.test_ratio)

        # Shuffle deterministically
        shuffled = full_data.sample(fraction=1.0, seed=self.seed, shuffle=True)

        if self.split == Split.TEST:
            result = shuffled.head(n_test)
        elif self.split == Split.DEV:
            result = shuffled.slice(n_test, n_dev)
        else:  # TRAIN
            result = shuffled.tail(n_total - n_test - n_dev)

        return result.drop("_idx")

    def get_rule_ids(self) -> list[str]:
        """Get all unique rule IDs in the current split."""
        return self.data[self.rule_id_column].unique().to_list()

    def filter_by_rule(self, rule_id: str) -> pl.DataFrame:
        """Filter dataset to only include items with a specific rule ID."""
        return self.data.filter(pl.col(self.rule_id_column) == rule_id)
