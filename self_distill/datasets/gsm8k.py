import polars as pl

from self_distill.datasets.base import BaseDataset, Split

# HuggingFace parquet paths for GSM8K
GSM8K_SPLITS = {
    "train": "main/train-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}
GSM8K_BASE_URL = "hf://datasets/openai/gsm8k/"

# Default dev split ratio (from train set)
DEFAULT_DEV_RATIO = 0.1
DEFAULT_SEED = 42


class GSM8KDataset(BaseDataset):
    """
    GSM8K (Grade School Math 8K) dataset.

    Contains 8.5K grade school math word problems with detailed solutions.
    Questions and answers are stored in 'question' and 'answer' columns.
    """

    def __init__(
        self,
        split: Split | str,
        dev_ratio: float = DEFAULT_DEV_RATIO,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(split)
        self.dev_ratio = dev_ratio
        self.seed = seed

    @property
    def question_column(self) -> str:
        return "question"

    @property
    def answer_column(self) -> str:
        return "answer"

    def load(self) -> pl.DataFrame:
        """Load GSM8K data for the specified split."""
        if self.split == Split.TEST:
            return pl.read_parquet(GSM8K_BASE_URL + GSM8K_SPLITS["test"])

        # For train and dev, we need to split the training data
        full_train = pl.read_parquet(GSM8K_BASE_URL + GSM8K_SPLITS["train"])

        # Add row index for splitting
        full_train = full_train.with_row_index("_idx")

        # Shuffle deterministically and split
        n_total = len(full_train)
        n_dev = int(n_total * self.dev_ratio)

        shuffled = full_train.sample(fraction=1.0, seed=self.seed, shuffle=True)

        if self.split == Split.DEV:
            result = shuffled.head(n_dev)
        else:  # TRAIN
            result = shuffled.tail(n_total - n_dev)

        # Remove the temporary index column
        return result.drop("_idx")
