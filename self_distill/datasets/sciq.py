"""
SciQ Dataset - Science Question Answering.

Contains 13,679 crowdsourced science exam questions about Physics, Chemistry,
and Biology with multiple-choice answers and supporting explanations.

Perfect for tool use with physics/math ontologies because the 'support' field
contains concepts, formulas, and explanations that tools can leverage.
"""

from dataclasses import dataclass

import polars as pl

from self_distill.datasets.base import BaseDataset, DatasetItem, Split

# HuggingFace parquet URLs for SciQ
SCIQ_BASE_URL = "https://huggingface.co/api/datasets/allenai/sciq/parquet/default"
SCIQ_SPLITS = {
    "train": f"{SCIQ_BASE_URL}/train/0.parquet",
    "validation": f"{SCIQ_BASE_URL}/validation/0.parquet",
    "test": f"{SCIQ_BASE_URL}/test/0.parquet",
}

# Default dev split ratio (from train set, if not using validation)
DEFAULT_DEV_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class SciQItem(DatasetItem):
    """
    Extended DatasetItem for SciQ with additional fields.

    Attributes:
        question: The science question
        answer: The correct answer
        support: Supporting explanation/evidence (contains physics concepts)
        distractors: List of incorrect answer options
    """

    support: str = ""
    distractors: list[str] | None = None


class SciQDataset(BaseDataset):
    """
    SciQ (Science Question Answering) dataset.

    Contains 13,679 crowdsourced science exam questions covering:
    - Physics (forces, energy, waves, thermodynamics, etc.)
    - Chemistry
    - Biology

    Each question includes:
    - Multiple choice format (1 correct + 3 distractors)
    - Supporting evidence paragraph with concepts/formulas

    The 'support' field is particularly useful for ontology-based tools
    as it contains the physics/science concepts needed to answer.

    Splits:
    - train: 11,679 questions
    - validation: 1,000 questions (mapped to DEV)
    - test: 1,000 questions
    """

    def __init__(
        self,
        split: Split | str,
        include_support: bool = True,
        include_distractors: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        """
        Initialize SciQ dataset.

        Args:
            split: Dataset split ("train", "dev", or "test")
            include_support: Include supporting explanation in answer
            include_distractors: Include wrong answers in the data
            seed: Random seed for any shuffling
        """
        super().__init__(split)
        self.include_support = include_support
        self.include_distractors = include_distractors
        self.seed = seed

    @property
    def question_column(self) -> str:
        return "question"

    @property
    def answer_column(self) -> str:
        return "correct_answer"

    def load(self) -> pl.DataFrame:
        """Load SciQ data for the specified split."""
        if self.split == Split.TEST:
            return pl.read_parquet(SCIQ_SPLITS["test"])
        elif self.split == Split.DEV:
            # Use the validation split as dev
            return pl.read_parquet(SCIQ_SPLITS["validation"])
        else:  # TRAIN
            return pl.read_parquet(SCIQ_SPLITS["train"])

    def __getitem__(self, idx: int) -> SciQItem:
        """Get a single item with extended SciQ fields."""
        row = self.data.row(idx, named=True)

        distractors = None
        if self.include_distractors:
            distractors = [
                row.get("distractor1", ""),
                row.get("distractor2", ""),
                row.get("distractor3", ""),
            ]

        return SciQItem(
            question=row[self.question_column],
            answer=row[self.answer_column],
            support=row.get("support", "") if self.include_support else "",
            distractors=distractors,
        )

    def get_formatted_question(self, idx: int, include_choices: bool = True) -> str:
        """
        Get a formatted question with answer choices.

        Args:
            idx: Index of the question
            include_choices: Whether to include multiple choice options

        Returns:
            Formatted question string
        """
        row = self.data.row(idx, named=True)
        question = row[self.question_column]

        if not include_choices:
            return question

        # Shuffle choices deterministically
        import random

        rng = random.Random(self.seed + idx)

        choices = [
            row[self.answer_column],
            row.get("distractor1", ""),
            row.get("distractor2", ""),
            row.get("distractor3", ""),
        ]
        rng.shuffle(choices)

        formatted = f"{question}\n\n"
        for i, choice in enumerate(choices):
            label = chr(ord("A") + i)
            formatted += f"{label}. {choice}\n"

        return formatted.strip()

    def get_with_support(self, idx: int) -> tuple[str, str, str]:
        """
        Get question, answer, and support text.

        Useful for ontology-based tool use where the support
        contains physics concepts and formulas.

        Returns:
            Tuple of (question, correct_answer, support_text)
        """
        row = self.data.row(idx, named=True)
        return (
            row[self.question_column],
            row[self.answer_column],
            row.get("support", ""),
        )


class SciQPhysicsDataset(SciQDataset):
    """
    Filtered SciQ dataset containing only physics-related questions.

    Filters questions based on physics keywords in the question or support text.
    """

    PHYSICS_KEYWORDS = [
        # Mechanics
        "force", "velocity", "acceleration", "momentum", "mass", "weight",
        "friction", "gravity", "newton", "kinetic", "potential", "energy",
        "work", "power", "torque", "angular", "rotation", "oscillation",
        # Waves & Optics
        "wave", "frequency", "wavelength", "amplitude", "sound", "light",
        "reflection", "refraction", "diffraction", "interference", "optic",
        "electromagnetic", "spectrum", "photon",
        # Thermodynamics
        "temperature", "heat", "thermal", "entropy", "thermodynamic",
        "conduction", "convection", "radiation", "pressure", "volume",
        # Electricity & Magnetism
        "electric", "magnetic", "current", "voltage", "resistance", "circuit",
        "charge", "electron", "proton", "neutron", "atom", "ion",
        "capacitor", "inductor", "field",
        # Modern Physics
        "quantum", "relativity", "nuclear", "radioactive", "decay",
        "fusion", "fission",
        # General
        "physics", "physical", "joule", "watt", "newton", "pascal",
        "hertz", "coulomb", "ohm", "farad", "tesla", "weber",
    ]

    def __init__(
        self,
        split: Split | str,
        include_support: bool = True,
        include_distractors: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(
            split=split,
            include_support=include_support,
            include_distractors=include_distractors,
            seed=seed,
        )

    def load(self) -> pl.DataFrame:
        """Load and filter for physics questions only."""
        df = super().load()

        # Build regex pattern for physics keywords (case insensitive)
        pattern = "|".join(self.PHYSICS_KEYWORDS)

        # Filter rows where question or support contains physics keywords
        physics_df = df.filter(
            pl.col("question").str.to_lowercase().str.contains(pattern)
            | pl.col("support").str.to_lowercase().str.contains(pattern)
        )

        return physics_df
