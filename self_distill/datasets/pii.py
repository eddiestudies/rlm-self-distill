"""
PII Detection and Masking Datasets.

Two dataset variants for PII tasks:
1. PIIDetectionDataset - Detect PII entities in text
2. PIIMaskingDataset - Mask PII in text with type labels
"""

import json
from dataclasses import dataclass, field

import polars as pl

from self_distill.datasets.base import BaseDataset, Split

DEFAULT_DEV_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class PIIEntity:
    """A single PII entity found in text."""

    text: str
    pii_type: str
    start: int
    end: int


@dataclass
class PIITask:
    """A single PII task with text and labeled entities."""

    text: str
    entities: list[PIIEntity] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return len(self.entities) > 0

    def get_masked_text(self) -> str:
        """Return text with PII replaced by type labels."""
        result = self.text
        for entity in sorted(self.entities, key=lambda e: e.start, reverse=True):
            result = result[:entity.start] + f"[{entity.pii_type}]" + result[entity.end:]
        return result

    def get_entities_json(self) -> str:
        """Return entities as JSON string for answer format."""
        entities_list = [
            {"text": e.text, "type": e.pii_type, "start": e.start, "end": e.end}
            for e in self.entities
        ]
        return json.dumps(entities_list)


# Sample PII tasks with various PII types
PII_TASKS = [
    # === EMAIL ===
    PIITask(
        "Contact me at john.doe@example.com for more information.",
        [PIIEntity("john.doe@example.com", "EMAIL", 14, 34)],
    ),
    PIITask(
        "Send your resume to hr@company.co.uk and careers@bigcorp.com",
        [
            PIIEntity("hr@company.co.uk", "EMAIL", 20, 36),
            PIIEntity("careers@bigcorp.com", "EMAIL", 41, 60),
        ],
    ),
    PIITask("The weather today is sunny and warm.", []),
    # === PHONE NUMBERS ===
    PIITask(
        "Call me at (555) 123-4567 or 555.987.6543",
        [
            PIIEntity("(555) 123-4567", "PHONE", 11, 25),
            PIIEntity("555.987.6543", "PHONE", 29, 41),
        ],
    ),
    PIITask(
        "My cell is 1-800-555-0199 and office is +1 (212) 555-1234",
        [
            PIIEntity("1-800-555-0199", "PHONE", 11, 25),
            PIIEntity("+1 (212) 555-1234", "PHONE", 41, 58),
        ],
    ),
    # === SSN ===
    PIITask(
        "My SSN is 123-45-6789 for the application.",
        [PIIEntity("123-45-6789", "SSN", 10, 21)],
    ),
    PIITask(
        "Please verify SSN: 987-65-4321 and 111-22-3333",
        [
            PIIEntity("987-65-4321", "SSN", 18, 29),
            PIIEntity("111-22-3333", "SSN", 34, 45),
        ],
    ),
    # === CREDIT CARD ===
    PIITask(
        "Payment with card 4111-1111-1111-1111 was successful.",
        [PIIEntity("4111-1111-1111-1111", "CREDIT_CARD", 18, 37)],
    ),
    PIITask(
        "Try card 5500 0000 0000 0004 or 3400-000000-00009",
        [
            PIIEntity("5500 0000 0000 0004", "CREDIT_CARD", 9, 28),
            PIIEntity("3400-000000-00009", "CREDIT_CARD", 32, 49),
        ],
    ),
    # === IP ADDRESS ===
    PIITask(
        "Server IP is 192.168.1.100 and backup is 10.0.0.1",
        [
            PIIEntity("192.168.1.100", "IP_ADDRESS", 13, 26),
            PIIEntity("10.0.0.1", "IP_ADDRESS", 41, 49),
        ],
    ),
    PIITask(
        "IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        [PIIEntity("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "IP_ADDRESS", 6, 45)],
    ),
    # === DATE OF BIRTH ===
    PIITask(
        "DOB: 01/15/1990 or January 15, 1990",
        [
            PIIEntity("01/15/1990", "DOB", 5, 15),
            PIIEntity("January 15, 1990", "DOB", 19, 35),
        ],
    ),
    # === ADDRESS ===
    PIITask(
        "Ship to 123 Main Street, Apt 4B, New York, NY 10001",
        [PIIEntity("123 Main Street, Apt 4B, New York, NY 10001", "ADDRESS", 8, 51)],
    ),
    PIITask(
        "Located at 456 Oak Ave, Suite 100, Los Angeles, CA 90210",
        [PIIEntity("456 Oak Ave, Suite 100, Los Angeles, CA 90210", "ADDRESS", 11, 56)],
    ),
    # === MIXED PII ===
    PIITask(
        "Patient John Smith (SSN: 555-12-3456) can be reached at john.smith@email.com or (555) 234-5678",
        [
            PIIEntity("John Smith", "NAME", 8, 18),
            PIIEntity("555-12-3456", "SSN", 25, 36),
            PIIEntity("john.smith@email.com", "EMAIL", 56, 76),
            PIIEntity("(555) 234-5678", "PHONE", 80, 94),
        ],
    ),
    PIITask(
        "Employee ID: EMP-12345, Email: alice.johnson@corp.net, DOB: 03/22/1985",
        [
            PIIEntity("EMP-12345", "EMPLOYEE_ID", 13, 22),
            PIIEntity("alice.johnson@corp.net", "EMAIL", 31, 53),
            PIIEntity("03/22/1985", "DOB", 60, 70),
        ],
    ),
    # === NO PII (for balance) ===
    PIITask("The quick brown fox jumps over the lazy dog.", []),
    PIITask("Please review the attached document and provide feedback.", []),
    PIITask("Meeting scheduled for next Tuesday at 3pm.", []),
    PIITask("The product costs $49.99 with free shipping.", []),
]


def _create_pii_dataframe() -> pl.DataFrame:
    """Create a polars DataFrame from PII tasks."""
    records = []
    for task in PII_TASKS:
        records.append(
            {
                "text": task.text,
                "entities_json": task.get_entities_json(),
                "masked_text": task.get_masked_text(),
                "has_pii": task.has_pii,
            }
        )
    return pl.DataFrame(records)


class PIIDetectionDataset(BaseDataset):
    """
    PII Detection dataset.

    Given text, the task is to identify all PII entities with their types and positions.
    Question: The input text
    Answer: JSON list of detected PII entities
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
        return "text"

    @property
    def answer_column(self) -> str:
        return "entities_json"

    def load(self) -> pl.DataFrame:
        """Load PII detection data for the specified split."""
        full_data = _create_pii_dataframe()

        if self.split == Split.TEST:
            # Use last 20% as test
            n_test = max(1, len(full_data) // 5)
            return full_data.tail(n_test)

        # For train and dev, use the first 80%
        n_test = max(1, len(full_data) // 5)
        train_dev = full_data.head(len(full_data) - n_test)

        # Add row index for splitting
        train_dev = train_dev.with_row_index("_idx")

        # Shuffle deterministically and split
        n_total = len(train_dev)
        n_dev = max(1, int(n_total * self.dev_ratio))

        shuffled = train_dev.sample(fraction=1.0, seed=self.seed, shuffle=True)

        if self.split == Split.DEV:
            result = shuffled.head(n_dev)
        else:  # TRAIN
            result = shuffled.tail(n_total - n_dev)

        return result.drop("_idx")


class PIIMaskingDataset(BaseDataset):
    """
    PII Masking dataset.

    Given text, the task is to return the text with all PII replaced by type labels.
    Question: The input text
    Answer: Text with PII masked (e.g., "[EMAIL]", "[SSN]")
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
        return "text"

    @property
    def answer_column(self) -> str:
        return "masked_text"

    def load(self) -> pl.DataFrame:
        """Load PII masking data for the specified split."""
        full_data = _create_pii_dataframe()

        if self.split == Split.TEST:
            # Use last 20% as test
            n_test = max(1, len(full_data) // 5)
            return full_data.tail(n_test)

        # For train and dev, use the first 80%
        n_test = max(1, len(full_data) // 5)
        train_dev = full_data.head(len(full_data) - n_test)

        # Add row index for splitting
        train_dev = train_dev.with_row_index("_idx")

        # Shuffle deterministically and split
        n_total = len(train_dev)
        n_dev = max(1, int(n_total * self.dev_ratio))

        shuffled = train_dev.sample(fraction=1.0, seed=self.seed, shuffle=True)

        if self.split == Split.DEV:
            result = shuffled.head(n_dev)
        else:  # TRAIN
            result = shuffled.tail(n_total - n_dev)

        return result.drop("_idx")
