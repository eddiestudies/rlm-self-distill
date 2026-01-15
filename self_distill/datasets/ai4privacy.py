"""
AI4Privacy PII Masking Dataset (200K samples)

A large dataset with diverse PII types including:
- FIRSTNAME, LASTNAME, EMAIL, PHONE
- SSN, CREDITCARD, IBAN
- ADDRESS, CITY, COUNTRY
- AGE, GENDER, HEIGHT
- VEHICLEVIN, PHONEIMEI
- And 40+ more categories

Source: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
"""

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class PIIEntity:
    """A single PII entity in the text."""

    label: str
    value: str
    start: int
    end: int


@dataclass
class AI4PrivacyItem:
    """A single item from AI4Privacy dataset."""

    question: str
    answer: str
    source_text: str
    masked_text: str
    entities: list[PIIEntity]
    language: str


class AI4PrivacyDataset:
    """
    AI4Privacy PII Masking dataset with 200K samples.

    Each sample contains:
    - source_text: Original text with PII
    - masked_text: Text with PII replaced by [LABEL] tokens
    - entities: List of PII entities with labels and positions

    Usage:
        dataset = AI4PrivacyDataset(split="train", limit=1000)
        for item in dataset:
            print(f"Text: {item.source_text}")
            print(f"Entities: {[e.label for e in item.entities]}")
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        pii_types: Optional[list[str]] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split: Dataset split ("train" only for this dataset)
            limit: Maximum number of samples to load
            pii_types: Filter to specific PII types (e.g., ["EMAIL", "SSN"])
            language: Filter to specific language (e.g., "en")
        """
        self.split = split
        self.limit = limit
        self.pii_types = set(pii_types) if pii_types else None
        self.language = language
        self._items: Optional[list[AI4PrivacyItem]] = None

    def _load(self) -> list[AI4PrivacyItem]:
        """Load the dataset from HuggingFace."""
        from datasets import load_dataset

        # Use streaming for efficiency with large dataset
        ds = load_dataset(
            "ai4privacy/pii-masking-200k",
            split=self.split,
            streaming=True,
        )

        items = []
        for i, row in enumerate(ds):
            if self.limit and i >= self.limit:
                break

            # Filter by language if specified
            if self.language and row.get("language") != self.language:
                continue

            # Parse entities
            entities = []
            for mask in row.get("privacy_mask", []):
                entity = PIIEntity(
                    label=mask["label"],
                    value=mask["value"],
                    start=mask["start"],
                    end=mask["end"],
                )
                entities.append(entity)

            # Filter by PII types if specified
            if self.pii_types:
                entity_labels = {e.label for e in entities}
                if not entity_labels.intersection(self.pii_types):
                    continue

            # Create question for the task
            entity_labels = [e.label for e in entities]
            question = f"Detect PII in: {row['source_text']}"
            answer = (
                f"Found PII: {', '.join(entity_labels)}" if entities else "No PII found"
            )

            item = AI4PrivacyItem(
                question=question,
                answer=answer,
                source_text=row["source_text"],
                masked_text=row["target_text"],
                entities=entities,
                language=row.get("language", "unknown"),
            )
            items.append(item)

        return items

    @property
    def items(self) -> list[AI4PrivacyItem]:
        """Lazy load items."""
        if self._items is None:
            self._items = self._load()
        return self._items

    def __iter__(self) -> Iterator[AI4PrivacyItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> AI4PrivacyItem:
        return self.items[idx]

    def get_pii_type_counts(self) -> dict[str, int]:
        """Get count of each PII type in the dataset."""
        counts: dict[str, int] = {}
        for item in self.items:
            for entity in item.entities:
                counts[entity.label] = counts.get(entity.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


def load_ai4privacy(
    limit: int = 1000,
    pii_types: Optional[list[str]] = None,
) -> AI4PrivacyDataset:
    """
    Convenience function to load AI4Privacy dataset.

    Args:
        limit: Maximum samples to load
        pii_types: Filter to specific PII types

    Returns:
        AI4PrivacyDataset instance
    """
    return AI4PrivacyDataset(limit=limit, pii_types=pii_types)
