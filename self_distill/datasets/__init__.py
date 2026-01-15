from enum import Enum

from self_distill.datasets.base import BaseDataset, DatasetItem, Split
from self_distill.datasets.cola import CoLADataset, CoLAItem
from self_distill.datasets.gsm8k import GSM8KDataset
from self_distill.datasets.pii import PIIDetectionDataset, PIIMaskingDataset
from self_distill.datasets.sciq import SciQDataset, SciQItem, SciQPhysicsDataset
from self_distill.datasets.ai4privacy import AI4PrivacyDataset, AI4PrivacyItem, load_ai4privacy


class DATA(Enum):
    """Available datasets."""

    GSM8K = "gsm8k"
    COLA = "cola"
    PII_DETECTION = "pii_detection"
    PII_MASKING = "pii_masking"
    SCIQ = "sciq"
    SCIQ_PHYSICS = "sciq_physics"
    AI4PRIVACY = "ai4privacy"  # 200K PII samples from HuggingFace


def load_dataset(
    dataset: DATA,
    split: Split | str,
    **kwargs,
) -> BaseDataset:
    """
    Load a dataset by name and split.

    Args:
        dataset: The dataset to load (e.g., DATA.GSM8K, DATA.COLA)
        split: The split to load ("train", "dev", or "test")
        **kwargs: Additional arguments passed to the dataset constructor

    Returns:
        A BaseDataset instance for the requested dataset and split

    Example:
        >>> from self_distill.datasets import DATA, load_dataset
        >>> train_data = load_dataset(DATA.GSM8K, "train")
        >>> for item in train_data:
        ...     print(item.question, item.answer)

        >>> # CoLA with rule IDs exposed
        >>> cola = load_dataset(DATA.COLA, "train", include_rule_id=True)
        >>> for item in cola:
        ...     print(item.question, item.answer, item.rule_id)
    """
    if dataset == DATA.GSM8K:
        return GSM8KDataset(split=split, **kwargs)
    elif dataset == DATA.COLA:
        return CoLADataset(split=split, **kwargs)
    elif dataset == DATA.PII_DETECTION:
        return PIIDetectionDataset(split=split, **kwargs)
    elif dataset == DATA.PII_MASKING:
        return PIIMaskingDataset(split=split, **kwargs)
    elif dataset == DATA.SCIQ:
        return SciQDataset(split=split, **kwargs)
    elif dataset == DATA.SCIQ_PHYSICS:
        return SciQPhysicsDataset(split=split, **kwargs)
    elif dataset == DATA.AI4PRIVACY:
        return AI4PrivacyDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


__all__ = [
    "DATA",
    "Split",
    "BaseDataset",
    "DatasetItem",
    "CoLADataset",
    "CoLAItem",
    "GSM8KDataset",
    "PIIDetectionDataset",
    "PIIMaskingDataset",
    "SciQDataset",
    "SciQItem",
    "SciQPhysicsDataset",
    "AI4PrivacyDataset",
    "AI4PrivacyItem",
    "load_ai4privacy",
    "load_dataset",
]
