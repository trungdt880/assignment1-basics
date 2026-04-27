from dataclasses import dataclass, field
from typing import Any


@dataclass
class SingleDatasetConfig:
    # Path to the dataset root
    dataset_path: str
    val_dataset_path: str


@dataclass
class DataConfig:
    # datasets: list[SingleDatasetConfig] = field(default_factory=list)
    dataset_path: str | None = None
    val_dataset_path: str | None = None

    # Data loading
    shuffle: bool = True
    seed: int = 42
