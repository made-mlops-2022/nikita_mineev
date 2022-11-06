from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PathConfig:
    raw: str
    train: str
    val: str
    test: str


@dataclass(frozen=True)
class FeaturesConfig:
    categorical_features: list[Union[int, str]]
    numerical_features: list[Union[int, str]]
    target_features: list[Union[int, str]]


@dataclass(frozen=True)
class SplitConfig:
    train_size: int
    val_size: int
    test_size: int


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str
    path: PathConfig
    features: FeaturesConfig
    split: SplitConfig
