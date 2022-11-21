from dataclasses import dataclass
from .dataset import DatasetConfig
from .model import ModelConfig


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    random_state: int
    eda_reports_dir: str
