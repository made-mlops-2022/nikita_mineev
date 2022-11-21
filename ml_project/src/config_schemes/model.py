from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    type: str
    params: dict
    transform_dataset: bool
    save_path: str
    transformer_save_path: str
    save_predict_path: str
