from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data.data_config import DataConfig
from .model import create_model_union_type
from .model.gpt2 import GPT2Config
from .training.training_config import TrainingConfig

ModelUnionType = create_model_union_type()


@dataclass
class Config:
    load_config_path: str | None = None
    model: ModelUnionType = field(default_factory=lambda: GPT2Config())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self, f)

    def load(self, path: Path):
        data = yaml.load(path.read_text(), Loader=yaml.Loader)

        if isinstance(data, dict):
            self.load_dict(data)
        else:
            raise NotImplementedError

        return self

    def load_dict(self, data: dict):
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
        if "training" in data:
            self.training = TrainingConfig(**data["training"])

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        return data

    def validate(self):
        return True


def get_default_config() -> "Config":
    return Config()
