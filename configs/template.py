from dataclasses import dataclass
from typing import Literal

import jax.random as jr
from jaxtyping import PRNGKeyArray, Shaped
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    n_graphs: int
    n_nodes: int
    key: Shaped[PRNGKeyArray, ""]

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class ModelConfig:
    hidden_dim: int
    mode: Literal["adjacency", "edges"]
    n_layers: int
    key: Shaped[PRNGKeyArray, ""]

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class TrainerConfig:
    batch_size: int
    evaluation_freq: int
    evaluation_iters: int
    learning_rate: float
    training_iters: int
    key: Shaped[PRNGKeyArray, ""]

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class WandbConfig:
    entity: str
    mode: Literal["online", "offline", "disabled"]


@dataclass
class MainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    wandb: WandbConfig

    @classmethod
    def from_dict(cls, config: DictConfig) -> "MainConfig":
        return cls(
            dataset=DatasetConfig(**config.dataset),
            model=ModelConfig(**config.model),
            trainer=TrainerConfig(**config.trainer),
            wandb=WandbConfig(**config.wandb),
        )
