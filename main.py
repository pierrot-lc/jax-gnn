import hydra
import jax.random as jr
import optax
import wandb
from configs.template import MainConfig
from omegaconf import DictConfig, OmegaConf
from src.dataset import Dataset
from src.gnn import RankingModel
from src.trainer import Trainer


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    config = MainConfig.from_dict(dict_config)

    dataset = Dataset.generate(
        config.dataset.n_nodes,
        config.dataset.n_graphs,
        jr.split(config.dataset.key)[0],
    )
    train_dataset, test_dataset = Dataset.split(
        dataset,
        split=0.8,
        key=jr.split(config.dataset.key)[1],
    )

    model = RankingModel(
        config.model.hidden_dim,
        config.model.n_layers,
        key=config.model.key,
    )

    trainer = Trainer(
        config.trainer.batch_size,
        config.trainer.evaluation_freq,
        config.trainer.evaluation_iters,
        optax.adamw(learning_rate=config.trainer.learning_rate),
        config.trainer.training_iters,
    )

    with wandb.init(
        project="gnn-tuto",
        config=OmegaConf.to_container(dict_config),
        entity=config.wandb.entity,
        mode=config.wandb.mode,
    ) as logger:
        trainer.train(
            model,
            train_dataset,
            test_dataset,
            logger,
            config.trainer.key,
        )


if __name__ == "__main__":
    main()
