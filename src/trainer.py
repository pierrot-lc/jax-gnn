import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped
from scipy.stats import kendalltau
from tqdm import tqdm
from wandb.wandb_run import Run

from .dataset import Dataset, GraphData
from .gnn import RankingModel
from .utils import count_params, keys_generator, pointwise_cross_entropy
from .utils import kendalltau as kd


class Trainer(eqx.Module):
    batch_size: int = eqx.field(static=True)
    evaluation_freq: int = eqx.field(static=True)
    evaluation_iters: int = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    training_iters: int = eqx.field(static=True)

    def train(
        self,
        model: RankingModel,
        train_dataset: Dataset,
        test_dataset: Dataset,
        logger: Run,
        key: Shaped[PRNGKeyArray, ""],
    ):
        logger.summary["experiment-dir"] = os.getcwd()
        logger.summary["n-edges"] = train_dataset.graphs[0].edges.shape[0]
        logger.summary["n-nodes"] = train_dataset.graphs[0].scores.shape[0]
        logger.summary["n-params"] = count_params(model)
        logger.summary["testing-size"] = len(test_dataset)
        logger.summary["training-size"] = len(train_dataset)

        keys = keys_generator(key)
        model = eqx.nn.inference_mode(model, False)
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        for batch_id, batch in tqdm(
            enumerate(train_dataset.iter(self.batch_size, self.training_iters, next(keys))),
            desc="Training iters",
            total=self.training_iters,
            leave=True,
        ):
            model, opt_state = self.batch_update(model, batch, opt_state, next(keys))

            if batch_id % self.evaluation_freq == 0:
                logger.log(
                    {
                        "train": self.eval(model, train_dataset, next(keys)),
                        "test": self.eval(model, test_dataset, next(keys)),
                    },
                    step=batch_id,
                )

    def eval(
        self, model: RankingModel, dataset: Dataset, key: Shaped[PRNGKeyArray, ""]
    ) -> dict[str, float]:
        model = eqx.nn.inference_mode(model, True)
        keys = keys_generator(key)
        metrics = [
            self.batch_metrics(model, batch, next(keys))
            for batch in tqdm(
                dataset.iter(self.batch_size, self.evaluation_iters, next(keys)),
                desc="Evaluation iters",
                total=self.evaluation_iters,
                leave=False,
            )
        ]
        metrics = jax.tree.map(lambda *xs: jnp.concat(xs), *metrics)
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics = jax.tree.map(float, metrics)
        return metrics

    @eqx.filter_jit
    def batch_update(
        self,
        model: RankingModel,
        batch: GraphData,
        opt_state: optax.OptState,
        key: Shaped[PRNGKeyArray, ""],
    ) -> tuple[RankingModel, optax.OptState]:
        def loss_fn(model: RankingModel):
            sk1, sk2 = jr.split(key)
            batch_size, _ = batch.scores.shape
            pred_scores = jax.vmap(model)(batch, jr.split(sk1, batch_size))
            losses = jax.vmap(pointwise_cross_entropy)(
                pred_scores,
                batch.scores,
                batch.mask,
                jr.split(sk2, batch_size),
            )
            return losses.mean()

        grads = eqx.filter_grad(loss_fn)(model)
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    @jaxtyped(typechecker=beartype)
    def batch_metrics(
        self, model: RankingModel, batch: GraphData, key: Shaped[PRNGKeyArray, ""]
    ) -> dict[str, Float[Array, " batch_size"]]:
        metrics = dict()
        batch_size, _ = batch.scores.shape
        sk1, sk2 = jr.split(key)

        pred_scores = jax.vmap(model)(batch, jr.split(sk1, batch_size))
        losses = jax.vmap(pointwise_cross_entropy)(
            pred_scores, batch.scores, batch.mask, jr.split(sk2, batch_size)
        )
        kt_scores = jnp.array(
            [
                kendalltau(pred[mask], true[mask], nan_policy="raise").statistic
                for pred, true, mask in zip(pred_scores, batch.scores, batch.mask)
            ]
        )
        # Score can be inf if all nodes have the same score.
        kt_scores = jnp.where(jnp.isfinite(kt_scores), kt_scores, 0.0)

        metrics["loss"] = losses
        metrics["KT-scores"] = kt_scores
        metrics["KT-scores (mine)"] = jax.vmap(kd)(pred_scores, batch.scores, batch.mask)
        return metrics
