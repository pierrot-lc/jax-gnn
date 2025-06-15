from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar, Shaped


@eqx.filter_jit
def pointwise_cross_entropy(
    pred_scores: Float[Array, " n_nodes"],
    true_scores: Float[Array, " n_nodes"],
    mask: Bool[Array, " n_nodes"],
    key: Shaped[PRNGKeyArray, ""],
    sampling_factor: float = 20.0,
) -> Scalar:
    """A simple ranking loss."""
    (n_nodes,) = pred_scores.shape
    total_sampling = int(n_nodes * sampling_factor)
    perms = jr.choice(
        key,
        jnp.arange(n_nodes),
        (2, total_sampling),
        replace=True,
        p=mask / mask.sum(),  # Ignore masked nodes.
    )

    x1_true, x2_true = true_scores[perms]
    x1_pred, x2_pred = pred_scores[perms]

    y = jnp.sign(x1_true - x2_true)
    loss = -jax.nn.log_sigmoid(y * (x1_pred - x2_pred))
    return loss.mean()


@eqx.filter_jit
def kendalltau(
    x: Float[Array, " n_points"], y: Float[Array, " n_points"], mask: Bool[Array, " n_points"]
) -> Scalar:
    """Kendall's tau correlation measure.

    Implements the tau-b variant in O(n²). Do not use this implementation on a large number of
    points. Its only advantage is that for a small number of points it will run fast on GPU.

    ---
    See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    """
    x_sign = jnp.sign(x[None, :] - x[:, None]).astype(int)
    y_sign = jnp.sign(y[None, :] - y[:, None]).astype(int)
    mask = mask[:, None] @ mask[None, :]
    mask = jnp.triu(mask, k=1)

    P = jnp.sum((x_sign == y_sign) & (x_sign != 0) & (y_sign != 0), where=mask)
    Q = jnp.sum((x_sign != y_sign) & (x_sign != 0) & (y_sign != 0), where=mask)
    T = jnp.sum((x_sign == 0) & (y_sign != 0), where=mask)
    U = jnp.sum((y_sign == 0) & (x_sign != 0), where=mask)

    score = (P - Q) / jnp.sqrt((P + Q + T) * (P + Q + U))
    score = jnp.where(jnp.isfinite(score), score, 0.0)
    return score


def keys_generator(key: Shaped[PRNGKeyArray, ""]) -> Iterator[Shaped[PRNGKeyArray, ""]]:
    """Generate infinitely many keys on demand."""
    while True:
        key, sk = jr.split(key)
        yield sk


def count_params(model: eqx.Module) -> int:
    """Count the number of parameters of the given equinox module."""
    params = eqx.filter(model, eqx.is_array)
    n_params = jax.tree.map(lambda p: jnp.prod(jnp.array(p.shape)), params)
    n_params = jnp.array(jax.tree.leaves(n_params))
    n_params = jnp.sum(n_params)
    return int(n_params)
