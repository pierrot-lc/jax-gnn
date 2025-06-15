import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jaxtyping import Bool, Float
from scipy.stats import kendalltau as kd_scipy
from src.utils import kendalltau as kd_jax


@pytest.mark.parametrize(
    "x, y, m",
    [
        (
            jnp.array([3, 1, 0]),
            jnp.array([3, 1, 0]),
            jnp.array([True, True, True]),
        ),
        (
            jnp.array([3, 1, 0]),
            jnp.array([0, 3, 1]),
            jnp.array([True, True, True]),
        ),
        (
            jnp.array([3, 1, 0]),
            jnp.array([4, 1, 0]),
            jnp.array([True, False, False]),
        ),
        (
            jnp.array([3, 1, 0]),
            jnp.array([3, 1, 0]),
            jnp.array([False, False, False]),
        ),
        (
            jr.normal(jr.key(0), (100,)),
            jr.normal(jr.key(1), (100,)),
            jr.uniform(jr.key(2), (100,)) < 0.5,
        ),
    ],
)
def test_kendalltau(
    x: Float[Array, " n_points"], y: Float[Array, " n_points"], m: Bool[Array, " n_points"]
):
    scipy_score = kd_scipy(
        np.array(x[m]),
        np.array(y[m]),
        variant="b",
    ).statistic
    jax_score = kd_jax(x, y, m)

    if np.isnan(scipy_score):
        scipy_score = 0.0

    assert jnp.allclose(scipy_score, jax_score)
