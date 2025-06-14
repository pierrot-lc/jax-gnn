import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Shaped, jaxtyped

from .dataset import GraphData


class GConvLayer(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: Shaped[PRNGKeyArray, ""]):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(
        self, x: Float[Array, "n_nodes hidden_dim"], a: Int[jsparse.BCOO, "n_nodes n_nodes"]
    ) -> Float[Array, "n_nodes hidden_dim"]:
        x_ = jax.vmap(self.norm)(x)
        x_ = jax.vmap(self.linear)(x_)
        x_ = a @ x_
        return x_ + x


class SwiGLU(eqx.Module):
    linear_1: nn.Linear
    linear_2: nn.Linear
    linear_3: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: Shaped[PRNGKeyArray, ""]):
        keys = iter(jr.split(key, 3))
        self.linear_1 = nn.Linear(hidden_dim, 8 * hidden_dim // 3, key=next(keys))
        self.linear_2 = nn.Linear(hidden_dim, 8 * hidden_dim // 3, key=next(keys))
        self.linear_3 = nn.Linear(8 * hidden_dim // 3, hidden_dim, key=next(keys))
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(self, x: Float[Array, " hidden_dim"]) -> Float[Array, " hidden_dim"]:
        x_ = self.norm(x)
        x_ = jax.nn.swish(self.linear_1(x_)) * self.linear_2(x_)
        x_ = self.linear_3(x_)
        return x_ + x


class RankingModel(eqx.Module):
    hidden_dim: int = eqx.field(static=True)
    convs: GConvLayer
    ffns: SwiGLU
    predict: nn.Linear

    def __init__(self, hidden_dim: int, n_layers: int, *, key: Shaped[PRNGKeyArray, ""]):
        keys = iter(jr.split(key, 3))
        make_conv = lambda k: GConvLayer(hidden_dim, key=k)
        make_ffn = lambda k: SwiGLU(hidden_dim, key=k)

        self.hidden_dim = hidden_dim
        self.convs = eqx.filter_vmap(make_conv)(jr.split(next(keys), n_layers))
        self.ffns = eqx.filter_vmap(make_ffn)(jr.split(next(keys), n_layers))
        self.predict = nn.Linear(hidden_dim, "scalar", use_bias=False, key=next(keys))

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(self, g: GraphData, key: Shaped[PRNGKeyArray, ""]) -> Float[Array, " n_nodes"]:
        dynamic, static = eqx.partition((self.convs, self.ffns), eqx.is_array)

        def scan_fn(x: Float[Array, "n_nodes hidden_dim"], dynamic: PyTree):
            conv, ffn = eqx.combine(dynamic, static)
            x = conv(x, g.adjacency)
            x = jax.vmap(ffn)(x)
            return x, None

        x = jr.uniform(key, (g.n_nodes, self.hidden_dim), dtype=jnp.float32)
        x, _ = jax.lax.scan(scan_fn, x, dynamic)
        x = jax.vmap(self.predict)(x)
        return x
