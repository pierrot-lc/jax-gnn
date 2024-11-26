import functools
from typing import Mapping, Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import optax

from datasets import load_dataset
from jaxtyping import Array, Float, Int
from tqdm import notebook as tqdm

from transformers import AutoTokenizer