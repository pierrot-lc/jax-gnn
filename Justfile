device-check:
    TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.extend.backend import get_backend; print('Backend:', get_backend().platform)"

tests:
    JAX_DISABLE_JIT=True JAX_PLATFORM_NAME=cpu python3 -m pytest -v tests/
