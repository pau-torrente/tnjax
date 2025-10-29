import unittest
import jax
import jax.numpy as jnp

tensors = {
    "A1": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (1, 2, 3))),
    "A2": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 3))),
    "A3": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 1))),
}

edges = [(("A1", 2), ("A2", 0)), (("A2", 2), ("A3", 0))]
