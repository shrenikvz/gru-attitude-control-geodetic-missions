'''
GRU network class using JAX

Author: Shrenik Zinage

'''


# Required libraries
import jax                                          # JAX library for accelerated numerical computing
import jax.numpy as jnp                             # NumPy-like API for JAX
import jax.lax as lax                               # Linear algebra library for JAX
import jax.random as jrandom                        # Random number generation for JAX
import equinox as eqx                               # Deep learning library for JAX
from typing import List, Tuple                             # Type hints for function signatures


class GRUNetwork(eqx.Module):

        hidden_sizes: List[int]
        cells: List[eqx.Module]
        linear: eqx.nn.Linear
        bias: jax.Array

        def __init__(self, in_size: int, out_size: int, hidden_sizes: List[int], *, key: jax.random.PRNGKey):

            keys = jrandom.split(key, num=len(hidden_sizes) + 1)
            self.hidden_sizes = hidden_sizes
            self.cells = [eqx.nn.GRUCell(in_size if i == 0 else hidden_sizes[i - 1],
                                        hidden_sizes[i],
                                        key=keys[i]) for i in range(len(hidden_sizes))]
            self.linear = eqx.nn.Linear(hidden_sizes[-1], out_size, use_bias=False, key=keys[-1])
            self.bias = jnp.zeros(out_size)

        def __call__(self, input: jax.Array) -> jax.Array:

            hiddens = [jnp.zeros((size,)) for size in self.hidden_sizes]

            # Define a function for each cell
            def cell_fn(cell: eqx.Module, carry: jax.Array, inp: jax.Array) -> Tuple[jax.Array, None]:
                return cell(inp, carry), None

            # Process input through each cell
            for i, cell in enumerate(self.cells):
                out, _ = lax.scan(lambda carry, inp: cell_fn(cell, carry, inp), hiddens[i], input)
                if i != len(self.cells) - 1:  # Do not apply ReLU after the last GRU layer
                    out = jax.nn.relu(out)
                input = out.reshape(-1,1).T

            return self.linear(out) + self.bias