# ('params', 'Dense_0', 'kernel')
# Classification head’s kernel at the very end.

# ('params', 'PatchEmbed_0', 'Conv_0', 'kernel')
# Convolutional kernel for patch embedding.

# ('params', 'ViTBlock_i', 'ViTSelfAttention_0', 'Dense_0', 'kernel')
# QKV projection matrix in the i-th ViT block’s self-attention.

# ('params', 'ViTBlock_i', 'ViTSelfAttention_0', 'Dense_1', 'kernel')
# Final linear projection in the i-th ViT block’s self-attention.

# ('params', 'ViTBlock_i', 'MLP_0', 'Dense_0', 'kernel')
# First Dense layer in the i-th ViT block’s MLP (expanding hidden dimension).

# ('params', 'ViTBlock_i', 'MLP_0', 'Dense_1', 'kernel')
# Second Dense layer in the i-th ViT block’s MLP (projecting back to the original dimension).

# Example usage: get_kernel_norms_flat(state.params)['params', 'ViTBlock_0', 'ViTSelfAttention_0', 'Dense_0', 'kernel']
# outputs the scalar value for the norm of this kernel matrix.

import jax
from flax.core import FrozenDict
import jax.numpy as jnp

def get_kernel_norms_flat(params: FrozenDict) -> dict:
    """
    Traverse a nested Flax 'params' FrozenDict and return a flat dictionary
    mapping the path of each 'kernel' to its L2 norm.
    
    The keys will be tuples representing the hierarchical path, for example:
        ('PatchEmbed_0', 'Conv_0', 'kernel')
    """
    kernel_norms = {}

    def traverse(tree, path=()):
        if isinstance(tree, (dict, FrozenDict)):
            # Recursively traverse sub-dicts
            for k, v in tree.items():
                traverse(v, path + (k,))
        else:
            # 'tree' is a leaf (ndarray, typically)
            # Check if the final name in 'path' is 'kernel'
            if path[-1] == 'kernel':
                kernel_norms[path] = float(jnp.linalg.norm(tree))

    traverse(params)
    return kernel_norms



@jax.jit
def entropy(probs):
  """Computes the entropy of a probability distribution.

  Args:
    probs: A JAX array representing the probability distribution.

  Returns:
    The entropy of the distribution.
  """
  probs = jnp.where(probs == 0, 1e-10, probs) # for numerical stability
  return -jnp.sum(probs * jnp.log(probs), axis=-1)

get_entropies = jax.jit(jax.vmap(entropy))