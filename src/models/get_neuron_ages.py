from .transformer import *

def init_neuron_ages(config):
  neuron_ages = {}
  for i in range(config.n_layer):
    neuron_ages[f'Block_{i}'] = jnp.zeros((config.n_embd * 4,))
  return neuron_ages

# TODO: get blocks programatically
blocks = ['Block_0', 'Block_1', 'Block_2']

@jax.jit
def update_neuron_ages(neuron_ages, neuron_pre_activ):
    for block in blocks:
        is_neuron_dead = jnp.all(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0] < 0.0, axis=(0,1))
        neuron_ages[block] = (neuron_ages[block] + 16*128) * is_neuron_dead
    return neuron_ages

@jax.jit
def get_neuron_pre_activ(train_state,x):
    return train_state.apply_fn(train_state.params, x, True, mutable='intermediates')[1]


# EXAMPLE USAGE

# neuron_ages = init_neuron_ages(config)
# neuron_pre_activ = get_neuron_pre_activ(train_state,x)
# neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)