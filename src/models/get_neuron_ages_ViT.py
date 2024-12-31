# from .transformer import *

# Creates a dictionary that stores neuron ages.
# A neuron's age is the number of batches since it last fired.
def init_neuron_ages_ViT(config):
  neuron_ages = {}
  for i in range(config.num_layers):
    neuron_ages[f'ViTBlock_{i}'] = jnp.zeros((config.n_neurons,), dtype=jnp.uint32)
  return neuron_ages


def get_neurons_ages_functions_ViT(config):
    
    blocks = []
    for i in range(config.num_layers):
        blocks.append('ViTBlock_'+str(i))

    # This function is updated so that neuron ages are in units of batches, not number of examples
    # Algorithmically, there is no difference since the "number of batches" is in multiples of batch_size * context_window
    # Changing from examples to batches reduces memory utilization
    @jax.jit
    def update_neuron_ages(neuron_ages, neuron_pre_activ):
        for block in blocks:
            is_neuron_dead = jnp.all(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0] < 0.0, axis=(0,1))
            neuron_ages[block] = (neuron_ages[block] + 1) * is_neuron_dead
        return neuron_ages

    # Get the preactivations for each neuron
    @jax.jit
    def get_neuron_pre_activ(train_state,x):
        return train_state.apply_fn(train_state.params, x, True, mutable='intermediates')[1]

    return update_neuron_ages, get_neuron_pre_activ

    # EXAMPLE USAGE

    # update_neuron_ages, get_neuron_pre_activ = get_neurons_ages_functions_ViT(config)

    # neuron_ages = init_neuron_ages(config)
    # neuron_pre_activ = get_neuron_pre_activ(train_state,x)
    # neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)