import jax
import jax.numpy as jnp
import jax.random as jr

# TODO
# Perhaps move this code to another file

# Create a generic function that resets neurons for ART, CBP, ReDO
# and create another function, or inside the above function, reset the appropriate weights in the Adam optimizer.
# It may make sense to move this into the Transformer factory function in order to deal with arbitrary architectures.

# Then create function/datastructures for ART/ReDO/CBP

def get_reset_methods(config, alg, alg_params):

    
    blocks = []
    for i in range(config.n_layer):
        blocks.append('Block_'+str(i))

    def init_reset_state(config, alg, alg_params):
        if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*':
            reset_state = {
                'thresholds': {},
                'threshold_expansion_factor': 2,
                'arrivals_sum': {},
                'arrivals_count': {},
                # reset_freq is how often the thresholds are updated
                'reset_freq': alg_params['reset_freq'],
                'reset_percentile': alg_params['reset_percentile']
            }

            # Get initial threshold
            init_threshold = alg_params['threshold']

            for i in range(config.n_layer):
                reset_state['thresholds'][blocks[i]] = init_threshold * jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
                reset_state['arrivals_sum'][blocks[i]] =  jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
                reset_state['arrivals_count'][blocks[i]] = jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
            
            return reset_state
        
        if alg == "CBP-L2":
            # TODO
            reset_state = {
                'a': {},
                'f':  {},
                'u': {},
                'reset_freq': alg_params['reset_freq']
            }
            for i in range(config.n_layer):
                reset_state['a'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32)  
                reset_state['f'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32)  
                reset_state['u'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32) 

            return reset_state

        if alg == "ReDO-L2":
            reset_state = {
                'threshold': alg_params['ReDO_threshold'],
                'reset_freq': alg_params['ReDO_reset_freq']
            }

            return reset_state

    def get_reset_neurons(config, alg, alg_params):
        
        # Initialize the new parameters. 
        n_embd = config.n_embd
        initializer = jax.nn.initializers.lecun_normal()

        # Create function that generates layers of the MLP according to the prior distribution
        @jax.jit
        def generate_layer(key):
            return initializer(key, (n_embd, config.n_neurons), jnp.float32)

        if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*':

            @jax.jit
            def update_thresholds(reset_state):

                reset_percentile = reset_state['reset_percentile']
                gamma = reset_state['threshold_expansion_factor']

                for block in blocks:
                    thresholds = reset_state['thresholds'][block]
                
                    # estimate lambda parameter
                    lambda_hat = reset_state['arrivals_count'][block] / reset_state['arrivals_sum'][block]
                    threshold_percentiles = - jnp.log(1 - reset_percentile) / lambda_hat
                    
                    expansion_mask = thresholds <= threshold_percentiles

                    # update thresholds:
                    thresholds = jnp.ceil(((gamma * thresholds) * expansion_mask) + (threshold_percentiles * (1 - expansion_mask))).astype(jnp.uint32)
                    
                    reset_state['thresholds'][block] = thresholds
                    reset_state['arrivals_count'][block] = (0 * reset_state['arrivals_count'][block] + 1).astype(jnp.uint32)
                    reset_state['arrivals_sum'][block] = (0 * reset_state['arrivals_sum'][block] + 1).astype(jnp.uint32)
                return reset_state
            
            # This function updates neuron_ages, therefore, do not need to call update neuron_ages in the training loop 
            # of the experiment
            @jax.jit
            def reset_neurons(train_state, reset_state, neuron_ages, neuron_pre_activ, key):
                
                params = train_state.params
                opt_state = train_state.opt_state
                reset_masks = {}

                for block in blocks:

                    # Generate random parameters for reset
                    key, split_key = jr.split(key)
                    params_rand = generate_layer(split_key)

                    thresholds = reset_state['thresholds'][block]

                    # Temporarilly store neuron_age
                    pre_neuron_ages = neuron_ages[block]

                    # Update neuron_ages, without considering resets (yet)
                    is_neuron_dead_on_batch = jnp.all(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0] < 0.0, axis=(0,1))
                    neuron_ages[block] = (neuron_ages[block] + 1) * is_neuron_dead_on_batch

                    # Get firing_mask
                    # firing_mask == True iff the neuron fired in the last batch
                    firing_mask = neuron_ages[block] == 0

                    # Get reset_mask
                    # reset_mask == True if the neuron is to be reset due to exceeding the reset-threshold
                    reset_mask = neuron_ages[block] >= thresholds
                    reset_masks[block] = reset_mask

                    ####################
                    # Reset Neurons and Reset Adam Optimizer Parameters
                    # MLP to MLP Adam Standard Reset based off reset_mask
                    # Reset bias terms to zero
                    params['params'][block]['MLP_0']['Dense_0']['bias'] = params['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['bias'] = opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['bias'] = opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)

                    # Reset incoming neuron weights according to initial distribution
                    params['params'][block]['MLP_0']['Dense_0']['kernel'] = (params['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)) + (params_rand * reset_mask)
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['kernel'] = opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['kernel'] = opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)
                    
                    # Reset outgoing weights to zero
                    params['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * params['params'][block]['MLP_0']['Dense_1']['kernel']
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * opt_state[0].mu['params'][block]['MLP_0']['Dense_1']['kernel'] 
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * opt_state[0].nu['params'][block]['MLP_0']['Dense_1']['kernel'] 
                    ####################

                    # Update Ages due to resets
                    neuron_ages[block] = neuron_ages[block] * (1 - reset_mask)                    

                    # Update arrival times 
                    arrival_mask = firing_mask + reset_mask
                    arrival_times = pre_neuron_ages + 1
                    reset_state['arrivals_sum'][block] = reset_state['arrivals_sum'][block] + (arrival_times * arrival_mask)
                    reset_state['arrivals_count'][block] = reset_state['arrivals_count'][block] + arrival_mask

                # TODO: Potentially make this deterministic
                key, split_key = jr.split(key)
                update_thresholds_cond = jr.bernoulli(split_key, reset_state['reset_freq'])
                reset_state = jax.lax.cond(update_thresholds_cond, update_thresholds, lambda x: x, reset_state)

                # TODO: Potentially return the reset_mask
                # We don't need to output the thresholds since we have those already in the experiment loop through reset_states
                return train_state.replace(params = params, opt_state = opt_state), reset_state, neuron_ages, reset_masks
            
            return reset_neurons


        if alg == 'ReDO-L2':

            @jax.jit
            def ReDO_reset(train_state, reset_state, neuron_ages, neuron_pre_activ, key):

                params = train_state.params
                opt_state = train_state.opt_state
                reset_masks = {}

                # ReDO has onlya single threshold
                threshold = reset_state['threshold']

                for block in blocks:

                    # Generate random parameters for reset
                    key, split_key = jr.split(key)
                    params_rand = generate_layer(split_key)


                    # Compute Neuron scores on the current batch
                    # That is, the s^l_i variables from the Sokar et al. paper.
                    # sum_abs_activations = jnp.sum(jnp.abs(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0]), axis=(0,1))

                    sum_activations = jnp.sum(
                        jnp.where(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0] > 0, 
                                neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0], 
                                0), 
                        axis=(0, 1)
                    )
                    normalization_constant = jnp.sum(sum_activations)
                    neuron_scores = sum_activations / normalization_constant

                    # Get reset_mask
                    # Reset Criteria is simply neuron_scores (s^l_i) <= threshold
                    reset_mask = neuron_scores <= threshold         
                    reset_masks[block] = reset_mask

                    ####################
                    # Reset Neurons and Reset Adam Optimizer Parameters
                    # MLP to MLP Adam Standard Reset based off reset_mask
                    # Reset bias terms to zero
                    params['params'][block]['MLP_0']['Dense_0']['bias'] = params['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['bias'] = opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['bias'] = opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['bias'] * (1 - reset_mask)

                    # Reset incoming neuron weights according to initial distribution
                    params['params'][block]['MLP_0']['Dense_0']['kernel'] = (params['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)) + (params_rand * reset_mask)
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['kernel'] = opt_state[0].mu['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['kernel'] = opt_state[0].nu['params'][block]['MLP_0']['Dense_0']['kernel'] * (1 - reset_mask)
                    
                    # Reset outgoing weights to zero
                    params['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * params['params'][block]['MLP_0']['Dense_1']['kernel']
                    opt_state[0].mu['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * opt_state[0].mu['params'][block]['MLP_0']['Dense_1']['kernel'] 
                    opt_state[0].nu['params'][block]['MLP_0']['Dense_1']['kernel'] = (1 - reset_mask)[:,None] * opt_state[0].nu['params'][block]['MLP_0']['Dense_1']['kernel'] 
                    ####################                 

                return train_state.replace(params = params, opt_state = opt_state), reset_state, neuron_ages, reset_masks
            
            
            # This function updates neuron_ages, therefore, do not need to call update neuron_ages in the training loop 
            # of the experiment
            @jax.jit
            def reset_neurons(train_state, reset_state, neuron_ages, neuron_pre_activ, key):
                
                # TODO: Potentially make this deterministic
                key, split_key = jr.split(key)
                ReDO = jr.bernoulli(split_key, reset_state['reset_freq'])
                train_state = jax.lax.cond(ReDO, ReDO_reset, lambda a,b,c,d,e: a,  train_state, reset_state, neuron_ages, neuron_pre_activ, key)

                return train_state

            return reset_neurons

        if alg == 'CBP-L2':

            @jax.jit
            def reset_neurons(train_state, reset_state, neuron_ages, neuron_pre_activ, key):
                # TODO: Potentially make this deterministic
                key, split_key = jr.split(key)
                ReDO = jr.bernoulli(split_key, reset_state['reset_freq'])
                train_state = jax.lax.cond(ReDO, ReDO_reset, lambda a,b,c,d,e: a,  train_state, reset_state, neuron_ages, neuron_pre_activ, key)

                return train_state
            
            return reset_neurons

    # TODO 
    return init_reset_state, get_reset_neurons(config, alg, alg_params)
    # return init_reset_state, get_reset_neurons
     