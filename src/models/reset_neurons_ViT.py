import jax
import jax.numpy as jnp
import jax.random as jr


# Create a generic function that resets neurons for SNR, CBP, ReDO
# and create another function, or inside the above function, reset the appropriate weights in the Adam optimizer.
# It may make sense to move this into the Transformer factory function in order to deal with arbitrary architectures.

# Then create function/datastructures for SNR/ReDO/CBP

def get_reset_methods_ViT(config, alg, alg_params):

    
    # NOTE: This is updated
    blocks = []
    for i in range(config.num_layers):
        blocks.append('ViTBlock_'+str(i))

    # NOTE: This does not appear to need more updating
    def init_reset_state(config, alg, alg_params):
        
        if alg == 'SNR' or alg == 'SNR-L2' or alg == 'SNR-L2*':
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

            for i in range(config.num_layers):
                reset_state['thresholds'][blocks[i]] = init_threshold * jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
                reset_state['arrivals_sum'][blocks[i]] =  jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
                reset_state['arrivals_count'][blocks[i]] = jnp.ones((config.n_neurons,), dtype=jnp.uint32)  
            
            return reset_state
        
        # NOTE: This does not appear to need updating
        if alg == "CBP":
            # decay_rate = 0.99, age_threshold = 100 are the original default hyperparameters from Dohare et al.
            reset_state = {
                'a': {},
                'f':  {},
                'u': {},
                'reset_freq': alg_params['CBP_reset_freq'],
                'decay_rate': 0.99,
                'age_threshold': 100
            }
            
            for i in range(config.num_layers):
                reset_state['a'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32)  
                reset_state['f'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32)  
                reset_state['u'][blocks[i]] = jnp.zeros((config.n_neurons,), dtype=jnp.float32) 

            return reset_state

        if alg == "ReDO":
            reset_state = {
                'threshold': alg_params['ReDO_threshold'],
                'reset_freq': alg_params['ReDO_reset_freq']
            }

            return reset_state

    # NOTE: I have only checked CBP here, TODO: need to look at SNR and ReDO
    def get_reset_neurons(config, alg, alg_params):
        
        # Initialize the new parameters. 
        hidden_dim = config.hidden_dim
        initializer = jax.nn.initializers.lecun_normal()

        # Create function that generates layers of the MLP according to the prior distribution
        @jax.jit
        def generate_layer(key):
            return initializer(key, (hidden_dim, config.n_neurons), jnp.float32)

        # TODO: Return to this after ReDO/CBP
        if alg == 'SNR' or alg == 'SNR-L2' or alg == 'SNR-L2*':

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
            def reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, key):
                
                params = state.params
                opt_state = state.opt_state
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
                return state.replace(params = params, opt_state = opt_state), reset_state, neuron_ages, reset_masks
            
            return reset_neurons

        # TODO: Need to look at this
        # NOTE: This seems to look good, but I haven't tested this on Colab
        if alg == 'ReDO':

            @jax.jit
            def ReDO_reset(state, reset_state, neuron_ages, neuron_pre_activ, reset_masks, key):

                params = state.params
                opt_state = state.opt_state
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

                return state.replace(params = params, opt_state = opt_state), reset_state, neuron_ages, reset_masks
            
            
            # This function updates neuron_ages, therefore, do not need to call update neuron_ages in the training loop 
            # of the experiment
            @jax.jit
            def reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, key):
                
                # Create a dummy reset_masks for jax compilation purposes
                reset_mask = jnp.sum(jnp.abs(neuron_pre_activ['intermediates']['ViTBlock_0']['MLP_0']['features'][0]), axis=(0,1))
                reset_masks = {}
                for block in blocks:
                    reset_masks[block] = reset_mask < reset_mask

                # TODO: Potentially make this deterministic
                key, split_key = jr.split(key)
                ReDO = jr.bernoulli(split_key, reset_state['reset_freq'])
                state, reset_state, neuron_ages, reset_masks = jax.lax.cond(ReDO, ReDO_reset, lambda a,b,c,d,e,f: (a,b,c,e), state, reset_state, neuron_ages, neuron_pre_activ, reset_masks, key)

                return state, reset_state, neuron_ages, reset_masks

            return reset_neurons

        # NOTE: This seems to look good, just need to double check neuron_pre_activ
        # NOTE: neuron_pre_activ looked good to me on Colab
        if alg == 'CBP':
            
            @jax.jit
            def kmin(x, k):
                # Argsort x and get the index of k smalelst element
                idx = jnp.argsort(x)[k-1]
                # Get the value of the k smallest element
                k_min = x[idx]
                return k_min

            @jax.jit
            def get_reset_mask_CBP_multi(a, u_hat, age_threshold, reset_freq, noise):
                mask = (a >= age_threshold)
                mask_complement = 1 - mask

                k = jnp.round(reset_freq).astype(jnp.int32)
                u_hat_old = (u_hat * mask) + (u_hat.max() * mask_complement)
                k_value = kmin(u_hat_old, k)
                # Multiplying by mask in the line below guards against degenerate cases in which 
                # mask == True everywhere or mask == False everywhere
                reset_mask = (u_hat_old <= k_value) * mask
                return reset_mask
                

            # Determine smallest eligible neuron
            @jax.jit
            def get_reset_mask_CBP_min_(a, u_hat, age_threshold, reset_freq, noise):
                mask = (a >= age_threshold)
                mask_complement = 1 - mask

                u_hat_old = (u_hat * mask) + (u_hat.max() * mask_complement)
                # Multiplying by mask in the line below guards against degenerate cases in which 
                # mask == True everywhere or mask == False everywhere
                reset_mask = (u_hat_old == u_hat_old.min()) * mask
                return reset_mask

            # Check if noise is below reset_freq
            @jax.jit
            def get_reset_mask_CBP_min(a, u_hat, age_threshold, reset_freq, noise):
                return jax.lax.cond(noise <= reset_freq, get_reset_mask_CBP_min_, lambda v,w,x,y,z: v == -1, a, u_hat, age_threshold, reset_freq, noise)
                
            @jax.jit
            def get_reset_mask_CBP(a, u_hat, age_threshold, reset_freq, noise):
                reset_freq_ = reset_freq * u_hat.shape[0]
                return jax.lax.cond(reset_freq_ > 1, get_reset_mask_CBP_multi, get_reset_mask_CBP_min, a, u_hat, age_threshold, reset_freq_, noise)

            @jax.jit
            def reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, key):
                
                params = state.params
                opt_state = state.opt_state
                reset_masks = {}

                # Get CBP params
                reset_freq = reset_state['reset_freq']
                a = reset_state['a']
                f = reset_state['f']
                u = reset_state['u']
                decay_rate = reset_state['decay_rate']
                age_threshold = reset_state['age_threshold']

                for block in blocks:

                    # Generate random parameters for reset
                    key, split_key = jr.split(key)
                    params_rand = generate_layer(split_key)
            
                    # Increment neuron_ages
                    a[block] += 1
                    bias_correction = (1 - decay_rate ** a[block])

                    activations = jnp.mean(
                        jnp.where(neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0] > 0, 
                                neuron_pre_activ['intermediates'][block]['MLP_0']['features'][0], 
                                0), 
                        axis=(0, 1)
                    )
                    f_hat = f[block] / bias_correction
                    f[block] = decay_rate * f[block] + (1 - decay_rate) * activations
                    
                    # TODO: Update the following lines for the transformer
                    # y_numerator = jnp.abs(f_hat - activations) * jnp.sum(jnp.abs(state.params[next_layer]['kernel']), axis = 1)
                    # y_denominator = jnp.sum(jnp.abs(state.params[layer]['kernel']), axis = 0)
                    y_numerator = jnp.abs(f_hat - activations) * jnp.sum(jnp.abs(params['params'][block]['MLP_0']['Dense_1']['kernel']), axis = 1)
                    y_denominator =  jnp.sum(jnp.abs(params['params'][block]['MLP_0']['Dense_0']['kernel']), axis = 0)
                    y = y_numerator / y_denominator

                    u_hat = u[block] / bias_correction 
                    u[block] = decay_rate * u[block] + (1 - decay_rate) * y

                    key, split_key = jr.split(key)
                    noise = jr.uniform(split_key)
                    reset_mask = get_reset_mask_CBP(a[block], u_hat, age_threshold, reset_freq, noise)
                    reset_masks[block] = reset_mask

                    a[block] = a[block] * (1 - reset_mask.astype(jnp.int32))
                    u[block] = u[block] * (1 - reset_mask.astype(jnp.int32))
                    f[block] = f[block] * (1 - reset_mask.astype(jnp.int32))

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

                # Update reset state
                reset_state['a'] = a             
                reset_state['f'] = f
                reset_state['u'] = u
            
                return state.replace(params = params, opt_state = opt_state), reset_state, neuron_ages, reset_masks
            
            return reset_neurons

    # TODO 
    return init_reset_state, get_reset_neurons(config, alg, alg_params)
    
     