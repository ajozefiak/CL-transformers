# TODO
# import datasets
# import tiktoken
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np

# TODO verify that this works
from ..models import *

####################
# Example Code for create datasets_A, datasets_B
####################
# import datasets
# datasets_CNN = []
# datasets_People = []
# for year in [2016, 2017, 2018, 2019]:
#     for i in range(1,13):
#         ds_temp_CNN = load_from_disk(f'/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/Late August/Process All the News/Data/CNN_tokenized_129/CNN_{i}_{year}_tkn_129.hf')
#         datasets_CNN.append(ds_temp_CNN)

#         ds_temp_People = load_from_disk(f'/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/Late August/Process All the News/Data/People_tokenized_129/People_{i}_{year}_tkn_129.hf')
#         datasets_People.append(ds_temp_People)
#########################
#########################

# art_per_task is how many articles a task has
# percent_mixing determines how we mix articles families
# for instance percent_mixing = 0.7 => 70-30, 30-70, 70-30, 30-70, ... splits of families A and B
def create_two_tasks(ds1, ds2, col, art_per_task, percent_mixing):
    idx = int(percent_mixing * art_per_task)
    # task1_tokens = ds1[col][:idx] + ds2[col][:idx]
    # task2_tokens = ds1[col][idx:art_per_task] + ds2[col][idx:art_per_task]
    
    task1_tokens = ds1[col][:idx] + ds2[col][idx:art_per_task]
    task2_tokens = ds1[col][idx:art_per_task] + ds2[col][:idx]
    return task1_tokens, task2_tokens

def permute_rows(tokens, T, key):
    # Randomize the order of examples
    num_rows = tokens.shape[0] // T
    tokens = tokens[:num_rows * T] # Trim excess tokens if any

    # Step 1: Reshape into contiguous batches
    rows = tokens.reshape((num_rows, T))

    # Step 2: Shuffle the batches
    # Generate a permutation of indices for the number of batches
    permuted_indices = jax.random.permutation(key, num_rows)

    # Apply permutation to shuffle batches
    shuffled_rows = rows[permuted_indices]

    # Step 3: Flatten back to 1D array if necessary
    shuffled_tokens = shuffled_rows.flatten()

    return shuffled_tokens

# NOTE: Because each batch is independent. We reduce the window size by 1 so as to not
# be predicting the first word of the next batch.
class DataLoader:
    def __init__(self, B, T, tokens, col, key):
        self.current_position = 0
        self.B = B
        self.T = T
        self.key = key

        self.tokens = jnp.array(np.concatenate(tokens))
        key, split_key = jr.split(self.key)
        self.key = key
        self. tokens = permute_rows(self.tokens, T, split_key)

        print(f"loaded {len(self.tokens)} tokens in the datasets" )
        print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        # Old implementation when predicting a long stream of text
        # x,y = jnp.reshape(buf[:-1],(B,T)), jnp.reshape(buf[1:],(B,T))

        # New implementation when each batch is its own independent
        # print(self.current_position)
        x,y = jnp.reshape(buf[:-1],(B,T)), jnp.reshape(buf[1:],(B,T))
        x,y = x[:,:-1], y[:,:-1]

        # The rest as usual
        self.current_position += B*T
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0

            # Randomize the order of examples
            num_rows = self.tokens.shape[0] // (T)
            tokens = self.tokens[:num_rows * self.T] # Trim excess tokens if any

            # Step 1: Reshape into contiguous batches
            rows = tokens.reshape((num_rows, self.T))

            # Step 2: Shuffle the batches
            # Generate a permutation of indices for the number of batches
            key, split_key = jr.split(self.key)
            self.key = key
            permuted_indices = jax.random.permutation(split_key, num_rows)

            # Apply permutation to shuffle batches
            shuffled_rows = rows[permuted_indices]

            # Step 3: Flatten back to 1D array if necessary
            shuffled_tokens = shuffled_rows.flatten()

            self.tokens = shuffled_tokens

        return x,y

    # Returns the number of batches in an expoch
    def epoch_len(self):
        return len(self.tokens)//(self.B*self.T)

# We replace N, number of tokens, with art_per_task: the number of articles (example) per task
# We used B = 16, T = 129 in our experiments in Colab
# tasks => months, because now we have twice as many tasks as months
def run_experiment_Alt_CATN(alg, alg_params, datasets_A, datasets_B, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq):
    
    # TODO: Remove
    # Get the data_loader_class
    # data_loader_class = get_dataloader(text)

    # Initialize the random key
    random_key = jr.PRNGKey(seed)
    print(f"Initial key: {random_key}, Type: {type(random_key)}")
    random_key, split_key = jr.split(random_key)

    # Get the total number of training steps
    epoch_len = art_per_task // B
    train_steps_per_task = epoch_len * epochs
    print(f"Epoch length: {epoch_len}")
    print(f"Training steps per task: {train_steps_per_task}")
    # TODO: remove the line below
    # train_steps_per_task = (N * epochs) // (B * T)

    # TODO: change this if we want to play with the ModelConfig - maybe make it an argument
    config = ModelConfig(vocab_size=50257)
    train_state, train_step = get_transformer_methods(config, alg, alg_params, split_key)
    neuron_ages = init_neuron_ages(config)
    if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*' or alg == 'ReDO' or alg == 'CBP':
        init_reset_state, reset_neurons = get_reset_methods(config, alg, alg_params)
        reset_state = init_reset_state(config, alg, alg_params)

        # Initialize reset_mask_array
        reset_mask_array = np.zeros((3, int(train_steps_per_task * months * 2), int(config.n_embd * 4)), dtype = bool)

    # initialize neuron_ages_array
    if save_neuron_ages:
        neuron_ages_array = np.zeros((3, int(train_steps_per_task * months * 2), int(config.n_embd * 4)), dtype = np.uint32)

    #  Initialize params_list
    if save_weights:
        params_list = []
        probs_list = []

    # Iniitialize loss_array
    loss_array = np.zeros(train_steps_per_task * months * 2)
    t = 0

    
    t0 = time.time()
    # TODO: Delete the line below
    # for task in range(tasks):
    for i in range(months):

        task_1_tokens, task_2_tokens = create_two_tasks(datasets_A[i], datasets_B[i], 'tokens_129', art_per_task, percent_mixing)

        # Alternate between the two tasks of each month
        for j in range(2):
            if verbose:
                print(f"Task {j} of Month {i}")
            if j == 0:
                # TODO: perhaps create datasets_A and datasets_B in this function
                tokens = task_1_tokens
            else:
                tokens = task_2_tokens


            # OLD CODE: 
            # Split the random key
            random_key, split_key = jr.split(random_key)
            data_loader = DataLoader(B=B, T=T, tokens=tokens, col='tokens_129', key=split_key)
            
            ###############
            # In principle, we should be able to copy the for loop below to other experiments as long as the 
            # data_loader API is unchanged.
            ###############
            for step in range(train_steps_per_task):
                
                ###############
                # COMMON APPLICATION of DATA LOADER
                ###############
                x,y = data_loader.next_batch()
                
                ###############
                # COMMON TRAINING LOOP:
                # (1) maps x,y -> (new) train_state (and reset_state)
                # (2) outputs loss and neuron_ages for logging purposes
                # In practice, it should be the case that this code can be copied to every experiment
                ###############
                neuron_pre_activ = get_neuron_pre_activ(train_state, x)
                random_key, split_key = jr.split(random_key)
                loss, train_state = train_step(train_state, x, y, split_key)
                # Perform reset step and 
                if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*' or alg == 'ReDO' or alg == 'CBP':
                    random_key, split_key = jr.split(random_key)
                    train_state, reset_state, neuron_ages, reset_mask = reset_neurons(train_state, reset_state, neuron_ages, neuron_pre_activ, split_key)
                    # Store reset_mask in reset_mask_array
                    reset_mask_array[0,t,:] = reset_mask['Block_0']
                    reset_mask_array[1,t,:] = reset_mask['Block_1']
                    reset_mask_array[2,t,:] = reset_mask['Block_2']
                else:
                    neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)  
                ###############
                ###############

                ###############
                # Save Weights
                ###############
                if save_weights & (t % save_weights_freq == 0):
                    params_list.append(train_state.params)
                    
                    temp_probs = []
                    temp_probs.append(neuron_pre_activ['intermediates']['Block_0']['CausalSelfAttention_0']['probs'])
                    temp_probs.append(neuron_pre_activ['intermediates']['Block_1']['CausalSelfAttention_0']['probs'])
                    temp_probs.append(neuron_pre_activ['intermediates']['Block_2']['CausalSelfAttention_0']['probs'])
                    probs_list.append(temp_probs)
                
                # Update loss_array and neuron_ages_array for logigng purposes
                loss_array[t] = loss
                if save_neuron_ages:
                    neuron_ages_array[0,t,:] = neuron_ages['Block_0']
                    neuron_ages_array[1,t,:] = neuron_ages['Block_1']
                    neuron_ages_array[2,t,:] = neuron_ages['Block_2']
                t += 1

                if verbose & (step % print_freq == 0):
                    print(f"step {step}/{train_steps_per_task} | loss {loss:.4f} |")

    print(f"Time to complete experiment: {time.time() - t0}")

    #  Save results
    if save_results:
        #  save and pickle loss_array and neuron_ages_array at "save_path"
        # Ensure the path exists (create if it does not)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define file names for the data
        loss_path = os.path.join(save_path, "loss_array.pkl")
        ages_path = os.path.join(save_path, "neuron_ages_array.pkl")
        reset_mask_path = os.path.join(save_path, "reset_mask_array.pkl")
        weights_path = os.path.join(save_path, "params_list.pkl")
        probs_path = os.path.join(save_path, "probs_list.pkl")

        # Save the loss_array to a pickle file
        with open(loss_path, 'wb') as f:
            pickle.dump(loss_array, f)
            print(f"Saved loss_array to {loss_path}")

        # Save the params_list
        if save_weights:
            with open(weights_path, 'wb') as f:
                pickle.dump(params_list, f)
                print(f"Saved params_list to {weights_path}")
            with open(probs_path, 'wb') as f:
                pickle.dump(probs_list, f)
                print(f"Saved probs_list to {probs_path}")

        if save_neuron_ages:
            # Save the neuron_ages_array to a pickle file
            with open(ages_path, 'wb') as f:
                pickle.dump(neuron_ages_array, f)
                print(f"Saved neuron_ages_array to {ages_path}")
        
        if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*' or alg == 'ReDO' or alg == 'CBP':
            # Save the reset_mask_array to a pkl file
            with open(reset_mask_path, 'wb') as f:
                pickle.dump(reset_mask_array, f)
                print(f"Saved reset_mask_array to {reset_mask_path}")

    # Return results in case we want to analyze/plot immediately
    if save_neuron_ages:
        if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*' or alg == 'ReDO' or alg == 'CBP':
            return loss_array, neuron_ages_array, reset_mask_array
        else:
            return loss_array, neuron_ages_array
    else:
        if alg == 'ART' or alg == 'ART-L2' or alg == 'ART-L2*' or alg == 'ReDO' or alg == 'CBP':
            return loss_array, reset_mask_array
        else:
            return loss_array