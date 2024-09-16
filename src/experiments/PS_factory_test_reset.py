# Permuted Shakespeare. TODO
import tiktoken
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np

# TODO verify that this works
from ..models import *


# Function that runs the experiment 
# Need to account for reduced vocabulary 
# Need to load or process the data
# Need to save the data somewhere, results should be loss list and neuron activations - for each experiment. 

def get_dataloader(text):

    def process_text(text):
        text = text.replace('\n', ' ')
        return text

    text = process_text(text)

    enc = tiktoken.get_encoding("gpt2")

    # Encode text as tokens, as a list and jax.numpy array
    tokens_list = enc.encode(text)
    tokens_jnp = jnp.array(tokens_list)

    # Get the unique tokens and their indices to build a map
    unique_tokens = jnp.unique(tokens_jnp)
    unique_tokens, unique_tokens_idx = jnp.unique(unique_tokens, return_index = True)

    # Create the dictionary mapping unique tokens to their indices
    token_to_index_dict = {int(unique_tokens[i]): int(unique_tokens_idx[i]) for i in range(len(unique_tokens))}
    index_to_token_dict = {int(unique_tokens_idx[i]): int(unique_tokens[i]) for i in range(len(unique_tokens))}

    def custom_encoder(x):
        return token_to_index_dict.get(x, None)  # Returns None if x is not in the dictionary

    def custom_decoder(x):
        return index_to_token_dict.get(x, None)  # Returns None if x is not in the dictionary

    def custom_token_list_encoding(tokens):
        tokens_new = []
        for i in range(len(tokens)):
            tokens_new.append(custom_encoder(tokens[i]))
        return tokens_new

    def custom_token_list_decoder(tokens):
        tokens_new = []
        for i in range(len(tokens)):
            tokens_new.append(custom_decoder(tokens[i]))
        return tokens_new

    num_unique_tokens = len(unique_tokens)
    print(f"Number of unique tokens: {num_unique_tokens}")

    def permute_text_vocabulary(text, key):
        # Split the text into a list of words or characters
        words = text.split()

        # Get the unique vocabulary
        unique_words = list(set(words))

        # Generate a random permutation of indices
        permuted_indices = jr.permutation(key, jnp.arange(len(unique_words)))

        # Create a mapping from original words to permuted counterparts using the permuted indices
        word_map = {original: unique_words[idx] for original, idx in zip(unique_words, permuted_indices)}

        # Map the original text to the permuted vocabulary
        permuted_text = ' '.join(word_map[word] for word in words)

        return permuted_text
    
    # B: batch size
    # T: Context window
    # N: Number of Tokens per Task
    class DataLoaderPermuteText:
        def __init__(self, text, B, T, N, key):
            self.current_position = 0
            self.B = B
            self.T = T
            self.N = N

            enc = tiktoken.get_encoding("gpt2")

            text = process_text(text)
            text = ' ' + permute_text_vocabulary(text, key)

            # Number of batches per epoch
            # B = 8
            # T = 128
            # num_bpe = 32
            tokens_list = enc.encode(text)[0:N]
            
            # limited vocab tokens
            tokens_lv = custom_token_list_encoding(tokens_list)

            self.tokens = jnp.array(tokens_lv)
            print(f"loaded {len(self.tokens)} tokens in the datasets" )
            print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

        def next_batch(self):
            B,T = self.B, self.T
            buf = self.tokens[self.current_position:self.current_position+B*T+1]
            x,y = jnp.reshape(buf[:-1],(B,T)), jnp.reshape(buf[1:],(B,T))
            self.current_position += B*T
            if self.current_position + B*T+1 > len(self.tokens):
                self.current_position = 0
            return x,y
    
    return DataLoaderPermuteText   

def run_experiment_PS_factory_test_reset(alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq):
    
    # Get the data_loader_class
    data_loader_class = get_dataloader(text)

    # Initialize the random key
    random_key = jr.PRNGKey(seed)
    print(f"Initial key: {random_key}, Type: {type(random_key)}")
    random_key, split_key = jr.split(random_key)

    # TODO: change this if we want to play with the ModelConfig - maybe make it an argument
    config = ModelConfig(vocab_size=11387)
    train_state, train_step = get_transformer_methods(config, alg, alg_params, split_key)
    neuron_ages = init_neuron_ages(config)
    if alg == 'ART' or alg == 'ReDO' or alg == 'CBP':
        init_reset_state, reset_neurons = get_reset_methods(config, alg, alg_params)
        reset_state = init_reset_state(config, alg, alg_params)

    # Get the total number of training steps
    train_steps_per_task = (N * epochs) // (B * T)

    # initialize neuron_ages_array
    if save_neuron_ages:
        neuron_ages_array = np.zeros((3, int(train_steps_per_task * tasks), int(config.n_embd * 4)), dtype = np.uint32)

    # Iniitialize loss_array
    loss_array = np.zeros(train_steps_per_task * tasks)
    t = 0

    
    t0 = time.time()
    for task in range(tasks):
    
        if verbose:
            print(f"Task: {task}")

        # Split the random key
        random_key, split_key = jr.split(random_key)
        data_loader = data_loader_class(text=text, B=B, T=T, N=N, key=split_key)
        
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
            if alg == 'ART' or alg == 'ReDO' or alg == 'CBP':
                random_key, split_key = jr.split(random_key)
                train_state, reset_state, neuron_ages = reset_neurons(train_state, reset_state, neuron_ages, neuron_pre_activ, split_key)
            else:
                neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)  
            ###############
            ###############
            
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

        # Save the loss_array to a pickle file
        with open(loss_path, 'wb') as f:
            pickle.dump(loss_array, f)
            print(f"Saved loss_array to {loss_path}")

        if save_neuron_ages:
            # Save the neuron_ages_array to a pickle file
            with open(ages_path, 'wb') as f:
                pickle.dump(neuron_ages_array, f)
                print(f"Saved neuron_ages_array to {ages_path}")

    if save_neuron_ages:
        return loss_array, neuron_ages_array
    else:
        return loss_array