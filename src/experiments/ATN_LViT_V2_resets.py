# TODO Immports
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np
# TODO: Check that datasets loads
import datasets

from ..models import *
from ..data import *

# Continual All the News (C-ATN) problem
# Version 2 (V2): We not load the (nearly) full dataset and can sample much more articles per task
# We may drop the Washington Post and the New Yorker due limited articles for these two publications, leaving us with 26 publications
# This is currently a binary problem, but we could turn it into a 24 (26) dimensioanl prediciton problem, not sure which is better.
def run_ATN_LViT_V2_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config={}):
    
    # Default Experimental Parameters
    batch_size = 16
    articles = batch_size*320 
    # articles = 5120
    epochs = 1
    tasks = 500

    # Extract Experiment Config Data
    if 'batch_size' in experiment_config:
        batch_size = experiment_config['batch_size']
    if 'images_per_class' in experiment_config:
        images_per_class = experiment_config['images_per_class']
    if 'epochs' in experiment_config:
        epochs = experiment_config['epochs']
    if 'tasks' in experiment_config:
        tasks = experiment_config['tasks']
    
    # Logging of:
    # test_acc.pkl
    # train_acc.pkl
    # train_loss.pkl
    train_loss = []
    train_acc = []
    test_acc = []

    # Get random keys
    key = jr.PRNGKey(seed)
    task_key = jr.PRNGKey(seed)
    
    # Get the state, and train_step, accuracy functions
    key, split_key = jr.split(key)
    state, train_step, accuracy = get_LViT_methods(config, alg, alg_params, split_key)
    
    # Initialize data structures associated with resets and neuron_ages
    update_neuron_ages, get_neuron_pre_activ = get_neurons_ages_functions_ViT(config)
    neuron_ages = init_neuron_ages_ViT(config)
    if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO' or alg == 'SNR-V2' or alg == 'SNR-L2' or alg == 'SNR-V2-L2' or alg == 'SNR-L2*' or alg == 'SNR-V2-L2*':
        init_reset_state, reset_neurons = get_reset_methods_ViT(config, alg, alg_params)
        reset_state = init_reset_state(config, alg, alg_params)

    # Load the (nearly) full ATN Dataset
    publications = ['Vox',
                    'Business Insider',
                    'Reuters',
                    'TMZ',
                    'Vice',
                    'Vice News',
                    'Hyperallergic',
                    'TechCrunch',
                    'Axios',
                    'Refinery 29',
                    'The Verge',
                    'Mashable',
                    'People',
                    'Economist',
                    'CNN',
                    'Gizmodo',
                    'Wired',
                    'CNBC',
                    'New Republic',
                    'Fox News',
                    'The Hill',
                    'Politico',
                    'The New York Times',
                    'Buzzfeed News'
                    #  'New Yorker',               
                    #  'Washington Post'
                    ]
    if cluster:
        ATN_dir = '/home/jozefiak/CL/Experiments/PS_fixed_111924/ATN_dataset_full/'
    else:
        ATN_dir = '/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/Late August/Process All the News/Data/tokenize_all_512/'
    data = get_ATN_data_full(ATN_dir, publications)
    

    for task in range(tasks):
        task_key, task_split_key = jr.split(task_key)
        data_task, labels_task = get_next_task_full(data, publications, task_split_key, articles)

        for epoch in range(epochs):
            task_key, task_split_key = jr.split(task_key)
            data_shuffled, labels_shuffled = shuffle_task(data_task, labels_task, task_split_key)
    
            dataset_length = len(data_shuffled)

            for start_idx in range(0, dataset_length, batch_size):
                
                # start_idx
                end_idx = start_idx + batch_size

                x_batch = data_shuffled[start_idx:end_idx]
                y_batch = labels_shuffled[start_idx:end_idx]                


                ###############
                # COMMON TRAINING LOOP:
                # (1) maps x,y -> (new) train_state (and reset_state)
                # (2) outputs loss and neuron_ages for logging purposes
                # In practice, it should be the case that this code can be copied to every experiment
                ###############
                neuron_pre_activ = get_neuron_pre_activ(state, x_batch)

                train_accuracy = accuracy(state, x_batch, y_batch)
                key, split_key = jr.split(key)
                loss, state = train_step(state, x_batch, y_batch, split_key)

                # Perform reset step and 
                if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO' or alg == 'SNR-V2' or alg == 'SNR-L2' or alg == 'SNR-V2-L2' or alg == 'SNR-L2*' or alg == 'SNR-V2-L2*':
                    key, split_key = jr.split(key)
                    state, reset_state, neuron_ages, reset_mask = reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, split_key)
                else:
                    neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)  
                ###############
                ###############
                
                # Log training loss
                train_loss.append(loss)

                # Log training accuracy
                train_acc.append(train_accuracy)


            # Log test accuracy at the end of the epoch
            # TODO: We need to extractt a test set for each class
            # test_accuracy = accuracy(state, X_test_, y_test_)
            # test_acc.append(test_accuracy)


    # TODO: Save the results
    # Check if the path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Define file names for the data
    train_loss_path = os.path.join(save_path, "train_loss.pkl")
    train_acc_path = os.path.join(save_path, "train_acc.pkl")
    test_acc_path = os.path.join(save_path, "test_acc.pkl")

    # Save the data
    with open(train_loss_path, 'wb') as f:
        pickle.dump(train_loss, f)

    with open(train_acc_path, 'wb') as f:
        pickle.dump(train_acc, f)

    with open(test_acc_path, 'wb') as f:
        pickle.dump(test_acc, f)

    return train_loss, train_acc, test_acc