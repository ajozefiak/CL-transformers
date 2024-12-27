# TODO Immports
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np

from ..models import *
from ..data import *

# Implementation of CI-R1 experiment with additional logic for resets
# i.e. running through the dataset once
# save_path is the base directory in which we save the results
def run_CI_ViT_R1_reset_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config={}):
    
    # Default Experimental Parameters
    batch_size = 100
    images_per_class = 600
    epochs = 10
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
    state, train_step, accuracy = get_ViT_methods(config, alg, alg_params, split_key)
    train_state, train_step = get_transformer_methods(config, alg, alg_params, split_key)
    
    # Initialize data structures associated with resets and neuron_ages
    update_neuron_ages, get_neuron_pre_activ = get_neurons_ages_functions_ViT(config)
    neuron_ages = init_neuron_ages_ViT(config)
    if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO':
        init_reset_state, reset_neurons = get_reset_methods_ViT(config, alg, alg_params)
        reset_state = init_reset_state(config, alg, alg_params)

    # Load the ImageNet-32 Dataset
    if cluster:
        CI_dir = '/home/jozefiak/CL/Experiments/PS_fixed_111924/ImageNet_dataset/'
    else:
        CI_dir = '/content/drive/MyDrive/ML_Data/'
    # Takes 51 second to load
    X_train, y_train, X_test, y_test = load_Imagenet32(CI_dir)

    # Generate random partition of classes into two families
    task_key, task_split_key = jr.split(task_key)
    label_partition_permutation = jr.permutation(task_split_key, 1000)
    family_0 = label_partition_permutation[0:500]
    family_1 = label_partition_permutation[500:1000]


    # TODO: For now we have a fixed count of 500 tasks in the CI-R1 experiment
    for task in range(tasks):
        X_train_, y_train_, X_test_, y_test_ = get_next_task_CI32(X_train, y_train, X_test, y_test, task, family_0, family_1, images_per_class)

        for epoch in range(epochs):
            task_key, task_split_key = jr.split(task_key)
            X_train_, y_train_ = permute_image_order(X_train_, y_train_, task_split_key)

            for batch_start in range(0, X_train_.shape[0], batch_size):
                batch_end = batch_start + batch_size

                x_batch = X_train_[batch_start:batch_end]
                y_batch = y_train_[batch_start:batch_end]

                ###############
                # COMMON TRAINING LOOP:
                # (1) maps x,y -> (new) train_state (and reset_state)
                # (2) outputs loss and neuron_ages for logging purposes
                # In practice, it should be the case that this code can be copied to every experiment
                ###############
                neuron_pre_activ = get_neuron_pre_activ(state, x)

                key, split_key = jr.split(key)
                loss, state = train_step(state, x_batch, y_batch, split_key)

                # Perform reset step and 
                if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO':
                    key, split_key = jr.split(key)
                    state, reset_state, neuron_ages, reset_mask = reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, split_key)
                    # Store reset_mask in reset_mask_array
                    # if save_neuron_ages:
                    #     reset_mask_array[0,t,:] = reset_mask['Block_0']
                        # reset_mask_array[1,t,:] = reset_mask['Block_1']
                        # reset_mask_array[2,t,:] = reset_mask['Block_2']
                else:
                    neuron_ages = update_neuron_ages(neuron_ages, neuron_pre_activ)  
                ###############
                ###############
                
                # Log training loss
                train_loss.append(loss)

                # Log training accuracy
                train_accuracy = accuracy(state, x_batch, y_batch)
                train_acc.append(train_accuracy)


            # Log test accuracy at the end of the epoch
            test_accuracy = accuracy(state, X_test_, y_test_)
            test_acc.append(test_accuracy)


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