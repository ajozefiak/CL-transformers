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
def run_CI_ViT_R1_log_correlates_2(config, alg, alg_params, seed, save_path, cluster, experiment_config={}):
    
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

    # T = number of time steps
    T = int(tasks * epochs * (2 * images_per_class / batch_size))
    T_test = int(tasks * epochs)
    num_layers = config.num_layers
    n_neurons = config.n_neurons

    # Logging of:
    # test_acc.pkl
    # train_acc.pkl
    # train_loss.pkl
    train_loss = []
    train_acc = []
    test_acc = []
    neuron_ages_array = np.zeros((T, num_layers, n_neurons), dtype=np.uint32)
    neuron_ages_array_2 = np.zeros((T, num_layers, n_neurons))
    resets_array = np.zeros((T, num_layers, n_neurons), dtype=bool)
    # TODO: store entropies of self attention layers -> compute these as a single average value at each time step or test set or both
    mean_entropies_train_array = np.zeros((T, num_layers))
    std_entropies_train_array = np.zeros((T, num_layers))
    mean_entropies_test_array = np.zeros((T_test, num_layers))
    std_entropies_test_array = np.zeros((T_test, num_layers))
    # DONE: store weight norms, for each weight matrix of the network
    param_norms = []

    # Get random keys
    key = jr.PRNGKey(seed)
    task_key = jr.PRNGKey(seed)
    
    # Get the state, and train_step, accuracy functions
    key, split_key = jr.split(key)
    state, train_step, accuracy = get_ViT_methods(config, alg, alg_params, split_key)
    
    # Initialize data structures associated with resets and neuron_ages
    update_neuron_ages, get_neuron_pre_activ, update_neuron_ages_2 = get_neurons_ages_functions_ViT_2(config)
    neuron_ages = init_neuron_ages_ViT(config)

    neuron_ages_log = init_neuron_ages_ViT(config)
    neuron_ages_log_2 = init_neuron_ages_ViT_2(config)

    if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO' or alg == 'SNR-V2' or alg == 'SNR-L2' or alg == 'SNR-V2-L2':
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

    # For logging purposes: t_train and t_test
    # t_train: current training step
    # t_test: current test eet evaluation
    t_train = 0
    t_test = 0

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
                neuron_pre_activ = get_neuron_pre_activ(state, x_batch)

                key, split_key = jr.split(key)
                loss, state = train_step(state, x_batch, y_batch, split_key)

                # Perform reset step and 
                if alg == 'SNR' or alg == 'CBP' or alg == 'ReDO' or alg == 'SNR-V2' or alg == 'SNR-L2' or alg == 'SNR-V2-L2':
                    key, split_key = jr.split(key)state, reset_state, neuron_ages, reset_mask = reset_neurons(state, reset_state, neuron_ages, neuron_pre_activ, split_key)
                    
                    # Store reset_mask in reset_mask_array
                    # TODO: Make this programatic, perhaps as a function, so that we do not need to do this manually
                    # if save_neuron_ages:
                    #     reset_mask_array[0,t,:] = reset_mask['Block_0']
                        # reset_mask_array[1,t,:] = reset_mask['Block_1']
                        # reset_mask_array[2,t,:] = reset_mask['Block_2']
                # if (alg != 'SNR' and alg != 'SNR-V2' and alg != 'SNR-L2' and alg != 'SNR-V2-L2'):
                neuron_ages_log = update_neuron_ages(neuron_ages_log, neuron_pre_activ)
                neuron_ages_log_2 = update_neuron_ages_2(neuron_ages_log_2, neuron_pre_activ)
                ###############
                ###############
                
                # Log training loss
                train_loss.append(loss)

                # Log training accuracy
                train_accuracy = accuracy(state, x_batch, y_batch)
                train_acc.append(train_accuracy)

                # Log neuron_ages
                for l in range(num_layers):
                    neuron_ages_array[t_train,l,:] = neuron_ages_log[f'ViTBlock_{l}']
                    neuron_ages_array_2[t_train,l,:] = neuron_ages_log_2[f'ViTBlock_{l}']

                # Log reset_mask
                # Check if reset_mask is defined, if not, do not log
                if 'reset_mask' in locals():
                    for l in range(num_layers):
                        resets_array[t_train,l,:] = reset_mask[f'ViTBlock_{l}']

                # Log training entropies
                for l in range(num_layers):
                    dists = neuron_pre_activ['intermediates'][f'ViTBlock_{l}']['ViTSelfAttention_0']['probs'][0]
                    entropies = get_entropies(dists)
                    mean_entropies_train_array[t_train,l] = jnp.mean(entropies)
                    std_entropies_train_array[t_train,l] = jnp.std(entropies)

                # Incremenet t_train
                t_train += 1


            # Log test accuracy at the end of the epoch
            test_accuracy = accuracy(state, X_test_, y_test_)
            test_acc.append(test_accuracy)

            # Log parameter norms (after every epoch)
            param_norms.append(get_kernel_norms_flat(state.params))

            # Log test entropies
            neuron_pre_activ_test = get_neuron_pre_activ(state, X_test_)
            for l in range(num_layers):
                dists = neuron_pre_activ_test['intermediates'][f'ViTBlock_{l}']['ViTSelfAttention_0']['probs'][0]
                entropies = get_entropies(dists)
                mean_entropies_test_array[t_test,l] = jnp.mean(entropies)
                std_entropies_test_array[t_test,l] = jnp.std(entropies)

            # Incremenet t_test
            t_test += 1


    # TODO: Save the results
    # Check if the path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Define file names for the data
    train_loss_path = os.path.join(save_path, "train_loss.pkl")
    train_acc_path = os.path.join(save_path, "train_acc.pkl")
    test_acc_path = os.path.join(save_path, "test_acc.pkl")
    neuron_ages_array_path = os.path.join(save_path, "neuron_ages_array.pkl")
    neuron_ages_array_path_2 = os.path.join(save_path, "neuron_ages_array_2.pkl")
    resets_array_path = os.path.join(save_path, "resets_array.pkl")
    param_norms_path = os.path.join(save_path, "param_norms.pkl")
    mean_entropies_train_array_path = os.path.join(save_path, "mean_entropies_train_array.pkl")
    std_entropies_train_array_path = os.path.join(save_path, "std_entropies_train_array.pkl")
    mean_entropies_test_array_path = os.path.join(save_path, "mean_entropies_test_array.pkl")
    std_entropies_test_array_path = os.path.join(save_path, "std_entropies_test_array.pkl")

    # Save the data
    with open(train_loss_path, 'wb') as f:
        pickle.dump(train_loss, f)

    with open(train_acc_path, 'wb') as f:
        pickle.dump(train_acc, f)

    with open(test_acc_path, 'wb') as f:
        pickle.dump(test_acc, f)

    with open(neuron_ages_array_path, 'wb') as f:
        pickle.dump(neuron_ages_array, f)

    with open(neuron_ages_array_path_2, 'wb') as f:
        pickle.dump(neuron_ages_array_2, f)

    with open(resets_array_path, 'wb') as f:
        pickle.dump(resets_array, f)

    with open(param_norms_path, 'wb') as f:
        pickle.dump(param_norms, f)

    with open(mean_entropies_train_array_path, 'wb') as f:
        pickle.dump(mean_entropies_train_array, f)

    with open(std_entropies_train_array_path, 'wb') as f:
        pickle.dump(std_entropies_train_array, f)

    with open(mean_entropies_test_array_path, 'wb') as f:
        pickle.dump(mean_entropies_test_array, f)

    with open(std_entropies_test_array_path, 'wb') as f:
        pickle.dump(std_entropies_test_array, f)

    return train_loss, train_acc, test_acc