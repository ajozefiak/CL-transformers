# 09/18/24
# In this experiment we run through Permuted Shakespare, so far the only experiment that is showing promising results
# after having fixed my code. 
# Here I am iterating over epochs 25, 50, 100, 200, 
# I am also testing regularization strengths: 1e-4, 1e-5, 1e-6 for L2 and L2Init
# I have also created an ablation of ART + L2
# We run the experiment for 50 tasks, 32 batches per task, batch size 8, learning rate 1e-3 with Adam

import CL_transformers

# COLAB CODE
# prompt: load the text /content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt
# with open('/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt', 'r') as f:
#   text = f.read()

# CLUSTER CODE
# Load text                                                                                                                                                                                               
with open('shakespeare.txt', 'r') as f:
  text = f.read()


B = 8
T = 128
N = int(B * T * 32)
tasks = 50
seed = int(0)
save_neuron_ages = True
save_results = True
# CLUSTER SAVE_PATH_ROOT
save_path_root = 'Results/PS_091824/'
verbose = True
print_freq = int(200)
save_weights = True
# Set config to False so that we use the default config
config = False

epochs_list = [100, 200, 50, 20]
for epochs in epochs_list:

    save_weights_freq = int(32 * epochs)

    # No Intervention
    alg = 'Vanilla'
    alg_params = {}

    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

    # Resets Adaptive
    alg = 'ART'
    alg_params = {'threshold': 16,
                'reset_percentile': 0.95,
                'reset_freq': (1e-4 * (epochs / 100))}
    save_path = save_path_root + f'/epochs_{epochs}/ART_adaptive/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

    # Resets Fixed
    alg = 'ART'
    alg_params = {'threshold': 16,
                'reset_percentile': 0.95,
                'reset_freq': 0.0}
    save_path = save_path_root + f'/epochs_{epochs}/ART_fixed_16/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

    reg_strs = [1e-4, 1e-5, 1e-6]
    for alg in ['ART-L2', 'L2', 'L2Init']:
        for reg_str in reg_strs:
        alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 100)),
                    'reg_str': reg_str}  
        alg_params = {'reg_str': reg_str}
        save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
        res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)