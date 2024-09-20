# 09/19/24
# We run the adaptive resets for 100 epochs on PS 
import sys
import CL_transformers

# COLAB CODE
# prompt: load the text /content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt
# with open('/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt', 'r') as f:
#   text = f.read()

# CLUSTER CODE
# Load text                                                                                                                                                                                               
with open('shakespeare.txt', 'r') as f:
  text = f.read()


# Experiment Parameters
B = 8
T = 128
N = int(B * T * 32)
tasks = 50
epochs = 100
verbose = True
print_freq = int(200)

# Set config to False so that we use the default config
config = False

# TODO: toggle save_neuron_ages, save_weights
save_results = True
save_neuron_ages = True
save_weights = True
save_weights_freq = 1e-4


seed = int(sys.argv[1])
print(f"Seed: {seed}")
# CLUSTER SAVE_PATH_ROOT
save_path_root = f'Results/PS_092024/seed_{seed}/'

# No Intervention
alg = 'Vanilla'
alg_params = {}

save_path = save_path_root + f'/epochs_{epochs}/' + alg + '/'
res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

reg_strs = [1e-4, 1e-5, 1e-6]
for reg_str in reg_strs:
    alg = 'L2'
    alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 100)),
                    'reg_str': reg_str}  
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
    
    alg = 'ART-L2*'
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
