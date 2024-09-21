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
tasks = 200
epochs = 100
verbose = True
print_freq = int(200)

# Fourth Pass: Increase Task Count to 200, use two heads with n_embd = 32, n_neurons = 256
# Here we run this with the residual connections as an ablation
config = CL_transformers.ModelConfig(vocab_size = 11387, n_head = 2, n_layer = 1, n_embd = 32, n_neurons = 256, use_resid=True)

seed = int(sys.argv[1])
print(f"Seed: {seed}")
# CLUSTER SAVE_PATH_ROOT
save_path_root = f'Results/PS_092024_constrained_5/seed_{seed}/'

# TODO: toggle save_neuron_ages, save_weights
save_results = True
if seed == 1:
    save_neuron_ages = True
    save_weights = True
else:
    save_neuron_ages = False
    save_weights = False
save_weights_freq = 3200




# No Intervention
alg = 'Vanilla'
alg_params = {'threshold': 16,
              'reset_percentile': 0.95,
              'reset_freq': (1e-4 * (epochs / 100))}  

save_path = save_path_root + f'/epochs_{epochs}/' + alg + '/'
res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)


# Basic Resets
alg = 'ART'
alg_params = {'threshold': 16,
              'reset_percentile': 0.95,
              'reset_freq': (1e-4 * (epochs / 100))}  

save_path = save_path_root + f'/epochs_{epochs}/' + alg + '/'
res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

reg_strs = [1e-3, 1e-4, 1e-5, 1e-6]
for alg in ['L2', 'ART-L2*', 'ART-L2']:
  for reg_str in reg_strs:
    alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 100)),
                    'reg_str': reg_str}  
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_PS_factory_test_reset(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
