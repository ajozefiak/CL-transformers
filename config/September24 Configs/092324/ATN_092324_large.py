import CL_transformers
import sys
from datasets import load_from_disk

seed = int(sys.argv[1])
print(f"Seed: {seed}")

# Load Dataset given the seed
ds_path = f'/nobackup1/jozefiak/CL/Data/ATN/256k_tokenized/256K_tokenized_{seed}.hf'
ds = load_from_disk(ds_path)


# Experiment Parameters
B = 8
T = 128+1
art_per_task = B * 32
tasks = 1000
epochs = 100
verbose = True
print_freq = int(200)

# Set config to False to use the default config
# First Pass
config = CL_transformers.ModelConfig(vocab_size = 50257, n_head = 2, n_layer = 1, n_embd = 32, n_neurons = 256)

# TODO: toggle save_neuron_ages, save_weights
save_results = True
save_neuron_ages = True
save_weights = True
save_weights_freq = 32*epochs*2

# CLUSTER SAVE_PATH_ROOT
save_path_root = f'/nobackup1/jozefiak/CL/Results/ATN/ATN_092324_large/seed_{seed}/'

# Run the experiment
reg_strs = [1e-4]
for alg in ['L2', 'ART-L2']:
  for reg_str in reg_strs:
    alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 100)),
                    'reg_str': reg_str}  
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_ATN(config, alg, alg_params, ds, B, T, art_per_task, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
