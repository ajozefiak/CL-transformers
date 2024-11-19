# 10/07/24
# We are running the final 500-task experiment from the paper that I submitted, however, the only change is that we have 
# fixed the use of layer norm so that we are using pre-LN correctly. The goal is to see if anything changes with this fix.
import sys
import CL_transformers

# COLAB CODE
# prompt: load the text /content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt
# with open('/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt', 'r') as f:
#   text = f.read()

# CLUSTER CODE
# Load text                                                                                                                                                                                               
with open('shakespeare_and_dickens.txt', 'r') as f:
  text = f.read()

# Scale params
scale = 256
# We increase model scale by doubling width and hence squaring the number of weights
width_factor = 16

# Experiment Parameters
B = 8
# Context window T = 128 + 1, to account for a buffer so as to shuffle (x,y) pairs correctly
T = 128+1


# 695 = ceil(32 * 256^0.74)
N = int(B * T * 1938)

# We reduce the problem size substantially
tasks = 50
epochs = 20

verbose = True
print_freq = int(epochs * 1938)

# vocab_size = 21013 for scale-256 of shakespeare_and_dickens.txt 
config = CL_transformers.ModelConfig(vocab_size = 21013, n_head = int(2 * width_factor), n_layer = 1, n_embd = 32, n_neurons = int(256 * width_factor), use_resid=True)

seed = int(sys.argv[1])
print(f"Seed: {seed}")
# CLUSTER SAVE_PATH_ROOT
save_path_root = f'/pool001/jozefiak/CL/Results/111924/PS_111924_scale_256_randomize/seed_{seed}/'

# TODO: toggle save_neuron_ages, save_weights
save_results = True
save_neuron_ages = True
save_weights = True
save_weights_freq = 1938
# Old save_weights_freq which I ran initially
# save_weights_freq = int((B * 1938) / (T-1))


reg_strs = [1e-4]
# for alg in ['L2', 'ART-L2', 'Vanilla', 'L2Init', 'ART']:
for alg in ['ART-L2']:
  for reg_str in reg_strs:
    alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 20)),
                    'reg_str': reg_str}  
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_PS_100724(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
