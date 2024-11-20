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
with open('all_shakespeare.txt', 'r') as f:
  text = f.read()

# Scale params
scale = 256
# We increase model scale by doubling width and hence squaring the number of weights
width_factor = 16

n_head = 2 * width_factor
n_embd = 32 * width_factor
n_neurons = 256 * width_factor

# Experiment Parameters
B = 8
# Context window T = 128 + 1, to account for a buffer so as to shuffle (x,y) pairs correctly
T = 128+1


# batches = ceil(32 * 256^0.74)
batches = int(32 * (scale ** (0.74)))
N = int(B * T * batches)

tasks = 50
epochs = 100

verbose = True
print_freq = int(epochs * batches - 1)

# We use 256 neurons and run this experiment for 1000 tasks
# vocab_size = 21013 for scale-256 of shakespeare_and_dickens.txt 
config = CL_transformers.ModelConfig(vocab_size = 18539, n_head = int(2 * width_factor), n_layer = 1, n_embd = 32, n_neurons = int(256 * width_factor), use_resid=True)

seed = int(sys.argv[1])
print(f"Seed: {seed}")
# CLUSTER SAVE_PATH_ROOT
save_path_root = f'/pool001/jozefiak/CL/Results/112024/PS_112024_scale_256_epochs_100_randomize/seed_{seed}/'

# TODO: toggle save_neuron_ages, save_weights
save_results = True
save_neuron_ages = False
save_weights = True
save_weights_freq = int(epochs*batches)
# Old save_weights_freq which I ran initially
# save_weights_freq = int((B * batches) / (T-1))


reg_strs = [1e-4]
# for alg in ['L2', 'ART-L2', 'Vanilla', 'L2Init', 'ART']:
for alg in ['L2']:
  for reg_str in reg_strs:
    alg_params = {'threshold': 16,
                    'reset_percentile': 0.95,
                    'reset_freq': (1e-4 * (epochs / 20)),
                    'reg_str': reg_str}  
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_' + str(reg_str) + '/'
    res = CL_transformers.run_experiment_PS_112024(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
