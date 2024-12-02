# 10/07/24
# We are running the final 500-task experiment from the paper that I submitted, however, the only change is that we have 
# fixed the use of layer norm so that we are using pre-LN correctly. The goal is to see if anything changes with this fix.
import sys
import math
import CL_transformers

# Check arguments
if len(sys.argv) != 7:
    print("Usage: python3 PS_112024_Scale_X_epochs_Y_randomize_ALG.py <seed> <scale> <epochs> <task> <alg> <lr>")
    sys.exit(1)

# Read arguments
seed = int(sys.argv[1])
scale = int(sys.argv[2])
epochs = int(sys.argv[3])
tasks = int(sys.argv[4])
alg = sys.argv[5]
lr = float(sys.argv[6])

print(f"Seed: {seed}, Scale: {scale}, Epochs: {epochs}, Algorithm: {alg}")

# COLAB CODE
# prompt: load the text /content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt
# with open('/content/drive/MyDrive/CL LLM/Louis Wang Tutorial/input.txt', 'r') as f:
#   text = f.read()

# CLUSTER CODE
# Load text                                                                                                                                                                                               
with open('all_shakespeare.txt', 'r') as f:
  text = f.read()

# Scale params
# scale = 256
# We increase model scale by doubling width and hence squaring the number of weights
width_factor = int(round(math.sqrt(scale)))

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

# tasks = 500
# epochs = 50

verbose = True
print_freq = int(epochs * batches - 1)

# We use 256 neurons and run this experiment for 1000 tasks
# vocab_size = 21013 for scale-256 of shakespeare_and_dickens.txt 
config = CL_transformers.ModelConfig(vocab_size = 18539, n_head = n_head, n_layer = 1, n_embd = n_embd, n_neurons = n_neurons, use_resid=True)

seed = int(sys.argv[1])
print(f"Seed: {seed}")
# CLUSTER SAVE_PATH_ROOT
save_path_root = f'/pool001/jozefiak/CL/Results/120124_General/PS_120124_scale_{scale}_epochs_{epochs}_randomize/seed_{seed}/'

# TODO: toggle save_neuron_ages, save_weights
save_results = True
save_neuron_ages = True
save_weights = True
save_weights_freq = int(epochs*batches)
# Old save_weights_freq which I ran initially
# save_weights_freq = int((B * batches) / (T-1))


reg_strs = [1e-4 / width_factor]
# lr = 1e-3
lr = lr
# NOTE: These ReDO hyperparams are from a hyperparam sweep on the scale-1 10-task problem over 8 seeds, 
# and choosing the hyperparams that achieve the smallest terminal (last epoch) loss over the last 5 tasks.
ReDO_reset_freq = 1 / (4 * epochs * batches)
ReDO_threshold = 0.01
CBP_reset_freq = 1e-6

# for alg in ['L2', 'ART-L2', 'Vanilla', 'L2Init', 'ART', 'ReDO-L2', 'ART-L2*', 'L2*']:

for reg_str in reg_strs:
  alg_params = {'threshold': 16,
                  'reset_percentile': 0.95,
                  'reset_freq': (1e-4 * (epochs / 20)),
                  'reg_str': reg_str,
                  'lr': lr,
                  'ReDO_reset_freq': ReDO_reset_freq,
                  'ReDO_threshold': ReDO_threshold,
                  'CBP_reset_freq': CBP_reset_freq}  
  save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_lr_' + str(lr) + '/'
  if alg == 'ReDO-L2':
    save_path = save_path_root + f'/epochs_{epochs}/' + alg + '_lr_' + str(lr) + '/'
  res = CL_transformers.run_experiment_PS_112024(config, alg, alg_params, text, B, T, N, epochs, tasks, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
