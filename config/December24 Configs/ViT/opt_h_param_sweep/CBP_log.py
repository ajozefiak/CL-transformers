import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])

# Optimal choice for L3 lr=1e-3, L12 lr=1e-3, and L12 lr=1e-4 according to h_param_sweep
reset_freq = 1e-2

# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'CBP'
alg_params = {"CBP_reset_freq": reset_freq}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/pool001/jozefiak/CI_ViT/opt_h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reset_freq_{reset_freq}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates(config, alg, alg_params, seed, save_path, cluster, experiment_config)