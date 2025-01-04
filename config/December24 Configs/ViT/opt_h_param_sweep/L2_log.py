import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])

# Optimal choices according to h_param_sweep
if layers == 3:
    reg_str = 1e-4
    # reg_str = 1e-3 is the correct one, but only slightly better
if layers == 12 and lr == 1e-3:
    reg_str = 1e-3
    # reg_str = 1e-4 is the correct one
if layers == 12 and lr == 1e-4:
    reg_str = 1e-5

# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'L2'
# No alg_params necessary for Vanilla algorithm
alg_params = {"reg_str": reg_str}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/pool001/jozefiak/CI_ViT/opt_h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reg_str_{reg_str}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates(config, alg, alg_params, seed, save_path, cluster, experiment_config)