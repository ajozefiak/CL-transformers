import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])

# Optimal choices according to h_param_sweep
if layers == 3:
    p = 0.99
    sigma = 1e-2
    # p = 0.99, sigma = 1e-2 is correct for L3 @ 50 tasks
if layers == 12 and lr == 1e-3:
    p = 0.7
    sigma = 1e-1
    # p = 0.99 is better for the last 10 tasks
if layers == 12 and lr == 1e-4:
    p = 0.99
    sigma = 1e-2
    # p = 0.975 is better for the last 10 tasks
        
# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'S&P'
alg_params = {"p": p,
                "sigma": sigma}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/pool001/jozefiak/CI_ViT/opt_h_param_sweep/{alg}_L{num_layers}_lr_{lr}_p_{p}_sigma_{sigma}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates(config, alg, alg_params, seed, save_path, cluster, experiment_config)