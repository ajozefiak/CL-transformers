import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])



epochs = 10
batches = 12

# Optimal choices according to h_param_sweep
if layers == 3:
    reset_percentile = 0.55
    reset_freq = 1 / (0.75 * batches * epochs)
    # The above are the results of the finer h_param_sweep and are correct
if layers == 12 and lr == 1e-3:
    reset_percentile = 0.6
    reset_freq = 1 / (8 * batches * epochs)
    # Need to still do a finer h_param_sweep since this is only the first pass
if layers == 12 and lr == 1e-4:
    reset_percentile = 0.98
    reset_freq = 1 / (4 * batches * epochs)
    
# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'SNR-V2'
alg_params = {'threshold': 10,
        'reset_percentile': reset_percentile,
        'reset_freq': reset_freq}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/pool001/jozefiak/CI_ViT/opt_h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reset_percentile_{reset_percentile}_reset_freq_{reset_freq}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates(config, alg, alg_params, seed, save_path, cluster, experiment_config)