import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])



epochs = 10
batches = 12


# 1 / (0.5 * batches * epochs) == 0.0166....

# Optimal choices according to h_param_sweep
if layers == 3:
    ReDO_threshold = 0.01
    ReDO_reset_freq = 0.03333333333333333
if layers == 12 and lr == 1e-3:
    ReDO_threshold = 0.02
    ReDO_reset_freq = 0.03333333333333333
if layers == 12 and lr == 1e-4:
    ReDO_threshold = 0.005
    ReDO_reset_freq = 0.03333333333333333

# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'ReDO'
alg_params = {'ReDO_threshold': ReDO_threshold,
            'ReDO_reset_freq': ReDO_reset_freq}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/nobackup1/jozefiak/CI_ViT/opt_h_param_sweep/{alg}_L{num_layers}_lr_{lr}_threshold_{ReDO_threshold}_reset_freq_{ReDO_reset_freq}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates_2(config, alg, alg_params, seed, save_path, cluster, experiment_config)