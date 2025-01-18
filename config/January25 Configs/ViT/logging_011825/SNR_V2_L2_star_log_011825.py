import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])
option = int(sys.argv[4])



epochs = 10
batches = 12

# Optimal choices according to h_param_sweep
# if layers == 3:
#     reg_str = 1e-3
#     reset_percentile = 0.84
#     reset_freq = 0.0020833333333333333
#     reset_freq = 1 / (4 * batches * epochs)
# if layers == 12 and lr == 1e-3:
#     reg_str = 1e-4
#     reset_percentile = 0.55
#     reset_freq = 0.016666666666666666
# if layers == 12 and lr == 1e-4:
#     reg_str = 1e-7
#     reset_percentile = 0.98
#     reset_freq = 0.00026041666666666666

if option == 1:
    reg_str = 1e-3
    reset_percentile = 0.8
    reset_freq = 0.016666666666666666
if option == 2:
    reg_str = 1e-4
    reset_percentile = 0.55
    reset_freq = 0.016666666666666666
if option == 3:
    reg_str = 0.5*1e-4
    reset_percentile = 0.55
    reset_freq = 0.016666666666666666
if option == 4:
    reg_str = 1e-5
    reset_percentile = 0.55
    reset_freq = 0.016666666666666666


    
# config:
num_layers = layers
hidden_dim = int(num_layers * 16)
n_neurons = int(4 * hidden_dim)
config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

alg = 'SNR-V2-L2*'
alg_name = 'SNR-V2-L2-star'
alg_params = {
        'reg_str': reg_str,
        'threshold': 10,
        'reset_percentile': reset_percentile,
        'reset_freq': reset_freq}

cluster = True
# Use default experiment config, so we can leave this blank
experiment_config = {}

# TODO:
save_path = f'/nobackup1/jozefiak/CI_ViT/opt_h_param_sweep/{alg_name}_L{num_layers}_lr_{lr}_reg_str_{reg_str}_reset_percentile_{reset_percentile}_reset_freq_{reset_freq}/seed_{seed}/'

res = CL_transformers.run_CI_ViT_R1_log_correlates_2(config, alg, alg_params, seed, save_path, cluster, experiment_config)