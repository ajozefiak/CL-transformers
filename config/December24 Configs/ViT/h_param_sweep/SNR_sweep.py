import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])



epochs = 10
batches = 12

# reset_percentiles = 1 - 0.01*2^i for i = 1,2,3,4,
reset_percentiles = [0.99,0.98,0.96,0.92,0.84,0.68]
# The reset frequency is in terms of i resets (in expectation) per task = epochs * batches
reset_freqs = [1 / (i * epochs * batches) for i in [1,2,4,8,16,32]]

# Run the experiment for each hyperparameter
for reset_percentile in reset_percentiles:
    for reset_freq in reset_freqs:       
    
        # config:
        num_layers = layers
        hidden_dim = int(num_layers * 16)
        n_neurons = int(4 * hidden_dim)
        config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

        alg = 'SNR'
        alg_params = {'threshold': 10,
              'reset_percentile': reset_percentile,
              'reset_freq': reset_freq}

        cluster = True
        # Use default experiment config, so we can leave this blank
        experiment_config = {}

        # TODO:
        save_path = f'/pool001/jozefiak/CI_ViT/h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reset_percentile_{reset_percentile}_reset_freq_{reset_freq}/seed_{seed}/'

        res = CL_transformers.run_CI_ViT_R1_reset_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config)