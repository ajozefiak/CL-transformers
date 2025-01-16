import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])



epochs = 10
batches = 12

reg_strs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
reset_percentiles = [0.5,0.55,0.6,0.7,0.8]
# The reset frequency is in terms of i resets (in expectation) per task = epochs * batches
reset_freqs = [1 / (i * epochs * batches) for i in [0.5]]

# Run the experiment for each hyperparameter
for reg_str in reg_strs:
    for reset_percentile in reset_percentiles:
        for reset_freq in reset_freqs:       
    
        # config:
        num_layers = layers
        hidden_dim = int(num_layers * 16)
        n_neurons = int(4 * hidden_dim)
        config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, n_neurons=n_neurons, lr=lr, num_classes=2)

        alg = 'SNR-V2-L2'
        alg_params = {
              'reg_str': reg_str,
              'threshold': 10,
              'reset_percentile': reset_percentile,
              'reset_freq': reset_freq}

        cluster = True
        # Use default experiment config, so we can leave this blank
        experiment_config = {}

        # TODO:
        save_path = f'/nobackup1/jozefiak/CI_ViT/h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reg_str_{reg_str}_reset_percentile_{reset_percentile}_reset_freq_{reset_freq}/seed_{seed}/'

        res = CL_transformers.run_CI_ViT_R1_reset_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config)