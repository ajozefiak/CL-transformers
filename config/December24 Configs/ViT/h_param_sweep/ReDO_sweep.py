import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])



epochs = 10
batches = 12

ReDO_thresholds = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
# The reset frequency is in terms of i resets (in expectation) per task = epochs * batches
ReDO_reset_freqs = [1 / (i * epochs * batches) for i in [0.25,0.5,1,2,4]]

# Run the experiment for each hyperparameter
for ReDO_threshold in ReDO_thresholds:
    for ReDO_reset_freq in ReDO_reset_freqs:       
    
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
        save_path = f'/pool001/jozefiak/CI_ViT/h_param_sweep/{alg}_L{num_layers}_lr_{lr}_threshold_{ReDO_threshold}_reset_freq_{ReDO_reset_freq}/seed_{seed}/'

        res = CL_transformers.run_CI_ViT_R1_reset_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config)