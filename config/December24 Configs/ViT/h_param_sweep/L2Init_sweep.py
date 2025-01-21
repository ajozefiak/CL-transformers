import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])
lr = float(sys.argv[3])

reg_strs = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

# Run the experiment for each learning rate
for reg_str in reg_strs:
    
    # config:
    num_layers = layers
    hidden_dim = int(num_layers * 16)
    config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, num_classes=2)

    alg = 'L2Init'
    # No alg_params necessary for Vanilla algorithm
    alg_params = {"reg_str": reg_str}

    cluster = True
    # Use default experiment config, so we can leave this blank
    experiment_config = {}

    # TODO:
    save_path = f'/pool001/jozefiak/CI_ViT/h_param_sweep/{alg}_L{num_layers}_lr_{lr}_reg_str_{reg_str}/seed_{seed}/'

    res = CL_transformers.run_CI_ViT_R1_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config)