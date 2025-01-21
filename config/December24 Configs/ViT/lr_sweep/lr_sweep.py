import sys
import math
import CL_transformers

# Read arguments
seed = int(sys.argv[1])
layers = int(sys.argv[2])

learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]

# Run the experiment for each learning rate
for lr in learning_rates:
    
    # config:
    num_layers = layers
    hidden_dim = int(num_layers * 16)
    config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, num_classes=2)

    alg = 'Vanilla'
    # No alg_params necessary for Vanilla algorithm
    alg_params = {}

    cluster = True
    # Use default experiment config, so we can leave this blank
    experiment_config = {}

    # TODO:
    save_path = f'/pool001/jozefiak/CI_ViT/lr_sweep/{alg}_L{num_layers}_lr_{lr}/seed_{seed}/'

    res = CL_transformers.run_CI_ViT_R1_experiment(config, alg, alg_params, seed, save_path, cluster, experiment_config)