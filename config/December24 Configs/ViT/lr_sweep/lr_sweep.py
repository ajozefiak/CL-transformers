# config:
num_layers = 3
hidden_dim = int(num_layers * 16)
lr = 1e-3
print(f"Number of layers: {num_layers}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Learning rate: {lr}")

config = CL_transformers.ModelConfigViT(num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, num_classes=2)
alg = 'Vanilla'
alg_params = {}
seed = 0

cluster = True
# Use default experiment config
experiment_config = {}

# TODO:
save_path = '/content/drive/MyDrive/CL LLM/Test ViT/Test_pkg/Vanilla/seed_0/'

res = CL_transformers.run_CI_ViT_R1_experiment(config, alg, alg_params, seed, save_path, cluster)