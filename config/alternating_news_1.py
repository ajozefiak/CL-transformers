import CL_transformers

# import datasets
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np

#  This took 1-2 minutes to load.

datasets_CNN = []
datasets_People = []

path = '../..'

for year in [2016, 2017, 2018, 2019]:
    for i in range(1,13):
        ds_temp_CNN = load_from_disk(path + f'/Data/CNN_tokenized_129/CNN_{i}_{year}_tkn_129.hf')
        datasets_CNN.append(ds_temp_CNN)

        ds_temp_People = load_from_disk(path + f'/Data/People_tokenized_129/People_{i}_{year}_tkn_129.hf')
        datasets_People.append(ds_temp_People)

B = 16
T = 129
art_per_task = 1696
epochs = 20
months = 40
# This means no mixing 100-0, 0-100, 100-0, ...
percent_mixing = 1.0
seed = 0
save_neuron_ages = True
save_results = True
save_weights = True
save_weights_freq = 2120
save_path_root = '/Results/Fixed Experiments/Alternating News/100-0'
verbose = True
print_freq = 100


alg = 'Vanilla'
save_path = save_path_root + '/' + alg
alg_params = {}
res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

alg = 'ART'
alg_params = {'threshold': 16,
              'reset_freq': 0.0,
              'reset_percentile': 0.95}
save_path = save_path_root + '/ART_Fixed_16' 
res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

alg = 'ART'
alg_params = {'threshold': 1,
              'reset_freq': 0.0,
              'reset_percentile': 0.95}
save_path = save_path_root + '/ART_Fixed_1' 
res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

alg = 'ART'
alg_params = {'threshold': 16,
              'reset_freq': 1e-4,
              'reset_percentile': 0.99}
save_path = save_path_root + '/ART_Adaptive_16' 
res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

reg_strs = [1e-3, 1e-4, 1e-5]
for reg_str in reg_strs:
    alg = 'L2'
    alg_params = {'reg_str': reg_str}
    save_path = save_path_root + '/' + alg + '_' + str(reg_str)
    res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)

    alg = 'L2Init'
    alg_params = {'reg_str': reg_str}
    save_path = save_path_root + '/' + alg + '_' + str(reg_str)
    res = CL_transformers.run_experiment_Alt_CATN(alg, alg_params, datasets_CNN, datasets_People, B, T, art_per_task, epochs, months, percent_mixing, seed, save_neuron_ages, save_results, save_path, verbose, print_freq, save_weights, save_weights_freq)
