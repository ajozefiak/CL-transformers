import numpy as np
import pickle

X = np.ones((1000,1000))
path = "/nobackup1/jozefiak/CL/Results/PS/Test_Dir/"

with open(path + 'X.pkl', 'wb') as f:
   pickle.dump(neuron_ages_array, f)