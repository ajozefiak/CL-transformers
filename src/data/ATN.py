import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import pickle

def get_ATN_data(path):
    path = path = "/ds_tokenized_512.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


