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


def select_two_rand_pubs(publications, key):
  key, subkey1, subkey2 = jax.random.split(key, 3)
  num_publications = len(publications)
  idx_1 = jax.random.randint(subkey1, shape=(1,), minval=0, maxval=num_publications)[0]
  # Ensure idx_2 is different from idx_1
  idx_2 = jax.random.randint(subkey2, shape=(1,), minval=0, maxval=num_publications)[0]
  while idx_2 == idx_1:
    key, subkey2 = jax.random.split(key)
    idx_2 = jax.random.randint(subkey2, shape=(1,), minval=0, maxval=num_publications)[0]
  return idx_1, idx_2

def get_next_task(data, publications, key, articles):
  key, split_key = jr.split(key)
  idx_0, idx_1 = select_two_rand_pubs(publications, key)
  pub_0 = publications[idx_0]
  pub_1 = publications[idx_1]
  ds_0 = data[pub_0].select(range(articles))
  ds_1 = data[pub_1].select(range(articles))
  ds_0 = ds_0.add_column("label", [0] * len(ds_0))
  ds_1 = ds_1.add_column("label", [1] * len(ds_0))
  union_ds = datasets.concatenate_datasets([ds_0, ds_1])
  union_ds = union_ds.with_format("jax", columns=["tokens_512", "label"])
  return union_ds
            