import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import pickle
import datasets
from datasets import load_from_disk

def get_ATN_data(path):
    path = path + "/ds_tokenized_512.pkl"
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


publications = ['Vox',
 'Business Insider',
 'Reuters',
 'TMZ',
 'Vice',
 'Vice News',
 'Hyperallergic',
 'TechCrunch',
 'Axios',
 'Refinery 29',
 'The Verge',
 'Mashable',
 'People',
 'Economist',
 'CNN',
 'Gizmodo',
 'Wired',
 'CNBC',
 'New Republic',
 'Fox News',
 'The Hill',
 'Politico',
 'The New York Times',
 'Buzzfeed News'
#  'New Yorker',               
#  'Washington Post'
 ]

# New implementation that uses (nearly) the full dataset

def get_ATN_data_full(path, publications):
    data = []
    for pub in publications:
        data.append(datasets.load_from_disk(path + f'{pub}.hf'))
    return data

# OLD Implementation, before 01/24/25:
# Uses too many hugging face datasets objects and causes too much disk space to be consumed

# def get_next_task_full(data, publications, key, articles):
#   # Sample two random indices
#   key, split_key = jr.split(key)
#   idx_0, idx_1 = select_two_rand_pubs(publications, split_key)

#   # shuflfe datasets and get exact number of articles from each dataset
#   ds_0 = data[idx_0].shuffle(seed=int(split_key[0])).select(range(articles))
#   key, split_key = jr.split(key)
#   ds_1 = data[idx_1].shuffle(seed=int(split_key[0])).select(range(articles))

#   # Add labels
#   ds_0 = ds_0.add_column("label", np.zeros(len(ds_0), dtype=int))
#   ds_1 = ds_1.add_column("label", np.ones(len(ds_1), dtype=int))

#   union_ds = datasets.concatenate_datasets([ds_0, ds_1])
#   union_ds = union_ds.with_format("jax", columns=["tokens_512", "label"])

#   return union_ds

import numpy as np

def get_next_task_full(data, publications, key, articles):
  # Sample two random indices
  key, split_key = jr.split(key)
  idx_0, idx_1 = select_two_rand_pubs(publications, key)

  # shuflfe datasets and get exact number of articles from each dataset
  idxs = np.array(range(len(data[idx_0])))
  key, split_key = jr.split(key)
  idxs = jr.permutation(split_key, idxs)
  idxs = idxs[0:articles]
  data_0 = data[idx_0][idxs]['tokens_512']

  idxs = np.array(range(len(data[idx_1])))
  key, split_key = jr.split(key)
  idxs = jr.permutation(split_key, idxs)
  idxs = idxs[0:articles]
  data_1 = data[idx_1][idxs]['tokens_512']

  # Add labels
  labels = [0] * articles + [1] * articles
  
  data_union = data_0 + data_1

  return data_union, labels

# Ideally, take in as input data_union
def shuffle_task(data, labels, key):
    permutation = jr.permutation(key, len(data))
    data = np.array(data)[permutation]
    labels = np.array(labels)[permutation]
    return jnp.array(data), jnp.array(labels)


            