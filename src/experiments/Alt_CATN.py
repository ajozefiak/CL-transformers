# TODO
import datasets
import tiktoken
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np

# TODO verify that this works
from ..models import *

# def concatenate_two_ds(ds1, ds2, col, art_per_task_1, art_per_task_2):
#     idx = int(art_per_task/2)
#     task1_tokens = np.concatenate(ds1[col][:art_per_task_1] + ds2[col][:idx])
#     task2_tokens = np.concatenate(ds1[col][idx:art_per_task] + ds2[col][idx:art_per_task])
#     return task1_tokens, task2_tokens

# art_per_task is how many articles a task has
# percent_mixing determines how we mix articles families
# for instance percent_mixing = 0.7 => 70-30, 30-70, 70-30, 30-70, ... splits of families A and B
def create_two_tasks(ds1, ds2, col, art_per_task, percent_mixing):
    idx = int(percent_mixing * art_per_task)
    # task1_tokens = ds1[col][:idx] + ds2[col][:idx]
    # task2_tokens = ds1[col][idx:art_per_task] + ds2[col][idx:art_per_task]
    
    task1_tokens = ds1[col][:idx] + ds2[col][idx:art_per_task]
    task2_tokens = ds1[col][idx:art_per_task] + ds2[col][:idx]
    return task1_tokens, task2_tokens