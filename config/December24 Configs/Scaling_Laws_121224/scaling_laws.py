import sys
import math
import CL_transformers

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np
import optax

# Check arguments
if len(sys.argv) != 10:
    print("Not enough arguments")
    sys.exit()

seed = int(sys.argv[1])
n_embd = int(sys.argv[2])
n_head = int(sys.argv[3])
n_layer = int(sys.argv[4])
D = int(float(sys.argv[5]))
alg = sys.argv[6]
lr = float(sys.argv[7])
name = sys.argv[8]
log_freq = int(sys.argv[9])

# No regularization so far
# TODO: move this to the actual transformer file
@jax.jit
def loss_fn(train_state,x,y):
    logits = train_state.apply_fn(train_state.params, x, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

# Parameters regarding training examples
batch_size = 8
context_window = 128

# Load into RAM
data_dir = '/home/jozefiak/CL/Experiments/PS_fixed_111924'
train_data = jnp.array(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))
val_data = jnp.array(np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r'))

batches_train = D // (batch_size * context_window)
batches_test = len(val_data) // (batch_size * context_window)
test_freq = batches_train // log_freq


# Dataloader class
class DataLoader:
        def __init__(self, data, batch_size, context_window):
            self.data = data
            self.batch_size = batch_size
            self.context_window = context_window
            self.current_position = 0
            self.num_batches = len(data) // (batch_size * (context_window + 1))


        def next_batch(self):
            batch_size =self.batch_size
            context_window = self.context_window
            current_position = self.current_position
            num_batches = self.num_batches

            # Get indices for current batch
            start_idx = current_position * batch_size * context_window
            end_idx = start_idx + batch_size * context_window + 1  # Include next token for y
            batch = self.data[start_idx:end_idx]

            # Update self.current_position
            self.current_position = (current_position + 1) % num_batches

            # Reshape into (B, T+1), then split into x and y
            # batch_x = batch.reshape(batch_size, context_window)
            x = batch[:-1].reshape(batch_size, context_window)  # Input
            y = batch[1:].reshape(batch_size, context_window)   # Target (shifted by one token)

            # Convert to JAX arrays
            return jnp.array(x), jnp.array(y)

# Load dataloaders
dataloader_train = DataLoader(train_data, batch_size, context_window)
dataloader_test = DataLoader(val_data, batch_size, context_window)

reg_str = 1e-4 / jnp.sqrt(2)
lr = 1e-3

# Model Parameters
n_neurons = 4 * n_embd
config = CL_transformers.ModelConfig(vocab_size = 50257, n_head = n_head, n_layer = n_layer, n_embd = n_embd, n_neurons = n_neurons, use_resid=True)

epochs = 1

ReDO_reset_freq = 1 / (4 * epochs)
ReDO_threshold = 0.01
CBP_reset_freq = 1e-6
alg_params = {'threshold': 16,
                  'reset_percentile': 0.95,
                  'reset_freq': (1e-4 * (epochs / 20)),
                  'reg_str': reg_str,
                  'lr': lr,
                  'ReDO_reset_freq': ReDO_reset_freq,
                  'ReDO_threshold': ReDO_threshold,
                  'CBP_reset_freq': CBP_reset_freq}

# load the model
key = jr.PRNGKey(seed)
key, split_key = jr.split(key)

train_state, train_step = CL_transformers.get_transformer_methods(config, alg, alg_params, split_key)

train_losses = []
test_losses = []

for step in range(batches_train):

    # Compute test loss
    if step % test_freq == 0:
        test_loss = 0
        for _ in range(batches_test):
          x,y = dataloader_test.next_batch()
          # TODO use a faster non-train_step function
          loss = loss_fn(train_state, x, y)
          test_loss += loss
        test_loss /= batches_test
        test_losses.append(test_loss)
        print(f"Step: {step}, test loss: {test_loss:.4f}")

    x,y = dataloader_train.next_batch()

    key, split_key = jr.split(key)
    loss, train_state = train_step(train_state, x, y, split_key)
    train_losses.append(loss)

test_save_dir = "/pool001/jozefiak/CL/SL/test/name"
test_save_path = test_save_dir + name

train_save_dir = "/pool001/jozefiak/CL/SL/train/name"
train_save_path = train_save_dir + name

with open(test_save_path, 'wb') as f:
    pickle.dump(test_losses, f)

with open(train_save_path, 'wb') as f:
    pickle.dump(train_losses, f)