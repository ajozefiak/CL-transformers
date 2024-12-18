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

from threading import Thread
import queue


# Check arguments
if len(sys.argv) != 10:
    print("Not enough arguments")
    sys.exit()

seed = int(sys.argv[1])
n_scale = int(sys.argv[2])
D = int(float(sys.argv[3]))
alg = sys.argv[4]
lr = float(sys.argv[5])
name = sys.argv[6]
log_freq = int(sys.argv[7])
epochs = int(sys.argv[8])

n_head = 4
n_layer = n_scale
# aspect ratio of 32 is fixed
n_embd = int(32 * n_layer)


# No regularization so far
# TODO: move this to the actual transformer file
@jax.jit
def loss_fn(train_state,x,y):
    logits = train_state.apply_fn(train_state.params, x, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

# Parameters regarding training examples
batch_size = 128
context_window = 128

# Load into RAM
data_dir = '/home/jozefiak/CL/Experiments/PS_fixed_111924'
train_data = jnp.array(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))
val_data = jnp.array(np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r'))

# Dataloader class
class DataLoader:
    def __init__(self, data, batch_size, context_window, key, include_target=True):
        """
        Initializes the DataLoader.

        Args:
            data: Array of tokenized data.
            batch_size: Number of sequences per batch.
            context_window: Size of the context window (T).
            key: JAX random key for reproducibility.
            include_target: Whether to include the target token (T+1). Default is True.
        """
        self.data = data
        self.batch_size = batch_size
        self.context_window = context_window
        self.block_size = context_window + 1 if include_target else context_window
        self.include_target = include_target

        # Compute the number of blocks
        self.num_blocks = len(data) // self.block_size
        self.num_batches = self.num_blocks // batch_size

        # JAX random key for reproducibility
        self.key = key

        # Create initial shuffled block order
        self.block_order = jnp.arange(self.num_blocks)
        self.shuffle_blocks()  # Shuffle initially

        self.current_batch_index = 0

    def shuffle_blocks(self):
        """Shuffles the order of the blocks using JAX random functionality."""
        self.key, subkey = jax.random.split(self.key)
        self.block_order = jax.random.permutation(subkey, self.block_order)

    def next_batch(self):
        """
        Returns the next batch of data.

        Returns:
            x: Input data of shape (batch_size, context_window).
            y: Target data of shape (batch_size, context_window) if include_target is True,
               otherwise None.
        """
        if self.current_batch_index >= self.num_batches:
            # Reset for the next epoch
            self.current_batch_index = 0
            self.shuffle_blocks()

        # Fetch batch block indices
        start_block = self.current_batch_index * self.batch_size
        end_block = start_block + self.batch_size
        batch_block_indices = self.block_order[start_block:end_block]

        # Fetch corresponding blocks
        x_batch = []
        y_batch = []

        for block_idx in batch_block_indices:
            start_idx = block_idx * self.block_size
            end_idx = start_idx + self.block_size
            block = self.data[start_idx:end_idx]

            if self.include_target:
                x_batch.append(block[:-1])  # All except the last token
                y_batch.append(block[1:])   # All except the first token
            else:
                x_batch.append(block)

        # Increment batch index
        self.current_batch_index += 1

        # Convert to JAX arrays and return
        x = jnp.array(x_batch)
        y = jnp.array(y_batch) if self.include_target else None
        return x, y

class DataLoader_noshuffle:
    def __init__(self, data, batch_size, context_window, include_target=True):
        """
        Initializes the DataLoader.

        Args:
            data: Array of tokenized data.
            batch_size: Number of sequences per batch.
            context_window: Size of the context window (T).
            key: JAX random key for reproducibility.
            include_target: Whether to include the target token (T+1). Default is True.
        """
        self.data = data
        self.batch_size = batch_size
        self.context_window = context_window
        self.block_size = context_window + 1 if include_target else context_window
        self.include_target = include_target

        # Compute the number of blocks
        self.num_blocks = len(data) // self.block_size
        self.num_batches = self.num_blocks // batch_size

        # Create initial shuffled block order
        self.block_order = jnp.arange(self.num_blocks)

        self.current_batch_index = 0

    def next_batch(self):
        """
        Returns the next batch of data.

        Returns:
            x: Input data of shape (batch_size, context_window).
            y: Target data of shape (batch_size, context_window) if include_target is True,
               otherwise None.
        """
        if self.current_batch_index >= self.num_batches:
            # Reset for the next epoch
            self.current_batch_index = 0

        # Fetch batch block indices
        start_block = self.current_batch_index * self.batch_size
        end_block = start_block + self.batch_size
        batch_block_indices = self.block_order[start_block:end_block]

        # Fetch corresponding blocks
        x_batch = []
        y_batch = []

        for block_idx in batch_block_indices:
            start_idx = block_idx * self.block_size
            end_idx = start_idx + self.block_size
            block = self.data[start_idx:end_idx]

            if self.include_target:
                x_batch.append(block[:-1])  # All except the last token
                y_batch.append(block[1:])   # All except the first token
            else:
                x_batch.append(block)

        # Increment batch index
        self.current_batch_index += 1

        # Convert to JAX arrays and return
        x = jnp.array(x_batch)
        y = jnp.array(y_batch) if self.include_target else None
        return x, y

class AsyncDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.queue = queue.Queue(maxsize=10)
        self.thread = Thread(target=self._load_data)
        self.thread.daemon = True
        self.thread.start()

    def _load_data(self):
        while True:
            self.queue.put(self.dataloader.next_batch())

    def get_batch(self):
        return self.queue.get()


# Load dataloaders
# D_U is the number of unique tokens
# load the model
key = jr.PRNGKey(seed)
key, split_key = jr.split(key)

D_U = D

dataloader_train = DataLoader(train_data[0:int(D_U)], batch_size=batch_size, context_window=context_window, key=key, include_target=True)
dataloader_test_small = DataLoader_noshuffle(val_data[:int(5e5)], batch_size=batch_size, context_window=context_window, include_target=True)
async_dl_train = AsyncDataLoader(dataloader_train)


# Model Parameters
n_neurons = 4 * n_embd
config = CL_transformers.ModelConfig(vocab_size = 50257, n_head = n_head, n_layer = n_layer, n_embd = n_embd, n_neurons = n_neurons, use_resid=True)


reg_str = 0.0
lr = 1e-3

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



train_state, train_step = CL_transformers.get_transformer_methods(config, alg, alg_params, split_key)

train_losses = []
test_losses = []

test_batches_small = dataloader_test_small.num_batches
train_batches = dataloader_train.num_batches
test_freq = log_freq
t = 0

for epoch in range(epochs):

    for step in range(train_batches):
    
        # Compute test loss
        if t % test_freq == 0:
            test_loss = 0
          
            for _ in range(test_batches_small):
                x,y = dataloader_test_small.next_batch()
                _, loss = loss_fn(train_state, x, y)
                test_loss += loss
            test_loss /= test_batches_small
            test_losses.append(test_loss)
            print(f"Step: {step}, test loss: {test_loss:.4f}")

        x,y = async_dl_train.get_batch()

        t += 1

        key, split_key = jr.split(key)
        loss, train_state = train_step(train_state, x, y, split_key)
        train_losses.append(loss)

    # Save results at the end of the epoch
    test_save_dir = "/pool001/jozefiak/CL/SL/test/"
    test_save_path = test_save_dir + name

    train_save_dir = "/pool001/jozefiak/CL/SL/train/"
    train_save_path = train_save_dir + name

    with open(test_save_path, 'wb') as f:
        pickle.dump(test_losses, f)

    with open(train_save_path, 'wb') as f:
        pickle.dump(train_losses, f)