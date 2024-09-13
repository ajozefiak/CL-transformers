# These two are necessary for experiments, but not for the transformer
# import datasets
# from datasets import load_from_disk
# import tiktoken
# import pickle
# import numpy as np

# Imports

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.core import FrozenDict  # Import FrozenDict for type annotations
from flax import linen as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
  vocab_size: int = 50257
  block_size: int = 128

  n_layer: int = 3
  n_head: int = 4 
  n_embd: int = 64
  dropout_rate: float = 0.0
  gradient_accumulation_steps: int = 1

config = ModelConfig()

class CausalSelfAttention(nn.Module):

  config: ModelConfig

  @nn.compact
  def __call__(self, x, deterministic=True):

    assert len(x.shape) == 3

    b, l, d = x.shape

    q     = nn.Dense(self.config.n_embd)(x)
    k     = nn.Dense(self.config.n_embd)(x)
    v     = nn.Dense(self.config.n_embd)(x)
    # q*k / sqrt(dim) -> softmax -> @v
    q     = jnp.reshape(q, (b, l, d//self.config.n_head , self.config.n_head))
    k     = jnp.reshape(k, (b, l, d//self.config.n_head , self.config.n_head))
    v     = jnp.reshape(v, (b, l, d//self.config.n_head , self.config.n_head))
    norm  = jnp.sqrt(list(jnp.shape(k))[-1])
    attn  = jnp.matmul(q,jnp.transpose(k, (0,1,3,2))) / norm
    # My added line below to deal with Jax rounding QK^T => 0.0
    attn += 1e-6
    #
    mask  = jnp.tril(attn)
    attn  = jnp.where(mask[:,:,:l,:l], attn, float("-inf"))
    probs = jax.nn.softmax(attn, axis=-1)
    y     = jnp.matmul(probs, v)
    y     = jnp.reshape(y, (b,l,d))
    y     = nn.Dense(self.config.n_embd)(y)
    return y

class MLP(nn.Module):

  config: ModelConfig

  @nn.compact
  def __call__(self, x, deterministic=True):
    x = nn.Dense(self.config.n_embd*4)(x)
    # x = nn.gelu(x, approximate=True)
    # sow the features so that we can observe neuron death
    self.sow('intermediates', 'features', x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
    x = nn.Dense(self.config.n_embd)(x)
    x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
    return x

class Block(nn.Module):

  config: ModelConfig

  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm()(x)
    x = x + CausalSelfAttention(self.config)(x)

    x = nn.LayerNorm()(x)
    x = x + MLP(self.config)(x)
    return x

class GPT(nn.Module):

  config: ModelConfig

  @nn.compact
  def __call__(self, x, deterministic=False):

    B, T = x.shape
    assert T <= self.config.block_size

    pos     = jnp.arange(0, T)[None]
    pos_emb = nn.Embed(self.config.block_size, self.config.n_embd)(pos)
    wte     = nn.Embed(self.config.vocab_size, self.config.n_embd)
    tok_emb = wte(x)
    x = tok_emb + pos_emb

    for _ in range(self.config.n_layer):
      x = Block(self.config)(x)

    x = nn.LayerNorm()(x)

    logits = nn.Dense(config.vocab_size)(x)
    # logits = wte.attend(x) # parameter sharing
    return logits

  def init(self, rng):
    tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
    params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
    return params

import optax
from flax.training.train_state import TrainState
from typing import Tuple

def init_train_state(key, config) -> TrainState:
  model = GPT(config)
  params = model.init(key)

  # learning_rate = optax.warmup_cosine_decay_schedule(
  #     init_value=0.0,
  #     peak_value=2.5e-4,
  #     warmup_steps= 14000,
  #     decay_steps= 120000,
  #     end_value = 1e-5,
  #     )

  optimizer = optax.adam(1e-3, b1=0.9, b2=0.999, eps=1e-7)
  # optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-7, weight_decay=1e-2)

  if config.gradient_accumulation_steps>1:
    optimizer = optax.MultiSteps(
          optimizer, every_k_schedule=config.gradient_accumulation_steps
    )

  train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
  return train_state

@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, TrainState]:

  def loss_fn(params: FrozenDict) -> jnp.ndarray:
      logits = state.apply_fn(params, x, False)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
      return loss

  loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
  new_state = state.apply_gradients(grads=grads)

  return loss, new_state
  # return loss, new_state, grads