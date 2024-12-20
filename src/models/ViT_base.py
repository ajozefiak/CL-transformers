# Imports
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
from dataclasses import dataclass
from typing import Tuple
import optax

@dataclass
class ModelConfigViT:
    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 1000
    hidden_dim: int = 192
    num_heads: int = 3
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    lr: float = 1e-3
    gradient_accumulation_steps: int = 1


class PatchEmbed(nn.Module):
    """Convert image into patch embeddings."""
    config: ModelConfigViT

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C)
        # Use a Conv to get patch embeddings
        # This will produce shape: (B, H/patch_size, W/patch_size, hidden_dim)
        x = nn.Conv(features=self.config.hidden_dim, 
                     kernel_size=(self.config.patch_size, self.config.patch_size),
                     strides=(self.config.patch_size, self.config.patch_size))(x)
        # Flatten spatial dimensions:
        B, Hp, Wp, D = x.shape
        x = x.reshape(B, Hp * Wp, D)

        # Prepend a CLS token:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, D))
        cls = jnp.tile(cls, (B, 1, 1))  # (B, 1, D)
        x = jnp.concatenate([cls, x], axis=1)  # (B, 1 + num_patches, D)
        return x

class ViTSelfAttention(nn.Module):
    config: ModelConfigViT

    @nn.compact
    def __call__(self, x, deterministic=True):
        # x: (B, T, D)
        B, T, D = x.shape
        num_heads = self.config.num_heads
        head_dim = D // num_heads
        assert D % num_heads == 0, "Hidden dimension must be divisible by number of heads"

        qkv = nn.Dense(3 * D)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape to (B, T, num_heads, head_dim)
        q = q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        scale = 1.0 / jnp.sqrt(head_dim)
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale  # (B, num_heads, T, T)
        # For ViT, no causal masking; we allow full attention
        attn = nn.softmax(attn, axis=-1)

        # Apply attention
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = nn.Dense(D)(out)
        return out


class MLP(nn.Module):
    config: ModelConfigViT

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.config.hidden_dim * self.config.mlp_ratio)
        x = nn.Dense(hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.config.hidden_dim)(x)
        return x


class ViTBlock(nn.Module):
    config: ModelConfigViT

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Attention block
        h = nn.LayerNorm()(x)
        h = ViTSelfAttention(self.config)(h, deterministic=deterministic)
        x = x + h

        # MLP block
        h = nn.LayerNorm()(x)
        h = MLP(self.config)(h, deterministic=deterministic)
        x = x + h
        return x


class ViT(nn.Module):
    config: ModelConfigViT

    @nn.compact
    def __call__(self, x, deterministic=False):
        # x: (B, H, W, C)

        # Patch embedding
        x = PatchEmbed(self.config)(x)

        # Add learnable position embeddings
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        seq_len = num_patches + 1  # +1 for CLS
        pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, seq_len, self.config.hidden_dim))
        x = x + pos_embed

        # Apply Transformer layers
        for _ in range(self.config.num_layers):
            x = ViTBlock(self.config)(x, deterministic=deterministic)

        # Final LayerNorm
        x = nn.LayerNorm()(x)

        # Classification head: use the CLS token (x[:, 0])
        cls_token = x[:, 0]
        logits = nn.Dense(self.config.num_classes)(cls_token)
        return logits

    def init(self, rng, dummy_input):
        # dummy_input should be a batch of images, e.g. (1, 32, 32, 3)
        params = super().init(rng, dummy_input, True)
        return params


def init_train_state(key, config: ModelConfigViT) -> train_state.TrainState:
    model = ViT(config)
    dummy_input = jnp.zeros((1, config.image_size, config.image_size, 3), jnp.float32)
    params = model.init(key, dummy_input)
    optimizer = optax.adam(config.lr, b1=0.9, b2=0.999, eps=1e-7)

    if config.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=config.gradient_accumulation_steps
        )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    return state


@jax.jit
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, train_state.TrainState]:
    def loss_fn(params):
        logits = state.apply_fn(params, x, deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state