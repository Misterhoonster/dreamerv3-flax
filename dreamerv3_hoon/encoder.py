import math
from typing import Sequence

from chex import Array
import jax.numpy as jnp
import flax.linen as nn

from dreamerv3_hoon.flax_util import Conv, Dense


class CNNEncoder(nn.Module):
    """CNN encoder module."""

    in_shape: Sequence[int]
    chan: int = 96
    min_res: int = 4
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes an encoder."""
        # Convolutional layers
        num_layers = int(math.log2(self.in_shape[0] // self.min_res))
        out_chans = [2**i * self.chan for i in range(num_layers)]
        self.layers = [
            Conv(
                out_chan,
                kernel_size=(4, 4),
                strides=(2, 2),
                act_type=self.act_type,
                norm_type=self.norm_type,
            )
            for out_chan in out_chans
        ]

    def __call__(self, x: Array) -> Array:
        """Runs the forward pass of the encoder."""

        # Transform the input.
        x = x - 0.5

        # Apply the convolutional layers.
        for layer in self.layers:
            x = layer(x)
        x = jnp.reshape(x, (*x.shape[:-3], -1))

        return x


class MLPEncoder(nn.Module):
    """MLP encoder module for Catch environment."""

    in_shape: Sequence[int]
    out_size: int = 96
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes an encoder."""
        # Two-layer MLP
        self.linear1 = Dense(
            256,
            act_type=self.act_type,
            norm_type=self.norm_type,
        )
        self.linear2 = Dense(
            self.out_size,  # Final embedding size
            act_type="none",
            norm_type="none",
        )

    def __call__(self, x: Array) -> Array:
        """Runs the forward pass of the encoder."""
        # Flatten input while preserving batch dimensions
        x = jnp.reshape(x, (*x.shape[:-2], -1))  # (B, N, H*W)
        
        # Transform the input
        x = x - 0.5

        # Apply MLP layers
        x = self.linear1(x)
        x = self.linear2(x)

        return x
