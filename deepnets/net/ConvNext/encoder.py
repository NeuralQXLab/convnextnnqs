import jax.numpy as jnp
from flax import linen as nn
from netket.utils.types import DType
from deepnets.nn.blocks.convnext_utils import GRN
from typing import Optional


class ConvNext2Block(nn.Module):
    kernel_size: tuple
    n_features: int
    expansion_factor: int = 4
    param_dtype: DType = float

    @nn.compact
    def __call__(self, x_input):
        # convolution with unit stride, keeping same number of features.
        # This keeps the size of the input constant
        convolution = nn.Conv(
            self.n_features,
            self.kernel_size,
            padding="CIRCULAR",
            feature_group_count=self.n_features,
            param_dtype=self.param_dtype,
        )
        # Normalises according to the last axis
        layernorm = nn.LayerNorm()
        grn = GRN()
        dense_expand = nn.Dense(
            self.n_features * self.expansion_factor, param_dtype=self.param_dtype
        )
        dense_compress = nn.Dense(self.n_features, param_dtype=self.param_dtype)

        x = x_input
        x = convolution(x)
        x = layernorm(x)
        x = dense_expand(x)
        x = nn.activation.gelu(x)
        x = grn(x)
        x = dense_compress(x)
        return x_input + x


class ConvNextDownsamplingBlock(nn.Module):
    features: int
    kernel_size: tuple
    downsampling_factor: int | tuple = 2
    param_dtype: DType = float
    layernorm: bool = True

    @nn.compact
    def __call__(self, x):
        downsampling_factor = self.downsampling_factor
        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,) * len(self.kernel_size)

        x = nn.Conv(
            self.features,
            self.kernel_size,
            strides=downsampling_factor,
            padding="CIRCULAR",
            param_dtype=self.param_dtype,
        )(x)

        if self.layernorm:
            x = nn.LayerNorm(epsilon=1e-6, param_dtype=self.param_dtype)(x)
        return x


class Encoder(nn.Module):
    """
    A modified ConvNextV2 encoder, without the stem
    """

    kernel_size: tuple
    features: int | tuple = (48, 96)
    """
        Number of features in each stage.
    """
    n_blocks: int | tuple = (3, 3)
    """
        Number of blocks in each stage.
    """
    downsample_factor: int = 2
    """The stride of each downsampling convolution, which is executed before every stage of blocks.

    If you want to keep the same size, set to 1.
    """
    expansion_factor: int = 4
    param_dtype: int = float

    def setup(self):
        # Check selfs variables
        if isinstance(self.features, int) and isinstance(self.n_blocks, int):
            self.features = (self.features,)
            self.n_blocks = (self.n_blocks,)
        assert len(self.n_blocks) == len(self.features)

        if not isinstance(self.kernel_size, tuple):
            raise TypeError("kernel_size must be a tuple")

        stages = []
        for i, (features, blocks) in enumerate(zip(self.features, self.n_blocks)):
            if i != 0:
                downsample_layer = ConvNextDownsamplingBlock(
                    features,
                    kernel_size=self.kernel_size,
                    downsampling_factor=self.downsample_factor,
                    param_dtype=self.param_dtype,
                )

            blocks_in_stage = [
                ConvNext2Block(
                    kernel_size=self.kernel_size,
                    expansion_factor=self.expansion_factor,
                    param_dtype=self.param_dtype,
                    n_features=features,
                )
                for _ in range(blocks)
            ]
            if i == 0:
                stages.append(nn.Sequential(blocks_in_stage))
            else:
                stages.append(nn.Sequential([downsample_layer] + blocks_in_stage))
        self.stages = stages

    def __call__(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
