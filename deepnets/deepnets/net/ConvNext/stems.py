import jax.numpy as jnp
from flax import linen as nn
from netket.utils.types import Callable
from netket.utils.types import DType
from deepnets.nn.blocks.convnext_utils import GRN


# Stems are the initial block of the stem - encoder - head structure
class ConvNext2StemBlock(nn.Module):
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


class PatchStem(nn.Module):
    """
    Initial downsampling block, using apatching then embedding.
    Equivalent to a convolution with stride = kernel_width.
    This is the same as used in the ViT
    """

    features: int
    lattice_shape: tuple
    linear_patch_size: tuple
    extract_patches: Callable

    @nn.compact
    def __call__(self, x):
        x = self.extract_patches(x, self.linear_patch_size, self.lattice_shape)
        # x = nn.LayerNorm(epsilon=1e-6, param_dtype=self.param_dtype)(x) # include only if have actual (x,y) image, i,e not in patches
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )(x)

        return x


class ConvStem(nn.Module):
    """
    Use a ConvNext encoder block as the stem.
    We do this so that we can use a separate kernel size for the stem as for the rest of the encoder.
    The whole network can be thought of as having no stem, but with an initial convolution different to the rest of the encoder.
    """

    kernel_size: tuple
    features: int
    lattice_shape: tuple
    reshape_function: Callable
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        convnext_block = ConvNext2StemBlock(
            kernel_size=self.kernel_size,
            expansion_factor=self.expansion_factor,
            n_features=self.features,
        )
        x = self.reshape_function(x, self.lattice_shape)
        return convnext_block(x)
