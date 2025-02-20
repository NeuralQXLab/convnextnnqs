from deepnets.net.ConvNext.heads import OutputHead, ExpHead, UnitCellHead
from deepnets.net.ConvNext.stems import PatchStem, ConvStem
from deepnets.net.ConvNext.encoder import Encoder, EncoderDoubleSkip
from flax import linen as nn
from netket.utils.types import Callable
import jax.numpy as jnp


class ConvNext(nn.Module):
    """
    Structure = Stem - Encoder - Head
    """

    lattice_shape: tuple
    """
        The number of sites along each dimension of the lattice, e.g (Lx, Ly)
    """
    n_blocks: tuple
    """
        The number of blocks in each stage in format (stage1, stage2, ...)
    """
    features: tuple
    """
        The number of features (channels) after downsampling for each stage, in format (stage1, stage2, ...)
    """
    expansion_factor: int
    """
        By what factor to expand and compress the features in each block
    """
    Encoder: nn.module
    Head: nn.module
    """
        Output head after ConvNextV2
    """
    kernel_size: tuple
    """
        Kernel size for all convolutions
    """
    downsample_factor: int
    """
        Stride for downsampling convolutions
    """
    final_features: int
    """
        Features of output head
    """
    extract_patches: Callable
    """
        Function to extract 2D images of patches from (...,Nsites) input
    """

    def setup(self):
        self.stem = PatchStem(
            features=self.features[0],
            lattice_shape=self.lattice_shape,
            linear_patch_size=self.downsample_factor,
            extract_patches=self.extract_patches,
        )
        self.encoder = self.Encoder(
            kernel_size=self.kernel_size,
            features=self.features,
            n_blocks=self.n_blocks,
            downsample_factor=self.downsample_factor,
            expansion_factor=self.expansion_factor,
            param_dtype=jnp.float64,
        )
        self.output = self.Head(
            lattice_shape=self.lattice_shape, final_features=self.final_features
        )

    def __call__(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.output(x)
        return x


class ConvNext_nopatching(nn.Module):
    init_kernel_size: tuple
    """
        The kernel size for the initial convolution
    """
    n_blocks: tuple
    """
        The number of blocks in each stage in format (stage1, stage2, ...)
    """
    features: tuple
    """
        The number of features (channels) after downsampling for each stage, in format (stage1, stage2, ...)
    """
    expansion_factor: int
    """
        By what factor to expand and compress the features in each block
    """
    kernel_size: tuple
    """
        Kernel size for convolutions in encoder
    """
    unitcell_shape: tuple
    """
        Shape of the primitive unit cell
    """
    lattice_shape: tuple
    """
       (Lx,Ly)
    """
    final_features: int
    """
        Features of output head
    """
    reshape_function: Callable
    """
        Function to reshape from (batch,nsites) -> (batch,x,y,1)
    """

    def setup(self):
        assert (
            len(self.n_blocks) == len(self.features) == 1
        )  # Only use for a single stage

        self.stem = ConvStem(
            kernel_size=self.init_kernel_size,
            features=self.features[0],
            expansion_factor=self.expansion_factor,
            lattice_shape=self.lattice_shape,
            reshape_function=self.reshape_function,
        )
        self.encoder = Encoder(
            kernel_size=self.kernel_size,
            features=self.features,
            n_blocks=self.n_blocks,
            downsample_factor=1,  # this is a dummy variable, since we assert that we only have one stage
            expansion_factor=self.expansion_factor,
            param_dtype=jnp.float64,
        )
        self.output = UnitCellHead(
            lattice_shape=self.lattice_shape,
            unitcell_shape=self.unitcell_shape,
            final_features=self.final_features,
        )

    def __call__(self, x):
        # print(x.shape)
        x = self.stem(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x

#need to define this for serialization (see bottom of this file)
def _create_output_head(output_depth):
    def output_head(lattice_shape, final_features):
        return OutputHead(lattice_shape, final_features, output_depth)
    return output_head

def ConvNextVanilla(
    lattice_shape: tuple,
    n_blocks: tuple,
    features: tuple,
    expansion_factor: int,
    kernel_size: tuple,
    downsample_factor: int,
    final_features: int,
    extract_patches: Callable,
    output_depth: int = 1,
):
    output_head = _create_output_head(output_depth)
    return ConvNext(
        lattice_shape=lattice_shape,
        n_blocks=n_blocks,
        features=features,
        expansion_factor=expansion_factor,
        Encoder=Encoder,
        Head=output_head,
        kernel_size=kernel_size,
        downsample_factor=downsample_factor,
        final_features=final_features,
        extract_patches=extract_patches,
    )

def ConvNextExp(
    lattice_shape: tuple,
    n_blocks: tuple,
    features: tuple,
    expansion_factor: int,
    kernel_size: tuple,
    downsample_factor: int,
    final_features: int,
    extract_patches: Callable,
):
    return ConvNext(
        lattice_shape=lattice_shape,
        n_blocks=n_blocks,
        features=features,
        expansion_factor=expansion_factor,
        Encoder=Encoder,
        Head=ExpHead,
        kernel_size=kernel_size,
        downsample_factor=downsample_factor,
        final_features=final_features,
        extract_patches=extract_patches,
    )

#register the output head function for serialization
from nqxpack._src.lib_v1.closure import register_closure_simple_serialization
from nqxpack._src.lib_v1.custom_types import register_serialization

register_closure_simple_serialization(
                    _create_output_head, "output_head", original_qualname='deepnets.net.ConvNext.net._create_output_head'
                    )