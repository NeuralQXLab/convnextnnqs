import jax.numpy as jnp

from flax import linen as nn

from netket.utils.types import DType


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer

    Transposed from the Pytorch implementation
    https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py#L105
    """

    feature_axis: int = -1
    normalise_axes: tuple = (-2, -3)
    param_dtype: DType = float

    @nn.compact
    def __call__(self, x):
        n_features = x.shape[self.feature_axis]
        gamma = self.param(
            "gamma", nn.initializers.zeros, (n_features,), self.param_dtype
        )
        beta = self.param(
            "beta", nn.initializers.zeros, (n_features,), self.param_dtype
        )

        gamma_shape = list(1 for _ in range(x.ndim))
        gamma_shape[self.feature_axis] = -1
        gamma = gamma.reshape(tuple(gamma_shape))
        beta = beta.reshape(tuple(gamma_shape))

        # Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Gx = jnp.linalg.norm(x, ord="fro", axis=self.normalise_axes, keepdims=True)
        Nx = Gx / (jnp.mean(Gx, axis=self.feature_axis, keepdims=True) + 1e-6)
        return gamma * (x * Nx) + beta + x
