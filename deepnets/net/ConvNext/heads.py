import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Callable
from netket import nn as nknn
from netket import jax as nkjax
from netket.jax import logsumexp_cplx as logsumexp
from netket.utils.types import DType

def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class OutputHead(nn.Module):
    """
    Same head as ViT but pooling over lattice dimensions
    """

    lattice_shape: tuple
    final_features: int
    output_depth: int = 1

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norms_real = [
            nn.LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )
            for layer in range(self.output_depth)
        ]

        self.norms_imag = [
            nn.LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )
            for layer in range(self.output_depth)
        ]

        self.output_layers_real = [
            nn.Dense(
                self.final_features,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )
            for layer in range(self.output_depth)
        ]

        self.output_layers_imag = [
            nn.Dense(
                self.final_features,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )
            for layer in range(self.output_depth)
        ]

    def __call__(self, x):
        x = jnp.sum(
            x, axis=tuple(-np.arange(len(self.lattice_shape)) - 2)
        )  # Sum pooling over lattice (..., lattice, features) -> (..., features)
        x = self.out_layer_norm(x)

        x_real = x
        x_imag = x
        for i in range(self.output_depth - 1):
            x_real = self.norms_real[i](
                nn.activation.gelu(self.output_layers_real[i](x_real))
            )
            x_imag = self.norms_imag[i](
                nn.activation.gelu(self.output_layers_imag[i](x_imag))
            )

        x_real = self.norms_real[-1](self.output_layers_real[-1](x_real))
        x_imag = self.norms_imag[-1](self.output_layers_imag[-1](x_imag))

        z = x_real + 1j * x_imag

        return jnp.sum(
            log_cosh(z), axis=-1
        )  # logPsi = log \prod_{i=1}^Nfeatures cosh(z)


class ExpHead(nn.Module):
    """
    Same as OutputHead but with logsumexp pooling over features at the end
    """

    lattice_shape: tuple
    final_features: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.output_layer0 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        x = jnp.sum(
            x, axis=tuple(-np.arange(len(self.lattice_shape)) - 2)
        )  # Sum pooling over lattice (..., lattice, features) -> (..., features)
        x = self.out_layer_norm(x)

        amp = self.norm2(self.output_layer0(x))
        sign = self.norm3(self.output_layer1(x))

        z = amp + 1j * sign

        return logsumexp(
            log_cosh(z), axis=-1
        )  # \Psi = log \sum_{i=1}^Nfeatures cosh(z)


class UnitCellHead(nn.Module):
    """
    For using after an unpatched encoder.
    Sum pool up to unit cell, (...,Lx,Ly,features) -> (...,ax,ay,features).
    Convolution with stride ax (=ay), (...,ax,ay,features) -> (...,features)
    Same complex RBM over features as in OutputHead
    """

    lattice_shape: tuple
    unitcell_shape: tuple
    final_features: int

    def setup(self):
        assert self.lattice_shape[0] % self.unitcell_shape[0] == 0
        assert self.lattice_shape[1] % self.unitcell_shape[1] == 0

        self.Lx = self.lattice_shape[0]
        self.Ly = self.lattice_shape[1]
        self.ax = self.unitcell_shape[0]
        self.ay = self.unitcell_shape[1]

        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.strided_conv = nn.Conv(
            features=self.final_features,
            kernel_size=self.unitcell_shape,
            strides=self.unitcell_shape,
            padding="CIRCULAR",  # Periodic boundaries
            feature_group_count=self.final_features,  # Depth-wise(?)
        )

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.output_layer0 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def partial_sum(self, x, Lx, Ly, ax, ay):
        """
        Perform partial sum pooling from (nbatch,Lx,Ly,features) -> (nbatch,ax,ay,features).
        Such that result[:,i,j,:] = sum_{i=0,ax-1,2*ax-1,...,Lx-1} sum_{j=0,ay-1,2*ay-1,...,Ly-1} input[:,i,j,:]
        """
        ys = [
            jnp.sum(x[:, i::ax, j::ay, :], axis=(-3, -2))
            for i in range(ax)
            for j in range(ay)
        ]
        y = jnp.array(ys)  # -> (ax*ay, nbatch, features)
        y = y.transpose((1, 2, 0))  # -> (nbatch,features,ax*ay)
        y = y.reshape((y.shape[0], y.shape[1], ax, ay))  # -> (nbatch, features, ax, ay)
        y = y.transpose((0, 2, 3, 1))  # -> (nbatch, ax, ay, features)
        return y

    def __call__(self, x):
        x = self.partial_sum(
            x, self.Lx, self.Ly, self.ax, self.ay
        )  # (nbatch,Lx,Ly,features) -> (nbatch,ax,ay,features)
        x = self.out_layer_norm(x)
        x = self.strided_conv(x)  # (nbatch,ax,ay,features) -> (nbatch,1,1,features)
        x = x[:, 0, 0, :]  # -> (nbatch,features)

        amp = self.norm2(self.output_layer0(x))
        sign = self.norm3(self.output_layer1(x))

        z = amp + 1j * sign

        return jnp.sum(log_cosh(z), axis=-1)  # sum over features at the end


class UnitCellHead(nn.Module):
    """
    For using after an unpatched encoder.
    Sum pool up to unit cell, (...,Lx,Ly,features) -> (...,ax,ay,features).
    Convolution with stride ax (=ay), (...,ax,ay,features) -> (...,features)
    Same complex RBM over features as in OutputHead
    """

    lattice_shape: tuple
    unitcell_shape: tuple
    final_features: int

    def setup(self):
        assert self.lattice_shape[0] % self.unitcell_shape[0] == 0
        assert self.lattice_shape[1] % self.unitcell_shape[1] == 0

        self.Lx = self.lattice_shape[0]
        self.Ly = self.lattice_shape[1]
        self.ax = self.unitcell_shape[0]
        self.ay = self.unitcell_shape[1]

        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.strided_conv = nn.Conv(
            features=self.final_features,
            kernel_size=self.unitcell_shape,
            strides=self.unitcell_shape,
            padding="CIRCULAR",  # Periodic boundaries
            feature_group_count=self.final_features,  # Depth-wise(?)
        )

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.output_layer0 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.final_features,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def partial_sum(self, x, Lx, Ly, ax, ay):
        """
        Perform partial sum pooling from (nbatch,Lx,Ly,features) -> (nbatch,ax,ay,features).
        Such that result[:,i,j,:] = sum_{i=0,ax-1,2*ax-1,...,Lx-1} sum_{j=0,ay-1,2*ay-1,...,Ly-1} input[:,i,j,:]
        """
        ys = [
            jnp.sum(x[:, i::ax, j::ay, :], axis=(-3, -2))
            for i in range(ax)
            for j in range(ay)
        ]
        y = jnp.array(ys)  # -> (ax*ay, nbatch, features)
        y = y.transpose((1, 2, 0))  # -> (nbatch,features,ax*ay)
        y = y.reshape((y.shape[0], y.shape[1], ax, ay))  # -> (nbatch, features, ax, ay)
        y = y.transpose((0, 2, 3, 1))  # -> (nbatch, ax, ay, features)
        return y

    def __call__(self, x):
        x = self.partial_sum(
            x, self.Lx, self.Ly, self.ax, self.ay
        )  # (nbatch,Lx,Ly,features) -> (nbatch,ax,ay,features)
        x = self.out_layer_norm(x)
        x = self.strided_conv(x)  # (nbatch,ax,ay,features) -> (nbatch,1,1,features)
        x = x[:, 0, 0, :]  # -> (nbatch,features)

        amp = self.norm2(self.output_layer0(x))
        sign = self.norm3(self.output_layer1(x))

        z = amp + 1j * sign

        return jnp.sum(log_cosh(z), axis=-1)  # sum over features at the end
