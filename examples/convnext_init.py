#How to initialize the unwrapped vanilla convnext

from deepnets.net.ConvNext.net import ConvNextVanilla
from deepnets.system import Shastry_Sutherland, Square_Heisenberg
import jax
import jax.numpy as jnp

# System
L = 6
J = [0.8,1.0]
system = Shastry_Sutherland(L=L, J = J)

# Network
n_blocks = (2,)
features = (12,)
expansion_factor = 2
downsample_factor = 2  # linear_patch_size
kernel_width = 2
final_features = features[0]
output_depth = 1
net = ConvNextVanilla(
    lattice_shape = (L,L),
    n_blocks=n_blocks,
    features=features,
    expansion_factor=expansion_factor,
    kernel_size=(kernel_width,kernel_width),
    downsample_factor=downsample_factor,
    final_features=final_features,
    extract_patches = system.extract_patches_as2d,
    output_depth = output_depth
)

#Initialize
params = net.init(jax.random.PRNGKey(0), jnp.ones((L**2)))
print("Initialized params: ",params)
#Apply
output = net.apply(params, jnp.ones((L**2)))
print("Output: ",output)