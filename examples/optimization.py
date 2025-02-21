#This script demonstrates how to run an optimization using the `Protocol` 
# in `deepnets/optimization/protocols.py`.
# This can use multiple symmetry stages and a number of post-optimization steps 

import netket as nk
from deepnets.net import ConvNext
from deepnets.system import Shastry_Sutherland, Square_Heisenberg
from deepnets.optimization.protocols import Protocol
import deepnets.optimization.save_load as save_load
import argparse
import numpy as np
import optax
import jax
import json
import jax.numpy as jnp

#Parameters
#Model parameters
L = 6
J = [0.8,1.0]
system = Shastry_Sutherland(L=L, J = J)
#sign_rule = [False]
#Network parameters
#ConvNext
n_blocks = (1,)
features = (12,)
expansion_factor = 2
downsample_factor = 2 #linear_patch_size 
kernel_width=3
net_type="Vanilla"
final_features = features[0]
output_depth = 1

network = ConvNext(n_blocks = n_blocks,
                   features = features,
                   expansion_factor = expansion_factor,
                   net_type = net_type,
                   kernel_width = kernel_width,
                   downsample_factor = downsample_factor,
                   final_features = final_features,
                   init_kernel_width = 1,
                   output_depth = output_depth,
                   system = system)
args = {"lr": 1e-3,
        "lr_factor":0.5,
        "iters":[10],
        "alpha":1,
        "samples_per_rank": 128,
        "n_chains_per_rank": 16,
        "diag_shift":1e-6,
        "diag_shift_factor":1,
        "r": 1e-6,
        "symmetries": 0,
        "seed": 5,
        "discard_fraction":0,
        "save_every":50,
        "show_progress": True,
        "time_it": False,
        "save_base":"",
        "post_iters": 0,
        "chunk_size": 128,
        "momentum":0.9,
        "sweep_factor":1,
        "symmetry_ramping": 0 
}
opt = Protocol(system,network,args,compile_step=False)
opt.run()