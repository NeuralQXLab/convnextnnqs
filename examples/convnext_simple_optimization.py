#Optimization with the convnext without using `Protocol`

import netket as nk
import netket.experimental as nke
from deepnets.net import ConvNext
from deepnets.system import Square_Heisenberg, Shastry_Sutherland
import optax

# System
L = 6
J = [0.8,1.0]
system = Shastry_Sutherland(L=L, J = J)

# Network
n_blocks = (2,)
features = (12,)
expansion_factor = 2
downsample_factor = 2  # linear_patch_size
net_type = "Vanilla"
kernel_width = 2
final_features = features[0]
net_wrapped = ConvNext(
    n_blocks=n_blocks,
    features=features,
    expansion_factor=expansion_factor,
    net_type=net_type,
    kernel_width=kernel_width,
    downsample_factor=downsample_factor,
    final_features=final_features,
    init_kernel_width=1, #dummty if not using NoPatching
    system=system,
)
nets = [
    f(net_wrapped.network) for f in system.symmetrizing_functions
]  # set of symmetrized networks, nets[0] is unsymmetrized, nets[-1] fully symmetrized

# Sampler
n_chains_per_rank = 16
sweep_factor = 1
n_samples_per_rank = 100
chunk_size = None
n_discard_per_chain = 0
sampler = nk.sampler.MetropolisExchange(
    hilbert=system.hilbert_space,
    graph=system.graph,
    n_chains_per_rank=16,
    sweep_size=system.graph.n_nodes * sweep_factor,
)

# Log
save_base = ""
save_every = 2
log = nk.logging.JsonLog(
    save_base + "opt",
    mode="write",
    write_every=save_every,
    save_params=True,
    save_params_every=save_every,
)

# Optimization
iters = 5
lr = 5e-3
lr_factor = 0.5
diag_shift = 1e-2
diag_shift_factor = 1e-2

lr_scheduler = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=iters, alpha=lr_factor
)
diag_shift_scheduler = optax.cosine_decay_schedule(
    init_value=diag_shift, decay_steps=iters, alpha=diag_shift_factor
)
optimizer = nk.optimizer.Sgd(learning_rate=lr_scheduler)
solver = nk.optimizer.solver.cholesky

# Vstate
network = nets[-1]  # [0] is unsymmetrized, [-1] fully symmetrized
vstate = nk.vqs.MCState(
    sampler,
    model=network,
    n_samples_per_rank=n_samples_per_rank,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size,
)

# Run simulation
driver = nke.driver.VMC_SRt_ntk(
    hamiltonian=system.hamiltonian,
    optimizer=optimizer,
    linear_solver_fn=solver,
    diag_shift=diag_shift_scheduler,
    variational_state=vstate
)

callbacks = []
driver.run(
    n_iter=iters,
    out=log,
    show_progress=True,
    callback=callbacks,
)
