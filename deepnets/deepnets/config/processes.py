import os
import jax
import socket

print("Configuring sharding...")
if "MULTI_PROCESS_CPU" in os.environ:
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(cluster_detection_method="mpi4py")
elif "MULTI_PROCESS_GPU" in os.environ:
    jax.distributed.initialize()  # Checks via slurm or fails
import netket as nk

print(f"Number of distributed processes = {jax.process_count()}")
print(f"Number of total devices: {len(jax.devices())}")
print(
    f"{jax.process_index()}/{jax.process_count()} : global", jax.devices(), flush=True
)
print(
    f"{jax.process_index()}/{jax.process_count()} : local",
    jax.local_devices(),
    flush=True,
)
print(
    f"{jax.process_index()}/{jax.process_count()} : hostname: ",
    socket.gethostname(),
    flush=True,
)
