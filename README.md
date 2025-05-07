# ConvNext Neural Quantum States and lattice symmetries

Repository accompanying the article [Design principles of deep translationally-symmetric neural quantum states for frustrated magnets](https://arxiv.org/abs/2505.03466)

## Content

The repository is structured as follows:
- [data](data): Contains the data from simulations (in [data/files](data/files)) along with notebooks producing plots for the paper, and computing flops of a network.
Since the data takes up some space, it is not included on the default `main` branch, checkout the `with_data` branch in order to access it.
- [deepnets](deepnets): A python package containing code used for running the simulations, including the [ConvNext network](deepnets/net/ConvNext) and the [optimization protocol](deepnets/optimization). 
- [examples](examples): Contains examples of initializing and VMC optimization of the ConvNext.
    - `convnext_init.py` an example script with the hyperparameters to construct the ConvNext Flax model, which can directly be used with NetKet.
    - `convnext_simple_optimization.py` a simple optimization script showcasing how to train a Convnext NQS model on a 6x6 Lattice
    - `optimization.py` a sample optimization script we used for our runs, using a custom training protocol class.
- [scripts](scripts): Contains scripts for running VMC optimization of the ConvNext and computing expectation values from the optimized wavefunction

## Installation
A simple way to get up and running is to use [uv](https://docs.astral.sh/uv/getting-started/installation/).
To set up an environment and run once you have `uv` installed you just need to run `uv sync`, which will install the exact same versions of python packages that we used.
If you wish to use more recent software, you can run `uv sync --upgrade`, but incompatibilities might arise.

After you initialised the environment you can run commands using those exact versions of software installed with the commands.
Those commands will work as long as you are inside the root folder of this repository.
 - `uv run {script_to_run}` (this will not include the CUDA jax stack, so training will be very slow. Inference might be ok.)
 - `uv run --group cuda {script_to_run}` (to run with CUDA gpus enabled)
 - `uv run --group cuda --with jupyterlab` (to launch jupyterlab with cuda enabled and the packages we use)

You can also install directly the package used for our research called 'deepnets' by running `uv add {/path/to/connextnnqx/deepnets}` or `pip add {/path/to/connextnnqx/deepnets}`.

## Further Information
For further questions about what is in the repository, contact rajah.nutakki@polytechnique.edu.
