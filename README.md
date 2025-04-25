Repository accompanying the article [Design principles of deep translationally-symmetric neural quantum states for frustrated magnets](link)

The repository is structured as follows:
- [data](data): Contains the data from simulations (in [data/files](data/files)) along with notebooks producing plots for the paper, and computing flops of a network.
Since the data takes up some space, it is not included on the default `main` branch, checkout the `with_data` branch in order to access it.
- [deepnets](deepnets): Contains code used for running the simulations, including the [ConvNext network](deepnets/net/ConvNext) and the [optimization protocol](deepnets/optimization)
- [examples](examples): Contains examples of initializing and VMC optimization of the ConvNext.
- [scripts](scripts): Contains scripts for running VMC optimization of the ConvNext and computing expectation values from the optimized wavefunction

# Installation
A simple way to get up and running is to use [uv](https://docs.astral.sh/uv/getting-started/installation/).
To set up an environment and run once you have `uv` installed: 
1. Run `install.sh` in your terminal. This will set up an environment in `.venv` with all required packages installed.
2. Run the desired scripts, e.g `examples/convnext_init.py`, using the command `uv run {script_to_run}`

# Further Information
For further questions about what is in the repository, contact rajah.nutakki@polytechnique.edu.
