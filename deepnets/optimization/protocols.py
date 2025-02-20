import os
import netket as nk
import netket.experimental as nke
import deepnets.optimization.save_load as saver
import jax
import jaxlib
from jax.tree import structure as tree_structure
import optax
import time
from typing import Sequence
import numpy as np
import jax.numpy as jnp
import argparse
from deepnets.optimization.utils import process_print, to_sequence, add_module
from deepnets.callbacks import SaveVariationalState

class Protocol:
    callbacks = (nk.callbacks.InvalidLossStopping(),)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """
        Add optimization/post-optimization arguments to the parser.
        These arguments are read into the class instance in __init__.
        """
        parser.add_argument(
            "--samples_per_rank",
            type=int,
            help="Number of samples on each rank",
        )
        parser.add_argument(
            "--n_chains_per_rank",
            type=int,
            help="Number of MC chains per rank",
        )
        parser.add_argument(
            "--discard_fraction",
            type=float,
            help="Fraction of samples to discard per chain",
        )
        parser.add_argument(
            "--iters", type=int, action="append", help="Number of optimization steps"
        )
        parser.add_argument("--lr", type=float, action="append", help="Learning rate")
        parser.add_argument(
            "--diag_shift",
            type=float,
            action="append",
            help="Initial diagonal shift of schedule",
        )
        parser.add_argument(
            "--diag_shift_factor",
            type=float,
            action="append",
            help="The factor multiplied by diag_shift to give the final diag_shift",
        )
        parser.add_argument("--r", type=float, help="rtol and rtol_smooth for pinv")
        parser.add_argument(
            "--save_base",
            type=str,
            default="",
            help="File to save optimization results to",
        )
        parser.add_argument(
            "--save_num", type=int, default=0, help="Number to append to save file name"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1,
            help="Random seed for initializing state and sampler",
        )
        parser.add_argument(
            "--time_it", type=int, default=0, help="Whether to time the optimization"
        )
        parser.add_argument(
            "--show_progress",
            type=int,
            default=0,
            help="Whether to show progress bar during optimization",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=0,
            help="Number of samples to compute with at a time (reduces memory requirements), 0 = None",
        )
        parser.add_argument(
            "--lr_factor",
            type=float,
            action="append",
            help="Cosine decay scheduler factor for lr, min learning rate = lr_factor*lr",
        )
        parser.add_argument(
            "--save_every",
            type=int,
            help="Number of iterations between saving data",
        )
        parser.add_argument(
            "--symmetries",
            type=int,
            default=0,
            help="Whether to use symmetries when running the standard optimization",
        )
        parser.add_argument(
            "--symmetry_ramping",
            type=int,
            default=0,
            help="Whether to use symmetry ramping",
        )
        parser.add_argument(
            "--post_iters",
            type=int,
            default=0,
            help="Number of iterations to perform post optimization to choose lowest energy state",
        )
        parser.add_argument(
            "--sweep_factor",
            type=int,
            default = 1,
            help = "Factor for number of MC steps per sweep"
        )

    @staticmethod
    def check_args(args):
        """
        Check and modify the args to have the desired form.
        Returns:
            passed: bool - whether passed the tests
            chunk_size: int or None - the processed chunk_size argument
        """
        if args["chunk_size"] == 0:
            args["chunk_size"] = None

        # TODO: add more checks

        return True

    @staticmethod
    def post_optimization(
        post_iters: int,
        lr: float,
        diag_shift: float,
        save_base: str,
        n_chains_per_rank: int,
        chunk_size,
        seed: int,
        optimizer_t,
        sampler_t,
        SR_solver,
        old_vstate,
        system,
    ):
        """
        Run post_iters optimization steps with hyperparameters specified in arguments.
        Each iteration the variational state is saved and the iteration which had the minimum energy is determined, 
        returning the index corresponding to this iteration
        """
        process_print(f"Performing {post_iters} additional optimization steps...")
        start = time.time()
        #Make directory for saving vstates
        post_save_base = save_base + "post/"
        os.makedirs(post_save_base,exist_ok=False) #raises error if already exists

        saver_callback = SaveVariationalState(save_every=1, file_prefix=post_save_base)
        # New sampler for different number of chains
        sampler = sampler_t(
            system.hilbert_space,
            graph=system.graph,
            n_chains_per_rank=n_chains_per_rank,
            sweep_size=system.graph.n_nodes,
        )
        # New vstate with new sampler
        vstate = nk.vqs.MCState(
            sampler,
            model=old_vstate.model,
            n_samples=old_vstate.n_samples,
            seed=seed,
            n_discard_per_chain=old_vstate.n_samples
            // (
                n_chains_per_rank * jax.process_count()
            ),  # = n_samples_per_chain across all processes
            chunk_size=chunk_size,
        )
        vstate.variables = old_vstate.variables
        print("Sampling parameters:")
        print(f"n_chains = {vstate.sampler.n_chains}")
        print(f"n_samples = {vstate.n_samples}")
        print(f"n_discard_per_chain = {vstate.n_discard_per_chain}")
        optimizer = optimizer_t(learning_rate=lr)
        log = nk.logging.RuntimeLog()
    
        gs = nke.driver.VMC_SRt(
            system.hamiltonian.to_jax_operator(),
            optimizer,
            linear_solver_fn=SR_solver,
            diag_shift=diag_shift,
            variational_state=vstate
        )
        gs.run(
            n_iter=post_iters,  # run optimization for post_iters steps
            out=log,
            callback=saver_callback
        )
        # Now find lowest energy state
        energies = np.real(log["Energy"]["Mean"])
        print(f"Final {post_iters} energies = {energies}")
        min_index = np.arange(len(energies))[energies == min(energies)][0]
        print(f"min_index={min_index}")
        end = time.time()
        print("Finished additional optimization steps")
        print(f"Post-processing time = {end-start:.0f}s")
        return int(min_index), min(energies)

    def read_args(self, args: dict):
        """
        Set the variables self.{variable_name} = args["{variable_name}"]
        """
        for key, value in args.items():
            setattr(self, key, value)

    def __init__(self, system, network, args: dict, compile_step=True, log_mode="fail"):
        """
        Initialize all the objects for running the optimization protocol specified in args
        """
        # Check the arguments
        if not self.check_args(args):
            raise RuntimeError(
                "args failed check in check_args, because save_every does not divide iters"
            )
        # Set self.{variable_name} from args
        self.read_args(args)
        # Dependent parameters
        self.n_samples = (
            self.samples_per_rank * len(jax.devices())
        )
        self.n_discard_per_chain = int(
            self.discard_fraction
            * self.samples_per_rank
            // (self.n_chains_per_rank * len(jax.devices()))
        )
        self.compile_step = compile_step

        self.system = system
        self.network = network
        self.sampler_t = nk.sampler.MetropolisExchange
        self.optimizer_t = nk.optimizer.Sgd
        self.SR_solver = nk.optimizer.solver.pinv_smooth(
            rtol=self.r, rtol_smooth=self.r
        )

        self.sampler = self.sampler_t(
            system.hilbert_space,
            graph=system.graph,
            n_chains_per_rank=self.n_chains_per_rank,
            sweep_size=system.graph.n_nodes*self.sweep_factor,
        )
        self.log = nk.logging.JsonLog(
            self.save_base + "opt",
            mode=log_mode,
            write_every=self.save_every,
            save_params=True,
            save_params_every=self.save_every,
        )

        (
            self.lr,
            self.iters,
            self.lr_factor,
            self.diag_shift,
            self.iters,
            self.diag_shift_factor,
        ) = [
            to_sequence(arg)
            for arg in (
                self.lr,
                self.iters,
                self.lr_factor,
                self.diag_shift,
                self.iters,
                self.diag_shift_factor,
            )
        ]

        if self.symmetry_ramping:  # introducing symmetries along optimization
            print(f"Number of symmetry stages = {len(system.symmetrizing_functions)}")
            # lr and diag shift
            assert (
                len(self.iters)
                == len(system.symmetrizing_functions)
                == len(self.lr)
                == len(self.diag_shift)
                == len(self.diag_shift_factor)
                == len(self.lr_factor)
            )
            self.lr_schedulers = [
                optax.cosine_decay_schedule(
                    init_value=self.lr[i],
                    decay_steps=self.iters[i],
                    alpha=self.lr_factor[i],
                    exponent=1,
                )
                for i in range(len(self.iters))
            ]
            self.diag_shift_schedulers = [
                optax.cosine_decay_schedule(
                    init_value=self.diag_shift[i],
                    decay_steps=self.iters[i],
                    alpha=self.diag_shift_factor[i],
                    exponent=1,
                )
                for i in range(len(self.iters))
            ]
            # symmetrized networks
            self.nets = [f(network.network) for f in system.symmetrizing_functions]
        else:
            # lr and diag shift
            self.lr_schedulers = (
                optax.cosine_decay_schedule(
                    init_value=self.lr[0],
                    decay_steps=self.iters[0],
                    alpha=self.lr_factor[0],
                    exponent=1,
                ),
            )
            self.diag_shift_schedulers = (
                optax.cosine_decay_schedule(
                    init_value=self.diag_shift[0],
                    decay_steps=self.iters[0],
                    alpha=self.diag_shift_factor[0],
                    exponent=1,
                ),
            )
            # networks
            if self.symmetries:
                self.nets = (system.symmetrizing_functions[-1](network.network),)
            else:
                self.nets = (network.network,)

        self.n_symm_stages = len(self.nets)

        # Check maximum number of parameters vs no. of samples to decide whether to use SR or minSR
        params = self.nets[-1].init(
            jax.random.PRNGKey(5), jnp.zeros((1, system.graph.n_nodes))
        )
        max_nparams = nk.jax.tree_size(params)
        print(
            f"Maximum no. of parameters = {max_nparams}, total number of samples = {self.n_samples}"
        )
        use_minSR = max_nparams > self.n_samples
        print(f"Using minSR = {use_minSR}")
        if use_minSR:
            self.driver_t = lambda opt, dshift, vstate: nke.driver.VMC_SRt(
                hamiltonian=system.hamiltonian.to_jax_operator(),
                optimizer=opt,
                linear_solver_fn=self.SR_solver,
                diag_shift=dshift,
                variational_state=vstate,
            )
            # capture to call self.driver_t(optimizer, diag_shift, variational_state)
        else:
            preconditioner_t = lambda dshift: nk.optimizer.SR(
                qgt=nk.optimizer.qgt.QGTJacobianDense,
                solver=self.SR_solver,
                diag_shift=dshift,
                mode="complex",
            )
            self.driver_t = lambda opt, dshift, vstate: nk.driver.VMC(
                hamiltonian=system.hamiltonian.to_jax_operator(),
                optimizer=opt,
                variational_state=vstate,
                preconditioner=preconditioner_t(dshift),
            )
            # capture to call self.driver_t(optimizer, diag_shift, variational_state)

    def optimize(self):
        process_print("Running optimization...")
        old_vars = None  # dummy
        for i in range(self.n_symm_stages):
            print(
                f"Symmetry stage {i}/{self.n_symm_stages-1} on process {jax.process_index()}:"
            )
            while True:
                try:
                    self.vstate = nk.vqs.MCState(
                        self.sampler,
                        model=self.nets[i],
                        n_samples_per_rank=self.samples_per_rank,
                        seed=self.seed,
                        sampler_seed=self.seed,
                        n_discard_per_chain=self.n_discard_per_chain,
                        chunk_size=self.chunk_size,
                    )

                    if i > 0:
                        updated_params = add_module(
                            old_params=old_vars["params"],
                            new_params=self.vstate.variables["params"],
                        )
                        old_vars["params"] = updated_params
                        self.vstate.variables = old_vars
                        assert old_vars == self.vstate.variables

                    optimizer = self.optimizer_t(learning_rate=self.lr_schedulers[i])
                    driver = self.driver_t(
                        optimizer, self.diag_shift_schedulers[i], self.vstate
                    )

                    if self.compile_step:
                        process_print("Compiling...")
                        start_time = time.time()
                        driver.run(
                            n_iter=1,
                            out=self.log,
                            show_progress=self.show_progress,
                            timeit=self.time_it,
                            callback=self.callbacks,
                        )
                        end_time = time.time()
                        process_print(f"Compilation time = {end_time-start_time:.0f}s")

                    process_print("Running optimization...")
                    start_time = time.time()
                    all_start_time = time.time()
                    driver.run(
                        n_iter=self.iters[i],
                        out=self.log,
                        show_progress=self.show_progress,
                        timeit=self.time_it,
                        callback=self.callbacks,
                    )
                    old_vars = self.vstate.variables
                    end_time = time.time()
                    process_print(f"Optimization time = {end_time-start_time:.0f}s")
                    break  # succesfully completed without of of memory

                except jaxlib.xla_extension.XlaRuntimeError:
                    if self.chunk_size is None:
                        self.chunk_size = self.samples_per_rank
                    self.chunk_size = self.chunk_size // 2
                    if self.chunk_size <= 32:
                        raise RuntimeError(
                            "Whilst reducing chunk_size to not run out of memory, chunk_size <= 32, aborting simulation"
                        )
                    print(
                        f"Out of memory, reducing chunk_size to {self.chunk_size} and retrying"
                    )

        all_end_time = time.time()
        sim_time = all_end_time - all_start_time
        process_print("Finished optimization")
        final_lr = self.lr_schedulers[-1](driver.step_count)
        final_diag_shift = self.diag_shift_schedulers[-1](driver.step_count)

        return sim_time, final_lr, final_diag_shift

    def post_optimize(self, lr: float, diag_shift: float, n_chains_per_rank: int):
        min_index, min_energy = self.post_optimization(
            post_iters=self.post_iters,
            lr=lr,
            diag_shift=diag_shift,
            save_base=self.save_base,
            n_chains_per_rank=n_chains_per_rank,
            chunk_size=self.chunk_size,
            seed=self.seed,
            optimizer_t=self.optimizer_t,
            sampler_t=self.sampler_t,
            SR_solver=self.SR_solver,
            old_vstate=self.vstate,
            system=self.system
        )
        return min_index, min_energy

    def run(self):
        """
        Run the protocol, first performing the optimization and then computing expectation values
        in post-optimization if specified
        """
        sim_time, final_lr, final_diag_shift = self.optimize()
        n_chains_per_rank = 32
        # print(f"self.vstate = ", jax.tree.structure(self.vstate.variables))
        min_index, min_energy = self.post_optimize(
            lr=final_lr, diag_shift=final_diag_shift, n_chains_per_rank=n_chains_per_rank
        )

        # Check from arguments whether the final network is symmetrized
        if self.symmetries or self.symmetry_ramping:
            symmetrized = True
        else:
            symmetrized = False

        saver.save(
            self.system,
            self.network,
            self.save_base + "post.json",
            min_index=min_index,
            min_energy=min_energy,
            symmetrized=symmetrized,
            n_chains=n_chains_per_rank*len(jax.devices())
        )

        return sim_time, self.vstate.n_parameters
