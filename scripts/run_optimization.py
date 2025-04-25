#Run optimization with:
# uv run run_optimization.py --config_file config_opt.yaml --other_optional_parameters
# or
# python run_optimization.py --config_file config_opt.yaml --other_optional_parameters

import deepnets.config.processes
import deepnets.optimization.protocols as opt
import deepnets.config.args as args
from deepnets.data import save as saver
from deepnets.system import SquareHeisenberg, ShastrySutherland
from deepnets.net import ConvNext
import netket as nk
import jax

systems = {
    "SquareHeisenberg": SquareHeisenberg,
    "ShastrySutherland": ShastrySutherland,
}
networks = {"ConvNext": ConvNext}

if __name__ == "__main__":
    # For command line arguments
    parser = args.parser
    # Get which system and network class from command line arguments
    parser.add_argument(
        "--system",
        type=str,
        help="Which system class to use, options are: SquareHeisenberg, ShastrySutherland",
    )
    parser.add_argument(
        "--network",
        type=str,
        help="Which network to use, options are: resCNN, ViT2D, ConvNext",
    )

    args_setup, remaining_input = parser.parse_known_args()
    print("args_setup:", args_setup)
    system_class = systems[args_setup.system]
    network_class = networks[args_setup.network]

    # Get parameters for system, network and optimization
    system_class.add_arguments(parser)
    network_class.add_arguments(parser)
    opt.Protocol.add_arguments(parser)
    parser.add_argument(
        "--double_precision",
        type=int,
        default=1,
        help="Whether to use double precision (>=1) or single precision (0)",
    )
    args = parser.parse_args(remaining_input)
    print("args:", args)

    # Process the command line parameters
    nk.config.netket_enable_x64 = bool(args.double_precision)
    opt.process_print("Double precision enabled = ", nk.config.netket_enable_x64)
    system_args = system_class.read_arguments(args)
    system = system_class(*system_args)
    network_args = network_class.read_arguments(args)
    network = network_class(*network_args, system)

    # Initialize the protocol
    protocol = opt.Protocol(
        system,
        network,
        vars(args),
        compile_step=True,
        log_mode="write"
    )
    # Run it
    sim_time, n_parameters = protocol.run()

    # Save additional information to log file {args.save_base}opt.log, where results have already been saved
    if jax.process_index() == 0:
        fname = f"{protocol.save_base}opt"
        print(f"Saving simulation results to {fname}.log")
        if system.L <= 4:
            print("Diagonalizing for comparison...")
            E_gs = nk.exact.lanczos_ed(system.hamiltonian)
            print(f"Ground state energy = {E_gs}")
            print("Finished diagonalization")
            saver.write_attributes_json(
                filename=fname + ".log", group_name="Exact", E_gs=E_gs
            )
        saver.write_attributes_json(
            filename=fname + ".log",
            group_name="distrbuted_processes",
            ranks=jax.process_count(),
        )
        saver.write_attributes_json(
            filename=fname + ".log", group_name="Time", optimization=sim_time
        )
        saver.write_attributes_json(
            filename=fname + ".log", group_name="Network", n_parameters=n_parameters
        )
        saver.write_json(
            filename=fname + ".log",
            gnames=["Arguments", "System", "Network"],
            objs=[args, system, network],
        )

    print(f"Finished on process {jax.process_index()}")
