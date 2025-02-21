#Run expectation value computation with:
# uv run run_expvalues.py --config_file example_config_expvalues.yaml --other_optional_parameters
# or
# python run_expvalues.py --config_file example_config_expvalues.yaml --other_optional_parameters

import deepnets.config.processes
import deepnets.config.args as args
from deepnets.expectation_value import expectation_value
import numpy as np
import time
import argparse

if __name__ == "__main__":
    parser = args.parser

    parser.add_argument(
        "--directory",
        type=str,
        help="path to directory containing post.json and post directory (output of optimization)",
    )
    parser.add_argument(
        "--net",
        type=str,
        help="Name of neural network used, to load back in (ConvNext,...)",
    )
    parser.add_argument(
        "--n_samples_per_chain",
        type=int,
    )
    parser.add_argument(
        "--n_chains",
        type=int
    )
    parser.add_argument("--n_discard_per_chain", type=int)
    parser.add_argument("--chunk_size", type =int)
    parser.add_argument("--symmetry_stage", type=int, default=-1)

    args = parser.parse_args()
    print("Arguments:", args)
    print("Computing expectation values...")
    start = time.time()
    results = expectation_value.compute(
        dirname=args.directory,
        net_name=args.net,
        n_samples_per_chain=args.n_samples_per_chain,
        n_chains=args.n_chains,
        n_discard_per_chain=args.n_discard_per_chain,
        chunk_size=args.chunk_size,
        symmetry_stage = args.symmetry_stage
    )
    end = time.time()
    print("Finished computing expectation values")
    print(f"Time taken {end-start:.1f}s")
