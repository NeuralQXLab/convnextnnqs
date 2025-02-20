import deepnets.optimization.save_load as save_load
import netket as nk
import nqxpack
import json
import jax
import numpy as np


def compute(
    dirname: str,
    net_name: str,
    n_samples_per_chain: int,
    n_chains: int,
    n_discard_per_chain: int,
    chunk_size: int,
    symmetry_stage: int = -1,
):
    """
    Load in the system, network (net_name) and vstate of the minimum energy state in {dirname}/post/checkpoint, then compute the expectation value of
    of the operators with the parameters provided. Results are saved to {dirname}/expectation_values.json.
    Returns the results_dict of form {"operator_name": expectation_value,...}
    """
    if dirname == "":
        json_path = "post.json"
        post_path = "post"
    else:
        json_path = dirname + "/post.json"
        post_path = dirname + "/post"

    min_index, system, network = save_load.load(json_path, net_name, symmetry_stage = symmetry_stage)
    print(network)
    sampler = nk.sampler.MetropolisExchange(
        system.hilbert_space, graph=system.graph
    )
    
    vstate_load = nqxpack.load(f"{post_path}/vstate{min_index}.nk")
    sampler = nk.sampler.MetropolisExchange(
        system.hilbert_space, graph=system.graph, n_chains=n_chains
    )
    vstate = nk.vqs.MCState(sampler,model=network)
    vstate.variables = vstate_load.variables
    operators = {
        "energy": system.hamiltonian,
    }
    vstate.n_samples = n_samples_per_chain * n_chains
    vstate.n_discard_per_chain = n_discard_per_chain
    vstate.chunk_size = chunk_size
    print(f"vstate.chunk_size = {vstate.chunk_size}")
    print(f"vstate.n_chains = {vstate.sampler.n_chains}")
    print(f"vstates.n_samples = {vstate.n_samples}")
    print(f"vstate.n_discard_per_chain = {vstate.n_discard_per_chain}")
    if chunk_size:
        vstate.chunk_size = chunk_size
    results_dict = {
        "n_chains": vstate.sampler.n_chains,
        "n_samples": vstate.n_samples,
        "n_discard_per_chain": vstate.n_discard_per_chain,
    }
    for name, operator in operators.items():
        result = vstate.expect(operator.to_jax_operator())
        result_dict = (
            result.__dict__
        )  # convert all of the attributes and their values to a dictionary
        # Convert to types compatible with json
        for key, value in result_dict.items():
            if isinstance(value, jax.Array):
                result_dict[key] = float(
                    np.real(complex(value))
                )  # cannot go directly from jax.Array with complex dtype to float, so take real part
        results_dict[name] = result_dict

    if dirname == "":
        save_file = "expectation_values.json"
    else:
        save_file = dirname + "/expectation_values.json"

    # Save all results
    with open(save_file, "a") as f:
        json.dump(results_dict, f)

    return results_dict
