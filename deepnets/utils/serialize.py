import flax
import netket as nk
import json
import deepnets.system.system as sys
import deepnets.net.wrappers as net

def load_variables(mpack_name: str, vstate) -> nk.vqs.VariationalState:
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(vstate.variables, f.read())
    vstate.variables = variables
    return vstate

def save_variables(mpack_name: str, vstate):
    with open(mpack_name,"wb") as f:
        f.write(flax.serialization.to_bytes(vstate.variables))

def save(system, network, fname: str, **kwargs):
    kwarg_dict = kwargs  # kwargs is a dictionary of form
    system_dict = system.name_and_arguments_to_dict()
    net_dict = network.name_and_arguments_to_dict()
    save_dict = {"post": kwarg_dict, "system": system_dict, "network": net_dict}
    with open(fname, "w+") as f:
        json.dump(save_dict, f)


def load(fname: str, net_str: str, symmetry_stage = -1):
    with open(fname, "r") as f:
        load_dict = json.load(f)
    min_index = int(load_dict["post"]["min_index"])
    symmetrized = bool(load_dict["post"]["symmetrized"])
    system = sys.load(fname,"system")
    network = net.load(fname,system,"network")
    if symmetry_stage == -1: #return final symmetrized network
        if symmetrized:
            network = system.symmetrizing_functions[-1](network.network)
    else:
        network = system.symmetrizing_functions[symmetry_stage](network.network)
    return min_index, system, network