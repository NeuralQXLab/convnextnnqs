import numpy as np
import glob
import netket as nk
import flax
from deepnets.net import ViT2D
import re
from collections.abc import Sequence

def get_matching(dir: str, match_str: str):
    """
    Return list of paths to all files in dir containing match_str
    """
    return glob.glob(f"{dir}*{match_str}*")


def get_int(filename: str, prefix: str) -> int:
    with open(filename) as file:
        contents = file.read()
        match = re.search(rf"{prefix} (\d+)", contents)
        if match:
            number = int(match.group(1))
            return number
        else:
            print("Number not found")

def get_mpack_paths(dir: str):
    """
    Return list of paths to all .mpack files in dir, netket variational states are saved as .mpack at the end of optimization
    """
    return glob.glob(dir + "*.mpack")

def read_from_str(
    params: dict, key: str, match_str: str, return_type=float, end_str=","
):
    """
    Read the value in the string at params[key] = string, which comes immediately after match_str
    """
    value = params[key]
    start_index = value.index(match_str) + len(match_str)
    end_index = start_index + value[start_index:].index(end_str)
    number = return_type(value[start_index:end_index])
    return number

def load_vstate(mpack_name: str, network, sampler) -> nk.vqs.VariationalState:
    var_state = nk.vqs.MCState(sampler, model=network)
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(var_state.variables, f.read())
    var_state.parameters = variables
    return var_state


def get_indices(param_seq: Sequence, key: str, match_str: str):
    """
    Get all i for which params_seq[i][key] == match_str
    """
    return [i for i, param in enumerate(param_seq) if param[key] == match_str]


def sort_indices(param_seq: Sequence, sort_key: str):
    """
    Return a list of the indices of param_seq sorted according to the value of param_seq[i][sort_key]
    """
    return [
        i
        for i, _ in sorted(
            enumerate([p[sort_key] for p in param_seq]), key=lambda x: x[1]
        )
    ]
