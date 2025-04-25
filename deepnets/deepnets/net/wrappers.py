# Wrap the neural networks to define parameters and save
from deepnets.net.base_wrapper import NetBase
import argparse
import deepnets.system as system
from deepnets.net import ConvNext

class ConvNext(NetBase):
    nets = {
        "Vanilla": ConvNext.ConvNextVanilla,
        "FT": ConvNext.ConvNextFT,
        "NoPatching": ConvNext.ConvNext_nopatching,
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--n_blocks",
            type=int,
            nargs="+",
            required=True,
            help="Number of blocks in each stage format --depth stage1 stage2 ...",
        )
        parser.add_argument(
            "--features",
            type=int,
            nargs="+",
            required=True,
            help="Number of features after downsampling in each stage, format --features stage1 stage2 ...",
        )
        parser.add_argument(
            "--expansion_factor",
            type=int,
            required=True,
            help="Expansion factor of number features in every block",
        )
        parser.add_argument(
            "--net_type",
            type=str,
            required=True,
            help="The type of network to use (Vanilla, SkyNet or NoPatching)",
        )
        parser.add_argument(
            "--kernel_width",
            type=int,
            required=True,
            help="Width of all downsampling convolutional kernels",
        )
        parser.add_argument(
            "--downsample_factor",
            type=int,
            required=True,
            help="Stride for downsampling convolutions, i.e the linear size of patches",
        )
        parser.add_argument(
            "--final_features",
            type=int,
            required=True,
            help="Number of features in output head",
        )
        parser.add_argument(
            "--init_kernel_width",
            type=int,
            default=0,
            help="Width of initial downsampling kernel if not patching",
        )
        parser.add_argument(
            "--output_depth", type=int, default=1, help="Depth of output head"
        )
        parser.add_argument(
            "--q",
            type = float,
            action = "append",
            required = False,
            help = "Momentum of quantum state in units of pi"
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return (
            tuple(d for d in args.n_blocks),
            tuple(f for f in args.features),
            args.expansion_factor,
            args.net_type,
            args.kernel_width,
            args.downsample_factor,
            args.final_features,
            args.init_kernel_width,
            args.output_depth,
            args.q
        )

    def __init__(
        self,
        n_blocks: tuple,
        features: tuple,
        expansion_factor: int,
        net_type: str,
        kernel_width: int,
        downsample_factor: int,
        final_features: int,
        init_kernel_width: int,
        output_depth: int,
        q: tuple[float],
        system: system.SpinHalf,
    ):
        self.name = "ConvNext"
        self.lattice_shape = (system.L, system.L)
        self.n_blocks = n_blocks
        self.features = features
        self.expansion_factor = expansion_factor
        self.net_type = net_type
        self.kernel_width = kernel_width
        self.downsample_factor = downsample_factor
        self.final_features = final_features
        self.init_kernel_width = init_kernel_width
        self.output_depth = output_depth
        self.q = q

        if net_type == "Vanilla":
            print(f"Using net_type={net_type}")
            for i in range(len(self.lattice_shape)):
                assert (
                    self.lattice_shape[i] >= self.kernel_width * self.downsample_factor
                )  # check that kernel_size is not bigger than the lattice after patching

            self.network = self.nets[self.net_type](
                lattice_shape=self.lattice_shape,
                n_blocks=n_blocks,
                features=features,
                expansion_factor=expansion_factor,
                kernel_size=(kernel_width, kernel_width),
                downsample_factor=downsample_factor,
                final_features=final_features,
                extract_patches=system.extract_patches_as2d,
                output_depth=output_depth,
            )

        elif net_type == "NoPatching":
            print(f"Using net_type= NoPatching")
            self.network = self.nets[self.net_type](
                init_kernel_size=(init_kernel_width, init_kernel_width),
                n_blocks=n_blocks,
                features=features,
                expansion_factor=expansion_factor,
                kernel_size=(kernel_width, kernel_width),
                unitcell_shape=(downsample_factor, downsample_factor),
                lattice_shape=self.lattice_shape,
                final_features=final_features,
                reshape_function=system.reshape_xy,
            )

        elif net_type == "FT":
            print(f"Using net_type= FT")
            self.network = self.nets[self.net_type](
                n_blocks=n_blocks,
                features=features,
                expansion_factor=expansion_factor,
                kernel_size=(kernel_width, kernel_width),
                final_features=final_features,
                extract_patches=system.extract_patches_as2d,
                q=q,
                compute_positions=system.compute_positions,
            )

        else:
            raise NotImplementedError(
                "net_types implemented are Vanilla, FT or NoPatching"
            )

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ (except system) to a dictionary
        """
        arg_dict = {
            "name": self.name,
            "n_blocks": self.n_blocks,
            "features": self.features,
            "expansion_factor": self.expansion_factor,
            "net_type": self.net_type,
            "kernel_width": self.kernel_width,
            "downsample_factor": self.downsample_factor,
            "final_features": self.final_features,
            "init_kernel_width": self.init_kernel_width,
            "output_depth": self.output_depth,
            "q": self.q,
        }
        return arg_dict


networks = {"ConvNext": ConvNext}


def from_dict(arg_dict: dict, system, network_name="ConvNext"):
    """
    Return the wrapped network specified by the dictionary
    """
    try:
        network = networks[str(arg_dict["name"])]
        del arg_dict["name"]
    except KeyError:  # compatibility with old versions where it wasnt saved
        network = networks[network_name]
        arg_dict["net_type"] = arg_dict["output_head"]
        del arg_dict["output_head"]
        arg_dict["init_kernel_width"] = 1
    try:  # stupid fix for these being saved as lists
        arg_dict["n_blocks"] = tuple(arg_dict["n_blocks"])
        arg_dict["features"] = tuple(arg_dict["features"])
    except KeyError:
        pass
    # print(arg_dict)
    if "downsample_factor" in arg_dict.keys():
            del arg_dict["downsample_factor"]
    if not "output_depth" in arg_dict.keys():
        arg_dict["output_depth"] = 1
    if not "q" in arg_dict.keys():
        arg_dict["q"] = [0,0]

    try:
        if isinstance(arg_dict["n_blocks"], (tuple, list)):
            arg_dict["n_blocks"] = arg_dict["n_blocks"][0]
        if isinstance(arg_dict["features"], (tuple, list)):
            arg_dict["features"] = arg_dict["features"][0]
    except KeyError:
        pass
    
    return network(**arg_dict, system=system)


def load(file_name: str, system, prefix: str = None):
    """
    Return the wrapped network specified by the dictionary, dict[prefix], contained in
    the json file file_filename
    """
    arg_dict = NetBase.argument_loader(file_name, prefix)
    loaded_network = from_dict(arg_dict, system)
    return loaded_network
