# 2D spin models
import netket as nk
import numpy as np
import argparse
import warnings
import jax.numpy as jnp
import einops

from collections.abc import Sequence
from deepnets.system.base import SpinHalf
from deepnets.system.utils import reflect_and_translate, reflect_and_translate_group
from netket.utils.group._point_group import PointGroup
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import SpaceGroupBuilder
from netket.utils.types import Array
from netket.nn.blocks import SymmExpSum
from deepnets.nn.blocks import FlipExpSum


class SquareHeisenberg(SpinHalf):
    rotation_group = nk.utils.group.planar.C(4)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--L", type=int, required=True, help="Linear size of the square lattice"
        )
        parser.add_argument(
            "--J", type=float, action="append", required=True, help="List of J values"
        )
        parser.add_argument(
            "--sign_rule",
            type=int,
            action="append",
            required=True,
            help="List of boolean sign rules for each J value",
        )
        parser.add_argument(
            "--patching",
            type=int,
            required=True,
            help="Whether using patching for network",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return args.L, args.J, [bool(i) for i in args.sign_rule], args.patching

    @staticmethod
    def extract_patches_as1d(x, b):
        """
        Extract bxb patches from the (nbatches, nsites) input x.
        Returns x reshaped to (nbatches,npatches,patch_size), where patch_size = b**2
        """
        batch = x.shape[0]
        L_eff = int((x.shape[1] // b**2) ** 0.5)
        x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
        x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
        # flatten the patches
        x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
        x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
        return x

    @staticmethod
    def extract_patches_as2d(x: Array, b: int, lattice_shape=None) -> Array:
        """
        Extract bxb patches from the (nbatches, nsites) input x, lattice_shape is a dummy argument so the call has a consistent pattern
        across systems.
        Returns x reshaped to (nbatches,x,y,patch_size) where x,y are x,y coordinates of the patch_size = b**2 patches
        For the square lattice with site indexing I = j + i*Ly.
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        batch = x.shape[0]
        L_eff = int((x.shape[-1] // b**2) ** 0.5)
        x = x.reshape(batch, L_eff, b, L_eff, b)
        x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
        x = x.reshape(batch, L_eff, L_eff, b**2)  # [L_eff, L_eff, b*b]
        return x

    @staticmethod
    def reshape_xy(x: Array, lattice_shape: tuple) -> Array:
        """
        Reshape a (nbatch, nsites) array into an (nbatch,x,y,1) array, which can then have a 2d convolutional layer applied, where (x,y) label the real space coordinates of the point
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        x = x.reshape(
            (x.shape[0], lattice_shape[0], lattice_shape[1])
        )  # (nbatch, x, y)
        x = x.reshape(x.shape + (1,))  # shape (nbatch, x, y, 1)
        return x

    def __init__(
        self,
        L: int,
        J: Sequence[float],
        sign_rule: Sequence[bool] = [False],
        patching: bool = True,
        sz_sector=0,
    ):
        super().__init__(N=int(L**2), L=L, J=J, sz_sector=sz_sector)
        self.name = "SquareHeisenberg"
        self.patching = patching
        self.graph = nk.graph.Square(length=L, max_neighbor_order=len(J), pbc=self.pbc)
        self.graph_name = "Square"
        # Get all the symmetries we will use to symmetrize the wavefunction
        self.graph_symmetries = {
            "C4": self.graph.point_group(self.rotation_group),
            "Full point group": self.graph.point_group(),
        }
        # Get translation symmetries if patching
        if patching:
            spacegroupbuilder = SpaceGroupBuilder(
                self.graph, nk.utils.group.trivial_point_group(ndim=2)
            )
            trans_x = spacegroupbuilder._translations_along_axis(0)[1]  # translation +x
            trans_y = spacegroupbuilder._translations_along_axis(1)[1]  # translation +y
            trans_xy = trans_x @ trans_y  # translation +x+y
            translation_group = PermutationGroup(
                [Identity(), trans_x, trans_y, trans_xy], degree=self.graph.n_nodes
            )
            self.graph_symmetries.update(
                {
                    "Translation": translation_group,
                    "T@C4": translation_group
                    @ self.graph_symmetries[
                        "C4"
                    ],  # group consisting of rotations and translations
                    "T@Full": translation_group
                    @ self.graph_symmetries[
                        "Full point group"
                    ],  # Full point group and translations
                }
            )

        if (
            not patching
        ):  # unsymmetrized + unsymmetrized + C4 + full point + spin parity
            self.symmetrizing_functions = (
                lambda net: net,
                lambda net: net,
                lambda net: SymmExpSum(net, self.graph_symmetries["C4"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["Full point group"]),
                lambda net: FlipExpSum(
                    SymmExpSum(net, self.graph_symmetries["Full point group"])
                ),
            )
        else:  # unsymmetrized + translations + C4 + full point + spin parity
            self.symmetrizing_functions = (
                lambda net: net,
                lambda net: SymmExpSum(net, self.graph_symmetries["Translation"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["T@C4"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["T@Full"]),
                lambda net: FlipExpSum(
                    SymmExpSum(net, self.graph_symmetries["T@Full"])
                ),
            )

        if len(sign_rule) != len(J):
            warnings.warn(
                "len(sign_rule) and len(J) mismatched, increasing length of sign_rule by repeating first value..."
            )
            sign_rule = len(J) * [sign_rule[0]]
        self.sign_rule = sign_rule
        self.hamiltonian = nk.operator.Heisenberg(
            hilbert=self.hilbert_space,
            graph=self.graph,
            J=self.J,
            sign_rule=self.sign_rule,
        )
        self.hamiltonian_name = "Heisenberg"

    def name_and_arguments_to_dict(self):
        return {
            "name": self.name,
            "L": self.L,
            "J": self.J,
            "sign_rule": self.sign_rule,
            "patching": self.patching,
        }


class ShastrySutherland(SpinHalf):
    basis_vecs = np.array([[2, 0], [0, 2]])
    unit_cell = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    custom_edges = [
        (0, 1, [1, 0], 0),
        (0, 1, [-1, 0], 0),
        (0, 2, [0, 1], 0),
        (0, 2, [0, -1], 0),
        (1, 3, [0, 1], 0),
        (1, 3, [0, -1], 0),
        (2, 3, [1, 0], 0),
        (2, 3, [-1, 0], 0),
        (0, 3, [-1, 1], 1),
        (2, 1, [1, 1], 1),
    ]
    rotation_group = nk.utils.group.planar.C(4)
    reflection_group = reflect_and_translate_group(
        45, np.array([1, -1])
    )  # {I, sigma_xy}
    reflection = PointGroup(
        [reflect_and_translate(45, np.array([1, -1]))], ndim=2
    )  # sigma_xy
    point_group = nk.utils.group._point_group.product(
        rotation_group, reflection_group
    )  # C4 x {I, sigma_xy}, full point group length 8
    glide_group = nk.utils.group._point_group.product(
        rotation_group, reflection
    )  # C4 x sigma_xy, glide group, reflections*rotations, length 4

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--L",
            type=int,
            required=True,
            help="Linear size of the underlying square lattice",
        )
        parser.add_argument(
            "--J", type=float, action="append", required=True, help="List of J values"
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return args.L, args.J

    @staticmethod
    def get_graph(L: int) -> nk.graph.Graph:
        return nk.graph.Lattice(
            basis_vectors=ShastrySutherland.basis_vecs,
            site_offsets=ShastrySutherland.unit_cell,
            custom_edges=ShastrySutherland.custom_edges,
            extent=[L // 2, L // 2],
            pbc=True,
            point_group=ShastrySutherland.point_group,
        )

    @staticmethod
    def extract_patches_as1d(x: Array, b: int = 2) -> Array:
        """
        Extract flattened b x b patches from the (nbatch,nsites) input x on the Shastry-Sutherland lattice.
        A 2 x 2 patch should correspond to the unit cell, i.e a plaquette without diagonal bond.
        Returns x reshaped to (nbatch,npatches,patch_size)
        """
        patch_size = int(b**2)
        Npatches = x.size // patch_size
        x = x.reshape(Npatches, patch_size)
        return x

    @staticmethod
    def extract_patches_as2d(x: Array, b: int, lattice_shape: tuple) -> Array:
        """
        Reshape a (...,nsites) array into (...,x,y,b**2) where x and y are coordinates of the patch of b**2 sites
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        batch_size = x.shape[0]
        Lx = lattice_shape[0] // b
        Ly = lattice_shape[1] // b
        x = x.reshape(batch_size, Lx, Ly, b**2)
        return x

    @staticmethod
    def reshape_xy(x: Array, lattice_shape: tuple) -> Array:
        """
        Reshape a (nbatch, nsites) array into an (nbatch,x,y,1) array, which can then have a 2d convolutional layer applied, where (x,y) label the real space coordinates of the point
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        x = ShastrySutherland.extract_patches_as2d(
            x=x, b=2, lattice_shape=lattice_shape
        )  # -> (nbatch, ux, uy, 4) where ux,uy label unit cell coords
        x = x.reshape(
            (x.shape[0], lattice_shape[0] // 2, lattice_shape[1] // 2, 2, 2)
        )  # (nbatch, ux,uy, dy, dx) , where dx, dy are positions within the unit cell ux,uy
        x = einops.rearrange(
            x, "batch ux uy dy dx -> batch (ux dx) (uy dy)"
        )  # shape (nbatch, x, y)
        x = x.reshape(x.shape + (1,))  # shape (nbatch, x, y, 1)
        return x
    
    def __init__(self, L: int, J: Sequence[float], sz_sector=0):
        if len(J) != 2:
            raise ValueError("Shastry-Sutherland model requires J1 and J2")
        super().__init__(N=int(L**2), L=L, J=J, sz_sector=sz_sector)
        self.name = "ShastrySutherland"
        self.graph = nk.graph.Lattice(
            basis_vectors=self.basis_vecs,
            site_offsets=self.unit_cell,
            custom_edges=self.custom_edges,
            extent=[L // 2, L // 2],
            pbc=self.pbc,
            point_group=self.point_group,
        )
        self.graph_name = "ShastrySutherland"
        self.graph_symmetries = {
            "C4": self.graph.point_group(self.rotation_group),
            "Glides": self.graph.point_group(self.glide_group),
            "Full point group": self.graph.point_group(),
        }
        # Define the symmetrizing_functions used in a symmetry ramping optimization
        self.symmetrizing_functions = (
            lambda net: net,  # unsymmetrized
            lambda net: SymmExpSum(net, self.graph_symmetries["C4"]),  # rotations
            lambda net: SymmExpSum(
                net, self.graph_symmetries["Full point group"]
            ),  # rotations and glides = full point group
            lambda net: FlipExpSum(
                SymmExpSum(net, self.graph_symmetries["Full point group"])
            ),  # S^z parity and full point group
        )

        self.hamiltonian = nk.operator.Heisenberg(
            hilbert=self.hilbert_space, graph=self.graph, J=self.J
        )
        self.hamiltonian_name = "Heisenberg"
        self.positions = self.graph.positions

    def compute_positions(self) -> Array:
        """
            Return the positions of the centre of patches in shape (npx,npy, 2).
            The order of patches is the same as that from extract_patches_to2d
        """
        patched_positions = self.extract_patches_as2d(self.positions.transpose()) #(2,npx,npy,patch_size)
        patch_positions = jnp.mean(patched_positions,axis=-1) #(2,npx,npy), gives centre of sites making up patch
        return patch_positions.transpose((1,2,0)) #(npx,npy,2)

systems = {
    "Square_Heisenberg": SquareHeisenberg, #compatibility with old naming
    "Shastry_Sutherland": ShastrySutherland, #compatibility with old naming
    "SquareHeisenberg": SquareHeisenberg,
    "ShastrySutherland": ShastrySutherland
}


def from_dict(arg_dict: dict):
    """
    Return the system specified by the dictionary
    """
    try:
        system = systems[str(arg_dict["name"])]
        del arg_dict["name"]
    except KeyError:
        system = systems[str(arg_dict["Name"])]
        del arg_dict["Name"]
    return system(**arg_dict)


def load(file_name: str, prefix: str = None):
    """
    Return the system specified by the dictionary, dict[prefix], contained in
    the json file file_filename
    """
    arg_dict = SpinHalf.argument_loader(file_name, prefix)
    loaded_system = from_dict(arg_dict)
    return loaded_system
