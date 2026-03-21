import matplotlib as mpl
import matplotlib.pyplot as plt
from ase import Atoms as ASEAtoms
from ase.visualize.plot import plot_atoms
from matplotlib.patches import Patch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser


def mp_20_plot(cif: str) -> plt.Axes:
    """Visualize a single datapoint from the MP-20 dataset using ASE's plot_atoms."""
    parser = CifParser.from_str(cif_string=cif)
    structure = parser.parse_structures(primitive=True)[0]
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # Optional runtime check
    if not isinstance(atoms, ASEAtoms):
        msg = f"Expected ASE Atoms object, got {type(atoms)}"
        raise TypeError(msg)

    fig, ax = plt.subplots()

    # Get unique symbols
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(set(symbols))

    # Automatically assign colors using a colormap
    cmap = mpl.colormaps["tab20"](range(len(unique_symbols)))  # Up to 20 colors
    color_map = {s: cmap[i] for i, s in enumerate(unique_symbols)}

    # Map atom colors
    alpha_atoms = 0.7  # Opacity for atoms
    atom_colors = [
        (r, g, b, alpha_atoms) for s in symbols for r, g, b, _ in [color_map[s]]
    ]

    # Legend
    legend_elements = [Patch(facecolor=color_map[s], label=s) for s in unique_symbols]
    ax.legend(handles=legend_elements, loc="upper right", title="Atom types")

    plot_atoms(atoms, ax=ax, rotation="45x, 35y, 0z", colors=atom_colors)
    fig.suptitle(str(atoms.symbols))
    ax.set_axis_off()

    return ax
