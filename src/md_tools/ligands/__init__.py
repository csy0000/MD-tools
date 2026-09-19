"""Reusable small-molecule parameter packages, and the mapping of ligand instances onto them.

A compound keeps IDENTICAL parameters in every environment it is simulated in -- water, a
membrane, an organic solvent, a protein pocket -- so that results from those environments are
comparable. That is only true if the parameters are generated once, saved as they are, and loaded
by every build, rather than regenerated per build from a recipe whose output depends on a
conformer, a charge backend and a toolkit version. One measured exception: under OPC water the
System applies the water XML's 1-4 Coulomb scale, `0.833333`, where the package says 5/6 (about
4e-7 relative), and the build record names the value applied (`docs/ligand-packages.md`).

Three identities are kept apart (see `docs/ligand-packages.md`):

  compound     `CHEMBL112` -- which substance, with searchable aliases. Tautomers and protomers
               of one substance share it.
  parameters   `param_<12 hex>` -- one exact chemical state AND one saved parameter set, derived
               from their contents. Never read from a directory name.
  instance     one ligand in one structure, selected by chain, residue number, insertion code
               and assembly copy.

Nothing in this package runs dynamics. Charge generation happens in exactly one function,
`package.create_package`, and loading a package never calls it.
"""

from .identity import (CompoundIdError, chemical_state, check_compound_id,
                       local_compound_id)
from .package import (LigandPackage, PackageError, create_package, import_package_from_system,
                      load_package)

__all__ = [
    "CompoundIdError",
    "LigandPackage",
    "PackageError",
    "chemical_state",
    "check_compound_id",
    "create_package",
    "import_package_from_system",
    "load_package",
    "local_compound_id",
]
