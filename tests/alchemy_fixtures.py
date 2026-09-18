"""The miniature alchemical endpoint fixtures (version 1), and helpers every 0.7.0 test shares.

`tests/data/alchemy/v1/` holds three registered-form parameter packages made by
`md-openmm build-top --parameterize` with AM1-BCC (sqm) and openff-2.2.1, and one environment made
by `md-openmm build-top` from the ethane package: ethane in a 1.9 nm cube of TIP3P, PME, 0.9 nm
cutoff, HBonds, rigid water, no ions. See `tests/data/alchemy/README.md` for the commands.

    ETHANE        CC      LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4   residue ETA
    CHLOROETHANE  CCCl    LOCAL-HRYZWHHZPQKTII/param_5930c10b0577   residue CLE
    ETHANOL       CCO     LOCAL-LFQSCWFLJHTTHZ/param_d66453ef683a   residue EOH

All three share atom names C1 C2 H1..H5 for the ethyl fragment; ethane's H6 sits where chloroethane
has Cl1 and ethanol has O1 (whose hydrogen is ethanol's H6). So `CORE` below is a map by name that
a person can check against the three `.pdb` files in a minute, in both directions.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "alchemy" / "v1"
PACKAGES = FIXTURE_ROOT / "packages"
FIXTURE_VERSION = "alchemy-endpoints/1"

ETHANE = "LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4"
CHLOROETHANE = "LOCAL-HRYZWHHZPQKTII/param_5930c10b0577"
ETHANOL = "LOCAL-LFQSCWFLJHTTHZ/param_d66453ef683a"

#: The ethyl fragment every fixture shares, by package atom name.
CORE = ("C1", "C2", "H1", "H2", "H3", "H4", "H5")

#: Energy agreement per force class on the Reference platform (kJ/mol). Calibrated before any
#: comparison: float64 sums over |E| ~ 1e4 kJ/mol give residuals ~1e-11; the smallest real
#: effect the accounting must resolve, the dispersion-correction shift from one dummy particle in
#: this box, is ~1e-4. 1e-7 separates the two by three orders of magnitude on each side.
ENERGY_TOL_KJ = 1e-7


def package(reference: str):
    from md_tools.ligands import load_package

    return load_package(PACKAGES / reference)


def water_environment():
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    return Environment.from_files(FIXTURE_ROOT / "ethane-tip3p" / "built.xml",
                                  FIXTURE_ROOT / "ethane-tip3p" / "built.pdb",
                                  LigandSelector(resname="ETA"))


def vacuum_environment(pkg, *, constraints=None):
    """The package molecule alone, NoCutoff, built by OpenMM from the package's own ffxml."""
    from openmm import app

    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector
    from md_tools.ligands.parameters import topology_for_molecule

    topology = topology_for_molecule(pkg.mol, pkg.atom_names, pkg.residue_name)
    forcefield = app.ForceField(io.StringIO(pkg.ffxml_text))
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff,
                                     constraints=constraints, removeCMMotion=False,
                                     residueTemplates={next(topology.residues()):
                                                       pkg.template_name})
    conformer = pkg.mol.GetConformer()
    positions = np.array([list(conformer.GetAtomPosition(i)) for i in range(pkg.mol.GetNumAtoms())])
    return Environment(system=system, topology=topology, positions_nm=positions / 10.0,
                       ligand=LigandSelector(resname=pkg.residue_name))


def core_map(a, b, extra=None):
    from md_tools.alchemy.topology_mapping import AtomMap

    pairs = {name: name for name in CORE}
    pairs.update(extra or {})
    return AtomMap.from_pairs(a, b, pairs)


def independent_reference(plan, environment, pkg, side: str):
    """The PHYSICAL endpoint built from scratch by OpenMM's force field, not from the plan.

    The environment minus its ligand, plus *pkg* as a fresh residue at the plan's coordinates,
    parameterised by `ForceField(amber14/tip3p.xml)` with the package's ffxml loaded through the
    ligand module's own loader, under the environment's recorded nonbonded settings. Returns the
    System and, for each of its particles, the plan particle it is.
    """
    from openmm import NonbondedForce, app, unit

    from md_tools.ligands.mapping import load_packages_into
    from md_tools.ligands.parameters import topology_for_molecule

    hyb = plan.record["endpoints"][side]["hybrid_index_of_local_atom"]
    ligand_name = plan.record["endpoints"]["A"]["environment_residue"]["name"]
    x = plan.positions_nm
    env_nb = next(f for f in environment.system.getForces() if isinstance(f, NonbondedForce))
    periodic = environment.system.usesPeriodicBoundaryConditions()
    modeller = app.Modeller(environment.topology, environment.positions_nm * unit.nanometer)
    modeller.delete([r for r in environment.topology.residues() if r.name == ligand_name])
    kept = [a.index for a in environment.topology.atoms() if a.residue.name != ligand_name]
    modeller.add(topology_for_molecule(pkg.mol, pkg.atom_names, pkg.residue_name),
                 x[hyb] * unit.nanometer)
    forcefield = app.ForceField("amber14/tip3p.xml")
    load_packages_into(forcefield, [pkg])
    residue = [r for r in modeller.topology.residues() if r.name == pkg.residue_name][-1]
    constrained = environment.system.getNumConstraints() > 0
    kwargs = dict(constraints=app.HBonds if constrained else None, rigidWater=True,
                  removeCMMotion=False, residueTemplates={residue: pkg.template_name})
    if periodic:
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=app.PME,
            nonbondedCutoff=env_nb.getCutoffDistance(),
            ewaldErrorTolerance=env_nb.getEwaldErrorTolerance(), **kwargs)
        nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        nb.setUseDispersionCorrection(env_nb.getUseDispersionCorrection())
        nb.setUseSwitchingFunction(env_nb.getUseSwitchingFunction())
    else:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, **kwargs)
    return system, kept + list(hyb)
