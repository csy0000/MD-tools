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

#: A separate versioned fixture (`tests/data/alchemy/internal-v1/`), added without touching v1:
#: n-pentane, whose propyl group left unmapped by the ethyl CORE has internal 1-4 exceptions and
#: non-excluded 1-5 pairs -- the unique-group internal nonbonded terms v1 cannot see.
INTERNAL_PACKAGES = FIXTURE_ROOT.parent / "internal-v1" / "packages"
PENTANE = "LOCAL-OFBQJSOFQDEBGM/param_2ef743cf164f"

#: `tests/data/alchemy/xh-only-v1/`: methane, a ligand whose every bond is X-H.
XH_ONLY_PACKAGES = FIXTURE_ROOT.parent / "xh-only-v1" / "packages"
METHANE = "LOCAL-VNWKTOKETHGBQD/param_7e58d629eea7"

#: The ethyl fragment every fixture shares, by package atom name.
CORE = ("C1", "C2", "H1", "H2", "H3", "H4", "H5")

#: Energy agreement per force class on the Reference platform (kJ/mol). Calibrated before any
#: comparison: float64 sums over |E| ~ 1e4 kJ/mol give residuals ~1e-11; the smallest real
#: effect the accounting must resolve, the dispersion-correction shift from one dummy particle in
#: this box, is ~1e-4. 1e-7 separates the two by three orders of magnitude on each side.
ENERGY_TOL_KJ = 1e-7


def package(reference: str):
    from md_tools.ligands import load_package

    root = {PENTANE: INTERNAL_PACKAGES, METHANE: XH_ONLY_PACKAGES,
            ACETATE: CHARGED_ROOT / "packages",
            PROPANOATE: CHARGED_ROOT / "packages"}.get(reference, PACKAGES)
    return load_package(root / reference)


def water_environment():
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    return Environment.from_files(FIXTURE_ROOT / "ethane-tip3p" / "built.xml",
                                  FIXTURE_ROOT / "ethane-tip3p" / "built.pdb",
                                  LigandSelector(resname="ETA"),
                                  record=FIXTURE_ROOT / "ethane-tip3p" / "built.log")


#: `tests/data/alchemy/ethane-tip3p-v2/`: v1's ethane in TIP3P with a 2.7 nm cube, so an NPT
#: window's box fluctuations stay clear of twice the 0.9 nm cutoff.
WATER_V2_ROOT = FIXTURE_ROOT.parent / "ethane-tip3p-v2"
#: The margin a periodic fixture keeps above twice its cutoff (nm).
NPT_BOX_MARGIN_NM = 0.8


def water_environment_v2():
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    return Environment.from_files(WATER_V2_ROOT / "built.xml", WATER_V2_ROOT / "built.pdb",
                                  LigandSelector(resname="ETA"),
                                  record=WATER_V2_ROOT / "built.log")


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
    own = {k: pkg.conventions[k] for k in ("coulomb14scale", "lj14scale")}
    # Built here from the package's ffxml alone, so the System applies the package's own scales:
    # stated by the builder of the System, as an in-memory environment must.
    return Environment(system=system, topology=topology, positions_nm=positions / 10.0,
                       ligand=LigandSelector(resname=pkg.residue_name),
                       nonbonded_compatibility={"packages": [
                           {"reference": pkg.reference, **own, "applied": own}]},
                       compatibility_source="stated: System built from the package ffxml alone",
                       solvation="vacuum")


def environment_without_ligand(environment, forcefield_files=("amber14/tip3p.xml",)):
    """The environment with its ligand residue deleted, built independently by the force field.

    The reference a decoupled endpoint is measured against: at lambda 1 the plan's System must be
    this, plus the ligand's own Hamiltonian, plus the dispersion shift.
    """
    from openmm import NonbondedForce, app, unit

    ligand = environment.ligand.residues(environment.topology)[0].name
    modeller = app.Modeller(environment.topology, environment.positions_nm * unit.nanometer)
    modeller.delete([r for r in environment.topology.residues() if r.name == ligand])
    kept = [a.index for a in environment.topology.atoms() if a.residue.name != ligand]
    forcefield = app.ForceField(*forcefield_files)
    source = next(f for f in environment.system.getForces() if isinstance(f, NonbondedForce))
    constrained = environment.system.getNumConstraints() > 0
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=source.getCutoffDistance(),
        ewaldErrorTolerance=source.getEwaldErrorTolerance(),
        constraints=app.HBonds if constrained else None, rigidWater=True, removeCMMotion=False)
    next(f for f in system.getForces() if isinstance(f, NonbondedForce)).setUseDispersionCorrection(
        source.getUseDispersionCorrection())
    return system, kept


def core_map(a, b, extra=None):
    from md_tools.alchemy.topology_mapping import AtomMap

    pairs = {name: name for name in CORE}
    pairs.update(extra or {})
    return AtomMap.from_pairs(a, b, pairs)


#: `tests/data/alchemy/charged-v1/`: acetate and propanoate, both -1, and acetate + Na+ in TIP3P.
CHARGED_ROOT = FIXTURE_ROOT.parent / "charged-v1"
ACETATE = "LOCAL-QTBSBXVTEAMEQO/param_c565813e02ae"
PROPANOATE = "LOCAL-XBDQKXXYIPTUBI/param_fa700052a552"
#: acetate local name -> propanoate local name
ACETATE_TO_PROPANOATE = {"C1": "C2", "C2": "C3", "O1": "O1", "O2": "O2", "H1": "H4", "H2": "H5"}


def acetate_environment():
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    return Environment.from_files(CHARGED_ROOT / "acetate-tip3p" / "built.xml",
                                  CHARGED_ROOT / "acetate-tip3p" / "built.pdb",
                                  LigandSelector(resname="ACT"),
                                  record=CHARGED_ROOT / "acetate-tip3p" / "built.log")


#: `tests/data/alchemy/complex-v1/`: capped alanine (ff14SB) and one ethane in TIP3P, built by
#: `build-top` as a `kind: complex` structure; the ethane is chain B, resid 201.
COMPLEX_ROOT = FIXTURE_ROOT.parent / "complex-v1"
COMPLEX_FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")


def complex_environment(root=None):
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    root = root or COMPLEX_ROOT
    return Environment.from_files(root / "built.xml", root / "built.pdb",
                                  LigandSelector(chain="B", resid="201"),
                                  record=root / "built.log")


#: `tests/data/alchemy/complex-cmap-v1/`: the same structure built with ff19SB + OPC -- CMAP
#: present, and OPC's rounded 1-4 scale (0.833333) applied to every 1-4 pair, the ligand's too.
CMAP_ROOT = FIXTURE_ROOT.parent / "complex-cmap-v1"
CMAP_FORCEFIELD = ("amber19-all.xml", "amber19/opc.xml")


def independent_reference(plan, environment, pkg, side: str,
                          forcefield_files=("amber14/tip3p.xml",)):
    """The PHYSICAL endpoint built from scratch by OpenMM's force field, not from the plan.

    The environment minus its ligand, plus *pkg* as a fresh residue at the plan's coordinates,
    parameterised by `ForceField(*forcefield_files)` -- the files the environment's build record
    names -- with the package's ffxml loaded through the ligand module's own loader, under the environment's recorded nonbonded settings. Returns the
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
    forcefield = app.ForceField(*forcefield_files)
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
