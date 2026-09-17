"""What a parameter package's ffxml actually says, as one canonical, hashable table.

The parameter identity of a package is derived from the PARAMETERS, not from the bytes of the
ffxml file and not from the name of a directory. Two ffxml files that differ only in whitespace,
attribute order or generated type names describe the same Hamiltonian; two that differ in one
charge do not. So the file is loaded into OpenMM, a System is created for the molecule alone, and
every per-atom and per-term value is read back off that System in package atom order. That table
is what is hashed, and what a loaded package is re-verified against.

Supported force terms are exactly those the small-molecule force fields this package loads
produce: NonbondedForce (charges, Lennard-Jones, 1-4 exceptions), HarmonicBondForce,
HarmonicAngleForce and PeriodicTorsionForce. Anything else -- a virtual site, a CMAP, a custom
force -- is REFUSED rather than dropped, because a conversion that loses a term produces a package
that loads, runs, and is a different molecule.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
from typing import Any, Sequence

__all__ = [
    "PARAMETER_TABLE_SCHEMA",
    "SUPPORTED_FORCES",
    "UnsupportedParametersError",
    "canonical_json",
    "classify_torsion",
    "ligand_system",
    "parameter_digest",
    "parameter_table",
    "rename_ffxml",
    "topology_for_molecule",
]

PARAMETER_TABLE_SCHEMA = "md-tools-ligand-parameters/1"

SUPPORTED_FORCES = ("NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
                    "PeriodicTorsionForce")

#: The Amber/SMIRNOFF electrostatic 1-4 scale, written exactly. SMIRNOFF files state it to ten
#: decimals (0.8333333333), Amber's OpenMM files as the nearest double to 5/6. OpenMM merges
#: NonbondedForce tags whose scales agree within 1e-5 and silently keeps the FIRST file's value,
#: so a ligand loaded after a protein force field would be built with a different 1-4 scale than
#: the same ligand built alone. A package therefore stores the exact value, and records what its
#: source said.
EXACT_COULOMB14 = 5.0 / 6.0
_SCALE_SNAP = 1e-8


class UnsupportedParametersError(ValueError):
    """An ffxml or System carries a term this package format cannot represent faithfully."""


def canonical_json(document: Any) -> str:
    """One byte sequence per value: sorted keys, no whitespace, floats in shortest round-trip form."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def topology_for_molecule(mol, atom_names: Sequence[str], residue_name: str):
    """An OpenMM Topology holding exactly one residue: the package molecule, in package order."""
    from openmm.app import Element, Topology

    if len(atom_names) != mol.GetNumAtoms():
        raise ValueError(f"{len(atom_names)} atom names for a {mol.GetNumAtoms()}-atom molecule")
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue(residue_name, chain, "1")
    atoms = [topology.addAtom(name, Element.getByAtomicNumber(a.GetAtomicNum()), residue)
             for name, a in zip(atom_names, mol.GetAtoms())]
    for bond in mol.GetBonds():
        topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    return topology


def ligand_system(ffxml_text: str, mol, atom_names: Sequence[str], template_name: str):
    """The molecule alone, parameterised from the ffxml ONLY: no cutoff, no constraints.

    No constraints, because a constrained bond is removed from HarmonicBondForce and its force
    constant would vanish from the table. No centre-of-mass remover, because it is not a
    parameter.
    """
    from openmm import app

    topology = topology_for_molecule(mol, atom_names, template_name[:3])
    forcefield = app.ForceField(io.StringIO(ffxml_text))
    residue = next(iter(topology.residues()))
    return forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None,
                                   rigidWater=False, removeCMMotion=False,
                                   residueTemplates={residue: template_name})


def classify_torsion(atoms: Sequence[int], bonded: set[tuple[int, int]]) -> str:
    """`proper` for a bonded chain i-j-k-l; `improper` otherwise.

    Decided from the molecule's bond graph, as the REST2 scaler decides it, rather than from the
    tag the ffxml happened to use.
    """
    i, j, k, l = atoms

    def b(x, y):
        return (min(x, y), max(x, y)) in bonded

    return "proper" if b(i, j) and b(j, k) and b(k, l) else "improper"


def _q(value) -> float:
    from openmm import unit

    if hasattr(value, "value_in_unit_system"):
        value = value.value_in_unit_system(unit.md_unit_system)
    number = float(value)
    if not math.isfinite(number):
        raise UnsupportedParametersError(f"non-finite parameter value {number!r}")
    return number


def parameter_table(system, mol, *, coulomb14scale: float, lj14scale: float) -> dict[str, Any]:
    """Every parameter of *system*, in package atom order, as plain JSON-able values."""
    names = [type(system.getForce(i)).__name__ for i in range(system.getNumForces())]
    unsupported = sorted(set(names) - set(SUPPORTED_FORCES))
    if unsupported:
        raise UnsupportedParametersError(
            f"the parameterised molecule carries {', '.join(unsupported)}, which a ligand "
            f"parameter package cannot represent. Refusing rather than dropping the term: a "
            f"package missing a force loads and runs as a different molecule.")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise UnsupportedParametersError(f"more than one {', '.join(duplicated)} in the System")
    if any(system.isVirtualSite(i) for i in range(system.getNumParticles())):
        raise UnsupportedParametersError("the molecule has virtual sites; not supported by the "
                                         "ligand package format")
    n = mol.GetNumAtoms()
    if system.getNumParticles() != n:
        raise UnsupportedParametersError(
            f"{system.getNumParticles()} particles for a {n}-atom molecule")
    forces = {type(system.getForce(i)).__name__: system.getForce(i)
              for i in range(system.getNumForces())}
    bonded = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(),
               b.GetEndAtomIdx())) for b in mol.GetBonds()}

    nonbonded = forces.get("NonbondedForce")
    atoms = []
    for index, atom in enumerate(mol.GetAtoms()):
        charge = sigma = epsilon = 0.0
        if nonbonded is not None:
            charge, sigma, epsilon = (_q(x) for x in nonbonded.getParticleParameters(index))
        atoms.append([atom.GetSymbol(), _q(system.getParticleMass(index)), charge, sigma, epsilon])

    exceptions = []
    if nonbonded is not None:
        if nonbonded.getNumExceptionParameterOffsets() or nonbonded.getNumParticleParameterOffsets():
            raise UnsupportedParametersError("parameter offsets are not supported in a package")
        for k in range(nonbonded.getNumExceptions()):
            i, j, qq, sig, eps = nonbonded.getExceptionParameters(k)
            i, j = min(i, j), max(i, j)
            exceptions.append([i, j, _q(qq), _q(sig), _q(eps)])
    bonds = []
    if "HarmonicBondForce" in forces:
        f = forces["HarmonicBondForce"]
        for k in range(f.getNumBonds()):
            i, j, length, kk = f.getBondParameters(k)
            bonds.append([min(i, j), max(i, j), _q(length), _q(kk)])
    angles = []
    if "HarmonicAngleForce" in forces:
        f = forces["HarmonicAngleForce"]
        for k in range(f.getNumAngles()):
            i, j, l, theta, kk = f.getAngleParameters(k)
            if i > l:
                i, l = l, i
            angles.append([i, j, l, _q(theta), _q(kk)])
    propers, impropers = [], []
    if "PeriodicTorsionForce" in forces:
        f = forces["PeriodicTorsionForce"]
        for k in range(f.getNumTorsions()):
            i, j, kk, l, periodicity, phase, force_k = f.getTorsionParameters(k)
            quartet = [i, j, kk, l]
            if classify_torsion(quartet, bonded) == "proper":
                if (l, kk, j, i) < (i, j, kk, l):
                    quartet = [l, kk, j, i]
                propers.append([*quartet, int(periodicity), _q(phase), _q(force_k)])
            else:
                # An improper's energy depends on atom order, so the order is kept as built.
                impropers.append([*quartet, int(periodicity), _q(phase), _q(force_k)])

    return {
        "schema": PARAMETER_TABLE_SCHEMA,
        "units": "OpenMM md_unit_system: nm, kJ/mol, rad, amu, elementary charge",
        "conventions": {
            "coulomb14scale": float(coulomb14scale),
            "lj14scale": float(lj14scale),
            "combining_rule": "lorentz-berthelot",
            "exclusions": "1-2 and 1-3 pairs excluded; 1-4 pairs as the listed exceptions",
        },
        "atoms": atoms,
        "bonds": sorted(bonds),
        "angles": sorted(angles),
        "proper_torsions": sorted(propers),
        "improper_torsions": sorted(impropers),
        "exceptions": sorted(exceptions),
        "net_charge_e": sum(a[2] for a in atoms),
    }


def parameter_digest(table: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(table).encode()).hexdigest()


def nonbonded_scales(ffxml_text: str) -> tuple[float, float]:
    root = ET.fromstring(ffxml_text)
    tags = root.findall("NonbondedForce")
    if len(tags) != 1:
        raise UnsupportedParametersError(f"expected one <NonbondedForce>, found {len(tags)}")
    return float(tags[0].get("coulomb14scale")), float(tags[0].get("lj14scale"))


def rename_ffxml(ffxml_text: str, *, template_name: str, type_prefix: str,
                 atom_names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """Give the ffxml collision-free names and exact 1-4 scales. Parameters are untouched.

    openmmforcefields names atom types by the sha256 of the molecule's SMILES, and the residue
    template by the SMILES itself. Two packages for the SAME molecule -- two charge methods, two
    force-field versions -- then define the same type names with different parameters, and OpenMM
    refuses or, worse, whichever loads first wins. Every name here is derived from the parameter
    identity instead.

    Returns the rewritten text and a note of what was normalised.
    """
    root = ET.fromstring(ffxml_text)
    residues = root.findall("Residues/Residue")
    if len(residues) != 1:
        raise UnsupportedParametersError(
            f"a ligand ffxml must define exactly one residue template, found {len(residues)}")
    residue = residues[0]
    template_atoms = residue.findall("Atom")
    if len(template_atoms) != len(atom_names):
        raise UnsupportedParametersError(
            f"the template has {len(template_atoms)} atoms, the molecule {len(atom_names)}")
    if residue.findall("VirtualSite") or residue.findall("ExternalBond"):
        raise UnsupportedParametersError(
            "the template has virtual sites or external bonds; a single-residue ligand package "
            "supports neither (a covalently attached ligand needs an explicit representation)")

    old_types = [a.get("type") for a in template_atoms]
    if len(set(old_types)) != len(old_types):
        raise UnsupportedParametersError("template atoms share a type; expected one type per atom")
    type_map = {old: f"{type_prefix}-{i}" for i, old in enumerate(old_types)}
    name_map = {a.get("name"): new for a, new in zip(template_atoms, atom_names)}

    for element in root.iter():
        for attribute, value in list(element.attrib.items()):
            if value in type_map and (attribute in ("name", "type", "class")
                                      or attribute.startswith(("class", "type"))):
                element.set(attribute, type_map[value])
    for atom in template_atoms:
        atom.set("name", name_map[atom.get("name")])
    for bond in residue.findall("Bond"):
        for key in ("atomName1", "atomName2"):
            bond.set(key, name_map[bond.get(key)])
    # A constraint list inside a template would constrain bonds in every environment regardless
    # of the build's `constraints` setting; the build decides that, not the package.
    for constraint in residue.findall("Constraint"):
        residue.remove(constraint)
    residue.set("name", template_name)

    note: dict[str, Any] = {}
    nonbonded = root.find("NonbondedForce")
    if nonbonded is not None:
        stated = float(nonbonded.get("coulomb14scale"))
        note["source_coulomb14scale"] = nonbonded.get("coulomb14scale")
        if abs(stated - EXACT_COULOMB14) < 1e-5 and stated != EXACT_COULOMB14:
            if abs(stated - EXACT_COULOMB14) > 1e-9 and abs(stated - 0.8333333333) > _SCALE_SNAP:
                raise UnsupportedParametersError(
                    f"coulomb14scale {stated!r} is close to, but not a spelling of, 5/6")
            nonbonded.set("coulomb14scale", repr(EXACT_COULOMB14))
            note["coulomb14scale_normalised_to"] = repr(EXACT_COULOMB14)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n", note


def copy_table(table: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(table)
