"""What makes a saved parameter set reusable for a build: ONE definition, consulted twice.

A build that needs ligand parameters searches the catalog, and then either reuses what it finds or
parameterises the molecule itself. Those two branches must agree about what "the same ligand" is,
so the comparison lives here and both call it. Two places deciding it would drift, and the drift
would be invisible: the result of reusing parameters generated for a different protonation state,
or by a different implementation of the same charge method, is a System that builds, runs and
answers a different question.

Three things are compared, all of them, and a near match is a difference:

  topology     the heavy-atom skeleton: which atoms, bonded how. Tautomers and protomers of one
               substance share it, which is why it is not enough on its own.
  protonation  the exact chemical state: every hydrogen, formal charges, bond orders and
               stereochemistry (`identity.chemical_state`).
  charges      the method, the scheme it resolved to, and the IMPLEMENTATION that produced them --
               AM1-BCC through AmberTools' sqm is not AM1-BCC through OpenEye and neither is
               NAGL's graph model of it, which also names the trained model file.

The force field is compared too: Sage 2.2.1 and Sage 2.1.0 are different Hamiltonians for the same
molecule in the same state, and a package records which one it holds.

A verdict carries the reasons, so a build can record WHY it reused a package, or why it did not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .package import PackageError

__all__ = ["CRITERIA_SCHEMA", "TOPOLOGY_SCHEME", "BACKEND_IDS", "MatchVerdict", "criteria_of_metadata",
           "matches", "requested_criteria", "topology_identity"]

#: The shape of a criteria document. BUMPED whenever the derived document gains or loses a field,
#: because a stored `parameter.config` is compared with a freshly derived one: without a version
#: that moves, a file written by older code and a file that genuinely disagrees with its package
#: are indistinguishable, and adding one key made every existing file read as a contradiction.
CRITERIA_SCHEMA = "md-tools-ligand-criteria/2"

#: A stored document that states a different schema is RE-DERIVED rather than compared: it was
#: written in another vocabulary, which is not the same as disagreeing with its package. Strict
#: equality applies within one schema, and that is what catches an edited catalog.

#: The namespace the heavy-atom skeleton digest is computed in. SEPARATE from CRITERIA_SCHEMA, and
#: it does not move with it: the digest identifies a molecule's skeleton, and if it were derived
#: from the document's version then adding a field elsewhere would change what every stored
#: digest is compared against. It changes only if the skeleton canonicalisation itself changes.
TOPOLOGY_SCHEME = "md-tools-ligand-topology/1"

#: The charge implementations this package can name. A record that cannot say which one ran is
#: refused rather than written: "am1bcc" alone does not identify the numbers.
BACKEND_IDS = ("ambertools-sqm", "openeye", "openff-nagl")

#: The three fields of a verdict, in the order they are reported.
_COMPARED = ("topology", "protonation", "charges", "forcefield")


@dataclass
class MatchVerdict:
    """Whether a package may be reused, and what was compared to decide it."""

    matched: bool
    reasons: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def differences(self) -> list[str]:
        return [f"{name}: {why}" for name, why in self.reasons.items() if why != "same"]

    def as_dict(self) -> dict[str, Any]:
        return {"matched": self.matched, "compared": dict(self.reasons), "notes": list(self.notes)}


def topology_identity(mol) -> dict[str, Any]:
    """The heavy-atom skeleton, independent of hydrogens, charges and bond orders.

    Deliberately coarse: it is the first of the three comparisons, the one that says "this is the
    same substance", and the protonation comparison below carries everything it drops.
    """
    from rdkit import Chem

    skeleton = Chem.RWMol(Chem.RemoveHs(Chem.Mol(mol)))
    for atom in skeleton.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetIsAromatic(False)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    smiles = Chem.MolToSmiles(skeleton.GetMol(), canonical=True)
    key = Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(mol))) or ""
    return {"skeleton_smiles": smiles,
            "skeleton_digest": hashlib.sha256(f"{TOPOLOGY_SCHEME}|{smiles}".encode()).hexdigest(),
            "standard_inchikey_skeleton": key.split("-")[0] if key else None,
            "n_heavy_atoms": skeleton.GetMol().GetNumAtoms()}


def charge_identity(method: str, *, scheme: Optional[str], backend_id: Optional[str],
                    model: Optional[dict[str, Any]] = None,
                    allow_unknown: bool = False) -> dict[str, Any]:
    """What produced the charges, named exactly enough to compare.

    `allow_unknown` is for READING a package written before the implementation was recorded. Such
    a package loads and can be used when a configuration names it explicitly -- its numbers are
    whatever they are, and nothing about them changed -- but `backend_recorded: false` travels
    with it and `matches` never lets it satisfy a search. Requesting charges is never unknown: a
    build always knows which implementation it would run.
    """
    if backend_id not in BACKEND_IDS:
        if backend_id is None and allow_unknown:
            return {"method": str(method), "scheme": scheme, "backend_id": None,
                    "backend_recorded": False,
                    "model": ({"name": model.get("name"), "sha256": model.get("sha256")}
                              if model else None)}
        # PackageError, not a bare ValueError: build-top's ligand validation turns PackageError
        # into a refusal naming the configuration key, and a bare one escaped that handler and
        # reached the user as a traceback. PackageError IS a ValueError, so nothing else changes.
        raise PackageError(
            f"charge backend {backend_id!r} is not one of {BACKEND_IDS}. A charge method alone "
            f"does not identify the numbers: AM1-BCC through AmberTools' sqm, through OpenEye and "
            f"through NAGL's graph model are three different results, and a package that cannot "
            f"say which one produced it cannot be reused safely.")
    return {"method": str(method), "scheme": scheme, "backend_id": backend_id,
            "backend_recorded": True,
            "model": ({"name": model.get("name"), "sha256": model.get("sha256")}
                      if model else None)}


def requested_criteria(mol, *, charge_method: str, forcefield: str) -> dict[str, Any]:
    """What a build is ABOUT to produce, resolved without computing anything expensive.

    The charge implementation is resolved the way `create_package` would resolve it in this
    environment -- OpenEye if its toolkit is registered, NAGL's model file when NAGL is asked for
    -- so a search compares what would actually run, not the label in a configuration.
    """
    from .identity import chemical_state
    from .package import _prepared_molecule

    prepared = _prepared_molecule(mol)
    method = str(charge_method).lower()
    if method == "am1bcc":
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

        openeye = any("OpenEye" in t.__class__.__name__
                      for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits)
        charges = charge_identity("am1bcc", scheme="am1bccelf10" if openeye else "am1bcc",
                                  backend_id="openeye" if openeye else "ambertools-sqm")
    elif method in ("am1bcc_nagl", "nagl"):
        from ..openmm.system import resolve_nagl_am1bcc_model

        model = resolve_nagl_am1bcc_model()
        charges = charge_identity(method, scheme=model["name"], backend_id="openff-nagl",
                                  model=model)
    else:
        raise PackageError(f"charge method {charge_method!r} cannot be searched for; supported: "
                         f"am1bcc, am1bcc_nagl")
    return {"schema_version": CRITERIA_SCHEMA,
            "topology": topology_identity(prepared),
            "protonation": {k: chemical_state(prepared)[k] for k in
                            ("digest", "fixed_h_inchikey", "net_formal_charge", "canonical_smiles")},
            "charges": charges,
            "forcefield": {"family": "smirnoff", "resource": str(forcefield)}}


def criteria_of_metadata(metadata: dict[str, Any], mol) -> dict[str, Any]:
    """The same shape, derived from a package's metadata and molecule.

    Derived rather than stored twice: `parameter.config` is written from this, and re-derived and
    compared when a package is loaded, so the browsable file cannot drift from the parameters.
    """
    charges = metadata["charges"]
    state = metadata["chemical_state"]
    return {"schema_version": CRITERIA_SCHEMA,
            "topology": topology_identity(mol),
            "protonation": {k: state[k] for k in
                            ("digest", "fixed_h_inchikey", "net_formal_charge", "canonical_smiles")},
            # allow_unknown: a package written before the implementation was recorded still
            # loads. `matches` refuses to reuse it; see `charge_identity`.
            "charges": charge_identity(charges["method"], scheme=charges.get("scheme"),
                                       backend_id=charges.get("backend_id"),
                                       model=charges.get("model"), allow_unknown=True),
            "forcefield": {"family": metadata["forcefield"]["family"],
                           "resource": metadata["forcefield"]["resource"]}}


def matches(request: dict[str, Any], candidate: dict[str, Any]) -> MatchVerdict:
    """THE comparison. Used by the catalog search and by the decision to parameterise instead."""
    reasons: dict[str, str] = {}
    notes: list[str] = []
    if request.get("schema_version") != candidate.get("schema_version"):
        return MatchVerdict(False, {"schema": f"{candidate.get('schema_version')!r} is not "
                                              f"{request.get('schema_version')!r}"})
    for name in _COMPARED:
        want, have = request[name], candidate[name]
        if name == "topology":
            same = want["skeleton_digest"] == have["skeleton_digest"]
            why = "same" if same else (f"a different heavy-atom skeleton "
                                       f"({have['skeleton_smiles']} is not {want['skeleton_smiles']})")
        elif name == "protonation":
            same = want["digest"] == have["digest"]
            why = "same" if same else (f"a different chemical state ({have['canonical_smiles']} is "
                                       f"not {want['canonical_smiles']})")
        elif name == "charges":
            if not have.get("backend_recorded", True) or have.get("backend_id") is None:
                # NEVER a match. The package's numbers may be anything; what is missing is which
                # implementation produced them, and "am1bcc" alone does not identify a result. A
                # configuration may still name this package explicitly -- that is the user saying
                # they know what it is -- but a search must not present it as reuse of known
                # charges.
                reasons[name] = ("the package does not record which implementation produced its "
                                 "charges, so it can be used only by naming it explicitly")
                return MatchVerdict(False, reasons, notes)
            same = (want["method"] == have["method"] and want["scheme"] == have["scheme"]
                    and want["backend_id"] == have["backend_id"] and want["model"] == have["model"])
            why = "same" if same else (
                f"different charges ({have['method']}/{have['scheme']} by {have['backend_id']}"
                + (f" model {(have['model'] or {}).get('name')}" if have["model"] else "")
                + f" is not {want['method']}/{want['scheme']} by {want['backend_id']}"
                + (f" model {(want['model'] or {}).get('name')}" if want["model"] else "") + ")")
        else:
            same = want["resource"] == have["resource"]
            why = "same" if same else (f"a different force field ({have['resource']} is not "
                                       f"{want['resource']})")
        reasons[name] = why
        if not same:
            return MatchVerdict(False, reasons, notes)
    return MatchVerdict(True, reasons, notes)

