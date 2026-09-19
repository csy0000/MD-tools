"""Selective REST2: which atoms are hot and which torsions are hot, resolved once, from masks.

THE one resolver (0.6.1). `md-openmm build-top --rest2-scaler` calls it, and so does anything
that validates a selection afterwards: one parser (`md_tools.rest2.masks`), one resolver (this
module), one record (`md_tools.rest2.selection.ScalingSelection`, format 2.0).

    backbone_scaling_list:  ":45,46,59"
    sidechain_scaling_list: ":45-50"
    ligand_scaling_dict:
      L01: {mask: ":201", torsion_exclusions: L01-exclusions.yaml}
      L02: {mask: ":305", torsion_exclusions: auto}

TWO MODES, and they never blend. No selector at all is the LEGACY full-solute selection -- every
non-solvent, non-ion atom, byte-identical to 0.6.0. ANY selector is EXPLICIT: only the categories
and instances named are hot, an omitted category means none, and an explicit region that resolves
to nothing is refused.

TWO QUESTIONS, answered separately (policy `md-tools-selective-rest2/1`):

* **Which atoms carry the nonbonded factors.** Backbone atoms of the backbone-selected residues,
  sidechain atoms of the sidechain-selected residues, every atom of a selected ligand instance.
  S-S pairs and exceptions carry `(1-tau)^2`, S-E `(1-tau)`, E-E `1`.
* **Which torsions are scaled.** Every eligible PROPER Fourier term across a selected CENTRAL BOND,
  where a central bond is selected when the region owning it is. Ownership (`central_bond_owner`):

      both atoms backbone, one residue     -> that residue's backbone    (phi N-CA, psi CA-C)
      a sidechain atom involved, one residue -> that residue's sidechain (chi1 CA-CB: its quartet
                                               N-CA-CB-CG reaches backbone N and CA, and is
                                               still the sidechain's)
      peptide bond C(i)-N(i+1)             -> residue i's backbone (omega_i, IUPAC). Ordinary amide
                                               omegas stay unscaled whoever owns them; this
                                               decides the proline-like ones
      disulfide SG-SG                      -> JOINTLY both sidechains: scaled only when both are
                                               selected, and reported when only one is
      within a ligand instance             -> that instance
      anything else between residues       -> undefined, and refused when either side is selected

  The four atoms of a scaled torsion need not all be hot: a torsion exclusion and a nonbonded
  selection are different questions about the same atom, and neither removes an atom from the
  other.

MEMBERSHIP (`atom_category`). Backbone is N, H, H1, H2, H3 (the N-terminal amine), CA, HA,
HA2, HA3 (glycine), C, O and OXT (the C-terminal carboxylate); everything else in a supported
protein residue is sidechain. So glycine has no sidechain, proline's ring (CB, CG, CD and their
hydrogens) is sidechain while its N and CA are backbone, and a CYX's SG is sidechain. A CAP (ACE,
NME, NHE, NMA) is backbone throughout. A hydrogen follows the heavy atom it is named for. Nucleic
acids and modified residues have no definition here and are refused by name; a ligand is selected
only as an instance.

CMAP (`cmap_decision`). A CMAP term couples two backbone torsions, phi and psi. It is scaled when
BOTH their central bonds are selected, left unchanged and reported when only one is ("mixed"), and
left unchanged when neither is. A map shared between scaled and unscaled terms is duplicated
(`hamiltonian.duplicate_shared_cmaps`), so scaling one residue's map never scales another's.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .masks import MASK_GRAMMAR, MASK_NUMBERING, MaskError, parse_residue_mask
from .selection import SelectionError

__all__ = [
    "SELECTION_POLICY", "SELECTOR_KEYS", "EXCLUSIONS_FORMAT", "BACKBONE_ATOM_NAMES",
    "CAP_RESIDUES", "atom_category", "residue_kind", "central_bond_owner", "has_selectors",
    "parse_selectors", "resolve_region", "ResolvedRegion", "load_torsion_exclusions",
]

#: Membership, ownership and CMAP rules together, by name and version. Recorded in every selection.
SELECTION_POLICY = "md-tools-selective-rest2/1"

#: The configuration keys that switch a selection to explicit mode.
SELECTOR_KEYS = ("backbone_scaling_list", "sidechain_scaling_list", "ligand_scaling_dict")

#: A per-instance torsion-exclusion file, in PACKAGE-LOCAL atom names.
EXCLUSIONS_FORMAT = "md-tools-torsion-exclusions/1"

BACKBONE_ATOM_NAMES = frozenset({"N", "H", "H1", "H2", "H3", "CA", "HA", "HA2", "HA3", "C", "O",
                                 "OXT"})
CAP_RESIDUES = frozenset({"ACE", "NME", "NHE", "NMA"})

_NUCLEIC = frozenset({"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "RA", "RC", "RG",
                      "RU", "A3", "A5", "C3", "C5", "G3", "G5", "U3", "U5", "DA3", "DA5", "DC3",
                      "DC5", "DG3", "DG5", "DT3", "DT5"})


def _protein_residues() -> frozenset:
    from ..openmm.system import PROTEIN_RESIDUES

    return PROTEIN_RESIDUES


def _solvent_residues() -> frozenset:
    from ..md.stage import SOLVENT_RESIDUES

    return SOLVENT_RESIDUES


def residue_kind(residue) -> str:
    """`protein`, `cap`, `solvent`, `nucleic` or `other` (a ligand, or a residue nobody defined)."""
    name = residue.name.strip().upper()
    if name in _solvent_residues():
        return "solvent"
    if name in CAP_RESIDUES:
        return "cap"
    if name in _protein_residues():
        return "protein"
    if name in _NUCLEIC:
        return "nucleic"
    return "other"


def atom_category(atom) -> str | None:
    """`backbone` or `sidechain` for an atom of a protein residue or cap; None otherwise."""
    kind = residue_kind(atom.residue)
    if kind == "cap":
        return "backbone"
    if kind == "protein":
        return "backbone" if atom.name.strip().upper() in BACKBONE_ATOM_NAMES else "sidechain"
    return None


def _owner_of_one_residue(atom_a, atom_b):
    residue = atom_a.residue
    kind = residue_kind(residue)
    if kind in ("protein", "cap"):
        categories = {atom_category(atom_a), atom_category(atom_b)}
        return ((residue.index, "sidechain" if "sidechain" in categories else "backbone"),)
    if kind == "other":
        return ((residue.index, "ligand"),)
    return None


def central_bond_owner(atom_a, atom_b):
    """Who owns central bond a-b: a tuple of `(residue index, category)`, all of which must be
    selected for its torsions to be scaled; or None when ownership is undefined. See the module
    docstring for the table."""
    if atom_a.residue.index == atom_b.residue.index:
        return _owner_of_one_residue(atom_a, atom_b)
    kinds = {residue_kind(atom_a.residue), residue_kind(atom_b.residue)}
    if kinds <= {"protein", "cap"}:
        names = {atom_a.name.strip().upper(), atom_b.name.strip().upper()}
        if names == {"C", "N"}:
            carbon = atom_a if atom_a.name.strip().upper() == "C" else atom_b
            return ((carbon.residue.index, "backbone"),)
        if names == {"SG"}:
            return tuple(sorted(((atom_a.residue.index, "sidechain"),
                                 (atom_b.residue.index, "sidechain"))))
    return None


# --- the selectors -------------------------------------------------------------------------------

def has_selectors(config: Mapping[str, Any] | None) -> bool:
    """Whether any selective-REST2 key is PRESENT. Presence, not truth: an empty list is explicit."""
    return bool(config) and any(key in config and config[key] is not None
                                for key in SELECTOR_KEYS)


def parse_selectors(config: Mapping[str, Any]) -> dict[str, Any]:
    """The three selector keys, parsed, with the original strings kept exactly as written."""
    parsed: dict[str, Any] = {"backbone": None, "sidechain": None, "ligands": {}}
    for key, slot in (("backbone_scaling_list", "backbone"),
                      ("sidechain_scaling_list", "sidechain")):
        if config.get(key) is not None:
            text = config[key]
            parsed[slot] = {"mask": text, "residues": parse_residue_mask(text, where=key)}
    ligands = config.get("ligand_scaling_dict")
    if ligands is not None:
        if not isinstance(ligands, Mapping):
            raise SelectionError(f"ligand_scaling_dict maps an instance label to "
                                 f"{{mask, torsion_exclusions}}; got {type(ligands).__name__}")
        for label, entry in ligands.items():
            where = f"ligand_scaling_dict.{label}"
            if not isinstance(label, str) or not label:
                raise SelectionError(f"ligand_scaling_dict: {label!r} is not an instance label")
            if not isinstance(entry, Mapping):
                raise SelectionError(
                    f"{where}: the compact form `{label}: <exclusion file>` needs {label!r} to be "
                    f"a recorded ligand-INSTANCE alias, and the ligand mapping record "
                    f"(md-tools-ligand-mapping/1) records none -- a compound alias or a residue "
                    f"name cannot say which copy is meant. Write the explicit entry:\n"
                    f"  {label}:\n    mask: \":<topology residue index>\"\n"
                    f"    torsion_exclusions: {entry if isinstance(entry, str) else 'auto'}")
            unknown = sorted(set(entry) - {"mask", "torsion_exclusions"})
            if unknown:
                raise SelectionError(f"{where}: unknown key(s) {unknown}; an instance entry is "
                                     f"{{mask, torsion_exclusions}}")
            if "mask" not in entry:
                raise SelectionError(f"{where}: `mask` is required -- the one-based topology "
                                     f"residue index of THIS instance, e.g. mask: \":201\"")
            exclusions = entry.get("torsion_exclusions", "auto")
            if exclusions is None or not isinstance(exclusions, str) or not exclusions:
                raise SelectionError(f"{where}: torsion_exclusions is `auto` or the path of a "
                                     f"{EXCLUSIONS_FORMAT} file; got {exclusions!r}")
            residues = parse_residue_mask(entry["mask"], where=f"{where}.mask")
            if len(residues) != 1:
                raise SelectionError(
                    f"{where}: mask {entry['mask']!r} names {len(residues)} residues. A ligand "
                    f"entry is ONE instance -- one residue -- so that selecting it never selects "
                    f"its twin; give each instance its own entry.")
            parsed["ligands"][label] = {"mask": entry["mask"], "residues": residues,
                                        "torsion_exclusions": exclusions}
    return parsed


# --- the exclusion file ----------------------------------------------------------------------------

def load_torsion_exclusions(path: Path, *, where: str) -> dict[str, Any]:
    """Read a `md-tools-torsion-exclusions/1` file. Returns its parsed form and EXACT contents.

    Contents and sha256 go into the selection record: a path is not provenance, because the file
    can change after the record is written, and then the record describes a run that never
    happened.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise SelectionError(f"{where}: torsion_exclusions names {path}, which does not exist")
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as broken:
        raise SelectionError(f"{where}: {path} is not readable YAML: {broken}") from None
    if not isinstance(document, dict) or document.get("format") != EXCLUSIONS_FORMAT:
        found = document.get("format") if isinstance(document, dict) else type(document).__name__
        raise SelectionError(f"{where}: {path} is not a {EXCLUSIONS_FORMAT} file (format: "
                             f"{found!r})")
    unknown = sorted(set(document) - {"format", "parameters", "residue_name", "central_bonds",
                                      "note"})
    if unknown:
        raise SelectionError(f"{where}: {path}: unknown key(s) {unknown}")
    parameters = document.get("parameters")
    if not isinstance(parameters, str) or "/param_" not in parameters:
        raise SelectionError(
            f"{where}: {path} must name the parameter package its atom names belong to, as "
            f"`parameters: <compound>/param_<id>` (got {parameters!r}). A residue name cannot "
            f"bind it: two packages can share one, and names from one mean nothing in the other.")
    bonds = document.get("central_bonds")
    if not isinstance(bonds, list) or not all(
            isinstance(b, list) and len(b) == 2 and all(isinstance(n, str) for n in b)
            for b in bonds):
        raise SelectionError(f"{where}: {path}: central_bonds is a list of [atom name, atom name] "
                             f"pairs in the package's own atom names")
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "contents": raw.decode("utf-8"), "parameters": parameters,
            "residue_name": document.get("residue_name"),
            "central_bonds": [list(b) for b in bonds]}


# --- resolution ------------------------------------------------------------------------------------

class ResolvedRegion(dict):
    """The resolved selection, as a plain mapping (it is written to YAML verbatim)."""


def _neighbours(topology) -> dict[int, set[int]]:
    neighbours: dict[int, set[int]] = {}
    for bond in topology.bonds():
        neighbours.setdefault(bond.atom1.index, set()).add(bond.atom2.index)
        neighbours.setdefault(bond.atom2.index, set()).add(bond.atom1.index)
    return neighbours


def _residue_entry(residue) -> dict[str, Any]:
    return {"chain": residue.chain.id, "chain_index": int(residue.chain.index),
            "residue_id": str(residue.id).strip(),
            "insertion_code": (residue.insertionCode or "").strip(),
            "residue_name": residue.name}


def _instance_from_mapping(ligand_mapping: Mapping[str, Any] | None, residue) -> dict | None:
    """The `ligand_mapping.json` instance whose resolved residue is *residue*, or None."""
    if not ligand_mapping:
        return None
    for instance in ligand_mapping.get("instances") or ():
        resolved = instance.get("resolved") or {}
        if resolved.get("residue_index") == residue.index:
            return instance
    return None


def resolve_region(topology, config: Mapping[str, Any], *, config_dir: Path | None = None,
                   ligand_mapping: Mapping[str, Any] | None = None,
                   implicit: bool = False) -> ResolvedRegion:
    """Resolve the explicit selectors in *config* against *topology*. Refuses, never guesses.

    Returns the nonbonded atom set, the CANDIDATE central bonds with their owners, the per-residue
    map and the ligand instances. Which candidates are protected (amide, aromatic, double bond,
    per-instance exclusion files) is decided afterwards by the torsion classifier over exactly the
    residues this names (`classification_atoms`), and CMAP needs the System: see
    `md_tools.build.scaler`, which joins the three into one `ScalingSelection`.
    """
    if not has_selectors(config):
        raise SelectionError("resolve_region is the EXPLICIT resolver; with no selector present "
                             "the selection is the legacy full solute")
    parsed = parse_selectors(config)
    if implicit:
        raise SelectionError(
            "a selective REST2 region (" + ", ".join(k for k in SELECTOR_KEYS if
                                                    config.get(k) is not None) +
            ") was given for an implicit-solvent System. Selective generalised-Born scaling is "
            "outside 0.6.1: a GB energy is not separable per atom. Omit every selector to scale "
            "the whole system, which is the supported implicit ladder.")
    residues = list(topology.residues())
    count = len(residues)

    def lookup(number: int, where: str):
        if not 1 <= number <= count:
            raise MaskError(f"{where}: residue {number} is out of range; this topology has "
                            f"{count} residues, numbered 1..{count} across every chain without "
                            f"a reset")
        return residues[number - 1]

    selected_parts: set[tuple[int, str]] = set()
    residue_map: dict[int, dict[str, Any]] = {}
    notes: list[str] = []

    for category, key in (("backbone", "backbone_scaling_list"),
                          ("sidechain", "sidechain_scaling_list")):
        block = parsed[category]
        if block is None:
            continue
        for number in block["residues"]:
            residue = lookup(number, key)
            kind = residue_kind(residue)
            if kind not in ("protein", "cap"):
                what = {"solvent": "solvent or an ion", "nucleic": "a nucleic acid",
                        "other": "not a supported protein residue"}[kind]
                hint = ("; select a ligand through ligand_scaling_dict" if kind == "other" else
                        "; nucleic acids have no backbone/sidechain definition in "
                        + SELECTION_POLICY if kind == "nucleic" else "")
                raise SelectionError(
                    f"{key} {block['mask']!r}: residue {number} is {residue.name} "
                    f"(chain {residue.chain.id!r}, id {str(residue.id).strip()}), which is "
                    f"{what}{hint}. Backbone and sidechain are defined only for "
                    f"{sorted(_protein_residues())}.")
            entry = residue_map.setdefault(number, dict(_residue_entry(residue), categories=[],
                                                        ligand_instance=None))
            entry["categories"].append(category)
            members = [a for a in residue.atoms() if atom_category(a) == category]
            if not members:
                entry.setdefault("notes", []).append(f"no {category} atoms")
                notes.append(f"residue {number} {residue.name}: selected as {category}, has no "
                             f"{category} atoms (a glycine or a cap has no sidechain)")
            selected_parts.add((residue.index, category))

    instances: list[dict[str, Any]] = []
    for label, block in parsed["ligands"].items():
        where = f"ligand_scaling_dict.{label}"
        number = block["residues"][0]
        residue = lookup(number, f"{where}.mask")
        if residue_kind(residue) != "other":
            raise SelectionError(
                f"{where}: residue {number} is {residue.name}, which is "
                f"{residue_kind(residue)}, not a ligand. A ligand entry names one ligand "
                f"residue; protein residues go in backbone_scaling_list or "
                f"sidechain_scaling_list.")
        if number in residue_map:
            other = residue_map[number]["ligand_instance"]
            raise SelectionError(f"{where}: residue {number} is already selected as ligand "
                                 f"instance {other!r}; one instance, one entry")
        entry = residue_map.setdefault(number, dict(_residue_entry(residue), categories=[],
                                                    ligand_instance=None))
        entry["categories"].append("ligand")
        entry["ligand_instance"] = label
        selected_parts.add((residue.index, "ligand"))
        mapped = _instance_from_mapping(ligand_mapping, residue)
        names = [a.name for a in residue.atoms()]
        instance = {
            "label": label, "mask": block["mask"], "topology_residue": number,
            "residue": _residue_entry(residue),
            "residue_key": [residue.chain.id, str(residue.id).strip(),
                            (residue.insertionCode or "").strip()],
            "package": (mapped or {}).get("package"),
            "atom_identity": ("package atom names (ligand_mapping.json)" if mapped
                              else "residue atom names (no ligand mapping record)"),
            "atoms": {name: atom.index for name, atom in zip(names, residue.atoms())},
            "torsion_exclusions": {"mode": "auto"},
        }
        if len(set(names)) != len(names):
            raise SelectionError(f"{where}: residue {number} {residue.name} repeats atom names, "
                                 f"so a package-local atom identity cannot name its atoms")
        if block["torsion_exclusions"] != "auto":
            path = Path(block["torsion_exclusions"])
            if not path.is_absolute() and config_dir is not None:
                path = Path(config_dir) / path
            loaded = load_torsion_exclusions(path, where=f"{where}.torsion_exclusions")
            package = instance["package"] or {}
            bound = (f"{package.get('compound_id')}/{package.get('parameter_id')}"
                     if package.get("parameter_id") else None)
            if bound is None:
                raise SelectionError(
                    f"{where}.torsion_exclusions: {path} is bound to package "
                    f"{loaded['parameters']}, and residue {number} {residue.name} has no recorded "
                    f"parameter package (no ligand_mapping.json instance resolves to it), so its "
                    f"atom names cannot be shown to be that package's. Use `auto`, or build the "
                    f"structure from a registered package with `md-openmm build-top`.")
            if loaded["parameters"] != bound:
                raise SelectionError(
                    f"{where}.torsion_exclusions: {path} is written for package "
                    f"{loaded['parameters']}, and instance {label} (residue {number}) is package "
                    f"{bound}. Package-local atom names from one package mean nothing in another.")
            if loaded["residue_name"] not in (None, residue.name):
                raise SelectionError(
                    f"{where}.torsion_exclusions: {path} is written for residue "
                    f"{loaded['residue_name']!r}, and residue {number} is {residue.name}")
            resolved_pairs = []
            for first, second in loaded["central_bonds"]:
                missing = [n for n in (first, second) if n not in instance["atoms"]]
                if missing:
                    raise SelectionError(
                        f"{where}.torsion_exclusions: {path} names atom(s) {missing}, which "
                        f"{residue.name} {number} does not have (its atoms: {names})")
                resolved_pairs.append(sorted((instance["atoms"][first],
                                              instance["atoms"][second])))
            instance["torsion_exclusions"] = {
                "mode": "file", "file": block["torsion_exclusions"], "sha256": loaded["sha256"],
                "parameters": loaded["parameters"],
                "contents": loaded["contents"], "central_bonds": loaded["central_bonds"],
                "resolved_central_bonds": resolved_pairs}
        instances.append(instance)

    # Every nonbonded atom: backbone / sidechain of the selected residues, every ligand atom.
    nonbonded: set[int] = set()
    for residue in residues:
        for category in ("backbone", "sidechain"):
            if (residue.index, category) in selected_parts:
                nonbonded |= {a.index for a in residue.atoms() if atom_category(a) == category}
        if (residue.index, "ligand") in selected_parts:
            nonbonded |= {a.index for a in residue.atoms()}

    # Every candidate central bond (a torsion can run across it) owned by the selection.
    neighbours = _neighbours(topology)
    atoms = list(topology.atoms())
    candidates: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if len(neighbours.get(a.index, ())) < 2 or len(neighbours.get(b.index, ())) < 2:
            continue
        if residue_kind(a.residue) == "solvent" or residue_kind(b.residue) == "solvent":
            continue
        owner = central_bond_owner(a, b)
        touched = {(a.residue.index, c) for c in ("backbone", "sidechain", "ligand")} | \
                  {(b.residue.index, c) for c in ("backbone", "sidechain", "ligand")}
        if owner is None:
            if touched & selected_parts:
                raise SelectionError(
                    f"central bond {a.index}-{b.index} joins {a.residue.name}"
                    f"{a.residue.index + 1} {a.name} to {b.residue.name}{b.residue.index + 1} "
                    f"{b.name}: neither a peptide bond nor a disulfide, so which region owns its "
                    f"torsions is undefined under {SELECTION_POLICY}. Refusing rather than "
                    f"guessing.")
            continue
        chosen = [part in selected_parts for part in owner]
        entry = {"bond": sorted((a.index, b.index)),
                 "owner": [{"residue": index + 1, "category": category}
                           for index, category in owner]}
        if all(chosen):
            candidates.append(entry)
        elif any(chosen):
            partial.append(dict(entry, reason="jointly owned (disulfide) and only one side is "
                                              "selected: left unscaled"))

    if not nonbonded and not candidates:
        raise SelectionError(
            "the explicit selection resolves to NOTHING: no atom carries a nonbonded factor and "
            "no torsion is scaled. An explicit region that heats nothing is not the legacy "
            "selection -- omit every selector for that -- and is refused rather than run as an "
            "unscaled ladder. " + ("; ".join(notes) if notes else ""))

    # The residues a torsion classifier must see: both sides of every candidate bond.
    classify: set[int] = set()
    for entry in candidates:
        for index in entry["bond"]:
            classify |= {a.index for a in atoms[index].residue.atoms()}

    return ResolvedRegion(
        policy=SELECTION_POLICY,
        masks={"grammar": MASK_GRAMMAR, "numbering": MASK_NUMBERING,
               "backbone": (parsed["backbone"] or {}).get("mask"),
               "sidechain": (parsed["sidechain"] or {}).get("mask"),
               "ligands": {label: block["mask"] for label, block in parsed["ligands"].items()}},
        residue_map={str(k): residue_map[k] for k in sorted(residue_map)},
        selected_nonbonded_atoms=sorted(nonbonded),
        candidate_central_bonds=sorted(candidates, key=lambda e: e["bond"]),
        partially_owned_central_bonds=sorted(partial, key=lambda e: e["bond"]),
        ligand_instances=instances,
        classification_atoms=sorted(classify),
        notes=notes,
    )


def cmap_decisions(system, selected_bonds: Iterable[Iterable[int]]) -> list[dict[str, Any]]:
    """Per CMAP term: scaled when BOTH its torsions' central bonds are selected (see the module)."""
    from openmm import CMAPTorsionForce

    selected = {frozenset(int(i) for i in bond) for bond in selected_bonds}
    decisions: list[dict[str, Any]] = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if not isinstance(force, CMAPTorsionForce):
            continue
        for term in range(force.getNumTorsions()):
            parameters = force.getTorsionParameters(term)
            map_index, a = parameters[0], [int(x) for x in parameters[1:]]
            phi, psi = frozenset((a[1], a[2])), frozenset((a[5], a[6]))
            flags = (phi in selected, psi in selected)
            if all(flags):
                decision, reason = True, "both torsions selected"
            elif any(flags):
                decision, reason = False, ("MIXED: only the " + ("phi" if flags[0] else "psi")
                                           + " central bond is selected; left unscaled")
            else:
                decision, reason = False, "neither torsion selected"
            decisions.append({"term": term, "map": int(map_index), "scaled": decision,
                              "phi_central_bond": sorted(phi), "psi_central_bond": sorted(psi),
                              "reason": reason})
        break
    return decisions


def canonical_digest(document: Any) -> str:
    """sha256 of canonical JSON: the digest a selection is identified by."""
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def explicit_selection(topology, system, region: Mapping[str, Any],
                       classified: Mapping[str, Any]):
    """Join a resolved region, the torsion classification of its residues, and CMAP into ONE
    `ScalingSelection` (format 2.0, `selection_mode: explicit`).

    *classified* is `openmm.system.unscaled_torsions` over `region["classification_atoms"]`.
    A candidate central bond the classifier protects (ordinary amide omega, aromatic ring, double
    bond), or that an instance's exclusion file names, stays unscaled; the rest are the selected
    torsion central bonds. Impropers are always unscaled in a selective region.
    """
    from .selection import EXPLICIT_MODE, ScalingSelection, topology_digest

    if not classified.get("unscaled_impropers", True):
        raise SelectionError(
            "a selective REST2 region keeps every improper unscaled (an improper has no central "
            "bond, so no region owns it); `unscaled_torsions: false` is supported only for the "
            "legacy full-solute selection")
    candidates = {tuple(entry["bond"]): entry for entry in region["candidate_central_bonds"]}
    protected = {tuple(sorted(int(i) for i in bond))
                 for bond in classified["unscaled_central_bonds"]}

    from_files: dict[tuple[int, int], str] = {}
    for instance in region["ligand_instances"]:
        block = instance["torsion_exclusions"]
        if block["mode"] != "file":
            continue
        for names, pair in zip(block["central_bonds"], block["resolved_central_bonds"]):
            pair = tuple(pair)
            if pair not in candidates:
                raise SelectionError(
                    f"ligand_scaling_dict.{instance['label']}.torsion_exclusions: "
                    f"{block['file']} names {names[0]}-{names[1]}, which is not a central bond "
                    f"a torsion runs across in {instance['residue']['residue_name']} "
                    f"{instance['topology_residue']}. An exclusion that protects nothing would "
                    f"say a rotation keeps its barrier when no such rotation exists.")
            from_files[pair] = instance["label"]

    # ...and it must be the central bond of a PROPER torsion the System actually has. A bond a
    # torsion could run across but that the force field gave no term protects nothing either.
    if from_files:
        from openmm import PeriodicTorsionForce

        from .hamiltonian import system_bond_graph, torsion_kind

        graph = system_bond_graph(system)
        with_terms = set()
        for force in system.getForces():
            if isinstance(force, PeriodicTorsionForce):
                for term in range(force.getNumTorsions()):
                    i, j, k, l = (int(x) for x in force.getTorsionParameters(term)[:4])
                    if torsion_kind((i, j, k, l), graph) == "proper":
                        with_terms.add(tuple(sorted((j, k))))
        for pair, label in from_files.items():
            if pair not in with_terms:
                raise SelectionError(
                    f"ligand_scaling_dict.{label}.torsion_exclusions names bond {list(pair)}, "
                    f"which is the central bond of no proper torsion in this System: the "
                    f"exclusion would protect nothing, and the record would say otherwise.")
    excluded = sorted((set(candidates) & protected) | set(from_files))
    selected = sorted(set(candidates) - set(excluded))
    reasons = {tuple(sorted(e["bond"])): e["class"] for e in classified.get("central_bonds", [])}
    labels = [{"bond": list(pair),
               "residues": ", ".join(f"{o['category']}:{o['residue']}"
                                     for o in candidates[pair]["owner"]),
               "reason": (f"ligand instance {from_files[pair]}: torsion_exclusions file"
                          if pair in from_files else reasons.get(pair, "protected"))}
              for pair in excluded]
    for pair in classified.get("proline_like_scaled_bonds") or []:
        pair = tuple(sorted(int(i) for i in pair))
        if pair in candidates:
            labels.append({"bond": list(pair), "residues": None,
                           "reason": "proline-like: SCALED, no amide hydrogen to protect"})

    decisions = cmap_decisions(system, selected)
    details = (
        ("policy", SELECTION_POLICY),
        ("masks", region["masks"]),
        ("residue_map", region["residue_map"]),
        ("torsion_bond_owners", [dict(candidates[pair], protected=pair in excluded)
                                 for pair in sorted(candidates)]),
        ("cmap_decisions", decisions),
        ("ligand_instances", region["ligand_instances"]),
        ("partially_owned_central_bonds", region["partially_owned_central_bonds"]),
        ("notes", region["notes"]),
    )
    return ScalingSelection(
        solute_atoms=tuple(region["selected_nonbonded_atoms"]),
        excluded_bonds=tuple(excluded), topology_sha256=topology_digest(topology),
        labels=tuple(labels), detection=str(classified["detection_method"]),
        mode=EXPLICIT_MODE, torsion_bonds=tuple(selected),
        cmap_terms=tuple(d["term"] for d in decisions if d["scaled"]),
        unscaled_impropers=True, detector_version=classified.get("detector_version"),
        details=details)


def print_residue_map(selection, echo=print) -> None:
    """The resolved residue map, for a person: which residue each mask number actually matched."""
    document = selection.to_document() if hasattr(selection, "to_document") else selection
    residue_map = document.get("residue_map") or {}
    echo(f"selective REST2 ({document.get('policy')}): masks are one-based topology residue "
         f"indices")
    for number, entry in residue_map.items():
        where = f"chain {entry['chain']!r} id {entry['residue_id']}{entry['insertion_code']}"
        role = ", ".join(entry["categories"])
        instance = f"  instance {entry['ligand_instance']}" if entry["ligand_instance"] else ""
        extra = f"  ({'; '.join(entry['notes'])})" if entry.get("notes") else ""
        echo(f"  :{number:<5} {entry['residue_name']:<4} {where:<22} {role}{instance}{extra}")


def exclusion_file_changes(document: Mapping[str, Any], *, config_dir: Path) -> list[str]:
    """Which recorded torsion-exclusion files no longer hold the contents the record saved.

    The record, not the file, is authoritative for a Hamiltonian already built: its saved
    contents and resolved bonds reproduce the states. A changed file means a NEW resolution
    would describe a different Hamiltonian, so a consumer that re-resolves (a rebuild, or a
    workflow checking its configuration against the states it is about to use) must refuse
    rather than mix them. Returns one line per changed or missing file; empty when all agree.
    """
    changes = []
    for instance in document.get("ligand_instances") or ():
        block = instance.get("torsion_exclusions") or {}
        if block.get("mode") != "file":
            continue
        path = Path(block["file"])
        path = path if path.is_absolute() else Path(config_dir) / path
        if not path.is_file():
            changes.append(f"{instance['label']}: {path} no longer exists")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != block["sha256"]:
            changes.append(f"{instance['label']}: {path} has sha256 {digest[:12]}..., the "
                           f"record saved {block['sha256'][:12]}...")
    return changes


def claimed_region_differences(topology, claim: Mapping[str, Any], *, config_dir: Path,
                               ligand_mapping: Mapping[str, Any] | None, implicit: bool,
                               selection_document: Mapping[str, Any] | None) -> list[str]:
    """How a CLAIMED region (selector keys, e.g. from a build-md REST2 configuration) differs from
    the region RECORDED in a selection document (`scaler.yaml`'s `selection`). `[]` means agree.

    The ONE comparison, and it resolves the claim through `resolve_region`, the one resolver, so
    two spellings of one region (":45,46" and ":45-46") agree and a claim is never compared as
    text. What is compared is exactly the Hamiltonian-determining projection the identity hashes
    (`identity.hamiltonian_selection_projection`), as far as a region determines it: the hot
    atoms, each residue's role, and each ligand instance's residue, package and RESOLVED exclusion
    content. Labels, mask text, paths and file bytes are provenance and never differ here.
    Torsion classification settings are the scaler's and are not a claim.

    A claim that does not itself resolve RAISES (`SelectionError`): an invalid claim is refused,
    not reported as a difference. A legacy or 1.0 record, or a missing one, is a difference with
    a sentence, never a crash.
    """
    from .selection import (EXPLICIT_MODE, LEGACY_MODE, LEGACY_SELECTION_FORMAT,
                            SELECTION_FORMAT, topology_digest)

    claimed_explicit = has_selectors(claim)
    document = dict(selection_document or {})
    fmt = document.get("format")
    mode = document.get("selection_mode", LEGACY_MODE) if fmt == SELECTION_FORMAT else LEGACY_MODE
    region = None
    if claimed_explicit:
        # Resolved FIRST, so an invalid claim is refused whatever the record says.
        region = resolve_region(topology, claim, config_dir=config_dir,
                                ligand_mapping=ligand_mapping, implicit=implicit)
    if not document:
        return ["there is no recorded selection to compare the claim with; rebuild the states "
                "with `md-openmm build-top --rest2-scaler`"]
    if fmt not in (SELECTION_FORMAT, LEGACY_SELECTION_FORMAT):
        return [f"the recorded selection has format {fmt!r}, which this build cannot compare"]
    if not claimed_explicit:
        return ([] if mode == LEGACY_MODE else
                ["the claim names no selector (the whole solute), but the states were built for "
                 "an EXPLICIT region: " + _describe_masks(document.get("masks"))])
    if mode != EXPLICIT_MODE:
        what = "a 1.0 (pre-0.6.1) record" if fmt == LEGACY_SELECTION_FORMAT else "a legacy record"
        return [f"explicit claim against {what}: the states were built for the whole solute, and "
                f"the claim names a selective region ({_describe_masks(region['masks'])})"]

    differences: list[str] = []
    recorded_digest = document.get("topology_sha256")
    if recorded_digest and recorded_digest != topology_digest(topology):
        differences.append("the record was resolved against a different topology "
                           f"({str(recorded_digest)[:12]}... vs "
                           f"{topology_digest(topology)[:12]}...)")

    # THE SAME PROJECTION the Hamiltonian identity hashes (`identity.hamiltonian_selection_
    # projection`), applied to what the region resolution determines -- hot atoms and ligand
    # instances. Labels, mask text, paths and exclusion-file bytes are provenance, here as there.
    from .identity import hamiltonian_selection_projection

    claimed = hamiltonian_selection_projection({
        "format": SELECTION_FORMAT, "selection_mode": EXPLICIT_MODE,
        "selected_nonbonded_atoms": region["selected_nonbonded_atoms"],
        "ligand_instances": region["ligand_instances"]})
    recorded = hamiltonian_selection_projection(document)

    claimed_atoms = set(claimed["selected_nonbonded_atoms"])
    recorded_atoms = set(recorded["selected_nonbonded_atoms"])
    if claimed_atoms != recorded_atoms:
        differences.append(
            f"hot nonbonded atoms differ: {len(claimed_atoms - recorded_atoms)} claimed but not "
            f"recorded, {len(recorded_atoms - claimed_atoms)} recorded but not claimed")

    # Which ROLE each residue plays decides which torsions it owns; its display name and any
    # instance label do not.
    def roles(residue_map):
        return {str(k): (sorted(v.get("categories") or ()), v.get("residue_name"))
                for k, v in (residue_map or {}).items()}

    claimed_roles, recorded_roles = roles(region["residue_map"]), roles(document.get("residue_map"))
    for number in sorted(set(claimed_roles) | set(recorded_roles), key=int):
        was, now = recorded_roles.get(number), claimed_roles.get(number)
        if (was or ([], None))[0] != (now or ([], None))[0]:
            differences.append(
                f"residue {number}: recorded {_describe_role(was)}, claimed {_describe_role(now)}")

    def by_residue(projection):
        return {tuple(entry["residue_key"]): entry for entry in projection["ligand_instances"]}

    claimed_instances, recorded_instances = by_residue(claimed), by_residue(recorded)
    for key in sorted(set(claimed_instances) & set(recorded_instances)):
        was, now = recorded_instances[key], claimed_instances[key]
        for field, meaning in (("parameter_id", "parameter package"),
                               ("exclusions", "torsion exclusions (resolved content)")):
            if was[field] != now[field]:
                differences.append(f"ligand instance {list(key)}: {meaning} recorded "
                                   f"{was[field]!r}, claimed {now[field]!r}")
    return differences


def _describe_masks(masks) -> str:
    masks = masks or {}
    parts = [f"{k} {masks.get(k)!r}" for k in ("backbone", "sidechain") if masks.get(k)]
    parts += [f"ligand {label} {mask!r}" for label, mask in (masks.get("ligands") or {}).items()]
    return ", ".join(parts) or "no masks recorded"


def _describe_role(role) -> str:
    if role is None:
        return "not selected"
    categories, name = role
    return f"{name} as {'+'.join(categories)}"
