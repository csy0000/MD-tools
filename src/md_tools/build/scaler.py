"""Scaled Hamiltonians as a build step: what `md-openmm build-top --rest2-scaler` does.

    md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config

reads a built System and writes, beside it,

    build/<method>/system_state<n>.xml   one per tau of the schedule
    build/<method>/scaler.yaml           what those files are and how they were scaled
    build/<method>/scaler.log            the same, for a person

Nothing is rebuilt: no force field is loaded and no charge is computed. The scaling is
`remd.protocol.build_rung_systems`, the function every ladder has always integrated, and the omega
decision is `openmm.system.omega_exclusions`, the one enforcing entry point. This module adds
where the evidence comes from (§5 of docs/amber-like-fix/REST2-scaler.md), the record, and the
directory transaction; it decides nothing about the Hamiltonian those two functions do not already
decide. `md_tools.rest2.states` is the reading half.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .strict import ConfigError, Field, Schema, Section

#: The methods a scaled-state directory is named after. A fixed-tau hot run is `cMD`.
METHODS = ("REST2", "cMD")


def _check_schedule(resolved: dict[str, Any]) -> None:
    schedule = resolved["schedule"]
    n, low, high = schedule["n_states"], schedule["tau_min"], schedule["tau_max"]
    if n == 1 and low != high:
        raise ConfigError(
            f"schedule: n_states is 1, so there is one tau, but tau_min = {low} and tau_max = "
            f"{high}. Set both to the tau of the single state.")
    if n >= 2 and not low < high:
        raise ConfigError(
            f"schedule: {n} states need tau_min < tau_max, got tau_min = {low} and tau_max = "
            f"{high}; equal endpoints would write {n} copies of one Hamiltonian.")
    if resolved["method"] == "cMD" and n != 1:
        raise ConfigError(
            f"method cMD integrates ONE fixed-tau Hamiltonian, but the schedule has {n} states. "
            f"Use n_states: 1 with tau_min == tau_max, or method: REST2 for a ladder.")


def _check_residue_settings(resolved: dict[str, Any]) -> None:
    files = resolved["sdf_filelist"]
    if files is not None:
        bad = [f"{k!r}: {v!r}" for k, v in files.items()
               if not isinstance(k, str) or not isinstance(v, str)]
        if bad:
            raise ConfigError(f"sdf_filelist maps a residue NAME to an SDF PATH, both strings; "
                              f"got {', '.join(bad)}")
    names = resolved["proline_like_residues"]
    if not all(isinstance(name, str) for name in names):
        raise ConfigError(f"proline_like_residues must be residue names, got {names!r}")
    resolved["proline_like_residues"] = list(names)


SCALER_SCHEMA = Schema(
    "scaler.config",
    doc="Scaled Hamiltonians from a built System, for `md-openmm build-top --rest2-scaler`.",
    fields=[
        Field("method", str, enum=METHODS,
              doc="Which method these states are for; also the output directory "
                  "build/<method>/. REST2 is a ladder; cMD is one fixed-tau hot state."),
        Field("omega_exclusion", bool, default=True,
              doc="Leave ordinary amide omega torsions unscaled. false scales every solute "
                  "torsion, including ordinary amide omegas, and classifies nothing."),
        Field("sdf_filelist", dict, default=None, nullable=True,
              doc="Residue NAME -> SDF describing it, relative to this file. A non-standard "
                  "residue not listed is looked for as <NAME>.sdf beside the System, then as "
                  "built.sdf when it is the only non-standard residue."),
        Field("proline_like_residues", list, default=["PRO"],
              doc="Residue names whose amide nitrogen is ring-locked: those omegas stay ELIGIBLE "
                  "for scaling, and need no SDF."),
        Field("max_proline_ring_size", int, default=7, minimum=3, maximum=12,
              doc="From bond orders, an amide nitrogen in a ring of at most this many atoms is "
                  "proline-like. The bound is what keeps a macrocycle's omegas ordinary."),
    ],
    sections=[
        Section("schedule", [
            Field("kind", str, default="linear", enum=("linear",),
                  doc="How tau is spaced. linear is the one implemented kind."),
            Field("n_states", int, default=4, minimum=1, maximum=64,
                  doc="How many scaled Systems to write."),
            Field("tau_min", float, default=0.0, minimum=0.0, maximum=0.95,
                  doc="tau of state 0. Above 0, NO state is the physical Hamiltonian."),
            Field("tau_max", float, default=0.5, minimum=0.0, maximum=0.95,
                  doc="tau of the last state."),
        ], doc="The tau of every state."),
    ],
    checks=[_check_schedule, _check_residue_settings],
)


def schedule_taus(schedule: Mapping[str, Any]) -> list[float]:
    """The taus of a resolved schedule, from the ONE `tau_ladder`."""
    from ..remd.generated import tau_ladder

    if int(schedule["n_states"]) == 1:
        return [round(float(schedule["tau_max"]), 6)]
    return tau_ladder(int(schedule["n_states"]), float(schedule["tau_max"]),
                      tau_min=float(schedule["tau_min"]))


# --- §5: which SDF describes which residue ------------------------------------------------------

def resolve_residue_sdfs(topology, solute, *, system_dir, config_dir,
                         sdf_filelist: Mapping[str, str] | None,
                         proline_like_residues) -> dict[str, Path]:
    """`{residue name: SDF}` for every non-standard residue that holds an amide nitrogen.

    Looked for in order: `sdf_filelist[name]` (relative to the configuration), `<name>.sdf` beside
    the System, and `built.sdf` beside the System ONLY when the solute has exactly one non-standard
    residue name -- which is what `build-top` writes for a `.smi`/`.sdf` input. Anything else is
    refused with every path looked for: "the first residue" is an ordering accident, and handing
    one residue another's SDF classifies against the wrong molecule.

    A residue with no amide nitrogen needs no SDF, and a declared proline-like name needs none.
    """
    from ..openmm.system import PROTEIN_RESIDUES, _amide_candidates

    solute = {int(i) for i in solute}
    system_dir, config_dir = Path(system_dir), Path(config_dir)
    proline = {str(name).upper() for name in proline_like_residues}
    spelled: dict[str, str] = {}
    for residue in topology.residues():
        if any(atom.index in solute for atom in residue.atoms()):
            spelled.setdefault(residue.name.upper(), residue.name)
    non_standard = {name for name in spelled if name not in PROTEIN_RESIDUES and name not in proline}

    listed = {str(k).upper(): v for k, v in (sdf_filelist or {}).items()}
    strangers = sorted(set(listed) - non_standard)
    if strangers:
        raise ConfigError(
            f"sdf_filelist names {strangers}, which are not non-standard solute residues of this "
            f"topology (those are {sorted(non_standard) or 'none'}). A map naming a residue that is "
            f"not there describes a different system.")

    needed = sorted({cand["nitrogen_residue"].upper() for cand in _amide_candidates(topology, solute)
                     if cand["nitrogen_residue"].upper() in non_standard})
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in sorted(set(needed) | set(listed)):
        if name in listed:
            path = Path(listed[name])
            path = path if path.is_absolute() else config_dir / path
            if not path.is_file():
                missing.append(f"  {spelled[name]}: sdf_filelist names {path}, which does not exist")
                continue
            found[name] = path
            continue
        own = system_dir / f"{spelled[name]}.sdf"
        built = system_dir / "built.sdf"
        if own.is_file():
            found[name] = own
        elif len(non_standard) == 1 and built.is_file():
            found[name] = built
        else:
            looked = [str(own)] + ([str(built)] if len(non_standard) == 1 else [])
            why = ("" if len(non_standard) == 1 else
                   f" (built.sdf is not used: the solute has {len(non_standard)} non-standard "
                   f"residue names, and it cannot say which one it describes)")
            missing.append(f"  {spelled[name]}: looked for {', '.join(looked)}{why}")
    if missing:
        raise ConfigError(
            "these residues hold an amide nitrogen whose omega can only be classified from bond "
            "orders, and no SDF was found for them:\n" + "\n".join(missing) + "\n"
            "Put <NAME>.sdf beside the System, or map each name in sdf_filelist.")
    return found


def optional_residue_sdfs(topology, solute, *, system_dir, config_dir,
                          sdf_filelist: Mapping[str, str] | None,
                          proline_like_residues) -> dict[str, Path]:
    """The same lookup as `resolve_residue_sdfs`, for EVERY non-standard residue, refusing nothing.

    For the pictures: a residue with no amide needs no SDF to be classified, but its drawing still
    says "nothing here is left unscaled", which is worth seeing. A name with no SDF is simply absent.
    """
    from ..openmm.system import PROTEIN_RESIDUES

    solute = {int(i) for i in solute}
    system_dir, config_dir = Path(system_dir), Path(config_dir)
    proline = {str(name).upper() for name in proline_like_residues}
    spelled: dict[str, str] = {}
    for residue in topology.residues():
        if any(atom.index in solute for atom in residue.atoms()):
            spelled.setdefault(residue.name.upper(), residue.name)
    non_standard = {name for name in spelled if name not in PROTEIN_RESIDUES and name not in proline}
    listed = {str(k).upper(): v for k, v in (sdf_filelist or {}).items()}
    found: dict[str, Path] = {}
    for name in sorted(non_standard):
        if name in listed:
            path = Path(listed[name])
            path = path if path.is_absolute() else config_dir / path
        elif (system_dir / f"{spelled[name]}.sdf").is_file():
            path = system_dir / f"{spelled[name]}.sdf"
        elif len(non_standard) == 1:
            path = system_dir / "built.sdf"
        else:
            continue
        if path.is_file():
            found[name] = path
    return found


# --- a picture of what is left unscaled ----------------------------------------------------------

#: The one colour the pictures use for meaning. Atoms are drawn black and white, so red is never
#: an oxygen: it is only ever "left unscaled".
UNSCALED_RGB = (1.0, 0.0, 0.0)


def depict_unscaled_torsions(topology, solute, residue_sdfs: Mapping[str, Path],
                             unscaled_bonds, out_dir, *, omega_exclusion: bool = True,
                             size: tuple[int, int] = (600, 450)) -> dict[str, dict[str, Any]]:
    """`<RESNAME>-unscaled.png` per small-molecule residue: its unscaled torsions' bonds in RED.

    An excluded central bond leaves EVERY torsion across it unscaled, so the bond is what is
    coloured, with its two atoms. The drawing is the SDF's 2D depiction without hydrogens, in a
    black-and-white atom palette; the caption lists the same bonds by the same topology indices
    `scaler.yaml` uses, so the picture and the record can be checked against each other.

    Drawn from the FIRST instance of each residue; every instance has the same chemistry. An SDF
    whose atom count or element sequence does not match the residue is not drawn -- a picture of
    the wrong molecule is worse than none -- and is returned under `skipped` with the reason.

    Returns `{NAME: {file, unscaled_bonds, caption}}`; skipped names are logged by the caller.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    solute = {int(i) for i in solute}
    out_dir = Path(out_dir)
    bonds = [tuple(int(a) for a in bond) for bond in unscaled_bonds]
    drawn: dict[str, dict[str, Any]] = {}
    for name, sdf in sorted(residue_sdfs.items()):
        residue = next((r for r in topology.residues() if r.name.upper() == name.upper()
                        and any(a.index in solute for a in r.atoms())), None)
        if residue is None:
            continue
        atoms = sorted((a for a in residue.atoms() if a.index in solute), key=lambda a: a.index)
        mol = Chem.MolFromMolFile(str(sdf), removeHs=False)
        if (mol is None or mol.GetNumAtoms() != len(atoms)
                or any(mol.GetAtomWithIdx(i).GetSymbol() != (a.element.symbol if a.element else None)
                       for i, a in enumerate(atoms))):
            continue
        for i, atom in enumerate(atoms):
            mol.GetAtomWithIdx(i).SetIntProp("topology_index", int(atom.index))
        members = {a.index for a in atoms}
        mine = sorted(bond for bond in bonds if bond[0] in members and bond[1] in members)

        heavy = Chem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(heavy)
        where = {atom.GetIntProp("topology_index"): atom.GetIdx() for atom in heavy.GetAtoms()
                 if atom.HasProp("topology_index")}
        red_atoms, red_bonds = set(), set()
        for a, b in mine:
            if a in where and b in where:
                red_atoms |= {where[a], where[b]}
                bond = heavy.GetBondBetweenAtoms(where[a], where[b])
                if bond is not None:
                    red_bonds.add(bond.GetIdx())
        # The topology index beside each red atom, so "1-3" in the caption and in scaler.yaml can be
        # found in the picture without counting atoms.
        for index in red_atoms:
            atom = heavy.GetAtomWithIdx(index)
            atom.SetProp("atomNote", str(atom.GetIntProp("topology_index")))

        if not omega_exclusion:
            caption = f"{residue.name}: omega exclusion OFF -- no torsion is left unscaled"
        elif mine:
            caption = (f"{residue.name}: red = unscaled torsions across bond(s) "
                       + ", ".join(f"{a}-{b}" for a, b in mine))
        else:
            caption = f"{residue.name}: no torsion left unscaled"

        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        options = drawer.drawOptions()
        options.useBWAtomPalette()
        options.legendFontSize = 18
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, heavy, legend=caption,
            highlightAtoms=sorted(red_atoms), highlightBonds=sorted(red_bonds),
            highlightAtomColors={i: UNSCALED_RGB for i in red_atoms},
            highlightAtomRadii={i: 0.25 for i in red_atoms},
            highlightBondColors={i: UNSCALED_RGB for i in red_bonds})
        drawer.FinishDrawing()
        target = out_dir / f"{residue.name}-unscaled.png"
        target.write_bytes(drawer.GetDrawingText())
        drawn[name] = {"file": target.name, "unscaled_bonds": [list(b) for b in mine],
                       "caption": caption}
    return drawn


# --- the build ------------------------------------------------------------------------------------

def _plain(value):
    """Tuples to lists, and nothing YAML cannot write: the record is read back and compared."""
    return json.loads(json.dumps(value))


def _sha256(path: Path) -> str:
    from ..build.record import sha256_file

    return sha256_file(Path(path))


def build_scaled_states(*, system_path, topology_path, config_path, overwrite: bool = False,
                        check: bool = False, echo: bool = True) -> dict[str, Any]:
    """Scale a built System to every tau of `scaler.config`. Returns the record written.

    Every refusal happens before a file or directory exists. The states, `scaler.log` and
    `scaler.yaml` are written into a staging directory beside the target and the directory is then
    renamed into place, `scaler.yaml` last inside it, so `build/<method>/` is either the previous
    complete set or the new complete set. With `overwrite`, the previous set is MOVED ASIDE to
    `.<method>.replaced-<utc>`, never deleted.

    With `check`, everything is validated -- including constructing every scaled System -- and
    nothing is created. The return value then carries no file digests.
    """
    from openmm import XmlSerializer

    from ..md.stage import solute_atom_indices
    from ..openmm.system import UnclassifiedOmegaError, omega_exclusions
    from ..remd.protocol import build_rung_systems
    from ..rest2 import REST2_IMPLEMENTATION, scaling_for_tau, torsion_exclusion_report
    from ..rest2.states import (RECORD_FORMAT, RECORD_NAME, ScaledStateError,
                                scaled_state_identity, state_system_name)
    from ..run.preflight import PreflightError, load_inputs
    from .record import LogWriter, source_commit
    from .. import __version__

    system_path, topology_path = Path(system_path).resolve(), Path(topology_path).resolve()
    config_path = Path(config_path).resolve()
    config = SCALER_SCHEMA.load(config_path)
    method = config["method"]

    for role, path in (("-s", system_path), ("-p", topology_path)):
        if not path.is_file():
            raise ConfigError(f"{role} {path}: no such file")
    try:
        already = scaled_state_identity(system_path)
    except ScaledStateError as refusal:
        raise ConfigError(str(refusal)) from None
    if already is not None:
        raise ConfigError(
            f"-s {system_path} is already scaled: state {already['state']} (tau {already['tau']}) "
            f"of {already['record']}. Scaling it again takes solute-solute to (1-tau)^4. Pass the "
            f"unscaled System that record names as its source.")
    try:
        loaded = load_inputs(topology_path, system_path)
    except PreflightError as refusal:
        raise ConfigError(str(refusal)) from None

    parent = system_path.parent
    target = parent / method
    if target.exists() and not overwrite:
        raise ConfigError(
            f"{target} already exists. Pass --overwrite to replace it: the current states are then "
            f"moved aside, not deleted, and a run that used them refuses to continue on new ones.")

    topology = loaded.pdb.topology
    solute = solute_atom_indices(topology)
    taus = schedule_taus(config["schedule"])
    state0_is_physical = taus[0] == 0.0

    residue_sdfs: dict[str, Path] = {}
    if config["omega_exclusion"]:
        residue_sdfs = resolve_residue_sdfs(
            topology, solute, system_dir=parent, config_dir=config_path.parent,
            sdf_filelist=config["sdf_filelist"],
            proline_like_residues=config["proline_like_residues"])
        try:
            omega = omega_exclusions(topology, solute, residue_sdfs=residue_sdfs,
                                     proline_like_residues=config["proline_like_residues"],
                                     max_proline_ring_size=config["max_proline_ring_size"])
        except UnclassifiedOmegaError as refusal:
            raise ConfigError(f"{method} states for {topology_path.name}: {refusal}") from None
    else:
        omega = {"omega_unscaled_bonds": [], "omega_proline_like_scaled_bonds": [],
                 "omega_detection_method": "disabled (omega_exclusion: false): every solute "
                                           "torsion is scaled, including ordinary amide omegas",
                 "omega_detail": {"unscaled": [], "proline_like_scaled": []}}
    excluded = [tuple(int(a) for a in bond) for bond in omega["omega_unscaled_bonds"]]

    try:
        systems, audit = build_rung_systems(loaded.system, solute, tuple(taus),
                                            excluded_bonds=excluded)
    except Exception as broken:
        raise ConfigError(f"the scaled Systems could not be constructed: "
                          f"{type(broken).__name__}: {broken}") from None

    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "method": method,
        "schedule": dict(config["schedule"]),
        "state0_is_physical": state0_is_physical,
        "states": [],
        "source": {"system": system_path.name, "system_sha256": _sha256(system_path),
                   "topology": topology_path.name, "topology_sha256": _sha256(topology_path)},
        "config": {"file": config_path.name, "sha256": _sha256(config_path)},
        "solute": {"n_atoms": len(solute),
                   "atom_range": [min(solute), max(solute)] if solute else None},
        "omega": {
            "exclusion": config["omega_exclusion"],
            "method": omega["omega_detection_method"],
            "unscaled_bonds": [list(bond) for bond in omega["omega_unscaled_bonds"]],
            "proline_like_scaled_bonds": [list(bond)
                                          for bond in omega["omega_proline_like_scaled_bonds"]],
            "decisions": omega.get("omega_detail"),
            "residue_sdfs": {name: {"file": os.path.relpath(path, parent), "sha256": _sha256(path)}
                             for name, path in sorted(residue_sdfs.items())},
            "proline_like_residues": config["proline_like_residues"],
            "max_proline_ring_size": config["max_proline_ring_size"],
            "excluded_torsions": torsion_exclusion_report(loaded.system, solute, excluded),
        },
        "convention": dict(REST2_IMPLEMENTATION),
        "forces": {bucket: entries for bucket, entries in audit.items()
                   if bucket in ("scaled", "unscaled_by_convention", "energy_free")},
        "md_tools": {"version": __version__, "commit": source_commit()},
    }
    if check:
        record["omega"]["depictions"] = {}
        record["states"] = [{"state": i, "file": state_system_name(i), "tau": tau}
                            for i, tau in enumerate(taus)]
        return _plain(record)

    staging = Path(tempfile.mkdtemp(prefix=f".{method}.staging-", dir=parent))
    for index, (tau, system) in enumerate(zip(taus, systems)):
        path = staging / state_system_name(index)
        path.write_text(XmlSerializer.serialize(system), encoding="utf-8")
        solute_solute, solute_environment = scaling_for_tau(tau)
        record["states"].append({
            "state": index, "file": path.name, "tau": tau,
            "sha256": _sha256(path), "bytes": path.stat().st_size,
            "scaling": {"solute_solute": solute_solute, "solute_environment": solute_environment,
                        "solute_solute_expression": "(1 - tau)^2",
                        "solute_environment_expression": "1 - tau"}})
    pictures = optional_residue_sdfs(
        topology, solute, system_dir=parent, config_dir=config_path.parent,
        sdf_filelist=config["sdf_filelist"],
        proline_like_residues=config["proline_like_residues"])
    pictures.update(residue_sdfs)
    depictions = depict_unscaled_torsions(topology, solute, pictures, excluded, staging,
                                          omega_exclusion=config["omega_exclusion"])
    record["omega"]["depictions"] = {
        name: dict(facts, sha256=_sha256(staging / facts["file"]))
        for name, facts in sorted(depictions.items())}
    record = _plain(record)

    aside = None
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        aside = parent / f".{method}.replaced-{stamp}"
        suffix = 1
        while aside.exists():
            aside = parent / f".{method}.replaced-{stamp}-{suffix}"
            suffix += 1

    log = LogWriter(staging / "scaler.log", record_type="rest2-scaler", echo=echo)
    log.heading(f"{method} scaled states")
    log.field("source System", f"{system_path.name}  sha256 {record['source']['system_sha256']}")
    log.field("source topology", topology_path.name)
    log.field("configuration", f"{config_path.name}  sha256 {record['config']['sha256']}")
    log.field("schedule", f"{config['schedule']['kind']}, {len(taus)} state(s), "
                          f"tau {taus[0]} .. {taus[-1]}")
    if not state0_is_physical:
        log(f"  state 0 is at tau = {taus[0]}: it is NOT the physical Hamiltonian. REST2 recovers "
            f"the physical ensemble only from an unscaled state, so no state here samples it.")
    log.field("solute", f"{len(solute)} atom(s)")
    log.field("omega exclusion", "on" if config["omega_exclusion"] else
              "OFF -- every solute torsion scaled, including ordinary amide omegas")
    log.field("omega unscaled", f"{len(record['omega']['unscaled_bonds'])} bond(s), "
                                f"{record['omega']['excluded_torsions']['n_excluded_torsions']} "
                                f"torsion(s) protected")
    log.field("omega proline-like", f"{len(record['omega']['proline_like_scaled_bonds'])} bond(s), "
                                    f"scaled")
    for name, facts in record["omega"]["residue_sdfs"].items():
        log.field(f"SDF for {name}", facts["file"])
    for name, facts in record["omega"]["depictions"].items():
        log.field(f"picture of {name}", f"{facts['file']}  ({facts['caption']})")
    log.heading("States")
    for state in record["states"]:
        log(f"  {state['file']:<22} tau {state['tau']:<9} (1-tau)^2 "
            f"{state['scaling']['solute_solute']:.6f}   1-tau "
            f"{state['scaling']['solute_environment']:.6f}")
    if aside is not None:
        log.field("replaced", f"the previous {method}/ was moved to {aside.name}")
    log.update(scaler=record)
    log.complete()
    log.save()

    partial = staging / (RECORD_NAME + ".partial")
    import yaml

    partial.write_text(yaml.safe_dump(record, sort_keys=False, default_flow_style=False,
                                      width=100), encoding="utf-8")
    os.replace(partial, staging / RECORD_NAME)

    if aside is not None:
        os.rename(target, aside)
    os.rename(staging, target)
    return record
