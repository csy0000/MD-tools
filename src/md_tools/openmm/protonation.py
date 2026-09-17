"""Protein protonation: PROPKA3 predictions, their translation into residue variants, and warnings.

THREE SEPARATE THINGS, kept separate on purpose (plan section 4):

  prediction   PROPKA3 predicts pKa values. It does not choose force-field residues, and it does
               NOT decide between the two neutral histidine tautomers HID and HIE.
  assignment   md-tools translates a prediction into a variant OpenMM's `addHydrogens` can build,
               by one deterministic rule (`assign_variants`). A predicted state no supported
               variant represents is REPORTED, never quietly turned into a different one.
  warnings     A histidine within `histidine_proximity_angstrom` of a ligand or an ion is screened
               and printed with the measured distances. Proximity is not proof of coordination,
               so a warning never imposes a tautomer; an explicit per-residue override does.

`method: openmm` keeps the behaviour md-tools always had -- `addHydrogens(pH)` chooses -- and is
recorded as a choice, not as an absence. `method: propka` with PROPKA missing, or a prediction that
fails, is an ERROR: a silent fall back to OpenMM's defaults would build a different protein and
record that PROPKA had been asked for.

Residues are identified by (chain id, residue id, insertion code) everywhere -- in overrides, in
the record and in warnings -- never by position in a list, which any insertion of atoms or
residues shifts.

Ligand instances mapped to parameter packages are FROZEN: their hydrogens are the package's, and
nothing here deletes or adds an atom in them. PROPKA's view of a ligand is recorded as a warning
only; a different chemical state needs a different package.

Constant-pH MD is not implied: an ordinary run holds these states fixed.
"""
from __future__ import annotations

import hashlib
import io
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

#: The residue names PROPKA reports for protein titratable groups, and the variants OpenMM's
#: `Modeller.addHydrogens` can build for each (openmm/app/data/hydrogens.xml). Anything PROPKA
#: predicts outside these -- a deprotonated cysteine (CYM), tyrosinate, neutral arginine, a
#: neutral N terminus or a protonated C terminus -- is reported as UNSUPPORTED and the standard
#: state is kept, with the prediction in the record.
SUPPORTED_VARIANTS = {
    "ASP": ("ASP", "ASH"),
    "GLU": ("GLU", "GLH"),
    "HIS": ("HID", "HIE", "HIP"),
    "LYS": ("LYS", "LYN"),
    "CYS": ("CYS", "CYX"),
}

#: Every residue name a family may already carry in an input, mapped to its family.
FAMILY = {"ASP": "ASP", "ASH": "ASP", "GLU": "GLU", "GLH": "GLU",
          "HIS": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
          "LYS": "LYS", "LYN": "LYS", "CYS": "CYS", "CYX": "CYS", "CYM": "CYS"}

#: PROPKA's model pKa values (propka.cfg), used ONLY for a titratable residue PROPKA returned no
#: prediction for -- an incomplete side chain, say -- so its state is still a stated rule rather
#: than whatever `addHydrogens` does at some other pH. Every such residue is flagged.
MODEL_PKA = {"ASP": 3.8, "GLU": 4.5, "HIS": 6.5, "LYS": 10.5, "CYS": 9.0}

#: PROPKA group types that have no supported variant in either direction, with what a prediction
#: across the target pH would have meant.
UNSUPPORTED_GROUPS = {
    "TYR": ("deprotonated tyrosine (tyrosinate)", lambda pka, ph: pka < ph),
    "ARG": ("neutral arginine", lambda pka, ph: pka < ph),
    "N+": ("neutral N terminus", lambda pka, ph: pka < ph),
    "C-": ("protonated C terminus", lambda pka, ph: pka > ph),
}

METHODS = ("openmm", "propka")

#: Default screening distance for histidine proximity warnings, heavy atom to heavy atom.
DEFAULT_PROXIMITY_ANGSTROM = 5.0

#: A prediction within this many pH units of the target is flagged: small changes in structure
#: or method move it across, and the assigned state should be read as uncertain.
DEFAULT_NEAR_PH_WINDOW = 1.0


class ProtonationError(ValueError):
    """A protonation request that cannot be carried out as stated. Refused before any output."""


Key = tuple[str, str, str]


def residue_key(residue) -> Key:
    """(chain id, residue id, insertion code) of an OpenMM Residue."""
    return (str(residue.chain.id), str(residue.id).strip(), str(residue.insertionCode or "").strip())


def format_key(key: Key) -> str:
    chain, resid, icode = key
    return f"{chain}:{resid}{icode}"


# --- prediction -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    group: str            # PROPKA group type: ASP, GLU, HIS, LYS, CYS, TYR, ARG, N+, C-, or a ligand type
    key: Key
    pka: float
    model_pka: float
    coupled: bool


def propka_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("propka")
    except PackageNotFoundError:
        raise ProtonationError(
            "protonation.method is propka, but PROPKA3 is not installed in this environment. "
            "Install it (`pip install propka==3.5.1`, pinned in environment.yml) or set "
            "protonation.method: openmm deliberately. Nothing was built.") from None


def run_propka(topology, positions, *, ph: float, workdir: Path) -> dict[str, Any]:
    """Predict pKa values for the structure AS GIVEN, and keep everything PROPKA said.

    The structure is written to `workdir/propka_input.pdb` with its residue ids kept, so PROPKA's
    chain/residue labels are the topology's. Its log -- missing atoms, groups it could not set
    up, unsupported residues -- is captured in full; the `.pka` report is kept beside the input.
    """
    from openmm import app

    version = propka_version()
    import propka.run

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pdb_path = workdir / "propka_input.pdb"
    with pdb_path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
    digest = hashlib.sha256(pdb_path.read_bytes()).hexdigest()

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("propka")
    previous = (logger.level, logger.propagate)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # PROPKA prints its header and some diagnostics with print() as well as through logging, so
    # both go into the record rather than onto the build-top console mixed with md-tools' own.
    import contextlib

    try:
        with pdb_path.open("r") as handle, contextlib.redirect_stdout(stream), \
                contextlib.redirect_stderr(stream):
            molecule = propka.run.single(pdb_path.name, optargs=["--pH", str(float(ph))],
                                         stream=handle, write_pka=False)
            report = workdir / "propka_output.pka"
            molecule.write_pka(filename=str(report))
    except ProtonationError:
        raise
    except Exception as failure:
        raise ProtonationError(
            f"PROPKA {version} failed on {pdb_path}: {type(failure).__name__}: {failure}. "
            f"protonation.method is propka, so this is not replaced by OpenMM's defaults. "
            f"Nothing was built.") from None
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous[0])
        logger.propagate = previous[1]

    conformation = molecule.conformations["AVR"]
    predictions = []
    for group in conformation.groups:
        atom = group.atom
        icode = str(getattr(atom, "icode", "") or "").strip()
        predictions.append(Prediction(
            group=str(group.residue_type), key=(str(atom.chain_id), str(atom.res_num), icode),
            pka=float(group.pka_value), model_pka=float(group.model_pka),
            coupled=bool(getattr(group, "coupled_titrating_group", None))))
    log_text = stream.getvalue()
    return {
        "version": version,
        "ph": float(ph),
        "input_pdb": pdb_path.name,
        "input_sha256": digest,
        "report": report.name,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "log": [line for line in log_text.splitlines() if line.strip()],
        "predictions": predictions,
    }


# --- assignment ------------------------------------------------------------------------------------

@dataclass
class Assignment:
    key: Key
    residue: str                      # the residue name in the input
    family: str
    variant: str | None               # None: addHydrogens chooses (CYS/CYX by bond; neutral HIS)
    source: str
    pka: float | None = None
    model_pka: float | None = None
    near_ph: bool = False
    coupled: bool = False
    notes: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        chain, resid, icode = self.key
        return {"chain": chain, "resid": resid, "insertion_code": icode, "residue": self.residue,
                "variant": self.variant, "source": self.source, "pka": self.pka,
                "model_pka": self.model_pka, "near_ph": self.near_ph, "coupled": self.coupled,
                "notes": list(self.notes)}


def parse_overrides(overrides: Iterable[Mapping[str, Any]] | None) -> dict[Key, str]:
    """`[{select: {chain, resid, insertion_code}, variant}]` -> {key: variant}, strictly."""
    parsed: dict[Key, str] = {}
    for number, entry in enumerate(overrides or (), start=1):
        if not isinstance(entry, Mapping) or set(entry) - {"select", "variant"} \
                or "select" not in entry or "variant" not in entry:
            raise ProtonationError(
                f"protonation.overrides[{number}] must be {{select: {{chain, resid, "
                f"insertion_code}}, variant}}, got {entry!r}")
        select = entry["select"]
        if not isinstance(select, Mapping) or set(select) - {"chain", "resid", "insertion_code"} \
                or "chain" not in select or "resid" not in select:
            raise ProtonationError(
                f"protonation.overrides[{number}].select must name chain and resid (and "
                f"optionally insertion_code), got {select!r}")
        key = (str(select["chain"]), str(select["resid"]).strip(),
               str(select.get("insertion_code") or "").strip())
        variant = str(entry["variant"]).upper()
        if variant not in FAMILY or variant not in SUPPORTED_VARIANTS[FAMILY[variant]]:
            supported = sorted(v for vs in SUPPORTED_VARIANTS.values() for v in vs)
            raise ProtonationError(
                f"protonation.overrides[{number}]: variant {variant!r} is not one addHydrogens "
                f"can build. Supported: {', '.join(supported)}.")
        if key in parsed:
            raise ProtonationError(f"protonation.overrides names {format_key(key)} twice")
        parsed[key] = variant
    return parsed


def _disulfide_cysteines(topology) -> set[Key]:
    bonded = set()
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        if a.residue is not b.residue and a.name == "SG" and b.name == "SG":
            bonded.add(residue_key(a.residue))
            bonded.add(residue_key(b.residue))
    return bonded


def assign_variants(topology, *, method: str, ph: float,
                    predictions: Sequence[Prediction] = (),
                    overrides: Mapping[Key, str] | None = None,
                    frozen_residues: Iterable[Key] = (),
                    near_ph_window: float = DEFAULT_NEAR_PH_WINDOW
                    ) -> tuple[list[Assignment], list[dict[str, Any]]]:
    """Per-residue variants by one deterministic rule. Returns (assignments, unsupported).

    Rule, at target pH `ph`, for each titratable protein residue:
      override given       -> that variant (it must belong to the residue's family)
      CYS in a disulfide   -> CYX
      ASP/GLU  pKa >  pH   -> ASH/GLH,  else ASP/GLU
      LYS      pKa <  pH   -> LYN,      else LYS
      HIS      pKa >  pH   -> HIP,      else NEUTRAL: None, and OpenMM's hydrogen-bond heuristic
                              chooses HID or HIE (PROPKA does not resolve the tautomer)
      CYS      pKa <  pH   -> UNSUPPORTED (CYM is not an addHydrogens variant): CYS kept, reported
      no prediction        -> the same rule on PROPKA's MODEL pKa, flagged
    `method: openmm` assigns only overrides and disulfides and leaves every other choice to
    `addHydrogens(pH)`, as md-tools always did.

    An override naming a residue that does not exist, or a variant from another family, is an
    error. Frozen residues (mapped ligand instances) are never assigned.
    """
    if method not in METHODS:
        raise ProtonationError(f"protonation.method must be one of {METHODS}, got {method!r}")
    overrides = dict(overrides or {})
    frozen = set(frozen_residues)
    residues = {residue_key(r): r for r in topology.residues()}
    for key, variant in overrides.items():
        if key not in residues:
            raise ProtonationError(
                f"protonation.overrides names {format_key(key)}, which is not a residue of this "
                f"structure. Selectors use the chain id, residue id and insertion code the input "
                f"file carries.")
        name = residues[key].name
        if FAMILY.get(name) != FAMILY[variant]:
            raise ProtonationError(
                f"protonation.overrides sets {format_key(key)} ({name}) to {variant}, a variant of "
                f"{FAMILY[variant]}. An override chooses between states of the residue that is "
                f"there; it cannot change which residue it is.")

    by_key: dict[Key, Prediction] = {}
    unsupported: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.group in FAMILY:
            by_key.setdefault(prediction.key, prediction)
        elif prediction.group in UNSUPPORTED_GROUPS:
            meaning, crosses = UNSUPPORTED_GROUPS[prediction.group]
            if crosses(prediction.pka, ph):
                unsupported.append({
                    "group": prediction.group, "chain": prediction.key[0],
                    "resid": prediction.key[1], "insertion_code": prediction.key[2],
                    "pka": round(prediction.pka, 2), "ph": ph,
                    "predicted": meaning, "kept": "standard state",
                    "why": "no force-field variant md-tools can build represents it"})

    disulfides = _disulfide_cysteines(topology)
    assignments: list[Assignment] = []
    for key, residue in residues.items():
        family = FAMILY.get(residue.name)
        if family is None or key in frozen:
            continue
        prediction = by_key.get(key)
        assignment = Assignment(key=key, residue=residue.name, family=family, variant=None,
                                source="")
        if prediction is not None:
            assignment.pka = round(prediction.pka, 2)
            assignment.model_pka = prediction.model_pka
            assignment.coupled = prediction.coupled
            assignment.near_ph = abs(prediction.pka - ph) < near_ph_window
        if key in overrides:
            assignment.variant = overrides[key]
            assignment.source = "override"
            if method == "propka" and prediction is not None:
                rule = _rule(family, prediction.pka, ph, key in disulfides)
                if rule != overrides[key]:
                    assignment.notes.append(
                        f"override {overrides[key]} supersedes the prediction "
                        f"(pKa {prediction.pka:.2f} at pH {ph} -> "
                        f"{rule or 'neutral HID/HIE'})")
        elif family == "CYS" and key in disulfides:
            assignment.variant, assignment.source = "CYX", "disulfide"
        elif method == "openmm":
            assignment.source = f"openmm addHydrogens(pH {ph})"
        else:
            pka = prediction.pka if prediction is not None else MODEL_PKA[family]
            if prediction is None:
                assignment.model_pka = MODEL_PKA[family]
                assignment.notes.append("PROPKA returned no prediction; model pKa used")
            assignment.variant = _rule(family, pka, ph, False)
            assignment.source = "propka" if prediction is not None else "model-pka"
            if family == "HIS" and assignment.variant is None:
                assignment.source += " (neutral) + openmm hydrogen-bond heuristic for HID/HIE"
            if family == "CYS" and pka < ph:
                assignment.notes.append(
                    f"predicted deprotonated (pKa {pka:.2f} < pH {ph}); CYM is not an "
                    f"addHydrogens variant, so neutral CYS is kept")
                unsupported.append({
                    "group": "CYS", "chain": key[0], "resid": key[1], "insertion_code": key[2],
                    "pka": round(pka, 2), "ph": ph, "predicted": "deprotonated cysteine (CYM)",
                    "kept": "CYS", "why": "CYM is not an addHydrogens variant"})
        assignments.append(assignment)
    return assignments, unsupported


def _rule(family: str, pka: float, ph: float, disulfide: bool) -> str | None:
    if family == "ASP":
        return "ASH" if pka > ph else "ASP"
    if family == "GLU":
        return "GLH" if pka > ph else "GLU"
    if family == "LYS":
        return "LYN" if pka < ph else "LYS"
    if family == "HIS":
        return "HIP" if pka > ph else None
    if family == "CYS":
        return "CYX" if disulfide else None
    raise AssertionError(family)


# --- histidine proximity ---------------------------------------------------------------------------

def histidine_proximity(topology, positions, *, neighbours: Iterable[Key],
                        ions: Iterable[str], cutoff_angstrom: float,
                        variants: Mapping[Key, str | None],
                        assignments: Mapping[Key, Assignment]) -> list[dict[str, Any]]:
    """Every histidine with a heavy atom within `cutoff_angstrom` of a ligand or an ion.

    `neighbours` are ligand residue keys; `ions` are ion residue names. For an ion the ND1-ion and
    NE2-ion distances are reported separately, because which nitrogen faces a metal is what
    decides the tautomer a coordinating histidine needs. SCREENING, not a decision: nothing here
    changes a variant.
    """
    import numpy as np
    from openmm import unit

    xyz = np.asarray(positions.value_in_unit(unit.angstrom)
                     if hasattr(positions, "value_in_unit") else positions, dtype=float)
    ion_names = {name.upper() for name in ions}
    ligand_keys = set(neighbours)

    def heavy(residue):
        return [a for a in residue.atoms() if a.element is not None and a.element.symbol != "H"]

    partners = []
    for residue in topology.residues():
        key = residue_key(residue)
        if key in ligand_keys or residue.name.upper() in ion_names:
            atoms = heavy(residue)
            if atoms:
                partners.append((residue, key, residue.name.upper() in ion_names, atoms))

    warnings = []
    for residue in topology.residues():
        if FAMILY.get(residue.name) != "HIS":
            continue
        key = residue_key(residue)
        his_atoms = heavy(residue)
        if not his_atoms:
            continue
        his_xyz = xyz[[a.index for a in his_atoms]]
        named = {a.name: a for a in his_atoms}
        for partner, partner_key, is_ion, atoms in partners:
            other = xyz[[a.index for a in atoms]]
            distances = np.linalg.norm(his_xyz[:, None, :] - other[None, :, :], axis=2)
            i, j = np.unravel_index(int(np.argmin(distances)), distances.shape)
            nearest = float(distances[i, j])
            if nearest > cutoff_angstrom:
                continue
            assignment = assignments.get(key)
            entry = {
                "histidine": {"chain": key[0], "resid": key[1], "insertion_code": key[2]},
                "neighbour": {"chain": partner_key[0], "resid": partner_key[1],
                              "insertion_code": partner_key[2], "residue": partner.name,
                              "kind": "ion" if is_ion else "ligand"},
                "distance_angstrom": round(nearest, 2),
                "closest_atoms": [his_atoms[i].name, atoms[j].name],
                "pka": assignment.pka if assignment else None,
                "variant": variants.get(key),
                "source": assignment.source if assignment else "openmm addHydrogens",
            }
            if is_ion:
                for name in ("ND1", "NE2"):
                    if name in named:
                        entry[f"{name}_ion_distance_angstrom"] = round(float(
                            np.min(np.linalg.norm(other - xyz[named[name].index], axis=1))), 2)
            warnings.append(entry)
    return warnings


def format_proximity_warning(entry: Mapping[str, Any]) -> str:
    his, other = entry["histidine"], entry["neighbour"]
    text = (f"WARNING histidine {his['chain']}:{his['resid']}{his['insertion_code']} is "
            f"{entry['distance_angstrom']:.2f} A from {other['kind']} {other['residue']} "
            f"{other['chain']}:{other['resid']}{other['insertion_code']} "
            f"({entry['closest_atoms'][0]}-{entry['closest_atoms'][1]})")
    if "ND1_ion_distance_angstrom" in entry or "NE2_ion_distance_angstrom" in entry:
        text += (f"; ND1-ion {entry.get('ND1_ion_distance_angstrom', '-')} A, NE2-ion "
                 f"{entry.get('NE2_ion_distance_angstrom', '-')} A")
    pka = entry.get("pka")
    text += (f"; predicted pKa {pka:.2f}" if pka is not None else "; no pKa prediction")
    text += (f"; final variant {entry.get('variant') or 'unresolved'} ({entry['source']}). "
             f"Proximity is not proof of coordination: set protonation.overrides for this "
             f"residue if its tautomer matters.")
    return text


# --- the whole step --------------------------------------------------------------------------------

@dataclass
class ProtonationResult:
    topology: Any
    positions: Any
    record: dict[str, Any]
    warnings: list[str]


def protonate_structure(topology, positions, forcefield, protonation_cfg: Mapping[str, Any], *,
                        frozen_residues: Iterable[Key] = (),
                        residue_templates_for: Callable[[Any], dict] | None = None,
                        seed: int, workdir: Path,
                        ions: Iterable[str] | None = None,
                        echo: Callable[[str], None] | None = print) -> ProtonationResult:
    """Delete non-frozen hydrogens, predict (if asked), assign, add hydrogens, and screen.

    `protonation_cfg` keys: method (openmm|propka), ph, overrides, histidine_proximity_angstrom,
    near_ph_window, delete_existing_hydrogens. `residue_templates_for` is called on the topology
    `addHydrogens` actually receives -- a dict computed earlier would key residues of a topology
    `Modeller.delete` has already replaced.

    Hydrogen placement is seeded and relaxed on the Reference platform, as `system.protonate`
    does, so a build is bit-reproducible.
    """
    from openmm import Platform, app
    from openmm.app import element as elem

    from .solvation import ION_RESIDUE_NAMES

    method = str(protonation_cfg.get("method") or "openmm")
    ph = float(protonation_cfg.get("ph", 7.0))
    cutoff = float(protonation_cfg.get("histidine_proximity_angstrom",
                                       DEFAULT_PROXIMITY_ANGSTROM))
    window = float(protonation_cfg.get("near_ph_window", DEFAULT_NEAR_PH_WINDOW))
    overrides = parse_overrides(protonation_cfg.get("overrides"))
    frozen = set(frozen_residues)
    ions = set(ions if ions is not None else ION_RESIDUE_NAMES)
    workdir = Path(workdir)

    modeller = app.Modeller(topology, positions)
    n_before = sum(1 for a in modeller.topology.atoms() if a.element == elem.hydrogen)
    if protonation_cfg.get("delete_existing_hydrogens", True):
        modeller.delete([a for a in modeller.topology.atoms()
                         if a.element == elem.hydrogen and residue_key(a.residue) not in frozen])

    propka = None
    predictions: Sequence[Prediction] = ()
    if method == "propka":
        propka = run_propka(modeller.topology, modeller.positions, ph=ph, workdir=workdir)
        predictions = propka["predictions"]
    assignments, unsupported = assign_variants(
        modeller.topology, method=method, ph=ph, predictions=predictions, overrides=overrides,
        frozen_residues=frozen, near_ph_window=window)
    by_key = {a.key: a for a in assignments}

    variants = []
    for residue in modeller.topology.residues():
        chosen = by_key.get(residue_key(residue))
        variants.append(chosen.variant if chosen is not None else None)

    # THE pH addHydrogens RECEIVES. Under propka every titratable residue already has an explicit
    # variant except a neutral histidine and a free cysteine, and those two are exactly where
    # addHydrogens reads the pH: a histidine takes its hydrogen-bond HID/HIE heuristic only above
    # pH 6.5, and is made HIP below it. A prediction of NEUTRAL must stay neutral whatever the
    # target pH, so the heuristic is asked for at max(pH, 7.0). Under openmm the target pH is
    # passed unchanged, as it always was.
    add_ph = max(ph, 7.0) if method == "propka" else ph
    templates = residue_templates_for(modeller.topology) if residue_templates_for else None

    reference = Platform.getPlatformByName("Reference")
    state = random.getstate()
    random.seed(int(seed))
    try:
        kwargs = {"pH": add_ph, "variants": variants, "platform": reference}
        if templates:
            kwargs["residueTemplates"] = templates
        chosen_variants = modeller.addHydrogens(forcefield, **kwargs)
    finally:
        random.setstate(state)

    final = {}
    for residue, variant in zip(modeller.topology.residues(), chosen_variants):
        key = residue_key(residue)
        if key in by_key:
            final[key] = None if variant is None else str(variant)
            if by_key[key].variant is None and variant is not None:
                by_key[key].notes.append(f"addHydrogens chose {variant}")

    proximity = histidine_proximity(
        modeller.topology, modeller.positions, neighbours=frozen, ions=ions,
        cutoff_angstrom=cutoff, variants=final, assignments=by_key)
    warnings = [format_proximity_warning(entry) for entry in proximity]
    for entry in unsupported:
        warnings.append(
            f"WARNING {entry['group']} {entry['chain']}:{entry['resid']}{entry['insertion_code']} "
            f"predicted pKa {entry['pka']} at pH {entry['ph']}: {entry['predicted']} is not "
            f"supported ({entry['why']}); the {entry['kept']} is kept.")
    for assignment in assignments:
        for note in assignment.notes:
            if note.startswith("override"):
                warnings.append(f"NOTE {format_key(assignment.key)}: {note}")
    if echo is not None:
        for line in warnings:
            echo(line)

    n_after = sum(1 for a in modeller.topology.atoms() if a.element == elem.hydrogen)
    record = {
        "method": method,
        "ph": ph,
        "addhydrogens_ph": add_ph,
        "hydrogen_seed": int(seed),
        "near_ph_window": window,
        "histidine_proximity_angstrom": cutoff,
        "n_hydrogens_before": n_before,
        "n_hydrogens_after": n_after,
        "frozen_residues": [format_key(k) for k in sorted(frozen)],
        "overrides": [{"chain": k[0], "resid": k[1], "insertion_code": k[2], "variant": v}
                      for k, v in sorted(overrides.items())],
        "assignments": [dict(a.record(), final_variant=final.get(a.key)) for a in assignments],
        "unsupported": unsupported,
        "histidine_proximity": proximity,
        "warnings": warnings,
        "propka": None if propka is None else {
            key: value for key, value in propka.items() if key != "predictions"} | {
            "predictions": [{"group": p.group, "chain": p.key[0], "resid": p.key[1],
                             "insertion_code": p.key[2], "pka": round(p.pka, 2),
                             "model_pka": p.model_pka, "coupled": p.coupled}
                            for p in propka["predictions"]]},
    }
    return ProtonationResult(topology=modeller.topology, positions=modeller.positions,
                             record=record, warnings=warnings)
