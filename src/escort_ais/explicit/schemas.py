"""Versioned manifests for portable explicit-solvent REST2.

Two files, deliberately separate:

* a **system manifest** — molecular identity and how it is parameterised. It answers "what
  molecule is this, exactly", and it must be reproducible on another machine from the file alone.
* an **experiment manifest** — run controls: seeds, integrator, ladder, budget, precision.

Keeping them apart is what lets one system be run under several experiments (and one experiment
template be applied to several systems) without either file quietly encoding the other's choices.

The rules below exist because the failure they prevent has been seen. A force-field route inferred
from a filename or a molecule name silently reinterprets a Sage/AM1-BCC ligand calculation as an
ff19SB peptide one and the run still "works"; the numbers are then wrong for a reason nothing in
the output records. So:

* ``input.route`` is explicit and may never be ``auto`` in a portable manifest;
* the route and the parameterisation block must agree, and a ligand route may not name a protein
  force field;
* the molecule is pinned by its RDKit canonical isomeric SMILES **and** that string's SHA-256, so
  a manifest that has been edited to a different (even isomeric) molecule fails to load;
* ``cyclo_rgdfv`` is pinned harder still — that name may not be attached to an arbitrary neutral
  SMILES, because "a neutral macrocycle called RGD" is exactly what a copy-paste error produces.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

#: Routes a portable manifest may declare. ``auto`` is deliberately absent: it is resolvable only
#: against a specific invocation, which is the opposite of portable.
ROUTES = ("smiles", "pdb")

PRECISIONS = ("single", "mixed", "double")

#: Top-level keys each manifest may carry. Unknown keys are rejected rather than ignored: a typo
#: in a portable manifest otherwise leaves the default silently in force while the file appears to
#: set something.
SYSTEM_KEYS = frozenset({
    "schema_version", "system_id", "display_name", "input", "parameterization", "solvation",
    "chemistry", "notes",
})
EXPERIMENT_KEYS = frozenset({
    "schema_version", "experiment_id", "master_seed", "ladder_status", "integrator", "rest2",
    "platform", "equilibration", "overrides", "notes",
})
PLATFORMS = ("CPU", "CUDA", "OpenCL")

#: The project's cyclo-(RGDfV), as simulated: the zwitterion, C26H38N8O7, net charge zero.
#: Recorded here so the identity is enforced rather than described. Regenerate with
#: ``Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)`` and its SHA-256.
RGD_SYSTEM_ID = "cyclo_rgdfv"
RGD_CANONICAL_SMILES = (
    "CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)"
    "[C@H](CCCNC(N)=[NH2+])NC1=O"
)
RGD_CANONICAL_SMILES_SHA256 = (
    "59d4422635f77292ed94c609a76808df5c97110be09cd9779819270363143159"
)
RGD_EXPECTED_FORMAL_CHARGE = 0

#: Any absolute path under one of these roots in a shipped manifest or a generated artifact means
#: the file will not work on another machine.
_DEVELOPER_PATH_RE = re.compile(r"(?:^|[\s\"'=:])(/home/|/Users/|/mnt/home/)")


class ManifestError(ValueError):
    """A manifest is missing, malformed, or internally inconsistent."""


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def canonical_json(doc: Any) -> str:
    """Deterministic serialisation used for hashing.

    Sorted keys, no insignificant whitespace, UTF-8. Two manifests that differ only in key order
    or formatting must hash identically; two that differ in any *value* must not.
    """
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(system: dict, experiment: dict, *, length: int = 12) -> str:
    """Stable short hash of the (system, experiment) pair, for the run-directory name.

    Derived from the canonical serialisation of both manifests, so it is reproducible on any
    machine and changes whenever any scientific setting or seed changes. It deliberately does NOT
    include the platform or device: the same calculation launched on CPU and on CUDA is the same
    configuration, and giving them different hashes would hide that.
    """
    payload = canonical_json({"system": system, "experiment": experiment})
    return sha256_text(payload)[:length]


def find_developer_paths(text: str) -> list[str]:
    """Absolute developer-home paths in `text`; empty when the text is portable."""
    return [m.group(0).strip(" \"'=:") for m in _DEVELOPER_PATH_RE.finditer(text)]


def _require(doc: dict, key: str, where: str) -> Any:
    if key not in doc:
        raise ManifestError(f"{where}: required field {key!r} is missing")
    return doc[key]


def _load_yaml(path: Path) -> dict:
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:                      # noqa: BLE001 - report the file, not a traceback
        raise ManifestError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ManifestError(f"{path}: expected a mapping at the top level")
    return doc


# ---------------------------------------------------------------------------------------------
# system manifest
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemManifest:
    """A validated system manifest plus the directory it was loaded from."""

    doc: dict
    source: Path

    @property
    def system_id(self) -> str:
        return str(self.doc["system_id"])

    @property
    def route(self) -> str:
        return str(self.doc["input"]["route"])

    @property
    def display_name(self) -> str:
        return str(self.doc.get("display_name", self.system_id))

    @property
    def canonical_hash(self) -> str:
        """The molecular identity hash: canonical SMILES hash, or the PDB's content hash."""
        if self.route == "smiles":
            return str(self.doc["input"]["canonical_smiles_sha256"])
        return str(self.doc["input"]["pdb_sha256"])

    @property
    def formal_charge(self) -> Optional[int]:
        v = self.doc["input"].get("expected_formal_charge")
        return None if v is None else int(v)

    def input_path(self) -> Optional[Path]:
        """Absolute path to the PDB input, resolved relative to the manifest's own directory."""
        if self.route != "pdb":
            return None
        return (self.source.parent / str(self.doc["input"]["pdb"])).resolve()


def load_system(path: Path, *, check_chemistry: bool = True) -> SystemManifest:
    """Load and fully validate a system manifest. Raises `ManifestError` on any violation."""
    path = Path(path).resolve()
    doc = _load_yaml(path)
    validate_system(doc, source=path, check_chemistry=check_chemistry)
    return SystemManifest(doc=doc, source=path)


def validate_system(doc: dict, *, source: Path, check_chemistry: bool = True) -> None:
    """Every check the instruction requires, in the order a reader would want them to fail.

    `check_chemistry=False` skips only the RDKit-dependent checks (canonical SMILES, stereo,
    formal charge), so structural validation still works in an environment without RDKit. It is
    never the default, and `prepare`/`rest2` never use it.
    """
    where = str(source)

    unknown = sorted(set(doc) - SYSTEM_KEYS)
    if unknown:
        raise ManifestError(
            f"{where}: unknown top-level key(s) {unknown}; allowed: {sorted(SYSTEM_KEYS)}"
        )
    version = _require(doc, "schema_version", where)
    if int(version) != SCHEMA_VERSION:
        raise ManifestError(
            f"{where}: schema_version {version} is not supported (this build reads "
            f"schema_version {SCHEMA_VERSION})"
        )
    system_id = str(_require(doc, "system_id", where))
    if not re.fullmatch(r"[a-z0-9_]+", system_id):
        raise ManifestError(
            f"{where}: system_id {system_id!r} must be lowercase alphanumeric with underscores; "
            "it becomes part of a run-directory name"
        )
    _require(doc, "display_name", where)

    inp = _require(doc, "input", where)
    if not isinstance(inp, dict):
        raise ManifestError(f"{where}: 'input' must be a mapping")
    route = str(_require(inp, "route", f"{where}:input"))
    if route == "auto":
        raise ManifestError(
            f"{where}: input.route may not be 'auto' in a portable manifest. The force-field "
            "route must be stated, or a PDB invocation silently becomes a peptide calculation."
        )
    if route not in ROUTES:
        raise ManifestError(f"{where}: input.route {route!r} must be one of {ROUTES}")

    par = _require(doc, "parameterization", where)
    if not isinstance(par, dict):
        raise ManifestError(f"{where}: 'parameterization' must be a mapping")
    for key in ("small_molecule_forcefield", "charge_method", "protein_forcefield",
                "water_forcefield"):
        if key not in par:
            raise ManifestError(f"{where}:parameterization: required field {key!r} is missing")
    smff = par["small_molecule_forcefield"]
    prot = par["protein_forcefield"]
    if "auto" in (str(smff).lower(), str(prot).lower()):
        raise ManifestError(f"{where}: 'auto' is not a force field; name it explicitly")
    if not par["water_forcefield"]:
        raise ManifestError(f"{where}:parameterization: water_forcefield is required")

    # ---- route / parameterisation agreement --------------------------------------------------
    if route == "smiles":
        if not smff:
            raise ManifestError(
                f"{where}: the smiles route needs parameterization.small_molecule_forcefield"
            )
        if prot:
            raise ManifestError(
                f"{where}: a ligand (smiles) route may not load a protein force field "
                f"({prot!r}). Declare route: pdb if this is a peptide calculation."
            )
        if not par["charge_method"]:
            raise ManifestError(f"{where}: the smiles route needs parameterization.charge_method")
    else:
        if not prot:
            raise ManifestError(
                f"{where}: the pdb route needs parameterization.protein_forcefield"
            )
        if smff:
            raise ManifestError(
                f"{where}: a peptide (pdb) route may not also name a small-molecule force field "
                f"({smff!r}); that is how a peptide silently becomes a Sage ligand run."
            )

    # ---- the input itself ---------------------------------------------------------------------
    if route == "pdb":
        rel = _require(inp, "pdb", f"{where}:input")
        if Path(str(rel)).is_absolute():
            raise ManifestError(
                f"{where}: input.pdb must be relative to the manifest, not an absolute path "
                f"({rel!r}); absolute paths do not survive being copied to another machine"
            )
        pdb_path = (Path(source).parent / str(rel)).resolve()
        if not pdb_path.is_file():
            raise ManifestError(f"{where}: input.pdb does not exist: {pdb_path}")
        declared = str(_require(inp, "pdb_sha256", f"{where}:input"))
        actual = sha256_file(pdb_path)
        if declared != actual:
            raise ManifestError(
                f"{where}: input.pdb_sha256 does not match the file.\n"
                f"  declared {declared}\n  actual   {actual}\n"
                f"  file     {pdb_path}"
            )
        return

    # ---- smiles route -------------------------------------------------------------------------
    smiles = str(_require(inp, "smiles", f"{where}:input"))
    declared_canonical = str(_require(inp, "canonical_isomeric_smiles", f"{where}:input"))
    declared_hash = str(_require(inp, "canonical_smiles_sha256", f"{where}:input"))
    if "expected_formal_charge" not in inp:
        raise ManifestError(f"{where}:input: required field 'expected_formal_charge' is missing")
    expected_charge = int(inp["expected_formal_charge"])

    # the hash must match the string the manifest itself declares, with or without RDKit --
    # this catches an edited SMILES even in an environment that cannot parse chemistry
    hash_of_declared = sha256_text(declared_canonical)
    if hash_of_declared != declared_hash:
        raise ManifestError(
            f"{where}: canonical_smiles_sha256 does not hash canonical_isomeric_smiles.\n"
            f"  declared hash {declared_hash}\n  actual   hash {hash_of_declared}"
        )

    if system_id == RGD_SYSTEM_ID:
        # The name is not a label here, it is a claim about which molecule this is.
        if declared_hash != RGD_CANONICAL_SMILES_SHA256:
            raise ManifestError(
                f"{where}: system_id '{RGD_SYSTEM_ID}' is reserved for the project's "
                "cyclo-(RGDfV) zwitterion and does not match this molecule.\n"
                f"  expected canonical hash {RGD_CANONICAL_SMILES_SHA256}\n"
                f"  this manifest           {declared_hash}\n"
                f"  expected canonical SMILES {RGD_CANONICAL_SMILES}"
            )
        if expected_charge != RGD_EXPECTED_FORMAL_CHARGE:
            raise ManifestError(
                f"{where}: cyclo_rgdfv's formal charge is exactly "
                f"{RGD_EXPECTED_FORMAL_CHARGE} (the zwitterion), not {expected_charge}"
            )

    if not check_chemistry:
        return

    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError as exc:                    # pragma: no cover - env-dependent
        raise ManifestError(
            f"{where}: RDKit is required to validate a smiles-route manifest "
            "(run `escort-explicit validate-env`)"
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ManifestError(f"{where}: input.smiles does not parse: {smiles!r}")

    actual_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    if actual_canonical != declared_canonical:
        raise ManifestError(
            f"{where}: canonical_isomeric_smiles disagrees with RDKit.\n"
            f"  declared {declared_canonical}\n  RDKit    {actual_canonical}"
        )

    unassigned = [
        idx for idx, tag in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        ) if tag == "?"
    ]
    if unassigned:
        raise ManifestError(
            f"{where}: stereochemistry must be explicit; unassigned stereocentre(s) at atom "
            f"index {unassigned}. An unspecified centre makes the parameterisation "
            "non-reproducible."
        )

    actual_charge = Chem.GetFormalCharge(mol)
    if actual_charge != expected_charge:
        raise ManifestError(
            f"{where}: expected_formal_charge is {expected_charge} but the SMILES carries "
            f"{actual_charge}. A wrong net charge changes the ion count and the electrostatics."
        )


# ---------------------------------------------------------------------------------------------
# experiment manifest
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentManifest:
    doc: dict
    source: Path

    @property
    def experiment_id(self) -> str:
        return str(self.doc["experiment_id"])

    @property
    def scale_factors(self) -> list[float]:
        return [float(v) for v in self.doc["rest2"]["scale_factors"]]

    @property
    def n_rungs(self) -> int:
        return len(self.scale_factors)

    @property
    def master_seed(self) -> int:
        return int(self.doc["master_seed"])

    @property
    def ladder_status(self) -> str:
        """``validated`` | ``unvalidated``. Absent means unvalidated — silence is not evidence."""
        return str(self.doc.get("ladder_status", "unvalidated"))


def load_experiment(path: Path) -> ExperimentManifest:
    path = Path(path).resolve()
    doc = _load_yaml(path)
    validate_experiment(doc, source=path)
    return ExperimentManifest(doc=doc, source=path)


def validate_experiment(doc: dict, *, source: Path) -> None:
    where = str(source)
    unknown = sorted(set(doc) - EXPERIMENT_KEYS)
    if unknown:
        raise ManifestError(
            f"{where}: unknown top-level key(s) {unknown}; allowed: {sorted(EXPERIMENT_KEYS)}"
        )
    version = _require(doc, "schema_version", where)
    if int(version) != SCHEMA_VERSION:
        raise ManifestError(f"{where}: schema_version {version} is not supported")
    _require(doc, "experiment_id", where)
    seed = _require(doc, "master_seed", where)
    if not isinstance(seed, int):
        raise ManifestError(f"{where}: master_seed must be an integer, got {seed!r}")

    integ = _require(doc, "integrator", where)
    for key in ("kind", "temperature_k", "timestep_fs"):
        _require(integ, key, f"{where}:integrator")
    if float(integ["temperature_k"]) <= 0:
        raise ManifestError(f"{where}: integrator.temperature_k must be positive")
    if float(integ["timestep_fs"]) <= 0:
        raise ManifestError(f"{where}: integrator.timestep_fs must be positive")

    rest2 = _require(doc, "rest2", where)
    scales = _require(rest2, "scale_factors", f"{where}:rest2")
    if not isinstance(scales, (list, tuple)) or len(scales) < 2:
        raise ManifestError(f"{where}:rest2.scale_factors needs at least two rungs")
    vals = [float(v) for v in scales]
    if abs(vals[0] - 1.0) > 1e-9:
        raise ManifestError(
            f"{where}:rest2.scale_factors must start at the cold rung s=1.0, got {vals[0]}"
        )
    if any(b >= a for a, b in zip(vals, vals[1:])):
        raise ManifestError(f"{where}:rest2.scale_factors must be strictly descending: {vals}")
    if not all(0.0 < v <= 1.0 for v in vals):
        raise ManifestError(f"{where}:rest2.scale_factors must all lie in (0, 1]")
    for key in ("exchange_interval_ps", "relaxation_ps", "total_ns_per_replica", "chunk_ns"):
        v = float(_require(rest2, key, f"{where}:rest2"))
        if v <= 0:
            raise ManifestError(f"{where}:rest2.{key} must be positive, got {v}")
    if float(rest2["chunk_ns"]) > float(rest2["total_ns_per_replica"]) + 1e-12:
        raise ManifestError(
            f"{where}:rest2.chunk_ns ({rest2['chunk_ns']}) exceeds total_ns_per_replica "
            f"({rest2['total_ns_per_replica']})"
        )

    plat = _require(doc, "platform", where)
    precision = str(_require(plat, "precision", f"{where}:platform"))
    if precision not in PRECISIONS:
        raise ManifestError(f"{where}:platform.precision {precision!r} must be one of {PRECISIONS}")

    for optional in ("equilibration", "overrides"):
        if optional in doc and not isinstance(doc[optional], dict):
            raise ManifestError(f"{where}: '{optional}' must be a mapping")

    status = str(doc.get("ladder_status", "unvalidated"))
    if status not in ("validated", "unvalidated"):
        raise ManifestError(
            f"{where}: ladder_status {status!r} must be 'validated' or 'unvalidated'"
        )


# ---------------------------------------------------------------------------------------------
# shipped manifests
# ---------------------------------------------------------------------------------------------

def manifests_dir() -> Path:
    """Directory of the manifests shipped inside the installed package.

    NOT named `data/`: this repository gitignores `data/` at ANY depth, so a package directory by
    that name is silently excluded from version control. The manifests would then be absent from a
    fresh clone and the wheel built from it would ship no shipped manifests at all — while working
    perfectly on the machine that wrote them.
    """
    return Path(__file__).resolve().parent / "manifests"


def shipped_system(name: str) -> Path:
    return manifests_dir() / "systems" / f"{name}.yaml"


def shipped_experiment(name: str) -> Path:
    return manifests_dir() / "experiments" / f"{name}.yaml"


def list_shipped() -> dict[str, list[str]]:
    return {
        "systems": sorted(p.stem for p in (manifests_dir() / "systems").glob("*.yaml")),
        "experiments": sorted(p.stem for p in (manifests_dir() / "experiments").glob("*.yaml")),
    }
