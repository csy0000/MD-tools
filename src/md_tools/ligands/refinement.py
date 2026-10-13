"""A bespoke-torsion refinement of a small molecule's force field, as a record a package carries.

BespokeFit refits a molecule's torsions against a quantum or machine-learned reference and writes
one `.offxml`: the base force field with new proper-torsion parameters appended. This module takes
that file, checks that it is what it claims to be, and describes it. It runs no fit and imports
nothing from BespokeFit, QCEngine, Psi4 or MACE: those live in an environment the user installs
and names (`solute.bespokefit_env`, `solute.mace_env`), never in this package's.

What a package records is the METHOD and the file's sha256, because those decide whether two
packages are the same Hamiltonian; the base force field and the new parameter ids are recorded so a
person can read what was changed.

The file is verified, not trusted. It must parse, it must hold every proper torsion of the base
force field UNCHANGED (so it really is the base plus additions, not a different force field wearing
its name), and it must add at least one: a "refinement" that adds nothing is a stock package under
another name, and the one thing that must not happen is two packages that differ in a label but not
in a parameter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "REFINEMENT_METHODS",
    "RefinementError",
    "package_refinement",
    "refinement_identity",
    "refinement_record",
    "resolve_refinement",
]

#: What `solute.torsion_refinement` may name besides `none`, and the reference each one fits to.
REFINEMENT_METHODS = {
    "bespokefit": {"program": "psi4", "model": "B3LYP-D3BJ/DZVP"},
    "mace-omol-bespokefit": {"program": "mace-omol", "model": "MACE-OMOL-0 extra_large"},
    "mace-off23-bespokefit": {"program": "mace", "model": "MACE-OFF23"},
}


class RefinementError(ValueError):
    """A bespoke force field that cannot be used, with the reason and the way out."""


def _base_offxml(base_forcefield: str) -> str:
    """BespokeFit fits against the UNCONSTRAINED variant of a Sage release.

    `openff-2.2.1` is the constrained file; its parameters are the unconstrained file's, plus a
    Constraints handler, and h-bond constraints are applied by the System anyway. So the unconstrained
    file is what a fitted `.offxml` is compared to.
    """
    text = str(base_forcefield)
    if text.startswith("openff-") and not text.startswith("openff_unconstrained-"):
        return "openff_unconstrained-" + text[len("openff-"):] + ".offxml"
    if text.endswith(".offxml"):
        return text
    return text + ".offxml"


def _torsion_key(parameter) -> tuple:
    """A proper-torsion parameter as comparable text: SMIRKS plus every physical attribute."""
    fields = {k: str(v) for k, v in parameter.to_dict().items()
              if k not in ("id", "parameterize") and not k.startswith("_")}
    return tuple(sorted(fields.items()))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def refinement_record(method: str, offxml: str | Path, *, base_forcefield: str,
                      base_dir: Optional[Path] = None) -> dict[str, Any]:
    """Verify a fitted `.offxml` and describe it. Raises `RefinementError`; never returns a guess.

    The returned record carries `offxml_path` (absolute, for the generator that reads it) beside the
    fields a package stores; `package_refinement` drops the path.
    """
    if method not in REFINEMENT_METHODS:
        raise RefinementError(f"torsion_refinement {method!r} is not one of "
                              f"{', '.join(REFINEMENT_METHODS)}")
    path = Path(offxml)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise RefinementError(f"solute.refinement_offxml {str(offxml)!r} does not exist"
                              f"{f' (looked for {path})' if str(path) != str(offxml) else ''}. "
                              f"A fit that has not run yet cannot be used: run BespokeFit in the "
                              f"environment you installed, then name its `.offxml` here.")
    from openff.toolkit import ForceField

    try:
        fitted = ForceField(str(path), allow_cosmetic_attributes=True)
    except Exception as exc:                       # noqa: BLE001 -- the toolkit raises many types
        raise RefinementError(f"{path} is not a force field the toolkit can read: {exc}") from exc
    base_name = _base_offxml(base_forcefield)
    try:
        base = ForceField(base_name)
    except Exception as exc:                       # noqa: BLE001
        raise RefinementError(f"the base force field {base_forcefield!r} ({base_name}) cannot be "
                              f"loaded to check {path.name} against it: {exc}") from exc
    if "ProperTorsions" not in fitted.registered_parameter_handlers:
        raise RefinementError(f"{path.name} has no ProperTorsions section, so it refines nothing")

    have = {}
    for parameter in fitted.get_parameter_handler("ProperTorsions").parameters:
        have[str(parameter.id)] = _torsion_key(parameter)
    base_torsions = base.get_parameter_handler("ProperTorsions").parameters
    changed = [str(p.id) for p in base_torsions
               if have.get(str(p.id)) != _torsion_key(p)]
    if changed:
        raise RefinementError(
            f"{path.name} does not contain {base_name} unchanged: proper torsion(s) "
            f"{', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''} are missing or differ. "
            f"A refinement ADDS torsions to the base force field named by "
            f"solute.ligand_forcefield ({base_forcefield}); check that the fit started from it.")
    base_ids = {str(p.id) for p in base_torsions}
    added = [i for i in have if i not in base_ids]
    if not added:
        raise RefinementError(f"{path.name} adds no proper torsion to {base_name}: it is the stock "
                              f"force field, and a package built from it would be a stock package "
                              f"under another name. Use torsion_refinement: none.")
    return {
        "method": method,
        "reference": dict(REFINEMENT_METHODS[method]),
        "source": "supplied .offxml (the fit was not run by this build)",
        "base_forcefield": str(base_forcefield),
        "base_file": base_name,
        "offxml": path.name,
        "offxml_sha256": sha256_of(path),
        "added_proper_torsion_ids": added,
        "offxml_path": str(path),
    }


def resolve_refinement(solute: dict[str, Any], *, base_forcefield: str,
                       config_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The refinement a resolved `solute` block asks for, verified; None for `none`.

    Today the fit is supplied, not run: a method with no `refinement_offxml` is refused HERE, before
    any charge is generated, saying so. Running the fit is a separate, hours-long, GPU job that this
    build does not start, and quietly parameterising with the stock force field instead would be the
    substitution the option exists to avoid.
    """
    method = str(solute.get("torsion_refinement") or "none")
    if method == "none":
        return None
    offxml = solute.get("refinement_offxml")
    if not offxml:
        raise RefinementError(
            f"solute.torsion_refinement is {method!r} but no fitted force field was given. This "
            f"build does not run the fit: run BespokeFit in your own environment and name its "
            f"`.offxml` in solute.refinement_offxml.")
    return refinement_record(method, offxml, base_forcefield=base_forcefield, base_dir=config_dir)


def package_refinement(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """What a package stores: the record without the path of a file it does not keep."""
    if record is None:
        return None
    return {k: v for k, v in record.items() if k != "offxml_path"}


def refinement_identity(record: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    """What decides whether two packages are the same Hamiltonian: the method and the file's hash."""
    if not record:
        return None
    return {"method": str(record["method"]), "offxml_sha256": str(record["offxml_sha256"])}
