"""Which small-molecule force field a ligand is parameterised with, resolved to an exact version.

Two families are supported:

* **OpenFF Sage** (the default) through ``SMIRNOFFTemplateGenerator``;
* **GAFF** through ``GAFFTemplateGenerator``.

The point of this module is that a *label* is not a force field. ``sage-2.2.1`` and ``gaff2`` are
things a person writes; ``openff-2.2.1`` and ``gaff-2.2.20`` are things that get loaded, and only
the second kind may be recorded. A record saying "GAFF2" does not identify a Hamiltonian: GAFF2 has
been distributed as 2.1, 2.11 and 2.2.20, they differ, and a trajectory belongs to exactly one of
them.

So an alias is accepted at the boundary and resolved immediately, and every record downstream
carries the resolved version. An alias never reaches a record.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "LigandForcefieldError",
    "installed_gaff_versions",
    "is_gaff",
    "resolve_ligand_forcefield",
    "gaff_provenance",
]

#: Aliases a user may write for "the current GAFF 2". They are resolved to the newest installed
#: 2.x at build time, which is why the resolved value -- never the alias -- is what gets recorded.
_GAFF2_ALIASES = ("gaff2", "gaff-2", "gaff 2")


class LigandForcefieldError(ValueError):
    """A ligand force field that cannot be loaded, with the installed alternatives named."""


def installed_gaff_versions() -> tuple[str, ...]:
    """The GAFF force fields ``openmmforcefields`` actually ships in this environment.

    Read from the generator rather than hard-coded: the list depends on the installed version of
    `openmmforcefields`, and a hard-coded copy would eventually promise a file that is not there.
    """
    from openmmforcefields.generators import GAFFTemplateGenerator

    return tuple(GAFFTemplateGenerator.INSTALLED_FORCEFIELDS)


def is_gaff(resource: str | None) -> bool:
    """Whether a RESOLVED resource name names the GAFF family."""
    return bool(resource) and str(resource).lower().startswith("gaff")


def _newest_gaff2() -> str:
    """The newest installed GAFF 2.x, in the order `openmmforcefields` declares.

    NOT sorted by parsing the version. GAFF's numbering mixes conventions -- the installed list is
    `1.4, 1.8, 1.81, 2.1, 2.11, 2.2.20` -- so `1.81` is newer than `1.8` while `2.2.20` is newer
    than `2.11`. There is no single arithmetic that orders both: read as decimals `2.11 > 2.2`,
    read component-wise `(1, 81) > (1, 8)` but `(2, 11) > (2, 2, 20)`. The first version of this
    function compared component-wise and resolved `gaff2` to `gaff-2.11`, which is the wrong file.

    `INSTALLED_FORCEFIELDS` is maintained upstream in ascending order, so the last GAFF 2.x entry
    is the newest. That is a fact about the package rather than a guess about the numbering, and
    if upstream ever stops maintaining the order the pinned test on this function fails.
    """
    candidates = [name for name in installed_gaff_versions()
                  if name.lower().startswith("gaff-2")]
    if not candidates:
        raise LigandForcefieldError(
            "no GAFF 2.x force field is installed. `openmmforcefields` provides "
            f"{', '.join(installed_gaff_versions()) or '(none)'}.")
    return candidates[-1]


def resolve_ligand_forcefield(name: str | None) -> str | None:
    """Turn what a user wrote into the exact resource that will be loaded.

    ``sage-2.2.1`` -> ``openff-2.2.1``   (the installed `openforcefields` file name)
    ``gaff2``      -> ``gaff-2.2.20``    (the newest installed GAFF 2.x, resolved here)
    ``gaff-2.11``  -> ``gaff-2.11``      (an exact request, checked against what is installed)

    A GAFF version that is not installed is refused HERE, naming the installed list, rather than
    at the point antechamber is invoked -- by then a solvated box may already have been built.
    """
    if not name:
        return None
    text = str(name).strip().lower()

    if text.startswith("sage-"):
        return "openff-" + text[len("sage-"):]

    if text in _GAFF2_ALIASES:
        return _newest_gaff2()

    if text.startswith("gaff"):
        installed = installed_gaff_versions()
        if text in installed:
            return text
        raise LigandForcefieldError(
            f"solute.ligand_forcefield {name!r} is not installed. `openmmforcefields` provides "
            f"{', '.join(installed)}. Write one of those exactly, or the alias 'gaff2' for the "
            f"newest installed GAFF 2.x ({_newest_gaff2()}); an ambiguous label is not recorded "
            f"because GAFF2 has been distributed as several different parameter sets.")

    # Anything else is passed through: SMIRNOFFTemplateGenerator resolves openff-* names itself
    # and raises at the point the parameters would have been assigned.
    return text


def _ambertools_version(executable_path: str | None) -> dict[str, Any]:
    """The AmberTools build that supplies `antechamber` and `sqm`.

    Read from the conda package record beside the executable, not from the program's own output:
    antechamber has no version flag, and its usage banner merely mentions "gaff2 (beta-version)".
    An earlier version of this function scraped that line and recorded it as the AmberTools
    version, ANSI escapes and all -- a record that looks precise and says nothing.

    `null` when the executable did not come from a conda environment. That is honest: this is
    provenance, and "I could not establish it" is information a reader can act on, whereas a
    scraped string is not.
    """
    if not executable_path:
        return {"source": None, "version": None}
    meta = Path(executable_path).resolve().parent.parent / "conda-meta"
    if meta.is_dir():
        for record in sorted(meta.glob("ambertools-*.json")):
            # `ambertools-26.0-cuda_None_nompi_py312hd652fd9_100.json`
            parts = record.stem.split("-")
            if len(parts) >= 2:
                return {"source": "conda-meta", "version": parts[1],
                        "package": record.stem}
    return {"source": None, "version": None}


def gaff_provenance(resolved: str) -> dict[str, Any]:
    """What identifies a GAFF parameterisation beyond the version string.

    GAFF types the molecule with antechamber [Wang 2006] and charges it through `sqm`
    [Jakalian 2002], so the AmberTools build in the environment is part of this Hamiltonian's
    provenance in a way it is not for a SMIRNOFF force field, where the offxml file is the whole
    specification.

    Nothing here raises. Failing to describe a build must not fail a build that succeeded.
    """
    import shutil

    antechamber = shutil.which("antechamber")
    return {
        "family": "gaff",
        "typing": "antechamber",
        "installed_versions": list(installed_gaff_versions()),
        "antechamber": antechamber,
        "sqm": shutil.which("sqm"),
        "ambertools": _ambertools_version(antechamber),
    }
