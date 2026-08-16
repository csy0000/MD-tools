"""Migration from the currently shipped system + experiment manifests into the canonical model.

The old format split one calculation across two documents with unitless numbers whose units lived
in the field names (`timestep_fs`, `padding_nm`, `chunk_ns`). The canonical model carries explicit
quantities instead, so migration attaches the unit the old NAME declared -- never a guessed one.
`timestep_fs: 4.0` becomes `"4 fs"`, which is the same physical value, and that equivalence is what
makes the migration reportable rather than a reinterpretation.

Every semantic change is returned alongside the result. Nothing is written in place unless the
caller asks for it.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["migrate_manifests", "MigrationError"]


class MigrationError(ValueError):
    """An old document cannot be expressed in the canonical model without a decision."""


def _q(value: Any, unit: str) -> Optional[str]:
    """Attach the unit the old FIELD NAME declared. `None` stays `None`."""
    return None if value is None else f"{value} {unit}"


def migrate_manifests(system_doc: dict, experiment_doc: dict) -> tuple[dict, list[str]]:
    """Return `(canonical_document, notes)`.

    `notes` lists every semantic change, including the ones that are merely a renaming, so a
    reviewer can see what the migration decided rather than diffing two formats by eye.
    """
    notes: list[str] = []
    inp = system_doc.get("input") or {}
    par = system_doc.get("parameterization") or {}
    sol = system_doc.get("solvation") or {}
    route = inp.get("route")
    if route not in ("smiles", "pdb"):
        raise MigrationError(
            f"system.input.route is {route!r}; the canonical model requires a declared route of "
            "'smiles' or 'pdb' and never infers it."
        )

    system = {
        "schema_version": 1,
        "system_id": system_doc.get("system_id"),
        "display_name": system_doc.get("display_name"),
        "route": route,
        "expected_formal_charge": inp.get("expected_formal_charge", 0),
    }
    if route == "smiles":
        system["smiles"] = inp.get("smiles")
        system["canonical_isomeric_smiles"] = inp.get("canonical_isomeric_smiles")
        system["canonical_smiles_sha256"] = inp.get("canonical_smiles_sha256")
    else:
        system["pdb"] = inp.get("pdb")
        system["pdb_sha256"] = inp.get("pdb_sha256")

    build = {
        "schema_version": 1,
        "forcefield": {
            "small_molecule": par.get("small_molecule_forcefield"),
            "charge_method": par.get("charge_method"),
            "protein": par.get("protein_forcefield"),
            "water": par.get("water_forcefield"),
        },
        "solvation": {
            "water_model": sol.get("water_model"),
            "box_shape": sol.get("box_shape"),
            "padding": _q(sol.get("padding_nm"), "nm"),
            "ionic_strength_molar": sol.get("ionic_strength_molar", 0.0),
            "positive_ion": sol.get("positive_ion", "Na+"),
            "negative_ion": sol.get("negative_ion", "Cl-"),
        },
    }
    notes.append("solvation.padding_nm -> build.solvation.padding, as an explicit 'nm' quantity")

    integ = experiment_doc.get("integrator") or {}
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "integrator": {
            "kind": integ.get("kind", "langevin-middle"),
            "timestep": _q(integ.get("timestep_fs"), "fs"),
            "temperature": _q(integ.get("temperature_k"), "K"),
        },
    }
    notes.append("integrator.timestep_fs -> protocol.integrator.timestep, as an explicit 'fs' "
                 "quantity (same physical value)")
    notes.append("integrator.temperature_k -> protocol.integrator.temperature ('K')")

    rest2 = experiment_doc.get("rest2") or {}
    md = experiment_doc.get("md") or {}
    if md:
        protocol["production"] = {
            "method": "md",
            "n_chunks": md.get("n_chunks"),
            "chunk": _q(md.get("chunk_ns"), "ns"),
            "scale_factor": md.get("scale_factor", 1.0),
        }
        notes.append("md.chunk_ns -> protocol.production.chunk ('ns')")
    elif rest2:
        if "total_ns_per_replica" in rest2:
            raise MigrationError(
                "rest2.total_ns_per_replica belongs to a retired schema: the chunk count is an "
                "input, not a rounded quotient. Convert it to n_chunks and chunk_ns first."
            )
        protocol["production"] = {
            "method": "rest2",
            "n_chunks": rest2.get("n_chunks"),
            "chunk": _q(rest2.get("chunk_ns"), "ns"),
            "scale_factors": rest2.get("scale_factors"),
            "exchange_interval": _q(rest2.get("exchange_interval_ps"), "ps"),
            "relaxation": _q(rest2.get("relaxation_ps"), "ps"),
        }
        notes.append("rest2.chunk_ns / exchange_interval_ps / relaxation_ps -> explicit quantities")
    else:
        raise MigrationError(
            "the experiment declares neither an `md:` nor a `rest2:` block, so the method is "
            "undetermined. The canonical model requires protocol.production.method."
        )

    platform = experiment_doc.get("platform") or {}
    execution = {"schema_version": 1, "precision": platform.get("precision", "mixed")}
    notes.append("platform.precision -> execution.precision (execution-only; not in any hash)")

    if experiment_doc.get("ladder_status"):
        notes.append(
            f"ladder_status {experiment_doc['ladder_status']!r} is NOT carried into the canonical "
            "model: it is a claim about evidence, not a simulation setting. Record it in "
            "documentation or bundle provenance instead."
        )
    if experiment_doc.get("master_seed") is not None:
        notes.append(
            f"master_seed {experiment_doc['master_seed']} is not copied automatically; set "
            "protocol.production.seed explicitly if the run must reproduce a previous stream."
        )

    document = {"system": system, "build": build, "protocol": protocol, "execution": execution}
    return document, notes
