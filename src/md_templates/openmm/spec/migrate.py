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



def _tau_ladder_from_scale_factors(scale_factors) -> tuple[dict, str]:
    """Express an explicit `s` ladder as a tau ladder, or refuse.

    tau = 1 - sqrt(s). A ladder that is not linear in tau cannot be written in the new form
    without changing it, so it is refused rather than approximated -- a silently respaced ladder
    changes exchange acceptance and therefore the run.
    """
    import math

    from ..tau import build_tau_ladder

    if not scale_factors:
        raise MigrationError(
            "rest2.scale_factors is missing; the ladder cannot be migrated without it."
        )
    taus = [1.0 - math.sqrt(float(s)) for s in scale_factors]
    minimum, maximum, count = round(taus[0], 12), round(taus[-1], 12), len(taus)
    rebuilt = build_tau_ladder(minimum, maximum, count)
    worst = max(abs(a - b) for a, b in zip(taus, rebuilt))
    if worst > 1e-9:
        raise MigrationError(
            "rest2.scale_factors is not linear in tau (worst deviation "
            f"{worst:.2e}), so it cannot be expressed as a tau ladder without respacing it. "
            "Respacing changes exchange acceptance and therefore the run. Convert the ladder "
            "deliberately, or keep the old ladder by naming each rung."
        )
    return ({"minimum": minimum, "maximum": maximum, "count": count, "interpolation": "linear"},
            f"rest2.scale_factors -> tau_ladder[{minimum}, {maximum}] x {count} "
            f"(s = (1 - tau)^2 reproduces the old ladder to {worst:.1e})")


def _exchanges_per_segment(chunk_ns, interval_ps) -> int:
    """How many exchange rounds the old chunk contained, as an exact integer."""
    if chunk_ns is None or interval_ps is None:
        raise MigrationError(
            "rest2.chunk_ns and rest2.exchange_interval_ps are both required to derive the "
            "exchange count per segment."
        )
    count = (float(chunk_ns) * 1000.0) / float(interval_ps)
    if abs(count - round(count)) > 1e-9:
        raise MigrationError(
            f"rest2.chunk_ns ({chunk_ns} ns) is not a whole number of exchange intervals "
            f"({interval_ps} ps): {count:.6f}. The old configuration was already inconsistent; "
            "fix it before migrating."
        )
    return int(round(count))


class MigrationError(ValueError):
    """An old document cannot be expressed in the canonical model without a decision."""


def _q(value: Any, unit: str) -> Optional[str]:
    """Attach the unit the old FIELD NAME declared. `None` stays `None`."""
    return None if value is None else f"{value} {unit}"


def migrate_manifests(system_doc: dict, experiment_doc: dict, *,
                      method: Optional[str] = None) -> tuple[dict, list[str]]:
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
    if md and rest2 and method is None:
        # A legacy experiment may declare BOTH, because the pair described a system that could be
        # run either way. The canonical model has ONE method, so choosing by if/elif order would
        # silently label a REST2 experiment "md" -- which it did, and a legacy REST2 bundle then
        # refused to run. Ambiguity is reported, not resolved by statement order.
        raise MigrationError(
            "this experiment declares BOTH an `md:` and a `rest2:` block, so the canonical method "
            "is ambiguous. Pass method='md' or method='rest2' to say which calculation this is; "
            "it is not inferred from the order the blocks appear in."
        )
    if method == "md" or (md and not rest2):
        protocol["production"] = {
            "method": "md",
            "duration_per_segment": _q(md.get("chunk_ns"), "ns"),
            "scale_factor": md.get("scale_factor", 1.0),
        }
        notes.append("md.chunk_ns -> protocol.production.duration_per_segment ('ns')")
        if md.get("n_chunks") is not None:
            notes.append(
                f"md.n_chunks ({md['n_chunks']}) was DROPPED: segment count is an execution "
                "choice, not a scientific input. Request that many segments from the driver "
                "script instead; the run manifest records how many committed."
            )
    elif rest2 or method == "rest2":
        if "total_ns_per_replica" in rest2:
            raise MigrationError(
                "rest2.total_ns_per_replica belongs to a retired schema: the chunk count is an "
                "input, not a rounded quotient. Convert it to n_chunks and chunk_ns first."
            )
        chunk_ns = rest2.get("chunk_ns")
        interval_ps = rest2.get("exchange_interval_ps")
        tau_ladder, ladder_note = _tau_ladder_from_scale_factors(rest2.get("scale_factors"))
        exchanges = _exchanges_per_segment(chunk_ns, interval_ps)
        protocol["production"] = {
            "method": "rest2",
            "tau_ladder": tau_ladder,
            "duration_per_segment": _q(chunk_ns, "ns"),
            "exchange": {"number_of_exchanges_per_segment": exchanges},
            "relaxation": _q(rest2.get("relaxation_ps"), "ps"),
        }
        notes.append(
            f"rest2.chunk_ns ({chunk_ns} ns) -> production.duration_per_segment, the length of ONE "
            "segment; how many segments to run stays an execution choice"
        )
        notes.append(ladder_note)
        notes.append(
            f"rest2.exchange_interval_ps ({interval_ps}) is no longer stated: the interval is "
            f"DERIVED as duration_per_segment / number_of_exchanges_per_segment = {chunk_ns} ns / "
            f"{exchanges}, which is the same {interval_ps} ps. Both divisions are exact in step "
            "space and are refused rather than rounded"
        )
        if rest2.get("n_chunks") is not None:
            notes.append(
                f"rest2.n_chunks ({rest2['n_chunks']}) was DROPPED: segment count is an execution "
                "choice, not a scientific input."
            )
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
    randomness = {"schema_version": 1}
    if experiment_doc.get("master_seed") is not None:
        randomness["master_seed"] = int(experiment_doc["master_seed"])
        notes.append(
            f"master_seed {experiment_doc['master_seed']} -> randomness.master_seed. Stage seeds "
            "derive from it by the unchanged rule (master + structure/equilibration/md/rest2 "
            "offset 0..3), so migrated inputs reproduce their existing trajectories."
        )

    document = {"system": system, "build": build, "protocol": protocol,
                "execution": execution, "randomness": randomness}
    return document, notes
