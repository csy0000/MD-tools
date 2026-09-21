"""One LEG of a thermodynamic cycle: a plan, its Hamiltonian, its windows, and its free energy.

    prepare_leg   plan + Hamiltonian -> a leg directory: leg.json, system.xml, topology.pdb
    run_leg       every window of the leg (or the ones named), through `windows.run_window`
    analyze_leg   the windows' sample streams -> `estimators.analyze` -> a `cycles.Leg`

A leg directory is the unit a window runs against: `system.xml` is `hamiltonian.system` exactly
(`run_window` compares the serialisations before writing anything), `topology.pdb` is the plan's
combined topology, and `leg.json` records the plan digest, the path, the states and how the
Hamiltonian was built. The window outputs sit in `windows/`.

The Hamiltonian is REBUILT from the plan by the caller on every invocation (`from_plan`), never
unpickled from a previous process: a System and its Context parameters are what reproduce the
energies (shared contracts section 5), and the window fingerprint binds every state's Context
parameters, so a Hamiltonian that rebuilt differently could not continue an old window.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from md_tools.alchemy.cycles import Leg
from md_tools.alchemy.paths import AlchemicalPath
from md_tools.alchemy.samples import ThermodynamicState, concatenate, window_states
from md_tools.alchemy.windows import (WindowError, WindowSettings, read_window_samples,
                                      run_window, window_paths)

LEG_SCHEMA = "md-tools-alchemical-leg/1"


def prepare_leg(directory, *, plan, hamiltonian, path: AlchemicalPath, s_values: Sequence[float],
                temperature_k: float, pressure_bar: float | None, environment: str,
                endpoint_a: str, endpoint_b: str, scheme: str) -> dict[str, Any]:
    """Write a leg directory. Refuses an existing directory rather than mixing two legs."""
    import openmm

    directory = Path(directory)
    if directory.exists():
        raise WindowError(f"{directory} exists; a leg is prepared once, into a new directory")
    states = window_states(path, s_values, temperature_k=temperature_k, pressure_bar=pressure_bar)
    solvation = plan.record.get("environment", {}).get("solvation")
    if solvation is None:
        raise WindowError("the plan records no environment solvation (S2's "
                          "plan.record['environment']['solvation']); a window cannot know whether "
                          "it is a vacuum leg, and must not guess from periodicity")
    directory.mkdir(parents=True)
    system_xml = openmm.XmlSerializer.serialize(hamiltonian.system)
    (directory / "system.xml").write_text(system_xml)
    pdb = plan.pdb_text
    if pdb is None:
        import io
        buf = io.StringIO()
        openmm.app.PDBFile.writeFile(plan.topology, plan.positions_nm * 10.0, buf, keepIds=True)
        pdb = buf.getvalue()
    (directory / "topology.pdb").write_text(pdb)
    # The whole plan record, so the cycle can run S2's `matched_legs` on the two legs later
    # without rebuilding either plan.
    (directory / "plan.json").write_text(json.dumps(plan.record, indent=1, sort_keys=True) + "\n")
    record = {
        "schema": LEG_SCHEMA, "environment": environment, "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b, "alchemical_scheme": scheme,
        "plan_sha256": plan.sha256, "plan_mode": plan.mode,
        # S2's ligand-side Hamiltonian digest; None until S2's plan record carries it, and the
        # relative cycles refuse a leg without it.
        "ligand_hamiltonian_sha256": plan.record.get("ligand_hamiltonian_sha256"),
        "solvation": solvation,
        "hamiltonian": {k: v for k, v in (getattr(hamiltonian, "record", {}) or {}).items()
                        if k in ("schema", "softcore", "settings")},
        "system_sha256": hashlib.sha256(system_xml.encode()).hexdigest(),
        "path": path.to_record(), "states": [s.to_record() for s in states],
    }
    (directory / "leg.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def read_leg(directory) -> tuple[dict[str, Any], AlchemicalPath, tuple[ThermodynamicState, ...]]:
    record = json.loads((Path(directory) / "leg.json").read_text())
    if record.get("schema") != LEG_SCHEMA:
        raise WindowError(f"{directory}/leg.json schema {record.get('schema')!r}")
    return (record, AlchemicalPath.from_record(record["path"]),
            tuple(ThermodynamicState.from_record(s) for s in record["states"]))


def run_leg(directory, *, hamiltonian, settings: WindowSettings, windows: Sequence[str] | None = None,
            repeat: str = "r1", **run_kw) -> list[dict[str, Any]]:
    """Run (continue, or verify) the leg's windows into `windows/<repeat>/`."""
    directory = Path(directory)
    record, path, states = read_leg(directory)
    import openmm
    current = hashlib.sha256(openmm.XmlSerializer.serialize(hamiltonian.system).encode()).hexdigest()
    if current != record["system_sha256"]:
        raise WindowError(f"the Hamiltonian given is not the one {directory} was prepared with "
                          f"(system sha256 {current[:12]}... != {record['system_sha256'][:12]}...)")
    wanted = list(windows) if windows else [s.state_id for s in states]
    out = directory / "windows" / repeat
    return [run_window(topology=directory / "topology.pdb", system=directory / "system.xml",
                       hamiltonian=hamiltonian, path=path, states=states, window_id=wid,
                       out_dir=out, settings=settings, solvation=record["solvation"], **run_kw)
            for wid in wanted]


def analyze_leg(directory, *, repeat: str = "r1", estimator: str = "MBAR",
                name: str | None = None) -> tuple[Leg, dict[str, Any]]:
    """The leg's free energy, from every window's committed and completed stream."""
    from md_tools.alchemy import estimators as est

    directory = Path(directory)
    record, _, states = read_leg(directory)
    out = directory / "windows" / repeat
    parts = []
    for st in states:
        p = window_paths(out, st.state_id)
        if not p["completion"].is_file():
            raise WindowError(f"window {st.state_id} of {directory} ({repeat}) is not complete")
        parts.append(read_window_samples(p["samples"], json.loads(p["record"].read_text())))
    analysis = est.analyze(concatenate(parts))
    leg = Leg.from_estimate(name or f"{record['environment']} {repeat}", record["environment"],
                            analysis["estimates"][estimator], path=record["path"],
                            temperature_k=states[0].temperature_k,
                            alchemical_scheme=record["alchemical_scheme"],
                            ligand_hamiltonian_sha256=record.get("ligand_hamiltonian_sha256"))
    return leg, analysis


def combine_repeats(legs: Sequence[Leg]) -> Leg:
    """Combine independent repeats of ONE leg, and error-bar them by the REPEATS.

    The value is the inverse-variance mean. The uncertainty is the LARGER of

        the inverse-variance uncertainty   -- what the estimators claim, and
        the spread of the repeats          -- std(values, ddof=1) / sqrt(n),

    because on the M2 campaign (2026-09-20, `handoffs/S4.md`) the repeats of one leg differed by
    up to 0.80 kcal/mol while MBAR claimed 0.09: an asymptotic covariance computed where
    neighbouring states barely overlap is not a measurement of the run-to-run spread, and
    publishing it as one understates the error by an order of magnitude. Both numbers are kept in
    the record, with which one was used.

    With a single repeat there is no spread to measure; the estimator's own uncertainty is
    returned and the record says `repeats: 1`, which is not evidence of precision.
    """
    if not legs:
        raise WindowError("no repeats")
    first = legs[0]
    for leg in legs[1:]:
        if (leg.environment, leg.endpoint_a, leg.endpoint_b, leg.alchemical_scheme,
                leg.ligand_hamiltonian_sha256) != \
                (first.environment, first.endpoint_a, first.endpoint_b, first.alchemical_scheme,
                 first.ligand_hamiltonian_sha256):
            raise WindowError("repeats of different legs cannot be combined")
    values = [leg.delta_g_kj_mol for leg in legs]
    w = [1.0 / leg.sigma_kj_mol ** 2 for leg in legs]
    value = sum(wi * v for wi, v in zip(w, values)) / sum(w)
    estimator_sigma = (1.0 / sum(w)) ** 0.5
    spread_sigma = (statistics.stdev(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    sigma = max(estimator_sigma, spread_sigma)
    return Leg(f"{first.environment} ({len(legs)} repeats)", first.environment,
               first.endpoint_a, first.endpoint_b, value, sigma,
               first.temperature_k, first.estimator, first.alchemical_scheme,
               first.restraint_digest,
               {"repeats": [leg.to_record() for leg in legs], "n_repeats": len(legs),
                "estimator_sigma_kJ_mol": estimator_sigma,
                "repeat_spread_sigma_kJ_mol": spread_sigma,
                "sigma_used": "repeat spread" if spread_sigma > estimator_sigma else "estimator",
                "rule": "the larger of the estimator's uncertainty and the repeat spread"},
               first.ligand_hamiltonian_sha256)

def matched_leg_report(first_dir, second_dir) -> dict[str, Any]:
    """S2's `matched_legs` on the plans two leg directories were prepared from.

    Refuses (TopologyError) unless both legs carry one ligand Hamiltonian; also refuses a
    plan.json that is not the plan its leg.json names.
    """
    from types import SimpleNamespace

    from md_tools.alchemy.topology import matched_legs

    plans = []
    for d in (Path(first_dir), Path(second_dir)):
        record, _, _ = read_leg(d)
        plan = json.loads((d / "plan.json").read_text())
        if plan.get("plan_sha256") != record["plan_sha256"]:
            raise WindowError(f"{d}/plan.json is not the plan {d}/leg.json was prepared from")
        plans.append(SimpleNamespace(record=plan, sha256=plan["plan_sha256"]))
    return matched_legs(*plans)


def relative_hydration_from_legs(vacuum_dir, solvent_dir, *, repeats: Sequence[str],
                                 estimator: str = "MBAR") -> dict[str, Any]:
    """ddG_hyd(A->B) from a vacuum and a solvent leg, each over its independent repeats.

    `matched_legs` runs first: two legs that do not share one ligand Hamiltonian are refused
    before any sample is read.
    """
    from md_tools.alchemy.cycles import relative_hydration

    report = matched_leg_report(vacuum_dir, solvent_dir)
    legs, analyses = {}, {}
    for role, d in (("vacuum", vacuum_dir), ("solvent", solvent_dir)):
        per = [analyze_leg(d, repeat=r, estimator=estimator) for r in repeats]
        for (leg, _), r in zip(per, repeats):
            if leg.ligand_hamiltonian_sha256 != report["ligand_hamiltonian_sha256"]:
                raise WindowError(f"{d} ({r}) carries ligand Hamiltonian "
                                  f"{str(leg.ligand_hamiltonian_sha256)[:12]}..., matched_legs "
                                  f"says {report['ligand_hamiltonian_sha256'][:12]}...")
        legs[role] = combine_repeats([leg for leg, _ in per])
        analyses[role] = {r: a for (_, a), r in zip(per, repeats)}
    result = relative_hydration(vacuum=legs["vacuum"], solvent=legs["solvent"])
    result["matched_legs"] = report
    result["repeats"] = list(repeats)
    result["analyses"] = analyses
    return result
