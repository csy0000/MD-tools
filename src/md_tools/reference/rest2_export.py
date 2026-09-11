"""Turn a finished REST2 ladder into a standalone bundle that needs only OpenMM and numpy.

HOW THIS DIFFERS FROM THE cMD EXPORT, AND WHY IT IS SAFER

`reference/export.py` writes the loop out as a template, because a cMD stage's control flow is
about thirty lines and a template is the honest way to say thirty lines. A ladder's is not: N
contexts, an exchange schedule, a reduced-potential matrix, an acceptance rule and the
state-to-walker bookkeeping. Writing that out again would be a second implementation of the
physics, and the cMD export already demonstrated what second implementations do -- it carried the
BUILD System instead of the integrated one, used the config seed instead of the derived one, and
started from the wrong coordinates, all while running perfectly.

So this export does not reimplement anything. The modules that decide what happens --

    core.py        the pV unit and the named-substream seed derivation
    rules.py       the acceptance criterion and the odd/even sweep
    engine.py      the Contexts, propagation, and the reduced potential
    statistics.py  acceptance accounting
    rem_log.py     the Amber-format exchange log

-- are copied VERBATIM into the bundle. They already import nothing from md_tools: after
`BAR_NM3_TO_KJ_PER_MOL` and `stream_seed` moved into `core.py`, every one of them needs only the
standard library, numpy and OpenMM. A test asserts the copies are byte-identical to the package's
own, which makes drift impossible rather than merely detectable.

What is left to write is the wiring, and only the wiring is new code that has to be verified.

WHAT IS DELIBERATELY NOT CARRIED

Resume, extend, MPI coordination, checkpointing, fail-closed identity checks, reservoir refresh:
`driver.py` is 1993 lines and most of them are those. A frozen artefact must not resume -- it has
one job, which is to run the ladder that was run -- and a bundle carrying a checkpoint format
would be claiming compatibility it cannot keep.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .export import _continue_from, _digest, _locate, user_inputs_plan, write_user_inputs

#: Copied byte for byte. Each is dependency-free by construction; see the module docstring.
VENDORED = ("core.py", "rules.py", "engine.py", "statistics.py", "rem_log.py")

#: `md_tools/rest2/hamiltonian.py`, copied byte for byte beside them: the code that turns the
#: unscaled System into a rung. With it, `verify_rungs.py` rebuilds every bundled rung from rung 0
#: using OpenMM alone, so how the scaled Systems were derived is checkable rather than described.
SCALING_MODULE = "hamiltonian.py"

PACKAGE_INIT = '''"""The ladder's own decision-making, copied verbatim from md-tools {version}.

Every module beside this one is a byte-for-byte copy at commit {commit}: `md_tools/remd/<name>`,
and `hamiltonian.py`, which is `md_tools/rest2/hamiltonian.py` -- the code that built the rungs.
None of them imports md_tools; they need the standard library, numpy and OpenMM.

This file is the exception: it is written by the export, because the package's own `__init__`
imports a great deal that a bundle has no use for.
"""
'''

RUNNER = '''#!/usr/bin/env python
"""{title}

Standalone. Needs OpenMM and numpy -- no MD-tools, no AmberTools, no MPI.

    python run.py                      the whole ladder
    python run.py --exchanges 20       a short check
    python run.py --platform CPU

WHAT IS REPRODUCED

    The ensemble, and -- on the same platform with the same OpenMM -- the run itself, exchange
    for exchange. The acceptance rule, the odd/even sweep and the seed derivation in `ladder/`
    are the same bytes md-tools ran, not a reimplementation of them.

    The propagation between exchanges is issued as one `step()` call per segment. The original
    run split those segments further wherever a trajectory or checkpoint event fell, which is
    dynamically neutral -- the same integration steps drawn from the same stream -- and the
    equivalence test in md-tools asserts exactly that against the engine's own output.

HOW THE RUNGS WERE BUILT

    system_rung0.xml is the unscaled System; every other rung is that System with the solute
    scaled at its tau. `ladder/hamiltonian.py` is the code that did it, and `python
    verify_rungs.py` rebuilds each rung from rung 0 with it and checks the result is identical.
    provenance.json's `derivation` block holds the solute atoms and the omega bonds it needs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openmm import XmlSerializer, unit
from openmm.app import PDBFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ladder.core import stream_seed                      # noqa: E402
from ladder.engine import Configuration, ReplicaEngine    # noqa: E402
from ladder.rules import ExchangeContext, NeighbouringExchangeRule  # noqa: E402

SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))
PROVENANCE = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))


class Protocol:
    """Exactly what `ReplicaEngine` and the exchange rule read, and nothing else.

    md-tools' own `Protocol` validates a configuration and builds Systems. Both jobs are already
    done: the Systems are on disk beside this file, built by the engine that ran. Carrying the
    real class would mean carrying `md_tools.rest2` behind it for no behaviour a bundle uses.
    """

    def __init__(self, settings):
        self.tau = [float(t) for t in settings["tau"]]
        self.temperature_k = float(settings["temperature_K"])
        self.timestep_fs = float(settings["timestep_fs"])
        self.friction_per_ps = float(settings["friction_per_ps"])
        self.constraint_tolerance = float(settings["constraint_tolerance"])
        self.random_seed = int(settings["seed"])
        #: NVT by contract. Every rung of a REST2 ladder shares one temperature and one volume;
        #: a barostat would sample a different distribution at every rung but the unscaled one.
        self.pressure_bar = None

    @property
    def n_states(self):
        return len(self.tau)

    @property
    def beta(self):
        # 1 / kT in mol/kJ. MOLAR_GAS_CONSTANT_R carries units; strip them once, here.
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoule_per_mole
        kt = (MOLAR_GAS_CONSTANT_R * (self.temperature_k * kelvin)).value_in_unit(
            kilojoule_per_mole)
        return 1.0 / kt


def _check_openmm():
    """Say when the running OpenMM is not the one that produced the data beside this script.

    Not a refusal: running a reference on a newer OpenMM is frequently the whole point. But the
    random stream and the order of force summation both change with it, so the frames will
    differ, and that must not be something a reader discovers by accident.
    """
    import openmm

    recorded = PROVENANCE.get("openmm")
    running = openmm.__version__
    print(f"# openmm            : {{running}} (this bundle was produced with {{recorded}})")
    if recorded and running != recorded:
        print(f"# NOTE              : OpenMM differs from the recorded one. The ensemble is "
              f"reproduced; individual frames will not be.")


def _start_configuration(periodic):
    opened = XmlSerializer.deserialize((HERE / "start.xml").read_text(encoding="utf-8"))
    positions = opened.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    try:
        velocities = opened.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond)
    except Exception:
        velocities = np.zeros_like(np.asarray(positions))
    box = None
    if periodic:
        box = np.array(opened.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                       dtype=float)
    return Configuration(positions, velocities, box)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run this reference REST2 ladder.")
    parser.add_argument("--exchanges", type=int, default=SETTINGS["number_of_exchanges"],
                        help="override the number of exchange attempts, for a quick check")
    parser.add_argument("--platform", default=None, help="CUDA, CPU, ... (default: OpenMM's)")
    parser.add_argument("--out", default="exchange.csv", metavar="FILE",
                        help="where to write the exchange record (default: exchange.csv)")
    args = parser.parse_args(argv)

    _check_openmm()
    protocol = Protocol(SETTINGS)
    n = protocol.n_states
    systems = [XmlSerializer.deserialize((HERE / f"system_rung{{i}}.xml").read_text(
        encoding="utf-8")) for i in range(n)]
    pdb = PDBFile(str(HERE / "topology.pdb"))

    from openmm import Platform
    platform = Platform.getPlatformByName(args.platform) if args.platform else None
    engine = ReplicaEngine(protocol, systems, pdb.topology, platform=platform,
                           seed=protocol.random_seed)

    # Every rung starts from the same equilibrated configuration, as the ladder did.
    start = _start_configuration(engine.periodic)
    configurations = [start.copy() for _ in range(n)]
    state_to_walker = list(range(n))
    for index in range(n):
        engine.set_configuration(index, configurations[state_to_walker[index]])

    # Per-state relaxation before the first exchange, exactly as the engine's `_equilibrate` does
    # it: every rung propagated under ITS OWN Hamiltonian from the shared start, on the rung's own
    # seeded integrator, and not counted as production -- the step counter and the exchange stream
    # are untouched. This runner used to skip it, so a ladder with `equilibration_steps` > 0 was
    # reproduced from the wrong configurations and diverged at its first close exchange.
    equilibration = int(SETTINGS.get("equilibration_steps") or 0)
    if equilibration:
        print(f"# equilibration      : {{equilibration}} step(s) per state, NOT counted as "
              f"production")
        for index in range(n):
            engine.propagate(index, equilibration)
        for index in range(n):
            configurations[state_to_walker[index]] = engine.get_configuration(index)

    rule = NeighbouringExchangeRule()
    rng = np.random.default_rng(stream_seed(protocol.random_seed, "exchange"))
    interval = int(SETTINGS["exchange_interval_steps"])

    rows = ["exchange,step,state_i,state_j,log_alpha,accepted"]
    for attempt in range(int(args.exchanges)):
        for index in range(n):
            engine.propagate(index, interval)
        step = (attempt + 1) * interval
        for index in range(n):
            configurations[state_to_walker[index]] = engine.get_configuration(index)

        matrix = [[engine.reduced_potential_of(i, configurations[w]) for w in range(n)]
                  for i in range(n)]
        outcome = rule.propose(ExchangeContext(
            iteration=attempt, segment=step, protocol=protocol,
            state_to_walker=state_to_walker,
            reduced_potential=lambda i, w: matrix[i][w],
            rng=rng, exchange_index=attempt))
        for state_i, state_j, log_alpha, accepted in outcome.proposals:
            rows.append(f"{{attempt}},{{step}},{{state_i}},{{state_j}},{{log_alpha!r}},"
                        f"{{int(bool(accepted))}}")
        for state_i, state_j in outcome.swaps:
            state_to_walker[state_i], state_to_walker[state_j] = \\
                state_to_walker[state_j], state_to_walker[state_i]
        # Installed AFTER the swap, exactly as the engine does it: the configuration now assigned
        # to a state is the one that state's Context must continue from.
        for index in range(n):
            engine.set_configuration(index, configurations[state_to_walker[index]])

    (HERE / args.out).write_text("\\n".join(rows) + "\\n", encoding="utf-8")
    for index in range(n):
        final = engine._simulations[index].context.getState(getPositions=True, getVelocities=True)
        (HERE / f"final_rung{{index}}.xml").write_text(XmlSerializer.serialize(final),
                                                     encoding="utf-8")
    print(f"done: {{args.exchanges}} exchange attempt(s); record in {{args.out}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

SHELL = '''#!/usr/bin/env bash
# Run this reference REST2 ladder. Needs a Python with OpenMM and numpy on it.
#
#   ./run.sh                     the full {exchanges} exchange attempt(s)
#   ./run.sh --exchanges 20      a short check first, which is the sensible way to start
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec python run.py "$@"
'''


VERIFY = '''#!/usr/bin/env python
"""Rebuild every rung of this ladder from rung 0, and check it is the rung this bundle carries.

Standalone. Needs OpenMM only. `ladder/hamiltonian.py` is the code md-tools built the rungs with,
copied byte for byte, so this is the derivation that ran rather than a restatement of it.

    python verify_rungs.py

Each rung is compared after both Systems are serialised by the RUNNING OpenMM, so a newer OpenMM
that formats its XML differently still compares like with like. Exits 0 when every rung is
identical, and 1 naming the rungs that are not.
"""
import json
import sys
from pathlib import Path

from openmm import XmlSerializer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ladder.hamiltonian import build_scaled_system    # noqa: E402


def main():
    settings = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))
    derivation = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))["derivation"]
    solute = [int(i) for i in derivation["solute_atom_indices"]]
    excluded = [tuple(int(a) for a in pair) for pair in derivation["excluded_bonds"]]
    base = XmlSerializer.deserialize((HERE / "system_rung0.xml").read_text(encoding="utf-8"))

    differing = []
    for index, tau in enumerate(settings["tau"]):
        bundled = XmlSerializer.serialize(XmlSerializer.deserialize(
            (HERE / f"system_rung{index}.xml").read_text(encoding="utf-8")))
        rebuilt = XmlSerializer.serialize(
            build_scaled_system(base, solute, float(tau), excluded_bonds=excluded))
        same = rebuilt == bundled
        print(f"rung {index}  tau {float(tau):<10g} {'identical' if same else 'DIFFERS'}")
        if not same:
            differing.append(index)
    if differing:
        print(f"rebuilt from rung 0, rung(s) {differing} do not match the bundled System(s)")
        return 1
    print(f"all {len(settings['tau'])} rungs rebuild identically from rung 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _ladder_record(run_dir: Path, stage: str) -> dict[str, Any]:
    from ..build.record import read_record

    log = run_dir / f"{stage}.log"
    if not log.is_file():
        raise FileNotFoundError(f"{log} does not exist, so there is no finished {stage} to export")
    record = read_record(log)
    if record.get("status") != "completed":
        raise ValueError(f"{log} reports status {record.get('status')!r}, not 'completed'. A "
                         f"reference is exported from a finished run, never from a partial one.")
    kind = str(record.get("record_type") or "")
    if not kind.startswith("md-replica:"):
        raise ValueError(f"{log} is a {kind!r} record, not a replica ladder. For a single-stage "
                         f"run use `export-reference`. Nothing has been written.")
    return record


def export_rest2_reference(run_dir: Path, out_dir: Path, *, stage: str = "REST2") -> dict[str, Any]:
    """Write a standalone bundle for one finished REST2 ladder. Returns its manifest."""
    from openmm import XmlSerializer

    from ..build.record import source_commit
    from ..md.stage import solute_atom_indices
    from ..openmm.system import classify_omega_bonds
    from ..remd.protocol import build_rung_systems
    from ..run.preflight import load_inputs

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    record = _ladder_record(run_dir, stage)
    ladder = record.get("ladder") or {}
    inputs = record.get("inputs") or {}
    taus = [float(t) for t in record["tau"]]

    if ladder.get("reservoir", {}).get("enabled"):
        raise ValueError(
            f"{run_dir} ran with a reservoir (rREST2). A reservoir refresh replaces a rung's "
            f"configuration from a prepared phase-space file, and that file is not part of this "
            f"bundle; running it without one would be a different sampler. Nothing written.")
    if (ladder.get("collective_variables") or {}).get("file"):
        raise ValueError(
            f"{run_dir} ran with a collective-variable definition, which this bundle does not "
            f"carry or report. Nothing has been written.")

    found = {}
    for role in ("topology", "system"):
        source = _locate(run_dir, inputs[role])
        if source is None:
            raise FileNotFoundError(
                f"{role} {inputs[role]['path']!r} (sha256 {inputs[role]['sha256'][:16]}...) is "
                f"not beside {run_dir} or above it; it has to be found by digest.")
        found[role] = source

    parent = _continue_from(record)
    start = run_dir / Path(parent).name if parent else None
    if start is None or not start.is_file():
        raise FileNotFoundError(
            f"{run_dir} names no starting state that exists ({parent!r}). Every rung of this "
            f"ladder began from one equilibrated configuration; without it the bundle would "
            f"start from the built coordinates and call the result a reproduction.")

    # -- the N Systems the ladder PROPAGATED ---------------------------------------------------
    #
    # `build_rung_systems` is the function `Protocol.build_systems` delegates to and the one the
    # preflight prepares the rungs with, so these are the ladder's own Systems rather than a
    # second construction of them. Unlike a cMD stage there is no restraint force and no
    # barostat: a ladder is NVT by contract and its rungs carry neither.
    loaded = load_inputs(found["topology"], found["system"])
    solute = solute_atom_indices(loaded.pdb.topology)
    omega = classify_omega_bonds(loaded.pdb.topology, solute, route="peptide", ligand_sdf=None)
    excluded = [tuple(int(a) for a in bond) for bond in omega.get("omega_unscaled_bonds", [])]
    systems, audit = build_rung_systems(loaded.system, list(solute), tuple(taus),
                                        excluded_bonds=excluded, pressure_bar=None)
    if len(systems) != len(taus):
        raise ValueError(f"built {len(systems)} rung System(s) for {len(taus)} tau value(s)")

    # Proven before the directory exists, like every other refusal here.
    inputs_plan = user_inputs_plan(run_dir, found["system"], found["topology"])

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found["topology"], out_dir / "topology.pdb")
    shutil.copy2(start, out_dir / "start.xml")
    user_inputs = write_user_inputs(inputs_plan, out_dir)
    for index, system in enumerate(systems):
        (out_dir / f"system_rung{index}.xml").write_text(XmlSerializer.serialize(system),
                                                         encoding="utf-8")

    # -- the ladder's decision-making, copied rather than rewritten -----------------------------
    package = out_dir / "ladder"
    package.mkdir(exist_ok=True)
    environment = record.get("environment") or {}
    packages = environment.get("packages") or {}
    # Two commits, answering two different questions. `md_tools_commit` is the engine that RAN
    # the ladder, exactly as the run recorded it -- null when it recorded none, as the cMD export
    # leaves it. The modules in ladder/ are copied from THIS installation: the exporter's commit,
    # which is not necessarily the engine's. This used to fill the first from the second whenever
    # the run had recorded nothing, so a ladder run on one commit shipped a bundle naming the
    # exporter's commit beside the run's own version string and timestamps. Confidently wrong is
    # worse than honestly empty.
    from .. import __version__ as exporter_version
    copied_from = {"md_tools_version": exporter_version, "md_tools_commit": source_commit()}
    (package / "__init__.py").write_text(
        PACKAGE_INIT.format(version=exporter_version,
                            commit=copied_from["md_tools_commit"]
                            or "unknown (this build baked none)"),
        encoding="utf-8")
    remd = Path(__import__("md_tools.remd", fromlist=["__file__"]).__file__).parent
    for name in VENDORED:
        shutil.copy2(remd / name, package / name)
    rest2 = Path(__import__("md_tools.rest2", fromlist=["__file__"]).__file__).parent
    shutil.copy2(rest2 / SCALING_MODULE, package / SCALING_MODULE)

    dynamics = ladder.get("dynamics") or {}
    settings = {
        "name": stage,
        "ensemble": "NVT",
        "tau": taus,
        "n_states": len(taus),
        "exchange_interval_steps": int(ladder["exchange_interval_steps"]),
        "number_of_exchanges": int(ladder["number_of_exchanges"]),
        "timestep_fs": float(dynamics["timestep_fs"]),
        "temperature_K": float(dynamics["temperature_K"]),
        "friction_per_ps": float(dynamics["friction_per_ps"]),
        "constraint_tolerance": 1.0e-8,
        # The ladder's recorded seed, used exactly as the driver uses it: the per-rung integrator
        # streams are seed + 977*rung, and the acceptance draw is its own named substream.
        "seed": int(dynamics["seed"]),
        "equilibration_steps": int(ladder.get("equilibration_steps") or 0),
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "produced_by": "md-tools",
        "md_tools_version": packages.get("md-tools"),
        "md_tools_commit": environment.get("md_tools_commit"),
        "ladder_modules_from": copied_from,
        "openmm": packages.get("openmm"),
        "python": packages.get("python"),
        # Everything the run recorded. None of it is needed to RUN this bundle -- the Systems are
        # frozen -- but these are the tools that decided the Hamiltonian, and a reader asking
        # which openff produced these charges should not have to find the original run.
        "built_with": dict(sorted(packages.items())),
        "started_utc": record.get("started_utc"),
        "finished_utc": record.get("finished_utc"),
        "built_from": {
            "topology": {"path": inputs["topology"]["path"],
                         "sha256": _digest(found["topology"])},
            "system": {"path": inputs["system"]["path"], "sha256": _digest(found["system"])},
            "continued_from": parent,
        },
        "inputs": user_inputs,
        "force_audit": audit,
        # What rung i IS, in terms a reader can execute: rung 0 and these two lists are every
        # input `build_scaled_system` takes besides tau. The solute list used to be absent, so a
        # bundle stated the scaling rules but not which atoms they applied to.
        "derivation": {
            "rung_0": "system_rung0.xml, the unscaled System every other rung is derived from",
            "code": "ladder/hamiltonian.py: build_scaled_system(rung_0, solute_atom_indices, "
                    "tau, excluded_bonds)",
            "check": "python verify_rungs.py",
            "solute_atom_indices": [int(i) for i in solute],
            "excluded_bonds": [[int(a), int(b)] for a, b in excluded],
        },
        "note": "The modules in ladder/ are byte-for-byte copies of md_tools/remd/* and "
                "md_tools/rest2/hamiltonian.py at `ladder_modules_from` -- the exporter's commit, "
                "which is not necessarily the engine that ran (`md_tools_commit`). They are "
                "md_tools' own acceptance criterion, sweep schedule and rung construction, not a "
                "reimplementation. system_rung<i>.xml are the Systems the ladder propagated; "
                "verify_rungs.py rebuilds them from rung 0. Needs OpenMM and numpy only.",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n",
                                             encoding="utf-8")

    total_ps = (settings["exchange_interval_steps"] * settings["number_of_exchanges"]
                * settings["timestep_fs"] * 1e-3)
    title = (f"{stage}: {len(taus)} rungs, tau {taus[0]:g}..{taus[-1]:g}, "
             f"{total_ps * 1e-3:g} ns per rung")
    (out_dir / "run.py").write_text(RUNNER.format(title=title), encoding="utf-8")
    (out_dir / "verify_rungs.py").write_text(VERIFY, encoding="utf-8")
    shell = out_dir / "run.sh"
    shell.write_text(SHELL.format(exchanges=settings["number_of_exchanges"]), encoding="utf-8")
    shell.chmod(0o755)

    lines = []
    for path in sorted(p for p in out_dir.rglob("*")
                       if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts):
        lines.append(f"{_digest(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"settings": settings, "provenance": provenance, "files": len(lines) + 1,
            "vendored": list(VENDORED), "scaling_module": SCALING_MODULE}
