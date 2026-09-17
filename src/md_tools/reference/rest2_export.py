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
    rung_equilibration.py  per-tau equilibration, when the ladder ran with it

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
#: `rung_equilibration.py` is the per-tau equilibration (`rest2.equilibration_per_tau`), which the
#: runner performs with the same calls the driver makes.
VENDORED = ("core.py", "rules.py", "engine.py", "statistics.py", "rem_log.py",
            "rung_equilibration.py")

#: `md_tools/rest2/hamiltonian.py`, copied byte for byte beside them: the code that turned the
#: built System into the saved states (`md-openmm build-top --rest2-scaler`). With it,
#: `verify_rungs.py` rebuilds every bundled rung from `system_unscaled.xml` using OpenMM alone, so
#: how the scaled Systems were derived is checkable rather than described.
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

    system_rung<i>.xml are the saved states the ladder integrated, written by `md-openmm
    build-top --rest2-scaler`: system_unscaled.xml with the solute scaled at each rung's tau.
    `ladder/hamiltonian.py` is the code that did it, and `python verify_rungs.py` rebuilds each
    rung from system_unscaled.xml with it and checks the result is identical. provenance.json's
    `derivation` block, and scaler.yaml, hold the solute atoms and what stays unscaled.
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

    running_build = getattr(openmm.version, "version", None) or openmm.__version__
    revision = getattr(openmm.version, "git_revision", None)
    recorded_build = PROVENANCE.get("openmm_build")
    recorded = recorded_build or PROVENANCE.get("openmm")

    print(f"# openmm            : {{running_build}} (this bundle was produced with {{recorded}})")
    if revision:
        print(f"# openmm commit     : {{revision}}")
    if recorded_build and running_build != recorded_build:
        print(f"# NOTE              : OpenMM is a DIFFERENT BUILD from the one that produced "
              f"this data. The ensemble is reproduced; the exchange decisions will not be.")
    elif not recorded_build and recorded and openmm.__version__ != recorded:
        print(f"# NOTE              : OpenMM differs from the recorded one. The ensemble is "
              f"reproduced; individual frames will not be.")
    elif not recorded_build:
        print(f"# NOTE              : this bundle records only OpenMM's major.minor version, so "
              f"a different 8.x.y build cannot be detected here.")


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

    # Per-tau equilibration (`rest2.equilibration_per_tau`), exactly as the engine's
    # `_equilibrate_per_tau` does it: the equilibration stages on every rung under ITS OWN tau, on
    # a restrained copy of that rung, from the shared start, with `ladder/rung_equilibration.py`
    # -- the same bytes the engine ran. Before the per-state relaxation below, as there.
    per_tau = SETTINGS.get("per_tau_equilibration") or []
    if per_tau:
        from ladder.rung_equilibration import equilibrate_rung

        print("# per-tau equilibration: "
              + ", ".join(f"{{stage['name']}} {{stage['steps']}} step(s)" for stage in per_tau)
              + " on every rung, under its own tau, NOT counted as production")
        solute = [int(i) for i in PROVENANCE["derivation"]["solute_atom_indices"]]
        for index in range(n):
            configurations[index], _stages = equilibrate_rung(
                systems[index], configurations[index], per_tau,
                reference_positions=pdb.positions, restrained_atoms=solute,
                state_index=index, seed=protocol.random_seed,
                temperature_k=protocol.temperature_k, friction_per_ps=protocol.friction_per_ps,
                timestep_fs=protocol.timestep_fs,
                constraint_tolerance=protocol.constraint_tolerance, platform=platform)
            engine.set_configuration(index, configurations[index])

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
"""Rebuild every rung of this ladder from the built System, and check it is the rung bundled.

Standalone. Needs OpenMM only. `ladder/hamiltonian.py` is the code md-tools built the saved states
with (`md-openmm build-top --rest2-scaler`), copied byte for byte, so this is the derivation that
ran rather than a restatement of it. `system_unscaled.xml` is the built System they came from.

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
    impropers = bool(derivation["unscaled_impropers"])
    base = XmlSerializer.deserialize((HERE / "system_unscaled.xml").read_text(encoding="utf-8"))

    differing = []
    for index, tau in enumerate(settings["tau"]):
        bundled = XmlSerializer.serialize(XmlSerializer.deserialize(
            (HERE / f"system_rung{index}.xml").read_text(encoding="utf-8")))
        rebuilt = XmlSerializer.serialize(
            build_scaled_system(base, solute, float(tau), excluded_bonds=excluded,
                                unscaled_impropers=impropers))
        same = rebuilt == bundled
        print(f"rung {index}  tau {float(tau):<10g} {'identical' if same else 'DIFFERS'}")
        if not same:
            differing.append(index)
    if differing:
        print(f"rebuilt from the built System, rung(s) {differing} do not match the bundled "
              f"System(s)")
        return 1
    print(f"all {len(settings['tau'])} rungs rebuild identically from the built System")
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


def _openmm_build(record: dict[str, Any], packages: dict[str, Any],
                  run_dir: Path) -> str | None:
    """The PRECISE OpenMM build this ladder ran, from whichever record carries it.

    `packages.openmm` is `openmm.__version__`, which is only major.minor: it reads "8.6" for
    every 8.6.x, so a different patch release and a different dev build both compare equal to it
    -- and both change the random stream and the order of force summation, which for a ladder
    means different exchange decisions. A bundle has to be able to say "this is not the OpenMM
    that produced the data", and with the coarse string it cannot.

    Three sources, in order, because no single record is the richer one:

      1. `acceleration.openmm_version` -- what a STAGE record carries. A replica record has no
         platform block at all, so this is absent for ladders.
      2. `packages.openmm_build` -- recorded for every record type from 0.5.3 onward.
      3. `restart.json`'s `versions.openmm` -- which 0.5.2 ladders ALREADY wrote, beside the log
         this bundle is exported from. Without this a ladder finished before 0.5.3 could never
         gain the comparison, and those are exactly the finished reference runs someone wants a
         bundle of. It is read defensively: a missing or unreadable manifest yields None and the
         runner then says it can only see major.minor, rather than the export failing over a
         provenance nicety.
    """
    from_platform = (record.get("acceleration") or {}).get("openmm_version")
    if from_platform:
        return str(from_platform)
    from_packages = packages.get("openmm_build")
    if from_packages:
        return str(from_packages)
    manifest = Path(run_dir) / "restart.json"
    try:
        versions = (json.loads(manifest.read_text(encoding="utf-8")) or {}).get("versions") or {}
    except (OSError, ValueError):
        return None
    recorded = versions.get("openmm")
    return str(recorded) if recorded else None


def export_rest2_reference(run_dir: Path, out_dir: Path, *, stage: str = "REST2") -> dict[str, Any]:
    """Write a standalone bundle for one finished REST2 ladder. Returns its manifest."""
    from ..build.record import source_commit
    from ..run.preflight import load_inputs

    # Resolved before anything is searched; see `export_reference`.
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
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
            f"carry or report"
            + (", and torsion restraints, which its `verify_rungs.py` would rebuild by scaling "
               "the built System -- a restraint is added after scaling and is not part of what "
               "that check reconstructs" if (ladder.get("umbrella") or {}).get("file") else "")
            + ". Nothing has been written.")

    # A 0.5.4 LADDER INTEGRATED SAVED STATES and recorded each as `system_state<i>`; there is no
    # single `system` input. A record that has one ran before 0.5.4, when the ladder scaled at run
    # time under an earlier torsion convention, and rebuilding it with this code would not be the
    # Hamiltonian that ran.
    if "system" in inputs and "system_state0" not in inputs:
        raise ValueError(
            f"{run_dir} was run before 0.5.4: it records one -s and scaled its rungs at run time, "
            f"under an earlier unscaled-torsion convention. Export it with the md-tools that ran "
            f"it. Nothing has been written.")
    found = {}
    roles = ["topology"] + [f"system_state{index}" for index in range(len(taus))]
    for role in roles:
        if role not in inputs:
            raise ValueError(f"{run_dir}: the ladder record names no {role} input, so the "
                             f"{len(taus)}-state ladder it describes cannot be bundled. Nothing "
                             f"has been written.")
        source = _locate(run_dir, inputs[role])
        if source is None:
            raise FileNotFoundError(
                f"{role} {inputs[role]['path']!r} (sha256 {inputs[role]['sha256'][:16]}...) is "
                f"not beside {run_dir} or above it; it has to be found by digest.")
        found[role] = source

    # EVERY STATE IS FOLLOWED BACK TO THE BUILT SYSTEM through the one `scaler.yaml` they share,
    # as `export-reference` does for a hot stage: the build-top record proves built.xml, the scaler
    # record proves each state came from it, at the tau the ladder recorded.
    from ..rest2.states import ScaledStateError, load_scaler_record, scaled_state_identity

    identities = []
    for index in range(len(taus)):
        try:
            identity = scaled_state_identity(found[f"system_state{index}"])
        except ScaledStateError as refusal:
            raise ValueError(f"{refusal} Nothing has been written.") from None
        if identity is None or identity["method"] != "REST2" or identity["state"] != index \
                or abs(float(identity["tau"]) - taus[index]) > 1e-9:
            raise ValueError(
                f"{found[f'system_state{index}']} is not state {index} (tau {taus[index]:g}) of a "
                f"REST2 scaler record beside it, which is what the ladder recorded integrating. "
                f"Nothing has been written.")
        identities.append(identity)
    record_paths = {str(Path(identity["record"]).resolve()) for identity in identities}
    if len(record_paths) != 1:
        raise ValueError(f"{run_dir}: the ladder's states come from {len(record_paths)} scaler "
                         f"records; a ladder is one schedule. Nothing has been written.")
    scaler_path = Path(next(iter(record_paths)))
    scaler = load_scaler_record(scaler_path)
    built = scaler_path.parent.parent / scaler["source"]["system"]
    if not built.is_file() or _digest(built) != scaler["source"]["system_sha256"]:
        raise FileNotFoundError(
            f"the ladder's states were made from {scaler['source']['system']!r} (sha256 "
            f"{scaler['source']['system_sha256'][:16]}...), which is not at {built} with that "
            f"digest, so the bundle could not prove how they were derived. Nothing has been "
            f"written.")
    if _digest(found["topology"]) != scaler["source"]["topology_sha256"]:
        raise ValueError(f"the ladder's states were scaled against a topology with a different "
                         f"sha256 from the one the ladder ran with. Nothing has been written.")
    found["system"] = built

    # THE RECORDED `-c`, RESOLVED AGAINST THE RUN DIRECTORY -- then the bare name beside the
    # records. The fifth site holding this same assumption, after `_stage_inputs`, `_locate` and
    # the cMD `-c` parent in `export.py`: a ladder now starts from `eq/eq.xml` or `../min/eq.xml`
    # because preparation stages write into their own directories, so stripping to the basename
    # looked at the run root and refused every ladder built under this layout. The basename stays
    # as the fallback for a run generated before the split, when every stage wrote to one place.
    parent = _continue_from(record)
    start = None
    if not parent and "group_file" in inputs:
        # A 0.5.4 ladder has no `-c` on its command line: every group line names its own, and the
        # lines are homogeneous (`remd.executor` refuses a ladder whose starts differ). The group
        # file is a recorded input, found by digest like the others; its paths are relative to it.
        from ..remd.executor import GroupFileError, parse_group_file

        group_path = _locate(run_dir, inputs["group_file"])
        if group_path is not None:
            try:
                parent = parse_group_file(group_path)[0].get("coordinates")
            except GroupFileError as refusal:
                raise ValueError(f"{refusal} Nothing has been written.") from None
    if parent:
        recorded = Path(parent)
        options = [run_dir / recorded] if not recorded.is_absolute() else [recorded]
        options.append(run_dir / recorded.name)
        start = next((option for option in options if option.is_file()), None)
    if start is None or not start.is_file():
        raise FileNotFoundError(
            f"{run_dir} names no starting state that exists ({parent!r}). Every rung of this "
            f"ladder began from one equilibrated configuration; without it the bundle would "
            f"start from the built coordinates and call the result a reproduction.")

    # -- the N Systems the ladder PROPAGATED ---------------------------------------------------
    #
    # The saved states themselves, byte for byte: the ladder integrated these files as they are,
    # so the bundle carries them rather than a second construction. What they leave unscaled is
    # read from the record that made them, never re-classified here.
    loaded = load_inputs(found["topology"], found["system"])
    solute = [int(i) for i in scaler["solute"]["atom_indices"]]
    torsions = scaler["unscaled_torsions"]
    excluded = [tuple(int(a) for a in bond) for bond in torsions["unscaled_central_bonds"]]
    impropers = bool(torsions["unscaled_impropers"])
    from ..rest2.scaler import UnclassifiedForceError, audit_force_classes

    try:
        audit = audit_force_classes(loaded.system, where=f"{stage} reference export")
    except UnclassifiedForceError as unknown:
        raise ValueError(f"{unknown} Nothing has been written.") from None

    # Proven before the directory exists, like every other refusal here.
    inputs_plan = user_inputs_plan(run_dir, found["system"], found["topology"])
    # How the states are MADE again: the scaler configuration `scaler.yaml` records by digest.
    wanted_config = (scaler.get("config") or {}).get("sha256")
    config_file = (scaler.get("config") or {}).get("file")
    scaler_config = None
    if config_file:
        scaler_config = next((path for path in (scaler_path.parent / config_file,
                                                scaler_path.parent.parent / config_file,
                                                scaler_path.parent.parent.parent / config_file)
                              if path.is_file() and _digest(path) == wanted_config), None)
    inputs_plan["scaled_ladder"] = {"record": scaler_path, "config": scaler_config}

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found["topology"], out_dir / "topology.pdb")
    shutil.copy2(start, out_dir / "start.xml")
    user_inputs = write_user_inputs(inputs_plan, out_dir)
    for index in range(len(taus)):
        shutil.copy2(found[f"system_state{index}"], out_dir / f"system_rung{index}.xml")
    shutil.copy2(found["system"], out_dir / "system_unscaled.xml")
    shutil.copy2(scaler_path, out_dir / "scaler.yaml")

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
        # `rest2.equilibration_per_tau`, as the stages the ladder ran on every rung; [] when off.
        "per_tau_equilibration": [dict(stage) for stage in
                                  (ladder.get("per_tau_equilibration") or [])],
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "produced_by": "md-tools",
        "md_tools_version": packages.get("md-tools"),
        "md_tools_commit": environment.get("md_tools_commit"),
        "ladder_modules_from": copied_from,
        "openmm": packages.get("openmm"),
        "openmm_build": _openmm_build(record, packages, run_dir),
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
            "system": {"path": scaler["source"]["system"], "sha256": _digest(found["system"])},
            "states": [{"path": inputs[f"system_state{index}"]["path"],
                        "sha256": _digest(found[f"system_state{index}"])}
                       for index in range(len(taus))],
            "continued_from": parent,
        },
        "inputs": user_inputs,
        "force_audit": audit,
        # What rung i IS, in terms a reader can execute: rung 0 and these two lists are every
        # input `build_scaled_system` takes besides tau. The solute list used to be absent, so a
        # bundle stated the scaling rules but not which atoms they applied to.
        "derivation": {
            "unscaled": "system_unscaled.xml, the built System every rung is derived from",
            "record": "scaler.yaml, written by `md-openmm build-top --rest2-scaler`",
            "code": "ladder/hamiltonian.py: build_scaled_system(unscaled, solute_atom_indices, "
                    "tau, excluded_bonds, unscaled_impropers)",
            "check": "python verify_rungs.py",
            "solute_atom_indices": [int(i) for i in solute],
            "excluded_bonds": [[int(a), int(b)] for a, b in excluded],
            "unscaled_impropers": impropers,
        },
        "note": "The modules in ladder/ are byte-for-byte copies of md_tools/remd/* and "
                "md_tools/rest2/hamiltonian.py at `ladder_modules_from` -- the exporter's commit, "
                "which is not necessarily the engine that ran (`md_tools_commit`). They are "
                "md_tools' own acceptance criterion, sweep schedule and rung construction, not a "
                "reimplementation. system_rung<i>.xml are the saved states the ladder "
                "propagated; verify_rungs.py rebuilds them from system_unscaled.xml. Needs OpenMM "
                "and numpy only.",
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
