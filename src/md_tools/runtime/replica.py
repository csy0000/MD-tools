"""Drive a REST2 / rREST2 ladder from a generated script.

The science here is not new and is not re-derived. The Hamiltonian scaling, the exchange
algorithm, the fixed state trajectories, the `rem.log` projection, the neighbouring-pair
acceptance report and every restart rule are the implementation already validated on this branch,
reached through the same executor entry point the previous generated projects used.

What this module replaces is the *packaging* around it. The old design copied a dozen runtime
modules into every generated directory and launched them through a separate `openmm-md`
executable. A generated directory now holds one small script; the runtime is imported from the
installed distribution, and the executor is called as a function.

Three things have to be prepared before the executor runs, and all three are derived from the
built system rather than carried in configuration that could go stale:

  solute.yaml    which atoms are solute, and which amide omega bonds must NOT be scaled. Derived
                 with the same `classify_omega_bonds` the builder uses, so the ladder scales
                 exactly what the build recorded.
  _protocol.py   one REST2Protocol describing the ladder.
  ladder.group   one group line per state, inputs only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def templates_directory() -> Path:
    """The installed runtime modules the executor imports by bare name.

    `openmm_md.run_grouped` does `from replica_driver import ReplicaRun`, a bare import that the
    copy-based generated projects satisfied by having the modules beside the script. Putting the
    INSTALLED directory on `sys.path` satisfies the same import from the wheel, which is what lets
    a generated directory hold one file instead of a dozen copies.
    """
    from importlib.resources import files
    # `templates` ships as a directory of modules rather than a package -- it has no __init__.py,
    # because the generated projects imported its files by bare name. It therefore has no
    # __file__, and importlib.resources is the supported way to locate it in an installed wheel
    # as well as in a source tree.
    return Path(str(files("md_tools.openmm").joinpath("templates"))).resolve()


def _ensure_runtime_importable() -> Path:
    directory = templates_directory()
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return directory


def tau_ladder(n_states: int, tau_max: float) -> list[float]:
    """A linear ladder from 0 to tau_max inclusive. State 0 is always the unscaled Hamiltonian."""
    if n_states < 2:
        raise SystemExit(f"a ladder needs at least 2 states, not {n_states}")
    step = float(tau_max) / (n_states - 1)
    return [round(index * step, 6) for index in range(n_states)]


def write_solute_document(topology_path: Path, system_path: Path, out: Path, *,
                          route: str = "peptide") -> dict[str, Any]:
    """Derive and write solute.yaml from the built system.

    Derived, never configured. The omega classification decides which torsions keep their physical
    barrier at the hot rungs; a stale hand-written list would change the Hamiltonian silently.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..openmm.config import write_yaml
    from ..openmm.system import classify_omega_bonds
    from ..openmm.sysgen import _solute_document
    from .stage import solute_atom_indices

    topology = PDBFile(str(topology_path)).topology
    system = XmlSerializer.deserialize(Path(system_path).read_text(encoding="utf-8"))
    indices = solute_atom_indices(topology)
    omega = classify_omega_bonds(topology, indices, route=route, ligand_sdf=None)
    document = _solute_document(topology, indices, omega, route=route, system=system)
    ambiguous = (document.get("rest2") or {}).get("omega_ambiguous_candidates") or []
    if ambiguous:
        raise SystemExit(
            f"{len(ambiguous)} amide candidate(s) could not be classified as ordinary or "
            f"proline-like. Guessing either way silently changes the Hamiltonian, so the ladder "
            f"is refused rather than run. Candidates: {ambiguous[:3]}")
    write_yaml(out, document)
    return document


PROTOCOL_TEMPLATE = '''#!/usr/bin/env python
"""{protocol}: {n_states} states, tau 0.0 to {tau_max}, NVT at {temperature} K.

Written by the generated {protocol}.py at run time. Every state is thermostatted at the SAME
temperature and differs only by Hamiltonian: this is Hamiltonian scaling, not temperature REMD,
and an exchange never rescales velocities.

    tau ladder : {ladder}
    (1-tau)^2 on solute-solute terms, (1-tau) on solute-environment terms, omega left unscaled
"""
from replica_runtime import REST2Protocol

protocol = REST2Protocol(
    tau={ladder},
    temperature_k={temperature},
    timestep_fs={timestep},
    exchange_interval_ps={exchange_ps},
    whole_output_interval_ps={whole_ps},
    solute_output_interval_ps={solute_ps},
    number_of_exchanges={exchanges},
    friction_per_ps={friction},
    equilibration_ps={equilibration_ps},
    random_seed={seed},
    hydrogen_mass_amu=None,
    platform={platform!r},
    precision=None,
)
'''


def replica_main(ladder: dict[str, Any], argv: list[str] | None = None) -> int:
    """Prepare the ladder's inputs and hand them to the validated executor."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{ladder['protocol']}: one coordinated replica-exchange ladder")
    parser.add_argument("-p", "--topology", required=True, metavar="PDB")
    parser.add_argument("-s", "--system", required=True, metavar="XML")
    parser.add_argument("-c", "--continue-from", default=None, metavar="XML",
                        help="equilibrated state every replica starts from")
    parser.add_argument("-log", "--log", default=None, metavar="LOG")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="where the ladder's outputs are written (default: here)")
    parser.add_argument("--route", default="peptide", choices=("peptide", "ligand"),
                        help="how the omega classifier reads the solute")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extend", type=int, default=0, metavar="N")
    parser.add_argument("--extend-from", default=None, metavar="DIRECTORY")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    directory = _ensure_runtime_importable()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    protocol_name = ladder["protocol"]
    dynamics = ladder["dynamics"]
    timestep = float(dynamics["timestep_fs"])
    states = int(ladder["n_states"])
    taus = tau_ladder(states, float(ladder["tau_max"]))
    exchange_ps = int(ladder["exchange_interval_steps"]) * timestep / 1000.0

    solute_yaml = out / "solute.yaml"
    if not solute_yaml.is_file():
        write_solute_document(Path(args.topology), Path(args.system), solute_yaml,
                              route=args.route)

    protocol_file = out / "_protocol.py"
    protocol_file.write_text(PROTOCOL_TEMPLATE.format(
        protocol=protocol_name, n_states=states, tau_max=ladder["tau_max"],
        temperature=float(dynamics["temperature_K"]), ladder=taus, timestep=timestep,
        exchange_ps=exchange_ps, whole_ps=exchange_ps, solute_ps=exchange_ps,
        exchanges=int(ladder["number_of_exchanges"]),
        friction=float(dynamics["friction_per_ps"]),
        equilibration_ps=0.0, seed=int(dynamics["seed"]),
        platform=dynamics.get("platform"),
    ), encoding="utf-8")

    group_file = out / f"{protocol_name}.group"
    lines = [f"# {protocol_name}: {states} states, tau 0.0 to {ladder['tau_max']}.",
             "# One group per line, inputs only. Run-level outputs go on the executor call,",
             "# because they describe the coordinated run rather than one replica.",
             ""]
    for index in range(states):
        parts = [f"-i {protocol_file.name}", f"-p {args.topology}", f"-s {args.system}"]
        if args.continue_from:
            parts.append(f"-c {args.continue_from}")
        parts += [f"--solute {solute_yaml.name}", f"--group-index {index}"]
        lines.append(" ".join(parts))
    group_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    executor_argv = [
        "-ng", str(states),
        "--groupfile", str(group_file),
        "-o", str(out / f"{protocol_name}.out"),
        "-x", str(out / f"{protocol_name}.nc"),
        "-r", str(out / "restart.json"),
        "--checkpoint", str(out / f"{protocol_name}_checkpoint.nc"),
    ]
    if ladder.get("rem_log", True):
        executor_argv += ["--rem", str(out / "rem.log")]
    if ladder.get("reservoir", {}).get("enabled"):
        executor_argv += ["--reservoir", str(Path(ladder["reservoir"]["path"]).resolve())]
    if args.resume:
        executor_argv.append("--resume")
    if args.verify_only:
        executor_argv.append("--verify-only")
    if args.force:
        executor_argv.append("--force")
    if args.extend:
        executor_argv += ["--extend", str(args.extend)]
    if args.extend_from:
        executor_argv += ["--extend-from", args.extend_from]

    from ..build.record import LogWriter, file_facts
    from ..openmm.templates import openmm_md

    log_path = Path(args.log) if args.log else out / f"{protocol_name}.log"
    log = LogWriter(log_path, record_type=f"md-replica:{protocol_name}", echo=False)
    log(f"md-openmm {protocol_name}")
    log("=" * 68)
    log.heading("Ladder")
    log.field("states", states)
    log.field("tau", taus)
    log.field("exchange every", f"{ladder['exchange_interval_steps']} steps = {exchange_ps:g} ps")
    log.field("attempts", ladder["number_of_exchanges"])
    log.field("temperature", f"{dynamics['temperature_K']} K (every state, NVT)")
    log.update(ladder=dict(ladder), tau=taus, exchange_interval_ps=exchange_ps,
               inputs={"topology": file_facts(Path(args.topology)),
                       "system": file_facts(Path(args.system))})

    code = int(openmm_md.main(executor_argv) or 0)

    # The executor owns the run and writes its own authoritative records. This log exists so that
    # every artefact this package produces carries the SAME machine record, and so registration
    # never has to read the executor's prose summary to decide whether a ladder finished.
    restart = out / "restart.json"
    if code == 0 and restart.is_file():
        outputs = {"restart_json": file_facts(restart, relative_to=out)}
        for name in sorted(out.glob("remd*.nc")):
            outputs[name.name] = file_facts(name, relative_to=out)
        for extra in (out / f"{protocol_name}.nc", out / f"{protocol_name}_checkpoint.nc",
                      out / "rem.log"):
            if extra.is_file():
                outputs[extra.name] = file_facts(extra, relative_to=out)
        log.update(outputs=outputs, state_trajectories=len(list(out.glob("remd*.nc"))))
        log.complete()
        log.heading("Summary")
        log(f"  {protocol_name}: {states} states, {ladder['number_of_exchanges']} exchanges")
        log("  status: completed")
    else:
        log.fail(f"the executor returned {code}")
    log.save()
    return code
