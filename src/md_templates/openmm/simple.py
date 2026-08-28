"""`md-openmm setup`: one small request in, a runnable OpenMM directory out.

The generated directory is the point of this module. It looks like an Amber or GROMACS run
directory -- one readable input per stage, run directly, writing a `.out` beside it -- and it is
**detached**: no `md_templates` import, no YAML read at run time, no Git, no component checkout.
Delete this package after generating and every stage still runs.

That is the whole trade. Everything the old generated scripts did at run time -- resolving force
fields, validating the timestep against hydrogen mass, converting picoseconds to steps, deriving
seeds -- happens HERE, once, and reaches the script as a literal. A reader sees `4` and not
`int(config["common"]["timestep_fs"])`.

What is deliberately NOT here: dataset registration, checksum manifests, component locking,
continuation fingerprints. Those belong to whoever registers the result, and putting them in a
stage script is what made the previous generation 286 lines to say twelve lines of OpenMM.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import defaults as D
from .config import ConfigError

#: Bumped when the `.out` header grammar changes. Parsers key off it.
OUTPUT_VERSION = 1

#: Stage order for conventional MD. Explicit solvent equilibrates through two NPT stages; implicit
#: has no box, so there is no barostat and the free stage is NVT.
EXPLICIT_STAGES = ("nvt_1kcal", "npt_1kcal", "npt_free")
IMPLICIT_STAGES = ("nvt_1kcal", "nvt_free")


def _duration_ps(value: Any, *, field_name: str) -> float:
    """Accept `1 ns`, `250 ps`, `0.5`, or a number. Picoseconds unless a unit says otherwise."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    for suffix, scale in (("ns", 1000.0), ("ps", 1.0), ("fs", 0.001)):
        if text.endswith(suffix):
            head = text[: -len(suffix)].strip()
            try:
                return float(head) * scale
            except ValueError:
                raise ConfigError(f"{field_name}: {value!r} is not a number followed by {suffix}")
    try:
        return float(text)
    except ValueError:
        raise ConfigError(
            f"{field_name}: {value!r} is not a duration. Write `1 ns`, `250 ps` or a number of ps.")


def _steps(duration_ps: float, timestep_fs: float, *, field_name: str) -> int:
    """Picoseconds to steps, refusing anything that is not a whole number of them.

    A 7 ps interval at 2 fs is 3500 steps and fine; at 3 fs it is 2333.33, and a reporter cannot
    write a third of a step. Rounding silently would put frames at times the header does not state.
    """
    exact = duration_ps * 1000.0 / timestep_fs
    rounded = round(exact)
    if abs(exact - rounded) > 1e-9:
        raise ConfigError(
            f"{field_name}: {duration_ps} ps at a {timestep_fs} fs timestep is {exact} steps, "
            f"which is not a whole number. Choose an interval that divides exactly -- "
            f"{rounded * timestep_fs / 1000.0} ps would.")
    if rounded < 1:
        raise ConfigError(f"{field_name}: {duration_ps} ps is less than one step at {timestep_fs} fs")
    return int(rounded)


@dataclass
class SetupRequest:
    """The small portable request. Everything else comes from the preset."""

    system: str
    input: str
    type: str = "peptide"
    solvent: str = "explicit"
    protocol: str = "cMD"
    production: Any = "1 ns"
    output_interval: Any = "5 ps"
    platform: str = "automatic"
    output: Optional[str] = None
    advanced: dict = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: dict) -> "SetupRequest":
        known = {f for f in cls.__dataclass_fields__ if f != "advanced"}
        unknown = set(document) - known - {"advanced"}
        if unknown:
            raise ConfigError(
                f"setup request has unknown field(s): {sorted(unknown)}. Advanced settings go "
                f"under `advanced:`, so a typo at the top level is a typo and not a silent default.")
        for required in ("system", "input"):
            if not document.get(required):
                raise ConfigError(f"setup request needs `{required}`")
        return cls(**{k: v for k, v in document.items() if k in known or k == "advanced"})


def resolve(request: SetupRequest) -> dict[str, Any]:
    """The request plus this repository's validated defaults, with every number made concrete.

    Nothing scientific is invented here: `sys_defaults` and `md_defaults` are the same functions
    the older `sys-config` command uses, so a preset and a hand-written configuration resolve to
    the same force fields, box and integrator.
    """
    if request.type not in ("peptide", "ligand"):
        raise ConfigError(f"type must be `peptide` or `ligand`, not {request.type!r}")
    if request.solvent not in ("explicit", "implicit"):
        raise ConfigError(f"solvent must be `explicit` or `implicit`, not {request.solvent!r}")
    if str(request.protocol).lower() not in ("cmd",):
        raise ConfigError(
            f"protocol must be `cMD` in this milestone, not {request.protocol!r}. REST2 and AIS "
            f"keep their existing commands and are untouched here.")

    implicit = request.solvent == "implicit"
    solvent_name = "GBn2" if implicit else "TIP3P"
    peptide = request.type == "peptide"

    system_config = D.sys_defaults(peptide=peptide, solvent=solvent_name)
    protocol = D.md_defaults(methods=("cMD",), solvent=solvent_name)
    advanced = dict(request.advanced or {})

    # Advanced overrides are applied to the SAME documents the defaults produced, so an override
    # is visibly a departure from a known baseline rather than a separate code path.
    for dotted, value in advanced.items():
        target: Any = None
        parts = dotted.split(".")
        for document in (system_config, protocol):
            probe = document
            for key in parts[:-1]:
                probe = probe.get(key) if isinstance(probe, dict) else None
                if probe is None:
                    break
            if isinstance(probe, dict) and parts[-1] in probe:
                target = probe
                break
        if target is None:
            raise ConfigError(
                f"advanced setting {dotted!r} matches no field in the resolved configuration")
        target[parts[-1]] = value

    common = protocol["common"]
    timestep_fs = float(common["timestep_fs"])
    hydrogen_mass = (system_config.get("constraints") or {}).get("hydrogen_mass_amu")

    # The invariant this repository has always enforced, now enforced once, at generation.
    if timestep_fs > 3.0 and not hydrogen_mass:
        raise ConfigError(
            f"a {timestep_fs} fs timestep needs hydrogen-mass repartitioning. Set "
            f"`advanced: {{constraints.hydrogen_mass_amu: 4.0}}` or lower the timestep. "
            f"Bonds to hydrogen are constrained, but their angular motion is not, and above about "
            f"3 fs the integration is no longer stable without repartitioning.")

    production_ps = _duration_ps(request.production, field_name="production")
    interval_ps = _duration_ps(request.output_interval, field_name="output_interval")

    stages: list[dict[str, Any]] = []
    equilibration = protocol["equilibration"]
    for name in (IMPLICIT_STAGES if implicit else EXPLICIT_STAGES):
        key = {"nvt_1kcal": "nvt_restrained_duration_ps",
               "npt_1kcal": "npt_restrained_duration_ps",
               "npt_free": "npt_free_duration_ps",
               "nvt_free": "nvt_free_duration_ps"}[name]
        duration = equilibration.get(key)
        if duration is None:
            raise ConfigError(f"equilibration.{key} is null but stage {name} needs it")
        stages.append({
            "name": name,
            "duration_ps": float(duration),
            "steps": _steps(float(duration), timestep_fs, field_name=f"equilibration.{key}"),
            "ensemble": "NVT" if name.startswith("nvt") else "NPT",
            "restraint_kcal": (float(equilibration["restraint_k_kcal_mol_a2"])
                               if name.endswith("1kcal") else 0.0),
        })

    return {
        "system_id": request.system,
        "title": advanced.get("title") or request.system,
        "type": request.type,
        "solvent": request.solvent,
        "implicit": implicit,
        "protocol": "cMD",
        "platform": request.platform,
        "sys_config": system_config,
        "protocol_config": protocol,
        "timestep_fs": timestep_fs,
        "temperature_K": float(common["temperature_kelvin"]),
        "friction_per_ps": float(common["friction_per_ps"]),
        "pressure_bar": None if implicit else float(common["pressure_bar"]),
        "barostat_interval": None if implicit else int(common["barostat_frequency_steps"]),
        "hydrogen_mass_amu": hydrogen_mass,
        "minimization_max_iterations": int(protocol["minimization"]["max_iterations"]),
        "equilibration_stages": stages,
        "production_ps": production_ps,
        "production_steps": _steps(production_ps, timestep_fs, field_name="production"),
        "output_interval_ps": interval_ps,
        "output_interval_steps": _steps(interval_ps, timestep_fs, field_name="output_interval"),
    }


def format_preset(resolved: dict[str, Any]) -> str:
    """What the user confirms before anything is written."""
    ff = resolved["sys_config"]["forcefield"]
    solvent = resolved["sys_config"].get("solvent") or {}
    implicit = resolved["sys_config"].get("implicit_solvent") or {}
    lines = [
        f"system        {resolved['system_id']}  ({resolved['type']}, {resolved['solvent']} solvent)",
        f"protocol      {resolved['protocol']}",
    ]
    if resolved["implicit"]:
        lines.append(f"force field   {ff.get('protein')} + {implicit.get('model')}/{implicit.get('radii')}")
    else:
        lines.append(f"force field   {ff.get('protein')} + {ff.get('water')}")
        lines.append(f"solvent box   {solvent.get('model')}, {solvent.get('padding_nm')} nm padding, "
                     f"{solvent.get('box_shape')}, {solvent.get('ionic_strength_molar')} M, "
                     f"{solvent.get('cutoff_nm')} nm cutoff")
    if resolved["type"] == "ligand":
        solute = resolved["sys_config"]["solute"]
        lines.append(f"ligand        {solute['ligand_forcefield']} + {solute['ligand_charge_method']}")
    lines += [
        f"integrator    LangevinMiddle, {resolved['temperature_K']} K, "
        f"{resolved['friction_per_ps']} /ps, {resolved['timestep_fs']} fs",
    ]
    if not resolved["implicit"]:
        lines.append(f"barostat      MonteCarlo, {resolved['pressure_bar']} bar, "
                     f"every {resolved['barostat_interval']} steps")
    lines.append("minimization  %d iterations" % resolved["minimization_max_iterations"])
    for stage in resolved["equilibration_stages"]:
        lines.append(f"equilibration {stage['name']:<10} {stage['duration_ps']:>8.3f} ps  "
                     f"{stage['steps']:>10,d} steps  {stage['ensemble']}"
                     + (f"  restraint {stage['restraint_kcal']} kcal/mol/A^2"
                        if stage["restraint_kcal"] else ""))
    lines += [
        f"production    {resolved['production_ps']:.3f} ps  {resolved['production_steps']:,d} steps",
        f"output every  {resolved['output_interval_ps']:.3f} ps  "
        f"{resolved['output_interval_steps']:,d} steps  "
        f"({resolved['production_steps'] // resolved['output_interval_steps']:,d} frames)",
        f"platform      {resolved['platform']}",
    ]
    return "\n".join("  " + line for line in lines)


def origin_commit() -> Optional[str]:
    """The MD-templates commit generating this directory, read ONCE, at generation.

    Recorded as a literal in `config.yaml`. No generated script ever runs `git`.
    """
    here = Path(__file__).resolve()
    try:
        result = subprocess.run(["git", "-C", str(here.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def refuse_unsafe_output(root: Path) -> None:
    """A data root inside a Git worktree will eventually be committed by accident."""
    root = Path(root).resolve()
    probe = subprocess.run(["git", "-C", str(root if root.is_dir() else root.parent),
                            "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if probe.returncode == 0:
        raise ConfigError(
            f"{root} is inside the Git working tree at {probe.stdout.strip()}. Generated systems, "
            f"trajectories and checkpoints must live outside every repository -- an ignore rule is "
            f"not protection, because it is one `git add -f` from being wrong. Point --output at a "
            f"managed storage root instead.")


def generate(resolved: dict[str, Any], *, input_path: Path, output_root: Path,
             overwrite: bool = False, echo: bool = True) -> dict[str, Any]:
    """Build the system, then write the stage scripts beside it.

    The system is built by `sysgen.generate_system`, unchanged: force fields, solvation, ligand
    parameterisation and charges are the validated code paths this repository already has, and this
    milestone is not the place to reimplement them. What changes is everything after -- the scripts.
    """
    from . import emit
    from .sysgen import generate_system

    output_root = Path(output_root).resolve()
    refuse_unsafe_output(output_root)
    system_dir = output_root / resolved["system_id"]

    if system_dir.exists() and any(system_dir.iterdir()) and not overwrite:
        raise ConfigError(
            f"{system_dir} already exists and is not empty. Generating again would overwrite stage "
            f"scripts beside trajectories produced by the previous ones, leaving a directory whose "
            f"`.out` files describe a run its scripts no longer perform. Choose another system name, "
            f"or pass --overwrite if the directory is genuinely scratch.")

    common = system_dir / "common"
    common.mkdir(parents=True, exist_ok=True)

    # The system configuration sysgen consumes. Written into the generated directory so the build
    # is reproducible from what is on disk, then never read again by anything at run time.
    sys_config_path = system_dir / "common" / "sys.config.yaml"
    document = dict(resolved["sys_config"])
    document["dataset"] = {"enabled": False}
    sys_config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    generate_system(input_path=Path(input_path), config_path=sys_config_path,
                    output_folder=common, echo=echo)

    # The solute atom range, read ONCE here so the restrained stages can name it as a literal
    # instead of parsing YAML at run time. sysgen states whether the indices are contiguous; if
    # they ever are not, refusing is better than emitting a range that silently restrains the
    # wrong atoms.
    solute = yaml.safe_load((common / "solute.yaml").read_text(encoding="utf-8"))
    if not solute.get("solute_atom_indices_are_contiguous", False):
        raise ConfigError(
            f"{common / 'solute.yaml'} reports non-contiguous solute indices, which this "
            f"generator cannot express as a literal range in a restrained stage script.")
    resolved = dict(resolved, solute_range=tuple(solute["solute_atom_range"]))

    written: list[str] = []

    (system_dir / "min").mkdir(exist_ok=True)
    (system_dir / "min" / "min.py").write_text(emit.minimization_script(resolved), encoding="utf-8")
    written.append("min/min.py")

    # The first equilibration stage runs from eq/<name>/, two levels below the system root,
    # so it reaches minimisation through `../../min`. Later stages are siblings in eq/.
    parent = "../../min/min.state.xml"
    for stage in resolved["equilibration_stages"]:
        directory = system_dir / "eq" / stage["name"]
        directory.mkdir(parents=True, exist_ok=True)
        script = emit.equilibration_script(resolved, stage, parent)
        (directory / f"{stage['name']}.py").write_text(script, encoding="utf-8")
        written.append(f"eq/{stage['name']}/{stage['name']}.py")
        parent = f"../{stage['name']}/{stage['name']}.state.xml"

    # The production stage sits one level up from the eq/ stages, so its parent path differs.
    last = resolved["equilibration_stages"][-1]["name"]
    (system_dir / "cMD").mkdir(exist_ok=True)
    (system_dir / "cMD" / "cmd.py").write_text(
        emit.production_script(resolved, f"../eq/{last}/{last}.state.xml"), encoding="utf-8")
    written.append("cMD/cmd.py")

    (system_dir / "run.sh").write_text(_run_script(resolved), encoding="utf-8")
    (system_dir / "run.sh").chmod(0o755)
    written.append("run.sh")

    (system_dir / "config.yaml").write_text(_system_config(resolved, input_path), encoding="utf-8")
    written.append("config.yaml")

    return {"system_dir": system_dir, "written": written}


def _run_script(resolved: dict[str, Any]) -> str:
    """Every stage in order. Short enough to read before trusting it."""
    lines = ["min/min.py"]
    lines += [f"eq/{s['name']}/{s['name']}.py" for s in resolved["equilibration_stages"]]
    lines.append("cMD/cmd.py")
    body = "\n".join(f"run {path}" for path in lines)
    return f'''#!/usr/bin/env bash
# Every stage of {resolved["system_id"]}, in order. Each writes its own .out beside its script.
#
# Runs any stage on its own just as well:
#     cd eq/nvt_1kcal && python nvt_1kcal.py > nvt_1kcal.out 2>&1
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"

run () {{
    local script="$1"
    local directory; directory="$(dirname "$script")"
    local name; name="$(basename "$script" .py)"
    echo "== $name =="
    # `|| true` so a failing stage reaches the check below and is named, rather than
    # `set -e` aborting with nothing said about which stage stopped it.
    ( cd "$directory" && python "$name.py" > "$name.out" 2>&1 ) || true
    grep -q '^run_status: completed$' "$directory/$name.out" || {{
        echo "$name did not complete -- see $directory/$name.out" >&2
        tail -n 5 "$directory/$name.out" >&2
        exit 1
    }}
}}

{body}
echo "== done =="
'''


def _system_config(resolved: dict[str, Any], input_path: Path) -> str:
    """System-level metadata. Deliberately small: the parameters live in the scripts and the .out."""
    document = {
        "schema_version": 1,
        "system_id": resolved["system_id"],
        "title": resolved["title"],
        "created": _dt.date.today().isoformat(),
        "source": {
            "kind": "structure" if resolved["type"] == "peptide" else "ligand",
            "identity": Path(input_path).name,
        },
        "generated_by": {
            "tool": "md-openmm setup",
            "md_templates_commit": origin_commit(),
        },
        "protocol": resolved["protocol"],
        "route": resolved["type"],
        "solvent": resolved["solvent"],
    }
    header = ("# System-level metadata. Small on purpose: every simulation parameter is a literal\n"
              "# in the stage scripts and is echoed into each .out, so duplicating them here would\n"
              "# create a second copy to drift.\n")
    return header + yaml.safe_dump(document, sort_keys=False)
