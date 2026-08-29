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

import os
import datetime as _dt
import random
import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import defaults as D
from .config import ConfigError

#: Where the copied runner and helpers live.
TEMPLATES = Path(__file__).resolve().parent / "templates"

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
    #: The explicit-water model, which selects a COUPLED force-field pair. Ignored under implicit
    #: solvent, which has no water at all.
    #:
    #: TIP3P is the default because it is what this repository's validated defaults have used and
    #: what every existing dataset was generated with; `DEFAULT_SOLVENT` in defaults.py agrees, so
    #: `setup` and the older sys-config path resolve the same force fields. OPC is a supported
    #: selection, not a silent upgrade -- ff19SB and OPC were parameterised together and switching
    #: to them changes the physics of every new system.
    water: str = "TIP3P"
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


def _refuse_mixed_solvent(system_config: dict, *, requested: str) -> None:
    """The protein force field, the water force field and `solvent.model` must agree.

    An `advanced:` override can reach any one of the three. Setting only `forcefield.water` used to
    produce ff14SB protein with OPC water and a `solvent.model` still reading TIP3P -- three
    mutually contradictory statements, silently, in a file that claims to record what ran. The
    resulting System would be built from a combination nobody parameterised.

    So the trio is checked against the supported pairs after overrides are applied, and a
    combination that is not one of them is refused by name.
    """
    forcefield = system_config.get("forcefield") or {}
    solvent = system_config.get("solvent") or {}
    protein, water = forcefield.get("protein"), forcefield.get("water")
    model = solvent.get("model")

    for name, pair in D.EXPLICIT_COMBINATIONS.items():
        if (protein == pair["protein"] and water == pair["water"]
                and D.canonical_solvent(model or name) == name):
            return

    supported = "; ".join(
        f"{name}: protein {pair['protein']}, water {pair['water']}"
        for name, pair in D.EXPLICIT_COMBINATIONS.items())
    raise ConfigError(
        f"the solvent selection is not internally consistent: protein {protein!r}, water "
        f"{water!r}, solvent.model {model!r}. These force fields were parameterised together and "
        f"only the coupled sets are supported -- {supported}. Select one with `water: "
        f"{requested}` in the request rather than overriding a single field."
    )

def _exact_multiple(numerator: float, denominator: float, *, what: str, per: str) -> int:
    """`numerator / denominator` as a whole number, or an error naming both.

    Rejected rather than rounded, for the same reason every other duration in this repository is:
    a rounded interval means the physical time the records claim is not the one simulated.
    """
    if denominator <= 0:
        raise ConfigError(f"{per} must be positive; got {denominator} ps")
    ratio = numerator / denominator
    count = int(round(ratio))
    if abs(ratio - count) > 1e-9:
        raise ConfigError(
            f"{what} ({numerator} ps) must be a whole multiple of {per} ({denominator} ps); "
            f"the ratio is {ratio}.")
    if count < 1:
        raise ConfigError(f"{what} ({numerator} ps) is shorter than {per} ({denominator} ps).")
    return count


def _resolve_rest2(protocol: dict[str, Any], *, production_ps: float, solute_interval_ps: float,
                   timestep_fs: float) -> dict[str, Any]:
    """The REST2 ladder and its intervals, every conversion made exact here rather than at runtime.

    The request's `production` is production PER REPLICA and its `output_interval` is the SOLUTE
    output interval -- the fine one, which is the whole point of the OpenMMTools iteration model.
    The exchange interval and the whole-system interval come from this repository's validated
    REST2 defaults and are overridable through `advanced:` like any other default.
    """
    block = protocol["REST2"]
    # Refused here, at generation, rather than at run time: an unknown scheme would otherwise be
    # discovered by the launcher after a system had been built.
    scheme = str(block.get("replica_mixing_scheme", "swap-all"))
    owner = {"swap-all": "openmmtools", "swap-neighbors": "md-templates"}.get(scheme)
    if owner is None:
        raise ConfigError(
            f"REST2.replica_mixing_scheme must be `swap-all` or `swap-neighbors`, not {scheme!r}. "
            f"`swap-all` is stock OpenMMTools and is the default; `swap-neighbors` needs this "
            f"repository's fix for an upstream defect and makes md-templates the owner of the "
            f"exchange decision.")
    tau_min = float(block["tau_min"])
    tau_max = float(block["tau_max"])
    replicas = int(block["number_of_replicas"])
    if replicas < 2:
        raise ConfigError(f"REST2.number_of_replicas must be at least 2; got {replicas}")
    if not 0.0 <= tau_min < 1.0 or not 0.0 <= tau_max < 1.0:
        raise ConfigError(f"REST2 tau must lie in [0, 1); got {tau_min} to {tau_max}")
    if tau_max <= tau_min:
        raise ConfigError(f"REST2.tau_max ({tau_max}) must exceed tau_min ({tau_min})")

    exchange_ps = float(block["duration_per_segment_ps"])
    whole_ps = float(block["whole_system_interval_ps"])

    # The iteration IS the solute interval, so every other interval is counted in iterations.
    exchange_stride = _exact_multiple(exchange_ps, solute_interval_ps,
                                      what="the REST2 exchange interval",
                                      per="the solute output interval")
    checkpoint_stride = _exact_multiple(whole_ps, solute_interval_ps,
                                        what="the REST2 whole-system output interval",
                                        per="the solute output interval")
    exchanges = _exact_multiple(production_ps, exchange_ps,
                                what="the REST2 production duration per replica",
                                per="the exchange interval")
    steps_per_iteration = _steps(solute_interval_ps, timestep_fs,
                                 field_name="output_interval (the REST2 solute interval)")
    equilibration_ps = float(block["equilibration_duration_ps"])
    if equilibration_ps:
        _exact_multiple(equilibration_ps, solute_interval_ps,
                        what="the REST2 per-tau equilibration duration",
                        per="the solute output interval")

    step = (tau_max - tau_min) / (replicas - 1)
    return {
        "tau_min": tau_min,
        "tau_max": tau_max,
        "number_of_replicas": replicas,
        "taus": [tau_min + step * i for i in range(replicas)],
        "scale_factors": [(1.0 - (tau_min + step * i)) ** 2 for i in range(replicas)],
        "exchange_interval_ps": exchange_ps,
        "solute_interval_ps": solute_interval_ps,
        "whole_interval_ps": whole_ps,
        "equilibration_duration_ps": equilibration_ps,
        "number_of_exchanges": exchanges,
        "exchange_stride_iterations": exchange_stride,
        "checkpoint_interval_iterations": checkpoint_stride,
        "steps_per_iteration": steps_per_iteration,
        "total_iterations": exchanges * exchange_stride,
        "production_per_replica_ps": production_ps,
        "omega_exclusion": bool(block["omega_exclusion"]),
        "enhanced_region": block["enhanced_region"],
        "replica_mixing_scheme": scheme,
        "exchange_decision_owner": owner,
    }


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
    if str(request.protocol).lower() not in ("cmd", "rest2"):
        raise ConfigError(
            f"protocol must be `cMD` or `REST2`, not {request.protocol!r}. AIS keeps its existing "
            f"command and is untouched here.")
    rest2_requested = str(request.protocol).lower() == "rest2"

    implicit = request.solvent == "implicit"
    peptide = request.type == "peptide"

    # ONE user-facing choice selects the whole coupled set. The protein and water force fields
    # were parameterised together -- ff19SB's CMAPs were fit in OPC, ff14SB's in TIP3P -- so
    # `water: OPC` has to move `forcefield.protein`, `forcefield.water` and `solvent.model`
    # together or it means nothing.
    if implicit:
        solvent_name = "GBn2"
    else:
        try:
            solvent_name = D.canonical_solvent(request.water)
        except ValueError:
            solvent_name = str(request.water)
        if solvent_name not in D.EXPLICIT_COMBINATIONS:
            raise ConfigError(
                f"water must be one of {sorted(D.EXPLICIT_COMBINATIONS)}, not {request.water!r}. "
                f"Each selects a coupled protein/water pair; there is no supported way to mix "
                f"them.")

    system_config = D.sys_defaults(peptide=peptide, solvent=solvent_name)
    protocol = D.md_defaults(methods=(("REST2",) if rest2_requested else ("cMD",)),
                             solvent=solvent_name)
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

    if not implicit:
        _refuse_mixed_solvent(system_config, requested=solvent_name)

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

    stage_names = [s["name"] for s in stages] + ["cMD", "REST2", "min"]
    # `advanced.common.random_seed` is honoured rather than accepted and ignored: it becomes the
    # base every stage seed is derived from, and each derived value is printed in the preset and
    # written as a literal into the protocol file.
    seeds = derive_seeds(common.get("random_seed"), stage_names)

    rest2 = _resolve_rest2(protocol, production_ps=production_ps, solute_interval_ps=interval_ps,
                           timestep_fs=timestep_fs) if rest2_requested else None

    return {
        "seeds": seeds,
        "rest2": rest2,
        "water_model": (None if implicit
                        else (system_config.get("solvent") or {}).get("model")),
        "system_id": request.system,
        "title": advanced.get("title") or request.system,
        "type": request.type,
        "solvent": request.solvent,
        "implicit": implicit,
        "protocol": "REST2" if rest2_requested else "cMD",
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
        lines.append(f"force field   {ff.get('protein')} + {ff.get('water')}"
                     f"   (water model {solvent.get('model')})")
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
    rest2 = resolved.get("rest2")
    if rest2:
        ladder = ", ".join(f"{tau:.3f}" for tau in rest2["taus"])
        scales = ", ".join(f"{s:.4f}" for s in rest2["scale_factors"])
        lines += [
            f"REST2 ladder  {rest2['number_of_replicas']} replicas, tau [{ladder}]",
            f"              s = (1-tau)^2 [{scales}]  -- one thermostat at "
            f"{resolved['temperature_K']} K, NOT temperature REMD",
            f"              omega-selective: omega torsions "
            + ("left UNSCALED" if rest2["omega_exclusion"] else "SCALED"),
            f"tau equil     {rest2['equilibration_duration_ps']:.3f} ps per replica "
            f"(not counted as production)",
            f"production    {rest2['production_per_replica_ps']:.3f} ps per replica  "
            f"{rest2['number_of_exchanges']:,d} exchange attempts",
            f"exchange      every {rest2['exchange_interval_ps']:.3f} ps  "
            f"({rest2['exchange_stride_iterations']} iterations)",
            f"solute out    every {rest2['solute_interval_ps']:.3f} ps  "
            f"({rest2['steps_per_iteration']:,d} steps = one iteration)",
            f"whole out     every {rest2['whole_interval_ps']:.3f} ps  "
            f"({rest2['checkpoint_interval_iterations']} iterations)",
            f"engine        openmmtools ReplicaExchangeSampler + MultiStateReporter NetCDF",
        ]
    else:
        lines += [
            f"production    {resolved['production_ps']:.3f} ps  "
            f"{resolved['production_steps']:,d} steps",
            f"output every  {resolved['output_interval_ps']:.3f} ps  "
            f"{resolved['output_interval_steps']:,d} steps  "
            f"({resolved['production_steps'] // resolved['output_interval_steps']:,d} frames)",
        ]
    method = resolved["protocol"]
    lines += [
        f"platform      {resolved['platform']}",
        f"base seed     {resolved['seeds']['base']['seed']}  "
        f"({method} integrator {resolved['seeds'][method]['integrator']}, "
        f"barostat {resolved['seeds'][method]['barostat']})",
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


def _nearest_existing(path: Path) -> Path:
    """The closest ancestor that exists. Git cannot be asked about a directory that is not there.

    The previous check looked only at the immediate parent, so `$REPO/a/b/c` with none of `a/b/c`
    created reported "not a repository" and was accepted -- inside a worktree.
    """
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _inside_git_worktree(path: Path) -> str | None:
    """The worktree containing `path`, or None. Asked of Git, walking up to something that exists."""
    probe = subprocess.run(["git", "-C", str(_nearest_existing(path)), "rev-parse",
                            "--show-toplevel"], capture_output=True, text=True)
    return probe.stdout.strip() if probe.returncode == 0 else None


def resolve_output_root(output: Optional[str]) -> tuple[Path, str]:
    """Where the system goes, and the portable project path relative to `$MD_DATA`.

    A generated system is addressed as `$MD_DATA/<relative project>/<system>` so `paths.sh` can
    contain no machine path. That only works if the output really is beneath `$MD_DATA`, so this
    refuses anything else rather than emitting a map that silently does not apply.
    """
    managed = os.environ.get("MD_DATA")
    if not managed:
        raise ConfigError(
            "MD_DATA is not set. A generated system is addressed relative to the managed storage "
            "root so that paths.sh carries no machine path; without it there is nothing to be "
            "relative to. Export MD_DATA to the storage root and pass --output beneath it.")
    managed_path = Path(managed).expanduser()
    if not managed_path.is_dir():
        raise ConfigError(f"MD_DATA={managed} is not an existing directory")
    managed_path = managed_path.resolve()

    root = Path(output).expanduser().resolve() if output else managed_path
    try:
        relative = root.relative_to(managed_path)
    except ValueError:
        raise ConfigError(
            f"--output {root} is not beneath MD_DATA={managed_path}. The generated paths.sh "
            f"resolves the system from $MD_DATA, so an output elsewhere could not be described "
            f"portably.") from None
    if any(part == ".." for part in relative.parts):
        raise ConfigError(f"--output {root} traverses outside {managed_path}")

    worktree = _inside_git_worktree(root)
    if worktree:
        raise ConfigError(
            f"{root} is inside the Git working tree at {worktree}. Generated systems, "
            f"trajectories and checkpoints must live outside every repository -- an ignore rule is "
            f"not protection, because it is one `git add -f` from being wrong.")

    return root, relative.as_posix()


def refuse_unsafe_output(root: Path) -> None:
    """Kept for callers that only need the worktree question."""
    worktree = _inside_git_worktree(Path(root))
    if worktree:
        raise ConfigError(
            f"{Path(root).resolve()} is inside the Git working tree at {worktree}. Generated "
            f"systems must live outside every repository.")


def resolve_contributor(document: dict, *, interactive: bool) -> dict:
    """Who generated this. Explicit field, then $MD_CONTRIBUTOR, then a prompt, then an error.

    Never invented. An attributed record naming someone who did not do the work is worse than one
    that admits it does not know.
    """
    # Request, then environment, then a prompt if one is allowed, then an error. Never a guess:
    # an attributed record naming someone who did not do the work is worse than one that admits
    # it does not know. There is deliberately no fallback to $USER or the Git config.
    contributor = document.get("contributor") or os.environ.get("MD_CONTRIBUTOR")
    if not contributor and interactive:
        contributor = input("  contributor (name, or name <email>): ").strip() or None
    if not contributor:
        raise ConfigError(
            "no contributor. Set `contributor:` in the setup request, export MD_CONTRIBUTOR, or "
            "run where a prompt is possible. `--yes` never prompts, by design. This is recorded "
            "in config.yaml as who generated the system, and it is not something to guess at.")
    text = str(contributor).strip()
    if "<" in text and text.endswith(">"):
        name, _, email = text.partition("<")
        return {"name": name.strip(), "email": email.rstrip(">").strip()}
    return {"name": text, "email": None}


def derive_seeds(base: Optional[int], stages: list[str]) -> dict[str, dict[str, int]]:
    """One base seed, then a distinct literal seed per stage and per use.

    Resolved here so the seed appears as a number in the protocol file and in the `.out`. A run
    whose seed was chosen at run time cannot be repeated from its own record.
    """
    if base is None:
        base = random.SystemRandom().randrange(1, 2**31 - 1)
    base = int(base)
    seeds: dict[str, dict[str, int]] = {}
    for stage in stages:
        digest = hashlib.sha256(f"{base}:{stage}".encode()).digest()
        seeds[stage] = {
            "integrator": int.from_bytes(digest[0:4], "big") % (2**31 - 1) or 1,
            "barostat": int.from_bytes(digest[4:8], "big") % (2**31 - 1) or 1,
        }
    seeds["base"] = {"seed": base}
    return seeds


def package_version() -> Optional[str]:
    """The installed distribution version, which exists even when there is no Git checkout."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                            # pragma: no cover
        return None
    for name in ("md-templates", "md_templates"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    try:
        from .. import __version__            # type: ignore[attr-defined]
        return str(__version__)
    except Exception:
        return None


def generate(resolved: dict[str, Any], *, input_path: Path, output_root: Path,
             relative_project: str, contributor: dict, overwrite: bool = False,
             echo: bool = True) -> dict[str, Any]:
    """Build the system, then write the protocol files, launchers, path map and runner.

    `sysgen.generate_system` still builds every system, unchanged: force fields, solvation and
    ligand charges are the validated code paths. What this writes is the interface around them.
    """
    from . import emit
    from .sysgen import generate_system

    output_root = Path(output_root).resolve()
    system_dir = output_root / resolved["system_id"]

    if system_dir.exists() and any(system_dir.iterdir()) and not overwrite:
        raise ConfigError(
            f"{system_dir} already exists and is not empty. Generating again would put new stage "
            f"files beside outputs produced by the old ones, leaving a directory whose .out files "
            f"describe runs its scripts no longer perform. Choose another system name, or pass "
            f"--overwrite if the directory is genuinely scratch.")

    inputs = system_dir / "input"
    inputs.mkdir(parents=True, exist_ok=True)

    sys_config_path = inputs / "sys.config.yaml"
    document = dict(resolved["sys_config"])
    document["dataset"] = {"enabled": False}
    sys_config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    generate_system(input_path=Path(input_path), config_path=sys_config_path,
                    output_folder=inputs, echo=echo)

    solute = yaml.safe_load((inputs / "solute.yaml").read_text(encoding="utf-8"))
    if not solute.get("solute_atom_indices_are_contiguous", False):
        raise ConfigError(
            f"{inputs / 'solute.yaml'} reports non-contiguous solute indices, which cannot be "
            f"expressed as a literal range in a restrained stage.")
    resolved = dict(resolved, solute_range=tuple(solute["solute_atom_range"]),
                    provenance_hint="input/provenance.yaml")

    written: list[str] = []

    # The runner. Copied in, so the system needs no installed command.
    binaries = system_dir / "bin"
    binaries.mkdir(exist_ok=True)
    runner = binaries / "openmm-md"
    shutil.copy2(TEMPLATES / "openmm_md.py", runner)
    runner.chmod(0o755)
    written.append("bin/openmm-md")

    rest2 = resolved.get("rest2")
    if rest2:
        # The REST2 ladder has its own file interface, because its outputs are not one trajectory
        # and one restart: they are a multistate NetCDF, a checkpoint NetCDF and a manifest.
        rest2_runner = binaries / "openmm-rest2"
        shutil.copy2(TEMPLATES / "openmm_rest2.py", rest2_runner)
        rest2_runner.chmod(0o755)
        written.append("bin/openmm-rest2")

    stage_names = [s["name"] for s in resolved["equilibration_stages"]] + [resolved["protocol"]]
    (system_dir / "paths.sh").write_text(
        emit.paths_sh(relative_project, resolved["system_id"], stage_names), encoding="utf-8")
    (system_dir / "paths.sh").chmod(0o755)
    written.append("paths.sh")

    launchers: list[str] = []

    (system_dir / "min").mkdir(exist_ok=True)
    (system_dir / "min" / "min.py").write_text(emit.minimization_protocol(resolved),
                                               encoding="utf-8")
    minimisation = emit.launcher("min", directory_var="MIN_DIR", depth=1,
                                 parent_restart="${INPUT_DIR}/initial_state.xml",
                                 trajectory=False, checkpoint=False)
    (system_dir / "min" / "min.sh").write_text(minimisation, encoding="utf-8")
    (system_dir / "min" / "min.sh").chmod(0o755)
    written += ["min/min.py", "min/min.sh"]
    launchers.append("min/min.sh")

    directory_variable = {"nvt_1kcal": "NVT_DIR", "npt_1kcal": "NPT_RESTRAINED_DIR",
                          "npt_free": "NPT_FREE_DIR", "nvt_free": "NVT_FREE_DIR"}
    parent = "${MIN_DIR}/min.state.xml"
    for stage in resolved["equilibration_stages"]:
        name = stage["name"]
        directory = system_dir / "eq" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.py").write_text(emit.equilibration_protocol(resolved, stage),
                                              encoding="utf-8")
        script = emit.launcher(name, directory_var=directory_variable[name], depth=2,
                               parent_restart=parent, trajectory=True, checkpoint=True)
        (directory / f"{name}.sh").write_text(script, encoding="utf-8")
        (directory / f"{name}.sh").chmod(0o755)
        written += [f"eq/{name}/{name}.py", f"eq/{name}/{name}.sh"]
        launchers.append(f"eq/{name}/{name}.sh")
        parent = f"${{{directory_variable[name]}}}/{name}.state.xml"

    if rest2:
        directory = system_dir / "REST2"
        directory.mkdir(exist_ok=True)
        (directory / "rest2.py").write_text(emit.rest2_protocol(resolved), encoding="utf-8")
        (directory / "rest2.sh").write_text(
            emit.rest2_launcher(parent_restart=parent), encoding="utf-8")
        (directory / "rest2.sh").chmod(0o755)
        # The engine and the scaling convention travel WITH the project. `rest2_scaling.py` is the
        # same file cMD and AIS get, so a ladder cannot drift from the fixed-tau walker it is meant
        # to match, and `rest2_openmmtools.py` carries the version pin its overrides depend on.
        for helper in ("rest2_runtime.py", "rest2_openmmtools.py", "rest2_scaling.py",
                       "rest2_statistics.py", "rest2_validate.py"):
            shutil.copy2(TEMPLATES / helper, directory / helper)
            written.append(f"REST2/{helper}")
        written += ["REST2/rest2.py", "REST2/rest2.sh"]
        launchers.append("REST2/rest2.sh")
    else:
        (system_dir / "cMD").mkdir(exist_ok=True)
        (system_dir / "cMD" / "cmd.py").write_text(emit.production_protocol(resolved),
                                                   encoding="utf-8")
        production = emit.launcher("cmd", directory_var="CMD_DIR", depth=1, parent_restart=parent,
                                   trajectory=True, checkpoint=True)
        (system_dir / "cMD" / "cmd.sh").write_text(production, encoding="utf-8")
        (system_dir / "cMD" / "cmd.sh").chmod(0o755)
        written += ["cMD/cmd.py", "cMD/cmd.sh"]
        launchers.append("cMD/cmd.sh")

    (system_dir / "run.sh").write_text(emit.run_all_sh(launchers), encoding="utf-8")
    (system_dir / "run.sh").chmod(0o755)
    written.append("run.sh")

    (system_dir / "config.yaml").write_text(
        _system_config(resolved, input_path, contributor), encoding="utf-8")
    written.append("config.yaml")

    return {"system_dir": system_dir, "written": written}


def _system_config(resolved: dict[str, Any], input_path: Path, contributor: dict) -> str:
    """System-level metadata. Small on purpose: parameters live in the protocols and the .out."""
    commit = origin_commit()
    version = package_version()
    if not commit and not version:
        raise ConfigError(
            "neither an MD-templates commit nor an installed package version could be determined, "
            "so the generated system could not say what produced it.")
    document = {
        "schema_version": 1,
        "system_id": resolved["system_id"],
        "title": resolved["title"],
        "created": _dt.date.today().isoformat(),
        "contributor": contributor,
        "source": {
            "kind": "structure" if resolved["type"] == "peptide" else "ligand",
            "identity": Path(input_path).name,
        },
        "route": resolved["type"],
        "solvent": {
            "mode": resolved["solvent"],
            # The coupled selection, recorded as one fact. `water_model` is the user-facing
            # choice; the two XMLs are what it resolved to, and they move together or not at all.
            "water_model": resolved["water_model"],
            "forcefield_protein": (resolved["sys_config"].get("forcefield") or {}).get("protein"),
            "forcefield_water": (resolved["sys_config"].get("forcefield") or {}).get("water"),
        },
        "protocol": resolved["protocol"],
        "generated_by": {
            "tool": "md-openmm setup",
            "md_templates_version": version,
            # null only when the package version is present: a wheel install legitimately has no
            # checkout, but something concrete must identify the generator.
            "md_templates_commit": commit,
        },
        "random_seed_base": resolved["seeds"]["base"]["seed"],
    }
    header = ("# System metadata. Small on purpose: every simulation parameter is a literal in the\n"
              "# stage protocol files and is echoed into each .out, so duplicating them here would\n"
              "# create a second copy to drift. Force-field detail lives in input/provenance.yaml\n"
              "# and input/resolved_sys.config.yaml.\n")
    return header + yaml.safe_dump(document, sort_keys=False)
