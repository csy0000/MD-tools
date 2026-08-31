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


def _cmd_directory(tau: float) -> str:
    """`cMD` at tau = 0, `cMD_tau0p5` at 0.5. A NAME, never evidence.

    The directory is a convenience for a reader. Nothing downstream infers tau from it: rREST2
    reads the source run's own `resolved_run.yaml`, and refuses when there is none.
    """
    if tau == 0.0:
        return "cMD"
    return "cMD_tau" + f"{tau:g}".replace(".", "p")


def _resolve_replica(protocol: dict[str, Any], *, production_ps: float, solute_interval_ps: float,
                     timestep_fs: float, method: str) -> dict[str, Any]:
    """The ladder and its schedules, every conversion made exact here rather than at run time.

    The request's `production` is production PER REPLICA and its `output_interval` is the SOLUTE
    output interval -- the fine, frequent stream. The exchange, whole-system and checkpoint
    intervals come from this repository's validated REST2 defaults and are independent of it and
    of each other. There is no `segment_ps`: nothing here is a propagation quantum.
    """
    block = protocol["REST2"]
    tau_min = float(block["tau_min"])
    tau_max = float(block["tau_max"])
    states = int(block["number_of_replicas"])
    if states < 2:
        raise ConfigError(f"a ladder needs at least 2 states; got {states}")
    if not 0.0 <= tau_min < 1.0 or not 0.0 <= tau_max < 1.0:
        raise ConfigError(f"REST2 tau must lie in [0, 1); got {tau_min} to {tau_max}")
    if tau_max <= tau_min:
        raise ConfigError(f"REST2.tau_max ({tau_max}) must exceed tau_min ({tau_min})")

    exchange_ps = float(block["exchange_interval_ps"])
    whole_ps = float(block["whole_system_interval_ps"])
    checkpoint_ps = block.get("checkpoint_interval_ps")
    equilibration_ps = float(block["equilibration_duration_ps"])
    exchanges = _exact_multiple(production_ps, exchange_ps,
                                what="the production duration per replica",
                                per="the exchange interval")

    # Every interval must be a whole number of integration steps. Refused here, at generation,
    # rather than at run time: a system should not be built for a protocol that cannot run.
    steps = {}
    for name, value in (("exchange", exchange_ps), ("whole_output", whole_ps),
                        ("solute_output", solute_interval_ps),
                        ("checkpoint", checkpoint_ps if checkpoint_ps else exchange_ps),
                        ("equilibration", equilibration_ps)):
        if value in (None, 0):
            steps[name] = 0
            continue
        steps[name] = _steps(float(value), timestep_fs, field_name=f"REST2 {name} interval")

    step = (tau_max - tau_min) / (states - 1)
    resolved = {
        "method": method,
        "tau": [tau_min + step * i for i in range(states)],
        "scale_factors": [(1.0 - (tau_min + step * i)) ** 2 for i in range(states)],
        "n_states": states,
        "tau_min": tau_min, "tau_max": tau_max,
        "exchange_interval_ps": exchange_ps,
        "whole_output_interval_ps": whole_ps,
        "solute_output_interval_ps": solute_interval_ps,
        "checkpoint_interval_ps": checkpoint_ps,
        "equilibration_ps": equilibration_ps,
        "number_of_exchanges": exchanges,
        "steps": steps,
        "total_steps": exchanges * steps["exchange"],
        "production_per_replica_ps": production_ps,
        "omega_exclusion": bool(block["omega_exclusion"]),
        "enhanced_region": block["enhanced_region"],
        "exchange_rule": ("built-in neighbouring" if method == "REST2"
                          else "generated rrest2_exchange.py"),
    }
    if method == "rREST2":
        resolved["reservoir"] = _resolve_reservoir(
            protocol["rREST2"]["reservoir"], resolved, production_ps)
    return resolved


def _resolve_reservoir(block: dict, replica: dict, production_ps: float) -> dict[str, Any]:
    """The reservoir request. Its SOURCE is named here; what that source IS comes from its record.

    Defaults point at the fixed-tau cMD directory for tau_max, which is the run this repository
    generates for exactly this purpose. The path is a starting point for a reader; `rREST2` reads
    the source's own `resolved_run.yaml` for tau, temperature and frame timing, and refuses when
    there is none.
    """
    tau_max = replica["tau_max"]
    directory = _cmd_directory(tau_max)
    if str(block.get("weighting", "boltzmann")).lower() != "boltzmann":
        raise ConfigError(
            f"rREST2.reservoir.weighting must be `boltzmann`; got "
            f"{block.get('weighting')!r}. A non-Boltzmann reservoir needs its own separately "
            f"derived acceptance rule, and reusing the Boltzmann one would bias every replica.")
    if str(block.get("ensemble", "NVT")).upper() != "NVT":
        raise ConfigError(
            f"rREST2.reservoir.ensemble must be `NVT`; got {block.get('ensemble')!r}. An NPT "
            f"reservoir carries a distribution of volumes and is refused rather than approximated.")
    frames = int(block["frames"])
    if frames < 1:
        raise ConfigError(f"rREST2.reservoir.frames must be >= 1; got {frames}")
    interval = int(block["refresh_interval_exchanges"])
    if interval < 1:
        raise ConfigError(
            f"rREST2.reservoir.refresh_interval_exchanges must be >= 1; got {interval}")
    policy = str(block.get("velocity_policy", "stored")).lower()
    if policy not in ("stored", "maxwell"):
        raise ConfigError(
            f"rREST2.reservoir.velocity_policy must be `stored` or `maxwell`; got {policy!r}. "
            f"`stored` installs the recorded momentum and is what the probability-one rule "
            f"assumes; `maxwell` redraws and must be asked for explicitly.")
    return {
        "format": "md-templates-reservoir-request/v2",
        "phase_space": block.get("phase_space") or f"{directory}/cmd.phase_space.nc",
        "velocity_policy": policy,
        "start_time_ps": float(block.get("start_time_ps") or 0.0),
        "end_time_ps": float(block["end_time_ps"] if block.get("end_time_ps") is not None
                             else production_ps),
        "frames": frames,
        "refresh_interval_exchanges": interval,
        "random_seed": int(block["random_seed"]),
        "source_directory": directory,
        "weighting": "boltzmann",
        "ensemble": "NVT",
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
    protocol_name = {"cmd": "cMD", "rest2": "REST2", "rrest2": "rREST2"}.get(
        str(request.protocol).lower())
    if protocol_name is None:
        raise ConfigError(
            f"protocol must be `cMD`, `REST2` or `rREST2`, not {request.protocol!r}. AIS keeps "
            f"its existing command and is untouched here.")
    replica_requested = protocol_name in ("REST2", "rREST2")

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
    if protocol_name == "rREST2":
        # The fixed-tau source is generated WITH the ladder, not left to a second project the
        # user has to build by hand and hope agrees. Correction: the emitted declaration named a
        # sibling `cMD_tau<max>/` directory that no invocation of this generator could produce,
        # so the default source path could only ever dangle. Co-generating it also makes the
        # same-Hamiltonian contract true by construction -- one input, one force field, one
        # solvation model, one equilibrated box -- rather than something two runs might happen
        # to share.
        methods = ("cMD", "REST2", "rREST2")
    elif protocol_name == "REST2":
        methods = ("REST2",)
    else:
        methods = ("cMD",)
    protocol = D.md_defaults(methods=methods, solvent=solvent_name)
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

    stage_names = [s["name"] for s in stages] + ["cMD", "REST2", "rREST2", "min"]
    # `advanced.common.random_seed` is honoured rather than accepted and ignored: it becomes the
    # base every stage seed is derived from, and each derived value is printed in the preset and
    # written as a literal into the protocol file.
    seeds = derive_seeds(common.get("random_seed"), stage_names)

    replica = _resolve_replica(protocol, production_ps=production_ps,
                               solute_interval_ps=interval_ps, timestep_fs=timestep_fs,
                               method=protocol_name) if replica_requested else None
    # A fixed-tau cMD walker. tau = 0 is ordinary conventional MD and the System is untouched;
    # tau > 0 scales the solute Hamiltonian exactly as the matching REST2 rung does, which is what
    # makes such a run a legitimate reservoir source for rREST2 at that tau.
    # Only meaningful when cMD is the protocol: a REST2 request has no cMD block at all, and
    # validating a default that will never be used refused perfectly good implicit ladders.
    cmd_block = protocol.get("cMD") or {}
    cmd_ensemble = str(cmd_block.get("ensemble", "NVT" if implicit else "NPT")).upper()
    if cmd_block:
        if cmd_ensemble not in ("NVT", "NPT"):
            raise ConfigError(f"cMD.ensemble must be NVT or NPT; got {cmd_ensemble!r}")
        if implicit and cmd_ensemble == "NPT":
            raise ConfigError("implicit solvent has no box, so cMD.ensemble cannot be NPT")
    cmd_tau = float(cmd_block.get("tau", 0.0) or 0.0)
    if not 0.0 <= cmd_tau < 1.0:
        raise ConfigError(f"cMD.tau must lie in [0, 1); got {cmd_tau}")
    cmd_phase_ps = cmd_block.get("phase_space_interval_ps")
    if protocol_name == "rREST2":
        # The source of an rREST2 reservoir is not a free choice. It has to sit on the SAME rung
        # the reservoir refreshes -- same tau, same temperature, same fixed volume -- because the
        # probability-one acceptance is derived from that identity and from nothing else.
        tau_max = replica["tau_max"]
        # `cmd_block["tau"]` always exists -- it comes from the defaults -- so the question is
        # what the USER asked for, not what the baseline left there.
        if "cMD.tau" in advanced and float(advanced["cMD.tau"] or 0.0) != tau_max:
            raise ConfigError(
                f"cMD.tau is {float(advanced['cMD.tau'] or 0.0)} but this rREST2 ladder tops out at "
                f"tau = {tau_max}. The reservoir source must sit on the rung it refreshes; a "
                f"source at another tau is a different Hamiltonian and the probability-one "
                f"acceptance would not hold. Change REST2.tau_max, or drop cMD.tau.")
        cmd_tau = tau_max
        # As with tau: the baseline for explicit solvent is NPT, and refusing a default the user
        # never asked for would make rREST2 unusable in explicit solvent. Refuse only a deliberate
        # request, and otherwise pin the source to the fixed-volume ensemble the ladder runs in.
        if "cMD.ensemble" in advanced and str(advanced["cMD.ensemble"]).upper() != "NVT":
            raise ConfigError(
                f"cMD.ensemble is {str(advanced['cMD.ensemble']).upper()!r}, but an rREST2 "
                f"reservoir source must be NVT. "
                f"An NPT source carries a distribution of volumes, and the reservoir refuses one "
                f"rather than approximating it.")
        cmd_ensemble = "NVT"
        # A source with no phase-space stream is a source with no velocities, and
        # `velocity_policy: stored` would then be unsatisfiable. Default it to the solute stream
        # interval so the run this generator emits can actually feed the reservoir it declares.
        if cmd_phase_ps is None:
            cmd_phase_ps = interval_ps

    return {
        "seeds": seeds,
        "replica": replica,
        "cmd_tau": cmd_tau,
        "cmd_ensemble": cmd_ensemble,
        "cmd_phase_space_interval_ps": cmd_phase_ps,
        "cmd_directory": _cmd_directory(cmd_tau),
        "water_model": (None if implicit
                        else (system_config.get("solvent") or {}).get("model")),
        "system_id": request.system,
        "title": advanced.get("title") or request.system,
        "type": request.type,
        "solvent": request.solvent,
        "implicit": implicit,
        "protocol": protocol_name,
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
    replica = resolved.get("replica")
    if replica:
        ladder = ", ".join(f"{tau:.3f}" for tau in replica["tau"])
        scales = ", ".join(f"{s:.4f}" for s in replica["scale_factors"])
        lines += [
            f"{replica['method']} ladder  {replica['n_states']} states, tau [{ladder}]",
            f"              solute-solute (1-tau)^2 [{scales}]  -- one thermostat at "
            f"{resolved['temperature_K']} K, NOT temperature REMD",
            "              omega-selective: omega torsions "
            + ("left UNSCALED" if replica["omega_exclusion"] else "SCALED"),
            f"equilibration {replica['equilibration_ps']:.3f} ps per state "
            f"(not counted as production)",
            f"production    {replica['production_per_replica_ps']:.3f} ps per replica  "
            f"{replica['number_of_exchanges']:,d} exchange attempts",
            f"exchange      every {replica['exchange_interval_ps']:.3f} ps  "
            f"({replica['steps']['exchange']:,d} steps)",
            f"whole out     every {replica['whole_output_interval_ps']:.3f} ps  "
            f"({replica['steps']['whole_output']:,d} steps)",
            f"solute out    every {replica['solute_output_interval_ps']:.3f} ps  "
            f"({replica['steps']['solute_output']:,d} steps)  -- an independent stream",
            f"checkpoint    every {replica['steps']['checkpoint']:,d} steps",
            f"total         {replica['total_steps']:,d} steps "
            f"({replica['production_per_replica_ps']:.3f} ps per replica)",
            f"exchange rule {replica['exchange_rule']}",
            "executor      openmm-md -ng N --groupfile ...  (owned runtime, owned NetCDF)",
        ]
        if replica.get("reservoir"):
            r = replica["reservoir"]
            lines += [
                f"reservoir     Boltzmann phase space, {r['frames']} sample(s) from "
                f"{r['phase_space']}",
                f"              velocity_policy: {r['velocity_policy']}"
                + ("  (the recorded momentum is installed unchanged)"
                   if r["velocity_policy"] == "stored"
                   else "  (momenta REDRAWN -- asked for explicitly)"),
                f"              refresh every {r['refresh_interval_exchanges']} exchange(s) at "
                f"tau_max; the Hamiltonian fingerprint, tau and temperature come from the "
                f"source's own record",
            ]
    else:
        lines += [
            f"production    {resolved['production_ps']:.3f} ps  "
            f"{resolved['production_steps']:,d} steps"
            + (f"  at fixed tau = {resolved['cmd_tau']:g}" if resolved.get("cmd_tau") else ""),
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
    if worktree and not _staging_opt_in(root):
        raise ConfigError(
            f"{root} is inside the Git working tree at {worktree}. Generated systems, "
            f"trajectories and checkpoints must live outside every repository -- an ignore rule is "
            f"not protection, because it is one `git add -f` from being wrong.\n"
            f"  A project-local STAGING area, the kind `md-data-register` later moves out, is the "
            f"one exception: create {root}/.md-staging, make sure `git check-ignore` agrees "
            f"that paths beneath it are ignored, and set MD_TEMPLATES_ALLOW_STAGING=1.")
    if worktree:
        print(f"# staging            : {root} is inside {worktree} and is a declared staging "
              f"area. It is git-ignored and MUST be registered out before it is cited.")

    return root, relative.as_posix()


def _staging_opt_in(root: Path) -> bool:
    """Is this an explicitly declared, genuinely ignored staging area?

    The worktree rule below exists because "it is in .gitignore" is not protection: an ignore rule
    is one `git add -f` from being wrong. That reasoning is sound and it stays. What it did not
    allow for is the staging step the registration pipeline is built on -- a dataset is generated
    project-locally, finished, verified, and only then moved under $MD_DATA by `md-data-register`.

    So the exception is narrow and it is CHECKED rather than asserted. All three must hold:

      * the caller opted in for this run, by setting MD_TEMPLATES_ALLOW_STAGING=1;
      * a marker file `.md-staging` sits in the directory, so the intent is visible to anyone who
        looks at the tree and does not depend on the environment of whoever ran the command;
      * `git check-ignore` agrees that what is created beneath the root is actually ignored --
        asking git rather than trusting that a rule was written correctly. The root itself is
        often tracked (it holds a README); what must be ignored is where the data lands.

    A directory that is merely inside a repository still fails, which is the case the rule is for.
    """
    if os.environ.get("MD_TEMPLATES_ALLOW_STAGING") != "1":
        return False
    if not (root / ".md-staging").is_file():
        return False
    # What must be ignored is where the DATA lands -- the system directories created beneath the
    # root -- not the root itself, which is often a tracked directory holding a README. Asking
    # about a child is asking the question that matters. `check-ignore` matches paths, so the
    # probe does not need to exist.
    probe = root / "any-generated-system"
    try:
        result = subprocess.run(["git", "check-ignore", "--quiet", str(probe)],
                                cwd=str(root), capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


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

    replica = resolved.get("replica")

    # Which directory needs an extra `*_DIR` export beyond the ones paths.sh always writes.
    # For rREST2 that is the co-generated fixed-tau source, whose name follows tau_max.
    production_directory = (resolved["cmd_directory"]
                            if resolved["protocol"] == "rREST2" or not resolved.get("replica")
                            else resolved["protocol"])
    stage_names = [s["name"] for s in resolved["equilibration_stages"]] + [resolved["protocol"]]
    (system_dir / "paths.sh").write_text(
        emit.paths_sh(relative_project, resolved["system_id"], stage_names,
                      production_dir=production_directory), encoding="utf-8")
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
        # The group file is read after the launcher has cd'd to SYSTEM_ROOT, so its paths are
        # relative to the system root rather than to the shell variable map.
        parent_relative = f"eq/{name}/{name}.state.xml"

    def write_cmd_stage(parent_restart: str) -> str:
        """The fixed-tau (or unscaled) single-walker stage. Returns its launcher path."""
        name = resolved["cmd_directory"]
        directory = system_dir / name
        directory.mkdir(exist_ok=True)
        (directory / "cmd.py").write_text(emit.production_protocol(resolved), encoding="utf-8")
        # The companion runtime record is what makes this trajectory usable as an AIS source or an
        # rREST2 reservoir, so every cMD run writes one.
        shutil.copy2(TEMPLATES / "runtime_record.py", directory / "runtime_record.py")
        written.append(f"{name}/runtime_record.py")
        if resolved.get("cmd_tau"):
            # A fixed-tau walker scales through the same module the ladder uses, so it needs it.
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            written.append(f"{name}/rest2_scaling.py")
        if resolved.get("cmd_phase_space_interval_ps"):
            # A reservoir source records positions AND velocities AND box, and the Hamiltonian
            # fingerprint the reservoir validator will recompute and compare.
            for helper in ("phase_space.py", "hamiltonian_identity.py"):
                shutil.copy2(TEMPLATES / helper, directory / helper)
                written.append(f"{name}/{helper}")
        production = emit.launcher("cmd", directory_var=emit.cmd_directory_variable(name),
                                   depth=1, parent_restart=parent_restart, trajectory=True,
                                   checkpoint=True)
        (directory / "cmd.sh").write_text(production, encoding="utf-8")
        (directory / "cmd.sh").chmod(0o755)
        written.extend([f"{name}/cmd.py", f"{name}/cmd.sh"])
        return f"{name}/cmd.sh"

    if replica:
        method = resolved["protocol"]
        stem = "rest2" if method == "REST2" else "rrest2"
        directory = system_dir / method
        directory.mkdir(exist_ok=True)
        directory_var = "REST2_DIR" if method == "REST2" else "RREST2_DIR"

        (directory / f"{stem}.py").write_text(
            emit.replica_protocol_file(resolved, method=method), encoding="utf-8")
        (directory / f"{stem}.group").write_text(
            emit.replica_group_file(resolved, method=method, protocol_name=f"{stem}.py",
                                    parent_state=parent_relative), encoding="utf-8")
        rule = reservoir = None
        # Every module `replica_driver` imports must travel with the generated project: it runs
        # from this directory with nothing but these files on the path, so a helper left out here
        # is an ImportError at launch, not a missing feature.
        helpers = ["replica_runtime.py", "replica_protocol.py", "replica_schedule.py",
                   "replica_engine.py", "replica_driver.py", "replica_storage.py",
                   "replica_statistics.py", "replica_validate.py", "exchange_rules.py",
                   "rest2_scaling.py", "hamiltonian_identity.py",
                   "rem_log.py", "amber_trajectory.py", "state_trajectories.py"]
        if method == "rREST2":
            rule, reservoir = "rrest2_exchange.py", "reservoir.yaml"
            (directory / "reservoir.yaml").write_text(
                emit.reservoir_declaration(resolved), encoding="utf-8")
            written.append(f"{method}/reservoir.yaml")
            # The reservoir reader and the shared source-ensemble rules travel with the project.
            helpers += ["rrest2_exchange.py", "rrest2_reservoir.py", "source_ensemble.py",
                        "phase_space.py"]

        (directory / f"{stem}.sh").write_text(
            emit.replica_launcher(resolved, method=method, directory_var=directory_var,
                                  protocol_name=f"{stem}.py", stem=stem,
                                  exchange_rule=rule, reservoir=reservoir), encoding="utf-8")
        (directory / f"{stem}.sh").chmod(0o755)
        (directory / f"{stem}_extend.sh").write_text(
            emit.replica_extension_launcher(resolved, method=method,
                                            directory_var=directory_var, stem=stem,
                                            exchange_rule=rule, reservoir=reservoir),
            encoding="utf-8")
        (directory / f"{stem}_extend.sh").chmod(0o755)
        written.append(f"{method}/{stem}_extend.sh")
        if method == "rREST2":
            # Emitted BEFORE the ladder launcher so `run.sh` fills the reservoir source before
            # anything tries to read it.
            launchers.append(write_cmd_stage(parent))
        for helper in helpers:
            shutil.copy2(TEMPLATES / helper, directory / helper)
            written.append(f"{method}/{helper}")
        written += [f"{method}/{stem}.py", f"{method}/{stem}.group", f"{method}/{stem}.sh"]
        launchers.append(f"{method}/{stem}.sh")
    else:
        launchers.append(write_cmd_stage(parent))


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
