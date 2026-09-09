"""The small Amber-like input files `md-openmm md-run -i` reads.

Deliberately not an Amber parser. It reads namelist-*shaped* sections because that shape is what
someone coming from `pmemd` expects to see, and nothing more:

    &cntrl
      ! minimisation of the built system
      stage            = min,
      minimization_iterations = 1000,
      timestep_fs      = auto,
      temperature_K    = 300.0,
    /

Strict on purpose. A run input is read once, by a machine, and every mistake it can contain is
cheaper to refuse than to discover in the output: an unknown key is a setting that did nothing, a
duplicate is a file that says two things, a `timestep = 2` where `timestep_fs` was meant is a
number in the wrong units. All of them are refused with the key named.

**This parser defines no defaults.** It produces the same resolved model
`md_tools.build.md.resolve_md_config` produces, by projecting the input onto that schema and
resolving it there. Two files that independently decided what "the default timestep" is would
eventually disagree, and the one that lost would be invisible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..build.strict import ConfigError

__all__ = ["RunInput", "parse_run_input", "SECTION_KEYS", "BOOLEAN_WORDS"]

#: Booleans have ONE spelling set, documented here and nowhere else. `yes/no` and `1/0` are
#: deliberately absent: a configuration language with three spellings for true has three ways to
#: typo it.
BOOLEAN_WORDS = {"true": True, "false": False}

#: The sections and the keys each accepts, projected onto the YAML schema that owns the defaults.
#: `(section, key) -> dotted path in the resolved model`.
SECTION_KEYS: dict[str, dict[str, str]] = {
    "cntrl": {
        "protocol": "protocol",
        "solvent": "solvent",
        "stage": "_stage",
        "timestep_fs": "dynamics.timestep_fs",
        "temperature_K": "dynamics.temperature_K",
        "pressure_bar": "dynamics.pressure_bar",
        "friction_per_ps": "dynamics.friction_per_ps",
        "barostat_interval_steps": "dynamics.barostat_interval_steps",
        "restraint_kcal_per_mol_A2": "dynamics.restraint_kcal_per_mol_A2",
        "tau": "dynamics.tau",
        "phase_space_printout": "dynamics.phase_space_printout",
        "random_seed": "dynamics.seed",
        "minimization_iterations": "stages.minimization_iterations",
        "restrained_nvt_steps": "stages.restrained_nvt_steps",
        "restrained_npt_steps": "stages.restrained_npt_steps",
        "unrestrained_npt_steps": "stages.unrestrained_npt_steps",
        "production_steps": "stages.production_steps",
        "solute_printout": "reporting.solute_printout",
        "system_printout": "reporting.system_printout",
        "checkpoint_printout": "reporting.checkpoint_printout",
        # Torsion collective-variable reporting. `cv_file` rather than `file`, because a bare
        # `file` in an &cntrl block reads as "the input file" to anyone who has written an mdin.
        "cv_file": "collective_variables.file",
        "cv_interval_steps": "collective_variables.interval_steps",
    },
    "remd": {
        "number_of_replicas": "rest2.number_of_replicas",
        "tau_max": "rest2.tau_max",
        "exchange_interval_steps": "rest2.exchange_interval_steps",
        "number_of_exchanges": "rest2.number_of_exchanges",
        # Per-state relaxation before the first exchange, at each rung's own Hamiltonian. Spelled
        # out rather than abbreviated: Amber has no counterpart, so there is no established short
        # name to borrow and inventing one would only be a second thing to remember.
        "equilibration_steps": "rest2.equilibration_steps",
        "state_trajectory": "rest2.state_trajectory",
        "rem_log": "rest2.rem_log",
        "neighbour_acceptance_report": "rest2.neighbour_acceptance_report",
        "reservoir_enabled": "reservoir.enabled",
        "reservoir_path": "reservoir.path",
        "refresh_interval_exchanges": "reservoir.refresh_interval_exchanges",
        "reservoir_velocities": "reservoir.velocities",
    },
    "AIS": {
        "number_of_paths": "ais.number_of_paths",
        "tau_start": "ais.tau_start",
        "tau_end": "ais.tau_end",
        "switching_steps": "ais.switching_steps",
        "observation_interval_steps": "ais.observation_interval_steps",
        "parameter_update_interval_steps": "ais.parameter_update_interval_steps",
        # Spelled out. This one decides what the run COSTS and what it can be reweighted with
        # afterwards, and an abbreviation would make the most consequential line in an AIS input
        # the least readable one.
        "work_measurement": "ais.work_measurement",
        "verify_every_updates": "ais.verify_every_updates",
        "source_frame_start": "ais_source.first_frame",
        "source_frame_end": "ais_source.last_frame",
        "source_frame_stride": "ais_source.frame_stride",
        "source_frame_selection": "ais_source.selection",
        "allow_repeated_frames": "ais_source.allow_repeated_frames",
        "source_traj": "ais_source.trajectory",
        "source_topology": "ais_source.topology",
        # accepted in this section too, so an AIS input reads as one block
        "timestep_fs": "dynamics.timestep_fs",
        "temperature_K": "dynamics.temperature_K",
        "friction_per_ps": "dynamics.friction_per_ps",
        "random_seed": "dynamics.seed",
        "solute_printout": "reporting.solute_printout",
        "system_printout": "reporting.system_printout",
        "checkpoint_printout": "reporting.checkpoint_printout",
        # Torsion collective-variable reporting. `cv_file` rather than `file`, because a bare
        # `file` in an &cntrl block reads as "the input file" to anyone who has written an mdin.
        "cv_file": "collective_variables.file",
        "cv_interval_steps": "collective_variables.interval_steps",
    },
}

#: Keys people write when they mean a different one. Named explicitly because a suggestion built
#: from string distance alone would propose `tau` for `tau_max`.
_CONFUSIONS = {
    "timestep": "timestep_fs",
    "dt": "timestep_fs",
    "temperature": "temperature_K",
    "temp": "temperature_K",
    "temp0": "temperature_K",
    "pressure": "pressure_bar",
    "ntpr": "system_printout",
    "ntwx": "solute_printout",
    "ntwr": "checkpoint_printout",
    "nstlim": "production_steps",
    "irest": None,
    "ntb": None,
    "cut": None,
    "gamma_ln": "friction_per_ps",
    "ig": "random_seed",
    "numexchg": "number_of_exchanges",
    "nreplicas": "number_of_replicas",
    "npaths": "number_of_paths",
    # Retired rather than misspelled, and the suggestion says where it went.
    "platform": None,
    "device": None,
}

_SECTION_START = re.compile(r"^\s*&(\w+)\s*$")
_SECTION_END = re.compile(r"^\s*/\s*$")
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*,?\s*$")


@dataclass(frozen=True)
class RunInput:
    """A parsed run input: which stage it names, and the resolved workflow it projects onto."""

    path: Path
    protocol: str
    stage: str | None
    resolved: dict[str, Any]
    sections: tuple[str, ...]


def _coerce(section: str, key: str, raw: str, *, where: str) -> Any:
    """Turn one written value into a typed one, naming the key and the value when it cannot."""
    text = raw.strip().strip(",").strip()
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    lowered = text.lower()
    if lowered in BOOLEAN_WORDS:
        return BOOLEAN_WORDS[lowered]
    if lowered in ("yes", "no", "on", "off", ".true.", ".false."):
        raise ConfigError(
            f"{where}: {key} = {text!r}. Booleans are written {' or '.join(sorted(BOOLEAN_WORDS))} "
            f"-- one spelling set, so there is one way to get it right.")
    if lowered == "auto":
        return "auto"
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
        return float(text)
    if re.fullmatch(r"[A-Za-z][\w./\\-]*", text) or "/" in text or "." in text:
        return text
    raise ConfigError(f"{where}: {key} = {text!r} is not a number, a boolean or a bare word.")


def _suggest(section: str, key: str) -> str:
    """What the writer probably meant, when that can be said without guessing."""
    known = SECTION_KEYS.get(section, {})
    lowered = key.lower()
    if lowered in _CONFUSIONS:
        target = _CONFUSIONS[lowered]
        if target is None:
            if lowered in ("platform", "device"):
                return (f" `{key}` is a property of the MACHINE, not of the experiment, and is "
                        f"retired from protocol configuration. It is machine.openmm.platform "
                        f"(and machine.openmm.device_policy) in the user configuration; `--cpu` "
                        f"and `--device` override it for one run.")
            return (f" `{key}` is an Amber control that has no counterpart here: this runner takes "
                    f"its box, restart and cutoff behaviour from the built System rather than from "
                    f"the run input.")
        if target in known:
            return (f" Did you mean `{target}`? The unit is part of the name here, deliberately: "
                    f"a bare `{key}` cannot say which unit it is in.")
    import difflib

    close = difflib.get_close_matches(key, list(known), n=3, cutoff=0.6)
    if close:
        return f" Did you mean {', '.join(repr(c) for c in close)}?"
    return f" Known keys in &{section}: {', '.join(sorted(known))}."


def parse_run_input(path: str | Path, *, source_trajectory: str | None = None) -> RunInput:
    """Read one Amber-like input and resolve it through the schema that owns the defaults.

    Parsing and every semantic check finish before any caller opens OpenMM or creates an output.
    """
    from ..build.md import resolve_md_config

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"-i {path}: no such run input file")

    values: dict[str, dict[str, Any]] = {}
    seen_sections: list[str] = []
    current: str | None = None

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        where = f"{path}:{number}"
        stripped = line.split("!", 1)[0].split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        start = _SECTION_START.match(stripped)
        if start:
            if current is not None:
                raise ConfigError(f"{where}: &{start.group(1)} opens inside &{current}. "
                                  f"Close the previous section with `/` first.")
            name = start.group(1)
            if name not in SECTION_KEYS:
                raise ConfigError(
                    f"{where}: unknown section &{name}. This runner reads "
                    f"{', '.join('&' + s for s in SECTION_KEYS)}.")
            if name in seen_sections:
                raise ConfigError(
                    f"{where}: &{name} appears twice. A repeated section would silently merge or "
                    f"overwrite, and the losing values would appear nowhere.")
            seen_sections.append(name)
            values[name] = {}
            current = name
            continue
        if _SECTION_END.match(stripped):
            if current is None:
                raise ConfigError(f"{where}: `/` closes a section that was never opened.")
            current = None
            continue
        if current is None:
            raise ConfigError(
                f"{where}: {stripped.strip()!r} is outside any section. Every setting belongs in "
                f"a namelist block such as &cntrl ... /")
        assignment = _ASSIGNMENT.match(stripped)
        if not assignment:
            raise ConfigError(f"{where}: {stripped.strip()!r} is not `key = value`.")
        key, raw = assignment.group(1), assignment.group(2)
        if key not in SECTION_KEYS[current]:
            raise ConfigError(f"{where}: unknown key {key!r} in &{current}.{_suggest(current, key)}")
        if key in values[current]:
            raise ConfigError(
                f"{where}: {key!r} is set twice in &{current}. The first value would be dropped "
                f"without appearing in the resolved configuration, the log or the record.")
        values[current][key] = _coerce(current, key, raw, where=where)

    if current is not None:
        raise ConfigError(f"{path}: &{current} was never closed with `/`.")
    if not seen_sections:
        raise ConfigError(f"{path}: no namelist section found. Expected at least &cntrl.")

    # Project onto the YAML model, which owns every default and every cross-field rule.
    document: dict[str, Any] = {}
    stage = None
    for section, entries in values.items():
        for key, value in entries.items():
            target = SECTION_KEYS[section][key]
            if target == "_stage":
                stage = str(value)
                continue
            if "." in target:
                block, leaf = target.split(".", 1)
                document.setdefault(block, {})[leaf] = value
            else:
                document[target] = value
    if source_trajectory is not None:
        document.setdefault("ais_source", {})["trajectory"] = source_trajectory

    import tempfile

    import yaml

    projected = Path(tempfile.mkdtemp()) / "projected.config"
    projected.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    try:
        resolved = resolve_md_config(projected)
    except ConfigError as invalid:
        # Re-point the message at the file the user actually wrote.
        raise ConfigError(str(invalid).replace(str(projected), str(path))) from None

    return RunInput(path=path, protocol=resolved["protocol"], stage=stage, resolved=resolved,
                    sections=tuple(seen_sections))
