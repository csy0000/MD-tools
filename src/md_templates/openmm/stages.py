"""The common equilibration stages, and the order they depend on each other in.

One place derives the stage list, the directory names and the parent of every stage. `md-gen`
writes directories from it and the tests assert against it, so a generated tree and the contract
it is supposed to satisfy cannot disagree.

The chain, for explicit solvent:

    inputs -> minimization -> restrained NVT -> restrained NPT -> free NPT
                                                                   |-> cMD production
                                                                   `-> REST2 per-tau equilibration
                                                                         -> REST2 exchange

Under implicit solvent there is no box, so there is no barostat and no NPT: the free stage is NVT.

Each stage reads only the finalized state of its immediate parent. cMD and REST2 are siblings
hanging off the LAST common stage -- neither has to run before the other.
"""
from __future__ import annotations

from typing import Any

#: The restraint strength the default configuration uses, and the folder label that names it.
DEFAULT_RESTRAINT_KCAL = 1.0


def restraint_label(k_kcal_mol_a2: float) -> str:
    """`1kcal` only when the restraint really is 1 kcal/mol/A^2.

    A directory called `eq1_nvt_1kcal` holding a 5 kcal/mol/A^2 run is a lie told by a filename,
    and filenames are what people read months later when the log is gone.
    """
    if float(k_kcal_mol_a2) == DEFAULT_RESTRAINT_KCAL:
        return "1kcal"
    return "restrained"


def stage_plan(config: dict[str, Any], *, implicit: bool) -> list[dict[str, Any]]:
    """The ordered common stages for this configuration.

    Every entry carries what the generated `stage.yaml` needs: what kind of stage it is, how long,
    how strongly restrained, and which file it reads. `input_state` is relative to the stage's own
    directory, so a generated project can be moved anywhere as one piece.
    """
    equilibration = config.get("equilibration") or {}
    minimization = config.get("minimization") or {}
    common = config.get("common") or {}

    restraint_k = float(equilibration.get(
        "restraint_k_kcal_mol_a2", minimization.get("restraint_k_kcal_mol_a2",
                                                    DEFAULT_RESTRAINT_KCAL)))
    label = restraint_label(restraint_k)
    pressure = None if implicit else common.get("pressure_bar")

    plan: list[dict[str, Any]] = [{
        "name": "minimization",
        "group": None,                      # minimisation is not equilibration; it stays at the top
        "kind": "minimization",
        "ensemble": "none",
        "restraint_k_kcal_mol_a2": float(minimization.get("restraint_k_kcal_mol_a2",
                                                          DEFAULT_RESTRAINT_KCAL)),
        "max_iterations": int(minimization.get("max_iterations", 1000)),
        "duration_ps": None,
        "barostat_active": False,           # a barostat during minimisation moves the box against
                                            # forces that are still enormous
    }, {
        "name": f"nvt_{label}",
        "group": "eq",
        "kind": "nvt_restrained",
        "ensemble": "NVT",
        "restraint_k_kcal_mol_a2": restraint_k,
        "max_iterations": None,
        "duration_ps": _duration(equilibration, "nvt_restrained_duration_ps"),
        "barostat_active": False,
    }]

    if implicit:
        plan.append({
            "name": "nvt_free",
            "group": "eq",
            "kind": "nvt_free",
            "ensemble": "NVT",
            "restraint_k_kcal_mol_a2": 0.0,
            "max_iterations": None,
            "duration_ps": _duration(equilibration, "nvt_free_duration_ps"),
            "barostat_active": False,
        })
    else:
        plan.append({
            "name": f"npt_{label}",
            "group": "eq",
            "kind": "npt_restrained",
            "ensemble": "NPT",
            "restraint_k_kcal_mol_a2": restraint_k,
            "max_iterations": None,
            "duration_ps": _duration(equilibration, "npt_restrained_duration_ps"),
            "barostat_active": True,
        })
        plan.append({
            "name": "npt_free",
            "group": "eq",
            "kind": "npt_free",
            "ensemble": "NPT",
            "restraint_k_kcal_mol_a2": 0.0,
            "max_iterations": None,
            "duration_ps": _duration(equilibration, "npt_free_duration_ps"),
            "barostat_active": True,
        })

    # Wire the chain: stage 0 reads the built inputs, every later stage reads its parent's
    # finalized state -- never a checkpoint, which is a mid-stage artifact.
    for index, stage in enumerate(plan):
        stage["implicit"] = bool(implicit)
        stage["output_state"] = "final_state.xml"
        # Where the directory lives under MD/. The equilibration stages are grouped so the project
        # root shows the four things that matter -- minimisation, equilibration, and the two
        # production methods -- instead of a flat list that grows with every stage added.
        stage["path"] = (f"{stage['group']}/{stage['name']}" if stage.get("group")
                         else stage["name"])
        # Under explicit solvent EVERY stage carries a barostat in its System, so a State written
        # by an NPT stage -- which holds the barostat's context parameters -- loads into an NVT
        # stage's Context. `barostat_active` alone decides whether it attempts a move, and
        # `pressure_bar` is the applicable pressure: null when nothing is controlling it.
        stage["system_pressure_bar"] = pressure
        stage["pressure_bar"] = pressure if stage["barostat_active"] else None
        if index == 0:
            stage["parent"] = None
            stage["parent_path"] = None
            stage["input_state"] = None      # md-gen fills this from the inputs folder
        else:
            parent = plan[index - 1]
            stage["parent"] = parent["name"]
            stage["parent_path"] = parent["path"]
            # Relative from THIS stage's directory to the parent's, so the depths of the two
            # groups are accounted for rather than assumed equal.
            stage["input_state"] = _relative_state(stage["path"], parent["path"])
    return plan


def _relative_state(from_path: str, to_path: str) -> str:
    """`../nvt_1kcal/final_state.xml`, or `../../minimization/final_state.xml` across groups.

    Relative to the stage's OWN directory, which is where its run.py executes -- not to the group
    it sits in. Getting that wrong points a grouped stage one level too shallow.
    """
    import posixpath

    return posixpath.join(posixpath.relpath(to_path, from_path), "final_state.xml")


def _duration(equilibration: dict[str, Any], key: str) -> float:
    value = equilibration.get(key)
    if value is None:
        raise ValueError(
            f"equilibration.{key} is null, but this solvent needs that stage. Regenerate the "
            f"protocol with `md-openmm sys-config` or set a duration.")
    return float(value)


def stage_names(config: dict[str, Any], *, implicit: bool) -> list[str]:
    return [stage["name"] for stage in stage_plan(config, implicit=implicit)]


def stage_paths(config: dict[str, Any], *, implicit: bool) -> list[str]:
    """Directory of every common stage, relative to `MD/`."""
    return [stage["path"] for stage in stage_plan(config, implicit=implicit)]


def final_common_stage(config: dict[str, Any], *, implicit: bool) -> str:
    """The directory both production methods branch from, relative to `MD/`."""
    return stage_plan(config, implicit=implicit)[-1]["path"]
