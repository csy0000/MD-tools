#!/usr/bin/env python
"""Shrink the configs in the working directory to smoke sizes.

Used by the release-packaging workflow, which GENERATES a project and checks its tree without
running any dynamics -- its runner has no GPU. Kept as a file rather than a block of YAML pasted
into the workflow so that what CI does and what a developer can run are the same thing.

The same shrink is what the CUDA smoke tests use locally, where the dynamics do run.

    md-openmm sys-config --method cMD REST2 --solvent OPC
    python scripts/shrink_configs_for_ci.py

This makes a SMOKE configuration: a 0.5 nm box, 25 minimisation steps and picoseconds of dynamics.
It is not a scientific setting and nothing produced with it is a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def shrink(folder: Path = Path(".")) -> None:
    system = folder / "sys.config.yaml"
    document = yaml.safe_load(system.read_text())
    # As small as the cutoff allows: what is under test is the plumbing, not a box size.
    document["solvent"]["padding_nm"] = 0.5
    document["solvent"]["cutoff_nm"] = 0.5
    system.write_text(yaml.safe_dump(document, sort_keys=False))

    protocol = folder / "md.config.yaml"
    document = yaml.safe_load(protocol.read_text())
    document["minimization"]["max_iterations"] = 25
    # Shrink every stage the solvent actually has; a null duration means the stage does not apply
    # and must stay null, or the generator will build a directory for it.
    for key, value in list(document["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            document["equilibration"][key] = 0.02
    # Two DIFFERENT trajectory intervals in each method, so one written at the other's interval
    # is visible rather than plausible.
    document["cMD"].update({"duration_ns": 0.0002, "checkpoint_interval_ps": 0.1,
                            "whole_system_interval_ps": 0.1, "solute_interval_ps": 0.05})
    document["REST2"].update({"number_of_replicas": 2, "equilibration_duration_ps": 0.02,
                              "duration_per_segment_ps": 0.02,
                              "number_of_exchanges": 3, "tau_max": 0.05,
                              "checkpoint_interval_ps": 0.02,
                              "whole_system_interval_ps": 0.04, "solute_interval_ps": 0.02})
    protocol.write_text(yaml.safe_dump(document, sort_keys=False))
    print(f"shrunk {system.name} and {protocol.name} to smoke sizes")


if __name__ == "__main__":
    shrink(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
