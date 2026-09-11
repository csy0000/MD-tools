"""`rest2.equilibration_per_tau` on CUDA: one rank per rung, through the generated `run.sh`.

The CPU half -- resolution, the stage plan, the default-off golden files, the driver's stage
order and interruption, a bit-for-bit reconstruction of a rung's end state, the explicit box, the
refused resume and the fault seam -- is `test_rest2_equilibration_per_tau.py`. This file is the
CUDA evidence for `remd/rung_equilibration.py::run_stage`, and so carries no platform exemption.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from .test_rest2_equilibration_per_tau import (IMPLICIT, RUNGS, _build_md, _build_top,
                                               _needs_mpirun, _sha256)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_per_tau_equilibration_under_mpi_on_cuda(tmp_path):
    """Minimisation, then the ladder with per-tau equilibration and `equilibration_steps` on top,
    to completion, every rank on a CUDA device."""
    _needs_mpirun()
    from md_tools.remd.engine import visible_cuda_devices

    if len(visible_cuda_devices(probe=True)) < RUNGS:
        pytest.skip(f"{RUNGS} CUDA devices are needed, one per rung")
    _build_top(tmp_path, "solvent:\n  model: GBn2\n")
    document = json.loads(json.dumps(IMPLICIT))
    document["dynamics"].pop("timestep_fs")           # `auto`, as a user would leave it
    run = _build_md(tmp_path, document, odir="run")
    done = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"], cwd=run,
                          capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["execution"]["platform"] == "CUDA", manifest["execution"]
    record = manifest["per_tau_equilibration"]
    assert [entry["state_index"] for entry in record["states"]] == list(range(RUNGS))
    for entry in record["states"]:
        assert _sha256(run / entry["file"]) == entry["sha256"]
