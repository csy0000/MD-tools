"""S4 acceptance row G1: the window runtime on CUDA, not `--cpu`.

The same models and gates as `test_alchemy_windows.py`, with the platform reached only through
the machine configuration and `md_tools.openmm.platform_policy` -- the tests never name a
platform. A machine configuration is written into the test's own temporary directory, so the run
depends on no user file (and nothing near `$MD_DATA`).

This is the CUDA evidence lane for `alchemy/windows.py`: a missing GPU FAILS here, it does not
skip. Run it only on a card allocated to S4, pinned with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
openmm = pytest.importorskip("openmm")

from md_tools.alchemy import estimators as est  # noqa: E402
from md_tools.alchemy.samples import KJ_PER_KCAL, concatenate, window_states  # noqa: E402
from md_tools.alchemy.windows import (WindowError, WindowSettings, read_window_samples,  # noqa: E402
                                      run_window, window_paths)

from tests import test_alchemy_windows as cpu  # noqa: E402
from tests.test_alchemy_windows import model  # noqa: E402,F401  (the fixture)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SHORT = cpu.SHORT


@pytest.fixture(scope="module", autouse=True)
def cuda_present():
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail("OpenMM offers no CUDA platform: an unmet acceptance criterion, not a skip")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("CUDA_VISIBLE_DEVICES is not set: this lane runs only on a card allocated to "
                    "it, pinned explicitly")


@pytest.fixture()
def machine(tmp_path):
    p = tmp_path / "machine.config"
    p.write_text("schema_version: '1.0'\nmachine:\n  openmm:\n    platform: CUDA\n"
                 "    precision: mixed\n    device_policy: local_rank\n")
    return p


def _run(model, machine, wid, settings, out, **kw):
    return run_window(topology=model["pdb"], system=model["xml"], hamiltonian=model["ham"],
                      path=model["path"], states=model["states"], window_id=wid, out_dir=out,
                      settings=settings, cpu=False, machine_config=machine, **kw)


def _record(out, wid):
    return json.loads(window_paths(out, wid)["record"].read_text())


def test_g1_window_runs_on_cuda_and_says_so(model, machine):
    out = model["tmp"] / "g1"
    res = _run(model, machine, "w002", SHORT, out)
    assert res["rows"] == 21
    acc = _record(out, "w002")["acceleration"]
    print(f"\nG1 acceleration record: {json.dumps(acc, sort_keys=True)[:600]}")
    # the key is `resolved_platform` (platform_policy.acceleration_record); the first G1 run
    # asserted a `platform` key that record does not have, and failed on that alone
    assert acc["resolved_platform"] == "CUDA"
    assert acc["platform_selection"] == "machine-config" and acc["explicit_cpu"] is False
    done = json.loads(window_paths(out, "w002")["completion"].read_text())
    assert abs(done["evaluation_self_check_kJ_mol"]) < 1e-3


def test_g1_interrupted_window_resumes_on_cuda(model, machine):
    whole = _run(model, machine, "w001", SHORT, model["tmp"] / "whole")
    assert _run(model, machine, "w001", SHORT, model["tmp"] / "part",
                stop_after_steps=1200)["disposition"] == "interrupted"
    done = _run(model, machine, "w001", SHORT, model["tmp"] / "part")
    assert done["disposition"] == "resumed" and done["rows"] == whole["rows"]
    a = window_paths(model["tmp"] / "whole", "w001")["samples"].read_text().splitlines()
    b = window_paths(model["tmp"] / "part", "w001")["samples"].read_text().splitlines()
    assert [r.split(",")[2] for r in a] == [r.split(",")[2] for r in b]
    # RECORDED, not asserted: whether a CUDA continuation reproduces the uninterrupted stream.
    print(f"\nG1 CUDA resume byte-identical to the uninterrupted run: {a == b}")


def test_g1_refused_continuations_touch_nothing_on_cuda(model, machine):
    out = model["tmp"] / "tamper"
    _run(model, machine, "w003", SHORT, out, stop_after_steps=1200)
    p = window_paths(out, "w003")
    lines = p["samples"].read_text().splitlines()
    fields = lines[2].split(",")
    fields[-1] = repr(float(fields[-1]) + 1.0)
    lines[2] = ",".join(fields)
    p["samples"].write_text("\n".join(lines) + "\n")
    snapshot = {q: q.read_bytes() for q in out.rglob("*") if q.is_file()}
    with pytest.raises(WindowError, match="not the rows the checkpoint committed"):
        _run(model, machine, "w003", SHORT, out)
    other = WindowSettings(steps=2000, report_interval=100, checkpoint_interval=500,
                           equilibration_steps=200, timestep_fs=2.0, seed=99)
    with pytest.raises(WindowError, match="different window definition"):
        _run(model, machine, "w003", other, out)
    assert {q: q.read_bytes() for q in out.rglob("*") if q.is_file()} == snapshot


def test_g1_w8_on_cuda(model, machine):
    states = window_states(model["path"], [k / 16 for k in range(17)], temperature_k=cpu.T)
    m = dict(model, states=states)
    settings = WindowSettings(steps=100_000, report_interval=200, checkpoint_interval=20_000,
                              equilibration_steps=2000, timestep_fs=2.0, seed=8)
    out = model["tmp"] / "w8"
    parts = []
    for st in states:
        _run(m, machine, st.state_id, settings, out)
        parts.append(read_window_samples(window_paths(out, st.state_id)["samples"],
                                         _record(out, st.state_id)))
    result = est.analyze(concatenate(parts))
    exact = cpu.exact_kj_mol() / KJ_PER_KCAL
    verdicts = cpu._verdicts(result, exact)
    print(f"\nG1 W8 on CUDA: exact {exact:.4f} kcal/mol")
    for name, v in verdicts.items():
        e = result["estimates"][name]
        print(f"  {name:12s} {e['delta_g_kcal_mol']:.4f} +- {e['sigma_kcal_mol']:.4f}  "
              f"{v['verdict']}")
    for name, v in verdicts.items():
        assert v["verdict"] == "PASS", (name, v)


def test_g1_npt_on_cuda(tmp_path, machine):
    m = cpu._npt_model(tmp_path)
    settings = WindowSettings(steps=10_000, report_interval=50, checkpoint_interval=2_500,
                              equilibration_steps=1_000, timestep_fs=2.0, seed=5)
    out = tmp_path / "npt"
    parts = []
    for st in m["states"]:
        if st.state_id == "w002":
            assert _run(m, machine, st.state_id, settings, out,
                        stop_after_steps=4_000)["disposition"] == "interrupted"
        _run(m, machine, st.state_id, settings, out)
        parts.append(read_window_samples(window_paths(out, st.state_id)["samples"],
                                         _record(out, st.state_id)))
    samples = concatenate(parts)
    assert np.ptp(samples.volume_nm3) > 0.05
    u = samples.reduced_potential()
    pv = 1.01325 * 0.0602214076 * samples.volume_nm3 / samples.kt
    np.testing.assert_allclose(u, samples.potential_kj_mol / samples.kt + pv[:, None], rtol=1e-12)
    result = est.analyze(samples)
    exact = (1.5 * cpu.kt_kj_mol(cpu.T) * np.log(cpu.K1 / cpu.K0) + cpu.C) / KJ_PER_KCAL
    print(f"\nG1 N1 on CUDA: exact {exact:.4f} kcal/mol")
    for name, e in result["estimates"].items():
        print(f"  {name:12s} {e['delta_g_kcal_mol']:.4f} +- {e['sigma_kcal_mol']:.4f}")
    for name in ("MBAR", "BAR"):
        e = result["estimates"][name]
        gate = est.agreement_gate(e["delta_g_kcal_mol"], e["sigma_kcal_mol"], exact)
        assert gate["verdict"] == "PASS", (name, gate)
