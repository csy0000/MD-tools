"""A REST2 ladder integrates SAVED scaled states, and neither `build-md` nor the runtime scales.

Step 4 of docs/amber-like-fix/REST2-scaler.md. `build-md` used to scale and serialise rungs into
every run directory (`remd<n>/build_state<n>.xml`), and the grouped runtime then ignored all but
the first and REBUILT every rung from it -- re-classifying torsions from an SDF it looked for beside
`build_state0.xml`, where none ever is. That is why 0.5.3 refused a paracetamol ladder.

Now the states are `build/REST2/system_state<n>.xml` from `md-openmm build-top --rest2-scaler`, the
group file names them one per line, and the preflight hands exactly those Systems to the ladder.
`rest2.number_of_replicas` and `rest2.tau_max` are claims about them.

PLATFORM_POLICY_EXEMPTION: generation and preflight planning only; `cpu=True` so no Context is placed
on a device and nothing is propagated.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .conftest import make_dataset_root, make_scaled_ladder


def _config(root: Path, **rest2) -> Path:
    path = root / "REST2.config"
    path.write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                   "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                   "production_steps": 5},
        "rest2": {"number_of_replicas": 4, "tau_max": 0.5, "exchange_interval_steps": 5,
                  "number_of_exchanges": 2, **rest2},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5},
    }), encoding="utf-8")
    return path


def _generate(root: Path, **rest2):
    from md_tools.build.md import build_scripts

    build_scripts(config_path=_config(root, **rest2), out_dir=root / "REST2-run1", echo=False)
    return root / "REST2-run1"


def test_build_md_refuses_a_ladder_without_saved_states(tmp_path):
    from md_tools.build.md import ConfigError

    make_dataset_root(tmp_path)
    with pytest.raises(ConfigError) as refused:
        _generate(tmp_path)
    assert "build-top --rest2-scaler" in str(refused.value)
    assert not (tmp_path / "REST2-run1").exists()
    assert not (tmp_path / "input").exists()


def test_build_md_names_the_saved_states_and_writes_no_rungs_of_its_own(tmp_path):
    make_dataset_root(tmp_path)
    make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    lines = [line for line in (run / "remd_groupfile.1").read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    assert len(lines) == 4
    for index, line in enumerate(lines):
        assert f"-s ../build/REST2/system_state{index}.xml" in line
        assert f"--group-index {index}" in line
    assert not list(run.rglob("build_state*.xml")), "build-md must not scale rungs of its own"
    assert not (run / "build_states.log").exists()


@pytest.mark.parametrize("rest2, words", [
    ({"number_of_replicas": 3}, "3 states"),
    ({"tau_max": 0.6}, "0.6"),
])
def test_a_configuration_that_does_not_describe_the_states_is_refused(tmp_path, rest2, words):
    from md_tools.build.md import ConfigError

    make_dataset_root(tmp_path)
    make_scaled_ladder(tmp_path)
    with pytest.raises(ConfigError, match=words):
        _generate(tmp_path, **rest2)
    assert not (tmp_path / "REST2-run1").exists()


def _preflight(run: Path, root: Path):
    from md_tools.run.preflight import preflight_ladder

    from .conftest import write_starting_state

    write_starting_state(root, run)

    destination = run / "planned"
    ladder = {"protocol": "REST2", "n_states": 4, "tau_max": 0.5,
              "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 1,
                           "friction_per_ps": 1.0},
              "exchange_interval_steps": 5, "number_of_exchanges": 2}
    return preflight_ladder(
        topology=str(root / "build" / "built.pdb"), system=None,
        groupfile=str(run / "remd_groupfile.1"), replicas=4,
        output=destination / "REST2.out", log=destination / "REST2.log",
        trajectory=destination / "REST2.nc", cpu=True, protocol="REST2",
        timestep_fs=2.0, ladder=ladder, out_dir=destination, tau=0.5), destination


def _torsions(system):
    from openmm import PeriodicTorsionForce, unit

    force = next(f for f in system.getForces() if isinstance(f, PeriodicTorsionForce))
    return [force.getTorsionParameters(i)[6].value_in_unit(unit.kilojoule_per_mole)
            for i in range(force.getNumTorsions())]


def test_the_ladder_integrates_exactly_the_saved_states(tmp_path):
    from openmm import XmlSerializer

    make_dataset_root(tmp_path)
    states = make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    checked, destination = _preflight(run, tmp_path)

    record = yaml.safe_load((states / "scaler.yaml").read_text(encoding="utf-8"))
    assert list(checked.tau_list) == [s["tau"] for s in record["states"]]
    for index, system in enumerate(checked.rung_systems):
        saved = XmlSerializer.deserialize(
            (states / f"system_state{index}.xml").read_text(encoding="utf-8"))
        assert _torsions(system) == _torsions(saved), f"state {index} was scaled again"
    assert sorted(list(b) for b in checked.excluded_bonds) == sorted(
        record["unscaled_torsions"]["unscaled_central_bonds"])
    assert not destination.exists(), "the preflight created output"


def test_solute_yaml_names_the_scaler_record_relative_to_the_run(tmp_path):
    """An absolute path stops resolving when the run tree is moved, and would make `solute.yaml`
    -- which is content-addressed -- differ between two copies of one run."""
    import os

    make_dataset_root(tmp_path)
    states = make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    checked, destination = _preflight(run, tmp_path)
    recorded = checked.notes["solute_document"]["rest2"]["scaled_states"]["record"]
    assert not os.path.isabs(recorded), recorded
    assert (destination / recorded).resolve() == (states / "scaler.yaml").resolve()


def test_the_ladder_classifies_nothing_itself(tmp_path):
    """States built with `unscaled_torsions: false` from a topology the classifier would refuse
    (an unknown residue, no SDF). A runtime that classified again would refuse this ladder."""
    make_dataset_root(tmp_path)
    pdb = tmp_path / "build" / "built.pdb"
    pdb.write_text("".join(line[:17] + "XAA" + line[20:]
                           if line.startswith(("ATOM", "HETATM")) and line[17:20] == "ALA"
                           else line
                           for line in pdb.read_text(encoding="utf-8").splitlines(keepends=True)),
                   encoding="utf-8")
    make_scaled_ladder(tmp_path, extra="unscaled_torsions: false\n")
    run = _generate(tmp_path)
    checked, _destination = _preflight(run, tmp_path)
    assert len(checked.rung_systems) == 4


def test_a_line_naming_another_states_file_is_refused(tmp_path):
    from md_tools.run.preflight import PreflightError

    make_dataset_root(tmp_path)
    make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    group = run / "remd_groupfile.1"
    text = group.read_text(encoding="utf-8")
    group.write_text(text.replace("system_state1.xml", "__S__").replace(
        "system_state2.xml", "system_state1.xml").replace("__S__", "system_state2.xml"),
        encoding="utf-8")
    with pytest.raises((PreflightError, SystemExit)) as refused:
        _preflight(run, tmp_path)
    assert "state" in str(refused.value)


def test_an_odir_that_is_not_the_group_files_protocol_directory_is_refused(tmp_path):
    """The group file's `-i _protocol.py` resolves beside the group file; the ladder writes it into
    `-odir`. Launched with `-odir` elsewhere, the preflight passed and the executor died importing a
    protocol file that was never written. Refused by name now, before anything exists."""
    import subprocess
    import sys

    from .conftest import write_starting_state

    make_dataset_root(tmp_path)
    make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    write_starting_state(tmp_path, run)
    elsewhere = tmp_path / "elsewhere"
    done = subprocess.run(
        [sys.executable, str(run / "REST2.py"), "-p", "../build/built.pdb", "-ng", "4",
         "--groupfile", "remd_groupfile.1", "-odir", str(elsewhere), "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=600)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "_protocol.py" in done.stderr and "Nothing was written" in done.stderr, done.stderr
    assert not elsewhere.exists()
