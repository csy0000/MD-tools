"""One complete ladder plan, built before any output exists and CONSUMED by the driver.

WHAT WAS WRONG

    The preflight built exactly ONE System -- the top rung -- to validate the force layout. The
    driver then called `protocol.build_systems` again at runtime and built all N. So rungs
    0..N-2 were never validated until propagation was about to begin, with `-odir`, both logs,
    `solute.yaml`, `_protocol.py` and the group file already on disk. A force that classifies at
    tau_max and fails at an intermediate rung surfaced attached to a tree that looks exactly like
    a run that started.

    Two constructions of "what System is rung i" is also two answers waiting to disagree, and the
    disagreement would be invisible -- both produce a plausible ladder and only the numbers
    differ. `build_scaled_system`'s `prepare_for_switching` makes that concrete: it changes what
    rung 0 IS, and the preflight's `check_scaling_plan` passes it while the driver's
    `build_systems` does not.

PLATFORM_POLICY_EXEMPTION: no dynamics. These build Systems and inspect the plan; the one test
that runs a ladder does so through the generated script under `--cpu`, because what is under test
is which objects are constructed and when, which is identical on every platform.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"


# --- the one implementation of "what System is rung i" -----------------------------------------

def test_the_protocol_and_the_preflight_build_rungs_through_the_same_function():
    """`Protocol.build_systems` must delegate, not carry its own copy of the construction.

    Source-level, because the property is about there being ONE implementation -- which is not
    observable from behaviour until the day the two copies drift, at which point every test that
    checks behaviour still passes and the ladder is wrong.
    """
    source = (REPO / "src" / "md_tools" / "remd" / "protocol.py").read_text(encoding="utf-8")
    body = source[source.index("def build_systems(self"):]
    body = body[:body.index("\ndef ")] if "\ndef " in body else body
    assert "build_rung_systems(" in body, "build_systems no longer delegates"
    assert "build_scaled_system(" not in body, (
        "build_systems has its own construction again; there must be exactly one")


def test_a_prepared_rung_zero_is_the_one_the_driver_would_have_built():
    """THE subtle one: `prepare_for_switching` changes what rung 0 is.

    `check_scaling_plan` passes it (AIS needs it); `build_rung_systems` must not. If the preflight
    handed the driver a rung 0 built the other way, the ladder's cold replica would silently carry
    a CustomGBForce global parameter the driver never puts there -- a different System, presented
    as the prepared one.
    """
    pytest.importorskip("openmm")
    from openmm import XmlSerializer

    from md_tools.remd.protocol import build_rung_systems
    from md_tools.rest2.scaler import build_scaled_system

    system = _small_system()
    prepared, _audit = build_rung_systems(system, [0, 1], (0.0, 0.5), excluded_bonds=())
    expected = build_scaled_system(system, [0, 1], 0.0, excluded_bonds=())

    assert (XmlSerializer.serialize(prepared[0])
            == XmlSerializer.serialize(expected)), "rung 0 is not the System the driver builds"
    # And the two rungs are genuinely different objects, so the tuple is a ladder and not one
    # System repeated -- which would pass every "length is N" assertion.
    assert (XmlSerializer.serialize(prepared[0])
            != XmlSerializer.serialize(prepared[1]))


def _small_system():
    """Two solute particles and one environment particle, with a scalable nonbonded force."""
    import openmm
    from openmm import unit

    system = openmm.System()
    force = openmm.NonbondedForce()
    for _ in range(3):
        system.addParticle(12.0 * unit.amu)
        force.addParticle(0.1, 0.3, 0.5)
    system.addForce(force)
    return system


def test_every_rung_is_built_not_only_the_top_one():
    """The count is the claim: a plan with one System is not a plan for an N-rung ladder."""
    pytest.importorskip("openmm")
    from md_tools.remd.protocol import build_rung_systems

    taus = (0.0, 0.25, 0.5)
    systems, audit = build_rung_systems(_small_system(), [0, 1], taus)
    assert len(systems) == len(taus)
    # The audit is the COMPLETE one, carrying what the single-tau path never produced.
    assert "omega_exclusion" in audit and "rest2_implementation" in audit


def test_a_pressure_makes_the_whole_plan_refuse():
    """NVT by contract: a barostat would need the pV work the exchange rule does not carry."""
    pytest.importorskip("openmm")
    from md_tools.remd.protocol import ProtocolError, build_rung_systems

    with pytest.raises(ProtocolError, match="NVT"):
        build_rung_systems(_small_system(), [0, 1], (0.0, 0.5), pressure_bar=1.0)


# --- end to end: the plan reaches the driver ---------------------------------------------------

@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ladder-plan")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                   "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                   "production_steps": 5},
        "rest2": {"number_of_replicas": 3, "exchange_interval_steps": 5,
                  "number_of_exchanges": 2},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


@pytest.mark.slow
def test_the_preflight_prepares_one_system_per_rung_before_anything_is_written(project, tmp_path):
    """Called directly, so the plan itself can be inspected rather than its consequences."""
    pytest.importorskip("openmm")
    from md_tools.run.preflight import preflight_ladder

    destination = tmp_path / "planned"
    ladder = {"protocol": "REST2", "n_states": 3, "tau_max": 0.5,
              "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 1,
                           "friction_per_ps": 1.0},
              "exchange_interval_steps": 5, "number_of_exchanges": 2}

    checked = preflight_ladder(
        topology=str(project / "built.pdb"), system=str(project / "built.xml"),
        replicas=3, output=destination / "REST2.out", log=destination / "REST2.log",
        trajectory=destination / "REST2.nc", cpu=True, protocol="REST2",
        timestep_fs=2.0, route="peptide", ladder=ladder, out_dir=destination, tau=0.5)

    assert len(checked.rung_systems) == 3, "the plan does not carry one System per rung"
    assert checked.tau_list == (0.0, 0.25, 0.5), checked.tau_list
    assert checked.solute_indices, "the plan does not carry the solute selection"
    assert not destination.exists(), "the preflight created output"


def test_the_driver_refuses_a_plan_whose_rung_count_disagrees_with_the_ladder():
    """A plan and a protocol describing different ladders must refuse, not silently pick one.

    Calls the driver's own `_rung_systems`, so this exercises the guard rather than restating it.
    """
    pytest.importorskip("openmm")
    from types import SimpleNamespace

    from md_tools.remd.driver import DriverError, ReplicaRun

    run = object.__new__(ReplicaRun)
    run.prepared = SimpleNamespace(rung_systems=(object(), object()), force_audit={})
    run.protocol = SimpleNamespace(n_states=3)

    with pytest.raises(DriverError, match="must be the same ladder"):
        run._rung_systems()


def test_the_driver_returns_the_prepared_systems_unchanged_when_the_counts_agree():
    """Identity, not equality: the plan is consumed, not copied."""
    pytest.importorskip("openmm")
    from types import SimpleNamespace

    from md_tools.remd.driver import ReplicaRun

    rungs = (object(), object(), object())
    run = object.__new__(ReplicaRun)
    run.prepared = SimpleNamespace(rung_systems=rungs, force_audit={"scaled": []})
    run.protocol = SimpleNamespace(n_states=3)

    systems, audit = run._rung_systems()
    assert [s is r for s, r in zip(systems, rungs)] == [True, True, True]
    assert audit == {"scaled": []}


def test_without_a_prepared_plan_the_driver_still_builds_the_ladder_itself():
    """A direct caller that built no plan is not broken by the plan becoming the default path."""
    pytest.importorskip("openmm")
    from types import SimpleNamespace

    from md_tools.remd.driver import ReplicaRun

    built = ([object()], {"scaled": []})
    run = object.__new__(ReplicaRun)
    run.prepared = None
    run.base_system = object()
    run.solute_indices = [0]
    run.excluded_bonds = []
    run.protocol = SimpleNamespace(n_states=1, build_systems=lambda *a, **k: built)

    assert run._rung_systems() == built


@pytest.mark.slow
def test_a_real_ladder_runs_on_the_prepared_systems(project, tmp_path):
    """End to end, with the consumption made observable.

    The driver is asked, at runtime, whether the Systems it holds are the exact objects the
    preflight prepared -- identity, not equality -- which is the difference between consuming the
    plan and rebuilding something that resembles it.
    """
    import os

    probe = project / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "sys.path.insert(0, %r)\n"
        "from md_tools.remd import driver as d\n"
        "seen = {}\n"
        "original = d.ReplicaRun._begin\n"
        "def spy(self, identity, systems, *a, **k):\n"
        "    prepared = tuple(getattr(self.prepared, 'rung_systems', ()) or ())\n"
        "    seen['n'] = len(systems)\n"
        "    seen['same'] = [s is p for s, p in zip(systems, prepared)]\n"
        "    open(os.environ['PROBE_OUT'], 'w').write(json.dumps(seen))\n"
        "    return original(self, identity, systems, *a, **k)\n"
        "d.ReplicaRun._begin = spy\n" % str(REPO / "src"),
        encoding="utf-8")

    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile
    import openmm

    pdb = PDBFile(str(project / "built.pdb"))
    system = XmlSerializer.deserialize((project / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    state = context.getState(getPositions=True, getVelocities=True)
    initial = project / "initial_state.xml"
    initial.write_text(XmlSerializer.serialize(state), encoding="utf-8")

    destination = tmp_path / "run"
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": os.pathsep.join(
            [str(REPO / "src"),
             *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])]),
        "MD_TOOLS_CONFIG": str(user),
        "PROBE_OUT": str(tmp_path / "probe.json"),
        "PYTHONSTARTUP": "",
    })
    done = subprocess.run(
        [sys.executable, "-c",
         f"exec(open({str(probe)!r}).read()); "
         f"import runpy, sys; "
         f"sys.argv = ['REST2.py', '-p', {str(project / 'built.pdb')!r}, "
         f"'-s', {str(project / 'built.xml')!r}, '-c', {str(initial)!r}, "
         f"'-odir', {str(destination)!r}, '--cpu']; "
         f"runpy.run_path({str(project / 'REST2' / 'REST2.py')!r}, run_name='__main__')"],
        cwd=project / "REST2", capture_output=True, text=True, timeout=900, env=environment)

    probe_out = tmp_path / "probe.json"
    assert probe_out.is_file(), done.stdout[-3000:] + done.stderr[-3000:]
    import json

    seen = json.loads(probe_out.read_text(encoding="utf-8"))
    assert seen["n"] == 3, seen
    assert seen["same"] == [True, True, True], (
        "the driver did not propagate the exact Systems the preflight prepared")
