"""A stage never scales: `dynamics.tau` is a CLAIM about the System it is given.

Step 3 of docs/amber-like-fix/REST2-scaler.md, decided by the user on 2026-09-16: scaling happens in
ONE place, `build-top --rest2-scaler`, and a stage checks what it was given against the record that
place wrote. The rule, for every stage except minimisation:

  * `-s` is a saved scaled state -> nothing is scaled; `dynamics.tau` must equal the state's tau.
    A mismatch is refused, INCLUDING tau = 0 on a tau = 0.5 state -- which is exactly an NPT
    equilibration of a scaled System, and nothing used to notice it.
  * `-s` is not a saved state and tau > 0 -> refused, naming the command that builds the state.
    It used to scale in memory, which made the same Hamiltonian two ways and saved neither.
  * `-s` is not a saved state and tau = 0 -> an ordinary stage, unchanged.
  * a record beside `-s` that does not vouch for it -> refused.

Minimisation scales nothing and claims nothing: it minimises the file it is given, and `min/` is
shared by every method on the system.

PLATFORM_POLICY_EXEMPTION: preflight planning, System construction and file generation. No Context
is created.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import make_dataset_root


def _build(root: Path, *, tau=None, method="cMD") -> Path:
    """`build/` with the ACE-ALA-NME stand-in System, and a saved state at `tau` if given."""
    build = make_dataset_root(root) / "build"
    if tau is not None:
        from md_tools.build.scaler import build_scaled_states

        config = build / "scaler.config"
        config.write_text(f"method: {method}\nschedule:\n  n_states: 1\n"
                          f"  tau_min: {tau}\n  tau_max: {tau}\n", encoding="utf-8")
        build_scaled_states(system_path=build / "built.xml", topology_path=build / "built.pdb",
                            config_path=config, echo=False)
    return build


def _stage(name="prod", tau=0.5, ensemble="NVT"):
    return {"name": name, "tau": tau, "ensemble": ensemble, "seed": 1, "temperature_K": 300.0,
            "pressure_bar": 1.0}


def _prepare(system, topology, stage):
    from md_tools.run.preflight import _prepare_stage, load_inputs

    loaded = load_inputs(topology, system)
    return loaded, _prepare_stage(loaded, stage=stage, name=stage["name"], where=stage["name"])


def _torsion_constants(system):
    from openmm import PeriodicTorsionForce, unit

    force = next(f for f in system.getForces() if isinstance(f, PeriodicTorsionForce))
    return [force.getTorsionParameters(i)[6].value_in_unit(unit.kilojoule_per_mole)
            for i in range(force.getNumTorsions())]


# --- the four cases -----------------------------------------------------------------------------

def test_a_hot_stage_on_the_unscaled_system_is_refused_and_told_what_to_run(tmp_path):
    from md_tools.run.preflight import PreflightError

    build = _build(tmp_path / "ALA")
    with pytest.raises(PreflightError) as refused:
        _prepare(build / "built.xml", build / "built.pdb", _stage(tau=0.5))
    assert "build-top --rest2-scaler" in str(refused.value)


def test_a_hot_stage_on_its_saved_state_scales_nothing(tmp_path):
    build = _build(tmp_path / "ALA", tau=0.5)
    state = build / "cMD" / "system_state0.xml"
    loaded, prepared = _prepare(state, build / "built.pdb", _stage(tau=0.5))
    assert _torsion_constants(prepared["prepared_system"]) == _torsion_constants(loaded.system), (
        "the saved state is already scaled; scaling it again takes solute-solute to (1-tau)^4")


def test_a_claim_that_disagrees_with_the_state_is_refused(tmp_path):
    from md_tools.run.preflight import PreflightError

    build = _build(tmp_path / "ALA", tau=0.5)
    state = build / "cMD" / "system_state0.xml"
    with pytest.raises(PreflightError, match="0.3"):
        _prepare(state, build / "built.pdb", _stage(tau=0.3))


def test_an_unscaled_claim_on_a_scaled_state_is_refused(tmp_path):
    """tau = 0 would let an NPT equilibration run on a scaled System, and nothing refused it.
    (NVT here only because the stand-in System is implicit and would refuse NPT for another
    reason first.)"""
    from md_tools.run.preflight import PreflightError

    build = _build(tmp_path / "ALA", tau=0.5)
    state = build / "cMD" / "system_state0.xml"
    with pytest.raises(PreflightError, match="tau = 0.0"):
        _prepare(state, build / "built.pdb", _stage(tau=0.0))


def test_an_ordinary_stage_is_unchanged(tmp_path):
    build = _build(tmp_path / "ALA")
    loaded, prepared = _prepare(build / "built.xml", build / "built.pdb", _stage(tau=0.0))
    assert _torsion_constants(prepared["prepared_system"]) == _torsion_constants(loaded.system)


def test_a_state_edited_after_the_build_is_refused(tmp_path):
    from md_tools.run.preflight import PreflightError

    build = _build(tmp_path / "ALA", tau=0.5)
    state = build / "cMD" / "system_state0.xml"
    state.write_text(state.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="sha256"):
        _prepare(state, build / "built.pdb", _stage(tau=0.5))


def test_minimisation_scales_nothing_whatever_it_is_told(tmp_path):
    build = _build(tmp_path / "ALA")
    loaded, prepared = _prepare(build / "built.xml", build / "built.pdb",
                                _stage(name="min", tau=0.5))
    assert _torsion_constants(prepared["prepared_system"]) == _torsion_constants(loaded.system)


def test_the_saved_states_exclusions_are_the_ones_recorded(tmp_path):
    """The Hamiltonian identity of a hot stage must name what the state left unscaled."""
    import yaml

    build = _build(tmp_path / "ALA", tau=0.5)
    record = yaml.safe_load((build / "cMD" / "scaler.yaml").read_text(encoding="utf-8"))
    _loaded, prepared = _prepare(build / "cMD" / "system_state0.xml", build / "built.pdb",
                                 _stage(tau=0.5))
    assert sorted(list(b) for b in prepared["excluded_bonds"]) == sorted(
        record["unscaled_torsions"]["unscaled_central_bonds"])


# --- the plan and build-md ------------------------------------------------------------------------

def _resolve(document):
    from md_tools.build.md import MD_SCHEMA

    return MD_SCHEMA.resolve(document)


def test_minimisation_carries_no_tau_in_a_hot_plan():
    from md_tools.build.md import stage_plan

    plan = stage_plan(_resolve({"protocol": "cMD", "solvent": "implicit",
                                "dynamics": {"tau": 0.5}}))
    assert plan[0]["name"] == "min" and float(plan[0]["tau"]) == 0.0
    assert all(float(stage["tau"]) == 0.5 for stage in plan[1:])


def test_build_md_refuses_a_hot_run_without_its_saved_state(tmp_path):
    from md_tools.build.md import ConfigError, build_scripts

    root = tmp_path / "ALA"
    _build(root)
    config = tmp_path / "hot.config"
    config.write_text("protocol: cMD\nsolvent: implicit\ndynamics:\n  tau: 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as refused:
        build_scripts(config_path=config, out_dir=root / "cMD-run1", echo=False)
    assert "build-top --rest2-scaler" in str(refused.value)
    assert "cMD/system_state0.xml" in str(refused.value)
    assert not (root / "cMD-run1").exists()


def test_build_md_points_the_hot_stages_at_the_saved_state_and_min_at_the_built_system(tmp_path):
    from md_tools.build.md import build_scripts

    root = tmp_path / "ALA"
    _build(root, tau=0.5)
    config = tmp_path / "hot.config"
    config.write_text("protocol: cMD\nsolvent: implicit\ndynamics:\n  tau: 0.5\n", encoding="utf-8")
    build_scripts(config_path=config, out_dir=root / "cMD-run1", echo=False)
    run_sh = (root / "cMD-run1" / "run.sh").read_text(encoding="utf-8")

    blocks = run_sh.split('echo "== ')
    min_block = next(b for b in blocks if b.startswith("min =="))
    hot_blocks = [b for b in blocks if b.startswith(("eq_", "cMD =="))]
    assert '-s "${SYSTEM}"' in min_block
    assert hot_blocks and all('-s "${SCALED_SYSTEM}"' in b for b in hot_blocks)
    assert "build/cMD/system_state0.xml" in run_sh
