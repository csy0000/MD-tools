"""Selective REST2 ladders on CUDA (0.6.1 S1-G): the lane, ready for when a card is.

A peptide in explicit water with two copies of one ligand and a different ligand
(`selective_rest2_fixture`), run as a three-state ladder for each selection the aims name:
selective backbone, selective sidechains, one ligand instance, and all three combined. For each:

  * the ladder completes on CUDA, and its record says CUDA -- a run that fell back to the CPU
    would otherwise finish and look identical;
  * its Hamiltonian identity is v3 with `selection_mode: explicit`;
  * every recorded u[k][i][w] at the last exchange is the saved state i's energy at walker w's
    stored coordinates, on the platform the ladder ran on, within the calibrated tolerance of
    `test_selective_rest2_integration`, and the whole-solute Hamiltonian is caught.

PLATFORM_POLICY_EXEMPTION: none needed; this propagates on CUDA. `SELECTIVE_REST2_LANE_PLATFORM=CPU`
exists ONLY to check the harness itself on a machine whose cards are reserved; a CPU run is never
CUDA evidence, and the platform assertion below fails any such run that claims otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from .selective_rest2_fixture import RESIDUE, build_fixture
from .test_selective_rest2_integration import (CLI, EXCHANGE_ENERGY_TOLERANCE_KT, REPO,
                                               _env, _run, _saved_states, _whole_solute_states,
                                               exchange_energy_discrepancy)

PLATFORM = os.environ.get("SELECTIVE_REST2_LANE_PLATFORM", "CUDA")
#: `gpu` unless the harness override asked for CPU -- so an override run can never be collected,
#: counted or reported as part of the GPU lane.
pytestmark = [pytest.mark.slow] + ([pytest.mark.gpu] if PLATFORM == "CUDA" else [])
STATES, EXCHANGE_EVERY, EXCHANGES = 3, 50, 4

SELECTIONS = {
    "backbone": {"backbone_scaling_list": f":{RESIDUE['A.SER']}-{RESIDUE['A.PRO']}"},
    "sidechain": {"sidechain_scaling_list": f":{RESIDUE['A.SER']},{RESIDUE['A.PHE']}"},
    "ligand": {"ligand_scaling_dict": {"L01": {"mask": f":{RESIDUE['LGA#1']}"}}},
    "combined": {"backbone_scaling_list": f":{RESIDUE['A.SER']}-{RESIDUE['A.PRO']}",
                 "sidechain_scaling_list": f":{RESIDUE['A.PHE']}",
                 "ligand_scaling_dict": {"L01": {"mask": f":{RESIDUE['LGA#1']}"},
                                         "L02": {"mask": f":{RESIDUE['LGB']}"}}},
}


def _environment(root: Path) -> dict:
    env = _env(root, REPO / "src")
    if PLATFORM == "CUDA":
        # The lane's card, as the conftest assigned it or the caller confined it.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = visible
    return env


def _minimised_start(fixture, root: Path):
    """The fixture is unrelaxed (chain B was translated into place): minimise before dynamics."""
    import openmm
    from openmm import XmlSerializer, unit

    context = openmm.Context(fixture.system, openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds),
        openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(fixture.positions)
    openmm.LocalEnergyMinimizer.minimize(context, 10.0, 2000)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)),
        encoding="utf-8")


@pytest.fixture(scope="module", params=sorted(SELECTIONS))
def ladder(request, tmp_path_factory):
    name = request.param
    root = tmp_path_factory.mktemp(f"cuda-{name}")
    fixture = build_fixture(padding_nm=0.8)
    fixture.write(root / "build")
    env = _environment(root)
    text = (f"method: REST2\nschedule:\n  n_states: {STATES}\n  tau_max: 0.5\n"
            + yaml.safe_dump(SELECTIONS[name]))
    (root / "build" / "scaler.config").write_text(text, encoding="utf-8")
    _run(CLI + ["build-top", "--rest2-scaler", "-s", "build/built.xml", "-p", "build/built.pdb",
                "--config", "build/scaler.config"], cwd=root, env=env)
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 0, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "dynamics": {"seed": 20260919, "timestep_fs": 1.0},
    }), encoding="utf-8")
    _run(CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
         cwd=root, env=env)
    _minimised_start(fixture, root)

    from .conftest import ladder_group_file

    run = root / "run"
    _run([str(root / "REST2-run1" / "REST2.py"), "-p", str(root / "build" / "built.pdb"),
          "--groupfile", str(ladder_group_file(root, run)), "-odir", str(run),
          *(["--cpu"] if PLATFORM == "CPU" else [])],
         cwd=root / "REST2-run1", env=env, timeout=3600)
    return name, root, run


def test_the_ladder_completed_on_the_platform_it_claims(ladder):
    _name, _root, run = ladder
    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    platform = json.dumps(manifest.get("execution"))
    assert f'"{PLATFORM}"' in platform, platform[:2000]


def test_the_identity_is_the_explicit_selection(ladder):
    _name, root, run = ladder
    hamiltonian = json.loads((run / "restart.json").read_text(encoding="utf-8"))[
        "scientific_identity"]["hamiltonian"]
    assert hamiltonian["format"] == "md-tools-hamiltonian-identity/v3"
    assert hamiltonian["selection_mode"] == "explicit"


def test_exchange_energies_are_the_saved_states_energies_at_stored_coordinates(ladder):
    name, root, run = ladder
    absolute, cross, compared = exchange_energy_discrepancy(
        run, _saved_states(root, STATES), platform=PLATFORM)
    print(f"{name} on {PLATFORM}: {compared} values, worst |u| {absolute:.3g} kT, worst "
          f"cross-Hamiltonian {cross:.3g} kT")
    assert absolute < EXCHANGE_ENERGY_TOLERANCE_KT and cross < EXCHANGE_ENERGY_TOLERANCE_KT
    _, wrong, _ = exchange_energy_discrepancy(run, _whole_solute_states(root, STATES),
                                              platform=PLATFORM)
    assert wrong > 20 * EXCHANGE_ENERGY_TOLERANCE_KT, wrong
