"""The throughput measurement benchmarks the coordinates the run is about to integrate.

Placement by measured throughput opens a Context and integrates this run's System for a moment to
learn how fast each card is. It used to integrate `pair.pdb.positions` -- the coordinates inside
`-p`, the TOPOLOGY. Nothing ever relaxes those: `build/built.pdb` is deliberately never rewritten,
so the minimiser's result goes to `min/` and the equilibration's to `eq/`, and the topology keeps
its as-built geometry with every close contact solvation left in it.

A VerletIntegrator conserves energy and so has nothing to dissipate a contact with. Barnase-barstar
(29923 atoms) went from -188350 kJ/mol to `Particle coordinate is NaN` in ten steps, and a 12-rung
ladder was refused before writing anything. Its equilibrated `-c` was -402289 kJ/mol, required,
already validated, and named in the same group-file line.

`-c` must be read WITH its box. An NPT equilibration shrinks the box -- 7.661 -> 7.500 nm here --
and molecules are wrapped for the box they were saved in. Put those positions in the System's
default box and a bond across the boundary is stretched by a whole box length: +7e11 kJ/mol, worse
than the NaN it replaced. The two travel together or not at all.

These run without a GPU: they pin the coordinates the measurement is HANDED, which is where the
defect was. `tests/test_placement_cuda.py` covers the measurement running on real cards.
"""
from __future__ import annotations

import pytest

from md_tools.run.preflight import _measurement_state

openmm = pytest.importorskip("openmm")


def _state_file(tmp_path, *, box_nm: float, offset_nm: float):
    """A serialised State: two particles, a box, and positions that depend on `offset_nm`."""
    from openmm import unit

    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(box_nm, 0, 0),
                                        openmm.Vec3(0, box_nm, 0),
                                        openmm.Vec3(0, 0, box_nm))
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(openmm.Vec3(box_nm, 0, 0), openmm.Vec3(0, box_nm, 0),
                                  openmm.Vec3(0, 0, box_nm))
    context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(offset_nm, 0, 0)] * unit.nanometer)
    state = context.getState(getPositions=True)
    path = tmp_path / "eq_3.xml"
    path.write_text(openmm.XmlSerializer.serialize(state))
    return path


def test_the_restart_supplies_both_its_positions_and_its_box(tmp_path):
    """Both, together. Positions alone are the +7e11 case this exists to prevent."""
    from openmm import unit

    path = _state_file(tmp_path, box_nm=7.5, offset_nm=0.25)

    read = _measurement_state(str(path))
    assert read is not None, "the run's own -c was not read"
    positions, box = read

    assert positions[1][0].value_in_unit(unit.nanometer) == pytest.approx(0.25, abs=1e-6)
    assert box is not None, "positions were returned WITHOUT the box they belong to"
    assert box[0][0].value_in_unit(unit.nanometer) == pytest.approx(7.5, abs=1e-6)


def test_an_absent_or_unreadable_restart_falls_back_rather_than_refusing(tmp_path):
    """The measurement is a placement heuristic; every real check of `-c` happens elsewhere.

    A launch must not be refused because a benchmark could not parse a file -- but it must not
    silently measure garbage either, so the answer is None and the caller keeps its old input.
    """
    assert _measurement_state(None) is None
    assert _measurement_state("") is None
    assert _measurement_state(str(tmp_path / "missing.xml")) is None

    garbage = tmp_path / "not-a-state.xml"
    garbage.write_text("<State>this is not a serialised OpenMM state</State>")
    assert _measurement_state(str(garbage)) is None


def test_a_launch_with_no_restart_relaxes_before_measuring(tmp_path):
    """AIS has no `-c`, and the first version of this fix assumed every multi-rank launch did.

    A ladder's `-c` is required and always present, so reading it fixed ladders completely. AIS
    starts its paths from frames of a source trajectory and carries no restart at all, so
    `_measurement_state` returns None and the measurement falls back to the topology's
    coordinates -- the same unrelaxed `build/built.pdb` that produced the original NaN. A
    multi-rank AIS launch on barnase-barstar then died with exactly the message the ladder fix
    was written to remove:

        rank 0 of 4: AIS: measuring device throughput failed: Particle coordinate is NaN

    The fallback is a short minimisation, and the contract that matters is the one pinned here:
    when there is nothing to read, the caller asks for relaxation rather than benchmarking a
    structure nothing has minimised.
    """
    import inspect

    from md_tools.openmm import placement as placing

    signature = inspect.signature(placing.measure_device_throughput)
    assert "relax_iterations" in signature.parameters, \
        "the measurement cannot relax coordinates it was handed"
    assert signature.parameters["relax_iterations"].default == 0, \
        "relaxation must be opt-in: a caller passing a real -c has nothing to relax"

    source = inspect.getsource(placing.measure_device_throughput)
    assert "LocalEnergyMinimizer" in source, "relax_iterations does not actually minimise"

    # The preflight asks for it exactly when it has no state to read, and not otherwise.
    from md_tools.run import preflight

    plan = inspect.getsource(preflight._plan_placement)
    assert "relax_iterations=relax" in plan
    assert "0 if state is not None else" in plan, \
        "relaxation must be tied to the ABSENCE of -c, not applied unconditionally"
