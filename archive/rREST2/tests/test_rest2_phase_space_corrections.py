"""rREST2 cases moved out of tests/test_rest2_phase_space_corrections.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""
import pytest

def test_stored_is_the_default_velocity_policy():
    """The probability-one acceptance assumes the recorded momentum is installed unchanged, so
    redrawing must be a decision someone made, not what happens when nothing is said."""
    from md_tools.remd import reservoir as rrest2_reservoir
    assert rrest2_reservoir.DEFAULT_VELOCITY_POLICY == "stored"
    assert set(rrest2_reservoir.VELOCITY_POLICIES) == {"stored", "maxwell"}



def test_an_unknown_velocity_policy_is_refused_rather_than_defaulted(tmp_path):
    from md_tools.remd import reservoir as rrest2_reservoir
    declaration = tmp_path / "reservoir.yaml"
    declaration.write_text(
        "format: md-tools-reservoir-request/v2\nvelocity_policy: whatever\n"
        "weighting: boltzmann\nensemble: NVT\nprepared_directory: reservoir\n"
        "refresh_interval_exchanges: 1\nrandom_seed: 1\n"
        "source:\n  phase_space: nowhere.nc\n  frames: 1\n", encoding="utf-8")
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            declaration, protocol=None, topology_path=None, periodic=False, system=None)
    assert "whatever" in str(caught.value) and "stored" in str(caught.value)
