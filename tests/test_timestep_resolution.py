"""`timestep_fs: auto` is decided by the masses in the System, not by the configuration.

The distinction this file exists for: `build-md` writes run scripts from a configuration and never
opens `built.xml`. A configuration that says hydrogen mass repartitioning was applied is a claim
about a file that was written by a different command at a different time. The masses serialised in
the System are the fact.

The unit tests below use a synthetic two-particle System so they need no build and no GPU. The
end-to-end resolution against real ff14SB/TIP3P Systems -- one repartitioned, one not -- is the
`gpu`-marked test at the bottom, because building them is what makes it real.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.timestep import (AUTO, HMR_TIMESTEP_FS, ORDINARY_TIMESTEP_FS,
                                      heaviest_hydrogen_mass_amu, resolve_timestep_fs)

pytest.importorskip("openmm")


def _system_with_hydrogen(mass_amu: float):
    """A minimal System and Topology holding one carbon and one hydrogen of the given mass."""
    from openmm import System
    from openmm.app import Element, Topology

    system = System()
    system.addParticle(12.011)
    system.addParticle(mass_amu)

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    carbon = topology.addAtom("C", Element.getBySymbol("C"), residue)
    hydrogen = topology.addAtom("H", Element.getBySymbol("H"), residue)
    topology.addBond(carbon, hydrogen)
    return system, topology


def test_the_heaviest_hydrogen_is_what_is_measured():
    """Not the mean. Water is never repartitioned, so a correct HMR system holds both kinds and
    an average would sit between them and answer neither question."""
    system, topology = _system_with_hydrogen(3.024)
    assert heaviest_hydrogen_mass_amu(system, topology) == pytest.approx(3.024)


def test_auto_resolves_to_two_femtoseconds_on_ordinary_masses():
    system, topology = _system_with_hydrogen(1.008)
    resolved = resolve_timestep_fs(AUTO, system, topology)
    assert resolved["timestep_fs"] == ORDINARY_TIMESTEP_FS
    assert resolved["basis"] == "ordinary_masses"
    assert resolved["hmr_detected"] is False


def test_auto_resolves_to_four_femtoseconds_on_repartitioned_masses():
    system, topology = _system_with_hydrogen(3.024)
    resolved = resolve_timestep_fs(AUTO, system, topology)
    assert resolved["timestep_fs"] == HMR_TIMESTEP_FS
    assert resolved["basis"] == "hmr_masses"
    assert resolved["hmr_detected"] is True


def test_an_explicit_four_femtoseconds_without_hmr_is_refused():
    """The combination nothing else catches: 4 fs is reasonable, an unrepartitioned System is
    reasonable, and together they integrate a ~10 fs motion with a 4 fs step and produce a
    trajectory that is wrong rather than a run that fails."""
    system, topology = _system_with_hydrogen(1.008)
    with pytest.raises(SystemExit) as refusal:
        resolve_timestep_fs(4.0, system, topology)
    message = str(refusal.value)
    assert "was NOT applied" in message
    assert "Nothing has been integrated" in message


def test_an_explicit_two_femtoseconds_with_hmr_is_allowed():
    """Slower than it needs to be is not wrong, and refusing it would be this tool overruling a
    deliberate choice."""
    system, topology = _system_with_hydrogen(3.024)
    assert resolve_timestep_fs(2.0, system, topology)["timestep_fs"] == 2.0


def test_a_value_that_is_neither_a_number_nor_auto_is_refused():
    system, topology = _system_with_hydrogen(1.008)
    with pytest.raises(SystemExit, match="auto"):
        resolve_timestep_fs("fast", system, topology)


def test_the_basis_is_recorded_alongside_the_value():
    """A number in a log without its basis cannot be audited: 2 fs from `auto` on an ordinary
    System and 2 fs written by hand on an HMR System are different decisions."""
    system, topology = _system_with_hydrogen(1.008)
    resolved = resolve_timestep_fs(AUTO, system, topology)
    assert set(resolved) == {"timestep_fs", "requested", "basis", "heaviest_hydrogen_amu",
                             "hmr_detected"}
    assert resolved["requested"] == AUTO
    assert resolved["heaviest_hydrogen_amu"] == pytest.approx(1.008)


@pytest.mark.slow
@pytest.mark.gpu
def test_one_generated_script_resolves_two_and_four_fs_from_two_real_systems(tmp_path):
    """The whole point, end to end: the SAME generated stage gives 2 fs or 4 fs depending only on
    which built.xml it is pointed at."""
    import subprocess
    import sys

    import yaml

    from md_tools.build.record import read_record

    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    ala = repo / "tests" / "data" / "ALA.pdb"

    built = {}
    for enabled in (False, True):
        config = tmp_path / f"b{int(enabled)}.config"
        config.write_text(yaml.safe_dump(
            {"solvent": {"padding_nm": 0.5, "cutoff_nm": 0.6},
             "hydrogen_mass_repartitioning": {"enabled": enabled}}), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ala),
             "-os", f"b{int(enabled)}.xml", "-op", f"b{int(enabled)}.pdb",
             "-log", f"b{int(enabled)}.log", "--config", str(config)],
            cwd=tmp_path, capture_output=True, text=True, timeout=1800)
        assert done.returncode == 0, done.stdout + done.stderr
        built[enabled] = (f"b{int(enabled)}.pdb", f"b{int(enabled)}.xml")

    md = tmp_path / "md.config"
    md.write_text(yaml.safe_dump(
        {"protocol": "cMD", "solvent": "explicit",
         "dynamics": {},
         "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                    "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                    "production_steps": 5},
         "reporting": {"crd_printout_solute": 5, "info_printout": 5,
                       "checkpoint_printout": 5}}), encoding="utf-8")
    for enabled, expected, basis in ((False, 2.0, "ordinary_masses"), (True, 4.0, "hmr_masses")):
        # A directory PER SYSTEM. Running the same stage twice against two different Systems in
        # one directory is correctly refused -- the checkpoint fingerprint binds the System -- and
        # that refusal is asserted in its own test rather than worked around here.
        out = f"md{int(enabled)}"
        assert subprocess.run(
            [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", f"./{out}",
             "--config", str(md)], cwd=tmp_path, capture_output=True, text=True).returncode == 0
        pdb, xml = built[enabled]
        done = subprocess.run(
            [sys.executable, "min.py", "-p", f"../{pdb}", "-s", f"../{xml}",
             "-r", "m.xml", "-log", "m.log"],
            cwd=tmp_path / out, capture_output=True, text=True, timeout=1800)
        assert done.returncode == 0, done.stdout + done.stderr
        record = read_record(tmp_path / out / "m.log")["timestep"]
        assert record["timestep_fs"] == expected, record
        assert record["basis"] == basis
        assert record["requested"] == "auto"
