"""`md_tools.rest2` is the one scaler, and `ScalingSelection` is a file that can be checked.

The property worth protecting is structural: fixed-tau cMD, REST2, rREST2 and AIS reach the same
implementation, so a change to the scaling rules cannot arrive in one protocol and miss another.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openmm")

from md_tools.rest2 import REST2Scaler, ScalingSelection, SelectionError   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"


@pytest.fixture(scope="module")
def topology():
    from openmm.app import PDBFile

    return PDBFile(str(ALA)).topology


# --- the public surface -------------------------------------------------------------------------

def test_the_public_names_are_importable():
    import md_tools.rest2 as rest2

    assert {"REST2Scaler", "ScalingSelection"} <= set(rest2.__all__)
    assert callable(rest2.REST2Scaler)


def test_the_scaler_knows_nothing_about_replica_exchange():
    """Three of the four protocols that scale never exchange, so the scaler must not need to.

    Asserted on the module's own source: exchange mathematics lived here and was duplicated in the
    engine, and this is what stops it coming back.
    """
    from md_tools.rest2 import scaler

    text = Path(scaler.__file__).read_text(encoding="utf-8")
    body = text.split('"""', 2)[-1]
    for gone in ("def reduced_potential", "def exchange_log_acceptance", "def exchange_pairs"):
        assert gone not in body, f"{gone} is back in the scaler; it belongs to md_tools.remd"


def test_one_implementation_of_the_exchange_mathematics():
    """It was in two places. The duplicate was removed only after proving they agreed."""
    root = REPO / "src" / "md_tools"
    definitions = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                         if "def exchange_log_acceptance" in p.read_text(encoding="utf-8"))
    assert len(definitions) == 1, definitions


# --- the selection ------------------------------------------------------------------------------

def test_a_derived_selection_names_omega_central_bonds(topology):
    """Central bonds, not torsion indices: several torsions share one peptide C-N bond, and a
    force field may enumerate them differently between builds."""
    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    assert selection.excluded_bonds, "alanine dipeptide has two ordinary amide omegas"
    present = {tuple(sorted((b.atom1.index, b.atom2.index))) for b in topology.bonds()}
    for bond in selection.excluded_bonds:
        assert bond in present
    assert selection.topology_sha256


def test_a_selection_round_trips_through_a_file(topology, tmp_path):
    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    path = selection.write(tmp_path / "solute.yaml")
    again = ScalingSelection.load(path, topology=topology)
    assert again.solute_atoms == selection.solute_atoms
    assert again.excluded_bonds == selection.excluded_bonds
    assert again.topology_sha256 == selection.topology_sha256


def test_a_selection_derived_against_another_topology_is_refused(topology, tmp_path):
    """The failure this prevents is silent: the indices still resolve, they just mean something
    else, and the run produces a plausible trajectory at the wrong Hamiltonian."""
    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    path = tmp_path / "solute.yaml"
    document = selection.to_document()
    document["topology_sha256"] = "0" * 64
    from md_tools.openmm.yaml_io import write_yaml

    write_yaml(path, document)
    with pytest.raises(SelectionError, match="does not match the topology"):
        ScalingSelection.load(path, topology=topology)


def test_an_excluded_bond_that_does_not_exist_is_refused(topology, tmp_path):
    from md_tools.openmm.yaml_io import write_yaml

    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    document = selection.to_document()
    document["unscaled_torsion_central_bonds"].append([0, 21])       # not a bond in ALA
    path = tmp_path / "solute.yaml"
    write_yaml(path, document)
    with pytest.raises(SelectionError, match="do not exist"):
        ScalingSelection.load(path, topology=topology)


def test_a_duplicate_entry_is_refused(topology, tmp_path):
    from md_tools.openmm.yaml_io import write_yaml

    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    document = selection.to_document()
    document["unscaled_torsion_central_bonds"] *= 2
    path = tmp_path / "solute.yaml"
    write_yaml(path, document)
    with pytest.raises(SelectionError, match="duplicates"):
        ScalingSelection.load(path, topology=topology)


def test_a_selection_from_a_different_format_version_is_refused(topology, tmp_path):
    from md_tools.openmm.yaml_io import write_yaml

    document = ScalingSelection.derive(topology, range(topology.getNumAtoms())).to_document()
    document["format"] = "something-else/9.9"
    path = tmp_path / "solute.yaml"
    write_yaml(path, document)
    with pytest.raises(SelectionError, match="format"):
        ScalingSelection.load(path, topology=topology)


# --- the scaler ---------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_an_exclusion_matching_no_torsion_is_refused(tmp_path):
    """Silently ignoring it is the failure: the file would claim a rotation keeps its barrier and
    the run would soften it anyway."""
    import subprocess
    import sys

    import yaml
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    config = tmp_path / "b.config"
    config.write_text(yaml.safe_dump(
        {"solvent": {"model": "GBn2"}}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ALA),
         "-os", "b.xml", "-op", "b.pdb", "-log", "b.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr

    topology = PDBFile(str(tmp_path / "b.pdb")).topology
    system = XmlSerializer.deserialize((tmp_path / "b.xml").read_text(encoding="utf-8"))
    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))

    # A real bond that carries no torsion term: an N-H bond has no torsion centred on it.
    bonded = None
    for bond in topology.bonds():
        pair = tuple(sorted((bond.atom1.index, bond.atom2.index)))
        elements = {bond.atom1.element.symbol, bond.atom2.element.symbol}
        if "H" in elements and pair not in selection.excluded_bonds:
            bonded = pair
            break
    assert bonded, "no X-H bond found to use as a non-torsion central bond"

    invented = ScalingSelection(solute_atoms=selection.solute_atoms,
                                excluded_bonds=selection.excluded_bonds + (bonded,),
                                topology_sha256=selection.topology_sha256)
    with pytest.raises(SelectionError, match="match no torsion"):
        invented.check_matches_torsions(system)


@pytest.mark.slow
@pytest.mark.gpu
def test_the_scaler_is_the_same_object_for_every_protocol(tmp_path):
    """`REST2Scaler` builds a fixed rung and a live switcher from ONE selection, which is what
    makes the shared-scaler claim structural rather than a promise."""
    import subprocess
    import sys

    import yaml
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    config = tmp_path / "b.config"
    config.write_text(yaml.safe_dump({"solvent": {"model": "GBn2"}}), encoding="utf-8")
    assert subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ALA),
         "-os", "b.xml", "-op", "b.pdb", "-log", "b.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800).returncode == 0

    topology = PDBFile(str(tmp_path / "b.pdb")).topology
    base = XmlSerializer.deserialize((tmp_path / "b.xml").read_text(encoding="utf-8"))
    selection = ScalingSelection.derive(topology, range(topology.getNumAtoms()))
    scaler = REST2Scaler(base, selection)

    ladder = scaler.ladder(4, 0.5)
    assert ladder[0] == 0.0 and ladder[-1] == 0.5, ladder
    assert ladder == pytest.approx([0.0, 1/6, 1/3, 0.5]), ladder
    assert scaler.scaling_factors(0.5) == (0.25, 0.5)
    # A fixed rung (REST2, fixed-tau cMD) and a live switcher (AIS) from one selection.
    assert scaler.scaled_system(0.5).getNumParticles() == base.getNumParticles()
    switcher = scaler.switcher()
    assert hasattr(switcher, "set_tau") and hasattr(switcher, "prepared_system")
    # tau is the only persisted coordinate; s and sqrt(s) are derived on demand.
    identity = scaler.identity(0.5, temperature_k=300.0, ensemble="NVT")
    assert identity["tau"] == 0.5
    assert "s" not in identity and "sqrt_s" not in identity
