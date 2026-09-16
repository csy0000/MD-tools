"""Rungs serialised at build time, and the record that says how.

THE PREMISE BEING TESTED. Moving REST2 scaling from run time to build time means each rung
becomes a file, so a group line can name `remd<n>/build_state<n>.xml` and the input stops carrying
a scaling instruction. That is only sound if the file IS the Hamiltonian the run-time scaler would
have built -- so the test that matters here deserialises what was written and compares it against
`build_rung_systems` in memory, term by term.

THE HAZARD IT DOES NOT COVER, named so nobody mistakes this for the whole change: a System that
is already scaled must never be scaled again, or solute-solute goes as `(1-tau)^4` instead of
`(1-tau)^2` -- a wrong Hamiltonian producing entirely plausible numbers. That refusal belongs in
the runtime and is a separate change; until it exists, nothing should point `-s` at these files.

PLATFORM_POLICY_EXEMPTION: serialisation and deserialisation only. No Context is created, no
platform is selected, nothing is propagated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools import layout as L

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference"
BUILT = REFERENCE / "ALA-explicit-HMR" / "build"


# -- the run index ------------------------------------------------------------------------------

def test_the_first_run_of_a_method_is_one(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    assert dataset.next_run_index("REST2") == 1
    assert dataset.next_run("REST2").root.name == "REST2-run1"


def test_the_next_index_is_one_past_the_highest(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run2"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1, 2]
    assert dataset.next_run_index("REST2") == 3


def test_a_gap_is_never_reused(tmp_path):
    """THE POINT, not an oversight.

    A gap means a run was moved, archived or registered elsewhere -- `data-register` takes the
    source away and leaves a symlink. Handing its number to a new run would make two different
    experiments share an identity that some manifest already refers to.
    """
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run3"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1, 3]
    assert dataset.next_run_index("REST2") == 4, "2 is a gap and must not be handed out again"


def test_methods_are_counted_separately(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run2", "cMD-run1"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.next_run_index("REST2") == 3
    assert dataset.next_run_index("cMD") == 2
    assert dataset.next_run_index("AIS") == 1


def test_a_non_numeric_suffix_is_ignored_rather_than_fatal(tmp_path):
    """A directory someone named `REST2-run-old` must not block every new run."""
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run-old", "REST2-runX", "REST2-run2.bak"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1]
    assert dataset.next_run_index("REST2") == 2


def test_a_file_named_like_a_run_is_not_counted(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    dataset.root.mkdir(parents=True)
    (dataset.root / "REST2-run7").write_text("not a directory", encoding="utf-8")
    assert dataset.existing_runs("REST2") == []
    assert dataset.next_run_index("REST2") == 1


# -- the rungs themselves -----------------------------------------------------------------------

#: TWO MARKS, APPLIED TOGETHER, and they do different jobs. `reference_data` is what CI DESELECTS:
#: `data/` is gitignored, so on a runner these can never run, and a skip there is indistinguishable
#: from a test that quietly stopped working -- which is why `ci.yml` treats any skip in that lane
#: as a failure. The `skipif` stays for a local checkout that simply has no migrated dataset.
#: Locally, where one exists, they run for real.
#:
#: Composed as a decorator rather than nested: `pytest.mark.reference_data(pytest.mark.skipif(...))`
#: applies the skipif as an ARGUMENT to the marker rather than as a second mark.
def pytestmark_reference(test):
    """Apply both marks to *test*."""
    skip_without_data = pytest.mark.skipif(
        not (BUILT / "built.xml").is_file(),
        reason="needs a built System; data/ is gitignored, so this runs where one was migrated")
    return pytest.mark.reference_data(skip_without_data(test))


@pytest.fixture
def written(tmp_path):
    """Four rungs written from the real ALA-explicit built System."""
    from md_tools.build.rungs import write_rung_systems
    from md_tools.remd.generated import tau_ladder

    dataset = L.DatasetLayout(tmp_path / "ALA-explicit-HMR")
    run = dataset.next_run("REST2")
    taus = tau_ladder(4, 0.5)
    record = write_rung_systems(run, system_path=BUILT / "built.xml",
                                topology_path=BUILT / "built.pdb", taus=taus)
    return run, taus, record


@pytestmark_reference
def test_one_file_per_rung_lands_in_its_own_state_directory(written):
    run, taus, record = written
    for index in range(len(taus)):
        path = run.state_system(index)
        assert path.is_file(), path
        assert path.name == f"build_state{index}.xml"
        assert path.parent.name == f"remd{index}"


@pytestmark_reference
def test_the_written_rung_is_the_system_the_runtime_would_have_built(written):
    """THE CHECK THAT MATTERS: the file must BE the Hamiltonian, not resemble it."""
    from openmm import XmlSerializer

    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import classify_omega_bonds
    from md_tools.remd.protocol import build_rung_systems
    from openmm.app import PDBFile

    run, taus, _ = written
    base = XmlSerializer.deserialize((BUILT / "built.xml").read_text(encoding="utf-8"))
    pdb = PDBFile(str(BUILT / "built.pdb"))
    solute = solute_atom_indices(pdb.topology)
    omega = classify_omega_bonds(pdb.topology, solute, route="peptide", ligand_sdf=None)
    excluded = [tuple(int(a) for a in b) for b in omega.get("omega_unscaled_bonds", [])]
    expected, _audit = build_rung_systems(base, solute, tuple(taus), excluded_bonds=excluded)

    for index, reference in enumerate(expected):
        from_disk = XmlSerializer.deserialize(
            run.state_system(index).read_text(encoding="utf-8"))
        assert XmlSerializer.serialize(from_disk) == XmlSerializer.serialize(reference), (
            f"rung {index} on disk differs from the one the runtime would build")


@pytestmark_reference
def test_the_log_records_every_rung_with_its_factors_and_digest(written):
    run, taus, record = written
    assert record["format"] == "md-tools-rung-scaling/v1"
    assert len(record["states"]) == len(taus)
    for index, state in enumerate(record["states"]):
        assert state["state"] == index
        assert state["tau"] == taus[index]
        assert len(state["sha256"]) == 64
        # The factors, and the expressions that produced them.
        scaling = state["scaling"]
        assert scaling["solute_solute"] == pytest.approx((1 - taus[index]) ** 2)
        assert scaling["solute_environment"] == pytest.approx(1 - taus[index])
        assert scaling["solute_solute_expression"] == "(1 - tau)^2"


@pytestmark_reference
def test_the_recorded_digest_is_the_file_on_disk(written):
    import hashlib

    run, taus, record = written
    for state in record["states"]:
        path = run.state_system(state["state"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == state["sha256"]


@pytestmark_reference
def test_the_tau_ladder_is_taken_not_recomputed(written):
    """A second spelling of the ladder agreeing to six decimals made every four-rung ladder
    unresumable once; the record must hold the values that were used."""
    _run, taus, record = written
    assert record["ladder"]["taus"] == taus
    assert "not recomputed" in record["ladder"]["source"]


@pytestmark_reference
def test_the_omega_exclusion_is_recorded_as_the_torsions_it_protected(written):
    """A stored atom pair needs a force field to mean anything; the torsions are the result."""
    _run, _taus, record = written
    omega = record["omega_exclusion"]
    assert "detector_version" in omega
    assert "excluded_central_bonds" in omega
    assert "excluded_torsion_indices" in omega
    assert omega["n_excluded_torsions"] >= 0
    assert omega["n_scaled_solute_torsions"] >= 0


@pytestmark_reference
def test_the_convention_is_recorded_so_the_scaling_is_readable(written):
    _run, _taus, record = written
    convention = record["convention"]
    assert convention["solute_solute_nonbonded_scale"] == "(1-tau)^2"
    assert convention["solute_environment_nonbonded_scale"] == "1-tau"
    assert convention["ordinary_amide_omega"] == "unscaled"


@pytestmark_reference
def test_a_log_file_is_written_beside_the_run(written):
    run, _taus, record = written
    log = run.root / "build_states.log"
    assert log.is_file()
    import json
    assert json.loads(log.read_text(encoding="utf-8"))["format"] == record["format"]


@pytestmark_reference
def test_rewriting_without_overwrite_is_refused(tmp_path):
    """A rung file silently reused would describe a Hamiltonian this build did not produce."""
    from md_tools.build.rungs import RungWriteError, write_rung_systems
    from md_tools.remd.generated import tau_ladder

    dataset = L.DatasetLayout(tmp_path / "ALA")
    run = dataset.next_run("REST2")
    taus = tau_ladder(2, 0.5)
    arguments = dict(system_path=BUILT / "built.xml", topology_path=BUILT / "built.pdb",
                     taus=taus)
    write_rung_systems(run, **arguments)
    with pytest.raises(RungWriteError, match="already exists"):
        write_rung_systems(run, **arguments)
    write_rung_systems(run, **arguments, overwrite=True)      # explicitly asked for


@pytestmark_reference
def test_a_one_rung_ladder_is_refused(tmp_path):
    from md_tools.build.rungs import RungWriteError, write_rung_systems

    dataset = L.DatasetLayout(tmp_path / "ALA")
    with pytest.raises(RungWriteError, match="at least 2 rungs"):
        write_rung_systems(dataset.next_run("REST2"), system_path=BUILT / "built.xml",
                           topology_path=BUILT / "built.pdb", taus=[0.0])


@pytestmark_reference
def test_the_prose_report_states_the_factors_a_reader_needs(written):
    from md_tools.build.rungs import format_scaling_report

    _run, taus, record = written
    text = format_scaling_report(record)
    assert "rest2-no-bond-angle-omega" in text
    assert "(1-tau)^2" in text
    assert "omega exclusion" in text
    for tau in taus:
        assert f"{tau:.3f}" in text
