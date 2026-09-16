"""An amide omega nobody could classify is REFUSED by every surface that scales torsions.

`classify_omega_bonds` has always said "a non-empty unclassified list must block production", and
until this file only ONE of its production consumers acted on it: the ladder preflight, through
`remd.generated.solute_document`. Every other surface took `omega_unscaled_bonds` and ignored the
rest -- so a candidate the classifier could not name was simply left out of the exclusions and
SCALED, like any other solute torsion:

  * `build/rungs.write_rung_systems`, which serialises the rungs a grouped ladder actually
    integrates -- and which `build-md` called WITHOUT the `built.sdf` beside `built.xml`, so every
    amide of a `peptide-like` or `ligand` solute arrived here unclassified;
  * the fixed-tau cMD stage preflight and the AIS preflight;
  * the REST2 reference export;
  * `ScalingSelection.derive`, which did not even get as far as scaling: it iterated a candidate
    DICT as if it were an atom pair and died on `int('bond')`.

The run completes, the acceptance ratios look plausible, and the ordinary-amide invariant is
broken with nothing saying so. That is the defect. `omega_exclusions` is now the one enforcing
entry point, and the last test here keeps it that way.

THE FIXTURE. `conftest.make_dataset_root`'s ACE-ALA-NME with its alanine renamed `XAA`, and no
SDF. The ACE->XAA omega then has no residue evidence and no bond-order evidence, which is exactly
the state a modified residue -- or a whole-molecule solute whose SDF was not passed -- produces. The
System is untouched: the classifier reads the topology only. It is built from `tests/data/`, NOT
from the gitignored `data/reference/` this file first used: there, every test but one skipped on a
clean checkout, which CI's `--error-on-skip` lane would have reported as a failure and a laxer lane
as nothing at all.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping, System serialisation and preflight refusals. No
Context is created and nothing is propagated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from .conftest import make_dataset_root

REPO = Path(__file__).resolve().parents[1]


def _classifiable(root: Path) -> Path:
    """`build/` holding the ACE-ALA-NME System and PDB `make_dataset_root` builds (implicit)."""
    return make_dataset_root(root) / "build"


def _unclassifiable(root: Path) -> Path:
    """The same `build/`, its alanine renamed `XAA` in the PDB, and deliberately no `built.sdf`."""
    build = _classifiable(root)
    lines = (build / "built.pdb").read_text(encoding="utf-8").splitlines(keepends=True)
    renamed = [line[:17] + "XAA" + line[20:]
               if line.startswith(("ATOM", "HETATM")) and line[17:20] == "ALA" else line
               for line in lines]
    assert renamed != lines, "the fixture PDB no longer has the residue this fixture renames"
    (build / "built.pdb").write_text("".join(renamed), encoding="utf-8")
    return build


@pytest.fixture
def unclassifiable(tmp_path):
    return _unclassifiable(tmp_path / "XAA")


def _loaded(directory):
    from md_tools.run.preflight import load_inputs

    return load_inputs(directory / "built.pdb", directory / "built.xml")


# --- the one entry point -------------------------------------------------------------------------

def test_the_enforcing_entry_point_refuses_and_names_the_candidate(unclassifiable):
    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import UnclassifiedOmegaError, omega_exclusions

    loaded = _loaded(unclassifiable)
    with pytest.raises(UnclassifiedOmegaError) as refused:
        omega_exclusions(loaded.pdb.topology, solute_atom_indices(loaded.pdb.topology))
    message = str(refused.value)
    assert "XAA" in message, message
    assert "SDF" in message, message
    assert "4-6" in message or "(4, 6)" in message or "4, 6" in message, (
        "the refusal must name the bond, or a reader cannot find it")


def test_the_refusal_does_not_offer_a_setting_nobody_can_set(unclassifiable):
    """`rest2.proline_like_residues` is not a key of any user configuration: `_legacy_cfg`
    rebuilds `rest2` from DEFAULTS and copies nothing into it. Naming it as the remedy sends a
    reader to write a key that `build-top.config` refuses as unknown."""
    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import UnclassifiedOmegaError, omega_exclusions

    loaded = _loaded(unclassifiable)
    with pytest.raises(UnclassifiedOmegaError) as refused:
        omega_exclusions(loaded.pdb.topology, solute_atom_indices(loaded.pdb.topology))
    assert "proline_like_residues" not in str(refused.value), str(refused.value)


def test_a_classifiable_solute_passes_through_unchanged(tmp_path):
    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import classify_omega_bonds, omega_exclusions

    loaded = _loaded(_classifiable(tmp_path / "ALA"))
    solute = solute_atom_indices(loaded.pdb.topology)
    enforced = omega_exclusions(loaded.pdb.topology, solute)
    assert enforced == classify_omega_bonds(loaded.pdb.topology, solute)
    assert len(enforced["omega_unscaled_bonds"]) == 2


# --- every surface that scales ---------------------------------------------------------------------

def test_a_fixed_tau_stage_is_refused_before_anything_is_written(unclassifiable):
    from md_tools.run.preflight import PreflightError, _prepare_stage

    loaded = _loaded(unclassifiable)
    stage = {"tau": 0.3, "ensemble": "NVT", "seed": 1, "temperature_K": 300.0,
             "pressure_bar": 1.0}
    with pytest.raises(PreflightError) as refused:
        _prepare_stage(loaded, stage=stage, name="prod", where="prod")
    assert "XAA" in str(refused.value)


def test_a_tau_zero_stage_is_not_refused(unclassifiable):
    """Nothing is scaled at tau = 0, so nothing is exempted and there is nothing to classify.
    Refusing here would block every ordinary cMD run of a molecule with a modified residue."""
    from md_tools.run.preflight import _prepare_stage

    loaded = _loaded(unclassifiable)
    stage = {"tau": 0.0, "ensemble": "NVT", "seed": 1, "temperature_K": 300.0,
             "pressure_bar": 1.0}
    _prepare_stage(loaded, stage=stage, name="prod", where="prod")


def test_the_rung_writer_refuses_and_writes_no_rung(unclassifiable, tmp_path):
    from md_tools import layout as L
    from md_tools.build.rungs import RungWriteError, write_rung_systems

    run = L.DatasetLayout(tmp_path / "ALA").next_run("REST2")
    with pytest.raises(RungWriteError) as refused:
        write_rung_systems(run, system_path=unclassifiable / "built.xml",
                           topology_path=unclassifiable / "built.pdb", taus=[0.0, 0.3])
    assert "XAA" in str(refused.value)
    assert not run.state_system(0).exists() and not run.state_system(1).exists()


def test_the_ladder_solute_document_still_refuses(unclassifiable):
    from md_tools.remd.generated import solute_document

    loaded = _loaded(unclassifiable)
    with pytest.raises(SystemExit) as refused:
        solute_document(loaded.pdb.topology, loaded.system)
    assert "XAA" in str(refused.value)


def test_a_scaling_selection_refuses_rather_than_crashing(unclassifiable):
    """It used to iterate a candidate dict as an atom pair and raise `int('bond')`."""
    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2 import ScalingSelection, SelectionError

    loaded = _loaded(unclassifiable)
    with pytest.raises(SelectionError) as refused:
        ScalingSelection.derive(loaded.pdb.topology, solute_atom_indices(loaded.pdb.topology))
    assert "XAA" in str(refused.value)


def test_build_md_refuses_a_ladder_before_the_run_directory_exists(tmp_path):
    from md_tools.build.md import ConfigError, build_scripts

    dataset = tmp_path / "XAA"
    _unclassifiable(dataset)
    config = tmp_path / "REST2.config"
    # Implicit, to match the stand-in System: `build-md` validates the whole chain against it.
    config.write_text("protocol: REST2\nsolvent: implicit\n", encoding="utf-8")
    out = dataset / "REST2-run1"
    with pytest.raises(ConfigError) as refused:
        build_scripts(config_path=config, out_dir=out, echo=False)
    assert "XAA" in str(refused.value)
    assert not out.exists(), "a refused generation must leave no run directory behind"
    assert not (dataset / "input").exists(), "nor anything in the SHARED input/"


def test_a_group_files_solute_yaml_with_unresolved_candidates_is_refused(tmp_path):
    """The executor's fallback reads `solute.yaml` when no plan resolved a selection. A document
    recording unresolved candidates describes a ladder that must not run."""
    from md_tools.remd.executor import _excluded_bonds_from_solute_document

    document = {"n_solute_atoms": 22,
                "rest2": {"omega_excluded_bonds": [],
                          "omega_ambiguous_candidates": [{"bond": [4, 6],
                                                          "nitrogen_residue": "XAA",
                                                          "evidence": "no SDF"}]}}
    with pytest.raises(RuntimeError) as refused:
        _excluded_bonds_from_solute_document(document, source="solute.yaml")
    assert "XAA" in str(refused.value)


# --- and it stays the one entry point --------------------------------------------------------------

#: Where the classifier may still be called directly. Both are BUILD RECORDS: they describe what
#: the classifier found, unclassified candidates included, for a System that may only ever run at
#: tau = 0. Neither decides what a scaler excludes.
_RECORDING_SITES = {
    "src/md_tools/openmm/system.py",        # `build_system`'s record, and `omega_exclusions`
    "src/md_tools/openmm/implicit.py",      # the implicit build record
}


def test_no_scaling_surface_calls_the_classifier_directly():
    """Taking `omega_unscaled_bonds` and ignoring the unclassified list is the defect, and every
    call site that did it looked reasonable on its own."""
    offenders = []
    for path in sorted((REPO / "src" / "md_tools").rglob("*.py")):
        relative = str(path.relative_to(REPO))
        if relative in _RECORDING_SITES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "classify_omega_bonds"):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, (
        "call `omega_exclusions`, which refuses unclassified candidates, instead of "
        f"`classify_omega_bonds`: {offenders}")


# --- the case that was actually broken, end to end -----------------------------------------------

#: cyclo(Gly-Asp-Arg): three backbone amides in a 9-membered ring, above the proline-like bound of
#: 7, so all three are ORDINARY and must stay unscaled.
CYCLO_GDR = "O=C1NCC(=O)N[C@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]"


@pytest.mark.slow
def test_build_md_rungs_of_a_peptide_like_solute_exclude_its_omegas(tmp_path):
    """THE REPRODUCTION. Before the fix `build-md` wrote these rungs with
    `excluded_central_bonds: []` and `n_excluded_torsions: 0` under a header claiming
    `ordinary_amide_omega: unscaled` -- every omega of the macrocycle scaled, in the files the
    grouped ladder integrates, because the rung writer was never handed `built.sdf`.
    """
    import json
    import subprocess
    import sys

    from openmm.app import PDBFile

    from md_tools.build.md import build_scripts
    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import omega_exclusions

    dataset = tmp_path / "CYC"
    build = dataset / "build"
    build.mkdir(parents=True)
    (build / "in.smi").write_text(f"{CYCLO_GDR} CYC\n", encoding="utf-8")
    (build / "sys.config").write_text(
        "solute:\n  kind: peptide-like\n"
        "constraints:\n  type: HBonds\nhydrogen_mass_repartitioning:\n  enabled: false\n",
        encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", "in.smi",
         "-os", "built.xml", "-op", "built.pdb", "-log", "built.log", "--config", "sys.config"],
        cwd=build, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert (build / "built.sdf").is_file(), "an explicit peptide-like build must retain its SDF"

    topology = PDBFile(str(build / "built.pdb")).topology
    expected = {tuple(sorted(b)) for b in omega_exclusions(
        topology, solute_atom_indices(topology),
        ligand_sdf=build / "built.sdf")["omega_unscaled_bonds"]}
    assert len(expected) == 3, expected

    config = tmp_path / "REST2.config"
    config.write_text("protocol: REST2\n", encoding="utf-8")
    build_scripts(config_path=config, out_dir=dataset / "REST2-run1", echo=False)

    record = json.loads((dataset / "REST2-run1" / "build_states.log").read_text(encoding="utf-8"))
    written = {tuple(sorted(b)) for b in record["omega_exclusion"]["excluded_central_bonds"]}
    assert written == expected, (written, expected)
    assert record["omega_exclusion"]["n_excluded_torsions"] > 0
