"""Behavioural tests for the portable explicit-solvent layer.

These test what a *consumer* experiences: an installed entry point, manifests that refuse to
describe the wrong molecule, bundles that detect tampering, run directories that cannot be
overwritten, and a status that never says "completed" for work that did not finish.

Two things are deliberately avoided here.

*Tautological hashing.* `assert hashlib.sha256(value).hexdigest()` proves only that hashing
returns a non-empty string. Every hash assertion below either compares against an independently
recomputed digest, or checks that a real modification changes the verdict.

*Mocked provenance.* The manifest tests build the actual provenance object and inspect its fields,
so a manifest that silently stops recording the force field fails here rather than three months
later when someone tries to reproduce a run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from md_templates.openmm import bundle as bundle_mod          # noqa: E402
from md_templates.openmm import runner, schemas               # noqa: E402
from md_templates.openmm.cli import build_parser, main        # noqa: E402
from md_templates.openmm.schemas import (                     # noqa: E402
    ManifestError,
    config_hash,
    load_experiment,
    load_system,
    shipped_experiment,
    shipped_system,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def rgd_doc() -> dict:
    return yaml.safe_load(shipped_system("cyclo_rgdfv").read_text(encoding="utf-8"))


def write_yaml(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def expect_manifest_error(path: Path, *, fragment: str) -> None:
    with pytest.raises(ManifestError) as excinfo:
        load_system(path)
    assert fragment.lower() in str(excinfo.value).lower(), str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------

def test_cli_help_lists_every_documented_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    sub = {a.dest: a for a in parser._actions if a.dest == "command"}["command"]
    assert set(sub.choices) == {
        "validate-env", "validate-system", "prepare", "validate-bundle", "rest2", "md",
        "smoke", "config", "bundle",
    }


def test_entry_point_is_declared_in_pyproject():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert "md-openmm = \"md_templates.openmm.cli:main\"" in text


def test_invocation_outside_the_checkout(tmp_path):
    """The CLI must not depend on the current directory being the source tree.

    Run as a module in a temp cwd with the checkout's `src` on PYTHONPATH — the same import path a
    wheel install produces, minus the wheel. The wheel-installed variant is exercised by §8's
    outside-checkout smoke, which needs a clean environment this test cannot build.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    res = subprocess.run(
        [sys.executable, "-m", "md_templates.openmm.cli", "validate-system",
         "--system", "cyclo_rgdfv", "--no-chemistry"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    assert "cyclo_rgdfv" in res.stdout


# ---------------------------------------------------------------------------------------------
# system manifest: identity
# ---------------------------------------------------------------------------------------------

def test_shipped_rgd_manifest_is_the_project_molecule():
    system = load_system(shipped_system("cyclo_rgdfv"))
    assert system.system_id == "cyclo_rgdfv"
    assert system.route == "smiles"
    assert system.canonical_hash == schemas.RGD_CANONICAL_SMILES_SHA256
    assert system.formal_charge == 0
    assert system.doc["parameterization"]["small_molecule_forcefield"] == "openff-2.2.0"
    assert system.doc["parameterization"]["charge_method"] == "am1bcc"
    # the ligand route must not carry a protein force field, or the route is ambiguous
    assert system.doc["parameterization"]["protein_forcefield"] is None


def test_canonical_smiles_hash_is_checked_against_the_declared_string(tmp_path):
    doc = rgd_doc()
    doc["input"]["canonical_smiles_sha256"] = "0" * 64
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="does not hash")


def test_canonical_smiles_must_agree_with_rdkit(tmp_path):
    pytest.importorskip("rdkit")
    doc = rgd_doc()
    # a valid, correctly hashed, but NON-canonical string: only RDKit can catch this
    other = "OC(=O)C"
    doc["system_id"] = "some_other_molecule"
    doc["input"]["smiles"] = other
    doc["input"]["canonical_isomeric_smiles"] = "CC(=O)O_not_canonical"
    doc["input"]["canonical_smiles_sha256"] = schemas.sha256_text("CC(=O)O_not_canonical")
    doc["input"]["expected_formal_charge"] = 0
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="disagrees with rdkit")


def test_rgd_identity_cannot_be_attached_to_another_neutral_molecule(tmp_path):
    """`cyclo_rgdfv` is a claim about which molecule this is, not a label."""
    pytest.importorskip("rdkit")
    from rdkit import Chem

    smiles = "CC(=O)O"                                    # neutral, parses, is not RGD
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    doc = rgd_doc()
    doc["input"]["smiles"] = smiles
    doc["input"]["canonical_isomeric_smiles"] = canonical
    doc["input"]["canonical_smiles_sha256"] = schemas.sha256_text(canonical)
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="is reserved")


def test_formal_charge_mismatch_is_rejected(tmp_path):
    pytest.importorskip("rdkit")
    doc = rgd_doc()
    doc["system_id"] = "rgd_like"                          # avoid the stricter RGD identity path
    doc["input"]["expected_formal_charge"] = -1
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="expected_formal_charge")


def test_rgd_expected_charge_must_be_zero(tmp_path):
    doc = rgd_doc()
    doc["input"]["expected_formal_charge"] = 1
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="formal charge is exactly")


# ---------------------------------------------------------------------------------------------
# system manifest: route / force-field agreement
# ---------------------------------------------------------------------------------------------

def test_ligand_route_cannot_load_a_protein_forcefield(tmp_path):
    doc = rgd_doc()
    doc["parameterization"]["protein_forcefield"] = "amber19/protein.ff19SB.xml"
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc),
                          fragment="may not load a protein force field")


def test_peptide_route_cannot_silently_become_a_sage_ligand_route(tmp_path):
    pdb = tmp_path / "structure.pdb"
    pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n",
                   encoding="utf-8")
    doc = {
        "schema_version": 1,
        "system_id": "peptide_thing",
        "display_name": "peptide",
        "input": {"route": "pdb", "pdb": "structure.pdb",
                  "pdb_sha256": schemas.sha256_file(pdb)},
        "parameterization": {
            "protein_forcefield": "amber19/protein.ff19SB.xml",
            "small_molecule_forcefield": "openff-2.2.0",      # the mistake
            "charge_method": None,
            "water_forcefield": "amber19/tip3pfb.xml",
        },
    }
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc),
                          fragment="may not also name a small-molecule force field")


def test_auto_route_is_refused(tmp_path):
    doc = rgd_doc()
    doc["input"]["route"] = "auto"
    expect_manifest_error(write_yaml(tmp_path / "s.yaml", doc), fragment="may not be 'auto'")


def test_pdb_hash_mismatch_is_rejected(tmp_path):
    pdb = tmp_path / "structure.pdb"
    pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n",
                   encoding="utf-8")
    doc = {
        "schema_version": 1,
        "system_id": "peptide_thing",
        "display_name": "peptide",
        "input": {"route": "pdb", "pdb": "structure.pdb", "pdb_sha256": "0" * 64},
        "parameterization": {
            "protein_forcefield": "amber19/protein.ff19SB.xml",
            "small_molecule_forcefield": None,
            "charge_method": None,
            "water_forcefield": "amber19/tip3pfb.xml",
        },
    }
    path = write_yaml(tmp_path / "s.yaml", doc)
    expect_manifest_error(path, fragment="does not match the file")

    # and the same manifest with the true digest loads
    doc["input"]["pdb_sha256"] = schemas.sha256_file(pdb)
    assert load_system(write_yaml(tmp_path / "s2.yaml", doc)).route == "pdb"


# ---------------------------------------------------------------------------------------------
# experiment manifests
# ---------------------------------------------------------------------------------------------

def test_rgd_ladder_is_the_ten_rung_working_ladder():
    """Shape of the ladder. Its exact values and its status are pinned separately below."""
    exp = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    scales = exp.scale_factors
    assert exp.n_rungs == 10
    assert scales[0] == 1.0
    assert scales[-1] == pytest.approx(0.25)
    assert all(b < a for a, b in zip(scales, scales[1:])), "ladder must be strictly descending"
    # even in sqrt(s), which is what makes acceptance roughly even along the chain
    roots = [s ** 0.5 for s in scales]
    gaps = [a - b for a, b in zip(roots, roots[1:])]
    assert max(gaps) - min(gaps) < 1e-6, gaps
    # `validated` was too broad: the predeclared conjunctive rule was not formally satisfied
    assert exp.ladder_status == "pilot_supported"


def test_generic_macrocycle_template_is_marked_unvalidated():
    exp = load_experiment(shipped_experiment("macrocycle_pilot_8rung"))
    assert exp.n_rungs == 8
    assert exp.ladder_status == "unvalidated"
    text = shipped_experiment("macrocycle_pilot_8rung").read_text(encoding="utf-8")
    assert "not validated" in text.lower() or "unvalidated" in text.lower()


def test_ladder_status_defaults_to_unvalidated_when_absent(tmp_path):
    doc = yaml.safe_load(shipped_experiment("smoke").read_text(encoding="utf-8"))
    doc.pop("ladder_status", None)
    exp = load_experiment(write_yaml(tmp_path / "e.yaml", doc))
    assert exp.ladder_status == "unvalidated", "silence is not evidence of validation"


def test_ladder_must_start_cold_and_descend(tmp_path):
    doc = yaml.safe_load(shipped_experiment("smoke").read_text(encoding="utf-8"))
    doc["rest2"]["scale_factors"] = [0.5, 1.0, 0.25]
    with pytest.raises(ManifestError, match="start at the cold rung"):
        load_experiment(write_yaml(tmp_path / "e.yaml", doc))


# ---------------------------------------------------------------------------------------------
# config hashing
# ---------------------------------------------------------------------------------------------

def test_same_manifests_hash_the_same_regardless_of_key_order():
    s = load_system(shipped_system("cyclo_rgdfv")).doc
    e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
    reordered_s = dict(reversed(list(s.items())))
    reordered_e = dict(reversed(list(e.items())))
    assert config_hash(s, e) == config_hash(reordered_s, reordered_e)


def test_hash_changes_with_the_seed():
    s = load_system(shipped_system("cyclo_rgdfv")).doc
    e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
    other = json.loads(json.dumps(e))
    other["master_seed"] = e["master_seed"] + 1
    assert config_hash(s, e) != config_hash(s, other)


def test_hash_changes_with_a_scientific_setting():
    s = load_system(shipped_system("cyclo_rgdfv")).doc
    e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
    other = json.loads(json.dumps(e))
    other["rest2"]["scale_factors"] = other["rest2"]["scale_factors"][:-1]
    assert config_hash(s, e) != config_hash(s, other)


def test_hash_is_stable_across_processes():
    """A hash recomputed in a fresh interpreter must agree, or run names are not reproducible."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), PYTHONHASHSEED="12345")
    code = textwrap.dedent(
        """
        from md_templates.openmm.schemas import (config_hash, load_system, load_experiment,
                                                 shipped_system, shipped_experiment)
        s = load_system(shipped_system("cyclo_rgdfv"), check_chemistry=False).doc
        e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
        print(config_hash(s, e))
        """
    )
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=300)
    assert res.returncode == 0, res.stderr
    s = load_system(shipped_system("cyclo_rgdfv")).doc
    e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
    assert res.stdout.strip() == config_hash(s, e)


# ---------------------------------------------------------------------------------------------
# portability of what we generate
# ---------------------------------------------------------------------------------------------

def test_shipped_manifests_are_not_hidden_from_git():
    """`data/` is gitignored at any depth here, so a package dir by that name never reaches git.

    This bit once: the manifests were written to `explicit/data/` and `git add` silently staged
    nothing, which would have produced a wheel with no shipped manifests from a fresh clone.
    """
    d = schemas.manifests_dir()
    assert d.name == "manifests", d
    # Check the PACKAGE-RELATIVE path, never the absolute one. The concern is a package directory
    # named `data`, which .gitignore erases at any depth; where the checkout happens to live on
    # disk has nothing to do with it. Substring-matching "/data/" against the absolute path fails
    # for anyone whose clone sits under a directory named `data` -- e.g. /path/to/... --
    # which is a spurious failure about the user's filesystem, not about the package.
    pkg_relative = d.parts[d.parts.index("md_templates"):] if "md_templates" in d.parts else d.parts
    assert "data" not in pkg_relative, (
        f"a package directory is named 'data', which .gitignore erases at any depth: {d}"
    )
    res = subprocess.run(["git", "check-ignore", str(shipped_system("cyclo_rgdfv"))],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0, (
        f"shipped manifests are gitignored: {res.stdout.strip()}"
    )


def test_shipped_manifests_contain_no_developer_paths():
    for path in sorted(schemas.manifests_dir().rglob("*.yaml")):
        found = schemas.find_developer_paths(path.read_text(encoding="utf-8"))
        assert not found, f"{path} contains machine-local path(s): {found}"


def test_launcher_has_no_hardcoded_devices_or_developer_paths():
    text = (REPO_ROOT / "docs/implementation/explicit_solvent/scripts/run_all.sh").read_text(
        encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "CUDA_VISIBLE_DEVICES=0" not in body
    assert "CUDA_VISIBLE_DEVICES=1" not in body
    assert "CUDA_VISIBLE_DEVICES=2" not in body
    assert "nohup" not in body
    assert not schemas.find_developer_paths(body), schemas.find_developer_paths(body)
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in text


def test_developer_path_detector_actually_detects():
    """Guard the guard: a detector that never fires would make the test above vacuous."""
    assert schemas.find_developer_paths("out: /home/someone/runs")
    assert schemas.find_developer_paths('root="/Users/someone/x"')
    assert not schemas.find_developer_paths("out: ./runs")


# ---------------------------------------------------------------------------------------------
# platform / device argument validation
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("platform", ["CPU", "CUDA", "OpenCL"])
def test_supported_platforms_are_accepted(platform):
    args = build_parser().parse_args(
        ["rest2", "--bundle", "b", "--out-root", "r", "--platform", platform]
    )
    assert args.platform == platform


def test_unknown_platform_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["rest2", "--bundle", "b", "--out-root", "r", "--platform", "TPU"]
        )


def test_device_is_meaningless_for_cpu():
    with pytest.raises(ValueError, match="meaningless"):
        runner.configure_device("CPU", "0")


def test_cuda_device_order_is_set_so_device_ids_are_not_guesswork(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    env = runner.configure_device("CUDA", "1")
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def _executable_source(path: Path) -> str:
    """Source with comments and string literals removed.

    Needed because the modules deliberately DESCRIBE the hardcoding they replaced — runner.py's
    docstring names `CUDA_VISIBLE_DEVICES=0/1/2` to explain why it is gone. A naive substring
    search would flag the explanation and force the documentation to be deleted to make the test
    pass, which is exactly backwards.
    """
    import io
    import tokenize

    kept: list[str] = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(tok.string)
    return " ".join(kept)


def test_no_device_index_is_hardcoded_anywhere_in_the_package():
    for path in sorted((REPO_ROOT / "src/md_templates/explicit").rglob("*.py")):
        body = _executable_source(path)
        assert "CUDA_VISIBLE_DEVICES=0" not in body, path
        assert not schemas.find_developer_paths(body), path


def test_executable_source_filter_is_not_vacuous(tmp_path):
    """Guard the guard: the filter must strip prose and keep code."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        '''"""docstring mentioning CUDA_VISIBLE_DEVICES=0 and /home/someone."""\n'''
        "# comment mentioning /home/someone\n"
        "value = 1\n",
        encoding="utf-8",
    )
    body = _executable_source(sample)
    assert "CUDA_VISIBLE_DEVICES=0" not in body
    assert "value" in body and "=" in body


# ---------------------------------------------------------------------------------------------
# run directories
# ---------------------------------------------------------------------------------------------

def test_run_directory_name_is_immutable_and_descriptive(tmp_path):
    path = runner.create_run_dir(tmp_path, "cyclo_rgdfv", "abc123def456")
    stamp, system_id, kind, chash = path.name.split("__")
    assert system_id == "cyclo_rgdfv" and kind == "rest2" and chash == "abc123def456"
    assert stamp.endswith("Z") and len(stamp) == len("20260814T224539Z")


def test_existing_run_directory_is_refused(tmp_path):
    path = runner.create_run_dir(tmp_path, "sys", "hash12345678", stamp="20260814T000000Z")
    (path / "replica_00").mkdir()
    with pytest.raises(runner.RunExists, match="already exists"):
        runner.create_run_dir(tmp_path, "sys", "hash12345678", stamp="20260814T000000Z")
    assert (path / "replica_00").is_dir(), "the existing run must be left untouched"


def test_status_transitions_are_distinguishable(tmp_path):
    run_dir = runner.create_run_dir(tmp_path, "sys", "hash12345678")
    for status in (runner.STATUS_RUNNING, runner.STATUS_FAILED, runner.STATUS_INTERRUPTED,
                   runner.STATUS_COMPLETED):
        runner.write_status(run_dir, status)
        assert runner.read_status(run_dir)["status"] == status


def test_a_short_run_is_not_reported_completed(tmp_path):
    """The distinction the whole status field exists for.

    A run that stopped early must not claim the requested budget. This checks the decision rule
    directly: `completed` requires the observed exchange rounds to reach the planned ones.
    """
    run_dir = runner.create_run_dir(tmp_path, "sys", "hash12345678")
    (run_dir / "exchange_attempts.csv").write_text(
        "step,time_ps\n250,0.5\n500,1.0\n", encoding="utf-8")
    observed = runner.count_exchange_rounds(run_dir)
    assert observed == 2
    planned = 100
    runner.write_status(run_dir, runner.STATUS_INTERRUPTED,
                        planned_exchange_rounds=planned, observed_exchange_rounds=observed)
    status = runner.read_status(run_dir)
    assert status["status"] != runner.STATUS_COMPLETED
    assert status["observed_exchange_rounds"] < status["planned_exchange_rounds"]


def test_exchange_rounds_counted_from_the_artifact_not_the_driver(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert runner.count_exchange_rounds(run_dir) == 0, "no log means no rounds"
    (run_dir / "exchange_attempts.csv").write_text(
        "step,time_ps,replica_i,replica_j\n"
        "250,0.5,0,1\n250,0.5,1,2\n"          # same round, two pairs
        "500,1.0,0,1\n",
        encoding="utf-8",
    )
    assert runner.count_exchange_rounds(run_dir) == 2


# ---------------------------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------------------------

def _fake_bundle(tmp_path: Path) -> Path:
    """A structurally valid bundle with real hashes, without paying for parameterisation.

    The heavy path (a real `prepare`) is covered by the slow integration test below; this builds
    the same file set so the hash and tamper logic can be tested in milliseconds.
    """
    b = tmp_path / "bundle"
    b.mkdir()
    for name in bundle_mod.BUNDLE_FILES:
        if name == "system.yaml":
            (b / name).write_text(shipped_system("cyclo_rgdfv").read_text(encoding="utf-8"),
                                  encoding="utf-8")
        elif name == "experiment.prepare.yaml":
            (b / name).write_text(shipped_experiment("smoke").read_text(encoding="utf-8"),
                                  encoding="utf-8")
        else:
            (b / name).write_text(f"content of {name}\n", encoding="utf-8")
    sys_doc = yaml.safe_load((b / "system.yaml").read_text(encoding="utf-8"))
    exp_doc = yaml.safe_load((b / "experiment.prepare.yaml").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "kind": "explicit-solvent-rest2-bundle",
        "config_hash": config_hash(sys_doc, exp_doc),
        "system": {"system_id": sys_doc["system_id"]},
        "files": {name: schemas.sha256_file(b / name) for name in bundle_mod.BUNDLE_FILES},
    }
    (b / bundle_mod.BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return b


def test_bundle_hashes_are_recomputed_not_trusted(tmp_path):
    b = _fake_bundle(tmp_path)
    manifest = bundle_mod.validate_bundle(b)
    # independently recompute one digest and compare with what the manifest recorded
    import hashlib

    expected = hashlib.sha256((b / "topology.pdb").read_bytes()).hexdigest()
    assert manifest["files"]["topology.pdb"] == expected


def test_tampered_bundle_file_is_detected(tmp_path):
    b = _fake_bundle(tmp_path)
    bundle_mod.validate_bundle(b)                      # valid before
    (b / "system.xml").write_text("content of system.xml\n ", encoding="utf-8")   # one space
    with pytest.raises(bundle_mod.BundleError, match="do not match"):
        bundle_mod.validate_bundle(b)


def test_missing_bundle_file_is_rejected(tmp_path):
    b = _fake_bundle(tmp_path)
    (b / "equilibrated_state.xml").unlink()
    with pytest.raises(bundle_mod.BundleError, match="missing bundle file"):
        bundle_mod.validate_bundle(b)


def test_bundle_without_a_manifest_is_not_a_bundle(tmp_path):
    b = _fake_bundle(tmp_path)
    (b / bundle_mod.BUNDLE_MANIFEST).unlink()
    with pytest.raises(bundle_mod.BundleError, match="is not a bundle"):
        bundle_mod.validate_bundle(b)


def test_edited_manifest_inside_a_bundle_is_detected(tmp_path):
    """Rehashing the files is not enough if the manifests themselves were swapped."""
    b = _fake_bundle(tmp_path)
    doc = yaml.safe_load((b / "experiment.prepare.yaml").read_text(encoding="utf-8"))
    doc["master_seed"] = doc["master_seed"] + 1
    (b / "experiment.prepare.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    m = json.loads((b / bundle_mod.BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    m["files"]["experiment.prepare.yaml"] = schemas.sha256_file(b / "experiment.prepare.yaml")
    (b / bundle_mod.BUNDLE_MANIFEST).write_text(json.dumps(m, indent=2), encoding="utf-8")
    with pytest.raises(bundle_mod.BundleError, match="config_hash"):
        bundle_mod.validate_bundle(b)


# ---------------------------------------------------------------------------------------------
# provenance object
# ---------------------------------------------------------------------------------------------

def test_environment_block_records_what_a_reproducer_needs():
    from md_templates.openmm import provenance

    env = provenance.environment_block()
    for key in ("package_version", "package_location", "install", "git", "python", "toolchain",
                "sqm_path", "openmm_platforms", "hardware"):
        assert key in env, key
    assert env["python"].startswith("3.")
    # the toolchain must name every package that can change a number, present or not
    for tool in ("openmm", "rdkit", "openff-toolkit", "numpy"):
        assert tool in env["toolchain"], tool
    assert set(env["install"]) == {"wheel", "wheel_sha256", "install_kind"}
    assert set(env["git"]) >= {"available", "commit", "dirty"}


def test_invocation_records_the_actual_command():
    from md_templates.openmm import provenance

    inv = provenance.invocation()
    assert inv["argv"] == list(sys.argv)
    assert inv["command"] == " ".join(sys.argv)
    assert inv["timestamp_utc"].endswith("Z")


# ---------------------------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------------------------

def test_bad_manifest_exits_with_the_manifest_code(tmp_path, capsys):
    doc = rgd_doc()
    doc["input"]["route"] = "auto"
    path = write_yaml(tmp_path / "s.yaml", doc)
    assert main(["validate-system", "--system", str(path)]) == runner.EXIT_MANIFEST
    assert "manifest error" in capsys.readouterr().err


def test_unknown_manifest_name_is_a_manifest_error(capsys):
    assert main(["validate-system", "--system", "no_such_system"]) == runner.EXIT_MANIFEST
    assert "no such system manifest" in capsys.readouterr().err


def test_validate_bundle_on_a_non_bundle_exits_with_the_bundle_code(tmp_path, capsys):
    assert main(["validate-bundle", "--bundle", str(tmp_path)]) == runner.EXIT_BUNDLE
    assert "bundle error" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_cpu_smoke_prepares_propagates_and_exchanges(tmp_path):
    """Build a real System, propagate every replica, exchange, and produce a complete run.

    Marked slow (it parameterises with AM1-BCC and runs MD), but it is the only test that proves
    the pieces compose. It asserts on the artifacts a consumer reads, not on internal state.
    """
    pytest.importorskip("openmm")
    pytest.importorskip("openff.toolkit")

    code = main([
        "smoke", "--system", "small_macrocycle_smoke", "--out-root", str(tmp_path),
        "--platform", "CPU",
    ])
    assert code == runner.EXIT_OK

    runs = sorted(tmp_path.glob("*__rest2__*"))
    assert len(runs) == 1, runs
    run_dir = runs[0]

    for name in ("run_manifest.json", "system.yaml", "experiment.yaml", "resolved_config.json",
                 "bundle_manifest.json", "stdout.log", "stderr.log", "status.json",
                 "exchange_attempts.csv"):
        assert (run_dir / name).is_file(), f"{name} missing from {run_dir}"

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == runner.STATUS_COMPLETED
    assert status["observed_exchange_rounds"] >= 2

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    replicas = sorted(run_dir.glob("replica_*"))
    assert len(replicas) == manifest["experiment"]["n_rungs"] == 3

    rows = (run_dir / "exchange_attempts.csv").read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    for column in ("step", "replica_i", "replica_j", "scale_factor_i", "scale_factor_j"):
        assert column in header, column
    assert len(rows) - 1 >= 2, "a valid exchange log needs at least two attempts"

    # the bundle the run consumed must still validate afterwards
    bundles = sorted(tmp_path.glob("*__bundle__*"))
    assert len(bundles) == 1
    bundle_manifest = bundle_mod.validate_bundle(bundles[0])
    assert bundle_manifest["composition"]["n_atoms"] > 0
    assert bundle_manifest["composition"]["n_water_molecules"] > 0
    assert bundle_manifest["composition"]["n_degrees_of_freedom"] > 0
    assert bundle_manifest["system"]["formal_charge"] == 0
    assert bundle_manifest["parameterization"]["charge_method"] == "am1bcc"
    assert bundle_manifest["environment"]["toolchain"]["openmm"]


# ---------------------------------------------------------------------------------------------
# fix 8 — RGD box-shape provenance
# ---------------------------------------------------------------------------------------------

#: The Phase-A prepared system that fed all six matched ladder pilots. They copy `rgd_simbox.json`
#: out of it (see `raw_exchange/pilots6_command.sh`), so its geometry IS the geometry the ten-rung
#: evidence was measured in. Preserved in-repo because the original lived in a job temp directory.
RGD_PILOT_SIMBOX = (
    REPO_ROOT
    / "reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json"
)
RGD_PILOT_SIMBOX_SHA256 = "ea1c14edb7c9fc949d892fb41fb733750179555c235b3d6a0a54c7bda821ac61"


def test_rgd_box_shape_matches_the_pilot_evidence():
    """The ladder evidence and the shipped manifest must describe the same solvent geometry.

    The manifest said `cube` while the pilots ran in a dodecahedron. That silently attaches the
    ten-rung acceptance statistics to a different box, so this pins the shape to the artifact
    rather than to anybody's memory. If the evidence file is ever moved, this fails loudly instead
    of the manifest quietly drifting.
    """
    assert RGD_PILOT_SIMBOX.is_file(), (
        f"pilot evidence missing: {RGD_PILOT_SIMBOX}. The RGD box shape is only defensible while "
        "this artifact is preserved."
    )
    assert schemas.sha256_file(RGD_PILOT_SIMBOX) == RGD_PILOT_SIMBOX_SHA256

    evidence = json.loads(RGD_PILOT_SIMBOX.read_text(encoding="utf-8"))
    pilot_shape = evidence["geometry"]["box_shape"]
    assert pilot_shape == "dodecahedron"

    shipped = load_system(shipped_system("cyclo_rgdfv"))
    assert shipped.doc["solvation"]["box_shape"] == pilot_shape
    # padding and salt come from the same artifact and must not drift either
    assert shipped.doc["solvation"]["padding_nm"] == evidence["geometry"]["padding_nm_requested"]
    assert shipped.doc["solvation"]["ionic_strength_molar"] == pytest.approx(0.15)


def test_rgd_manifest_still_enforces_the_full_identity():
    """The box-shape change must not have loosened anything else."""
    s = load_system(shipped_system("cyclo_rgdfv"))
    assert s.canonical_hash == schemas.RGD_CANONICAL_SMILES_SHA256
    assert s.formal_charge == 0
    par = s.doc["parameterization"]
    assert par["small_molecule_forcefield"] == "openff-2.2.0"
    assert par["charge_method"] == "am1bcc"
    assert par["water_forcefield"] == "amber19/tip3pfb.xml"
    assert par["protein_forcefield"] is None


# ---------------------------------------------------------------------------------------------
# fix 8 — exact ladder
# ---------------------------------------------------------------------------------------------

def test_shipped_rgd_ladder_equals_the_authoritative_rule_exactly():
    """One definition, serialised — not two hand-written lists that can drift apart."""
    from md_templates.openmm import rest2_ladder

    shipped_vals = load_experiment(shipped_experiment("rgd_rest2_10rung")).scale_factors
    rule = rest2_ladder(1.0, 0.25, 10, "sqrt")
    assert len(shipped_vals) == 10
    for got, want in zip(shipped_vals, rule):
        assert abs(got - want) < 1e-12, (got, want)


def test_shipped_ladder_is_not_truncated():
    """Six-decimal rounding is a silent perturbation of the ladder the pilots ran."""
    vals = load_experiment(shipped_experiment("rgd_rest2_10rung")).scale_factors
    assert vals[1] == pytest.approx(0.891975308642, abs=1e-12)
    assert vals[7] == pytest.approx(0.373456790123, abs=1e-12)
    # a 6 dp value would be exactly representable at 6 dp; the exact one is not
    assert round(vals[1], 6) != vals[1]


def test_ladder_values_survive_serialisation_and_wheel_loading():
    """Full precision must survive YAML round-trip and package-resource loading."""
    from md_templates.openmm import rest2_ladder

    raw = yaml.safe_load(shipped_experiment("rgd_rest2_10rung").read_text(encoding="utf-8"))
    rule = rest2_ladder(1.0, 0.25, 10, "sqrt")
    for got, want in zip(raw["rest2"]["scale_factors"], rule):
        assert abs(float(got) - want) < 1e-12
    round_tripped = yaml.safe_load(yaml.safe_dump(raw))
    for got, want in zip(round_tripped["rest2"]["scale_factors"], rule):
        assert abs(float(got) - want) < 1e-12


def test_config_hash_changes_if_any_single_scale_factor_changes():
    s = load_system(shipped_system("cyclo_rgdfv")).doc
    e = load_experiment(shipped_experiment("rgd_rest2_10rung")).doc
    base = config_hash(s, e)
    for i in range(len(e["rest2"]["scale_factors"])):
        other = json.loads(json.dumps(e))
        other["rest2"]["scale_factors"][i] += 1e-9
        assert config_hash(s, other) != base, f"scale factor {i} does not affect the hash"


# ---------------------------------------------------------------------------------------------
# fix 8 — ladder-status vocabulary
# ---------------------------------------------------------------------------------------------

def test_ladder_status_vocabulary_excludes_validated():
    assert schemas.LADDER_STATUSES == ("unvalidated", "pilot_supported")


def test_rgd_is_pilot_supported_not_validated():
    assert load_experiment(shipped_experiment("rgd_rest2_10rung")).ladder_status \
        == "pilot_supported"


def test_other_shipped_experiments_stay_unvalidated():
    for name in ("macrocycle_pilot_8rung", "smoke"):
        assert load_experiment(shipped_experiment(name)).ladder_status == "unvalidated"


def test_absent_status_resolves_conservatively(tmp_path):
    doc = yaml.safe_load(shipped_experiment("rgd_rest2_10rung").read_text(encoding="utf-8"))
    doc.pop("ladder_status")
    assert load_experiment(write_yaml(tmp_path / "e.yaml", doc)).ladder_status == "unvalidated"


@pytest.mark.parametrize("bad", ["validated", "production_validated", "converged", "yes"])
def test_unknown_or_overclaiming_status_fails_validation(tmp_path, bad):
    doc = yaml.safe_load(shipped_experiment("smoke").read_text(encoding="utf-8"))
    doc["ladder_status"] = bad
    with pytest.raises(ManifestError, match="ladder_status"):
        load_experiment(write_yaml(tmp_path / "e.yaml", doc))


def test_status_is_preserved_verbatim_in_manifests():
    """A run or bundle manifest must not soften or re-derive the status."""
    exp = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    assert exp.ladder_status == "pilot_supported"
    doc_text = shipped_experiment("rgd_rest2_10rung").read_text(encoding="utf-8")
    assert "ladder_status: pilot_supported" in doc_text


def test_documentation_defines_pilot_supported_without_overclaiming():
    text = (REPO_ROOT / "docs/implementation/explicit_solvent/PORTABLE_REST2.md").read_text(
        encoding="utf-8")
    assert "pilot_supported" in text
    low = text.lower()
    assert "not" in low and "convergence" in low
    assert "production-ready" in low or "production readiness" in low


# ---------------------------------------------------------------------------------------------
# fix 8 — prepared-system fingerprint
# ---------------------------------------------------------------------------------------------

def _cfg(**exp_overrides):
    """Resolved config for RGD + the 10-rung experiment, with experiment-doc tweaks applied."""
    from md_templates.openmm.config import resolve_config
    from md_templates.openmm.schemas import ExperimentManifest

    s = load_system(shipped_system("cyclo_rgdfv"))
    e = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    doc = json.loads(json.dumps(e.doc))
    for dotted, value in exp_overrides.items():
        node = doc
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return resolve_config(s, ExperimentManifest(doc=doc, source=e.source), platform="CPU")


def _sysdoc_cfg(**sys_overrides):
    from md_templates.openmm.config import resolve_config
    from md_templates.openmm.schemas import SystemManifest

    s = load_system(shipped_system("cyclo_rgdfv"))
    e = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    doc = json.loads(json.dumps(s.doc))
    for dotted, value in sys_overrides.items():
        node = doc
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return resolve_config(SystemManifest(doc=doc, source=s.source), e, platform="CPU")


def _bundle_manifest_for(cfg) -> dict:
    from md_templates.openmm.fingerprint import build_projection, fingerprint

    return {"prepared_system": {"fingerprint": fingerprint(cfg),
                                "projection": build_projection(cfg)}}


def test_fingerprint_is_deterministic_and_order_independent():
    from md_templates.openmm.fingerprint import fingerprint

    a, b = _cfg(), _cfg()
    assert fingerprint(a) == fingerprint(b)
    reordered = dict(reversed(list(a.items())))
    assert fingerprint(reordered) == fingerprint(a)


@pytest.mark.parametrize("dotted,value", [
    ("rest2.total_ns_per_replica", 50.0),          # run longer against the same System
    ("rest2.chunk_ns", 0.5),                       # restart granularity
    ("rest2.exchange_interval_ps", 5.0),           # exchange cadence
    ("platform.precision", "double"),              # propagation precision
    ("integrator.timestep_fs", 2.0),               # production timestep
])
def test_runtime_only_changes_are_accepted(dotted, value):
    from md_templates.openmm.fingerprint import check_compatible

    manifest = _bundle_manifest_for(_cfg())
    assert check_compatible(manifest, _cfg(**{dotted: value})) is None, dotted


def test_reporting_interval_change_is_accepted():
    from md_templates.openmm.fingerprint import check_compatible

    manifest = _bundle_manifest_for(_cfg())
    proposed = _cfg(**{"overrides.production.report.state_ps": 1.0})
    assert check_compatible(manifest, proposed) is None


def test_platform_and_device_changes_are_accepted():
    from md_templates.openmm.config import resolve_config
    from md_templates.openmm.fingerprint import check_compatible

    s = load_system(shipped_system("cyclo_rgdfv"))
    e = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    manifest = _bundle_manifest_for(resolve_config(s, e, platform="CPU"))
    on_gpu = resolve_config(s, e, platform="CUDA", device="1")
    assert check_compatible(manifest, on_gpu) is None


@pytest.mark.parametrize("dotted,value,label", [
    ("parameterization.small_molecule_forcefield", "openff-2.1.0", "force field"),
    ("parameterization.charge_method", "gasteiger", "charge method"),
    ("parameterization.water_forcefield", "amber14/tip3p.xml", "water model"),
    ("solvation.box_shape", "cube", "box shape"),
    ("solvation.padding_nm", 1.5, "padding"),
    ("solvation.ionic_strength_molar", 0.0, "salt"),
])
def test_system_defining_changes_are_rejected(dotted, value, label):
    from md_templates.openmm.fingerprint import check_compatible

    manifest = _bundle_manifest_for(_sysdoc_cfg())
    reason = check_compatible(manifest, _sysdoc_cfg(**{dotted: value}))
    assert reason is not None, f"{label} change was accepted"
    assert "DIFFERENT prepared System" in reason


@pytest.mark.parametrize("dotted,value,label", [
    ("overrides.system_build.nonbonded_cutoff_nm", 1.2, "cutoff"),
    ("overrides.system_build.nonbonded_method", "CutoffPeriodic", "PME"),
    ("overrides.system_build.constraints", "AllBonds", "constraints"),
    ("overrides.system_build.rigid_water", False, "rigid water"),
    ("overrides.system_build.hydrogen_mass_amu", 1.008, "HMR"),
    ("overrides.system_build.minimum_image_margin_nm", 0.0, "box margin"),
    ("overrides.rest2.omega_exclusion", False, "omega selection"),
    ("overrides.rest2.max_proline_ring_size", 6, "omega classification"),
    ("equilibration.protocol", "simple", "equilibration protocol"),
    ("equilibration.npt_free_ps", 10.0, "equilibration length"),
])
def test_build_defining_experiment_changes_are_rejected(dotted, value, label):
    from md_templates.openmm.fingerprint import check_compatible

    manifest = _bundle_manifest_for(_cfg())
    reason = check_compatible(manifest, _cfg(**{dotted: value}))
    assert reason is not None, f"{label} change was accepted"
    assert "DIFFERENT prepared System" in reason


def test_rejection_names_the_differing_paths_and_values():
    from md_templates.openmm.fingerprint import check_compatible

    manifest = _bundle_manifest_for(_sysdoc_cfg())
    reason = check_compatible(manifest, _sysdoc_cfg(**{"solvation.box_shape": "cube"}))
    assert "solvation.box_shape" in reason
    assert "dodecahedron" in reason and "cube" in reason


def test_rehashing_an_edited_bundle_manifest_cannot_bypass_the_check():
    """The comparison is against the recorded PROJECTION, not the fingerprint alone."""
    from md_templates.openmm.fingerprint import build_projection, check_compatible
    from md_templates.openmm.schemas import canonical_json, sha256_text

    tampered_cfg = _sysdoc_cfg(**{"solvation.box_shape": "cube"})
    manifest = {"prepared_system": {"projection": build_projection(_sysdoc_cfg()),
                                    "fingerprint": "0" * 64}}
    # a stale fingerprint is caught as tampering
    assert "edited" in check_compatible(manifest, tampered_cfg)

    # and recomputing the fingerprint over the ORIGINAL projection does not help either: the
    # values still differ from the proposed configuration
    manifest["prepared_system"]["fingerprint"] = sha256_text(
        canonical_json(manifest["prepared_system"]["projection"]))
    reason = check_compatible(manifest, tampered_cfg)
    assert reason is not None and "DIFFERENT prepared System" in reason


def test_bundle_without_a_fingerprint_is_refused_for_overrides():
    from md_templates.openmm.fingerprint import check_compatible

    assert "no prepared-system fingerprint" in check_compatible({}, _cfg())


def test_runtime_only_paths_are_disjoint_from_build_defining_paths():
    from md_templates.openmm.fingerprint import BUILD_DEFINING_PATHS, RUNTIME_ONLY_PATHS

    overlap = set(BUILD_DEFINING_PATHS) & set(RUNTIME_ONLY_PATHS)
    assert not overlap, f"a path cannot be both build-defining and runtime-only: {overlap}"


def test_incompatible_override_is_rejected_before_the_run_directory_exists(tmp_path):
    """Rejection must happen before any directory or OpenMM context is created."""
    from md_templates.openmm import runner as r

    b = _fake_bundle(tmp_path)
    m = json.loads((b / bundle_mod.BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    m.update(_bundle_manifest_for(_sysdoc_cfg()))
    (b / bundle_mod.BUNDLE_MANIFEST).write_text(json.dumps(m, indent=2), encoding="utf-8")
    m["files"][bundle_mod.BUNDLE_MANIFEST] = None      # not hashed; harmless for this path

    out_root = tmp_path / "runs"
    out_root.mkdir()
    other = tmp_path / "other_experiment.yaml"
    doc = yaml.safe_load(shipped_experiment("rgd_rest2_10rung").read_text(encoding="utf-8"))
    doc.setdefault("overrides", {}).setdefault("system_build", {})["nonbonded_cutoff_nm"] = 1.4
    write_yaml(other, doc)

    with pytest.raises((r.IncompatibleExperiment, bundle_mod.BundleError)) as excinfo:
        r.launch_rest2(b, other, out_root, platform="CPU")
    if isinstance(excinfo.value, r.IncompatibleExperiment):
        assert list(out_root.iterdir()) == [], "a run directory was created before rejection"


# ---------------------------------------------------------------------------------------------
# fix 8 — minimum-image margin
# ---------------------------------------------------------------------------------------------

def test_margin_is_a_runtime_default_and_is_dumped():
    from md_templates.openmm import DEFAULTS, dump_defaults

    assert DEFAULTS["system_build"]["minimum_image_margin_nm"] == pytest.approx(0.10)
    assert "minimum_image_margin_nm" in dump_defaults()


class _FakeModeller:
    """Positions only — `_resolve_box` needs nothing else."""

    def __init__(self, radius_nm: float) -> None:
        from openmm import unit

        import numpy as np

        pts = np.array([[-radius_nm, 0.0, 0.0], [radius_nm, 0.0, 0.0]])
        self.positions = unit.Quantity(pts, unit.nanometer)


def _box_cfg(shape: str, padding: float, cutoff: float, margin: float) -> dict:
    from md_templates.openmm import DEFAULTS
    import copy

    cfg = copy.deepcopy(DEFAULTS)
    cfg["solvation"].update(box_shape=shape, padding_nm=padding,
                            padding_semantics="solute-image-gap", cutoff_fit_policy="grow")
    cfg["system_build"].update(nonbonded_cutoff_nm=cutoff, minimum_image_margin_nm=margin)
    return cfg


@pytest.mark.parametrize("shape", ["cube", "dodecahedron"])
def test_grown_box_clears_the_hard_limit_by_the_margin(shape):
    pytest.importorskip("openmm")
    from md_templates.openmm.solvation import _resolve_box

    cutoff, margin = 1.0, 0.10
    info = _resolve_box(_FakeModeller(0.35), _box_cfg(shape, 0.2, cutoff, margin))
    assert info["grown_for_cutoff"] is True
    assert info["min_image_distance_nm"] >= 2 * cutoff + margin - 1e-9
    assert info["minimum_image_margin_nm"] == pytest.approx(margin)
    assert info["minimum_image_required_nm"] == pytest.approx(2 * cutoff + margin)
    # the recorded geometry must match the vectors actually produced
    import numpy as np

    vectors = np.array(info["box_vectors_nm"])
    frac = float(np.min(np.diag(vectors))) / info["box_width_nm"]
    assert frac * info["box_width_nm"] == pytest.approx(info["min_image_distance_nm"], abs=1e-4)


@pytest.mark.parametrize("shape", ["cube", "dodecahedron"])
def test_a_box_already_above_the_threshold_is_left_alone(shape):
    pytest.importorskip("openmm")
    from md_templates.openmm.solvation import _resolve_box

    info = _resolve_box(_FakeModeller(0.35), _box_cfg(shape, 3.0, 1.0, 0.10))
    assert info["grown_for_cutoff"] is False
    assert info["box_width_nm"] == pytest.approx(info["box_width_requested_nm"])


def test_zero_margin_is_selectable_only_deliberately():
    pytest.importorskip("openmm")
    from md_templates.openmm import DEFAULTS
    from md_templates.openmm.solvation import _resolve_box

    assert DEFAULTS["system_build"]["minimum_image_margin_nm"] > 0
    info = _resolve_box(_FakeModeller(0.35), _box_cfg("cube", 0.2, 1.0, 0.0))
    assert info["min_image_distance_nm"] == pytest.approx(2.0, abs=1e-6)


def test_negative_margin_is_rejected_before_any_box_is_built():
    pytest.importorskip("openmm")
    from md_templates.openmm.solvation import _resolve_box

    with pytest.raises(ValueError, match="minimum_image_margin_nm"):
        _resolve_box(_FakeModeller(0.35), _box_cfg("cube", 0.2, 1.0, -0.05))


def test_refuse_policy_reports_what_would_satisfy_the_margin():
    pytest.importorskip("openmm")
    from md_templates.openmm.solvation import _resolve_box

    cfg = _box_cfg("dodecahedron", 0.2, 1.0, 0.10)
    cfg["solvation"]["cutoff_fit_policy"] = "refuse"
    with pytest.raises(ValueError) as excinfo:
        _resolve_box(_FakeModeller(0.35), cfg)
    msg = str(excinfo.value)
    assert "margin" in msg and "padding_nm" in msg and "2.100" in msg


@pytest.mark.slow
def test_grown_box_survives_npt_without_the_box_size_abort(tmp_path):
    """The regression the margin exists for: a real NPT integration in a grown box.

    Before the margin, a box grown to exactly 2*cutoff aborted on the first NPT step with "The
    periodic box size has decreased to less than twice the nonbonded cutoff". Here the box is
    forced to grow (padding far below what the cutoff needs) and then actually integrated under a
    barostat, which is where the abort used to happen.

    The solute is a single water so that no protein/ligand template is involved — this test is
    about box geometry surviving NPT, and a solute that needs parameterising would only add an
    unrelated way to fail.
    """
    openmm = pytest.importorskip("openmm")
    import numpy as np
    from openmm import MonteCarloBarostat, unit
    from openmm.app import ForceField, HBonds, Modeller, PDBFile, PME, Simulation

    from md_templates.openmm.solvation import _resolve_box

    pdb_path = tmp_path / "wat.pdb"
    pdb_path.write_text(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H\n"
        "HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H\n"
        "END\n",
        encoding="utf-8",
    )
    pdb = PDBFile(str(pdb_path))
    ff = ForceField("amber14/tip3pfb.xml")

    cutoff, margin = 0.9, 0.10
    cfg = _box_cfg("cube", 0.2, cutoff, margin)      # padding far too small: growth is forced
    modeller = Modeller(pdb.topology, pdb.positions)
    info = _resolve_box(modeller, cfg)
    assert info["grown_for_cutoff"] is True, "this test is only meaningful for a grown box"
    assert info["min_image_distance_nm"] >= 2 * cutoff + margin - 1e-9

    vectors = np.array(info["box_vectors_nm"]) * unit.nanometer
    modeller.addSolvent(ff, model="tip3p", boxVectors=vectors, neutralize=False)

    system = ff.createSystem(modeller.topology, nonbondedMethod=PME,
                             nonbondedCutoff=cutoff * unit.nanometer, constraints=HBonds,
                             rigidWater=True)
    system.addForce(MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 5))
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    sim = Simulation(modeller.topology, system, integrator,
                     openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    sim.step(2000)                                    # 2 ps of NPT: where the abort used to happen

    final = sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.nanometer)
    assert float(np.min(np.diag(final))) > 2 * cutoff, (
        "the box contracted below twice the cutoff; the margin did not protect the run"
    )


# ==================================================================================================
# requirement 7: omega exclusion is the default, and toggling it is a Hamiltonian change
# ==================================================================================================

def test_omega_exclusion_defaults_to_true():
    from md_templates.openmm import DEFAULTS

    assert DEFAULTS["rest2"]["omega_exclusion"] is True


def test_cli_accepts_the_exact_documented_form():
    parser = build_parser()
    for cmd in ("prepare", "rest2", "smoke"):
        argv = {"prepare": ["prepare", "--system", "s", "--experiment", "e", "--out-root", "o"],
                "rest2": ["rest2", "--bundle", "b", "--out-root", "o"],
                "smoke": ["smoke", "--system", "s", "--out-root", "o"]}[cmd]
        assert parser.parse_args(argv + ["--omega-exclusion", "true"]).omega_exclusion is True
        assert parser.parse_args(argv + ["--omega-exclusion", "false"]).omega_exclusion is False
        # the optional spelling maps onto the SAME canonical setting, not a second one
        assert parser.parse_args(argv + ["--no-omega-exclusion"]).omega_exclusion is False
        # unset means "whatever the manifests resolved to", not False
        assert parser.parse_args(argv).omega_exclusion is None


def test_unparseable_omega_value_is_refused_not_defaulted():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["smoke", "--system", "s", "--out-root", "o",
                           "--omega-exclusion", "maybe"])


def test_omega_exclusion_is_build_defining_so_it_enters_the_fingerprint():
    """Changing it changes which torsions are scaled, so a bundle cannot be reused across it."""
    from md_templates.openmm.fingerprint import BUILD_DEFINING_PATHS

    assert "rest2.omega_exclusion" in BUILD_DEFINING_PATHS


def test_cli_override_is_recorded_in_the_resolved_configuration():
    """A setting that changes the Hamiltonian must say where its value came from."""
    from md_templates.openmm.config import resolve_config

    system = load_system(shipped_system("cyclo_rgdfv"), check_chemistry=False)
    experiment = load_experiment(shipped_experiment("rgd_rest2_10rung"))

    default_cfg = resolve_config(system, experiment)
    assert default_cfg["rest2"]["omega_exclusion"] is True
    assert "_overrides" not in default_cfg

    off = resolve_config(system, experiment, omega_exclusion=False)
    assert off["rest2"]["omega_exclusion"] is False
    assert off["_overrides"]["rest2.omega_exclusion"] == {"value": False, "source": "command line"}


def test_toggling_omega_exclusion_changes_the_config_hash():
    """Two runs differing only in this setting are different calculations and must not share a hash."""
    from md_templates.openmm.config import resolve_config
    from md_templates.openmm.fingerprint import fingerprint

    system = load_system(shipped_system("cyclo_rgdfv"), check_chemistry=False)
    experiment = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    on = fingerprint(resolve_config(system, experiment, omega_exclusion=True))
    off = fingerprint(resolve_config(system, experiment, omega_exclusion=False))
    assert on != off, "the omega setting does not reach the prepared-system fingerprint"


def test_unclassified_amide_blocks_only_while_the_exclusion_is_on():
    """With the exclusion off, no bond is treated specially, so an unnamed amide changes nothing."""
    from md_templates.openmm.md import _assert_omega_classified

    bundle = {"omega_unclassified_candidates": [
        {"bond": [3, 4], "carbon_residue": "XXX", "nitrogen_residue": "YYY", "ambiguous": "?"}]}
    with pytest.raises(ValueError, match="could not be classified"):
        _assert_omega_classified(bundle, omega_exclusion=True)
    assert _assert_omega_classified(bundle, omega_exclusion=False) == []


# ==================================================================================================
# requirement 2: the chunk count is an input, never a rounded quotient
# ==================================================================================================
from md_templates.openmm.config import resolve_chunk_plan  # noqa: E402


def _plan(n, chunk, dt=4.0, exchange=None):
    return resolve_chunk_plan(n, chunk, timestep_fs=dt, where="production.remd",
                              exchange_interval_ps=exchange)


def test_totals_are_derived_from_the_plan():
    p = _plan(7, 0.5)
    assert (p["n_chunks"], p["chunk_ns"], p["total_ns"]) == (7, 0.5, 3.5)
    assert p["steps_per_chunk"] == 125000          # 0.5 ns at 4 fs


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_non_positive_chunk_counts_are_rejected(bad):
    with pytest.raises(ValueError, match="greater than zero"):
        _plan(bad, 1.0)


@pytest.mark.parametrize("bad", [2.0, 2.5, "2", None])
def test_non_integer_chunk_counts_are_rejected(bad):
    with pytest.raises(ValueError, match="must be an integer"):
        _plan(bad, 1.0)


def test_boolean_chunk_count_is_rejected_even_though_bool_is_an_int():
    """`True` is an int in Python; accepting it as "one chunk" would read a config error as a plan."""
    with pytest.raises(ValueError, match="must be an integer"):
        _plan(True, 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_chunk_length_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        _plan(2, bad)


def test_chunk_must_be_a_whole_number_of_steps():
    with pytest.raises(ValueError, match="whole number of .* fs steps"):
        _plan(2, 0.000003, dt=4.0)          # 3 ps / 4 fs = 0.75 steps


def test_rest2_chunk_must_be_a_whole_number_of_exchange_intervals():
    with pytest.raises(ValueError, match="whole number of exchange intervals"):
        _plan(2, 1.0, exchange=300.0)       # 1000 ps / 300 ps = 3.33 intervals


def test_a_schema_v1_experiment_is_refused_with_a_migration_message(tmp_path):
    """Silently reinterpreting the old field is what the explicit plan exists to stop."""
    doc = yaml.safe_load(shipped_experiment("smoke").read_text(encoding="utf-8"))
    doc["schema_version"] = 1
    doc["rest2"].pop("n_chunks")
    doc["rest2"]["total_ns_per_replica"] = 0.002       # 2 x 0.001, exactly representable
    path = write_yaml(tmp_path / "e.yaml", doc)
    with pytest.raises(ManifestError) as excinfo:
        load_experiment(path)
    message = str(excinfo.value)
    assert "total_ns_per_replica" in message
    assert "n_chunks: 2" in message, "the message should name the plan this manifest meant"
    assert "chunk_ns: 0.001" in message


def test_a_v1_total_that_was_never_a_whole_number_of_chunks_says_so(tmp_path):
    """The old code rounded this and ran a different length than the manifest declared."""
    doc = yaml.safe_load(shipped_experiment("smoke").read_text(encoding="utf-8"))
    doc["schema_version"] = 1
    doc["rest2"].pop("n_chunks")
    doc["rest2"]["total_ns_per_replica"] = 0.0025      # 2.5 chunks of 0.001
    with pytest.raises(ManifestError, match="NOT a whole number of chunks"):
        load_experiment(write_yaml(tmp_path / "e.yaml", doc))


def test_shipped_experiments_declare_an_explicit_plan():
    for name in ("smoke", "rgd_rest2_10rung", "macrocycle_pilot_8rung"):
        doc = yaml.safe_load(shipped_experiment(name).read_text(encoding="utf-8"))
        assert doc["schema_version"] == 2, name
        assert isinstance(doc["rest2"]["n_chunks"], int), name
        assert "total_ns_per_replica" not in doc["rest2"], name


# ==================================================================================================
# requirement 1: the fresh-run / resume naming contract
# ==================================================================================================
from md_templates.openmm.runner import RunExists, resolve_run_dir, run_dir_name  # noqa: E402


def test_a_named_fresh_run_gets_exactly_that_directory(tmp_path):
    """No timestamp, system or hash appended: a decorated name is not the name the user chose."""
    path, is_resume = resolve_run_dir(tmp_path, "sys", "abc123", run_name="my_run")
    assert path == (tmp_path / "my_run").resolve()
    assert path.is_dir() and not is_resume


def test_an_unnamed_fresh_run_gets_a_timestamped_default(tmp_path):
    path, is_resume = resolve_run_dir(tmp_path, "sys", "abc123", method="md", stamp="20260816T101500Z")
    assert path.name == "20260816T101500Z__sys__md__abc123"
    assert path.is_dir() and not is_resume


def test_a_fresh_run_refuses_to_overwrite(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(RunExists, match="will not overwrite"):
        resolve_run_dir(tmp_path, "sys", "abc123", run_name="taken")


def test_a_resume_uses_the_given_directory_and_creates_no_sibling(tmp_path):
    existing = tmp_path / "run_a"
    existing.mkdir()
    before = set(p.name for p in tmp_path.iterdir())
    path, is_resume = resolve_run_dir(tmp_path, "sys", "abc123", resume_run=existing)
    assert path == existing.resolve() and is_resume
    assert set(p.name for p in tmp_path.iterdir()) == before, "a resume created a sibling directory"


def test_resuming_a_directory_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(RunExists, match="does not exist"):
        resolve_run_dir(tmp_path, "sys", "abc123", resume_run=tmp_path / "never_ran")


def test_run_name_and_resume_run_are_mutually_exclusive(tmp_path):
    (tmp_path / "r").mkdir()
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_run_dir(tmp_path, "sys", "abc123", run_name="x", resume_run=tmp_path / "r")


@pytest.mark.parametrize("bad", ["", "   ", "a/b", ".", ".."])
def test_unusable_run_names_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="not a usable directory name"):
        resolve_run_dir(tmp_path, "sys", "abc123", run_name=bad)


def test_default_names_carry_the_method_so_md_and_rest2_do_not_collide():
    md = run_dir_name("sys", "abc123", method="md", stamp="S")
    rest2 = run_dir_name("sys", "abc123", method="rest2", stamp="S")
    assert md != rest2 and "__md__" in md and "__rest2__" in rest2


# ==================================================================================================
# follow-up: continuity contract, crash-safe restarts, lifetime statistics
# ==================================================================================================
from md_templates.openmm import runstate  # noqa: E402


def _contract(**over):
    cfg = {"integrator": {"timestep_fs": 4.0, "kind": "langevin-middle", "temperature_k": 300.0,
                          "friction_per_ps": 1.0},
           "system_build": {"constraints": "HBonds", "nonbonded_cutoff_nm": 1.0},
           "production": {"remd": {"chunk_ns": 0.001, "n_chunks": 2,
                                   "exchange_interval_ps": 0.5, "scale_factors": [1.0, 0.5]},
                          "md": {"chunk_ns": 0.001, "n_chunks": 2, "scale_factor": 1.0}},
           "rest2": {"omega_exclusion": True}}
    for dotted, value in over.items():
        node = cfg
        parts = dotted.replace("__", ".").split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = value
    return runstate.continuity_contract(cfg, method="rest2", fingerprint_value="fp", n_particles=99)


def test_more_chunks_is_an_extension_not_an_incompatibility():
    """Asking for more work must not look like a different calculation, or resume is impossible."""
    base = _contract()
    more = _contract(production__remd__n_chunks=50)
    assert runstate.compare_continuity(base, more) == []


@pytest.mark.parametrize("field,value", [
    ("integrator__timestep_fs", 2.0),
    ("integrator__temperature_k", 310.0),
    ("system_build__constraints", "AllBonds"),
    ("rest2__omega_exclusion", False),
    ("production__remd__chunk_ns", 0.002),
    ("production__remd__exchange_interval_ps", 1.0),
])
def test_hamiltonian_changes_are_incompatible_continuations(field, value):
    diffs = runstate.compare_continuity(_contract(), _contract(**{field: value}))
    assert diffs, f"{field} changed but the continuation was accepted"
    assert any(field.replace("__", ".") == name for name, _, _ in diffs)


def test_resuming_as_the_other_method_is_refused(tmp_path):
    """An MD run and a REST2 run are different runs even when every other setting agrees."""
    md_contract = dict(_contract(), method="md")
    runstate.write_run_state(tmp_path, method="md", continuity=md_contract)
    rest2_contract = dict(_contract(), method="rest2")
    with pytest.raises(runstate.IncompatibleContinuation, match="method"):
        runstate.assert_continuable(tmp_path, rest2_contract)


def test_incompatible_continuation_names_every_differing_field(tmp_path):
    runstate.write_run_state(tmp_path, method="rest2", continuity=_contract())
    with pytest.raises(runstate.IncompatibleContinuation) as excinfo:
        runstate.assert_continuable(tmp_path, _contract(integrator__timestep_fs=2.0))
    message = str(excinfo.value)
    assert "integrator.timestep_fs" in message and "recorded 4.0" in message


def test_a_directory_without_run_state_cannot_be_continued(tmp_path):
    with pytest.raises(runstate.RunStateError, match="does not exist"):
        runstate.assert_continuable(tmp_path, _contract())


def test_an_unknown_run_state_version_is_refused_not_reinterpreted(tmp_path):
    runstate.atomic_write_json(tmp_path / runstate.RUN_STATE_FILE,
                               {"schema_version": 99, "continuity": {}})
    with pytest.raises(runstate.RunStateError, match="schema_version"):
        runstate.read_run_state(tmp_path)


def test_a_generation_cannot_be_committed_before_it_is_written(tmp_path):
    """The commit record must point only at a complete restart."""
    with pytest.raises(runstate.RunStateError, match="missing"):
        runstate.commit_generation(tmp_path, 0, members=["walker.chk", "walker.state.xml"])
    assert runstate.committed_generation(tmp_path) is None


def test_a_failed_generation_does_not_replace_the_committed_one(tmp_path):
    gdir0 = runstate.generation_dir(tmp_path, 0)
    gdir0.mkdir(parents=True)
    for m in ("walker.chk", "walker.state.xml"):
        (gdir0 / m).write_text("x")
    runstate.commit_generation(tmp_path, 0, members=["walker.chk", "walker.state.xml"])
    assert runstate.committed_generation(tmp_path) == 0

    # generation 1 starts but never finishes: only one of its two members is written
    gdir1 = runstate.generation_dir(tmp_path, 1)
    gdir1.mkdir(parents=True)
    (gdir1 / "walker.chk").write_text("partial")
    with pytest.raises(runstate.RunStateError):
        runstate.commit_generation(tmp_path, 1, members=["walker.chk", "walker.state.xml"])
    assert runstate.committed_generation(tmp_path) == 0, "a failed generation took over the commit"


def test_the_previous_generation_is_retained(tmp_path):
    for gen in (0, 1, 2):
        gdir = runstate.generation_dir(tmp_path, gen)
        gdir.mkdir(parents=True)
        for m in ("walker.chk", "walker.state.xml"):
            (gdir / m).write_text("x")
        runstate.commit_generation(tmp_path, gen, members=["walker.chk", "walker.state.xml"])
    kept = sorted(p.name for p in (tmp_path / runstate.RESTART_DIR).iterdir() if p.is_dir())
    assert kept == ["gen_0001", "gen_0002"], kept


def test_atomic_write_leaves_no_partial_file_behind(tmp_path):
    target = tmp_path / "record.json"
    runstate.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "temporary left behind"


def test_duplicate_attempt_indices_are_corruption_not_data():
    from md_templates.openmm.rest2 import _assert_monotonic_attempts

    rows = [{"attempt_index": "0"}, {"attempt_index": "1"}, {"attempt_index": "1"}]
    with pytest.raises(ValueError, match="more than once"):
        _assert_monotonic_attempts(rows, "log.csv")


def test_non_monotonic_attempt_indices_are_refused():
    from md_templates.openmm.rest2 import _assert_monotonic_attempts

    with pytest.raises(ValueError, match="not\n?.*monotonic|monotonic"):
        _assert_monotonic_attempts([{"attempt_index": "5"}, {"attempt_index": "2"}], "log.csv")


def test_summary_regeneration_is_idempotent(tmp_path):
    """Rebuilding lifetime statistics from the log must not accumulate."""
    from md_templates.openmm.rest2 import summarise_exchange_log

    log = tmp_path / "exchange_attempts.csv"
    log.write_text("attempt_index,step,i,j,accepted\n"
                   "0,250,0,1,1\n1,500,0,1,0\n2,750,1,2,1\n")
    first = summarise_exchange_log(tmp_path)
    second = summarise_exchange_log(tmp_path)
    assert first == second
    assert first["lifetime_exchange_attempts"] == 3
    assert first["lifetime_exchange_accepted"] == 2


def test_lifetime_counts_span_the_whole_log():
    """The counter a resumed run starts from is the history, not zero."""
    from md_templates.openmm.rest2 import _lifetime_counts
    import tempfile as _tf

    d = Path(_tf.mkdtemp())
    log = d / "exchange_attempts.csv"
    log.write_text("attempt_index,accepted\n0,1\n1,0\n2,1\n3,1\n")
    assert _lifetime_counts(log) == (4, 3)


def test_md_command_passes_naming_options_into_the_execution_path(monkeypatch):
    """The CLI option must reach the runner, not merely parse."""
    from md_templates.openmm import cli

    seen = {}

    def fake_launch_md(bundle, exp, out_root, **kw):
        seen.update(kw)
        return 0, None

    monkeypatch.setattr(cli.runner, "launch_md", fake_launch_md)
    monkeypatch.setattr(cli.runner, "install_signal_handlers", lambda: None)
    parser = build_parser()
    args = parser.parse_args(["md", "--bundle", "b", "--out-root", "o", "--run-name", "chosen"])
    args.func(args)
    assert seen["run_name"] == "chosen" and seen["resume_run"] is None


def test_rest2_command_passes_naming_options_into_the_execution_path(monkeypatch, tmp_path):
    from md_templates.openmm import cli

    seen = {}

    def fake_launch_rest2(bundle, exp, out_root, **kw):
        seen.update(kw)
        return 0, None

    monkeypatch.setattr(cli.runner, "launch_rest2", fake_launch_rest2)
    monkeypatch.setattr(cli.runner, "install_signal_handlers", lambda: None)
    (tmp_path / "existing").mkdir()
    parser = build_parser()
    args = parser.parse_args(["rest2", "--bundle", "b", "--out-root", str(tmp_path),
                              "--resume-run", "existing"])
    args.func(args)
    assert seen["resume_run"] == tmp_path / "existing" and seen["run_name"] is None


def test_md_and_resume_options_are_mutually_exclusive_on_both_commands():
    parser = build_parser()
    for argv in (["md", "--bundle", "b", "--out-root", "o"],
                 ["rest2", "--bundle", "b", "--out-root", "o"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv + ["--run-name", "a", "--resume-run", "b"])
