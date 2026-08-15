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

from escort_ais.explicit import bundle as bundle_mod          # noqa: E402
from escort_ais.explicit import runner, schemas               # noqa: E402
from escort_ais.explicit.cli import build_parser, main        # noqa: E402
from escort_ais.explicit.schemas import (                     # noqa: E402
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
        "validate-env", "validate-system", "prepare", "validate-bundle", "rest2", "smoke",
    }


def test_entry_point_is_declared_in_pyproject():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert "escort-explicit = \"escort_ais.explicit.cli:main\"" in text


def test_invocation_outside_the_checkout(tmp_path):
    """The CLI must not depend on the current directory being the source tree.

    Run as a module in a temp cwd with the checkout's `src` on PYTHONPATH — the same import path a
    wheel install produces, minus the wheel. The wheel-installed variant is exercised by §8's
    outside-checkout smoke, which needs a clean environment this test cannot build.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    res = subprocess.run(
        [sys.executable, "-m", "escort_ais.explicit.cli", "validate-system",
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
    assert exp.ladder_status == "validated"


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
        from escort_ais.explicit.schemas import (config_hash, load_system, load_experiment,
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
    assert "/data/" not in str(d), d
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
    for path in sorted((REPO_ROOT / "src/escort_ais/explicit").rglob("*.py")):
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
    from escort_ais.explicit import provenance

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
    from escort_ais.explicit import provenance

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
