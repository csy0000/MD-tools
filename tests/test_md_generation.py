"""`md-gen` produces runnable local scripts that do not reach back to this checkout."""
from __future__ import annotations

import shutil
import stat

import pytest
import yaml

from .conftest import ALA_PDB, REPO_ROOT, run_cli

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A complete inputs/ + MD/ project, tiny enough to run."""
    work = tmp_path_factory.mktemp("mdgen")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "REST2", "--solvent", "OPC", cwd=work)

    sys_config = work / "sys.config.yaml"
    document = yaml.safe_load(sys_config.read_text())
    document["solvent"]["padding_nm"] = 0.9
    sys_config.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    md_config = work / "md.config.yaml"
    protocol = yaml.safe_load(md_config.read_text())
    protocol["minimization"]["max_iterations"] = 50
    protocol["equilibration"] = {"nvt_duration_ps": 0.2, "npt_duration_ps": 0.2,
                                 "restraint_k_kcal_mol_a2": 1.0}
    protocol["cMD"].update({"duration_ns": 0.002, "checkpoint_interval_ps": 1,
                            "whole_system_interval_ps": 1, "solute_interval_ps": 1})
    protocol["REST2"].update({"number_of_replicas": 2, "duration_per_segment_ps": 0.2,
                              "number_of_exchanges": 1, "tau_max": 0.1})
    md_config.write_text(yaml.safe_dump(protocol, sort_keys=False))

    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work


def test_the_layout_is_what_the_documentation_promises(project):
    md = project / "MD"
    assert (md / "md.config.yaml").is_file()
    assert (md / "provenance.yaml").is_file()
    for name in ("run.py", "run.sh"):
        assert (md / "cMD" / name).is_file(), name
        assert (md / "REST2" / name).is_file(), name
    assert (md / "REST2" / "extend.sh").is_file()
    assert (md / "REST2" / "rest2_scaling.py").is_file()
    # Both run.py files import it beside themselves; a project without it cannot start at all.
    for folder in ("cMD", "REST2"):
        assert (md / folder / "md_stages.py").is_file(), folder


def test_the_launchers_are_executable(project):
    for path in ((project / "MD" / "cMD" / "run.sh"),
                 (project / "MD" / "REST2" / "run.sh"),
                 (project / "MD" / "REST2" / "extend.sh")):
        assert path.stat().st_mode & stat.S_IXUSR, path


def test_no_generated_file_contains_a_path_into_this_checkout(project):
    """A project that names the checkout stops working the moment either one moves."""
    offenders = {}
    for path in (project / "MD").rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if str(REPO_ROOT) in text:
            offenders[str(path.relative_to(project))] = [
                line.strip() for line in text.splitlines() if str(REPO_ROOT) in line][:2]
    assert not offenders, offenders


def test_the_scripts_do_not_import_this_package(project):
    """Generated scripts may depend on OpenMM; they must not depend on md_templates."""
    for path in (project / "MD").rglob("*.py"):
        text = path.read_text()
        assert "md_templates" not in text, f"{path.name} imports the template package"


def test_the_inputs_folder_is_addressed_relatively(project):
    document = yaml.safe_load((project / "MD" / "md.config.yaml").read_text())
    assert not document["paths"]["inputs_folder"].startswith("/")


def test_md_gen_refuses_an_incomplete_inputs_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    run_cli("md_openmm", "sys-config", cwd=tmp_path)
    result = run_cli("md_openmm", "md-gen", "-if", "./empty", "--config", "md.config.yaml",
                     "-of", "./MD", cwd=tmp_path)
    assert result.returncode != 0
    assert "sys-gen" in (result.stdout + result.stderr)


def test_a_four_femtosecond_timestep_without_hmr_is_refused(tmp_path, project):
    """The two settings live in different files and are individually reasonable."""
    config = tmp_path / "md.config.yaml"
    document = yaml.safe_load((project / "md.config.yaml").read_text())
    document["common"]["timestep_fs"] = 4.0
    config.write_text(yaml.safe_dump(document, sort_keys=False))
    result = run_cli("md_openmm", "md-gen", "-if", str(project / "inputs"),
                     "--config", str(config), "-of", str(tmp_path / "MD"), cwd=tmp_path)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "hydrogen_mass_amu" in combined and "3.024" in combined
