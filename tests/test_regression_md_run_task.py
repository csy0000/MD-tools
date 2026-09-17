"""Regressions for the crossed-pair policy, the `md-run` surface and the CUDA-first rule.

Written BEFORE the implementation, so each one names a defect that exists today:

* `build-top` rejects ff19SB + TIP3P before the warning machinery can run, contradicting the
  documented policy that both crossed pairs build with a recorded warning;
* there is no `md-openmm md-run`, so there is no Amber-like execution surface;
* a replica ladder may fall back to CPU where an ordinary stage refuses;
* AIS has no `-source-traj`, no per-path trajectory naming, and does not require its reporting
  intervals to divide the switching path;
* both YAML config loaders accept a duplicate key and silently keep the last value.

They are grouped here rather than scattered so the "before" state is legible in one file. As each
is fixed the test stays, asserting the new behaviour.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]


def _config(**blocks) -> Path:
    path = Path(tempfile.mkdtemp()) / "build.config"
    path.write_text(yaml.safe_dump(blocks, sort_keys=False), encoding="utf-8")
    return path


# --- 1. the crossed explicit pairs ---------------------------------------------------------------

@pytest.mark.parametrize("protein, water", [("ff14SB", "OPC"), ("ff19SB", "TIP3P")])
def test_a_crossed_explicit_pair_resolves_instead_of_being_rejected(protein, water):
    """Both crossed pairs are permitted-but-warned, not refused.

    ff19SB + TIP3P was refused by `_check_pairings` before `pairing_warnings()` could ever see it,
    so the warning policy was unreachable for half the combinations it described.
    """
    from md_tools.build.top import resolve_build_config

    resolved = resolve_build_config(_config(forcefield={"protein": protein},
                                            solvent={"model": water}))
    assert resolved["forcefield"]["protein"] == protein
    assert resolved["solvent"]["model"] == water


@pytest.mark.parametrize("protein, water", [("ff14SB", "TIP3P"), ("ff19SB", "OPC")])
def test_a_matched_pair_produces_no_pairing_warning(protein, water):
    from md_tools.build.top import resolve_build_config
    from md_tools.openmm.system_config import pairing_warnings, resolve_sys_config
    from md_tools.build.top import _sys_document

    resolved = resolve_build_config(_config(forcefield={"protein": protein},
                                            solvent={"model": water}))
    assert pairing_warnings(resolve_sys_config(_sys_document(resolved))) == []


def test_ff19sb_with_gbn2_remains_a_hard_error():
    """Softening the pairing policy must not soften a combination that cannot be built.

    Refused by `resolve_build_config`, which is the LAST point where the evidence still exists.
    Further in, `_sys_document` takes the protein force field from `sys_defaults` on the implicit
    branch, so by the time `_check_protein_solvation_pairing` runs the document says ff14SB and
    the pair it was written to refuse is no longer visible in it. An ff19SB + GBn2 request was
    therefore not refused anywhere -- it was silently built as ff14SB + GBn2.
    """
    from md_tools.build.strict import ConfigError
    from md_tools.build.top import resolve_build_config

    with pytest.raises(ConfigError, match="not parameterised for"):
        resolve_build_config(_config(forcefield={"protein": "ff19SB"},
                                     solvent={"model": "GBn2"}))


def test_an_implicit_build_uses_the_protein_forcefield_that_was_asked_for():
    """The silent substitution above, asserted directly: what is requested is what is mapped."""
    from md_tools.build.top import _sys_document, resolve_build_config

    resolved = resolve_build_config(_config(forcefield={"protein": "ff14SB"},
                                            solvent={"model": "GBn2"}))
    assert "ff14SB" in _sys_document(resolved)["forcefield"]["protein"]


# --- 2. the md-run surface ------------------------------------------------------------------------

def test_the_cli_offers_exactly_four_public_subcommands():
    done = subprocess.run(CLI + ["-h"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    for name in ("build-top", "build-md", "md-run", "data-register"):
        assert name in done.stdout, f"{name} is not offered"


def test_md_run_help_works_without_a_context():
    """`-h` must not initialise OpenMM: help on a machine with no GPU has to work."""
    done = subprocess.run(CLI + ["md-run", "-h"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    for flag in ("-i", "-p", "-x", "-o", "-r", "-log", "--cpu"):
        assert flag in done.stdout, f"{flag} is not documented"


def test_no_separate_md_run_executable_is_installed():
    """One executable. `md-run` is a subcommand of `md-openmm`, not a console script."""
    import tomllib

    scripts = tomllib.load(open(REPO / "pyproject.toml", "rb"))["project"]["scripts"]
    assert list(scripts) == ["md-openmm"], scripts


# --- 3. the CUDA-first platform policy -------------------------------------------------------------

def test_one_platform_policy_serves_every_runtime():
    """Stages, REMD and AIS must not each decide separately whether CPU is acceptable.

    The release notes recorded a real difference: a ladder given no explicit platform fell back to
    the CPU where a stage refused. One resolver removes the possibility of drift.
    """
    from md_tools.openmm import platform_policy

    assert hasattr(platform_policy, "resolve_platform_request")


def test_the_default_policy_is_cuda_and_cpu_needs_an_explicit_request():
    """`from_flags` became `from_machine`: the default now comes from machine.openmm, not a flag.

    The guarantee is unchanged and is what is asserted -- CUDA unless somebody says otherwise, and
    `--cpu` recorded as having been asked for. What changed is where "otherwise" is written down.
    """
    from md_tools.openmm.platform_policy import PlatformRequest

    default = PlatformRequest.default()
    assert default.name == "CUDA"
    assert default.explicit_cpu is False

    from_absent_config = PlatformRequest.from_machine({})
    assert from_absent_config.name == "CUDA" and from_absent_config.explicit_cpu is False

    explicit = PlatformRequest.from_machine({}, cpu=True)
    assert explicit.name == "CPU" and explicit.explicit_cpu is True
    assert explicit.origin == "--cpu (command line)"


# --- 4. AIS ---------------------------------------------------------------------------------------

def test_an_ais_interval_that_does_not_divide_the_switching_path_is_refused():
    """`switching_steps=250` with `info_printout=100` cannot produce a frame at the end."""
    from md_tools.build.md import resolve_md_config
    from md_tools.build.strict import ConfigError

    path = Path(tempfile.mkdtemp()) / "AIS.config"
    path.write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "explicit",
        "ais": {"number_of_paths": 4, "switching_steps": 250,
                "observation_interval_steps": 10},
        "ais_source": {"trajectory": "../source.nc"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 100,
                      "checkpoint_printout": 100}}), encoding="utf-8")
    with pytest.raises(ConfigError) as refusal:
        resolve_md_config(path)
    message = str(refusal.value)
    assert "250" in message and "100" in message, message


def test_ais_intervals_that_divide_the_switching_path_are_accepted():
    from md_tools.build.md import resolve_md_config

    path = Path(tempfile.mkdtemp()) / "AIS.config"
    path.write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "explicit",
        "ais": {"number_of_paths": 4, "switching_steps": 250,
                "observation_interval_steps": 10},
        "ais_source": {"trajectory": "../source.nc"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 50,
                      "checkpoint_printout": 50}}), encoding="utf-8")
    resolved = resolve_md_config(path)
    assert resolved["ais"]["switching_steps"] == 250


def test_ais_path_trajectory_names_are_zero_based_and_padded():
    """`AIS_traj0000.nc` … `AIS_traj0099.nc` for 100 paths, whatever the MPI worker count."""
    from md_tools.ais import path_trajectory_name

    assert path_trajectory_name(0, 100) == "AIS_traj0000.nc"
    assert path_trajectory_name(99, 100) == "AIS_traj0099.nc"
    # wider when the count needs it, so ordering by name stays ordering by path id
    assert path_trajectory_name(0, 100000) == "AIS_traj00000.nc"


def test_ais_path_assignment_is_invariant_to_worker_count():
    """MPI scheduling must never change which global path owns which filename."""
    from md_tools.ais import paths_for_rank

    total = 100
    for size in (1, 2, 3, 8):
        union = []
        for rank in range(size):
            union.extend(paths_for_rank(rank, size, total))
        assert sorted(union) == list(range(total)), f"world size {size} lost or duplicated a path"


# --- 5. duplicate YAML keys ------------------------------------------------------------------------

def test_a_duplicate_key_in_a_build_configuration_is_refused():
    """PyYAML keeps the last value silently, so a file can say two things and run as one."""
    from md_tools.build.strict import ConfigError
    from md_tools.build.top import resolve_build_config

    path = Path(tempfile.mkdtemp()) / "build.config"
    path.write_text("solvent:\n  padding_nm: 1.5\n  padding_nm: 2.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="padding_nm"):
        resolve_build_config(path)


def test_a_duplicate_key_in_a_protocol_configuration_is_refused():
    from md_tools.build.md import resolve_md_config
    from md_tools.build.strict import ConfigError

    path = Path(tempfile.mkdtemp()) / "md.config"
    path.write_text("protocol: cMD\nstages:\n  production_steps: 10\n  production_steps: 20\n",
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="production_steps"):
        resolve_md_config(path)


# --- 6. the configuration and documentation defects -------------------------------------------------

def test_no_shipped_config_has_a_section_key_that_parses_as_a_scalar():
    """MIGRATED from `test_the_cmd_example_has_exactly_one_dynamics_key`.

    THE DEFECT. `cMD.config` carried an uncommented prose line -- `dynamics: minimisation, ...` --
    above the real `dynamics:` block. YAML read the first one, so `dynamics` parsed as a STRING,
    the second occurrence was a duplicate key, and the resolver saw a scalar where a mapping
    belongs.

    WHY IT IS MIGRATED RATHER THAN DELETED. The original asserted that cMD's `dynamics` is a dict,
    which requires the file to STATE a dynamics block. The shipped configs are now minimal
    canonical examples -- cMD states only `protocol` and `solvent`, because everything else
    resolves to the model's own defaults -- so that assertion demands a key the file has no reason
    to carry, and the failure it produced was about the file being minimal, not about the defect.

    The defect class is not specific to `dynamics`, nor to cMD: any prose line above any block, in
    any of the five shipped files, does the same thing. So the check is now that shape -- for every
    shipped config, a key the schema declares as a Section either is absent or parses as a mapping,
    and no top-level key appears twice. That catches the original bug and every sibling of it.
    """
    from md_tools.build.md import MD_SCHEMA

    sections = set(MD_SCHEMA.sections)
    for name in ("cMD", "REST2", "AIS", "umbrella"):
        path = REPO / "configs" / "md" / f"{name}.config"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        document = yaml.safe_load(text) or {}

        for key, value in document.items():
            if key in sections:
                assert isinstance(value, dict), (
                    f"configs/md/{name}.config: `{key}` is a schema Section but parsed as "
                    f"{type(value).__name__} ({value!r}) -- an uncommented prose line above the "
                    f"block will do this")

        stated = [line.split(":", 1)[0] for line in text.splitlines()
                  if line and not line[0].isspace() and not line.startswith("#") and ":" in line]
        duplicates = sorted({k for k in stated if stated.count(k) > 1})
        assert not duplicates, (
            f"configs/md/{name}.config states {duplicates} more than once at the top level; "
            f"YAML keeps one of them and the other is silently lost")


def test_the_scientific_defaults_page_points_at_the_bibliography_that_exists():
    text = (REPO / "docs" / "scientific-defaults.md").read_text(encoding="utf-8")
    assert "md-defaults-references.bib" not in text
    assert "scientific-defaults.bib" in text
    assert (REPO / "docs" / "scientific-defaults.bib").is_file()


def test_the_shipped_build_config_describes_the_warning_only_pair_policy():
    text = (REPO / "configs" / "sys" / "build-top.config").read_text(encoding="utf-8")
    assert "md_build.config" not in text, "it names a configuration file that does not exist"
    assert "Pairing it with TIP3P is refused" not in text, "the retired rejection is still claimed"


def test_the_cli_help_describes_the_resolved_timestep_rather_than_a_fixed_two_femtoseconds():
    done = subprocess.run(CLI + ["build-md", "-h"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0
    assert "auto" in done.stdout, done.stdout
