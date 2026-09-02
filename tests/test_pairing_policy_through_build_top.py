"""The protein/water pairing policy as `md-openmm build-top` actually delivers it.

`pairing_warnings()` is tested directly elsewhere. That is not the same guarantee: for half the
combinations it described, the function was unreachable, because `_check_pairings` refused
ff19SB + TIP3P before anything could call it. A policy is only as good as the path a user takes to
it, so these run the real command and look at what it actually produced.

Three artefacts, and all three matter for different readers:

* **stderr** -- the person who typed the command, now;
* **built.log** -- the person reading the build a year later;
* **the machine record** -- everything downstream that must not have to parse prose.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from md_tools.build.record import read_record

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

MATCHED = [("ff14SB", "TIP3P"), ("ff19SB", "OPC")]
CROSSED = [("ff14SB", "OPC"), ("ff19SB", "TIP3P")]


def _build(work: Path, protein: str, water: str, *, ligand: str | None = None):
    config = work / "build.config"
    text = f"forcefield:\n  protein: {protein}\nsolvent:\n  model: {water}\n"
    if ligand:
        text += f"solute:\n  ligand_forcefield: {ligand}\n"
    config.write_text(text, encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(config)],
        cwd=work, capture_output=True, text=True, timeout=1800)


@pytest.mark.parametrize("protein, water", CROSSED)
def test_a_crossed_pair_builds_and_says_so_in_all_three_places(protein, water, tmp_path):
    done = _build(tmp_path, protein, water)
    assert done.returncode == 0, done.stdout + done.stderr

    # 1. the person who typed the command
    assert "CROSSED" in done.stderr.upper(), done.stderr

    # 2. the person reading the build later
    log = (tmp_path / "built.log").read_text(encoding="utf-8")
    assert "CROSSED" in log.upper(), log

    # 3. everything downstream
    record = read_record(tmp_path / "built.log")
    warnings = record.get("warnings") or []
    assert [w["code"] for w in warnings] == ["crossed_explicit_pair"], warnings
    warning = warnings[0]

    # It must name the combination chosen AND both matched pairs -- a warning that says only
    # "this is unusual" leaves the reader to find out what the usual thing was.
    assert set(warning["supported_pairs"]) == {"TIP3P", "OPC"}, warning
    assert warning["solvent_model"] == water
    message = warning["message"]
    assert "build and run" in message or "will build" in message, message

    assert (tmp_path / "built.xml").is_file() and (tmp_path / "built.pdb").is_file()
    assert record["status"] == "completed"


@pytest.mark.parametrize("protein, water", MATCHED)
def test_a_matched_pair_builds_silently(protein, water, tmp_path):
    done = _build(tmp_path, protein, water)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "CROSSED" not in done.stderr.upper(), done.stderr
    record = read_record(tmp_path / "built.log")
    assert not [w for w in (record.get("warnings") or [])
                if w["code"] == "crossed_explicit_pair"], record.get("warnings")


@pytest.mark.parametrize("ligand", ["sage-2.2.1", "gaff2"])
def test_the_ligand_forcefield_does_not_change_the_protein_water_verdict(ligand, tmp_path):
    """Sage and GAFF are an orthogonal choice: they parameterise the small molecule, not water."""
    done = _build(tmp_path, "ff14SB", "TIP3P", ligand=ligand)
    assert done.returncode == 0, done.stdout + done.stderr
    record = read_record(tmp_path / "built.log")
    assert not [w for w in (record.get("warnings") or [])
                if w["code"] == "crossed_explicit_pair"], record.get("warnings")


def test_ff19sb_with_gbn2_is_refused_by_the_command_itself(tmp_path):
    """The one combination that cannot be built. Refused, not warned about, and before any work."""
    config = tmp_path / "build.config"
    config.write_text("forcefield:\n  protein: ff19SB\nsolvent:\n  model: GBn2\n",
                      encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "not parameterised for" in done.stderr, done.stderr
    # Nothing was written: a refusal that leaves a half-built system behind is not a refusal.
    assert not (tmp_path / "built.xml").exists()
