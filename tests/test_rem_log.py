"""The Amber26 H-REMD log: exact grammar, and a real cpptraj parse.

The grammar was measured from genuine Amber output shipped with AmberTools26, not inferred from
column names. The decisive test is not that the text looks right -- it is that the installed
cpptraj reads what we write and reconstructs the exchange history we meant.

PLATFORM_POLICY_EXEMPTION: text formatting and a parser round-trip. No dynamics are propagated.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"
sys.path.insert(0, str(TEMPLATES))

import rem_log                                                     # noqa: E402

#: Genuine Amber output, shipped with AmberTools26. The grammar reference for this module.
AMBER_REFERENCE = Path("/path/to/software/md-stack/conda/ambertools26/"
                       "test/h_rem/rem.log.save")
CPPTRAJ = Path("/path/to/software/md-stack/conda/ambertools26/bin/cpptraj")


# --- the grammar, against Amber's own bytes ---------------------------------------------------

@pytest.mark.skipif(not AMBER_REFERENCE.is_file(), reason="AmberTools26 reference log not present")
def test_our_record_matches_amber_column_for_column():
    """Re-render Amber's own first row from its own values and require the bytes back."""
    rows = [l for l in AMBER_REFERENCE.read_text().splitlines() if not l.startswith("#")]
    original = rows[0]
    f = original.split()
    rendered = rem_log.format_row(
        rep=int(f[0]), neighbour=int(f[1]), temp0_k=float(f[2]),
        pot_x1=float(f[3]), pot_x2=float(f[4]), left_fe=float(f[5]), right_fe=float(f[6]),
        success=(f[7] == "T"), rate=float(f[8]))
    assert rendered == original, f"\n  ours: {rendered!r}\n  amber:{original!r}"
    assert len(rendered) == 79


@pytest.mark.skipif(not AMBER_REFERENCE.is_file(), reason="AmberTools26 reference log not present")
def test_every_amber_row_round_trips():
    for original in AMBER_REFERENCE.read_text().splitlines():
        if original.startswith("#"):
            continue
        f = original.split()
        assert rem_log.format_row(
            rep=int(f[0]), neighbour=int(f[1]), temp0_k=float(f[2]), pot_x1=float(f[3]),
            pot_x2=float(f[4]), left_fe=float(f[5]), right_fe=float(f[6]),
            success=(f[7] == "T"), rate=float(f[8])) == original


@pytest.mark.skipif(not AMBER_REFERENCE.is_file(), reason="AmberTools26 reference log not present")
def test_our_header_matches_amber(tmp_path):
    amber = AMBER_REFERENCE.read_text().splitlines()
    ours = rem_log.format_header(10)
    # Amber's line 4/5 name its own files; the rest must match exactly.
    assert ours[0] == amber[0]
    assert ours[1] == amber[1], f"{ours[1]!r} != {amber[1]!r}"
    assert ours[2] == amber[2]
    assert ours[5] == amber[5], "the column header must be byte-identical"
    assert rem_log.format_block_header(1) == amber[6]


def test_success_renders_as_T_and_F():
    row = rem_log.format_row(rep=1, neighbour=2, temp0_k=300.0, pot_x1=-1.0, pot_x2=-2.0,
                             left_fe=0.0, right_fe=0.0, success=True, rate=0.5)
    assert row.split()[7] == "T"
    row = rem_log.format_row(rep=1, neighbour=2, temp0_k=300.0, pot_x1=-1.0, pot_x2=-2.0,
                             left_fe=0.0, right_fe=0.0, success=False, rate=0.5)
    assert row.split()[7] == "F"


# --- projecting committed exchange rows -------------------------------------------------------

def _six_state_exchange(pairs, accept):
    """Committed-row shaped arrays: `proposed`, `accepted`, and a reduced-potential matrix."""
    n = 6
    proposed = np.zeros((n, n), dtype=np.int64)
    accepted = np.zeros((n, n), dtype=np.int64)
    for (i, j) in pairs:
        proposed[i, j] = proposed[j, i] = 1
        if (i, j) in accept:
            accepted[i, j] = accepted[j, i] = 1
    u = np.arange(n * n, dtype=float).reshape(n, n) * 0.1
    return proposed, accepted, u


def test_pairs_come_from_the_committed_matrix_not_the_rule():
    proposed, _, _ = _six_state_exchange([(0, 1), (2, 3), (4, 5)], set())
    assert rem_log.pairs_from_proposed(proposed) == {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def test_an_unpaired_state_uses_itself_as_neighbour():
    """Our sentinel, not Amber's -- Amber always pairs everyone via wrap-around."""
    proposed, accepted, u = _six_state_exchange([(1, 2), (3, 4)], set())
    rows = rem_log.block_rows(n_states=6, proposed=proposed, accepted=accepted, u=u,
                              beta=1.0 / 2.494, temperature_k=300.0, cumulative={})
    assert len(rows) == 6
    assert [int(r.split()[0]) for r in rows] == [1, 2, 3, 4, 5, 6], "Rep# ascending, 1-based"
    assert int(rows[0].split()[1]) == 1, "state 0 unpaired -> Neibr# equals Rep#"
    assert int(rows[5].split()[1]) == 6, "state 5 unpaired -> Neibr# equals Rep#"
    assert int(rows[1].split()[1]) == 3, "state 1 paired with state 2 -> 1-based 3"


def test_free_energies_are_antisymmetric_across_a_pair():
    proposed, accepted, u = _six_state_exchange([(0, 1)], set())
    left, right = rem_log.exchange_free_energies(u, {0: 1, 1: 0}, beta=1.0 / 2.494)
    assert right[0] == pytest.approx(-left[1])
    assert left[0] == 0.0 and right[1] == 0.0, "the ends of an unattempted pair stay zero"


def test_energies_are_converted_to_kcal_per_mole():
    beta = 1.0 / 2.494                                   # 1/kT at 300 K, kJ/mol
    proposed, accepted, u = _six_state_exchange([(0, 1)], {(0, 1)})
    rows = rem_log.block_rows(n_states=6, proposed=proposed, accepted=accepted, u=u,
                              beta=beta, temperature_k=300.0, cumulative={})
    expected = u[0, 0] / beta / rem_log.KJ_PER_KCAL
    assert float(rows[0].split()[3]) == pytest.approx(expected, abs=0.005)


def test_temp0_is_the_physical_temperature_for_every_state():
    """Never an effective temperature: every rung shares one thermostat."""
    proposed, accepted, u = _six_state_exchange([(0, 1), (2, 3), (4, 5)], set())
    rows = rem_log.block_rows(n_states=6, proposed=proposed, accepted=accepted, u=u,
                              beta=1.0 / 2.494, temperature_k=300.0, cumulative={})
    assert {float(r.split()[2]) for r in rows} == {300.0}


def test_the_rate_is_cumulative_across_blocks():
    beta = 1.0 / 2.494
    cumulative = {}
    first = rem_log.block_rows(n_states=6, beta=beta, temperature_k=300.0, cumulative=cumulative,
                               **dict(zip(("proposed", "accepted", "u"),
                                          _six_state_exchange([(0, 1)], {(0, 1)}))))
    assert float(first[0].split()[8]) == pytest.approx(1.0), "1 of 1"
    second = rem_log.block_rows(n_states=6, beta=beta, temperature_k=300.0, cumulative=cumulative,
                                **dict(zip(("proposed", "accepted", "u"),
                                           _six_state_exchange([(0, 1)], set()))))
    assert float(second[0].split()[8]) == pytest.approx(0.5), "1 of 2, carried across blocks"


def test_numexchg_matches_the_number_of_blocks(tmp_path):
    beta = 1.0 / 2.494
    exchanges = [_six_state_exchange([(0, 1), (2, 3), (4, 5)], {(0, 1)}) for _ in range(4)]
    blocks = rem_log.build(n_states=6, exchanges=exchanges, beta=beta, temperature_k=300.0)
    text = rem_log.render(blocks)
    assert "# numexchg is          4" in text
    assert text.count("# exchange ") == 4


def test_write_is_atomic_and_leaves_no_partial_file(tmp_path):
    beta = 1.0 / 2.494
    blocks = rem_log.build(n_states=6, beta=beta, temperature_k=300.0,
                           exchanges=[_six_state_exchange([(0, 1)], set())])
    path = rem_log.write(tmp_path / "rem.log", blocks)
    assert path.is_file()
    assert not list(tmp_path.glob("*.partial.*")), "a temporary file was left behind"
    # Rewriting replaces rather than appends.
    rem_log.write(tmp_path / "rem.log", blocks)
    assert path.read_text().count("# Replica Exchange log file") == 1


# --- the decisive test: the real parser --------------------------------------------------------

@pytest.mark.skipif(not CPPTRAJ.is_file(), reason="AmberTools26 cpptraj is not installed")
def test_cpptraj_parses_what_we_write(tmp_path):
    """Not text inspection: the installed cpptraj must read our log and agree on its shape."""
    beta = 1.0 / 2.494
    exchanges = []
    for step in range(6):
        pairs = [(0, 1), (2, 3), (4, 5)] if step % 2 == 0 else [(1, 2), (3, 4)]
        exchanges.append(_six_state_exchange(pairs, {pairs[0]}))
    blocks = rem_log.build(n_states=6, exchanges=exchanges, beta=beta, temperature_k=300.0)
    log = rem_log.write(tmp_path / "rem.log", blocks)

    (tmp_path / "in").write_text(f"readdata {log} name R\nlist dataset\n")
    done = subprocess.run([str(CPPTRAJ), "-i", str(tmp_path / "in")],
                          capture_output=True, text=True, cwd=tmp_path)
    out = done.stdout + done.stderr
    assert "Error" not in out, out
    assert "as Amber REM log" in out, out
    assert "6 Hamiltonian reps" in out, out
    assert "should contain 6 exchanges" in out, out


@pytest.mark.skipif(not CPPTRAJ.is_file(), reason="AmberTools26 cpptraj is not installed")
def test_cpptraj_reconstructs_the_exchange_history_we_recorded(tmp_path):
    """`remlog crdidx` walks the log and reports where each coordinate set ended up. It must
    follow the swaps we actually committed."""
    beta = 1.0 / 2.494
    # One accepted swap of states 0<->1 at the first exchange, nothing else accepted.
    exchanges = [_six_state_exchange([(0, 1), (2, 3), (4, 5)], {(0, 1)})]
    exchanges += [_six_state_exchange([(1, 2), (3, 4)], set()) for _ in range(3)]
    blocks = rem_log.build(n_states=6, exchanges=exchanges, beta=beta, temperature_k=300.0)
    log = rem_log.write(tmp_path / "rem.log", blocks)

    (tmp_path / "in").write_text(
        f"readdata {log} name R\nrunanalysis remlog R crdidx out crd.dat\n")
    done = subprocess.run([str(CPPTRAJ), "-i", str(tmp_path / "in")],
                          capture_output=True, text=True, cwd=tmp_path)
    assert "Error" not in done.stdout + done.stderr, done.stdout + done.stderr
    produced = (tmp_path / "crd.dat").read_text().splitlines()
    assert len(produced) >= 2, produced
    first = produced[1].split()[1:]
    # states 0 and 1 swapped their coordinate sets; the rest stayed put.
    assert first[0] == "2" and first[1] == "1", produced[:3]
    assert first[2:] == ["3", "4", "5", "6"], produced[:3]
