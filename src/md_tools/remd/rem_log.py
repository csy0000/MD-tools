"""The Amber26 Hamiltonian-REMD replica log, as a projection of committed exchange rows.

`rem.log` is NOT an authority. `exchange.nc` is. This module renders a deterministic, Amber-shaped
view of rows that are already committed there, so a reader with cpptraj can follow the same history
the NetCDF records. It is regenerated in full and replaced atomically, which is what makes it
crash-safe: a truncated or half-written log is never left behind, and nothing here can disagree
with the authoritative record because nothing here has its own state.

GRAMMAR, MEASURED NOT INFERRED
    The layout below was measured from genuine Amber output shipped with AmberTools26
    (`test/h_rem/rem.log.save`, a real 4-replica H-REMD log), and a log written to it was then
    parsed by the installed cpptraj, which reported the correct replica and exchange counts with no
    errors. The Amber26 manual PDF is not installed on this machine and the cpptraj parser source
    is not shipped, so the shipped log plus the parser binary are the ground truth used here.

        Rep#      %6d      1-based state index
        Neibr#    %6d      1-based partner state index
        Temp0     %10.2f   the ONE physical thermostat temperature, never an effective temperature
        PotE(x_1) %10.2f   U_i(x_i), this state's own configuration in its own Hamiltonian
        PotE(x_2) %10.2f   U_i(x_j), the partner's configuration in this state's Hamiltonian
        left_fe   %10.2f   the pair (i-1, i) exchange free-energy difference
        right_fe  %10.2f   the pair (i, i+1) exchange free-energy difference
        Success   %5s      T or F
        rate      %12.2f   cumulative acceptance for the pair (i, i+1)

    Energies are kcal/mol; OpenMM works in kJ/mol and the conversion happens here, once.

TWO POINTS THAT ARE OURS, NOT AMBER'S
    Amber's own logs always pair every replica within a block, using a wrap-around pairing. This
    repository's exchange rule uses alternating adjacent pairs WITHOUT wrap-around, so with an even
    number of states two replicas are unpaired on alternate attempts. There is no Amber example of
    an unpaired-replica sentinel; `Neibr# = Rep#` is used, which cpptraj parses cleanly, and it is
    recorded here as this repository's choice rather than an Amber convention.

    Amber's `Success rate` column could not be reproduced from any documented reading: it reports
    2.00 at the first exchange of the shipped log, and the two rows of one pair disagree. Its
    influence on the consumer was measured instead -- replacing the whole column with a constant
    leaves cpptraj's `remlog crdidx` reconstruction identical -- so the column is not load-bearing.
    What is written here is a plainly defined cumulative acceptance for the pair (i, i+1). It is
    NOT claimed to reproduce Amber's own value.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: The format this module writes, recorded in provenance beside the file it produces.
REM_LOG_FORMAT = "amber26-hremd/v1"

#: OpenMM reports kJ/mol; Amber logs kcal/mol.
KJ_PER_KCAL = 4.184

#: The header, byte-identical in shape to Amber's own.
HEADER_COLUMNS = ("# Rep#, Neibr#, Temp0, PotE(x_1), PotE(x_2), left_fe, right_fe, Success, "
                  "Success rate (i,i+1)")


def format_row(*, rep, neighbour, temp0_k, pot_x1, pot_x2, left_fe, right_fe, success, rate):
    """One record. 79 characters, in the field widths measured from Amber's own output."""
    return (f"{int(rep):6d}{int(neighbour):6d}{float(temp0_k):10.2f}"
            f"{float(pot_x1):10.2f}{float(pot_x2):10.2f}"
            f"{float(left_fe):10.2f}{float(right_fe):10.2f}"
            f"{('T' if success else 'F'):>5s}{float(rate):12.2f}")


def format_header(n_exchanges, *, remlog_name="rem.log", remtype_name="rem.type"):
    """The six-line preamble. `numexchg` must equal the number of blocks that follow: cpptraj
    reads it, and reports a mismatch against what it actually finds."""
    return [
        "# Replica Exchange log file",
        f"# numexchg is {int(n_exchanges):10d}",
        "# REMD filenames:",
        f"#   remlog= {remlog_name}",
        f"#   remtype= {remtype_name}",
        HEADER_COLUMNS,
    ]


def format_block_header(exchange_number):
    """1-based, as Amber writes it.

    ACROSS AN EXTENSION BOUNDARY the numbering is SEGMENT-LOCAL: an extension's `rem.log` starts
    again at exchange 1, exactly as Amber's does when a run is restarted, and its `numexchg` is
    that segment's own count. Two reasons, in order:

      * `numexchg` must equal the number of blocks in the file -- cpptraj checks it and reports a
        mismatch -- so a file numbered 7..12 with `numexchg 6` is self-inconsistent by that check;
      * it is what Amber does, and this format exists to be read by tools that expect Amber.

    cpptraj was measured on both conventions and parses either (it does not read the block number
    at all), so this is a deliberate choice rather than a constraint. The absolute position of the
    segment is not lost: the extension provenance records `first_exchange_number`, which is where
    this segment's block 1 falls in the chain.
    """
    return f"# exchange {int(exchange_number):8d}"


def render(blocks, **header):
    """The complete file: header, then one block per exchange, each with one row per state."""
    lines = format_header(len(blocks), **header)
    for number, rows in enumerate(blocks, start=1):
        lines.append(format_block_header(number))
        lines.extend(rows)
    return "\n".join(lines) + "\n"


# --- projecting committed exchange rows ------------------------------------------------------

def pairs_from_proposed(proposed):
    """`{state: partner}` for one exchange, from the committed proposal matrix.

    The matrix is the authority for who was paired with whom; nothing is re-derived from the rule,
    which could have been reconfigured since. A state with no proposal is unpaired.
    """
    proposed = np.asarray(proposed)
    partner = {}
    n = proposed.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and proposed[i, j]:
                partner[i] = j
    return partner


def exchange_free_energies(u, partner, beta):
    """`right_fe` per state, in kcal/mol, and its antisymmetric partner `left_fe`.

    For an attempted pair (i, j) the exchange argument is

        delta = (u_ii + u_jj) - (u_ij + u_ji)

    in units of kT. Dividing by beta gives kJ/mol, and the conversion to kcal/mol happens once, at
    the boundary. `right_fe` is carried by the lower-indexed member of a pair and `left_fe` by the
    higher one, with the opposite sign -- which is the antisymmetry Amber's own logs show between
    the two rows of a pair.
    """
    u = np.asarray(u, dtype=float)
    n = u.shape[0]
    right = np.zeros(n)
    left = np.zeros(n)
    for i, j in partner.items():
        if i >= j:
            continue
        delta_kt = (u[i, i] + u[j, j]) - (u[i, j] + u[j, i])
        delta_kcal = delta_kt / beta / KJ_PER_KCAL
        right[i] = delta_kcal
        left[j] = -delta_kcal
    return left, right


def block_rows(*, n_states, proposed, accepted, u, beta, temperature_k, cumulative):
    """One exchange block: `n_states` rows, `Rep#` ascending and 1-based.

    `cumulative` is mutated: `{pair_lower_index: [accepted, proposed]}` carried across blocks, so
    the rate column is genuinely cumulative over the run rather than per block.
    """
    proposed = np.asarray(proposed)
    accepted = np.asarray(accepted)
    u = np.asarray(u, dtype=float)
    partner = pairs_from_proposed(proposed)
    left, right = exchange_free_energies(u, partner, beta)

    for i, j in partner.items():
        if i < j:
            counts = cumulative.setdefault(i, [0, 0])
            counts[1] += 1
            if accepted[i, j]:
                counts[0] += 1

    rows = []
    for state in range(n_states):
        mate = partner.get(state)
        paired = mate is not None
        # No Amber ground truth for an unpaired replica; see the module docstring.
        neighbour = mate if paired else state
        success = bool(paired and accepted[state, mate])
        lower = min(state, mate) if paired else state
        got, tried = cumulative.get(lower, [0, 0])
        own = u[state, state] / beta / KJ_PER_KCAL
        other = (u[state, mate] / beta / KJ_PER_KCAL) if paired else own
        rows.append(format_row(
            rep=state + 1, neighbour=neighbour + 1, temp0_k=temperature_k,
            pot_x1=own, pot_x2=other, left_fe=left[state], right_fe=right[state],
            success=success, rate=(got / tried if tried else 0.0)))
    return rows


def build(*, n_states, exchanges, beta, temperature_k):
    """Every block, from an iterable of committed `(proposed, accepted, u)` triples."""
    cumulative = {}
    return [block_rows(n_states=n_states, proposed=p, accepted=a, u=u, beta=beta,
                       temperature_k=temperature_k, cumulative=cumulative)
            for p, a, u in exchanges]


def write(path, blocks, **header):
    """Replace the log atomically.

    A full rewrite from committed rows, never an append: an appended log can be torn by a crash and
    then disagree with the authoritative record, and there is no way to tell from the file which
    half is true. Rewriting cannot leave that state.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(render(blocks, **header), encoding="utf-8")
    os.replace(temporary, path)
    return path
