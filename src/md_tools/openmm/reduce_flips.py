"""Asn/Gln/His flips and histidine tautomers by Reduce, on the structure as it will be simulated.

WHAT THIS DECIDES, and nothing else. The terminal amide of Asn and Gln and the imidazole of His
look nearly the same in electron density turned through 180 degrees, so a deposited structure has
them backwards often enough that MolProbity flips them by default (Word et al. 1999, J. Mol. Biol.
285:1735; Williams et al. 2018, Protein Sci. 27:293). A wrong amide puts an N-H where an acceptor
belongs, and a wrong ring is a 180-degree side-chain rotation no equilibration undoes. Reduce
chooses, for each group, the orientation and (for His) which ring nitrogen carries a proton by
optimising the whole hydrogen-bond and contact network. This module runs it and returns:

  * per Asn/Gln/His: kept or flipped, Reduce's decision code and score, and whether it marked a
    clash;
  * per His: HID, HIE or HIP, read from the hydrogens Reduce placed (HD1 and/or HE2);
  * the heavy-atom coordinates of every FLIPPED residue, from Reduce's output, which since 4.x
    re-docks the flipped head onto the original atoms instead of renaming them in place.

Reduce's own hydrogens are discarded: `addHydrogens` places every hydrogen from the force field's
templates afterwards, at the force field's nuclear X-H geometry. `-NUClear` is passed anyway, so the
contacts Reduce scores are the nuclear-position ones and not its X-ray electron-cloud default, which
is 0.1-0.15 A shorter.

THE COMPLEX. Run after the ligands are placed with their package hydrogens, so a flip next to a
ligand sees the ligand's donors and acceptors. A frozen (mapped) ligand instance is written as
residue `UNL`, which Reduce's dictionary defines with no atoms: Reduce adds nothing to it, never
moves it, and scores contacts against its atoms, hydrogens included. Nothing here changes a frozen
residue.

Reduce missing, or failing, is an ERROR when `protonation.flips: reduce` -- a silent skip would
build a different protein and record that the flips had been decided. `flips: none` is the choice
to keep the deposited orientations and OpenMM's histidine heuristic.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .protonation import Key, ProtonationError, format_key, residue_key

FLIP_METHODS = ("reduce", "none")
DEFAULT_FLIPS = "reduce"

#: The flags, fixed. -FLIP is -HIS -OH -ROTEXOH plus NQH flips; -NUClear scores nuclear X-H
#: positions; -Quiet keeps the console out of the record. Nothing else is tunable from md-tools.
REDUCE_FLAGS = ("-FLIP", "-NUClear", "-Quiet")

#: The residue name a frozen instance is written under. `UNL` ("unknown ligand") is in Reduce's
#: wwPDB het dictionary with ZERO atoms, so Reduce adds no hydrogen to it. A package's own name
#: could be a real chemical-component code Reduce would protonate from the dictionary.
FROZEN_NAME = "UNL"

#: Residue names written for Reduce: every variant name back to the PDB name its tables know.
PDB_NAME = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
            "CYX": "CYS", "CYM": "CYS"}

#: What Reduce decides for. Everything else in its output is ignored.
FLIPPABLE = ("ASN", "GLN", "HIS")

#: A `USER  MOD` line for a flippable group, e.g.
#:   USER  MOD Single : A1146 HIS     :FLIP no HD1:sc=    2.57  F(o=2.1,f=2.6)
#:   USER  MOD Set 1.2: A1177 GLN     :FLIP  amide:sc=    1.72  F(o=3.9,f=4.8)
#: chain (1 char), resid (4, right-justified), insertion code (1), residue name, then the state,
#: the score of the chosen orientation (`!` appended when it clashes), and K (kept) / F (flipped) /
#: X (cannot decide) / C (clash) with the original (o) and flipped (f) scores, `!` marking a bad
#: bump.
_MOD = re.compile(
    r"^USER  MOD (?:Single|Set\s*[\d.]+)\s*:\s?(?P<chain>.)(?P<resid>.{4})(?P<icode>.)"
    r"(?P<resname>\S{3})\s*:(?P<flip>FLIP)?\s*(?P<state>[^:]*?)\s*:sc=\s*"
    r"(?P<score>-?[\d.]+(?:[eE][-+]?\d+)?)(?P<bang>!?)\s+(?P<code>[A-Z])\((?P<detail>[^)]*)\)")

#: A group Reduce declined to adjust, a histidine or an amide next to a metal, in either form:
#:   USER  MOD NoAdj-H:     2 HIS HE2 :     2 HIS NE2 : Z   1  ZNZN   :(H bumps)
#:   USER  MOD NoAdj  : E  81 ASNHD21 : E  81 ASN OD1 : E 280  CACA   :(metal ligand)
#: The atom field is four columns, so a four-character name (HD21) abuts the residue name.
#: Its orientation and tautomer are NOT Reduce's decision, and are not taken as one.
_NOADJ = re.compile(
    r"^USER  MOD NoAdj(?:-H)?\s*:\s?(?P<chain>.)(?P<resid>.{4})(?P<icode>.)(?P<resname>\S{3})"
    r"\s?(?P<atom>\S+)\s*:(?P<rest>.*)$")
#: The reason Reduce gives at the end of a NoAdj line: `(metal ligand)`, `(H bumps)`, ...
_NOADJ_REASON = re.compile(r"\(([^)]*)\)\s*$")


class ReduceError(ProtonationError):
    """Reduce asked for and not available, or a run that did not produce a usable answer."""


def reduce_executable() -> str:
    path = shutil.which("reduce")
    if path is None:
        raise ReduceError(
            "protonation.flips is reduce (the default), but Reduce is not on PATH in this "
            "environment. Install it (`conda install -c conda-forge -c bioconda reduce=4.15`, "
            "pinned in environment.yml) or set protonation.flips: none deliberately, which keeps "
            "the deposited Asn/Gln/His orientations and OpenMM's histidine heuristic. Nothing was "
            "built.")
    return path


def reduce_version(executable: str) -> str:
    """`reduce.4.15.250408` -> `4.15.250408`."""
    out = subprocess.run([executable, "-Version"], capture_output=True, text=True, timeout=60)
    text = (out.stdout or "") + (out.stderr or "")
    match = re.search(r"reduce\.(\d+\.\d+(?:\.\d+)?)", text)
    if match is None:
        raise ReduceError(f"could not read Reduce's version from `{executable} -Version`: "
                          f"{text.strip()[:200]!r}")
    return match.group(1)


def _pdb_key(key: Key) -> Key:
    """A residue key as it reads back from a PDB line: a blank chain id is the empty string."""
    return (key[0].strip(), key[1], key[2])


def candidates(topology, frozen: Iterable[Key] = ()) -> list[Key]:
    """The non-frozen Asn, Gln and His residues: what Reduce would be asked about."""
    frozen = set(frozen)
    return [residue_key(r) for r in topology.residues()
            if (PDB_NAME.get(r.name, r.name) in FLIPPABLE) and residue_key(r) not in frozen]


def _reduce_input(topology, positions, frozen: set[Key]) -> str:
    """The structure as PDB text with Reduce's residue names; ids kept, so keys round-trip."""
    import io

    from openmm import app

    if any(len(str(chain.id)) > 1 for chain in topology.chains()):
        raise ReduceError(
            "a chain id longer than one character cannot be written to the PDB file Reduce reads, "
            "so its residues could not be matched back. Rename the chains, or set "
            "protonation.flips: none.")
    buffer = io.StringIO()
    app.PDBFile.writeFile(topology, positions, buffer, keepIds=True)
    rename: dict[Key, str] = {}
    for residue in topology.residues():
        key = residue_key(residue)
        if key in frozen:
            rename[_pdb_key(key)] = FROZEN_NAME
        elif residue.name in PDB_NAME:
            rename[_pdb_key(key)] = PDB_NAME[residue.name]
    lines = []
    for line in buffer.getvalue().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM", "TER")) and len(line) > 26:
            key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            if key in rename:
                name = rename[key]
                record = "HETATM" if name == FROZEN_NAME and line[:3] != "TER" else line[:6]
                line = record + line[6:17] + f"{name:>3}" + line[20:]
        elif line.startswith("CONECT"):
            continue                 # Reduce finds bonds itself; OpenMM's CONECT lists are partial
        lines.append(line)
    return "".join(lines)


def _parse_output(text: str) -> tuple[dict[Key, dict[str, Any]], dict[Key, dict[str, tuple]]]:
    """(the USER MOD decision per residue, {key: {atom name: (x, y, z) in A}} for every residue)."""
    decisions: dict[Key, dict[str, Any]] = {}
    atoms: dict[Key, dict[str, tuple]] = {}
    for line in text.splitlines():
        if line.startswith("USER  MOD NoAdj"):
            match = _NOADJ.match(line)
            if match is None or match.group("resname") not in FLIPPABLE:
                continue
            key = (match.group("chain").strip(), match.group("resid").strip(),
                   match.group("icode").strip())
            # An Asn on a calcium gets one line per amide hydrogen; they say the same thing.
            reason = _NOADJ_REASON.search(match.group("rest"))
            decisions.setdefault(key, {
                "flipped": False, "code": "NoAdj", "state": "not adjusted", "score": None,
                "clash": "bump" in match.group("rest"),
                "scores": reason.group(1) if reason else match.group("rest").strip(" :"),
                "line": line.rstrip()})
        elif line.startswith("USER  MOD"):
            match = _MOD.match(line)
            if match is None or match.group("resname") not in FLIPPABLE:
                continue
            key = (match.group("chain").strip(), match.group("resid").strip(),
                   match.group("icode").strip())
            decisions[key] = {"flipped": bool(match.group("flip")), "code": match.group("code"),
                              "state": match.group("state").strip(),
                              "score": float(match.group("score")),
                              "clash": bool(match.group("bang")) or "!" in match.group("detail"),
                              "scores": match.group("detail"), "line": line.rstrip()}
        elif line.startswith(("ATOM  ", "HETATM")):
            key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            atoms.setdefault(key, {})[line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return decisions, atoms


def run_reduce(topology, positions, *, frozen_residues: Iterable[Key] = (),
               workdir: Path) -> dict[str, Any]:
    """Reduce on the structure AS GIVEN (protein hydrogens already deleted, frozen ones kept).

    Returns `decisions` ({key: record}), `coordinates` ({key: {atom: (x, y, z) A}} for the
    flipped residues only) and the provenance: version, flags, and the sha256 of the input and
    output files, which are kept in `workdir` as reduce_input.pdb / reduce_output.pdb / reduce.log.
    """
    frozen = set(frozen_residues)
    executable = reduce_executable()
    version = reduce_version(executable)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    inp, out, log = (workdir / name for name in ("reduce_input.pdb", "reduce_output.pdb",
                                                 "reduce.log"))
    inp.write_text(_reduce_input(topology, positions, frozen), encoding="utf-8")
    command = [executable, *REDUCE_FLAGS, inp.name]
    try:
        done = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise ReduceError(f"Reduce {version} did not finish within an hour on {inp}. "
                          f"Nothing was built.") from None
    out.write_text(done.stdout, encoding="utf-8")
    log.write_text(done.stderr, encoding="utf-8")
    # Reduce 4.15 exits 0 even when its input file does not exist, so the exit status is not the
    # test. An output with no coordinates is.
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in done.stdout.splitlines()):
        tail = " | ".join(done.stderr.strip().splitlines()[-5:])
        raise ReduceError(f"Reduce {version} wrote no coordinates for {inp} (exit "
                          f"{done.returncode}): {tail}. Nothing was built.")

    decisions, atoms = _parse_output(done.stdout)
    residues = {residue_key(r): r for r in topology.residues()}
    records: dict[Key, dict[str, Any]] = {}
    coordinates: dict[Key, dict[str, tuple]] = {}
    for key in candidates(topology, frozen):
        residue = residues[key]
        name = PDB_NAME.get(residue.name, residue.name)
        decision = decisions.get(_pdb_key(key))
        placed = atoms.get(_pdb_key(key), {})
        if decision is None:
            pdb = _pdb_key(key)
            tag = f"{pdb[0] or ' '}{pdb[1]:>4}{pdb[2] or ' '}{name}"
            unread = [line.rstrip() for line in done.stdout.splitlines()
                      if line.startswith("USER  MOD") and tag in line]
            said = (f" Reduce did write {len(unread)} line(s) about it that md-tools could not "
                    f"read: {' | '.join(unread[:3])}" if unread else "")
            raise ReduceError(
                f"Reduce {version} reported no decision for {format_key(key)} ({residue.name}); "
                f"its log is {log}.{said} A residue Reduce did not consider would silently keep "
                f"its deposited orientation. Nothing was built.")
        record = {"chain": key[0], "resid": key[1], "insertion_code": key[2],
                  "residue": residue.name, "group": "imidazole" if name == "HIS" else "amide",
                  "flipped": decision["flipped"], "decision": decision["code"],
                  "score": decision["score"], "scores": decision["scores"],
                  "clash": decision["clash"], "line": decision["line"]}
        if decision["code"] == "NoAdj":
            # Reduce would not move this group (a histidine on a metal, typically), so the
            # hydrogens it left are not a tautomer decision: the build's own rule decides, and
            # the record says Reduce declined.
            record["tautomer"] = None
        elif name == "HIS":
            hd1, he2 = "HD1" in placed, "HE2" in placed
            record["tautomer"] = {(True, True): "HIP", (True, False): "HID",
                                  (False, True): "HIE"}.get((hd1, he2))
            if record["tautomer"] is None:
                raise ReduceError(f"Reduce {version} placed neither HD1 nor HE2 on "
                                  f"{format_key(key)}; see {log}. Nothing was built.")
        if decision["flipped"]:
            heavy = [a.name for a in residue.atoms()
                     if a.element is not None and a.element.symbol != "H"]
            missing = [a for a in heavy if a not in placed]
            if missing:
                raise ReduceError(f"Reduce's output lacks {', '.join(missing)} of the flipped "
                                  f"{format_key(key)}; see {out}. Nothing was built.")
            coordinates[key] = {a: placed[a] for a in heavy}
            # In the record, so a rebuild (export-reference) applies these exact numbers and
            # never needs Reduce: the flip is part of the build's account, as a variant is.
            record["coordinates_angstrom"] = {a: list(placed[a]) for a in heavy}
        records[key] = record
    return {
        "version": version,
        "flags": list(REDUCE_FLAGS),
        "frozen_as": FROZEN_NAME,
        "input_pdb": inp.name,
        "input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
        "output_pdb": out.name,
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "log": log.name,
        "decisions": records,
        "coordinates": coordinates,
    }


def apply_flips(topology, positions, coordinates: Mapping[Key, Mapping[str, tuple]]):
    """Positions with every flipped residue's heavy atoms moved to Reduce's; nothing else moves."""
    from openmm import Vec3, unit

    xyz = list(positions.value_in_unit(unit.nanometer))
    for residue in topology.residues():
        placed = coordinates.get(residue_key(residue))
        if not placed:
            continue
        for atom in residue.atoms():
            if atom.name in placed:
                x, y, z = placed[atom.name]
                xyz[atom.index] = Vec3(x / 10.0, y / 10.0, z / 10.0)
    return unit.Quantity(xyz, unit.nanometer)


def his_states(decisions: Mapping[Key, Mapping[str, Any]]) -> dict[Key, str]:
    """{key: HID|HIE|HIP} for every histidine Reduce decided."""
    return {key: d["tautomer"] for key, d in decisions.items() if d.get("tautomer")}


def summary(decisions: Mapping[Key, Mapping[str, Any]]) -> dict[str, int]:
    flipped = [d for d in decisions.values() if d["flipped"]]
    return {"considered": len(decisions), "flipped": len(flipped),
            "uncertain": sum(1 for d in decisions.values() if d["decision"] == "X"),
            "not_adjusted": sum(1 for d in decisions.values() if d["decision"] == "NoAdj"),
            "clash": sum(1 for d in decisions.values() if d["clash"])}


__all__ = ["FLIP_METHODS", "DEFAULT_FLIPS", "ReduceError", "run_reduce", "apply_flips",
           "his_states", "candidates", "summary"]
