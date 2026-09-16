"""Serialise a ladder's rungs to `remd<n>/build_state<n>.xml`, and record how.

THE ARCHITECTURE THIS BELONGS TO. Scaling used to happen at RUN time: every group line named the
same unscaled `built.xml`, and `build_rung_systems` produced the N scaled Systems in memory each
time a ladder started. Moving it to BUILD time makes each rung a file -- so a group line can name
`remd<n>/build_state<n>.xml`, the input stops carrying a scaling instruction, and the Hamiltonian
a state ran under is readable rather than re-derivable.

WHAT THAT COSTS, stated because it is a real hazard rather than a detail: a System that is
already scaled must never be scaled again. Solute-solute would go as `(1-tau)^4` instead of
`(1-tau)^2` -- a wrong Hamiltonian that produces entirely plausible numbers, and nothing
downstream reports it. The refusal for that belongs in the runtime, alongside this; writing these
files is only half the change, which is why nothing should point `-s` at them until the other
half exists.

AIS IS THE EXCEPTION, and by construction rather than by preference. A default switching path
visits 50,001 distinct tau values (50,000 steps at a 1-step update interval), 100 paths to a
campaign: five million Systems. Tau moving continuously at frozen coordinates IS the method, so
AIS keeps the live `TauSwitcher`. Pre-scaled rungs serve the protocols that hold tau FIXED while
they propagate -- fixed-tau cMD, REST2 and rREST2.

THE LOG IS THE POINT AS MUCH AS THE FILES. A serialised System is opaque: it says what the
parameters ARE, never what was done to them. The record beside it says which factors were applied
to which terms, which torsions were left alone and why, and what each file hashes to -- so the
scaling is auditable without deserialising anything or trusting this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["write_rung_systems", "format_scaling_report", "RungWriteError"]


class RungWriteError(RuntimeError):
    """The rungs could not be written, or would not have described what they claim."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rung_systems(run_layout, *, system_path, topology_path, taus,
                       ligand_sdf=None, overwrite: bool = False) -> dict[str, Any]:
    """Write one `build_state<n>.xml` per rung, plus the scaling record. Returns that record.

    `taus` is TAKEN, never recomputed. `remd.generated.tau_ladder` is the one implementation, and
    its docstring records why: a second spelling of the ladder agreed to six decimals and no
    further, which was exactly enough to pass every eye and fail an exact comparison -- a
    `cv_stateN.json` held 0.166667 from the ladder that ran, a resume recomputed
    0.16666666666666666 from the ladder that did not, and every CV-enabled four-rung ladder was
    unresumable. Recomputing here would reintroduce that.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..md.stage import solute_atom_indices
    from ..openmm.system import UnclassifiedOmegaError, omega_exclusions
    from ..remd.protocol import build_rung_systems
    from ..rest2 import REST2_IMPLEMENTATION, scaling_for_tau, torsion_exclusion_report

    system_path, topology_path = Path(system_path), Path(topology_path)
    taus = [float(t) for t in taus]
    if len(taus) < 2:
        raise RungWriteError(f"a ladder needs at least 2 rungs; got {len(taus)}")

    base = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
    pdb = PDBFile(str(topology_path))

    # EXACTLY the sequence the preflight uses (`preflight.py` solute -> omega -> scale), so the
    # rung written here is the rung that would have been built at run time. A second derivation
    # of "what is the solute" is two answers waiting to disagree, and the disagreement would be
    # invisible: both produce a plausible ladder and only the numbers differ.
    solute = solute_atom_indices(pdb.topology)
    # No `route`: the classifier decides per candidate from the residue holding the amide
    # nitrogen, so the caller no longer has to know -- and can no longer get it wrong, which is
    # how the preflight and this writer came to disagree about the same ladder.
    #
    # ENFORCED, and this is the site where it matters most: these files ARE what a grouped ladder
    # integrates. A candidate left out of `omega_unscaled_bonds` used to be scaled here silently.
    try:
        omega = omega_exclusions(pdb.topology, solute, ligand_sdf=ligand_sdf)
    except UnclassifiedOmegaError as refusal:
        raise RungWriteError(str(refusal)) from None
    excluded = [tuple(int(a) for a in bond) for bond in omega.get("omega_unscaled_bonds", [])]

    systems, audit = build_rung_systems(base, solute, tuple(taus), excluded_bonds=excluded)
    if len(systems) != len(taus):
        raise RungWriteError(
            f"the scaler returned {len(systems)} System(s) for {len(taus)} rung(s)")

    states: list[dict[str, Any]] = []
    for index, (tau, system) in enumerate(zip(taus, systems)):
        target = run_layout.state_system(index)
        if target.exists() and not overwrite:
            raise RungWriteError(
                f"{target} already exists. Pass overwrite=True to replace it; a rung file that "
                f"is silently reused would describe a Hamiltonian this build did not produce.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(XmlSerializer.serialize(system), encoding="utf-8")

        solute_solute, solute_environment = scaling_for_tau(tau)
        states.append({
            "state": index,
            "tau": tau,
            "path": str(target.relative_to(run_layout.root.parent)),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
            "scaling": {
                # Written as the numbers AND as the expressions that produced them: a reader
                # checking a factor should not have to know which of tau or (1-tau) was squared.
                "solute_solute": solute_solute,
                "solute_environment": solute_environment,
                "solute_solute_expression": "(1 - tau)^2",
                "solute_environment_expression": "1 - tau",
            },
            "forces": {bucket: [[i, n] for i, n in entries]
                       for bucket, entries in audit.items()
                       if bucket in ("scaled", "unscaled_by_convention", "energy_free")},
        })

    record = {
        "format": "md-tools-rung-scaling/v1",
        "source": {
            "system": str(system_path),
            "system_sha256": _sha256(system_path),
            "topology": str(topology_path),
            "topology_sha256": _sha256(topology_path),
        },
        "convention": dict(REST2_IMPLEMENTATION),
        "solute": {
            "n_atoms": len(solute),
            "atom_range": [min(solute), max(solute)] if solute else None,
        },
        # The exclusion recorded as the TORSIONS it protected, not as bare atom pairs: a stored
        # pair needs a force field to mean anything, and re-deriving the mapping to check it uses
        # assumptions that may not match the ones used here.
        "omega_exclusion": torsion_exclusion_report(base, solute, excluded),
        "ladder": {"n_states": len(taus), "taus": taus,
                   "source": "remd.generated.tau_ladder (taken, not recomputed)"},
        "states": states,
    }

    log = run_layout.root / "build_states.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    record["log"] = str(log)
    return record


def format_scaling_report(record: dict[str, Any]) -> str:
    """The same record as prose, for a build log a person reads."""
    lines = ["REST2 rung Systems", ""]
    convention = record["convention"]
    lines.append(f"  convention            {convention['name']} v{convention['version']}")
    lines.append(f"  solute-solute         {convention['solute_solute_nonbonded_scale']}")
    lines.append(f"  solute-environment    {convention['solute_environment_nonbonded_scale']}")
    lines.append(f"  eligible torsions     {convention['eligible_solute_torsion_scale']}")
    lines.append(f"  left unscaled         bonds, angles, ordinary amide omega")
    lines.append(f"  solute                {record['solute']['n_atoms']} atom(s)")
    omega = record["omega_exclusion"]
    lines.append(f"  omega exclusion       {omega['n_excluded_torsions']} torsion(s) protected "
                 f"across {len(omega['excluded_central_bonds'])} bond(s); "
                 f"{omega['n_scaled_solute_torsions']} solute torsion(s) scaled")
    lines.append("")
    lines.append("  state    tau    (1-tau)^2      1-tau   file")
    for state in record["states"]:
        scaling = state["scaling"]
        lines.append(f"  {state['state']:>5}  {state['tau']:>5.3f}  "
                     f"{scaling['solute_solute']:>9.6f}  {scaling['solute_environment']:>9.6f}   "
                     f"{Path(state['path']).name}")
    return "\n".join(lines)
