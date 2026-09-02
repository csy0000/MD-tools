"""Resolve the integration timestep against the masses actually serialised in a System.

`build-md` cannot answer this question. It writes run scripts from a configuration file and never
opens `built.xml`, so it does not know whether hydrogen mass repartitioning was applied — and a
configuration that *claims* HMR is not evidence that the System has it. The two are written at
different times by different commands, and the one that matters is the System.

So the configuration may say `timestep_fs: auto`, and the value is resolved where the System is
first loaded: 2 fs on ordinary hydrogen masses, 4 fs on repartitioned ones. An explicit value is
honoured where it is safe and refused where it is not, with the same masses as the evidence.

Step counts stay authoritative throughout. A physical duration is only ever derived *after* the
numerical timestep is known, which is why nothing upstream of this module quotes picoseconds.
"""
from __future__ import annotations

from typing import Any

__all__ = ["AUTO", "ORDINARY_TIMESTEP_FS", "HMR_TIMESTEP_FS", "SAFE_UNREPARTITIONED_FS",
           "HMR_MASS_THRESHOLD_AMU", "heaviest_hydrogen_mass_amu", "resolve_timestep_fs"]

#: What a configuration writes to defer the decision.
AUTO = "auto"

#: Ordinary hydrogen masses: X-H stretch constrained, X-H *angle* motion around 10 fs.
ORDINARY_TIMESTEP_FS = 2.0

#: Repartitioned hydrogen masses. The value the HMR literature supports for the 3.024 amu target.
HMR_TIMESTEP_FS = 4.0

#: The largest step this package will run without repartitioning. Above it, HMR is required.
SAFE_UNREPARTITIONED_FS = 3.0

#: A hydrogen heavier than this had mass moved into it. An ordinary hydrogen is ~1.008 amu and a
#: repartitioned one ~3.024, so anything in between is unambiguous either way.
HMR_MASS_THRESHOLD_AMU = 2.0


def heaviest_hydrogen_mass_amu(system, topology) -> float | None:
    """The heaviest hydrogen in the System, or None if it contains none.

    Deliberately the heaviest rather than the mean: water is never repartitioned, so a correctly
    repartitioned solvated system contains BOTH ordinary and heavy hydrogens and an average would
    sit between them and answer neither question.
    """
    from openmm import unit

    masses = [system.getParticleMass(atom.index).value_in_unit(unit.dalton)
              for atom in topology.atoms()
              if atom.element is not None and atom.element.symbol == "H"]
    return max(masses) if masses else None


def resolve_timestep_fs(requested: Any, system, topology) -> dict[str, Any]:
    """Decide the numerical timestep, and say what decided it.

    ``requested`` is whatever the resolved configuration carries: the string ``"auto"``, or a
    number. Returns a record rather than a bare float, because the *basis* of the choice belongs
    in the log next to the value:

        {"timestep_fs": 4.0, "requested": "auto", "basis": "hmr_masses",
         "heaviest_hydrogen_amu": 3.024, "hmr_detected": True}

    Rules, all decided from the masses:

    ==========================  ===================  =========================================
    requested                   System               outcome
    ==========================  ===================  =========================================
    ``auto``                    ordinary hydrogens   2 fs
    ``auto``                    repartitioned        4 fs
    explicit <= 3 fs            either               honoured -- 2 fs on an HMR System is
                                                     slower than it needs to be, not wrong
    explicit > 3 fs             repartitioned        honoured
    explicit > 3 fs             ordinary hydrogens   **refused, before anything integrates**
    ==========================  ===================  =========================================
    """
    heaviest = heaviest_hydrogen_mass_amu(system, topology)
    hmr = heaviest is not None and heaviest >= HMR_MASS_THRESHOLD_AMU

    def record(value: float, basis: str) -> dict[str, Any]:
        return {"timestep_fs": float(value), "requested": requested, "basis": basis,
                "heaviest_hydrogen_amu": (round(heaviest, 6) if heaviest is not None else None),
                "hmr_detected": bool(hmr)}

    if isinstance(requested, str):
        if requested.strip().lower() != AUTO:
            raise SystemExit(
                f"dynamics.timestep_fs is {requested!r}. It must be a number, or {AUTO!r} to "
                f"resolve it from the masses in the built System.")
        if heaviest is None:
            # No hydrogens at all: nothing to repartition and nothing to be limited by. The
            # conservative value is the honest one, and the basis says why.
            return record(ORDINARY_TIMESTEP_FS, "no_hydrogens")
        return record(HMR_TIMESTEP_FS if hmr else ORDINARY_TIMESTEP_FS,
                      "hmr_masses" if hmr else "ordinary_masses")

    value = float(requested)
    if value <= SAFE_UNREPARTITIONED_FS or hmr:
        return record(value, "explicit")

    raise SystemExit(
        f"dynamics.timestep_fs is {value} fs, but the heaviest hydrogen in this System is "
        f"{heaviest:.3f} amu -- hydrogen mass repartitioning was NOT applied when it was built. "
        f"Above {SAFE_UNREPARTITIONED_FS} fs the X-H angle motion is no longer resolved, and the "
        f"run would produce a trajectory that is wrong rather than a failure that is obvious.\n"
        f"  Either rebuild with `hydrogen_mass_repartitioning.enabled: true` in the build "
        f"configuration, or set `dynamics.timestep_fs` to {ORDINARY_TIMESTEP_FS} or to '{AUTO}'.\n"
        f"  Nothing has been integrated.")
