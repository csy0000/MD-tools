#!/usr/bin/env python
"""A canonical fingerprint of the Hamiltonian a run actually propagated.

Copied verbatim into every generated project that produces or consumes a reservoir.

WHY TAU AND TEMPERATURE ARE NOT ENOUGH
    The probability-one Boltzmann reservoir rule is valid only when the source and the rung it
    refreshes sample the SAME configurational distribution. Matching tau, temperature, topology,
    atom order and box are necessary conditions and none of them is sufficient:

        ff14SB/TIP3P and ff19SB/OPC agree on all five and are different Hamiltonians.
        Two protonation states agree on all five.
        A different constraint set, hydrogen-mass repartitioning, cutoff, PME tolerance, switching
        function or dispersion correction agrees on all five.
        A different solute selection or unscaled-torsion set produces a different SCALED system at
        the same tau.

    So the identity is taken from the serialized OpenMM System itself -- the object that actually
    determines the energy -- reduced to a canonical form and hashed.

NEVER COMPARE TWO STORED CLAIMS
    Both sides recompute. The source records the fingerprint of the System it ran; the reservoir
    validator recomputes the fingerprint of the System the ladder is about to use and compares.
    Comparing two recorded strings would pass happily for two runs that were both mislabelled.

A PATH IS NOT IDENTITY
    A directory called `cMD_tau0p5` establishes nothing. It can be renamed, copied, or simply
    wrong, and a reservoir drawn from the wrong Hamiltonian runs to completion while being wrong.
"""
import hashlib
import re

#: Bumped when the canonicalisation changes, because a fingerprint is only comparable to one
#: produced the same way. v2: the selection records unscaled central bonds of every class and
#: whether impropers are unscaled (REST2 convention v3), where v1 recorded omega bonds only.
FINGERPRINT_FORMAT = "md-tools-hamiltonian-identity/v2"

#: Attributes stripped before hashing: they describe the file, not the physics.
_VOLATILE = (
    re.compile(rb'\s+version="[^"]*"'),
    re.compile(rb'\s+openmmVersion="[^"]*"'),
    re.compile(rb'\s+type="System"'),
)


def canonical_system_xml(system):
    """The serialized System with file-level attributes removed.

    OpenMM writes its own version into the XML, so two identical Systems serialized by different
    OpenMM builds differ byte-for-byte while being the same Hamiltonian. Those attributes are
    stripped; everything that carries a parameter is kept exactly as OpenMM wrote it.
    """
    from openmm import XmlSerializer

    text = XmlSerializer.serialize(system).encode("utf-8")
    for pattern in _VOLATILE:
        text = pattern.sub(b"", text)
    # Normalise line endings only. Whitespace INSIDE the document is OpenMM's own and is stable
    # for a given serializer, so it is not touched: rewriting it would risk hiding a real change.
    return text.replace(b"\r\n", b"\n")


def system_fingerprint(system):
    """A SHA-256 over the canonical System. This is the identity that matters."""
    return hashlib.sha256(canonical_system_xml(system)).hexdigest()


def force_summary(system):
    """A readable summary beside the digest, so a mismatch can be diagnosed without the XML.

    Deliberately NOT the identity: it is a human aid. Two systems agreeing here can still differ,
    which is exactly why the digest exists.
    """
    from openmm import (CustomGBForce, HarmonicAngleForce, HarmonicBondForce, NonbondedForce,
                        PeriodicTorsionForce)

    summary = {
        "particles": int(system.getNumParticles()),
        "constraints": int(system.getNumConstraints()),
        "forces": [system.getForce(i).__class__.__name__
                   for i in range(system.getNumForces())],
        "uses_periodic_boundary_conditions": bool(system.usesPeriodicBoundaryConditions()),
        "total_mass_amu": None,
    }
    try:
        from openmm import unit
        summary["total_mass_amu"] = float(sum(
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())))
    except Exception:
        pass
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce):
            summary["nonbonded"] = {
                "method": int(force.getNonbondedMethod()),
                "cutoff_nm": float(force.getCutoffDistance().value_in_unit_system(
                    __import__("openmm").unit.md_unit_system)),
                "uses_switching_function": bool(force.getUseSwitchingFunction()),
                "uses_dispersion_correction": bool(force.getUseDispersionCorrection()),
                "ewald_error_tolerance": float(force.getEwaldErrorTolerance()),
                "exceptions": int(force.getNumExceptions()),
            }
        elif isinstance(force, PeriodicTorsionForce):
            summary["torsions"] = int(force.getNumTorsions())
        elif isinstance(force, HarmonicBondForce):
            summary["bonds"] = int(force.getNumBonds())
        elif isinstance(force, HarmonicAngleForce):
            summary["angles"] = int(force.getNumAngles())
        elif isinstance(force, CustomGBForce):
            summary["generalised_born_terms"] = int(force.getNumEnergyTerms())
    return summary


def identity_record(system, *, tau, temperature_k, ensemble, solute_indices=(),
                    excluded_bonds=(), unscaled_impropers=True, extra=None):
    """Everything a consumer needs to decide "is this the same Hamiltonian I am about to run".

    `solute_indices` and `excluded_bonds` are included because they determine the SCALED system:
    two runs from one base System with different enhanced regions are different Hamiltonians at
    every tau above zero, and the base digest alone would not see it.
    """
    selection = hashlib.sha256(
        repr(({"solute": [int(i) for i in solute_indices],
               "unscaled_central_bonds": sorted([int(a), int(b)] for a, b in excluded_bonds),
               "unscaled_impropers": bool(unscaled_impropers)}
              )).encode("utf-8")).hexdigest()
    record = {
        "format": FINGERPRINT_FORMAT,
        "system_sha256": system_fingerprint(system),
        "selection_sha256": selection,
        "n_solute_atoms": len(list(solute_indices)),
        "n_unscaled_central_bonds": len(list(excluded_bonds)),
        "unscaled_impropers": bool(unscaled_impropers),
        # `None` means "the unscaled reference the ladder is built from", which is a different
        # claim from "the rung at tau = 0.0" and is recorded as a different value.
        "tau": None if tau is None else float(tau),
        "temperature_k": float(temperature_k),
        "ensemble": str(ensemble),
        "summary": force_summary(system),
        "note": ("the digest is over the canonical serialized OpenMM System: the object that "
                 "determines the energy. Tau, temperature, topology and box are necessary and not "
                 "sufficient -- ff14SB/TIP3P and ff19SB/OPC agree on all of them."),
    }
    if extra:
        record.update(extra)
    return record


class HamiltonianMismatch(ValueError):
    """The two Hamiltonians are not the same, so a probability-one transfer is not justified."""


def require_same_hamiltonian(recorded, current, *, what="reservoir"):
    """Compare a RECORDED identity with a freshly RECOMPUTED one, and refuse a difference.

    `current` must have been computed here, now, from the System about to be used. Comparing two
    stored claims would agree happily for two runs that were both mislabelled.
    """
    if not isinstance(recorded, dict):
        raise HamiltonianMismatch(
            f"the {what} records no Hamiltonian identity, so it cannot be shown to sample the "
            f"same distribution as the state it would refresh. Refusing rather than assuming.")
    if recorded.get("format") != FINGERPRINT_FORMAT:
        raise HamiltonianMismatch(
            f"the {what}'s identity is {recorded.get('format')!r}, not {FINGERPRINT_FORMAT!r}; "
            f"fingerprints are only comparable to ones produced the same way.")

    differences = []
    for key in ("system_sha256", "selection_sha256", "tau", "temperature_k", "ensemble",
                "n_solute_atoms", "n_unscaled_central_bonds", "unscaled_impropers"):
        if recorded.get(key) != current.get(key):
            differences.append((key, recorded.get(key), current.get(key)))
    if not differences:
        return True

    lines = []
    for key, was, now in differences:
        if key == "system_sha256":
            lines.append(
                f"    {key}: {str(was)[:16]}... vs {str(now)[:16]}...  -- the serialized OpenMM "
                f"System differs, so this is a different Hamiltonian even if every other field "
                f"matched (a crossed ff14SB/TIP3P vs ff19SB/OPC pair looks exactly like this)")
        elif key == "selection_sha256":
            lines.append(
                f"    {key}: {str(was)[:16]}... vs {str(now)[:16]}...  -- a different solute "
                f"selection or unscaled-torsion set produces a different SCALED system at every "
                f"tau above zero")
        else:
            lines.append(f"    {key}: was {was!r}, now {now!r}")
    summary_was = (recorded.get("summary") or {})
    summary_now = (current.get("summary") or {})
    for key in sorted(set(summary_was) | set(summary_now)):
        if summary_was.get(key) != summary_now.get(key):
            lines.append(f"    summary.{key}: was {summary_was.get(key)!r}, "
                         f"now {summary_now.get(key)!r}")
    raise HamiltonianMismatch(
        f"the {what} and the state it would refresh are not the same Hamiltonian:\n"
        + "\n".join(lines) + "\n"
        f"  The probability-one Boltzmann rule is justified ONLY when the two sample the same "
        f"configurational distribution. A path or directory name such as `cMD_tau0p5` is not "
        f"identity evidence.")
