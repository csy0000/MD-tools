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
#: v3 (0.6.1): `selection_sha256` hashes the Hamiltonian-determining projection of the
#: md-tools-solute-selection/2.0 document (`hamiltonian_selection_projection`): hot atoms, torsion
#: and CMAP decisions, protected bonds, ligand instances -- because under selective REST2 two
#: selections with one nonbonded atom set can still be two Hamiltonians -- and NOT its provenance.
FINGERPRINT_FORMAT = "md-tools-hamiltonian-identity/v3"
#: What 0.6.0 wrote. Read, and accepted ONLY through `_legacy_v2_agrees` (shared contract §3).
LEGACY_FINGERPRINT_FORMAT = "md-tools-hamiltonian-identity/v2"
LEGACY_SELECTION_MODE = "legacy-full-solute"

#: The fields a v2 identity carried, all of which must match for the compatibility branch.
V2_FIELDS = ("system_sha256", "selection_sha256", "tau", "temperature_k", "ensemble",
             "n_solute_atoms", "n_unscaled_central_bonds", "unscaled_impropers")

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


def _v2_selection_sha256(solute_indices, excluded_bonds, unscaled_impropers):
    """`selection_sha256` computed exactly as v2 did. Kept for the compatibility branch."""
    return hashlib.sha256(
        repr(({"solute": [int(i) for i in solute_indices],
               "unscaled_central_bonds": sorted([int(a), int(b)] for a, b in excluded_bonds),
               "unscaled_impropers": bool(unscaled_impropers)}
              )).encode("utf-8")).hexdigest()


def _legacy_selection_document(solute_indices, excluded_bonds, unscaled_impropers):
    """The md-tools-solute-selection/2.0 content of a legacy full-solute selection, for hashing
    when the caller holds only the three lists. Topology-free: the System digest already binds
    the atoms, and a caller that has the full document passes it instead."""
    bonds = sorted([int(a), int(b)] for a, b in excluded_bonds)
    return {"format": "md-tools-solute-selection/2.0", "selection_mode": LEGACY_SELECTION_MODE,
            "selected_nonbonded_atoms": [int(i) for i in solute_indices],
            "selected_torsion_central_bonds": None, "scaled_cmap_terms": None,
            "excluded_central_bonds": bonds,
            "improper_policy": {"unscaled_impropers": bool(unscaled_impropers)}}


def hamiltonian_selection_projection(document):
    """The part of a selection document (md-tools-solute-selection/2.0, or 1.0) that DETERMINES
    the Hamiltonian -- and nothing else. THE one definition: the identity hashes it,
    `ScalingSelection.digest()` hashes it, and `regions.claimed_region_differences` compares
    through it (S0 ruling, 2026-09-19).

    Kept: the hot nonbonded atoms; the scaled torsion central bonds and the protected ones; per
    CMAP term whether it is scaled; the improper policy; the detector and selection-policy
    versions; per ligand instance its residue identity, its parameter package and the RESOLVED
    content of its exclusion file (the package and the named bonds).

    Left out, as PROVENANCE: mask spellings, instance labels, file paths, the raw bytes of an
    exclusion file (a comment is not a Hamiltonian), the residue-map display fields, notes and
    labels. Two records differing only there describe one Hamiltonian, and a run must resume
    across them -- a field that does not determine the calculation must never make a run
    unresumable (the 20260909 defect).
    """
    document = dict(document or {})
    fmt = document.get("format")
    mode = document.get("selection_mode", LEGACY_SELECTION_MODE) \
        if fmt == "md-tools-solute-selection/2.0" else LEGACY_SELECTION_MODE
    atoms = document.get("selected_nonbonded_atoms")
    if atoms is None:
        atoms = document.get("solute_atoms") or ()
    excluded = document.get("excluded_central_bonds")
    if excluded is None:
        excluded = document.get("unscaled_torsion_central_bonds") or ()
    policy = document.get("improper_policy") or {}
    projection = {
        "selection_mode": mode,
        "selected_nonbonded_atoms": sorted(int(i) for i in atoms),
        "excluded_central_bonds": sorted(sorted(int(a) for a in b) for b in excluded),
        "unscaled_impropers": bool(policy.get("unscaled_impropers", True)),
    }
    if mode == LEGACY_SELECTION_MODE:
        return projection
    instances = []
    for entry in document.get("ligand_instances") or ():
        block = entry.get("torsion_exclusions") or {}
        instances.append({
            "residue_key": [str(x) for x in entry.get("residue_key") or ()],
            "parameter_id": (entry.get("package") or {}).get("parameter_id"),
            "exclusions": {"mode": block.get("mode", "auto"),
                           "parameters": block.get("parameters"),
                           "central_bonds": sorted(sorted(str(n) for n in pair)
                                                   for pair in block.get("central_bonds") or ())},
        })
    projection.update({
        "selected_torsion_central_bonds": sorted(
            sorted(int(a) for a in b) for b in document.get("selected_torsion_central_bonds")
            or ()),
        "cmap": sorted([int(d["term"]), int(d["map"]), bool(d["scaled"])]
                       for d in document.get("cmap_decisions") or ()),
        "detector_policy_version": document.get("detector_policy_version"),
        "policy": document.get("policy"),
        "ligand_instances": sorted(instances, key=lambda e: e["residue_key"]),
    })
    return projection


def selection_identity_sha256(document):
    """sha256 of `hamiltonian_selection_projection(document)`: what `selection_sha256` is."""
    return _canonical_sha256(hamiltonian_selection_projection(document))


def _canonical_sha256(document):
    import json

    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def identity_record(system, *, tau, temperature_k, ensemble, solute_indices=(),
                    excluded_bonds=(), unscaled_impropers=True, selection=None, extra=None):
    """Everything a consumer needs to decide "is this the same Hamiltonian I am about to run".

    `solute_indices` and `excluded_bonds` are included because they determine the SCALED system:
    two runs from one base System with different enhanced regions are different Hamiltonians at
    every tau above zero, and the base digest alone would not see it.

    `selection` is the md-tools-solute-selection/2.0 DOCUMENT (a `scaler.yaml`'s `selection`, or
    `ScalingSelection.to_document()`). Without it the selection is the legacy full solute given
    by the three lists. An explicit selection MUST be passed: from the three lists alone an
    explicit region is indistinguishable from the whole solute.
    """
    # A LEGACY document is ignored in favour of the three lists: a legacy Hamiltonian's identity
    # must not depend on whether its `scaler.yaml` was written before or after 0.6.1 recorded one.
    if selection is None or selection.get("selection_mode", LEGACY_SELECTION_MODE) == \
            LEGACY_SELECTION_MODE:
        selection = _legacy_selection_document(solute_indices, excluded_bonds,
                                               unscaled_impropers)
    mode = selection.get("selection_mode", LEGACY_SELECTION_MODE)
    record = {
        "format": FINGERPRINT_FORMAT,
        "system_sha256": system_fingerprint(system),
        "selection_mode": mode,
        # The Hamiltonian-determining projection ONLY. Provenance (mask text, labels, paths) is
        # recorded with the states and never part of the identity.
        "selection_sha256": selection_identity_sha256(selection),
        # The v2 way, so a 0.6.0 record of a legacy selection can be checked field for field.
        "v2_selection_sha256": _v2_selection_sha256(solute_indices, excluded_bonds,
                                                    unscaled_impropers),
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
    if recorded.get("format") == LEGACY_FINGERPRINT_FORMAT:
        return _legacy_v2_agrees(recorded, current, what=what)
    if recorded.get("format") != FINGERPRINT_FORMAT:
        raise HamiltonianMismatch(
            f"the {what}'s identity is {recorded.get('format')!r}, not {FINGERPRINT_FORMAT!r}; "
            f"fingerprints are only comparable to ones produced the same way.")

    differences = []
    for key in ("system_sha256", "selection_mode", "selection_sha256", "tau", "temperature_k",
                "ensemble", "n_solute_atoms", "n_unscaled_central_bonds", "unscaled_impropers"):
        if recorded.get(key) != current.get(key):
            differences.append((key, recorded.get(key), current.get(key)))
    if not differences:
        return True
    _raise_differences(recorded, current, differences, what=what)


def hamiltonian_identities_agree(recorded, current):
    """True when `require_same_hamiltonian` would accept; False otherwise. For a caller that
    compares a whole identity document key by key (the ladder's `compare_identity`) and must
    treat its `hamiltonian` entry through the ONE rule, compatibility branch included."""
    try:
        return bool(require_same_hamiltonian(recorded, current))
    except HamiltonianMismatch:
        return False


def _legacy_v2_agrees(recorded, current, *, what):
    """THE COMPATIBILITY BRANCH (shared contract §3, decided 2026-09-19): a 0.6.0 (v2) identity.

    Accepted if and only if the current selection is `legacy-full-solute` AND every v2 field
    matches, `selection_sha256` recomputed the v2 way. A legacy run's Hamiltonian is byte-identical
    under 0.6.1, so refusing it would strand every in-flight 0.6.0 ladder for nothing. An explicit
    selection never matches a v2 record: v2 could not describe one.
    """
    mode = current.get("selection_mode")
    if mode != LEGACY_SELECTION_MODE:
        raise HamiltonianMismatch(
            f"the {what} carries a {LEGACY_FINGERPRINT_FORMAT} identity (written before 0.6.1), "
            f"and the state it would continue uses a SELECTIVE REST2 region (selection_mode "
            f"{mode!r}). A v2 identity describes only the whole-solute selection, so it cannot "
            f"be the same Hamiltonian. Start a new run for the selective region.")
    differences = []
    for key in V2_FIELDS:
        now = current.get("v2_selection_sha256") if key == "selection_sha256" \
            else current.get(key)
        if recorded.get(key) != now:
            differences.append((key, recorded.get(key), now))
    if not differences:
        return True
    _raise_differences(recorded, current, differences, what=what)


def _raise_differences(recorded, current, differences, *, what):
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
