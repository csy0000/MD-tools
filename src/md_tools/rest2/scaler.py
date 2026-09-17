"""REST2 Hamiltonian scaling, at a fixed tau per System.

This is the AMBER-compatible convention, expressed in tau:

    s        = (1 - tau)^2        solute-solute terms
    (1 - tau)     solute-environment terms

so that for a pairwise nonbonded force

    U_tau = (1-tau)^2 * U_solute-solute  +  (1-tau) * U_solute-environment  +  U_environment

Charges scale by sqrt(s) and epsilons by s, which produces exactly that split without needing a
custom force. Solute torsions scale by s; CMAP maps that are wholly within the solute scale by s.
Torsions about an omega bond are LEFT ALONE -- scaling them lets a peptide bond rotate at the hot
rungs, so the ladder samples cis/trans interconversion the cold rung never sees, and the exchange
no longer connects two states of the same system.

Every rung is thermostatted at the same temperature. "Effective solute temperature" is a way of
describing the scaling, not a second thermostat: beta is common to the whole ladder.

A tau is fixed for the life of a System: a ladder builds one per rung and a fixed-tau cMD run
builds one. Nothing here moves tau on a live Context. That was `TauSwitcher`, and it served only
the single-topology AIS, which was retired in 0.5.4 -- AIS now mixes two end-state Systems
(`md_tools.ais.two_state`).
"""

# The rung-building code lives in `hamiltonian.py`, which imports only OpenMM so that a
# reference bundle can carry it verbatim. Re-exported here: one implementation, every caller
# unchanged.
from .hamiltonian import (  # noqa: F401
    REST2_IMPLEMENTATION,
    scaling_for_tau,
    clone_system,
    _scale_nonbonded,
    _scale_torsions,
    UNSCALED_TORSION_DETECTOR_VERSION,
    system_bond_graph,
    is_improper,
    torsion_kind,
    torsion_is_scaled,
    torsion_exclusion_report,
    cmap_map_roles,
    shared_cmap_originals,
    duplicate_shared_cmaps,
    cmap_targets,
    _scale_cmap,
    REST2_GB_SCALE_PARAMETER,
    _scale_customgb,
    SCALED_FORCE_CLASSES,
    DELIBERATELY_UNSCALED_FORCE_CLASSES,
    ENERGY_FREE_FORCE_CLASSES,
    UnclassifiedForceError,
    audit_force_classes,
    build_scaled_system,
)


#: Identities this build can read but must NOT continue. v1 scaled the whole generalised-Born
#: contribution by (1-tau)^2; v2 scales it by (1-tau). That is a different Hamiltonian, so a v1 run
#: cannot be extended or resumed under v2 -- the samples would come from two different ensembles.
#: v1 records stay exactly as written; nothing here rewrites history to pretend otherwise.
HISTORICAL_REST2_IMPLEMENTATIONS = {
    ("rest2-no-bond-angle-omega", 1): (
        "v1 scaled the complete generalised-Born energy by (1-tau)^2. v2 scales it by (1-tau), "
        "which is a different Hamiltonian: a v1 trajectory and a v2 trajectory do not sample the "
        "same implicit-solvent ensemble, so one cannot continue the other."),
    ("rest2-no-bond-angle-omega", 2): (
        "v2 left only the ordinary amide omega unscaled and scaled aromatic ring torsions, other "
        "double-bond torsions and every solute improper by (1-tau)^2. rest2-unscaled-torsions v3 "
        "leaves all of those unscaled, which is a different Hamiltonian at every tau > 0: a v2 "
        "ladder and a v3 ladder do not sample the same hot ensembles, so one cannot continue the "
        "other."),
}


def require_compatible_implementation(recorded, *, what="this run"):
    """Refuse to continue a run written under a superseded Hamiltonian identity.

    Two runs agree only if they agree about what REST2 meant. A version bump here is not
    bookkeeping -- it says the energy function changed -- so continuing across one would silently
    join samples from two different ensembles.
    """
    if not isinstance(recorded, dict) or not recorded:
        return
    name = recorded.get("name")
    version = recorded.get("version")
    if name == REST2_IMPLEMENTATION["name"] and version == REST2_IMPLEMENTATION["version"]:
        return
    reason = HISTORICAL_REST2_IMPLEMENTATIONS.get((name, version))
    current = f"{REST2_IMPLEMENTATION['name']}/v{REST2_IMPLEMENTATION['version']}"
    raise ValueError(
        f"{what} records the Hamiltonian identity {name}/v{version}, but this build implements "
        f"{current}.\n"
        f"  {reason or 'That identity is not one this build implements.'}\n"
        f"  The recorded run is left exactly as it is. Start a new dataset under {current} rather "
        f"than continuing one written under a different energy function.")


def linear_tau_ladder(minimum, maximum, count):
    """A linear ladder. ONE implementation, in `remd.generated.tau_ladder`; this is its name here.

    It used to compute the ladder itself, unrounded, while `tau_ladder` rounded to six places --
    two spellings of one quantity, agreeing to six decimals and no further. That is exactly enough
    to pass every eye and fail an exact comparison: a `cv_stateN.json` recorded 0.166667 from the
    ladder that RAN, a resume recomputed 0.16666666666666666 from the ladder that did not, the
    check allows 1e-12, and every CV-enabled four-rung ladder was unresumable. Found by
    interrupting a real ladder, because nothing that agrees to six places is visible in a test
    fixture.

    Making the two round identically would have left two implementations that happen to agree.
    This delegates, so there is one and they cannot drift apart again.

    `minimum` is kept in the signature -- it is public API -- and every caller passes 0.0, which
    is what `tau_ladder` assumes: state 0 is the unmodified physical Hamiltonian.
    """
    if count < 2:
        raise ValueError(f"a ladder needs at least 2 replicas; got {count}")
    if float(minimum) != 0.0:
        raise ValueError(
            f"a tau ladder starts at 0.0 -- state 0 is the unscaled Hamiltonian -- and this asks "
            f"for {minimum}. No caller in this package wants otherwise; if one does, the ladder "
            f"itself has to learn about it rather than this wrapper reimplementing it.")
    from ..remd.generated import tau_ladder

    return tau_ladder(int(count), float(maximum))
