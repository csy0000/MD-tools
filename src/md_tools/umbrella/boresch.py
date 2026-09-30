"""The Boresch restraint: six internal coordinates that hold a ligand's pose to a receptor.

WHAT IT IS, AND WHY A DISSOCIATION PROFILE NEEDS ONE

    Three receptor anchors R1, R2, R3 and three ligand anchors L1, L2, L3 define six internal
    coordinates between the two molecules -- one distance, two angles, three dihedrals:

        r    |R1 - L1|                the separation, and the dissociation coordinate
        thA  angle(R2, R1, L1)        how the ligand sits off the receptor's axis
        thB  angle(R1, L1, L2)        how the receptor sits off the ligand's axis
        phA  dihedral(R3, R2, R1, L1) rotation of the ligand about the receptor's axis
        phB  dihedral(R2, R1, L1, L2) the twist between the two axes
        phC  dihedral(R1, L1, L2, L3) rotation of the ligand about its own axis

    Together they fix all six rigid-body degrees of freedom of the ligand relative to the
    receptor: three translational (r, thA, phA -- a spherical polar coordinate about R1) and
    three rotational (thB, phB, phC). That is the whole point. Restrain the distance alone and
    the ligand orbits the site at fixed separation, wandering between exit channels, so two
    windows at the same r sample different pathways and no estimator can join them: the profile
    is an average over routes nobody chose, and it looks like a profile.

    So a dissociation profile SCANS r and HOLDS the other five. The five held terms are the
    pathway; changing any of them changes which route the profile describes.

WHY THE ANCHORS ARE CHECKED AND NOT TRUSTED

    The six coordinates are only well defined for anchors in general position. Two failures
    matter, both silent:

        a near-collinear triple makes its dihedral undefined -- rotate about an axis the three
        atoms nearly lie on and the dihedral swings wildly for an unmeasurable change in
        geometry, so a harmonic restraint on it pushes hard in a direction that means nothing;

        an angle restrained near 0 or 180 degrees is ill-conditioned in a way no wrapping fixes:
        the gradient of the angle with respect to the positions vanishes as the triple becomes
        collinear, so the restraint holds with almost no force exactly where it is needed most.

    Both produce windows that run to completion and sample something. `check_anchors` refuses
    them by name, with the measured geometry in the message, because a refusal a person can act
    on is worth more than a warning in a log nobody reads.

WHAT THIS MODULE DOES NOT DO

    It does not compute a binding free energy. A PMF along r with the orientational terms held is
    not a binding free energy on its own: it needs the standard-state correction for the volume
    and orientation the restraints confine, and the analytical form of that depends on which
    terms were held and how strongly. That belongs with the estimator, in the project asking the
    question, and it is documented in the tutorial rather than guessed at here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..cv.torsion import angle_degrees, distance_nm, torsion_degrees

__all__ = ["BORESCH_TERMS", "BoreschAnchors", "BoreschError", "boresch_cv_document",
           "boresch_window_document", "check_anchors", "measure_anchors"]


class BoreschError(ValueError):
    """An anchor choice or a reference geometry that cannot define a Boresch restraint."""


#: The six terms, in a fixed order, each with the kind it is and which anchors it spans.
#: `slots` indexes the concatenated anchor list (R1, R2, R3, L1, L2, L3).
BORESCH_TERMS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("r", "distance", (0, 3)),                 # R1 - L1
    ("thA", "angle", (1, 0, 3)),               # R2 - R1 - L1
    ("thB", "angle", (0, 3, 4)),               # R1 - L1 - L2
    ("phA", "torsion", (2, 1, 0, 3)),          # R3 - R2 - R1 - L1
    ("phB", "torsion", (1, 0, 3, 4)),          # R2 - R1 - L1 - L2
    ("phC", "torsion", (0, 3, 4, 5)),          # R1 - L1 - L2 - L3
)

#: How close to collinear a restrained angle may sit, in degrees. Outside [MARGIN, 180 - MARGIN]
#: the angle's gradient is small enough that the restraint cannot hold it, and any dihedral
#: measured about that axis is poorly determined. 20 degrees is the customary working margin.
ANGLE_MARGIN_DEG = 20.0

#: How close to collinear a dihedral's three defining bonds may sit, in degrees. A triple flatter
#: than this makes the dihedral about it undefined rather than merely noisy.
COLLINEAR_MARGIN_DEG = 15.0


@dataclass(frozen=True)
class BoreschAnchors:
    """Three receptor atoms and three ligand atoms, in the order the terms above assume."""

    receptor: tuple[int, int, int]
    ligand: tuple[int, int, int]

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(self.receptor) + tuple(self.ligand)

    def atoms_for(self, term: str) -> tuple[int, ...]:
        for name, _kind, slots in BORESCH_TERMS:
            if name == term:
                return tuple(self.slots[s] for s in slots)
        raise BoreschError(f"{term!r} is not one of the six Boresch terms")

    def __post_init__(self):
        if len(self.receptor) != 3 or len(self.ligand) != 3:
            raise BoreschError(
                f"a Boresch restraint needs exactly three receptor and three ligand anchors; got "
                f"{len(self.receptor)} and {len(self.ligand)}")
        every = self.slots
        if len(set(every)) != 6:
            raise BoreschError(
                f"the six anchors must be distinct atoms; got {list(every)}. A repeated anchor "
                f"collapses at least one of the six coordinates onto a constant, which then "
                f"restrains nothing while still appearing in the record")


def measure_anchors(positions, anchors: BoreschAnchors, box=None) -> dict[str, float]:
    """The six coordinates AS THEY ARE in `positions`, which is where reference values come from.

    A reference pose is measured, never chosen: the five held terms define the pathway, and
    numbers typed by hand describe a pose the complex was never in -- so the window starts by
    dragging the ligand somewhere it does not belong, and the work of that drag lands in the
    profile.
    """
    evaluate = {"distance": distance_nm, "angle": angle_degrees, "torsion": torsion_degrees}
    measured = {}
    for name, kind, _slots in BORESCH_TERMS:
        measured[name] = float(evaluate[kind](positions, anchors.atoms_for(name), box))
    return measured


def check_anchors(positions, anchors: BoreschAnchors, box=None, *,
                  angle_margin_deg: float = ANGLE_MARGIN_DEG,
                  collinear_margin_deg: float = COLLINEAR_MARGIN_DEG) -> dict[str, float]:
    """Refuse an anchor choice whose coordinates are not well defined. Returns the measurement.

    Called before a campaign is generated, not during it: every window in a profile shares the
    anchor choice, so a bad one wastes the whole campaign rather than one run.
    """
    problems: list[str] = []

    # THE CONDITIONING IS CHECKED BEFORE THE DIHEDRALS ARE MEASURED, because a dihedral about a
    # collinear triple cannot be measured at all -- `torsion_degrees` refuses it, correctly, with
    # a message about atom indices and an undefined dihedral. That is the right refusal from a
    # geometry function and the wrong one to hand somebody choosing anchors: it names no anchor,
    # no margin and no remedy. Measured first and this whole function was unreachable for exactly
    # the input it exists to diagnose.
    separation = distance_nm(positions, anchors.atoms_for("r"), box)
    if separation <= 0.0:
        problems.append("R1 and L1 coincide, so the separation is zero and its direction is "
                        "undefined")

    # The two restrained angles must be holdable.
    for term in ("thA", "thB"):
        try:
            value = angle_degrees(positions, anchors.atoms_for(term), box)
        except Exception as broken:                        # noqa: BLE001 - reported as a refusal
            problems.append(f"{term} cannot be measured: {broken}")
            continue
        if not angle_margin_deg <= value <= 180.0 - angle_margin_deg:
            atoms = anchors.atoms_for(term)
            problems.append(
                f"{term} = {value:.1f} deg on atoms {list(atoms)} is within "
                f"{angle_margin_deg:g} deg of collinear. The gradient of an angle with respect "
                f"to its atoms vanishes as the triple straightens, so a restraint there holds it "
                f"with almost no force, and any dihedral measured about that axis is poorly "
                f"determined. Choose an anchor that sits away from the axis")

    # Every dihedral needs its own three bonds out of line, checked on the bond angles rather
    # than by attempting the dihedral.
    for term in ("phA", "phB", "phC"):
        i, j, k, l = anchors.atoms_for(term)
        for triple in ((i, j, k), (j, k, l)):
            try:
                straightness = angle_degrees(positions, triple, box)
            except Exception as broken:                    # noqa: BLE001 - reported as a refusal
                problems.append(f"{term}: atoms {list(triple)} cannot form an angle: {broken}")
                continue
            if not collinear_margin_deg <= straightness <= 180.0 - collinear_margin_deg:
                problems.append(
                    f"{term}: atoms {list(triple)} are at {straightness:.1f} deg, within "
                    f"{collinear_margin_deg:g} deg of collinear, so the dihedral about them is "
                    f"undefined -- it swings through a wide range for a change in geometry too "
                    f"small to measure, and a harmonic restraint on it pushes hard in a "
                    f"direction that means nothing")

    if problems:
        raise BoreschError(
            "these anchors do not define a Boresch restraint:\n  - " + "\n  - ".join(problems)
            + "\n  Every window in a profile shares this choice, so it is refused here rather "
              "than after a campaign has run.")
    # Only now, with every coordinate known to be well defined, is the reference pose measured.
    return measure_anchors(positions, anchors, box)


def boresch_cv_document(anchors: BoreschAnchors, *, prefix: str = "boresch") -> dict:
    """A `cv.yaml` document defining the six coordinates, for `build-md` to copy and report.

    The names are prefixed so a definition can carry a Boresch set alongside other variables
    without a collision, and so a reader of a CSV header can tell at a glance which columns
    belong to the restraint.
    """
    return {
        "schema_version": 1,
        "collective_variables": [
            {"name": f"{prefix}_{name}", "type": kind,
             "atom_indices": [int(a) for a in anchors.atoms_for(name)]}
            for name, kind, _slots in BORESCH_TERMS
        ],
    }


def boresch_window_document(measured: dict[str, float], *, centre_nm: float,
                            distance_force_constant: float = 10000.0,
                            angular_force_constant: float = 100.0,
                            prefix: str = "boresch",
                            hold: Sequence[str] = ("thA", "thB", "phA", "phB", "phC")) -> dict:
    """An `umbrella.yaml` for ONE window: the distance at `centre_nm`, the rest held as measured.

    `measured` is what `measure_anchors` returned for the bound pose, so the five held terms keep
    the complex on the pathway it actually occupies rather than one chosen on paper.

    The two force constants are separate because their units are. A distance constant is
    kJ/mol/nm^2 and an angular one kJ/mol/rad^2, and the numbers are not comparable: 100 is a
    loose torsional restraint and an almost free distance, while 10000 kJ/mol/nm^2 corresponds to
    a root-mean-square excursion of about 0.016 nm at 300 K. Neither is a default anybody should
    accept without checking what it does to the sampling.
    """
    if centre_nm < 0.0:
        raise BoreschError(f"a window centre of {centre_nm} nm is not a separation")
    missing = [term for term in hold if term not in measured]
    if missing:
        raise BoreschError(f"no measured reference value for {missing}; measure the bound pose")

    restraints = [{"cv": f"{prefix}_r", "form": "harmonic", "centre_nm": float(centre_nm),
                   "force_constant": float(distance_force_constant)}]
    for term in hold:
        kind = next(k for name, k, _ in BORESCH_TERMS if name == term)
        if kind == "distance":
            raise BoreschError(
                f"{term} is the dissociation coordinate, not a held term; it is scanned by "
                f"`centre_nm` and cannot also be held at its bound value")
        restraints.append({
            "cv": f"{prefix}_{term}", "form": "harmonic",
            "centre_deg": round(float(measured[term]), 6),
            "force_constant": float(angular_force_constant),
        })
    return {"schema_version": 1, "restraints": restraints}


def standard_state_note(distance_force_constant: float, angular_force_constant: float,
                        temperature_k: float = 300.0) -> str:
    """What still has to be done to turn a profile into a binding free energy.

    Returned as text rather than a number on purpose. The correction depends on which terms were
    held, how strongly, and at what temperature, and quoting one computed here would invite it to
    be used as though this module had validated it. It has not.
    """
    kt = 0.0083144621 * float(temperature_k)
    sigma_nm = math.sqrt(kt / float(distance_force_constant))
    sigma_deg = math.degrees(math.sqrt(kt / float(angular_force_constant)))
    return (
        f"At {temperature_k:g} K, a distance constant of {distance_force_constant:g} "
        f"kJ/mol/nm^2 confines r to about {sigma_nm:.4f} nm rms, and an angular constant of "
        f"{angular_force_constant:g} kJ/mol/rad^2 confines each held angle to about "
        f"{sigma_deg:.1f} deg rms.\n"
        f"A PMF along r with those five terms held is NOT a binding free energy: it is the work "
        f"of separation along one pathway, with the ligand's orientation confined. Converting it "
        f"requires the standard-state correction for the volume and orientation the restraints "
        f"remove, which depends on exactly which terms were held and how strongly. That belongs "
        f"with the estimator, and this module deliberately does not compute it."
    )
