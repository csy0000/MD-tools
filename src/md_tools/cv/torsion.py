"""The dihedral itself: geometry, the minimum-image convention, and nothing else.

WHY THIS IS ARITHMETIC AND NOT A FORCE

    OpenMM will happily compute a torsion for you -- `CustomTorsionForce` with a zero force
    constant reports one per step. It is also the wrong tool here, and the reason is not
    performance. Adding a Force changes the System: its serialisation, its force-group layout, the
    checkpoint that records them, and -- through the force groups an integrator asks for -- the
    dynamics themselves. A run with CV reporting enabled would no longer be the same experiment as
    the run without it, which makes every comparison between them invalid and makes "does
    reporting change the physics" a question nobody can answer from the outputs.

    So the whole of CV evaluation is this file: positions in, degrees out, no OpenMM object
    touched. The positions are ones the reporting point already had.

THE MINIMUM-IMAGE CONVENTION IS NOT OPTIONAL

    Under periodic boundaries the four atoms of a torsion need not be in the same image. A
    backbone that straddles the boundary gives bond vectors the length of the box, and a dihedral
    computed from them is not merely imprecise -- it is unrelated to the molecular geometry.

    The convention is applied to the three sequential BOND VECTORS, not to the four positions.
    Wrapping positions independently into the primary cell is the intuitive thing to do and it is
    wrong: it can place two bonded atoms in opposite corners, which is exactly the artefact the
    convention exists to remove. Each bond vector is instead shifted by whichever lattice
    translation makes it shortest.

    For a triclinic cell that search is not `round(v / L)` per axis -- the box vectors are not
    orthogonal, so shortening along one axis can lengthen along another. OpenMM's reduced form
    guarantees `a` is along x and the off-diagonal terms are small, which makes a fixed
    three-step reduction (c, then b, then a) correct for it, and a small neighbour search over the
    27 surrounding images is used to prove that reduction found the true minimum.
"""

from __future__ import annotations

import math

import numpy as np


class TorsionError(ValueError):
    """A torsion that cannot be evaluated as asked."""


def minimum_image(vector, box=None):
    """The shortest lattice-equivalent of `vector`, as a length-3 float array.

    `box` is the 3x3 periodic box vectors in nanometres, rows `a, b, c`, exactly as OpenMM's
    `getPeriodicBoxVectors` returns them; `None` means a nonperiodic system, where the vector is
    already the only one there is.
    """
    vector = np.asarray(vector, dtype=float)
    if box is None:
        return vector
    box = np.asarray(box, dtype=float)
    if box.shape != (3, 3):
        raise TorsionError(f"periodic box must be 3x3 vectors, got shape {box.shape}")
    a, b, c = box[0], box[1], box[2]
    if a[0] <= 0 or b[1] <= 0 or c[2] <= 0:
        raise TorsionError(
            "the periodic box has a non-positive diagonal, so it encloses no volume and no "
            "minimum image is defined")

    # OpenMM's reduced form: `a` lies along x and `b` has no z component, so removing whole
    # multiples of c, then b, then a -- in that order, each using the component the previous step
    # cannot disturb -- lands within one image of the minimum.
    shortest = vector - c * round(vector[2] / c[2])
    shortest = shortest - b * round(shortest[1] / b[1])
    shortest = shortest - a * round(shortest[0] / a[0])

    # The reduction above is exact for a reduced cell and very nearly so for any other. The
    # neighbour search makes it exact for both: 27 candidates is cheap next to the position
    # retrieval that produced `vector`, and it removes the class of bug where a strongly skewed
    # cell silently reports a torsion from a second-nearest image.
    best = shortest
    best_length = float(np.dot(shortest, shortest))
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == j == k == 0:
                    continue
                candidate = shortest + i * a + j * b + k * c
                length = float(np.dot(candidate, candidate))
                if length < best_length:
                    best, best_length = candidate, length
    return best


def torsion_degrees(positions, indices, box=None) -> float:
    """The geometric dihedral of four atoms, in degrees, wrapped to `[-180, 180)`.

    `positions` is an (N, 3) array in nanometres; `indices` is the four atom indices in bonding
    order `i-j-k-l`. The sign follows the IUPAC/MDTraj convention: looking along `j -> k`, a
    positive angle turns the `i` end clockwise onto the `l` end.

    Wrapping is to `[-180, 180)` -- half-open at both ends of the same period, so exactly 180
    degrees reports as -180 and a value can never be written twice under two names.
    """
    positions = np.asarray(positions, dtype=float)
    i, j, k, l = (int(n) for n in indices)

    # The three sequential bond vectors, each independently minimum-imaged. See the module note:
    # the convention applies here, to the bonds, and not to the four positions.
    #
    # `b1` points i <- j, against the i-j-k-l chain, which is the canonical formulation and not a
    # slip. Taking it the other way negates both `x` and `y` below, and `atan2(-y, -x)` differs
    # from `atan2(y, x)` by exactly 180 degrees -- a whole column of plausible numbers in the
    # right range, every one of them wrong, which is precisely the failure this module's tests
    # exist to catch.
    b1 = minimum_image(positions[i] - positions[j], box)
    b2 = minimum_image(positions[k] - positions[j], box)
    b3 = minimum_image(positions[l] - positions[k], box)

    # The standard praxeolitic formulation: project out the b2 component so the two normals are
    # measured in the plane perpendicular to the central bond. `atan2` of (y, x) built this way
    # gives the signed angle directly, with no quadrant reconstruction and no `arccos` domain
    # error when the two planes are very nearly parallel.
    b2_length = math.sqrt(float(np.dot(b2, b2)))
    if b2_length == 0.0:
        raise TorsionError(
            f"atoms {j} and {k} are at the same position, so the torsion about the bond between "
            f"them is undefined")
    b2_unit = b2 / b2_length
    v = b1 - float(np.dot(b1, b2_unit)) * b2_unit
    w = b3 - float(np.dot(b3, b2_unit)) * b2_unit
    if not np.any(v) or not np.any(w):
        raise TorsionError(
            f"atoms {i}, {j}, {k}, {l} are collinear, so the dihedral about {j}-{k} is undefined")

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b2_unit, v), w))
    degrees = math.degrees(math.atan2(y, x))

    # `atan2` already returns (-180, 180]; the one value it produces that this convention does not
    # use is exactly +180.
    if degrees >= 180.0:
        degrees -= 360.0
    return degrees
