# Ligand dissociation with a Boresch restraint

A dissociation profile scans the receptor–ligand separation and **holds the other five degrees of
freedom**. This page is the method: what the six terms are, why the held five are the pathway, how
the anchors are chosen and checked, and what the resulting profile is and is not.

## The six coordinates

Three receptor anchors R1, R2, R3 and three ligand anchors L1, L2, L3 define six internal
coordinates between the two molecules:

```text
r    |R1 - L1|                 the separation, and the dissociation coordinate
thA  angle(R2, R1, L1)         the ligand off the receptor's axis
thB  angle(R1, L1, L2)         the receptor off the ligand's axis
phA  dihedral(R3, R2, R1, L1)  rotation of the ligand about the receptor's axis
phB  dihedral(R2, R1, L1, L2)  the twist between the two axes
phC  dihedral(R1, L1, L2, L3)  rotation of the ligand about its own axis
```

Together they fix all six rigid-body degrees of freedom of the ligand relative to the receptor:
three translational (`r`, `thA`, `phA` — a spherical polar coordinate about R1) and three
rotational (`thB`, `phB`, `phC`).

**Why hold five and scan one.** Restrain the separation alone and the ligand orbits the site at
fixed `r`, drifting between exit channels. Two windows at the same separation then sample
different routes, no estimator can join them, and the profile is an average over routes nobody
chose — while looking exactly like a profile. The five held terms *are* the pathway: change any
one and the profile describes a different route.

`md_tools.umbrella.boresch` builds the definitions; the restraints themselves are ordinary
umbrella restraints on named collective variables, because that is all a Boresch restraint is
once `cv.yaml` can express a distance and an angle.

## The anchors are chosen, checked, and never trusted

Two failure modes matter and both are silent:

* a **near-collinear triple** makes its dihedral undefined — it swings through a wide range for a
  change in geometry too small to measure, so a harmonic restraint on it pushes hard in a
  direction that means nothing;
* an **angle restrained near 0 or 180°** is ill-conditioned in a way no wrapping fixes: the
  gradient of an angle with respect to its atoms vanishes as the triple straightens, so the
  restraint holds it with almost no force exactly where it is needed.

`check_anchors` refuses both by name, with the measured geometry in the message, before a campaign
is generated — every window shares the anchor choice, so a bad one wastes the campaign rather than
a run.

```python
from md_tools.umbrella import BoreschAnchors, check_anchors, boresch_cv_document

anchors = BoreschAnchors(receptor=(1503, 2260, 625), ligand=(4687, 4673, 4675))
measured = check_anchors(positions, anchors)     # refuses, or returns the reference pose
```

**Reference values are measured, never chosen.** The five held terms keep the complex on the
pathway it actually occupies; numbers typed by hand describe a pose the complex was never in, so
the window opens by dragging the ligand there and the work of that drag lands in the profile.

**Prefer rigid anchors.** Backbone alpha carbons in ordered secondary structure, mutually
separated by more than about 0.6 nm. A side-chain tip is mobile, so a restraint referred to it
fights the side chain rather than holding the ligand.

**Align the pull with the exit.** Scanning `|R1 - L1|` moves the ligand roughly along the R1→L1
vector, so R1 should be chosen such that that vector points out of the site rather than into the
protein.

## Worked selection: TYK2 + ejm_31

From the 53,030-particle build of the TYK2/ejm_31 complex (ff14SB/TIP3P, ligand `L31`), sampling
anchor sets from the 51 alpha carbons lining the site and the 21 ligand heavy atoms:

```text
965 sets pass every conditioning check with >= 30 deg of margin
chosen: receptor LEU983:CA, LEU1030:CA, VAL929:CA   ligand O2, C5, C7
        margin 56.2 deg from any collinear limit
        alignment with the clearest exit direction 0.996

measured reference pose
  r    0.628 nm      thA  69.2 deg     thB  123.4 deg
  phA -31.7 deg      phB   8.9 deg     phC   69.4 deg
```

**The box is ample, which is worth stating because the obvious estimate says otherwise.** The
periodic box is cubic at 8.238 nm and the protein spans up to 6.26 nm, which looks tight. It is
not: with the ligand displaced 2.5 nm along the exit direction, the nearest periodic image of the
protein is still 3.8–4.1 nm away.

**There is no statically clear exit channel, and that shapes what the profile means.** Marching
the ligand centre of mass out to 2.5 nm along each of 600 directions on a sphere, not one keeps it
0.40 nm clear of every protein atom; the best direction holds 0.34 nm and falls to 0.11–0.16 nm
along most of the path. That is what a buried ATP site looks like rather than a defect — a
straight-line exit clips side chains, and real unbinding paths are curved with side chains
relaxing out of the way under force.

So a straight pull along any direction forces side chains aside, and **the profile is the work
along that route**, with its hysteresis unmeasured. It is not an unbinding free energy, and this
page does not call it one.

## Window spacing and force constants

The two constants are not comparable, because their units are not: a distance constant is
kJ/mol/nm² and an angular one kJ/mol/rad². What matters is the excursion each allows, and
`standard_state_note` reports it:

```text
2000 kJ/mol/nm^2  ->  r confined to about 0.035 nm rms at 300 K
100  kJ/mol/rad^2 ->  each held angle confined to about 9 deg rms
```

At 0.1 nm window spacing that is about 2.8σ between neighbours, which overlaps. Check the overlap
on the output rather than trusting the arithmetic: adjacent windows must share samples, and
`docs/tutorial/ALA/umbrella_analysis.py` refuses a PMF across a gap for that reason.

## What this does not give you

A PMF along `r` with five terms held is not a binding free energy. Converting one requires the
standard-state correction for the volume and orientation the restraints remove, and its analytical
form depends on exactly which terms were held and how strongly. That belongs with the estimator,
in the project asking the question — the same boundary every other estimator here sits behind.
`standard_state_note` states what is missing rather than returning a number, so that a figure
computed here cannot be mistaken for one this package has validated.
