# One protein–ligand system, four methods

**Planned, 2026-09-20, at the user's direction.** Nothing here is implemented or measured yet.
This page names the model system every method demonstrates on, what each branch must deliver on
it, and how the work divides. It is the shared target of 0.6.1, 0.7.0, 0.7.1 and 0.7.2, and the
source of the tutorials.

## The model system: TYK2, three congeneric ligands

From [OpenFE benchmarks](https://github.com/OpenFreeEnergy/openfe-benchmarks) (MIT, attribution
kept), `openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/`: `protein.pdb`, `ligands.sdf`,
`experimental_binding_data.json`, `PREPARATION_DETAILS.md`.

Why this one:

- **Small.** TYK2's kinase domain is the smallest of the JACS set, so 4-GPU ladders and ~20-window
  alchemical legs fit in hours, not days. A tutorial nobody can run is not a tutorial.
- **Congeneric.** Sixteen ligands of one series, which is what RBFE, REST2-FEP and EDS all need.
- **Measured.** Every ligand has an experimental ΔG, so a result can be *compared* to something —
  reported, and gated only where the comparison is fair.
- **Prepared, with its preparation stated.** Caps are ACE/NME, and the page records the manual
  bond fixes, so the input's provenance is legible rather than assumed.

The three ligands, chosen to span ~1.5 kcal/mol within one chemical series:

| ligand | experimental ΔG (kcal/mol) | role |
|---|---|---|
| `ejm_31` | −9.54 | the reference ligand: ABFE, REST2, the EDS/RE-EDS reference state |
| `ejm_42` | −9.78 | RBFE partner, ΔΔG ≈ −0.24 against `ejm_31` |
| `ejm_43` | −8.26 | second RBFE partner, ΔΔG ≈ +1.28 — a larger, easier-to-resolve signal |

Two partners on purpose: `ejm_42` is a near-null edge that tests precision, `ejm_43` a clear
signal that tests accuracy. Three ligands is also the minimum for EDS/RE-EDS to be a multi-state
method rather than a two-state one in disguise.

**Force field: ff14SB + TIP3P**, decided 2026-09-20. Not taste: `matched_legs` refuses to pair an
OPC-solvated leg with a vacuum leg, because OPC applies `0.833333` to a ligand's 1-4 pairs while a
vacuum build applies 5/6, and ABFE and RBFE both need exactly that pairing. ff19SB + OPC is a later
variant, unblocked by the backlog item that lets a vacuum leg apply the solvent leg's stated scale.

**Preparation is done ONCE, by one session, and every method reuses it.** One
`build-top` complex per ligand, one parameter package per ligand (created once, reused in every
environment, as 0.6.0 guarantees), and the prepared inputs kept as a versioned fixture with their
build records. Four methods preparing their own protein four times is four different proteins.

## What each branch delivers on it

### 0.6.1 — selective REST2, on 4 GPUs

1. **Ligand-only ladder.** The ligand hot, 8 rungs, tau 0 → 0.5.
2. **Ligand plus pocket sidechains.** The ligand and the sidechains of the residues lining the
   pocket, 8 rungs, tau 0 → 0.25.

**A ladder gets HOMOGENEOUS cards** (the user, 2026-09-21). On this machine card 0 is slightly
faster than the rest, so it is excluded from every REMD run: at each exchange every rank waits for
the slowest rung, so a faster card cannot make the ladder faster — it idles — and it makes
per-rung throughput figures incomparable. The TYK2 ladders run on cards 1–4. Independent work
(alchemical windows, AIS paths) may use card 0 freely, but must then say which card produced which
window rather than comparing their throughput as equals.

Both on 4 GPUs (8 rungs over 4 cards, 2 ranks per card, through the existing MPI/MPS rules), with
the scaling measured: ns/day against rungs and cards, exchange acceptance per neighbour pair, and
the round trip of a walker through the ladder.

What 0.6.1 still needs for this, beyond what it has:

- **Pocket residue selection.** The user asks for the sidechains *around the ligand*. Today the
  user writes the residue numbers. A helper that lists residues within a cutoff of a named ligand
  instance — printed, reviewable, and pasted into the configuration, never silently resolved at
  run time — is new work.
- **Sidechain rotatability.** A sidechain's rotatable and non-rotatable bonds differ per residue,
  and the set of amino acids is small and fixed, so this is a lookup table, not a perception
  problem: per residue, which chi torsions are scalable, and which bonds are not (amide in ASN/GLN,
  guanidinium in ARG, aromatic rings in PHE/TYR/TRP/HIS, carboxylate in ASP/GLU). S1's classifier
  already refuses what it cannot classify; the table makes the answer explicit and testable.

### 0.7.0 — Amber18 softcore, both binding free energies

1. **ABFE** of `ejm_31` in TYK2: the full cycle, Boresch restraints, restraint free energy and the
   standard-state correction.
2. **RBFE** `ejm_31` → `ejm_42` and `ejm_31` → `ejm_43`: solvent and complex legs.

Both across the topology-building modes the branch already supports, so the tutorial can show what
single, dual and hybrid topology actually change. Hydration on ethane/chloroethane stays the
development fixture; TYK2 is the demonstration.

#### Replica exchange over lambda (the user, 2026-09-21)

Today's windows are INDEPENDENT: each samples at a fixed lambda and the estimators combine them
afterwards. The user asks for exchange between lambda windows as the tutorial's advanced example,
so 0.7.0 gains it as a milestone of its own:

- **A3b — lambda exchange (Hamiltonian replica exchange).** Neighbouring lambda windows attempt
  swaps, through `md_tools.remd` — the one MPI and exchange authority — not a second
  implementation. Every window stays at the same physical temperature; what is exchanged is the
  Hamiltonian's lambda, exactly as REST2 exchanges tau.

The design question to settle first, because the two ladders differ where it matters: a REST2 rung
is a SAVED SYSTEM on disk and `md-run` reads it from the group file, while a lambda window is ONE
System with different Context parameter values. The saved-state rule exists so no Hamiltonian is
re-derived at run time; a lambda window does not re-derive anything, it sets a recorded parameter.
Whether that means the ladder driver learns a second rung kind, or alchemy supplies rungs through
the caller-supplied path with its declaration and per-rung provenance, is S3 and S4's to answer
together before anything is written.

What must hold either way: a series follows a STATE, not a walker (`cv_state<i>`-style, with the
walker recorded); exchanges never rescale velocities; acceptance uses independently evaluated
reduced potentials, which S4's sample record already carries per origin state; and the tutorial
reports acceptance per neighbour pair and round trips, as the REST2 pages do.

### 0.7.1 — REST2-TI / REST2-FEP

The composition of 0.6.1 and 0.7.0, following Wang, Berne and Friesner, *On achieving high
accuracy and reliability in the calculation of relative protein–ligand binding affinities*, PNAS
109(6):1937–1942 (2012), <https://doi.org/10.1073/pnas.1114017109> — FEP/REST: the alchemical
region *and* its surroundings are solute-tempered, so a binding-site rearrangement that FEP alone
cannot sample is reached.

On TYK2 that is the same two RBFE edges, with the 0.6.1 hot region (ligand plus pocket sidechains)
carried along. The acceptance test is not "it runs": it is that the two reductions hold — tau = 0
reproduces 0.7.0's result, and the physical alchemical endpoints reproduce 0.6.1's ensemble — and
that the RBFE agrees with 0.7.0's within combined uncertainty, with better overlap or fewer windows
for the same precision.

### 0.7.2 — EDS and RE-EDS

All three ligands at once, as one multi-state calculation (rinikerlab/reeds as the reference for
the method, not the implementation). The deliverable on TYK2 is ΔΔG for both edges from ONE
RE-EDS calculation, agreeing with 0.7.0's pairwise RBFE within combined uncertainty.

Grand-canonical water sampling stays 0.7.2's other half; the two are separate milestones and are
not combined until each is validated alone.

## Tutorials

One directory per method, under `docs/tutorial/protein-ligand-complex/`:

| directory | shows |
|---|---|
| `REST2/` | simultaneous ligand + pocket-sidechain scaling on 4 GPUs |
| `TI-FEP/` | ABFE and RBFE with Amber18 softcore, and — as the advanced example — the same edge with replica exchange over lambda |
| `REST2-TI-FEP/` | the combination, and when it is worth its cost |
| `EDS-RE-EDS/` | three ligands in one multi-state calculation |

Every page follows the rule the 0.5.4 and 0.6.0 tutorials set: **executed exactly as written, with
every number copied from the files that run produced**. A page that cannot be run as written is a
defect, not documentation. Each page states its hardware and wall time, so a reader knows what they
are committing to before they start.

`docs/tutorial/` is the existing location and keeps its name; the user's `tutorials/protein-ligand
complex/` is this, with the space replaced by a hyphen so the path is typable.

## Sequence

The methods depend on each other, so the order is not a preference:

1. **Preparation** (shared): TYK2 + three ligands, once.
2. **0.6.1** selective REST2 on it → its tutorial. Needs the pocket-selection helper and the
   sidechain table.
3. **0.7.0** ABFE and RBFE on it → its tutorial. Needs 0.6.1 integrated for ancestry, not for
   physics.
4. **0.7.1** REST2-TI/FEP → needs both above, and both reductions demonstrated.
5. **0.7.2** EDS/RE-EDS → needs 0.7.0's Hamiltonian and 0.7.1's multi-state runtime.

## What this costs, honestly

A TYK2 ABFE cycle, two RBFE edges in two legs each, and two 8-rung ladders are **days of GPU time**,
not hours, and they are production campaigns rather than tests. Every one of them is a request to
the user with cards and wall time named, and the acceptance rows are written before sampling. The
deterministic checks each method already has stay the gate; a campaign that has not passed those
does not start.
