# 0.6.1 — selective explicit-solvent REST2

**Under construction.** Nothing here is in a release. 0.6.0, which this builds on, is itself still
under the user's testing.

Baseline: `dev-0.6.0` at `e524e0e`.

## Purpose

Today a REST2 ladder scales the whole solute: `solute_atom_indices` takes every residue that is
not solvent or a counter-ion (`src/md_tools/md/stage.py:64-74`), and everything in that set is hot.
For a protein of any size that is wasteful and often counter-productive — the interesting degrees
of freedom are a loop, a set of sidechains, or one ligand, and heating the rest costs sampling
efficiency and can unfold what you wanted to keep folded.

0.6.1 lets the user choose the hot region: named backbone residues, named sidechain residues, and
named ligand *instances* — one copy of a compound, not every copy of it.

## User-visible outcome

New optional fields in the REST2 workflow configuration, forwarded to the build-time scaler:

~~~yaml
backbone_scaling_list: ":45,46,59"
sidechain_scaling_list: ":45-50"
ligand_scaling_dict:
  L01:
    mask: ":201"
    torsion_exclusions: "L01-exclusions.yaml"
  L02:
    mask: ":305"
    torsion_exclusions: auto
~~~

A compact form `L01: <path-to-exclusion-file>` is also accepted when `L01` is an existing,
unambiguous recorded instance alias. When it is not unambiguous the explicit entry is required;
the resolver never guesses from a ligand residue name, because two copies of one compound share
that name and selecting one must not select both.

Mask strings are quoted YAML. The supported grammar is a documented **subset** of AMBER's: a
numeric residue list (`":45,46,59"`), an inclusive range (`":45-50"`), and mixtures of the two.
Numbers are **one-based topology residue indices** — not Python indices, and not PDB author
numbering. Chain qualifiers may be added later through a compatible parser; until then the page
says so rather than implying the whole AMBER grammar works. Per-chain index resets are never
invented.

The resolved residue map is printed and kept with the preparation, so the user can see which
residue each number actually matched.

### Legacy behaviour is the default, and the modes do not blend

- **No new selector field present** → the legacy full-solute selection, byte-identical to 0.6.0.
- **Any new selector field present** → explicit mode: only the supplied categories and instances
  are hot. An omitted category means *none*, not *all*.
- An explicit mode whose hot region resolves to nothing is refused.

That distinction is a test, not a convention.

### Refusals

Unsupported operators, unknown or out-of-range residue numbers, an ambiguous instance alias, an
empty match, or an entirely empty explicit hot region are all refused before any output exists,
each naming what it could not resolve and what would fix it.

## Where resolution happens

The user may supply these settings through the REST2 workflow configuration, but the
**authoritative resolution and force construction stay in the build-time scaler**
(`md-openmm build-top --rest2-scaler`). Every scaled Hamiltonian a run integrates is still written
as a file before the run starts; nothing is scaled at run time, and no stale rung XML is reused.

The workflow configuration therefore *forwards* the selection to state construction. Where the
same selection is declared twice, the two must agree or the build is refused — a silent precedence
rule is how two documents come to describe different experiments.

One parser and one resolver serve preparation and validation alike. The runtime consumes and
verifies a resolved record; it never re-selects atoms from inputs that may have changed.

## Hamiltonian rules

For selected atoms `S` and the remainder `E`:

| interaction | factor |
|---|---|
| S–S nonbonded, and S–S exceptions (1-4) | `(1-tau)^2` |
| S–E nonbonded, and S–E exceptions | `1-tau` |
| E–E | `1` |

Electrostatics must stay PME-consistent: the current implementation reaches the S–E factor
through Lorentz–Berthelot from per-particle `charge*(1-tau)` and `epsilon*(1-tau)^2`
(`src/md_tools/rest2/hamiltonian.py:98-117`), which works because the hot set was the whole solute.
With a partial hot set the same construction must be re-derived and checked term by term against
an independent evaluation, not assumed to carry over.

Further rules:

- **Nonbonded selection and torsion eligibility are resolved separately.** A sidechain selection
  must reach chi1 even though the quartet defining chi1 includes backbone N and C-alpha; a
  backbone selection must handle phi and psi, which cross a residue boundary.
- Each central bond has a documented **owning residue**, and every eligible proper Fourier term
  belonging to that bond is scaled together.
- The existing exclusions survive unchanged: ordinary amide omega, aromatic ring bonds, other
  double bonds, and every improper stay unscaled, and the proline-like exception
  (a proline-like ring has no amide hydrogen to protect, so its omega is scaled) is preserved and
  documented.
- Backbone and sidechain membership is **defined**, in writing, for hydrogens, termini, caps,
  glycine, proline, disulfides and every supported modified residue. Chemistry that is not
  covered is refused, never approximated.
- **CMAP needs an explicit versioned rule.** The proposed rule: scale a CMAP term only when both
  of its underlying backbone torsions are selected; leave a mixed pair unchanged and report it.
  Shared-map duplication (`duplicate_shared_cmaps`) must be validated under partial selection, and
  the legacy all-solute behaviour preserved.
- Bonds and angles remain unscaled. A torsion exclusion never removes its atoms from nonbonded
  scaling — they are two different questions about the same atom.
- Ligand torsion exclusions are mapped from **package-local atom identities**, not from
  transferable global force indices. The resolved contents are saved into the record; an external
  file path alone is not provenance, because the file can change afterwards.
- **Selective implicit-solvent scaling is out of scope.** Whole-system GB runs keep working; a
  partial selection on an implicit system is refused before any output.

## Identity and resume

These selection semantics change what a Hamiltonian *is*, so the identity that guards resume must
change with them. `SELECTION_FORMAT` (`md-tools-solute-selection/1.0`) and
`FINGERPRINT_FORMAT` (`md-tools-hamiltonian-identity/v2`) are both versioned, and the REST2
convention record `REST2_IMPLEMENTATION` (`rest2-unscaled-torsions`, version 3) gains the
selection semantics it now depends on. An incompatible resume, and reuse of a saved state built
under different selection semantics, are refused rather than mixed. See
[shared contracts](../shared-contracts.md) for the exact record.

## Non-goals

- Selective implicit-solvent (GB) scaling.
- Temperature REMD. Every replica stays at the same physical temperature; the runtime stays NVT.
- Run-time scaling of any kind.
- Chain qualifiers or the full AMBER mask grammar in this release.
- Any change to cMD or AIS behaviour.

## Milestones

| | milestone | done when |
|---|---|---|
| S1-A | mask parser and resolver, with the resolved selection record | grammar accepted and refused as specified; round-trip through the record; residue map printed |
| S1-B | backbone / sidechain membership tables and torsion ownership | every supported residue class covered, with the boundary cases as tests |
| S1-C | partial-selection Hamiltonian construction | analytic S–S / S–E / E–E factors verified per pair and per exception |
| S1-D | ligand instances and their exclusion files | two copies of one compound, and two distinct compounds, each selected independently |
| S1-E | CMAP rule, versioned | shared and mixed maps; legacy behaviour unchanged |
| S1-F | identity, resume and refusal | stale saved state refused; changed selection refused; `--check` creates nothing |
| S1-G | CUDA ladders | the campaigns below, actually run |

## Supported chemistry and topology

| | supported in 0.6.1 | notes |
|---|---|---|
| protein backbone selection | yes | phi/psi across residue boundaries |
| protein sidechain selection | yes | chi1 reaching backbone N / C-alpha |
| glycine, proline, termini, caps | yes | membership defined explicitly |
| disulfides | yes | membership defined explicitly |
| modified residues | only those already supported | others refused by name |
| ligand instance selection | yes | per instance, via the recorded instance identity |
| several copies of one compound | yes | selecting one must not select the others |
| nucleic acids | no | refused; no backbone/sidechain definition exists for them |
| implicit solvent (GBn2) | whole-system only | partial selection refused |

## Required evidence

Deterministic, before any campaign:

- analytic pair and exception factors at several tau values;
- `tau = 0` identity — the scaled System is the unscaled System;
- legacy-selection equivalence — no new field present reproduces 0.6.0 exactly;
- phi/psi and chi1 boundary cases;
- several chains; two copies of one ligand; two distinct ligands;
- reordered atoms;
- mutation of an exclusion file after the record was written;
- CMAP shared and mixed maps.

Then actual **CUDA explicit-solvent ladders** for a small peptide and a protein–ligand fixture,
covering selective backbone, selective sidechains, ligand-only, and combined selections. Check
direct exchange energies at stored coordinates, resume identity and refusal, standalone export,
and data registration. Current NVT REST2 and the MPI/MPS rules are preserved.

Effective-temperature intuition is not evidence. The thing to test is the energy the
implementation computes.

## References

- **R1** Wang, Friesner and Berne, *Replica Exchange with Solute Scaling* (2011),
  <https://doi.org/10.1021/jp204407d> — scaling and exchange theory. MD-tools' explicit torsion
  exclusions are additional policy on top of it, not something R1 prescribes.
- **R2** AMBER atom-mask syntax, <https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/> —
  for syntax and numbering, and for documenting exactly which subset is supported.
- **R3** the current source, this repository at `dev-0.6.0`.
- Current implementation entry points: `src/md_tools/rest2/{selection,hamiltonian,identity,states}.py`,
  `src/md_tools/build/scaler.py`, `src/md_tools/openmm/system.py` (torsion classification),
  `src/md_tools/ligands/`, `src/md_tools/md/stage.py:64-74` (the legacy selector).
