# ALA reference campaign, 2026-09

Alanine dipeptide (ACE-ALA-NME), sampled in implicit and explicit solvent as reference data for
methods that are compared against it. Requested 2026-09-10, run 2026-09-10 to 2026-09-11.

## What was asked for

| arm | protocol | length | notes |
|---|---|---|---|
| hot | cMD at τ = 0.5 | 300 ns | NVT |
| cold | cMD at τ = 0, three seeds (1, 2, 3) | 1000 ns each | NVT requested; see below |
| REST2 | linear τ 0 → 0.5, exchange every 10 ps | 300 ns per state | 4 states implicit, 6 explicit |

Both solvents: ff14SB + GBn2 (implicit), ff14SB + TIP3P + 0.15 M NaCl (explicit). 2 fs, 300 K,
HBonds constraints, **no hydrogen mass repartitioning** -- none was requested, so the defaults
applied. Solute coordinates every 5 ps; explicit whole-system coordinates every 50 ns.

## What exists now

**The implicit arm is complete and registered** as shared reference data:

```text
$MD_DATA/common/2026/reference/2026-09/ALA-dipeptide-implicit/cMD-hot/run1
$MD_DATA/common/2026/reference/2026-09/ALA-dipeptide-implicit/cMD-cold/run1   (and run2, run3)
$MD_DATA/common/2026/reference/2026-09/ALA-dipeptide-implicit/REST2/run1
```

These five datasets were registered at `common/reference/...` and **moved** to the paths above when
the canonical common path gained its year segment; their manifests were rewritten to match. The
journals beside this file quote the original paths, and are left as written: they record where the
data went at the time. Any symlink still pointing at the old `common/reference/...` prefix dangles
and must be re-pointed where it lives.

Each carries an OpenMM-only bundle made by `md-openmm export-reference`. The REST2 dataset is the
whole run directory, so its stage-to-stage handoffs are verified by digest; the cMD ones were
staged as `bundle/`, `data/` and `provenance/`.

**The explicit arm was withdrawn** on 2026-09-11. Whole-system coordinates every 50 ns -- six
frames in a 300 ns run -- are too few for the analyses this data is for. `hot_explicit` and
`rest2_explicit` had been registered and were removed; the three `cold_explicit` runs were stopped
at 364-367 ns and removed. The explicit arm is to be rerun with a finer whole-system interval,
which is not decided yet.

## Engine

Every registered run ran the `md_tools` installed in the shared environment: version string
`0.5.0.dev0`, with all 99 installed files byte-identical to MD-tools commit `514e44b`. Those files
were installed 17 minutes before `514e44b` was committed, so the commit identifies the engine's
content rather than a commit the wheel was built from. The runs record `md_tools_commit: null`,
because that build could not report one.

The campaign directory's own README said "MD-tools `d157c63` (0.5.1)". That was wrong for every run
in it. Some explicit runs were briefly launched with `PYTHONPATH` pointing at a source tree
(2026-09-11, 07:11-07:48 UTC); all of them were discarded, and nothing registered comes from them.

## Things to know before using the data

1. **Equilibration.** Implicit runs: 100 ps restrained NVT (they predate the protocol below).
   Explicit runs: 10 ps restrained NVT + 10 ps restrained NPT. Both at 1.0 kcal/mol/Å². The
   protocol decided on 2026-09-11 is NVT 10 ps + NPT 10 ps for explicit solvent and NVT 20 ps for
   implicit, with solute positional restraints at 1 kcal/mol/Å².
2. **Ensembles.** The explicit cold runs were NPT; everything else is NVT, and a REST2 state is NVT
   at every τ including 0. So an explicit cold run is not the same ensemble as the bottom state of
   an explicit ladder. The implicit side has no such mismatch.
3. **A scaled explicit run cannot equilibrate its own box.** `hot_explicit` started from
   `cold_explicit_r1`'s NPT-equilibrated state (0.9934 g/cm³) rather than `build-top`'s box
   (0.947 g/cm³), through `run_from_equilibrated.sh`. A rerun needs the same.
4. **`cMD-hot/run1` (implicit) holds 59,980 solute frames**, 100 ps short of 300 ns, where every other
   300 ns run has 60,000. Not yet explained.

## Files

| path | what |
|---|---|
| `configs/` | the `build-top` (`sys.*.config`) and `build-md` configurations exactly as run |
| `systems/` | the built Systems and topologies: `implicit.xml` sha256 `b60c8451…`, `explicit.xml` sha256 `adb35cb7…` -- the digests every run record names |
| `schedule.py` | the GPU scheduler, with the device list made an argument |
| `run_from_equilibrated.sh` | the `hot_explicit` launcher described above |
| `journals/` | the two execution journals, copied unedited from `MD-project` at `94ad0cd` |

## Registering a run the same way

```bash
md-openmm data-register -idata <run directory> --common-data \
    -project_name reference -data_name 2026-09/ALA-dipeptide-<solvent>/<protocol>/run<N> -year 2026 \
    --notes "<engine, and how you know it>"
```
