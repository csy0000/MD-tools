# Deterministic bundle generation

**2026-08-21.** Building the same system bundle twice produced two different bundles. Found while
verifying an unrelated claim — that the alanine validation was unaffected by the ligand-route water
change — by rebuilding its bundle and comparing. The resolved configuration matched byte for byte,
which answered the original question, but `system.xml` did not.

Follow-on to [`2026-08-21_cmd-review-fixes-and-opc-default.md`](2026-08-21_cmd-review-fixes-and-opc-default.md).

## What was wrong

Two builds from byte-identical configuration differed in `system.xml`, `topology.pdb`,
`initial_state.xml` and `forcefield.json`. The total mass differed by **18.016 amu** — exactly one
water molecule — and the box edge by 0.04 Å.

This matters more than a hash mismatch. Bundle hashes, `relocate-check`, the golden configuration
hashes and the whole provenance story rest on a bundle being a function of its inputs. "Which of
these values did I choose, and which did the package choose for me?" cannot be answered by
comparing two bundles when rebuilding the same one gives a different answer every time.

Three independent causes, none of them seeded:

| # | source | measured effect |
|---|---|---|
| 1 | `Modeller.addHydrogens` places each new hydrogen from a **random direction** drawn from Python's global `random` | repeat calls moved hydrogens by up to **0.18 nm** |
| 2 | the relaxation `addHydrogens` runs afterwards is order-dependent on a threaded platform | ~1e-4 nm of residual drift once the RNG is fixed |
| 3 | `Modeller.addSolvent` chooses which waters become ions from the same global `random`; OpenMM 8.5.2 exposes **no** `randomSeed` parameter for it | different ion sites every build |

Cause 1 is the one that mattered. A tenth of a nanometre on a hydrogen is not rounding: the box is
sized from the solute's extent, so it changed the box, and that was enough to fit one more water.

## The fix

All three are now derived from the run's master seed, so solvation and protonation join the seed
map instead of being unrecorded sources of variation:

```
structure/protonation -> seeds addHydrogens, and pins its relaxation to the Reference platform
structure/solvation   -> seeds addSolvent's ion placement
```

Seeding alone was not enough — it left cause 2, and that residual is still enough to move the box.
The relaxation therefore runs on **Reference**, which is single-threaded and reproducible. Measured:

```
default platform, unseeded    max |delta| 1.796e-01 nm     NOT reproducible
seeded, default platform      max |delta| 4.371e-04 nm     NOT reproducible
seeded, CPU platform          max |delta| 2.298e-05 nm     NOT reproducible
seeded, Reference platform    max |delta| 0.000e+00 nm     reproducible
```

Reference is slower, but it minimises only the added hydrogens of a solute — a macrocycle or a
small peptide here — so the cost is seconds. Both seeds and the platform choice are recorded in the
bundle provenance rather than being silent.

The global RNG state is saved and restored around each call, so this changes protonation and
solvation and nothing else in the process.

## Evidence

`tests/test_bundle_reproducibility.py` (3 tests):

* hydrogen placement is reproducible under a fixed seed, **and** genuinely driven by that seed —
  a different seed must give a different answer, or the test would pass against a stub;
* two builds through `MD_system_gen.py` are byte-identical across all four defining files;
* the rebuild has the same water count, particle count and total mass — asserted directly, because
  "the hashes differ" does not tell you that one water molecule appeared.

| gate | result |
|---|---|
| reproducibility tests | **3 passed** |
| complete suite | **927 passed, 0 failed, 0 error** |
| `fast` / `integration-cpu` | green on the final SHA |

## The 1 ns validations were re-run

Both alanine validations were regenerated from freshly seeded bundles and re-checked, so the
recorded hashes now describe a build that can be reproduced. Every acceptance check passes
unchanged: 2 committed generations, exactly 1000.0 ps, checkpoint-preferred restart, 10 all-atom
and 100 selected frames, one log header, monotonic steps, no NaN or Inf, implicit nonperiodic with
a null box volume.

```
explicit  cMD_1_all_atoms.dcd       7fb8b0244a814cc5c561baf0ed1bdddf3093ae26c8133a402c12caf7e4701616
explicit  cMD_1_selected_atoms.dcd  21d6d4e3ff1fdb3fe4f1efaf7addfabe0d96afbed0ff4d2824db97dd378502e7
explicit  cMD_1.log                 591fbf791768404701a0c723b6dff364f42483f7ad7d425795b07f6a7d71701f
implicit  cMD_1_all_atoms.dcd       94296b1bd38f5df7d3958c1001561ff401d5137048ccb07c0d0cd984edd49595
implicit  cMD_1_selected_atoms.dcd  8335abfd10912ddec85fd5ac66a595dffa2606855f332a8679da160c92e4df93
implicit  cMD_1.log                 d957ff39190de44716fd4a4324f4bad0ed444c09bdc94652975d93abe737f8f2
```

**These supersede the hashes in the previous journal**, which were produced from unseeded bundles.
The physics is unchanged — same force fields, same protocol, same seeds for the dynamics — but the
starting coordinates came from a build that could not be repeated, so those hashes documented a run
without being reproducible by rebuilding it. These are.

Explicit on GPU 0 (RTX A5000, `GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1`), implicit on GPU 1
(RTX 3080, `GPU-96ce533d-9d42-acf6-8384-5e27150e9a85`), `CUDA_DEVICE_ORDER=PCI_BUS_ID`, driver
580.173.02, OpenMM 8.5.2, ParmEd 4.3.1, Python 3.12.13, CUDA mixed precision.

## What this does not fix

Reproducibility here means *same machine, same library versions*. A different OpenMM release may
place hydrogens differently or ship a different water box, and `topology.pdb` would change again.
That is a version-pinning question, not a seeding one, and the bundle already records the versions
it was built with.
