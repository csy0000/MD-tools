# Scope: what this repository should be, and what it should stop testing

Date: 2026-08-25
Status: **open question, raised by the repository owner.** Nothing has been deleted. This entry
records the measurements and the argument so the decision can be taken with numbers rather than
impressions.

## The position

> The repo should only be generating the input files for MD simulations (for now only OpenMM),
> instead of a huge self-contained application that makes sure everything works. It should provide
> scripts the user can use to generate inputs to run simulations on different engines, and that
> should be enough. […] We shouldn't do the checks on NAGL and AM1-BCC, or the effect of box size
> and barostat. These should all be OpenMM's job. […] From my view only the NpT REST2 is new from
> the current OpenMM version.

## Evidence that supports it

Every defect found during the 2026-08-24/25 work was **plumbing, not physics**:

| defect | class |
|---|---|
| `md-openmm config resolve` raised AttributeError on every configuration | input generation |
| the lock file recorded `commit: null` for any wheel install | provenance |
| `python -m …cli_system_gen` exited 0 having written nothing | CLI contract |
| eight new profiles made implicit profile selection ambiguous | configuration |
| the implicit builder silently discarded the requested hydrogen mass | input generation |
| `envcheck` shadowed its own parameter, breaking `prepare --config` on CPU | CLI contract |

Not one was a physics defect. Every one produced a wrong or unusable **input file**, which is
exactly the product this repository claims to deliver.

## Where the cost actually is

Measured 2026-08-25, `pytest tests/ -n 12 --dist loadfile`, 48-core machine:

```
total suite CPU                          1359 s
runs real MD to test restart/crash        980 s  (72%)
verifies physics/chemistry                 63 s  (5%)
everything else (config, provenance)      316 s  (23%)

wall time 207 s -- bounded by ONE module, test_bundle_portability.py at 198.9 s,
because --dist loadfile cannot split a file across workers
```

**Deleting the physics tests saves 5% of the runtime.** The scope objection is still correct on its
own terms -- maintenance burden and creep -- but it should not be expected to make the suite fast.
The 72% is real MD run to verify restart bookkeeping.

## Proposed triage

### Delete: OpenMM's or OpenFF's job

- `test_nagl_charges` -- comparing NAGL to AM1-BCC numerically. Keep exactly one test: requesting
  `am1bcc` without `sqm` must FAIL rather than silently substitute. That refusal is ours.
- the GBn2 potential/force invariance measurement -- it can only fail if OpenMM breaks, because no
  term in the potential reads a mass. It documents a premise; it guards nothing we control.
- `test_box_geometry` and `test_dodecahedron_geometry_audit` (22 tests). That audit concluded the
  implementation was right and our comments were wrong. The conclusion deserves a comment.
- anything measuring what the barostat DOES. Keep only "exactly one barostat is attached when NPT
  is declared" -- we attach it.
- the box-headroom fix made earlier today is the same category: a test that should never have been
  stressing box physics.

### The open question: the 72%

The crash/restart suite runs real dynamics to assert **file-level bookkeeping** -- committed
generation records, appended exchange history, walker mapping, monotonic step counters. None of
that needs real integration. A stub propagator that advances a counter would assert all of it, with
ONE end-to-end REST2 run kept as an integration smoke.

That is where the hours are, and it appears cuttable without losing anything this repository owns.

### Keep

- **NPT REST2** -- the general reduced potential `u_k(x,V) = b_k(U_k + p_k V)`, exchange
  acceptance, state swap, walker mapping. This is the one place the repository implements physics
  OpenMM does not provide, and it matches the owner's read that it is the only genuinely new piece.
- configuration resolution, the refusals, provenance and the lock file, and that a generated
  project still runs after relocation. That is the product.
- the built-System-vs-configuration guard, whose claim -- "the input we generated is the input you
  asked for" -- is the product's central promise, and which caught a real defect. Its
  implementation is heavy (8 bundle builds) and could check 2.

## Note on how this repository drifted

The recurring failure mode has been the same fact declared in several places and nothing checking
they agree: the ligand charge default lived in five, and a change updated two. The response each
time was another guard. Guards are cheap and have caught real drift -- but a repository that needs
many of them is telling you the declarations should be fewer, not that the guards should be more.
That is worth weighing alongside the scope question.

Nothing in the triage above has been actioned. `dev` is at the state described in
`2026-08-25_implicit-hmr-and-the-built-system-gate.md`, plus the dependency map and this entry.
