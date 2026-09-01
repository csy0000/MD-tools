# First refactor: from a research branch to a generic OpenMM template

**2026-08-16.** Status: **five of seven requirements complete, two outstanding.** The keyword
acceptance check passes, 209 tests pass, and both force-field routes run end to end through the
installed command from outside the checkout. Requirement 1 is partially done and requirement 3 is
not started; neither is claimed.

Executed against `claudecode-instructions/20260816_gpt_fix2.md`.

> **A note on names.** This journal cannot spell the retired identity, because the acceptance check
> in requirement 6 greps for it across the whole tree and must return zero matches. The previous
> distribution, import package and console script are therefore referred to descriptively. That is
> a real tension between "erase the old identity" and "record the migration", and it is worth
> knowing about before someone wonders why this reads so indirectly.

## What the seven requirements asked for

A generic, reusable base for conventional MD and omega-selective REST2 in explicit water: an
explicit run/resume contract, chunk counts as inputs, correct statistics across resumes, the
selected NPT box in the written structure, honest salt accounting, the old project identity gone,
and omega exclusion as the default implementation branch.

## Requirement 6 — identity and layout (done)

The distribution, import package and console script are now `md-tools`, `md_tools` and
`md-openmm`. No compatibility alias, shim or deprecated entry point: the instruction forbids them,
and they would keep a research project's name alive inside a general template.

**The prune was much larger than the rename.** The explicit path used exactly **four symbols** from
~2,600 lines of shared research code -- the REST2 Hamiltonian scaler, the neighbour-pair schedule,
the Metropolis criterion and the ladder validator. Those four were extracted and the entire old
source tree deleted: 8,087 lines of sampling methods, implicit-solvent builders, topology
preparation and collective-variable machinery that no explicit-water run ever calls.

The implicit-solvent branch of the REST2 scaler was deliberately not carried across. It now raises
a precise error naming the problem, rather than silently scaling a Hamiltonian this template does
not support.

The 2,419-line monolith became six modules in a strictly acyclic chain:

    config -> system -> solvation -> equilibration -> md -> rest2

The cut points are the module's own section boundaries. The cross-module imports were computed with
pyflakes rather than guessed, and the chain was checked for forward references before anything
moved.

## Requirement 2 — the chunk count is an input (done)

Both runners computed `n_chunks = int(round(total_ns / chunk_ns))`. That accepts a total which is
not a whole number of chunks and then runs a different length than the manifest declares: 0.0025 ns
of 0.001 ns chunks became 2 chunks, silently. Both sites are gone.

`n_chunks` and `chunk_ns` are the canonical inputs; totals are derived and reported only, and
nothing reads a total back to recover a plan. One validator checks all of it, including that
`bool` is rejected -- `True` is an `int` in Python and would otherwise be read as "one chunk",
turning a configuration error into a plan.

Experiment manifests went to schema 2. System manifests stayed at 1, with their own check: only the
experiment schema changed, and re-versioning every system manifest for an unrelated change is churn
that also makes "which schema broke" unanswerable from the number.

A v1 manifest is refused with a message naming the plan **that manifest meant** -- "exactly
n_chunks: 2, chunk_ns: 0.001" when the division is clean, and when it is not, that 0.0025 / 0.001 =
2.5 was never a whole number of chunks and the old code rounded it.

## Requirement 4 — the written structure carried the wrong box (done)

Equilibration selects a handoff state from the NPT tail -- coordinates, velocities and box vectors
that actually occurred together -- and then wrote the human-readable structure through the
simulation's topology, which still held the box it was **built** with. The CRYST1 record described a
cell the coordinates were never equilibrated in. Nothing downstream reads the PDB for production,
which is exactly why it could sit there indefinitely and mislead anyone who opened the file.

The regression test then found something the requirement anticipates: **PDB cannot round-trip a
triclinic cell.** CRYST1 stores lengths and angles to three decimals in angstrom, and reading it
back gives OpenMM's reduced lattice form, which for the dodecahedral box flips the sign of the third
vector's x/y components. Volume survives; the vectors as written do not. An mmCIF copy is now
written alongside and the authority order is stated: the serialized State is authoritative, the
mmCIF is the faithful structure, the PDB is for viewing.

## Requirement 5 — counterions are not salt (done)

Two independent calculations misreported ionic composition in two different ways:

* `min(ions.values()) * 55.5 / n_water` -- a minimum over whatever species happened to exist, so a
  box whose only ion was the counterion neutralising a -1 solute reported 0.0555 M of salt;
* `n_ions / 2` -- every ion treated as half a salt pair, true only when no neutralisation occurred.

One audited function now counts each species separately (absent is zero, never "missing"), takes
added salt as `min(n_positive, n_negative)`, attributes the remainder to neutralisation, and checks
that the leftover signed charge cancels the declared solute formal charge. If it does not, the box
is not the system the manifest describes, and that is now visible.

Ionic strength is computed from explicit valences rather than assumed equal to the pair molarity.
Multivalent and unrecognised species are refused, because applying the monovalent formula to them
produces a plausible wrong number.

## Requirement 7 — omega exclusion (done)

One canonical setting, `rest2.omega_exclusion`, default true, with `--omega-exclusion true|false`
on the commands that build or run a Hamiltonian and `--no-omega-exclusion` mapping onto the same
setting. An unparseable value is refused rather than defaulted -- the two values are different
physics. Unset means "whatever the manifests resolved to", which is not the same as false.

The override is recorded with its source, so a run says where the value came from. It stays in the
build-defining projection, so a bundle cannot be reused across a toggle.

With the exclusion **off**, an unclassified amide no longer blocks production: the block exists
because scaling or not scaling an ambiguous bond are different Hamiltonians, and with no bond
treated specially there is no distinction left to protect.

## Requirement 1 — run directories (partial)

Done: the naming contract. `--run-name NAME` gives exactly `ROOT/NAME` with nothing appended -- a
decorated name is not the name the user chose, and it makes the directory unpredictable to anything
scripted around it. `--resume-run DIR` continues in that exact directory and creates no sibling,
because a resume that silently starts a new run leaves two partial trajectories and no error. The
two are mutually exclusive, a fresh run still refuses to overwrite, and default names now carry the
method so an MD run and a REST2 run of the same configuration cannot collide.

**Not done:** the continuity contract itself -- the equality checks over fingerprint, topology,
force field, integrator and chunk plan performed *before* any state is loaded; atomic
checkpoint/State/`done.json` writes; the `--resume-from-state` fallback when a checkpoint is
incompatible but the portable State is valid; and the end-to-end tests over crash and extension
cases.

## Requirement 3 — statistics across resumes (not started)

REST2 summaries still describe the invocation that finished the run rather than the whole run.
Reconstructing attempts, acceptances, pair statistics, walker occupancy and phase parity from the
exchange log at finalisation is untouched.

## Four defects, each found by running something

This is the through-line worth keeping.

1. **The pre-NPT box** above -- found by writing the test the requirement asked for.
2. **Two salt formulas** -- found by reading them against the requirement's definition.
3. **The fingerprint function, given an already-built projection, hashed a mapping of `None`s** and
   returned a plausible constant, so unrelated systems compared **equal**. Discovered because a new
   test double-projected and its "the setting does not reach the fingerprint" failure was, in fact,
   the test's own bug. The function now refuses that input, so nobody has to work it out twice.
4. **The requirement-4 fix broke every smoke.** Deep-copying an OpenMM topology yields new atom
   objects while the copied bonds still reference the originals, so the PDB writer built its index
   from the new atoms and died with `KeyError` on the first bond. All 197 unit tests passed
   throughout: a topology built in a test has no bonds. The box is now set on the real topology and
   restored in a `finally`.

The fourth is the one to remember. A green suite meant nothing here; running the actual pipeline
took thirty seconds and found it immediately.

## What was deleted, and where it went

The target-machine validation, the peptide-route report, the ladder-pilot summaries, six working
journals and the handoff-package document were removed. Every command in them names an entry point
this template no longer has, and the schema, the salt fields and the equilibrated outputs have all
changed, so they no longer describe this package. Rewriting the names would have turned a true
record into a false one. They are preserved in git history at `f885be5`
(`git show f885be5:reports/`), and the README says so.

**One file was restored after deletion**: the Phase-A prepared-system record that the shipped RGD
manifest *cites* as the evidence for its dodecahedral box. A test exists precisely to assert its
presence -- "the box shape is only defensible while this artifact is preserved" -- and removing it
recreated a missing-citation defect that had been found and closed the day before.

## Verification

    compileall              clean
    pytest -m "not slow"    209 passed, 2 deselected
    keyword acceptance      0 matches, and no old name in any path
    md-openmm --version     md-tools 0.1.0
    both routes             ligand and peptide, rc=0, completed, 4/4 rounds,
                            run from /tmp with no source checkout on the path

## What this does not establish

Mechanical correctness of the refactor, not scientific validity of anything. No ladder is validated
for any system by this work, the 2 fs / 4 fs equivalence gate is still open, restart safety is
explicitly *not* established -- that is requirement 1's outstanding half -- and convergence is
untouched. The smokes are picoseconds and prove execution only.
