# Backlog

Known gaps that are **not fixed**. None of them is a release gate: none has been shown to
corrupt, discard or misreport scientific output, or to prevent recovery of an interrupted run.
Each entry says what it is, what it costs today, and the concrete thing that should cause it to
be picked up.

Nothing here should be read as "handled". If one of these turns out to affect a result, it stops
being backlog and becomes a defect.

---

## 1. The aggregate CV-cost record is not cross-checked against the prefix costs it sums

**What.** A campaign's aggregate cost record (`aggregation`, `per_state` / `per_path`) is parsed
strictly — every scope, every count, every `wall_seconds` — and each per-state or per-path entry
is checked against the prefix it belongs to. What is *not* checked is that the aggregate's own
totals equal the sum of those entries. A record whose parts are individually valid and whose
total is wrong is accepted.

**Impact.** Reporting only. The aggregate is never read back to decide a truncation, a resume
point or any scientific value; it is metadata describing how much CV evaluation a campaign cost.
A wrong total would mislead someone comparing the cost of two campaigns.

**Do not** use aggregate CV-cost metadata for efficiency comparisons between runs until this is
closed. The per-state and per-path entries are individually verified and are the trustworthy
numbers.

**Trigger.** Pick this up when CV cost is used to make a decision — scheduling, budgeting, or a
published efficiency claim — or when a campaign's reported total is questioned.

---

## 2. Missing completion-cost metadata, and exhaustive malformed-metadata cases

**What.** Two related gaps. A completion manifest that carries no cost record at all is not
refused the way a *committed prefix* without one is (that case is closed, and refuses). And the
malformed-metadata coverage is representative rather than exhaustive: it covers the field shapes
that decide a truncation or a restoration, not every field of every record under every mutation.

**Impact.** A completed campaign whose manifest lacks cost metadata reports no cost rather than a
wrong one. The unexercised mutations are in fields that no continuation reads.

**Trigger.** Pick this up when a manifest is found in the wild without its cost record, or when
manifest metadata starts being consumed by something other than a human reader — a dashboard, an
analysis package, an automated pin.

---

## 3. Broader provenance/schema hardening and diagnostic-file transactional behaviour

**What.** The read-only continuation boundary protects trajectories, CV/work/HS tables, committed
checkpoint generations and pointers, completion manifests and authoritative configuration. It
does not make *diagnostic* files transactional: a refused attempt may append to a `.out` or a
`.log`, and a crash mid-write can leave a partial diagnostic line. Schema-version handling is
strict where a record decides scientific behaviour and permissive elsewhere.

**Impact.** Diagnostics may be untidy after a refusal or a crash. This is deliberate under the
current preservation rules — a rejected attempt is allowed to say why it refused — and no
authoritative record is involved.

**Trigger.** Pick this up if a diagnostic file is ever made authoritative for anything, or if
partial diagnostic lines start breaking a downstream parser.

---

## 4. One unexplained AIS MPI failure, with no retained diagnostic

**What.** A single run of `tests/test_cv_mpi_cuda_ais.py` reported `1 failed, 10 passed`. The
diagnostic was lost to an output filter before it could be read, and the file has passed every
run since — thirteen consecutive clean full-file runs at the time it was recorded, plus the runs
in this pass. The cause is unknown. It is recorded as unresolved historical evidence, not as a
diagnosed and fixed defect and not as an accepted limitation.

**Impact.** Unknown, which is the problem. It has never recurred and has never been reproduced,
so there is nothing to characterise.

**Trigger.** Pick this up the moment it recurs. Every launcher call in that lane now carries a
subprocess timeout and retains complete stdout and stderr, so a recurrence will arrive with the
diagnostic this one lacked. A reproducible scientific, restart or MPI failure is a release
blocker; this one is not, because it is not reproducible.

---

## 5. `constraints.type` is ignored under implicit solvent, and the log says otherwise

**What.** The explicit-solvent route honours `constraints.type`; the implicit route passes
`app.HBonds` unconditionally. `AllBonds` + GBn2 builds a System identical to `HBonds` + GBn2 (12
constraints, none heavy-heavy, on the same input where TIP3P + `AllBonds` adds 9), while the build
log records `constraints.type  AllBonds  (set)`.

**Impact.** Not the ignored setting so much as the record naming a setting that did not apply. No
run in this repository is affected: `HBonds` is the default and what every validated
implicit-solvent run asks for, so request and outcome coincide. Anyone asking for `AllBonds` or
`HAngles` under implicit solvent gets `HBonds` and a log that says otherwise.

**Related, smaller.** The build schema's enum offers `constraints.type: "None"`, but
`_check_constraints` compares against Python `None` rather than the string, so the advertised
value is refused with "not supported". Conversely `_check_constraints` accepts `HAngles`, which
the schema enum does not offer; the enum is the tighter and intended policy.

**Trigger.** Pick this up when someone needs a non-`HBonds` constraint setting under implicit
solvent, or before any implicit-solvent build log is used as evidence of what was constrained.
Two defensible resolutions and the choice is a policy call: make the implicit route honour the
setting, or refuse anything but `HBonds` there so the request cannot disagree with the outcome.
Documented in `docs/scientific-defaults.md` section 11.2.

---

## 6. AM1-BCC charges are not reproducible between builds of the same input

**What.** Two `build-top` runs of the same `.smi`, same configuration, same machine, produce
different partial charges. Measured on a 43-atom cyclic peptide: **41 of 43 atoms differed**, one
example atom moving from -0.63047675 to -0.71247675 e.

**Cause, from the installed source.** The ETKDG embedding IS seeded, and `build-top` refuses an
unseeded one precisely so "this prepared system would not be reproducible" cannot happen. But the
OpenFF toolkit then discards that conformer: `AmberToolsToolkitWrapper.assign_partial_charges`
does `if use_conformers is None: mol_copy._conformers = None; mol_copy.generate_conformers(...)`,
and `md_tools.openmm.system` calls `offmol.assign_partial_charges(scheme)` without
`use_conformers`. The charges therefore come from an unseeded conformer that is neither the one
recorded in the build provenance nor the same one twice.

**Impact.** Two: no ligand or peptide-like build is bitwise reproducible, and the recorded ETKDG
conformer provenance does not describe the geometry the charges were computed on. Nothing is
wrong within a single build -- the charges are real AM1-BCC for the conformer that was used -- and
every published dataset remains internally consistent and self-describing. What cannot be done is
reproduce one from its inputs, or compare two builds expecting only an intended difference.

**Not fixed here, deliberately.** Passing `use_conformers=` would change the charges of every
future build, which is a scientific decision rather than a bug fix, and this was found while
implementing something else. It is also why `tests/test_peptide_like_hamiltonian.py` proves "only
the GB parameters change" on ONE prepared structure rather than by differencing two builds.

**Trigger.** Pick this up before any study that needs to reproduce a build from its inputs, before
comparing two builds for anything but their radii, or when deciding whether recorded conformer
provenance should be binding. The fix is small; its consequences are not.

---

## Cross-references

- `docs/release-notes/20260907-cv-validation-final-evidence.md` — the evidence behind the
  validation work these gaps were carved out of, including its own "Open item" section for
  entry 4.
- `docs/release-notes/20260907-readiness.md` — the readiness note for the pinned commit.
