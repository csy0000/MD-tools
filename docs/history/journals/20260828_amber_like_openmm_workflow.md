# An Amber-like OpenMM workflow

| | |
|---|---|
| Date | 2026-08-28 |
| MD-templates branch | `feat/amber-like-openmm`, from `dev` at `bdd3493254e8b3febb728735dda35e3c1508b0ac` |
| MD-project branch | `dev3-amber-like`, from `dev2-test-basic` at `53fda1e32cb502f29c4829d39a6415e19b3058cb` |
| Goal | make ordinary cMD look like `pmemd`/`gmx mdrun`: a short readable script, run directly, writing a `.out` |

Both remote heads matched the instruction exactly before any edit.

---

## Phase 1 — audit of the existing implementation

### The command surface today

```text
md-openmm sys-config     write sys.config.yaml and md.config.yaml
md-openmm show-default   print a default configuration block
md-openmm sys-gen        build the OpenMM system
md-openmm md-gen         generate run scripts
```

Four commands, and the user supplies two YAML documents before anything runs. There is no single
entry point that takes "this structure, 1 ns, 5 ps" and produces a runnable directory.

### Why the generated scripts are long

A generated cMD project carries **2,084 non-blank lines of Python** that a user has to trust:

| file | lines | what it is |
|---|---|---|
| `preflight.py` | 754 | the shared safety gate |
| `md_stages.py` | 503 | shared helpers, including `make_simulation` |
| `cMD/run.py` | 286 | the production stage |
| `minimization/run.py` | 277 | a common stage |
| `cMD/rest2_scaling.py` | 264 | tau scaling, present even at tau = 0 |

Of `cMD/run.py`'s 286 lines, the lines that are recognisably an OpenMM calculation number about
**twelve**: one `PDBFile`, three `DCDReporter`/`CheckpointReporter`, two `StateDataReporter`, one
`simulation.step`, one `saveCheckpoint`, and the imports. **The `Simulation` and the integrator are
not even in the script** -- they are built by `make_simulation()` in `md_stages.py:192`. So the
script a user opens to see "what was run" does not contain the run.

The remainder is configuration parsing, resolved-request fingerprinting, completion records,
invocation history, provenance dictionaries, continuation arithmetic, and the preflight call.

### Classification of current generated-script behaviour

| class | behaviour |
|---|---|
| **scientifically necessary in every stage** | topology/system/state loading; integrator; barostat; platform; `Simulation`; reporters; `step()`; final state and checkpoint |
| **necessary at generation time** | force-field resolution; solvation; timestep/HMR validation; interval-to-step arithmetic; seed derivation; stage ordering |
| **necessary only for safe execution** | parent-state existence; refusing to overwrite a completed stage; refusing a runtime request that contradicts the generated one |
| **registration-owned** | dataset manifests, checksums, MD-data validation, lifecycle |
| **historical compatibility** | `rest2_scaling.py` copied into cMD at tau = 0; `invocations.jsonl`; the dual `resolved_stage.yaml`/`resolved_run.yaml` pair |
| **unnecessary in generated scripts** | YAML parsing of the generator configuration; provenance dictionaries; environment capture; fingerprint recomputation; component/repository awareness |

The last row is the target of this milestone. The middle rows move to generation time or to a small
focused check; nothing scientific is dropped.

### Defaults, as they actually are

**A correction to the instruction's Phase 1 wording**, which refers to "the current ff19SB/OPC
peptide defaults". That is not the default:

```python
EXPLICIT_COMBINATIONS = {
    "TIP3P": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"},   # <- the default
    "OPC":   {"protein": "amber19-all.xml", "water": "amber19/opc.xml"},     # <- available
}
```

`sys_defaults(peptide=True, solvent="TIP3P")` resolves to **ff14SB + TIP3P**. ff19SB/OPC is a
supported selection, not the default, and `EXPLICIT_PROTEIN_FORCEFIELD` is a compatibility alias
kept "for readers of older records: the resource the 0.3.x default named". Every dataset in this
work was built on the ff14SB/TIP3P pair. **These defaults are preserved unchanged**; this milestone
changes the interface, not the science.

| | value |
|---|---|
| peptide, explicit | `amber14-all.xml` + `amber14/tip3p.xml` |
| peptide, implicit | `leaprc.protein.ff14SB` + GBn2/mbondi3 |
| ligand | OpenFF Sage 2.2.1 + AM1-BCC |
| solvent box | 1.5 nm padding, dodecahedron, 0.15 M, 1.0 nm cutoff |
| integrator | LangevinMiddle, 300 K, 1 ps⁻¹, 2 fs, HBonds, rigid water |
| equilibration | nvt_1kcal 10 ps → npt_1kcal 10 ps → npt_free 10 ps (explicit); nvt_1kcal → nvt_free (implicit) |

### Safety invariants to keep

* timestep above ~3 fs requires hydrogen-mass repartitioning, refused at generation;
* output intervals must be an integral number of steps;
* a completed stage must not be silently rerun or overwritten;
* a parent stage's final state must exist before a child runs;
* the data root must be outside every Git worktree;
* implicit solvent implies NVT and no barostat.

### Existing test coverage to preserve

cMD, REST2 and AIS stage layout and records; continuation safety; NAGL resolver; charge provenance;
the MD-project consumer gate. `-m "not gpu"` 354 passed, `-m gpu` 77 passed at `f3133d3`, whose tree
is the tree of `dev` at `bdd3493`.

### MD-project side

Examples live under `configs/{systems,simulations}`, data under `$MD_DATA` outside every worktree,
with `components/**` and `components-dev/**` ignored. There is no `md/` symlink convention yet.

---

## The MD-project side of the audit

This copy of the audit lives in MD-project because the classification governs both repositories:
MD-templates decides what a generated script contains, MD-project decides what it commits.

### What the project tracks today

| tracked | example |
|---|---|
| small source inputs | `inputs/structures/example_ace_ala_nme.pdb`, `inputs/ligands/*.smi` + `.identity.yaml` |
| generator configurations | 14 `configs/systems/*.sys.config.yaml` + 14 `configs/simulations/*.md.config.yaml` |
| declarations | `datasets.yaml`, `project.yaml` |
| evidence | `results/devtest/*`, `docs/journals/*` |
| dev-test tooling | `tests/devtest/*` |

Generated data live under `$MD_DATA` (`$DATA_ROOT/MD_DATA_fix`), outside every Git worktree,
proven by `git rev-parse --show-toplevel` failing there. `components/**` and `components-dev/**`
are ignored (`.gitignore:28-31`).

**There is no `md/` convention yet.** A reader of the project cannot follow a path from the
repository to the data; they have to know `$MD_DATA` and the canonical layout. Phase 3 adds the
ignored `md/<system>` symlink to close that gap without tracking machine paths.

### What this milestone must not disturb

* the 15 existing datasets and their recorded generators (3 × `822759da`, 12 × `30248428`);
* `datasets.yaml` and the 28 configuration files describing them;
* the consumer gate and its historical scoping;
* the component lock, which stays as historical compatibility evidence per Phase 11.

### The integration boundary this milestone establishes

| the project owns | `$MD_DATA` owns |
|---|---|
| project metadata, small source inputs, small setup requests | generated system files and exact stage scripts |
| project-developed simulation and analysis code | `.out` files, trajectories, states, checkpoints |
| `md/README.md` and ignored `md/<system>` symlinks | large derived data |
| accepted small results | |

The new workflow must run with no component checkout present, which is the sharpest difference from
the current architecture: today a generated project imports `preflight.py` and `md_stages.py`
copied beside it, and its stage script reads `md.config.yaml`. After this milestone a generated
stage is a standalone OpenMM script.

---

## The project side: what was added

| | |
|---|---|
| `md/README.md` | tracked; explains the boundary |
| `/md/*`, `!/md/README.md` | ignore rules |
| `tests/devtest/link_system.sh` | safe symlink helper |
| `examples/{ALA,phenol-IPH}/setup.yaml` | the documented 1 ns requests |
| `examples/{ALA,phenol-IPH}/setup.smoke.yaml` | 20 ps, for automated testing |
| `inputs/ligands/phenol_IPH.{smi,identity.yaml}` | the ligand source and its identity |
| `tests/devtest/test_amber_like_workflow.py` | 17 boundary tests |

### The link helper refuses on the right question

It asks Git, with `rev-parse --show-toplevel`, not `.gitignore`. An ignore rule is a statement
about what Git *shows*; the question here is whether the data are inside a repository at all, and
only Git can answer that. A trajectory inside a worktree is one `git add -f` from being committed
forever.

```console
$ tests/devtest/link_system.sh x docs/
refusing: .../docs is inside the Git working tree at $DATA_ROOT/scheme/MD-projects.

$ tests/devtest/link_system.sh ALA "$MD_DATA/.../ALA"
md/ALA -> $DATA_ROOT/MD_DATA_amber/smoke/ALA
```

### phenol/IPH

`IPH` is recorded as an RCSB **Chemical Component Dictionary** component identifier, with
`identifier_kind: rcsb_chemical_component_dictionary` and a `/ligand/IPH` URL. It is **not** a
four-character PDB entry accession, and the record says so in as many words so a later reader
cannot mistake it.

The SMILES is committed, not downloaded. The CCD is revised over time; a build that reaches the
network cannot say which revision it used, and a committed string can be checked against the
identity record — which the tests do.

### Results

```console
$ pytest -q                                        308 passed, 1 skipped
$ pytest tests/devtest/test_amber_like_workflow.py  17 passed
```

Up from 291. Nothing regressed: the existing dataset, coherence, component-pin and consumer-gate
coverage is untouched, and the 15 existing datasets and their recorded generators were not read,
modified or regenerated by any of this work.

### The boundary, as implemented

| this repository | `$MD_DATA` |
|---|---|
| `examples/*/setup.yaml`, small source inputs | generated `system.xml`, `topology.pdb` |
| project metadata and analysis code | the exact stage scripts that ran |
| `md/README.md`, ignored `md/<system>` links | `.out`, trajectories, states, checkpoints |

**The new workflow does not depend on component locking.** `components.yaml` and
`components.lock.yaml` remain, untouched, as historical compatibility evidence for the existing
contract-managed datasets, per the milestone's instruction not to combine a repository-wide
component-management cleanup with this implementation.

### Cleanup recommendation, for a separate task

Once REST2 and AIS adopt the stage-file convention and the contract-managed path is retired, the
following become dead weight rather than evidence, and should be removed together rather than
piecemeal:

* `components.yaml` / `components.lock.yaml` and the tests asserting the pin, if nothing runs from
  a pinned checkout any more;
* `tests/devtest/generate_example.sh` and `verify_example.py`, whose job `setup` plus the `.out`
  contract now covers for cMD;
* the dual `resolved_stage.yaml` / `resolved_run.yaml` record pair, if registration reads `.out`
  files instead.

None of that is done here. Each one is a decision about what the project's provenance story is,
and bundling them with an interface change would make both harder to review.

### Registration boundary

Unimplemented and unmodified — MD-data is untouched. The intended future operation is documented in
both READMEs:

```bash
md-data register-system md/ALA --config md/ALA/config.yaml --root "$MD_DATA"
```

No competing registry was added to either repository.

---

## A correction to commit `74ec758`

Its message states `pytest -q  322 passed, 1 skipped`. **The real figure is 311 passed, 1 skipped**,
confirmed by repeated runs. I wrote the number into the message before running the suite rather
than after, and 322 is not a measurement of anything -- it is arithmetic I did in my head and got
wrong.

The commit is pushed, so the message stands as written and this is the correction. The count that
holds for this branch:

| | |
|---|---|
| before this milestone | 291 passed, 1 skipped |
| after the workflow tests (17) | 308 passed, 1 skipped |
| after the three gap-closure tests | **311 passed, 1 skipped** |

The arithmetic is consistent: 291 + 17 = 308, + 3 = 311.

Worth recording rather than quietly fixing, because a test count in a commit message is the kind of
claim a reader takes at face value and never re-derives.

---

## The file-interface pass, project side

MD-templates gained the `openmm-md` file interface; this repository consumes it.

| | |
|---|---|
| MD-templates pushed | `20ca906e3835174568067f897f03d0063c20f5fb` on `feat/amber-like-openmm` |
| pushed | exactly once, after its full suite passed |
| MD-project pinned to | that commit, before any MD-project test was run |

### The pin names the feature branch, deliberately

`components.yaml` requests `feat/amber-like-openmm` and the lock records
`resolved_ref: feat/amber-like-openmm`, not `dev`.

**`md-openmm setup` exists only on that branch.** `dev` is at `bdd3493254e8`, which has no such
command, so a lock naming `dev` would send a fresh checkout to a version lacking the command this
repository's own examples invoke -- and the instruction requires that a documented setup obtains a
version really providing it. The comment in both files says this returns to `dev` when the
interface merges, and `tests/devtest/test_component_pin.py` still enforces the permanent rule: if
the lock ever says `dev`, the pinned commit must be reachable from `dev`.

All 14 system configurations follow the lock, because a configuration's `templates.commit` is what
a *fresh* generation stamps. **The datasets on disk were not touched** and keep the commits that
produced them.

### What changed here

| file | change |
|---|---|
| `components.yaml`, `components.lock.yaml` | pin and requested ref |
| 14 `configs/systems/*.sys.config.yaml` | `templates.commit` only -- verified by diff, nothing else |
| `examples/*/setup*.yaml` | contributor field, with the resolution order documented |
| `README.md` | the `openmm-md` interface, the four-piece layout, launchers |
| `md/README.md` | `paths.sh`, a real launcher, moving `$MD_DATA` |
| `tests/devtest/test_amber_like_workflow.py` | interface, contributor and README-agreement tests |

Eight of the changed files are named `*rest2*` or `*ais*`. Their entire diff is the one `commit:`
line; no REST2 or AIS behaviour is modified anywhere in this branch.

### A test that could have gone quiet

The instruction asks for README command extraction "without silently skipping because an unrelated
development checkout is absent". Mine did exactly that -- `pytest.skip` when no checkout was found,
which reads as a pass.

It still skips, because it genuinely cannot compare a README against a CLI that is not present. But
`test_a_checkout_is_available_to_check_the_readme_against` now **fails** in that case, so the
absence is reported rather than swallowed. This is the same failure shape that recurred through the
earlier work in this repository: a check and the thing it checks drifting apart, with the result
coming back green.

### Results at the pinned commit

```console
$ pytest -q                       316 passed, 1 skipped
$ pytest tests/upstream -q         20 passed, 4 skipped, 0 failed   (simulation environment)
$ pytest tests/upstream -q         18 passed, 6 skipped, 0 failed   (project environment)
$ pytest tests/devtest -q         all passed
```

Bounded runs, every stage through its generated launcher with `import md_templates` raising
`ImportError`:

| run | platform | wall clock | stages |
|---|---|---|---|
| ALA, peptide route | CPU | 2m27s | 5/5 `run_status: completed` |
| phenol/IPH, ligand route | CPU | 2m10s | 5/5 `run_status: completed` |
| ALA, CUDA smoke | CUDA | **12.1s** | 5/5, every `.out` says `platform: CUDA` |

### Hygiene

No tracked trajectory, state, checkpoint, `system.xml` or `topology.pdb`; no tracked `md/` symlink;
`md/*` ignored and `md/README.md` tracked; no machine path in any committed file under `examples/`,
`inputs/`, `md/`, `configs/` or the component files.

### Limitations

* Production simulations were not run. The bounded smokes prove the interface, not sampling.
* The CUDA smoke is one 4 ps ALA chain on one device; no multi-GPU or long-run behaviour is claimed.
* The ETKDG embedding-seed gap remains open upstream and is untouched here.
* Component management is unchanged. Its removal stays a separate cleanup after REST2 and AIS adopt
  the new convention.

---

## Correction pass: documentation measured against the implementation

The README claimed setup asks "the same seven questions". It asks nine. And the `.out` example
named `input_state` and `final_state`, which the current runner does not emit -- they were replaced
by `coordinates` and `restart` when the file interface landed, and the documentation kept the old
names.

Both are now **extracted from the implementation by tests** rather than asserted as prose:

| documented thing | extracted from |
|---|---|
| the nine interactive questions | `_ask("...")` calls in `cli/md_openmm.py` |
| the `.out` field names | the header literals in `openmm/emit.py` |
| the default water model | `water: str = "..."` in `openmm/simple.py` |
| every `md-openmm setup` flag | the CLI's own `add_argument` calls |

A count is fragile; the list is the contract. If either side changes, the test fails here rather
than in a reader's hands.

### What else was missing

* the three run modes stated separately -- interactive, config-driven, and `--yes` as the only
  thing that suppresses confirmation;
* the contributor resolution order, and that there is no `$USER` fallback;
* the coupled water table, so a reader can select OPC correctly instead of overriding one XML;
* stage lineage as *previous stage restart → next stage coordinates*, which the launcher names
  explicitly with `-c`;
* the distinction between the five generated files, including the one that matters for archiving:
  **`.state.xml` is portable across machines and platforms, `.chk` is not.**

The examples now name `water: TIP3P` explicitly rather than relying on a default, so a reader sees
which coupled set they get.

### Results

```console
$ pytest tests/devtest/test_amber_like_workflow.py -q    36 passed
$ pytest -q                                             all passed
```

Both examples were regenerated and inspected rather than described: `config.yaml` carrying the
water model with both force-field XMLs, the contributor, the generator version and commit, and the
base seed; `paths.sh` free of machine paths; protocols free of concrete paths; all five launchers
sourcing `paths.sh`; every file flag present. The phenol route correctly records no protein force
field.

### Not done

No REST2 or AIS work, no OpenMMTools, no registration, no storage-format change, no component
management redesign.
