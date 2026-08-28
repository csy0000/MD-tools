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

## Phases 2-5 — what was built

`md-openmm setup`, two new modules, and nothing removed. `simple.py` turns a request into a
resolved preset and writes the directory; `emit.py` writes the scripts. **System building still
goes through `sysgen.generate_system` unchanged**, so force fields, solvation and AM1-BCC are the
validated code paths this repository already had. This milestone changes the interface and the
generated scripts, not the science.

### The measurement that matters

| | old cMD project | `setup` |
|---|---|---|
| Python a user must trust | **2,084 lines** | **251 lines** |
| production script | 286 lines | 60 |
| where `Simulation` is built | `md_stages.py:192`, not in the script | in the script |
| `md_templates` import | yes | **no** |
| YAML read at run time | yes | **no** |

Per script, against the milestone's targets:

```text
min/min.py                 39 / 40
eq/nvt_1kcal/nvt_1kcal.py  57 / 60
eq/npt_1kcal/npt_1kcal.py  60 / 60
eq/npt_free/npt_free.py    51 / 60
cMD/cmd.py                 60 / 60
```

No script exceeds its target, so nothing needs justifying under Phase 4.

### Two design decisions

**The `.out` header is one f-string, not a helper module.** The milestone permits a small shared
utility for mechanical output formatting. Using one would have saved about a dozen lines per script
and cost the reader an indirection to answer "what does this header contain". The f-string form
means the text in the script reads exactly like the text in the file it produces.

**Restrained stages take the solute as a literal `range`.** That removed the only YAML read at run
time. The range is resolved at generation from `solute.yaml`, which records
`solute_atom_indices_are_contiguous`; if it is ever false, generation refuses rather than emitting
a range that would restrain the wrong atoms.

### Three bugs the first real runs found

None was visible from reading the generator; each needed the script to be executed.

1. **The first equilibration stage looked one directory too high.** `eq/nvt_1kcal/` reaches
   minimisation through `../../min`, not `../min`. Immediate `FileNotFoundError`.
2. **`run.sh` aborted silently.** `set -e` fired on the failing subshell before the diagnostic
   could run, so the operator saw a stopped script and no reason. It now reports the stage by name
   with the tail of its `.out`.
3. **The step counter climbed across stages**, so a 10,000-step production printed steps
   16,000-25,000 and a "time remaining" of 23:58:26. Each stage now resets its own clock; lineage
   is stated by `input_state` in the header.

### Detachment, tested rather than asserted

A stub on `PYTHONPATH` whose `md_templates` raises `ImportError`, then the full chain:

```console
$ python -c "import md_templates"
ImportError: md_templates is deliberately unavailable

$ cd $MD_DATA/.../ALA && ./run.sh
== min ==  == nvt_1kcal ==  == npt_1kcal ==  == npt_free ==  == cmd ==  == done ==
```

| run | platform | wall clock | stages completed |
|---|---|---|---|
| ALA, peptide route | CPU | 3m26s | 5/5 |
| phenol/IPH, ligand route | CPU | 3m32s | 5/5 |
| ALA, **CUDA smoke** | CUDA | **13.3s** | 5/5, every `.out` says `platform: CUDA` |

### Safety, all enforced at generation

```text
output root inside a Git worktree   refused, naming the toplevel
non-empty system directory          refused, unless --overwrite
4 fs without HMR                    refused; accepted with hydrogen_mass_amu: 4.0
0.003 ps interval at 2 fs           refused: "1.5 steps ... 0.004 ps would"
unknown `advanced:` key             refused rather than silently ignored
```

### Tests

```console
$ pytest tests/test_simple_setup.py -q     32 passed
$ pytest -m "not gpu" -q                  386 passed, 77 deselected     154.3s
$ pytest -m gpu -q                         77 passed, 386 deselected    182.2s
```

Non-GPU rose from 354 to 386: the 32 new tests. **REST2, AIS, continuation-safety, NAGL and
charge-provenance coverage is unchanged and green.** The suite asserts the scientific defaults, so
a change to force fields or the equilibration schedule fails here rather than reaching a generated
system.

### Not done in this milestone, deliberately

REST2 and AIS keep their existing generation and their longer launchers. They can adopt the stage
file convention later, once the cMD design has been reviewed. Nothing in this branch modifies them.

---

## The file interface pass

Turned the short generated scripts into reusable **protocol** files whose paths arrive from
generated launchers through an Amber-like command line. Four pieces, deliberately separate:

| piece | holds |
|---|---|
| `<stage>.py` | the science: integrator, ensemble, restraint, steps, reporters, `simulation.step()` |
| `paths.sh` | the directory map, resolved from `$MD_DATA`; no machine path |
| `<stage>.sh` | every concrete input and output, passed explicitly |
| `bin/openmm-md` | generic file handling; no scientific opinion at all |

### `openmm-md`

156 lines, standard library and OpenMM only -- asserted on **parsed imports**, not on a substring
search, because the docstring says the word `md_templates` while explaining that it does not use it,
and a test that cannot tell prose from a dependency would force the explanation out of the file.

```text
-i --input  -p --topology  -s --system  -c --coordinates
-o --output  -x --trajectory  -r --restart  --checkpoint  --solute-x  --force
```

Precedence is flag, then `OPENMM_*`, then a clear error. Everything is validated before any Context
exists or any byte is written: required paths resolved, inputs verified present, outputs proved
distinct from inputs and from each other, existing outputs refused unless `--force`. Output
directories are created only after validation passes -- a test asserts that an invalid invocation
leaves no directory behind.

**It never writes the completion marker.** The protocol prints it, and only after the outputs it
names exist. `openmm-md` additionally refuses a marker standing over missing outputs, which would
be a completed run with nothing to show for it.

### Line counts, measured rather than claimed

The instruction asks for the measured count and a justification of the unavoidable increase.

| file | total | `.out` header | protocol body | old target |
|---|---|---|---|---|
| `min.py` | 44 | 16 | **28** | 40 |
| `nvt_1kcal.py` | 65 | 23 | **42** | 60 |
| `npt_1kcal.py` | 72 | 25 | **47** | 60 |
| `npt_free.py` | 62 | 25 | **37** | 60 |
| `cmd.py` | 69 | 26 | **43** | 60 |

The growth is the header, and the header is what the output contract in this same instruction
requires: resolved topology, System, starting state, trajectory, restart and checkpoint paths, plus
explicit integrator and barostat seeds, plus the provenance reference. That is roughly ten lines per
stage of output formatting, not protocol.

Excluding it, the bodies are 28-47 lines -- within about two of the previous figures, and those two
are `def run(files):` and the explicit `setRandomNumberSeed`. Counting mandated output formatting
against a limit on scientific code would push the science into a helper, which is the one outcome
the design forbids. The test measures the header and excludes it, and says why.

One piece of real fat was found and removed: `pressure_bar: n/a` and `barostat_seed: n/a` printed on
NVT stages. A field that does not apply is not printed.

### Corrections in this pass

| | |
|---|---|
| contributor | request field → `$MD_CONTRIBUTOR` → prompt → error. Never invented |
| package version | recorded, so `md_templates_commit: null` is acceptable only beside a concrete version |
| seeds | `advanced.common.random_seed` honoured rather than accepted and dropped; derived per stage, written as literals, printed in preset and `.out` |
| confirmation | `--yes` is what makes a run unattended; both modes otherwise show the full preset and ask |
| question count | the interactive mode asks **nine** questions, not the seven the README claimed |
| OPC | documented as the **coupled** ff19SB/OPC selection. The old one-field example would have paired ff14SB with OPC -- a combination nobody validated |
| output collisions | refused by `openmm-md` with `--force` to override, replacing the old "directory not empty" check alone |
| nested worktrees | `$REPO/a/b/c` with none of `a/b/c` existing was accepted, because Git was asked about a directory that did not exist. It now walks up to the nearest existing ancestor |
| `$MD_DATA` containment | the output must resolve beneath it, and the portable relative project path is derived from it |

### Evidence

Both routes, full chain, with a stub making `import md_templates` raise `ImportError`:

| run | platform | wall clock | stages |
|---|---|---|---|
| ALA, peptide | CPU | 2m27s | 5/5 `run_status: completed` |
| phenol/IPH, ligand | CPU | 2m10s | 5/5 `run_status: completed` |

```console
$ pytest tests/test_openmm_md_interface.py -q     21 passed
$ pytest tests/test_simple_setup.py -q            38 passed
$ pytest -m "not gpu" -q                         425 passed, 77 deselected   238.1s
$ pytest -m gpu -q                                77 passed, 425 deselected  182.0s
```

Non-GPU rose from 387 to 425. REST2, AIS, continuation-safety, NAGL and charge-provenance coverage
is unchanged and green; no REST2 or AIS file was modified in this pass.

`paths.sh` portability and protocol reuse are both tested by execution rather than by inspection:
the managed root is copied elsewhere, `$MD_DATA` re-exported, and the same launcher runs untouched;
and one `min.py` is run against a second, unrelated set of paths through `openmm-md` directly.

### Not done here

REST2 and AIS keep their existing generation and their `sys-gen`/`md-gen` path. OpenMMTools is not
adopted, registration is not implemented, trajectory format is unchanged, and component management
is untouched. Those are the next task's, not this one's.
