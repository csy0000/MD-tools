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
