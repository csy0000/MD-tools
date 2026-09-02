# MD-tools v0.5: method documentation, scientific options, reusable APIs, and compact Python executables

Status: active, temporary execution instruction  
Target branch: `dev`  
Reviewed starting point: `2347690a3c59d33fbd8a6db17403f891576a9a6e`

This task changes both documentation and implementation. Work in staged, reviewable commits. Do not
merge to `main`, publish a package, create the final `v0.5.0` tag, or change
`0.5.0.dev0`.

## Outcome

MD-tools must retain one installed executable, `md-openmm`, with exactly:

```text
build-top
build-md
data-register
```

Do not add a `md-run` console script or a `run` subcommand. The generated Python files are the
executables and must still run as:

```bash
python ./md_script/min.py -p built.pdb -s built.xml -r min.xml -x min.dcd -log min.log
```

The generated files must become compact entry points. All common simulation, restraint, reporting,
REST2, REMD, rREST2 and AIS behavior must live in stable importable APIs inside the installed
`md_tools` package.

The Python distribution name is `MD-tools`; the import package is always lowercase
`md_tools`. Never generate `from MD-tools ...`.

---

## 1. Use the academic-research skill for scientific documentation

Claude has the skill `academic-research-skills`. Read its complete `SKILL.md` before changing
`docs/scientific-defaults.md` or `docs/scientific-defaults.bib`, then follow it.

Use primary literature and official OpenMM/OpenFF/openmmforcefields documentation where possible.
Do not invent references. Every scientific claim must have an inline citation and a matching,
complete bibliography entry. Include authors, title, journal or documentation owner, year, volume,
pages/article number, DOI or stable URL as applicable. Distinguish clearly between:

- software support;
- conventional practice;
- evidence from published validation;
- an MD-tools policy/default.

Do not turn the document into a development journal.

---

## 2. Make the root README a concise entry point

Rewrite `README.md` so it contains only:

1. what MD-tools is;
2. supported Python/OpenMM environment and installation;
3. the three-command interface;
4. one short end-to-end example: `build-top` → `build-md` → run the generated Python files →
   `data-register`;
5. a compact documentation index;
6. the unreleased/development status.

Move detailed method, force-field, replica-exchange, configuration, registration and recovery
explanations into `docs/`. Do not duplicate long passages between README and method pages.

---

## 3. Add method-focused documentation

Create:

```text
docs/openmm_methods/
├── README.md
├── cMD/
│   ├── README.md
│   └── example.config
├── REST2/
│   ├── README.md
│   └── example.config
├── rREST2/
│   ├── README.md
│   └── example.config
└── AIS/
    ├── README.md
    └── example.config
```

Each method README must explain:

- the ensemble and scientific purpose;
- what MD-tools implements and does not implement;
- required inputs;
- the minimal command sequence;
- generated files;
- restart/continuation behavior;
- important configuration fields;
- reporting and data-registration consequences;
- method-specific limitations;
- links to scientific defaults and the relevant primary references.

The four `example.config` files are minimal runnable examples, intentionally different in role
from the comprehensive root examples under `configs/md/`. They must not be byte-for-byte mirrors.
Validate them through the same real strict resolver in tests and CI.

Consolidate or remove older documents whose content is fully superseded. In particular, migrate
current material from `docs/replica-exchange.md` into the REST2/rREST2 pages and a shared method
index where appropriate. Do not leave two current authorities.

Use the exact directory name `openmm_methods` requested above.

---

## 4. Add data-registration documentation

Create:

```text
docs/data_register/
├── README.md
├── user.config.example
├── dataset.yaml.example
└── extension.yaml.example
```

The README must cover:

- `md-openmm data-register --init`;
- where user configuration is stored;
- `$MD_DATA/{year}/{project_name}/{data_name}/`;
- the `common/` variant;
- local ignored `data/` generation followed by registration;
- dry-run, verification, noninteractive operation and recovery from interruption;
- transactional move/symlink behavior;
- dataset v2 and extension v2;
- immutability and extension rules;
- concrete commands using the three example files.

These are documentation examples, not new schemas. Validate them against the actual user-config,
dataset and extension models. Keep `docs/data-contract.md` only as the authoritative schema-level
reference and link to it rather than duplicating it.

---

## 5. Scientific defaults and supported alternatives

Update the implementation, root comprehensive examples, method examples, support matrix,
scientific-defaults documentation and tests consistently.

### 5.1 Box shape

- Default remains rhombic dodecahedron with the current default padding.
- Add `cube` as a documented and tested alternative.
- If the current OpenMM version already supports `octahedron`, document it accurately as
  supported rather than pretending only two OpenMM shapes exist.
- Unknown box shapes must be refused before building.
- Record requested and realized box geometry in `built.log`.

### 5.2 Barostat

- Keep OpenMM `MonteCarloBarostat` as the supported isotropic production barostat and default.
- Discuss Berendsen pressure coupling formally because the user requested it, but do not expose a
  fake or unvalidated `BerendsenBarostat` option: OpenMM does not provide that standard class.
- Explain, with primary citations, that Berendsen weak coupling is useful historically or for
  relaxation but does not reproduce correct NPT volume fluctuations.
- If no independently validated implementation exists in this repository, label Berendsen
  `documented, not implemented`.
- Do not write a custom Berendsen implementation in this task.
- Implicit solvent remains boxless and barostat-free.

### 5.3 Ligand force fields

- Default remains Sage 2.2.1.
- Add a GAFF2-family alternative through `openmmforcefields.generators.GAFFTemplateGenerator`.
- Do not persist an ambiguous label such as only `GAFF2`. Resolve it to an explicit installed
  version such as `gaff-2.11` or `gaff-2.2.20`, chosen from
  `GAFFTemplateGenerator.INSTALLED_FORCEFIELDS`.
- The user-facing config may accept a documented alias only if the resolved exact version is
  recorded in `built.log`, the force-field record and the machine record.
- Record the charge method, AmberTools/Antechamber versions and cache provenance.
- Refuse an unavailable requested GAFF version with the installed supported list.
- Test the Sage and GAFF routes with a small ligand.

### 5.4 Hydrogen mass repartitioning

Replace the ambiguous public convention `hydrogen_mass_amu: null means off` with an explicit
human-readable block:

```yaml
hydrogen_mass_repartitioning:
  enabled: false
  hydrogen_mass_amu: 3.024
```

Requirements:

- default `enabled: false`;
- `enabled: true` is the documented alternative;
- conserve total mass;
- do not repartition water hydrogens;
- require compatible hydrogen-bond constraints;
- record enabled/disabled status, target mass, affected atoms and conservation evidence;
- keep a migration-quality error for obsolete or contradictory keys rather than silently ignoring
  them.

### 5.5 HMR-dependent timestep

Make the default MD value:

```yaml
dynamics:
  timestep_fs: auto
```

`build-md` does not see the serialized System, so it must not guess whether HMR was actually
applied. Resolve `auto` when the generated Python executable reads `built.xml`:

- ordinary hydrogen masses → 2 fs;
- verified HMR masses → 4 fs;
- explicit 4 fs without HMR → refuse before integration;
- explicit 2 fs with HMR → allow;
- explicit values above the safe non-HMR threshold without HMR → refuse.

Use the masses serialized in the System as the source of truth, not a repeated configuration claim.
Resolve the same way for cMD, REST2, rREST2 and AIS. Record the final numerical timestep and its
basis in every log and machine record. Step counts remain authoritative; physical times are derived
only after the numerical timestep is resolved.

---

## 6. Convert crossed explicit protein/water pairs from errors to warnings

The current resolver hard-refuses crossed explicit pairs. Change only this policy:

- ff14SB + TIP3P: recommended default, no warning;
- ff19SB + OPC: supported alternative, no warning;
- ff14SB + OPC: allow with a prominent warning;
- ff19SB + TIP3P: allow with a prominent warning.

The warning applies to the protein/water pairing, not the ligand force field:

- Sage + TIP3P or OPC: no ligand-related warning;
- GAFF2 + TIP3P or OPC: no ligand-related warning.

Emit warnings on stderr and store structured warnings in `built.log` and its machine record. Do
not silently continue.

Retain hard errors for incoherent physics or unsupported construction, including explicit-solvent
keys with GBn2 and ff19SB with the current GBn2 model unless new validation evidence explicitly
supports it.

---

## 7. Create coherent reusable APIs

Before moving code, build an import/call map for `openmm/templates/`, `runtime/`, `build/` and
their tests. Classify every module and public symbol as active, compatibility-only, obsolete or
uncertain. Resolve uncertainty from callers and tests. Never delete by filename alone.

Create stable APIs with these responsibilities. Exact internal file splitting may be adjusted when
the import graph proves a better boundary, but do not change the public concepts silently.

### 7.1 Ordinary MD

Expose from `md_tools.md`:

```python
PositionalRestraint
ReportingConfig
run_stage
run_generated_stage
run_generated_workflow
```

Centralize:

- adding a positional restraint;
- changing and reading its strength;
- selection of restrained atoms;
- integrator and Simulation construction;
- barostat creation and frequency;
- platform/device selection shared with REMD where possible;
- stage restart and checkpoint fingerprinting.

Generated scripts must not construct `CustomExternalForce`, integrators, barostats or reporters
directly.

### 7.2 Common reporting and provenance

Provide a coherent reporting interface that owns:

- readable logs;
- MD-data-compatible machine records;
- state/energy tables;
- solute and whole-system trajectories;
- phase-space output;
- checkpoints;
- restart truncation/repair;
- file hashes and inventories.

A single public `ReportingConfig` may compose specialized internal writers. Do not force ordinary
DCD/state reporting, REMD NetCDF and Amber `rem.log` into one giant implementation merely to
reduce the file count.

Consolidate duplicated seed derivation, hashing, timestamp, file-record and platform-selection
helpers. There must be one authoritative implementation of each low-level operation.

### 7.3 REST2 scaling

Expose from `md_tools.rest2`:

```python
REST2Scaler
ScalingSelection
```

The scaler is independent of REMD because fixed-tau cMD, REST2, rREST2 and AIS all use it. Move the
active behavior from `openmm/templates/rest2_scaling.py`; do not fork it.

`ScalingSelection` must accept or write a structured `solute.yaml` containing:

- schema version;
- topology digest;
- solute atom indices;
- unscaled torsion central bonds;
- readable residue/atom labels and reasons when available.

Use central bonds for omega exclusion so every torsion term sharing a peptide C–N bond is excluded.
Validate topology digest, atom ranges, bond existence, duplicates and solute membership. Refuse an
exclusion that matches no torsion rather than silently ignoring it. If no file is supplied, derive
the selection using the current validated classifier and write the resolved file.

Record the exact resolved exclusions. `tau` remains the only public/persisted scaling coordinate.

### 7.4 REMD and rREST2

Expose from `md_tools.remd`:

```python
REMDRunner
NeighborExchangeRule
run_remd
run_generated_remd
```

Expose the reservoir behavior from `md_tools.remd.reservoir`:

```python
ReservoirRefreshRule
```

Separate and consolidate:

- exchange scheduling and Metropolis acceptance;
- replica propagation and coordination;
- single-process/MPI device placement;
- state/walker mapping;
- checkpoints and continuation;
- state trajectories, NetCDF analysis and `rem.log`;
- reservoir selection, velocity policy and provenance.

rREST2 must compose `REST2Scaler + REMDRunner + NeighborExchangeRule +
ReservoirRefreshRule`. It must not carry a copied REMD driver. Future reservoir REMD development
must be able to reuse the generic reservoir rule.

Move exchange mathematics out of REST2 scaling if it is duplicated in the replica engine. Keep one
tested implementation.

### 7.5 AIS

Expose `run_generated_ais` from `md_tools.ais`. AIS must import the same `REST2Scaler` and
`ScalingSelection`; it must not retain a second scaling implementation.

Preserve the existing AIS work convention, observation-zero rule, fixed volume, exact step
schedule, independent seeds and restart/completion behavior.

### 7.6 Retire the misleading templates runtime layout

`src/md_tools/openmm/templates/` currently contains installed runtime code, not templates. Move
active code to the packages above and remove the directory when no active importer remains.

Small compatibility facades under `md_tools.runtime` are allowed only to keep already-generated
pre-v0.5 scripts executable. They must re-export/call the new implementation, contain no copied
scientific logic, be documented as compatibility-only and be tested. Do not retain parallel
authorities.

Aim for a small number of cohesive modules, not one huge `REMD.py`. File-count reduction is
secondary to one clear owner per responsibility.

---

## 8. Make every generated Python executable compact

Keep `resolved.config` as the single resolved workflow declaration beside the generated scripts.
It must be strictly validated at execution and included in checkpoint/config fingerprints.

Generate stage files equivalent to:

```python
#!/usr/bin/env python
from md_tools.md import run_generated_stage
raise SystemExit(run_generated_stage(__file__, "min"))
```

Generate all-in-one `md.py` equivalent to:

```python
#!/usr/bin/env python
from md_tools.md import run_generated_workflow
raise SystemExit(run_generated_workflow(__file__))
```

Generate REST2/rREST2 files equivalent to:

```python
#!/usr/bin/env python
from md_tools.remd import run_generated_remd
raise SystemExit(run_generated_remd(__file__, protocol="REST2"))
```

Generate AIS equivalently through `md_tools.ais.run_generated_ais`.

The helpers use `__file__` to locate sibling `resolved.config` and any resolved selection file.
They must not rely on the current working directory and must contain no checkout or machine path.

The familiar commands and flags remain valid:

```bash
python min.py -p built.pdb -s built.xml -r min.xml -x min.dcd -log min.log
python REST2.py -p built.pdb -s built.xml -c eq_npt_free.xml -log REST2.log
```

Acceptance checks for generated executables:

- no function or class definitions;
- no `argparse`;
- no direct OpenMM imports;
- no direct reporter, restraint, barostat, scaling or exchange construction;
- only the stable API import and invocation plus comments/docstring if genuinely useful;
- all current restart, `--check`, platform/device and output flags still work;
- moving the whole generated directory does not break it;
- changing `resolved.config` after checkpoint creation causes a fingerprint refusal.

Update `run.sh` to invoke these compact Python files without duplicating runtime logic.

---

## 9. Tests and scientific invariants

Do not delete a failing test to make the migration pass. Classify and migrate it.

Add focused tests for:

- documented minimal configs and registration examples;
- cube and dodecahedron construction;
- supported/unsupported box shapes;
- Sage and explicit-version GAFF ligand construction;
- exact GAFF provenance;
- crossed protein/water warnings and their log records;
- explicit/implicit invalid combinations remaining errors;
- HMR false/true construction and mass conservation;
- `timestep_fs: auto` resolving to 2/4 fs from serialized masses;
- explicit 4 fs without HMR refusing before any integration;
- all compact generated executables and unchanged CLI flags;
- public imports from `md_tools.md`, `md_tools.rest2`, `md_tools.remd` and
  `md_tools.remd.reservoir`;
- omega exclusion loaded from a valid file;
- digest mismatch, nonexistent bond and unmatched torsion refusals;
- shared REST2 scaler identity across fixed-tau cMD, REST2, rREST2 and AIS;
- neighboring exchange acceptance, no velocity rescaling, state trajectories and `rem.log`;
- reservoir refresh identity and velocity provenance;
- checkpoint/continuation and extension behavior;
- no importer of `openmm/templates/` after migration;
- wheel contents and operation outside the checkout.

Preserve all current scientific invariants in `CLAUDE.md`.

Run:

```bash
python -m pytest tests -m "not slow and not gpu" -q -rs
python -m pytest tests -m "gpu or slow" -q -rs
python -m build
```

No selected test may skip. GPU/slow tests must genuinely use CUDA, never CPU substitution.

Install the wheel in a clean environment and test outside the checkout:

- the three public commands;
- all root and documentation config examples;
- all registration examples;
- cMD split and all-in-one;
- cMD implicit;
- REST2;
- rREST2;
- AIS;
- compact-script structure;
- public imports;
- both schemas;
- absence of obsolete runtime/template modules from the wheel.

CI must build/install the wheel, test outside the checkout, validate the examples and public API,
compile every generated script, assert compactness, and fail on non-GPU skips. State plainly that CI
does not replace local CUDA scientific validation.

---

## 10. Documentation and completion

Update:

- `README.md`;
- `docs/README.md`;
- method and registration pages;
- `docs/scientific-defaults.md` and bibliography;
- `docs/support-matrix.md`;
- root comprehensive configs;
- `CLAUDE.md`;
- `CHANGELOG.md`;
- `docs/release-notes/v0.5.0.md`.

Document the new import API and make clear that generated Python files are executable entry points,
not copies of the implementation.

Use multiple meaningful commits. Push to `dev`, wait for the final implementation/documentation CI
run and require green.

Then delete this temporary `CLAUDE_TASK.md`, commit and push the deletion, and confirm the
current-head CI run is also green. Do not merge or tag.

## Completion report

Return:

1. final `dev` SHA;
2. commits and their responsibilities;
3. final documentation tree;
4. old-to-new module/symbol map;
5. deleted modules and evidence they had no remaining importer;
6. exact public import examples;
7. example compact generated scripts;
8. box, GAFF, force-field warning and HMR/timestep behavior;
9. exact fast and CUDA/slow test counts with zero skips;
10. wheel/outside-checkout results;
11. both final CI URLs and conclusions;
12. confirmation that `CLAUDE_TASK.md` was deleted;
13. confirmation that `main`, publication, version and final tags were untouched.
