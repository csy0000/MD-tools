# Claude Code instruction: make the implicit `-hmr-v1` profiles real, and gate the built System

Date: 2026-08-25  
Repository: `csy0000/MD-templates`  
Branch: `dev`  
Expected starting commit: `a8901ba7086241fd59031f8c3470e8187a878cbc` (tag `openmm-v0.1.0`) plus the
**uncommitted conservative-defaults work already in the worktree**, or a direct descendant of it

## Context: what the in-progress work already got right

The worktree contains an unfinished change that reverts every base profile to a conservative
`2 fs` with unmodified hydrogen masses, adds eight opt-in `…-hmr-v1` performance profiles at
`3.024 amu` / `4 fs`, sets `config.py`'s module defaults to `hydrogen_mass_amu: None` /
`hmr_scope: "none"`, adds an agreement check in `system.py` between `hmr_scope` and
`hydrogen_mass_amu`, and adds `tests/test_conservative_defaults.py`.

**That design is correct and this instruction does not undo any of it.** It removes the unstable
combination from the default path, which is what mattered most: before it, `openmm-v0.1.0` shipped
implicit profiles declaring 4 fs against a builder that returns 1.008 amu hydrogens.

Finish it. As it stands the opt-in half does not work for implicit solvent.

## The defect that remains, measured

`implicit-md-peptide-hmr-v1` declares `hydrogen_mass: 3.024 amu`, `hmr_scope: solute`,
`timestep: 4 fs`. Generating a project against it produces:

```
eq        5,000 steps   (20 ps, restrained)
cMD_1 1,250,000 steps   (5 ns)      ->  5 ns / 1,250,000 = 4 fs
```

on a System whose masses, read back from `system.xml`, are:

```
mass histogram   {1.008: 12, 12.01: 6, 16.0: 2, 14.01: 2}
total mass       144.176 amu        n constraints 12
```

So the implicit performance profiles are **4 fs on unrepartitioned hydrogens with `HBonds`
alone** — the exact combination the conservative revert exists to prevent, still reachable, now by
opting in. All four implicit `…-hmr-v1` profiles are affected.

### Why the opt-in cannot reach the builder

The implicit System is built by `MD_system_gen.py` from `system_config.json`. That file has no
profile and no timestep; the profile that declares the hydrogen mass is resolved later, by
`MD_input_gen.py`. **No implicit profile can change the System**, because the System is built
before any profile is read.

`system_config.json` *can* carry a `system_build` section — `system_prep.py:117` records it as
`"user input"` — so the explicit route has a path to HMR. The implicit branch then discards it:

```python
# system_prep.py, implicit branch, ~line 445
cfg["system_build"].update({
    ...
    "hydrogen_mass_amu": None,
    "hmr_scope": "none",
})
```

unconditionally, after the user's value was applied. And `implicit.py:build_implicit_system` calls
`structure.createSystem(...)` with no `hydrogenMass=` argument at all, while `implicit.py:297`
hard-codes `"hmr": {"scope": "none", "target_hydrogen_mass_amu": None}` into the provenance.

### Why the new guard does not catch it

`tests/test_conservative_defaults.py:136 test_4_fs_is_never_paired_with_unrepartitioned_hydrogens_anywhere`
iterates `_profiles()` and compares each profile's `timestep` against its `hydrogen_mass`. Every
`…-hmr-v1` profile states both, so it passes. Nothing compares a profile against the **System that
profile produces**, and that is exactly where the two disagree.

`tests/test_implicit_gbn2.py:199 test_the_implicit_system_is_not_hydrogen_mass_repartitioned`
still asserts `max(light) < 1.5` on a built bundle, and still passes.

## Outcome and non-negotiable completion rule

Selecting an implicit `…-hmr-v1` profile must produce a System with 3.024 amu hydrogens, or must
fail. The repository must gain a guard that compares a **built System** against the configuration
that asked for it.

Maintain the acceptance matrix below. You may write `PASS` only when every row is implemented,
tested and verified. Do not call a requirement "deferred", "follow-up" or "good enough" and still
return `PASS`. If a genuinely external blocker remains, return `BLOCKED`, name the exact unmet
gate, and preserve all evidence. Never relabel an unrun check as passed.

Work on `dev`, focused commits, push `dev`. OpenMM only.

The worktree also contains `build_backend/md_templates_build.py` and
`src/md_templates/_build_info.py`. Preserve them; they are unrelated to this task.

## Read before editing

- `CLAUDE.md`, `README.md`, `docs/configuration.md`;
- `src/md_templates/openmm/implicit.py` — `build_implicit_system`, `build_implicit_bundle_inputs`,
  `implicit_provenance`;
- `src/md_templates/openmm/system_prep.py` — `_runtime_cfg_from_system_config` (note line 117,
  which admits a user `system_build` section) and the implicit branch that overwrites it;
- `src/md_templates/openmm/system.py` — `repartition_hydrogen_mass`,
  `verify_hydrogen_mass_repartitioning`, and how `build_system` applies HMR on the explicit path;
- `src/md_templates/openmm/config.py` module defaults;
- `src/md_templates/openmm/{cli_system_gen,input_gen}.py` for the two-generator split;
- every `…-hmr-v1` profile and `tests/test_conservative_defaults.py`;
- `tests/test_implicit_gbn2.py` in full;
- the `e187c85` commit message, whose measurement stands and must not be re-litigated: mass enters
  the kinetic term only, and repartitioning cyclo-(RGDfV)'s 38 hydrogens left the GBn2 potential at
  -634.488588216 kJ/mol with every force component unchanged, total mass conserved at
  574.632036 amu.

Record the baseline test result before changing anything. Do not commit trajectories, checkpoints,
States, environments, caches or large logs.

## Requirements

### 1. Decide how a performance profile reaches the System builder, and document it

This is the design question; decide it deliberately rather than routing around it.

HMR is build-defining, but under the two-generator split the System is built before any profile is
read. A `…-hmr-v1` profile therefore cannot influence it today, on either route — the explicit
route only works because a user can restate the mass in `system_config.json`, which means the same
fact is written in two places that can silently disagree.

Choose one and justify it in the journal:

- `system_config.json` gains a first-class hydrogen-mass declaration, and `MD_input_gen.py`
  **refuses** a profile whose `build.hydrogen_mass` disagrees with the bundle it was built against;
- or the bundle records "no HMR applied" and the repartitioning moves to where the profile is
  known, with the bundle's System treated as pre-HMR;
- or another route you can defend.

Whatever is chosen, the mismatch must be **refused with both values named**, never averaged,
ignored, or resolved by precedence. A bundle built without HMR that is then run at 4 fs is the
failure this whole instruction exists to close.

### 2. Apply the repartitioning on the implicit route

`build_implicit_system` must repartition when the resolved build asks for it, through ParmEd's
`createSystem(hydrogenMass=...)` or the existing `system.repartition_hydrogen_mass`. Do not add a
third implementation.

Under implicit solvent the solute is the whole system, so `hmr_scope: "solute"` and `"all"`
coincide; state that rather than letting scope look ignored. Preserve total mass and keep the guard
that refuses to drive a heavy atom below a sane mass.

Stop the implicit branch of `system_prep.py` from unconditionally overwriting `hydrogen_mass_amu`
and `hmr_scope`. Overwriting the water-only fields (`rigid_water`, cutoff, PME) is correct and
should stay; hydrogen mass is not a water-only field.

### 3. Make the provenance state what actually happened

`implicit.py:297` and the `forcefield.json` `hmr` record must report the real scope, real target
mass, and the number of hydrogens repartitioned. If the superseded note about the pinned reference
survives anywhere, delete it rather than editing around it — the measurement it appeals to was
overturned in `e187c85`.

### 4. Replace the test that pins the defect

`tests/test_implicit_gbn2.py:199` asserts hydrogens stay below 1.5 amu. Replace it with a pair:
a default-profile bundle must be unrepartitioned, and an `…-hmr-v1` bundle must carry the declared
mass with total mass conserved and the repartitioned count matching the topology. A replacement
that would also have passed under the old behaviour is not acceptable.

### 5. Preserve the GBn2 energy identity, and measure it

The ParmEd-vs-`AmberPrmtopFile` evidence (16.05 kJ/mol in `CustomGBForce`) and the tau-zero
identity must still hold. Repartitioning changes mass only, so potential energy and every force
component must be unchanged to **0.000e+00** on ACE-ALA-NME. Add a test that measures it and record
the numbers in the journal.

### 6. Add the guard that was missing: built System vs resolved configuration

Deserialize a built `system.xml` and compare it against the configuration that produced it:
hydrogen masses against `build.hydrogen_mass`, constraint count against `build.constraints`,
periodicity against the solvation mode, and `CustomGBForce` present for implicit. Run it for
**every shipped profile that can be built**, default and `…-hmr-v1`, both routes.

`test_4_fs_is_never_paired_with_unrepartitioned_hydrogens_anywhere` compares a profile against
itself and will never catch this class of defect. This guard is the one that would have.

### 7. Repair `prepare --config` for implicit documents, or refuse cleanly

`md-openmm prepare --config` raises `AttributeError: 'NoneType' object has no attribute
'water_model'` at `cli.py:536` for any implicit document: `_synthetic_system_manifest` reads
`spec.build.solvation` unconditionally. Support the implicit route there or refuse it with a
message naming the supported route. A traceback is not a refusal.

## Consequences to handle explicitly

- Applying HMR is **build-defining**: implicit bundle hashes change and existing bundles become
  invalid. Say so in the changelog and journal, and make a stale bundle fail loudly, not resume.
- `openmm-v0.1.0` ships the unstable default combination. Decide with the user whether the
  conservative revert plus this fix warrants `v0.1.1`. Do not move an existing tag.
- `software/md-stack/conda/openfftools860` holds a non-editable `md_templates` in site-packages
  with pre-`e187c85` profiles. It needs reinstalling after this lands or `md-openmm` from that env
  keeps serving stale defaults.

## Acceptance matrix

| # | requirement | evidence required |
|---|---|---|
| 1 | how a performance profile reaches the builder is decided, documented, and mismatches are refused | journal entry; the refusal message with both values |
| 2 | an implicit `…-hmr-v1` bundle carries 3.024 amu hydrogens | mass histogram from `system.xml`, peptide and ligand routes |
| 3 | provenance reports real scope, mass and count | `forcefield.json` / `system_simbox.json` excerpts |
| 4 | the no-HMR test is replaced by a default/performance pair | test names and assertions |
| 5 | GBn2 potential and forces unchanged to 0.000e+00 | measured numbers on ACE-ALA-NME |
| 6 | built-System-vs-configuration guard covers every buildable profile | test name and what it compares |
| 7 | `prepare --config` on an implicit document works or refuses with a message | command and output |

Plus: the full supported suite passes, and a 1 ns implicit cMD and a 1 ns implicit REST2 each run
to a committed boundary with no NaN — at 2 fs on a default profile and at 4 fs on `…-hmr-v1`.

## Report

Starting and final SHA; the acceptance matrix with `PASS`/`BLOCKED` per row; mass histograms before
and after for both routes; the GBn2 energy and force deltas; test counts; and any decision that
changes a documented default. Write a dated journal entry under `docs/journal/`.

Do not describe any of this as scientific validation. It is a correctness fix and its evidence is
operational.
