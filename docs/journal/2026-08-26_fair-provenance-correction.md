# FAIR provenance correction

Date: 2026-08-26
Branch: `dev` (merged `origin/main` first, per the instruction; `main` untouched)
Version: `0.4.0.dev0` (unchanged)

Correcting `d39b665`, the first FAIR implementation. Everything here is metadata, lineage and
retrofit. No scientific default, algorithm, scaling, exchange, seed or restart behaviour changed.

## The central defect

`forcefield.json` was written from the configuration the user *asked for*, not from what the
builder *loaded*. Those differ in ways that matter:

| recorded before | actually loaded | why it matters |
|---|---|---|
| `opc.xml` | `amber19/opc.xml` | the short file is water-only; the qualified one also carries the Na+/Cl- templates. The recorded name describes a file that could not have solvated this box |
| `amber19-all.xml` on the ligand route | nothing | the ligand route deliberately loads no protein force field. Recording ff19SB attributes parameters to a file that contributed none |
| `amber19-all.xml` on the implicit route | `leaprc.protein.ff19SB` via tleap | no OpenMM protein XML touched that System |

Both builders already reported correctly — `build_system(...)["forcefield"]` and the tleap/ParmEd
record. The fix is to consume that report. `forcefield_record.py` no longer re-derives anything the
builder knows; the resolved configuration supplies only what the builder does not report (box
shape, requested padding, human-facing labels), and those are labelled `requested_*`.

### The four routes, verified by running them

| route | protein | ligand | solvent | builder |
|---|---|---|---|---|
| explicit peptide | `amber19-all.xml` | null | `amber19/opc.xml`, PME | `ForceField.createSystem` |
| explicit ligand | **null** + note | `openff-2.2.0` (label `sage-2.2.0`), am1bcc | `amber19/opc.xml` | `ForceField.createSystem`, XML `['amber19/opc.xml']` |
| implicit peptide | openmm null, tleap `leaprc.protein.ff19SB` | null | GBn2/mbondi3, NoCutoff | `parmed.Structure.createSystem` |
| implicit ligand | **null** | `openff-2.2.0`, am1bcc | GBn2/mbondi3, NoCutoff | `parmed.Structure.createSystem` |

## Two pre-existing bugs the ligand routes exposed

The SMILES route had never been run end to end in this codebase. Producing routes 2 and 4 required
fixing both:

* `initial_structure()` read `structure.etkdg.seed`, which `sys-config` writes as null, and
  `int(None)` raised. RDKit's own default is a *random* embedding — two runs of one SMILES would
  give different starting coordinates, which is the one thing a prepared system must not do. It now
  falls back to `run.seed` and refuses if both are null.
* `_build_explicit` asked for `prepared["pdb"]` / `["sdf"]`; the keys are `solute_pdb` /
  `solute_sdf`. `KeyError` before parameterisation.

## Retained preparation artifacts

Flattened to basenames before, so two routes' `solute.sdf` would collide and one checksum would
describe the wrong file. Now the staging-relative subpath is preserved
(`preparation/structure/solute.sdf`), a genuine content collision at the same subpath is refused,
and an identical rerun is still listed — the artifact list no longer depends on whether the
directory happened to be fresh.

## Runtime input lineage

cMD and every REST2 replica now hash their starting artifact **before** this invocation can
overwrite it. On a resume the production checkpoint is rewritten at completion, so a hash taken
afterwards identifies what the run *produced*, not what it *consumed* — the opposite of lineage.

Demonstrated on a real resume: recorded `sha256` equals the pre-resume checkpoint hash and differs
from the post-resume one.

```
#0  0->40   parent_final_state          8d3e3509d4314de4
#1 40->60   own_production_checkpoint   0ef0605f43717b25   == pre-overwrite, != post
```

## Stage-log ordering

`stage.log` was written *after* its own size and hash were read, so `outputs["stage.log"]`
described the previous run's log on a resume and nothing at all on a first run. The order is now
outputs → log → inspect → record. Verified: recorded bytes equal the finished file for all four
common stages.

## Retrofit corrections

* `common_root` was an absolute path in `file-inventory.yaml`; the manifest now describes its root
  instead of baking in a path that stops being true once archived elsewhere.
* Exclusion used string prefixes, so `/x/out` would have excluded `/x/outputs`. Now
  `Path.is_relative_to()`.
* The sidecar prefix was hardcoded `fair-registration/`, breaking every path for any other
  `--output`. It is computed, and a test verifies every entry resolves for both
  `fair-registration` and `out`.
* A verified `--original-input` is now retained at `<output>/original_inputs/<name>` and covered by
  the manifest — a verified original that is not kept leaves the candidate un-rebuildable, which is
  the entire A/B distinction. A supplied `--environment` is retained too, and its absolute source
  path is no longer recorded.
* `source_modified: false` was an unconditional literal — a claim, not a check, in the one field a
  reader trusts. Replaced by real before/after snapshots over path, size, mtime_ns and SHA-256,
  reporting method, counts and changed/added/removed lists.

### Grading

Grade A now requires the retained verified original, an **exact** identity (a recorded commit or
installed fingerprint — a version string names a release, not a build), the environment, and the
runtime records the *configured* protocol requires. The protocol is read from `md.config.yaml`; if
it cannot be read that is itself a gap, never inferred from folder names.

The old fixture (`git_commit: null`, no cMD record) correctly drops to **B** with
`implementation_identity_not_exact` and `cmd_runtime_record_missing`. A new fixture with exact
identity and every required record earns **A**.

## Tests

Corrected two of my own tests that encoded the wrong expectations: one asserted `opc.xml`, the
other compared `implicit_solvent` by whole-dict equality against a record that gained an
`applied_by` field.

```
pytest tests/ -q -m "not gpu"    116 passed, 38 deselected
pytest tests/ -q -m "gpu"         38 passed, 116 deselected   (CUDA devices 1-4)
```

One tiny CUDA project exercised the runtime path: four common stages, a cMD fresh start and resume,
and a two-replica REST2. No 1 ns/5 ns protocol was rerun.

## Remaining limitations

* Routes 2 and 4 were exercised with ethanol, not a macrocycle; the record shape is verified, the
  chemistry is not a validation.
* `ligand.charge_model` and `toolkit_registry` stay null under AM1-BCC — the builder reports them
  only for NAGL.
* The retrofit reads 0.3.x output; a 0.4.x directory carries these records natively and does not
  need it.
* MD-data still owes the dataset ID, `$MD_DATA` storage, archival trajectory checksums and
  lifecycle.
