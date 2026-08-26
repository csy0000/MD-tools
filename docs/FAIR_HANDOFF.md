# FAIR handoff: what MD-templates provides, and what it does not

MD-templates makes simulation data **FAIR-ready**. It does not make it FAIR. The difference is not
pedantry — it decides who is responsible when someone cannot find or reuse a dataset a year later.

## The boundary

| repository | owns |
|---|---|
| [MD-templates](https://github.com/csy0000/MD-templates) | system construction, the force-field record, the resolved protocol, seeds, generated scripts, execution provenance |
| [MD-data](https://github.com/csy0000/MD-data) | the permanent dataset ID, immutable storage under `$MD_DATA`, the complete archive checksum manifest, searchable metadata, locations, retention, replacement, access lifecycle |
| [MD-analysis](https://github.com/csy0000/MD-analysis) | analysis configuration, software identity, the dataset IDs and checksums consumed, derived-result lineage |
| project-template *(not yet available)* | project-specific inputs, configs and workflows, and exact locks for the three components plus the registered datasets |

A directory produced by `md-openmm` is **a registration candidate**, never a FAIR dataset by
itself. It carries no DOI, no permanent identifier, no access policy and no archival checksum over
its trajectories. Saying otherwise would promise something no code in this repository delivers.

## Against the four letters

**Findable** and **Accessible** are completed by MD-data, not here. MD-templates contributes the
content that makes a record worth finding — what the molecule was, how it was parameterised, what
ran — but identity and access are assigned at registration.

**Interoperable** is what this repository can deliver alone: documented YAML and JSON records with
explicit units, relative paths, and force-field resources named by their exact OpenMM resource
rather than a family label. `amber19-all.xml` is a file; "Amber19" is a conversation.

**Reusable** is the point of the provenance:

- `inputs/original_inputs/` — the user's own file, byte-for-byte;
- `inputs/forcefield.json` — how the System was parameterised, recorded where it was decided;
- `inputs/resolved_sys.config.yaml`, `MD/md.config.yaml` — what was asked for, after resolution;
- `implementation` in both provenance files — version, git commit when available, and an installed
  fingerprint that is never null;
- seeds for every stage and replica;
- the exact command as an argument list;
- `SHA256SUMS` and `generated-files.sha256` — deterministic, reproducible on demand.

## `unknown` is an answer

Every retrospective value produced by `scripts/retrofit_fair_v030.py` carries an evidence status:
`recorded`, `derived`, `user_supplied` or `unknown`. There is deliberately no `inferred`. A guessed
force field or a reconstructed charge method, written in the same field and the same font as a
recorded one, is worse than an acknowledged gap: the gap can be investigated, the guess gets cited.

Legacy 0.3.x output is classified **A** (rebuildable from the original input), **B** (prepared
system reusable, parameterisation not fully rebuildable) or **C** (archival and analysis only), and
every reason that lowered the grade is machine-readable. A grade is never raised because a filename
looks right.

## What MD-data still has to do

For any directory this repository produces:

1. assign the permanent dataset ID and landing page;
2. move it into `$MD_DATA` under an immutable layout;
3. compute the complete archival checksum manifest **including trajectories** — runtime records
   deliberately carry only paths, byte sizes and frame counts, because trajectories grow across
   invocations and hashing them on every invocation would cost more than it proves;
4. record retention, replacement and access policy.

Until those exist, the data is reproducible but not yet findable.
