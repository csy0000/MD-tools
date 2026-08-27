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

### Where the boundary actually runs

It is tempting to state it as "MD-templates never writes into `$MD_DATA`". That is wrong, and the
wrong version is the one that causes trouble, because a run's outputs have to land where the run
can read its own parent stage.

The accurate version:

- MD-templates **may** write generated simulation files and a contract-valid `dataset.yaml` into
  the single active dataset directory the user explicitly selected with `MD_DATA_LOCAL`.
- It **never** writes anywhere else under `$MD_DATA`, never walks or hashes the archive, never
  touches a second dataset, and never writes into one that is already `complete` or `archived`.
- MD-data owns the schema, the validator, identity, the `active -> complete -> archived` lifecycle,
  the catalogue, aliases, extensions, archival checksums and retention. MD-templates *imports*
  MD-data's validator rather than agreeing with it by hand: the failure mode being avoided is two
  repositories that each believe they implement the same contract and slowly stop doing so.

A dataset that MD-templates has written is still a registration candidate. Passing the contract
validator means the directory is *shaped* correctly; it does not mint an identifier, and the
`dataset_id` in the manifest is one the user obtained from MD-data, not one generated here.

## Against the four letters

**Findable** and **Accessible** are completed by MD-data, not here. MD-templates contributes the
content that makes a record worth finding — what the molecule was, how it was parameterised, what
ran — but identity and access are assigned at registration.

**Interoperable** is what this repository can deliver alone: documented YAML and JSON records with
explicit units, relative paths, and force-field resources named by their exact OpenMM resource
rather than a family label. `amber19-all.xml` is a file; "Amber19" is a conversation.

**Reusable** is the point of the provenance:

- `inputs/original_inputs/` — the user's own file, byte-for-byte;
- `inputs/forcefield.json` — how the System was parameterised, from the **builder's own report of
  what it loaded**, not from the configuration that was requested. `openmm_resource` is the exact
  resource (`amber19/opc.xml`, not the `opc.xml` label); `requested_label` keeps the human-facing
  name separately; `null` means the concept does not apply on this route — a ligand-only run loads
  no protein force field, and an implicit run loads no OpenMM protein XML at all;
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

Grade **A** requires the verified original input *retained in the candidate*, an **exact**
implementation identity — a recorded git commit or installed fingerprint, since a version string
names a release rather than a build — the build environment, and the runtime records the
*configured* protocol requires. The protocol is read from `md.config.yaml`; if it cannot be read,
that is itself a gap and is never inferred from directory names.

0.3.x did record the original input's SHA-256 in `inputs/provenance.yaml` when one was available,
but it generally did not retain the original bytes, and recorded neither the environment nor an
exact code identity. Most legacy data therefore lands at **B**, correctly.

## What MD-data still has to do

For any directory this repository produces:

1. assign the permanent dataset ID and landing page;
2. move it into `$MD_DATA` under an immutable layout;
3. compute the complete archival checksum manifest **including trajectories** — runtime records
   deliberately carry only paths, byte sizes and frame counts, because trajectories grow across
   invocations and hashing them on every invocation would cost more than it proves;
4. record retention, replacement and access policy.

Until those exist, the data is reproducible but not yet findable.
