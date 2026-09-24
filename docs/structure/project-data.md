# The dataset contract, and where the FAIR boundary runs

MD-tools makes simulation data **FAIR-ready**. It does not make it FAIR. The difference is not
pedantry — it decides who is answerable when someone cannot find or reuse a dataset a year later.

## Who owns what

This changed in v0.5.0, and the old division is still written down in places that predate it.

| owns | what |
|---|---|
| **MD-tools** | system construction, the force-field record, the resolved protocol, seeds, generated scripts, execution provenance, **the dataset contract and its validators**, **the inventory and its checksums**, **the transactional move into managed storage** |
| **MD-data** | the permanent dataset ID, long-term custody of `$MD_DATA`, searchable metadata, locations, retention, replacement, access lifecycle |
| **MD-analysis** | analysis configuration, software identity, the dataset IDs and checksums consumed, derived-result lineage |
| **the project** | its own inputs, configurations and analysis |

**`md_data` is not a runtime dependency of MD-tools and is never imported.** The contract
semantics were ported from
[MD-data](https://github.com/csy0000/MD-data) at commit `20d982eb`, with attribution in
`src/md_tools/data_contract/migration.md`; MD-tools now implements and enforces them itself.

A directory produced by `md-openmm` before registration is **a registration candidate**, never a
FAIR dataset. It carries no DOI, no permanent identifier and no access policy. Saying otherwise
would promise something no code here delivers.

## Where the boundary runs

It is tempting to say "MD-tools never writes into `$MD_DATA`". That is wrong, and the wrong version
is the one that causes trouble.

The accurate version:

- generated data are written into the project's **own ignored staging area**, never into
  `$MD_DATA`. The `MD_DATA_LOCAL` mechanism, which pointed a run directly at a managed dataset
  directory, is gone;
- `md-openmm data-register` writes into `$MD_DATA` **once**, transactionally, at exactly the
  canonical path the contract computes;
- it never walks the archive, never opens a second dataset, and never writes into one that is
  already `complete` or `archived`.

## Canonical paths

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/common/{year}/{project_name}/{data_name}/
```

There is no month segment, and no reserved `baseline/` namespace. `year` is the year the data were
**completed** — falling back to the creation year only while a dataset is still `active` — and it
is checked against the records rather than accepted as a label.

## What a dataset carries

```text
dataset.yaml            the authoritative manifest: identity, provenance, components, status
dataset.resolved.yaml   the records this data was derived from, and the inventory
SHA256SUMS              every file, by relative path, size and digest
extension.yaml          present only when this data continues another dataset
```

`dataset.yaml` is validated against the v2 model before the transaction commits. The exported
schemas — `dataset-v2.0.schema.json` and `extension-v2.0.schema.json` — are generated from the
pydantic models and drift-checked by tests, so a published schema cannot diverge from the validator
that actually runs.

## Completion is read, never inferred

Every log MD-tools writes carries one delimited, versioned YAML machine record. Registration parses
that block and nothing else.

A log with no block, two blocks, an unparseable block, or a `schema_version` this build does not
know is **not evidence of anything**. `status: completed` is written only after outputs are
flushed, reopened and re-read. A hopeful sentence in a log is printed because the code reached that
line, not because the data are whole — and it can be typed by hand.

## Continuations

Completed and archived datasets are immutable. Extending one creates a **new** dataset under the
same year-first layout, with a new stable id that records the parent dataset and component. New
output goes to a component the new dataset owns; a link into the parent is declared, read-only, and
never the output.

The digest of the checkpoint the continuation restarted from is required, and an in-place extension
must have **copied** that checkpoint first — it writes into the very component it restarted from,
so the run can overwrite its own restart point, and the join stops being reproducible the moment
the continuation passes it.

Registration validates the record, cross-checks it against `dataset.yaml`, and re-hashes the
checkpoint if it travelled with the data. It never opens the parent: the parent is named, and a
completed parent is immutable precisely so nothing needs to reach into it.

## The transaction

Registration writes state to disk **before** the step it authorises, so every interruption boundary
is recoverable:

```text
planned -> staged -> verified -> committed -> source-removed -> linked -> complete
```

Staging is a sibling of the destination, so the commit is a rename within one filesystem. The
crossing, when there is one, happens during an explicit file-by-file copy and is followed by
re-reading every digest at the destination. The source is removed only after that verification, and
it is the last other copy, so it goes last. `shutil.move` is never treated as evidence that a
cross-filesystem transfer arrived intact.

If any check fails the source is preserved unchanged, two datasets are never merged in place, and
completed data are never overwritten.
