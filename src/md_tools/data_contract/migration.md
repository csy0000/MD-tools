# Dataset contract: v1 to v2

Version 1 was defined in [MD-data](https://github.com/csy0000/MD-data), branch
`protect-md-project-dev-test`, commit `20d982eb463ed439095f1b95e00ff1b1d75906b4`. The semantics
below were ported from there into `md_tools.data_contract`; MD-data remains their origin, and
nothing in MD-tools imports `md_data` to validate a v2 manifest.

**v2 is a new model, not a widened v1.** The v1 validator was deliberately left alone rather than
relaxed until it happened to accept the new shape: one model accepting both is exactly the state in
which a wrong path goes undetected. A v1 manifest read by a v2 reader fails on `schema_version`,
with a message that names the difference.

## What changed

| | v1 | v2 |
|---|---|---|
| `schema_version` | `1.0` | `2.0` |
| canonical path | `{namespace}/{yyyy-mm}/{dataset_name}` | `{year}/{project_name}/{data_name}` |
| shared datasets | the reserved `baseline/` namespace | `{year}/common/{project_name}/{data_name}` |
| `role` | `baseline` \| `project` | `common` \| `project` |
| dated segment | creation **month**, `yyyy-mm` | completion **year**, `YYYY` |
| identity fields | `namespace`, `dataset_name` | `year`, `project_name`, `data_name` |
| software provenance | `templates` | `software` |
| `extension.yaml` | a separate operational record | not carried into v2; continuation provenance lives in the run records |
| catalogue | `catalogue-v1.schema.json` | dropped — nothing consumed it |

### The month segment is gone

v1's dated segment was the month the dataset was **created**, and was checked against `created_at`.
That was consistent, but it answered a question few readers were asking: data are usually finished
some time after they are started, so a dataset created on 31 August and completed in September sat
under `2026-08`.

v2 uses the year the data were **completed**, falling back to the creation year only while a
dataset is still `active` — and it is *checked* against the timestamps rather than accepted as a
label. `year` is a fact about the data, not a decision made at registration.

### Project before dataset

v1 grouped by namespace and then by month, so one project's datasets were scattered across twelve
directories a year. v2 groups by year and then by project, so everything one project produced in a
year sits together, and `--common-data` inserts a single `common/` segment rather than reserving a
top-level namespace.

## What was ported unchanged

Every one of these is a v1 rule that v2 keeps, because each one exists for a reason that did not
change:

* strict `schema_version` rejection, before anything else is reported;
* a stable `dataset_id` separate from the path, so a dataset can be cited when it moves;
* relative paths only — no absolute machine path in a manifest that travels;
* an explicit `role`, checked for consistency with the shape of the path;
* creator identity as a scientific person, not a machine login;
* timezone-aware `created_at` / `completed_at`, with `completed_at` never preceding `created_at`;
* dataset status and per-component status, with `complete` and `archived` read-only;
* exact repository provenance — a repository named without a commit records where to look but not
  what ran;
* component uniqueness, non-nesting, and a required `method` on every `simulation` component;
* linked components never `active`;
* a `complete` dataset never containing `active` components.

## Migrating an existing v1 dataset

**Nothing migrates automatically, and nothing already registered was rewritten.** Datasets
registered under v1 keep their v1 manifests and their `{namespace}/{yyyy-mm}/{dataset_name}` paths.
Their provenance is a record of what actually happened, and editing it to look like v2 would make
it false.

A v1 dataset that genuinely needs to be v2 is re-registered: `md-openmm data-register` reads the
run records, derives a fresh manifest, and writes it at the v2 path. The v1 copy is not touched by
that, and removing it afterwards is a deliberate act.

## The exported schema

`schemas/dataset-v2.0.schema.json` is generated from the pydantic model in `model.py`, never
maintained by hand, and a test regenerates it and compares. A published schema that has drifted
from the validator that actually runs is worse than no published schema.
