# Machine configuration

One file per machine, written once. It says **which platform this machine runs on** and **where
registered data goes** — two properties of the hardware and the filesystem, never of an
experiment.

```text
${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config
```

It is never written into the package, the repository, or `site-packages`.

## Writing it

```bash
md-openmm data-register --init
```

Interactive on a terminal; fully flag-driven otherwise
(`--noninteractive --name ... --person-id ... --md-data ...`). It asks for a stable lowercase
`person_id`, your name for provenance, and `$MD_DATA`.

Nothing but `data-register` requires the file — building and running work without it, on the
built-in defaults below.

## What it holds

```yaml
machine:
  md_data: /absolute/path/to/MD_DATA
  openmm:
    platform: CUDA        # or CPU, for a deliberate machine-wide CPU default
    precision: mixed
    device_policy: local_rank
```

Found through the order `--user-config`, `$MD_TOOLS_CONFIG`,
`${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`. A configuration written before the
`openmm:` block existed is still valid.

## The platform is a machine property

**CUDA is the default and it is mandatory**, and it is configured here rather than in any
protocol. `dynamics.platform` in a protocol configuration is **retired** and refused with a
migration message: a protocol is the same experiment wherever it runs, and a workflow carrying
`platform: CUDA` carried one machine's hardware into every repository it was shared through.

**There is no automatic fallback, in either direction.** A run that quietly moved to the CPU still
finishes, still writes a trajectory and still reports success — two orders of magnitude later, on
a machine whose GPU was simply not visible. The result is not obviously wrong, which is what makes
it expensive: it is found weeks later, if at all. CUDA that cannot be initialised is an error
*before* dynamics. Listing the platform is not the same as having a usable device — conda-forge
ships the plugin unconditionally — so the resolver opens and discards a one-particle Context to
prove it.

The CPU is reachable two ways, and both are choices somebody made:

| how | meaning |
|---|---|
| `machine.openmm.platform: CPU` | this machine has no GPU, or is deliberately CPU-only |
| `--cpu` | this one invocation, overriding the machine default |

There is no `--platform`: a per-run platform flag would be a second authority for a machine
property, and the two would disagree the first time somebody scripted one and configured the
other. `--device` says *which* GPU, never *whether*, and is refused together with `--cpu`.

`device_policy` decides where a rank's Context goes, and both values do something:

| value | behaviour |
|---|---|
| `local_rank` | one rank per visible device. The only policy that keeps a ladder off a single GPU |
| `openmm` | set no `DeviceIndex` and let OpenMM choose — right when a scheduler or MPS has already partitioned the GPUs |

`CUDA_VISIBLE_DEVICES` is yours to set. Nothing binds ranks to devices for you, and without it
every rank builds its Context on the default device and the whole ladder runs on one GPU,
silently:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh
```

## An absent configuration is not an invalid one

No file at all resolves to the built-in defaults — CUDA, mixed precision, local-rank placement. A
file that **exists and is malformed** — bad YAML, a duplicate key, an unknown field, an invalid
value — is fatal, and is never replaced by those defaults: the machine would then run on settings
nobody chose, and the file that said otherwise would never be mentioned again. `MD_TOOLS_CONFIG`
naming a file that does not exist is a broken reference, not an absence, because somebody meant
that path.

## What the record says

Every run distinguishes the three ways a platform can be chosen, because a CPU result has three
possible causes and only one of them is nobody's decision:

```yaml
acceleration:
  platform_selection: machine-config     # or built-in-default, or cli-override
  cli_cpu_override: false                # true ONLY for --cpu
  platform_origin: machine.openmm
  requested_policy: machine-cpu          # or default-cuda, or explicit-cpu
  resolved_platform: CUDA
  cuda_device_index: 3
  cuda_precision: mixed
  device_policy: local_rank
  gpus_on_host: [NVIDIA RTX A5000, ...]
  cuda_visible_devices: null
  cuda_driver_version: "580.95.05"
  openmm_version: "8.6"
  mpi: {rank: 3, size: 8, local_rank: 3}
```

`explicit_cpu` means the user typed `--cpu` on **this** invocation and nothing else; a machine
configured for CPU is `platform_selection: machine-config`. Conflating the two would make a file
somebody wrote months ago look like something they just typed.

One resolver — `md_tools.openmm.platform_policy` — serves stages, ladders and AIS alike, so they
cannot drift apart. This applies to OpenMM Context work; `build-top` assigns parameters with
OpenFF and AmberTools, which is CPU work and is not claimed to be anything else.

## Where registered data goes

`$MD_DATA` is resolved in this order, and `data-register` always prints which source supplied it:

1. `--md-data PATH`
2. `$MD_DATA`
3. `machine.md_data` in this file

```text
$MD_DATA/{year}/{project_name}/{data_name}/          # role: project
$MD_DATA/common/{year}/{project_name}/{data_name}/   # role: common
```

See [data registration](../basics/data-register/index.md) for the transaction itself, and
[the dataset contract](project-data.md) for the schema-level authority.
