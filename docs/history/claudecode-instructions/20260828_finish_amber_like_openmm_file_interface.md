# Finish the Amber-like OpenMM file interface

## Purpose

Finish the conventional-MD interface before beginning REST2.

This is a bounded cross-repository correction. It turns the current short generated OpenMM scripts
into reusable protocol inputs whose concrete input and output paths are supplied by generated shell
launchers through an Amber-like command-line interface.

Work in both repositories in the same Claude Code session:

```text
csy0000/MD-templates  branch feat/amber-like-openmm
csy0000/MD-project    branch dev3-amber-like
```

At the time this instruction was written, the relevant remote heads were:

```text
MD-templates feat/amber-like-openmm
df4629f7e2ba3f9fdd473aaaaa64ac57191ee7f0

MD-project dev3-amber-like before this instruction commit
07cce22b7ef5fe926f7e5f9e7308b23a037cbbb5
```

Fetch both repositories and inspect their actual remote heads before editing. Preserve newer work.
Do not reset, force-push, rewrite history, merge to `dev` or `main`, or begin REST2/AIS
implementation.

Use authenticated Git access already configured in the environment. If either repository is dirty,
diverged unexpectedly, or inaccessible, stop and report it rather than replacing work.

## One-push rule

Complete and test the change locally before publishing it.

- Local logical commits are allowed.
- Push MD-templates exactly once, after its complete suite passes.
- Then update and test MD-project against that exact pushed MD-templates commit.
- Push MD-project exactly once.
- Do not use iterative remote pushes as a testing loop.
- Do not open or merge pull requests in this task.

## Design boundary

The generated simulation has four distinct pieces:

1. A short Python protocol file contains scientific choices: integrator, ensemble, restraint,
   number of steps, output interval, and the direct OpenMM operations.
2. A generated `paths.sh` contains the portable directory mapping under `$MD_DATA`.
3. One generated shell launcher per stage supplies every concrete input and output path.
4. A small standalone `openmm-md` executable performs only generic file-interface work.

MD-templates is a generator. A generated system must run after the MD-templates checkout and Python
package have been removed.

Do not turn `openmm-md` into a simulation framework. It must not decide force fields, protocols,
equilibration schedules, restraints, temperatures, step counts, or ensembles. Those remain visible
in the Python protocol file.

## Required generated layout

For an explicit-solvent ALA system, generate:

```text
ALA/
├── config.yaml
├── paths.sh
├── bin/
│   └── openmm-md
├── input/
│   ├── topology.pdb
│   ├── system.xml
│   ├── initial_state.xml
│   ├── resolved_sys.config.yaml
│   └── provenance.yaml
├── min/
│   ├── min.py
│   └── min.sh
├── eq/
│   ├── nvt_1kcal/
│   │   ├── nvt_1kcal.py
│   │   └── nvt_1kcal.sh
│   ├── npt_1kcal/
│   │   ├── npt_1kcal.py
│   │   └── npt_1kcal.sh
│   └── npt_free/
│       ├── npt_free.py
│       └── npt_free.sh
├── cMD/
│   ├── cmd.py
│   └── cmd.sh
└── run.sh
```

Use `input/` consistently for the new `setup` path. Do not retain a second `common/` alias in
the same generated system.

The existing advanced `sys-gen` / `md-gen` layout is outside this task and may remain unchanged.
Do not break REST2 or AIS while changing the new `setup` path.

## Portable path file

Generate `paths.sh` at the system root. It must contain no machine-specific absolute path. It
must resolve the system from `$MD_DATA` plus the output path relative to that managed root.

For a system generated under `$MD_DATA/MY_PROJECT/2026-08/ALA`, its effective contract is:

```bash
#!/usr/bin/env bash

: "${MD_DATA:?MD_DATA is not set}"

export PROJECT_DATA="${MD_DATA}/MY_PROJECT/2026-08"
export SYSTEM_ROOT="${PROJECT_DATA}/ALA"

export INPUT_DIR="${SYSTEM_ROOT}/input"
export MIN_DIR="${SYSTEM_ROOT}/min"
export EQ_DIR="${SYSTEM_ROOT}/eq"
export NVT_DIR="${EQ_DIR}/nvt_1kcal"
export NPT_RESTRAINED_DIR="${EQ_DIR}/npt_1kcal"
export NPT_FREE_DIR="${EQ_DIR}/npt_free"
export CMD_DIR="${SYSTEM_ROOT}/cMD"
export CMD_TAU0P5_DIR="${SYSTEM_ROOT}/cMD_tau0p5"
export REST2_DIR="${SYSTEM_ROOT}/REST2"
export AIS_DIR="${SYSTEM_ROOT}/AIS"

export OPENMM_MD="${OPENMM_MD:-${SYSTEM_ROOT}/bin/openmm-md}"
```

Only define directories that are meaningful for the selected solvent and generated methods, except
that the method-root variables above may be reserved consistently for the later REST2/AIS work.

`md-openmm setup` must require its output root to resolve beneath `$MD_DATA`, then derive the
portable relative project path from it. It must refuse:

- an unset or unresolved `MD_DATA`;
- an output outside `$MD_DATA`;
- an output inside any Git worktree;
- a path containing traversal outside the managed root.

When checking a not-yet-created output path for Git ownership, walk upward to the nearest existing
ancestor before invoking Git. The present immediate-parent check is insufficient for nested missing
directories.

## The `openmm-md` file interface

Generate a standalone executable at `bin/openmm-md`. It may use only the Python standard library
and OpenMM. It must not import `md_templates`, read YAML, invoke Git, perform registration, or know
project schemas.

Required interface:

```text
openmm-md
  -i, --input       Python protocol file
  -p, --topology    topology PDB
  -s, --system      serialized OpenMM System XML
  -c, --coordinates starting State XML
  -o, --output      text .out
  -x, --trajectory  trajectory, optional for stages such as minimization
  -r, --restart     portable final State XML
      --checkpoint  binary checkpoint, optional
      --solute-x    solute-only trajectory, optional
      --force       deliberately replace existing runtime outputs
```

The short names deliberately follow the Amber mental model where possible. OpenMM needs both a
topology and a serialized System, so `-p` and `-s` are separate.

Resolution precedence is:

```text
explicit command-line flag
> corresponding exported OPENMM_* environment variable
> clear error before an OpenMM Context or output file is created
```

Generated stage shell files must nevertheless pass all applicable paths explicitly. Environment
fallback exists for manual use, not to hide the generated wiring.

Before execution, `openmm-md` must:

- resolve and validate every required input;
- ensure input and output paths are distinct;
- create no OpenMM Context during validation;
- refuse existing `.out`, trajectory, restart, or checkpoint files unless `--force` was given;
- create parent output directories only after validation succeeds;
- capture both stdout and stderr in `-o`;
- preserve a failed `.out` for diagnosis but never add a completion marker itself;
- return the protocol's nonzero exit status;
- require the protocol to print `run_status: completed` only after its requested restart and
  checkpoint outputs exist.

Load the input Python file without requiring it to be importable as a package. Define one minimal
call contract, such as a top-level `run(files)` function, and document it. The `files` object may
be a simple namespace supplied by `openmm-md`. Do not introduce plugins, registries, inheritance,
or a general workflow API.

The generated copy of `openmm-md` is authoritative for that system. An `OPENMM_MD` override may
select another compatible executable deliberately, but ordinary execution must not depend on a
globally installed MD-templates command.

## Stage Python files

A stage Python file must contain no concrete project path, no `Path(__file__)` directory traversal,
no YAML parsing, no Git operation, and no MD-templates import.

It receives paths through the runner's `files` argument. It should remain direct, readable OpenMM
application code. For example, the NVT input should visibly contain the equivalents of:

```python
def run(files):
    pdb = PDBFile(files.topology)
    system = XmlSerializer.deserialize(Path(files.system).read_text())
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        2 * unit.femtoseconds,
    )
    simulation = Simulation(pdb.topology, system, integrator, ...)
    simulation.context.setState(
        XmlSerializer.deserialize(Path(files.coordinates).read_text())
    )
    ...
```

Do not hide `Simulation`, the integrator, restraints, barostat, reporters, or `simulation.step()`
inside `openmm-md`.

The same generated Python protocol file must be reusable with another compatible system and another
set of output paths by changing only the shell invocation.

Keep the current readability limits. If the small `run(files)` interface requires a few additional
lines, report the measured count and justify only the unavoidable increase. Do not solve line-count
pressure by moving scientific operations into an opaque helper.

## Stage shell files

Each stage gets a short shell file. It locates and sources the system-root `paths.sh`, then calls
`$OPENMM_MD` with all applicable files.

The NVT launcher must be structurally equivalent to:

```bash
#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${STAGE_DIR}/../../paths.sh"

"${OPENMM_MD}" \
  -i "${NVT_DIR}/nvt_1kcal.py" \
  -p "${INPUT_DIR}/topology.pdb" \
  -s "${INPUT_DIR}/system.xml" \
  -c "${MIN_DIR}/min.state.xml" \
  -o "${NVT_DIR}/nvt_1kcal.out" \
  -x "${NVT_DIR}/nvt_1kcal.dcd" \
  -r "${NVT_DIR}/nvt_1kcal.state.xml" \
  --checkpoint "${NVT_DIR}/nvt_1kcal.chk"
```

Minimization may omit `-x` and `--checkpoint`. Every child stage must name the parent stage's
portable restart explicitly through `-c`.

The root `run.sh` calls the stage shell files in order. It must not reconstruct their commands,
redirect their output separately, or contain scientific settings.

## Output contract

The `.out` remains the authoritative human-readable execution report. It must record the resolved
values that actually ran, including:

- output grammar version;
- stage and system;
- OpenMM version and platform;
- integrator and ensemble;
- temperature, friction, timestep, restraint and pressure where applicable;
- explicit integrator and barostat random seeds;
- number of steps and physical duration;
- output interval;
- resolved topology, System, starting state, trajectory, restart and checkpoint paths;
- the force-field/provenance record path under `input/`;
- reporter output;
- `run_status: completed` printed last by the protocol only after requested outputs exist.

Do not duplicate the full force-field configuration in every `.out`. Reference the generated
`input/resolved_sys.config.yaml` and `input/provenance.yaml`; later registration computes
checksums.

## Metadata and seed corrections

The small setup request and generated `config.yaml` must record a confirmed contributor and
creation date.

Contributor resolution may use:

```text
explicit setup.yaml field
> MD_CONTRIBUTOR environment variable
> interactive prompt
> error
```

Never invent a person. Keep `config.yaml` small: identity, contributor, date, source identity,
route, solvent selection, generator version and exact commit when available.

Record the installed MD-templates package version even when Git metadata is unavailable. A wheel or
released installation may correctly have no Git checkout; `md_templates_commit: null` is
acceptable only when a concrete package version is also present.

Resolve stochastic seeds at generation time:

- accept an explicit base seed;
- otherwise generate a concrete base seed once;
- derive distinct literal stage seeds deterministically;
- set the integrator and barostat seeds explicitly;
- print them in the resolved preset and each relevant `.out`;
- do not accept `advanced.common.random_seed` and then silently ignore it.

## Correct the setup documentation

Make implementation and documentation agree:

- config-driven setup is explicitly non-interactive when `--yes` is supplied;
- without `--yes`, both interactive and config-driven setup show the complete resolved preset and
  ask before writing;
- examples used in unattended tests pass `--yes`;
- count the actual questions instead of calling nine prompts “seven questions”;
- remove or correct the broken one-field OPC example.

Do not change validated scientific defaults in this task. If OPC is shown, resolve the complete
coupled ff19SB/OPC selection through one supported user-facing choice or show every required coupled
override. Never claim that changing only the water XML selects ff19SB/OPC.

## MD-project integration

After MD-templates is complete, tested, committed, and pushed once:

1. Record its exact new 40-character commit in MD-project wherever the current transitional pin is
   still required.
2. Ensure a fresh MD-project checkout following its documented setup obtains a version that really
   provides `md-openmm setup`.
3. Do not add a runtime component dependency to generated systems.
4. Update the ALA and phenol/IPH examples to include contributor resolution and the managed output
   invocation.
5. Update README comparisons so the OpenMM example uses `openmm-md -i ... -o ...`, alongside the
   relevant `-p/-s/-c/-x/-r` options.
6. Update `md/README.md` and the examples to show `paths.sh` and stage shell launchers.
7. Preserve the rule that generated data live under `$MD_DATA` and `md/<system>` is an ignored
   symlink.
8. Do not broaden or redesign component management. Its later removal remains a separate cleanup
   after REST2 and AIS use the new convention.

## Tests

Add focused tests in MD-templates for:

- exact CLI option parsing and environment fallback;
- CLI values overriding environment values;
- missing required paths failing before output creation or Context construction;
- output collision refusal and explicit `--force`;
- input/output path distinctness;
- stdout and stderr capture;
- nonzero protocol exit propagation;
- no false completion marker;
- generated `paths.sh` portability under a changed `MD_DATA`;
- every generated stage shell file supplying the correct parent and outputs;
- Python protocol files containing no concrete generated paths;
- reuse of one protocol file with two temporary compatible path sets;
- detachment with MD-templates made unavailable;
- explicit seed setting and reporting;
- contributor/date/package-version metadata;
- nested missing output paths inside a Git worktree being refused;
- readable DCD and State XML;
- ALA peptide and phenol/IPH ligand generation;
- the complete existing non-GPU and GPU suites.

Add MD-project tests for:

- the exact MD-templates pin providing the documented command;
- example requests and commands matching the implemented interface;
- no generated data or machine absolute paths tracked;
- `md/*` symlinks ignored;
- README command extraction against the real pinned CLI without silently skipping because an
  unrelated development checkout is absent.

Use temporary data roots. Production simulations are not part of this task. Run only bounded smoke
dynamics needed to prove the interface, including one CUDA smoke when CUDA is available.

## Non-goals

Do not:

- implement or redesign REST2 or AIS;
- adopt OpenMMTools yet;
- change trajectory format to NetCDF;
- change force-field, solvent, timestep, box or equilibration defaults;
- implement MD-data registration;
- add Snakemake;
- create a generic plugin framework;
- merge branches;
- delete the older advanced path or component machinery;
- store generated data in either Git repository.

## Completion

Before either push:

1. Inspect both diffs for accidental scope growth.
2. Run formatting and linting.
3. Run the focused interface tests.
4. Run the complete MD-templates non-GPU suite.
5. Run the complete MD-templates GPU suite when CUDA is available.
6. Generate and execute bounded ALA and phenol/IPH examples.
7. Prove the generated systems run with MD-templates unavailable.
8. Read back the trajectory and portable restart.
9. Run the complete MD-project suite against the exact new MD-templates commit.
10. Verify README commands literally.
11. Confirm no generated binaries, trajectories, states, checkpoints, machine paths, or symlinks
    are tracked.
12. Inspect clean Git status.

Write or update execution journals in both repositories with exact commands, commits, test counts,
skips, hardware limitations, deviations and unresolved gaps. Do not claim an unavailable GPU test
passed.

Commit with informative messages. Push MD-templates once, then MD-project once. Report both final
branch heads. Stop there; REST2 is the next task.
