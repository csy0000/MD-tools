# MD-tools documentation

Start at the [root README](../README.md) for what the package is and how to install it. This
directory is the detail.

**Do not reconstruct how the package works from Git history.** It changed substantially during the
v0.5 development cycle: retired commands (`sys-gen`, `md-gen`, `sys-config`, `setup`, `openmm-md`),
a v1 dataset contract and a copy-based generated-project layout all existed and no longer do. What
is current is here.

## Methods

| page | |
|---|---|
| [Method index](openmm_methods/README.md) | what the four protocols are, and what they share |
| [cMD](openmm_methods/cMD/README.md) | ordinary MD; also the source ensemble for AIS |
| [REST2](openmm_methods/REST2/README.md) | Hamiltonian replica exchange at one temperature |
| [AIS](openmm_methods/AIS/README.md) | non-equilibrium switching and work |
| [umbrella](openmm_methods/umbrella/README.md) | biased sampling along a torsion, one window per run |

Each method page carries a minimal, runnable `example.config` beside it.

| page | |
|---|---|
| [Running](md-run.md) | `md-run`: the flags, the `.in` language, the platform policy, the MPI rules |
| [Configuration reference](md-configuration.md) | every key of every configuration, generated from the schemas |
| [The run layout](run-layout.md) | what `build-md` writes, and what is shared between runs |
| [Ligand parameter packages](ligand-packages.md) | one compound, one saved parameter set: package format, reuse, `kind: complex` ligand mapping, the catalog |

| page | |
|---|---|
| [Collective variables](collective_variables/README.md) | torsion CV reporting: the `cv.yaml` schema, conventions, cadences and outputs |

## Science

| page | |
|---|---|
| [Scientific defaults](scientific-defaults.md) | every consequential default, its evidence, and the limits of that evidence |
| [Bibliography](scientific-defaults.bib) | the sources, in BibTeX |
| [Support matrix](support-matrix.md) | supported, experimental and unsupported combinations |

## Data

| page | |
|---|---|
| [Data registration](data_register/README.md) | how to register a finished run, with worked examples |
| [The dataset contract](data-contract.md) | the schema-level authority: paths, records, immutability |

## Contributing a method

| page | |
|---|---|
| [Promoting a method](promoting-a-method.md) | the gate a method built elsewhere passes before this package carries it |

## Releases

| page | |
|---|---|
| [v0.6.1](release-notes/v0.6.1.md) | selective explicit-solvent REST2, validated on CUDA with one energy check recorded FAIL: — residue masks for backbone and sidechains, individually chosen ligand instances, and the two helpers that make a pocket selectable |
| [v0.6.0](release-notes/v0.6.0.md) | released 2026-09-19: reusable ligand parameter packages, protonation as a stated choice, assemblies and missing atoms, placement measured on real cards |
| [v0.5.4](release-notes/v0.5.4.md) | one shape of generated run, the whole chain checked before it is written, and AIS as a transformation between two topologies |
| [v0.5.3](release-notes/v0.5.3.md) | the run directory layout, an extension that can run its own group file, and the output work |
| [v0.5.2](release-notes/v0.5.2.md) | a finished run can leave this package behind: `export-reference` for cMD and REST2, and the correction that is the reason to trust it |
| [v0.5.1](release-notes/v0.5.1.md) | one fix: an interrupted CV-enabled REST2 ladder could not be resumed on any ladder whose tau is not exactly representable at six decimal places |
| [v0.5.0](release-notes/v0.5.0.md) | the interface, torsion collective variables, the dataset contract and the stable import API |
| [Runtime closure and CV evidence](release-notes/20260904-cv-and-runtime-closure-evidence.md) | the measured lanes, hardware, wheel and CUDA coverage behind the collective-variable and runtime-closure work |
| [CUDA coverage matrix](release-notes/cuda-coverage-matrix.md) | every CUDA-relevant source site and the lane that exercises it, generated from a real run |


## Development records

The campaign logs, release history, per-release aims and status, session handoffs and the backlog
are kept in a separate repository, `MD-tools-archive`, on the `development-records-20260923`
branch. They describe how the package came to work as it does, not how to use it.

