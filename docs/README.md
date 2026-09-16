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
| [cMD](openmm_methods/cMD/README.md) | ordinary MD; also the source ensemble for AIS and reservoirs |
| [REST2](openmm_methods/REST2/README.md) | Hamiltonian replica exchange at one temperature |
| [rREST2](openmm_methods/rREST2/README.md) | REST2 with a Boltzmann reservoir |
| [AIS](openmm_methods/AIS/README.md) | non-equilibrium switching and work |
| [umbrella](openmm_methods/umbrella/README.md) | biased sampling along a torsion, one window per run |

Each method page carries a minimal, runnable `example.config` beside it.

| page | |
|---|---|
| [Running](md-run.md) | `md-run`: the flags, the `.in` language, the platform policy, the MPI rules |
| [Configuration reference](md-configuration.md) | every key of every configuration, generated from the schemas |
| [The run layout](run-layout.md) | what `build-md` writes, and what is shared between runs |

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

## Reference campaigns

| page | |
|---|---|
| [ALA, 2026-09](campaigns/ala-2026-09/README.md) | the alanine-dipeptide reference campaign: configurations, built Systems, what was registered and what was withdrawn |
| [Validation throughput, 2026-08](reports/20260831_validation_matrix_speed.md) | measured ns/day for the 2026-08 validation matrix, with its CSVs |

## Contributing a method

| page | |
|---|---|
| [Promoting a method](promoting-a-method.md) | the gate a method built elsewhere passes before this package carries it |

## Releases

| page | |
|---|---|
| [v0.5.3](release-notes/v0.5.3.md) | the current cycle: the run directory layout, an extension that can run its own group file, and the output work |
| [v0.5.2](release-notes/v0.5.2.md) | a finished run can leave this package behind: `export-reference` for cMD and REST2, and the correction that is the reason to trust it |
| [v0.5.1](release-notes/v0.5.1.md) | one fix: an interrupted CV-enabled REST2 ladder could not be resumed on any ladder whose tau is not exactly representable at six decimal places |
| [v0.5.0](release-notes/v0.5.0.md) | the interface, torsion collective variables, the dataset contract and the stable import API |
| [Runtime closure and CV evidence](release-notes/20260904-cv-and-runtime-closure-evidence.md) | the measured lanes, hardware, wheel and CUDA coverage behind the collective-variable and runtime-closure work |
| [CUDA coverage matrix](release-notes/cuda-coverage-matrix.md) | every CUDA-relevant source site and the lane that exercises it, generated from a real run |

## History

| page | |
|---|---|
| [Development history](history/README.md) | the instructions and execution journals behind the REST2, rREST2 and file-interface work, kept as evidence. Not a description of how the package works now |
