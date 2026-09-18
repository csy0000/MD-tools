# Development history

The instructions that specified the REST2, rREST2 and Amber-like file-interface work, and the
execution journals that recorded doing it. They were written in the `MD-project` repository, where
that work happened before the code moved here, and were copied from its commit
`94ad0cd28d2a155302f3c28746f16cef25983fd0` on 2026-09-11 **byte for byte**. Nothing in them has been
edited, including what has since turned out to be wrong.

**This is history, not documentation.** It names things that no longer exist -- the `openmm-md`
executable, MD-templates, `md-openmm setup`, the pre-rename reporting keys -- because they existed
when it was written. How the package works today is in the [documentation index](../README.md).

## Why it is kept

Some tests state that their real-run evidence is a recorded end-to-end sequence rather than
anything a test fixture can reproduce. That evidence is in these journals:

| test | journal |
|---|---|
| `tests/test_migration_provenance.py` | [20260831_rest2-migration-provenance](journals/20260831_rest2-migration-provenance.md) |
| `tests/test_migration_transaction.py` | [20260831_rest2-migration-transaction](journals/20260831_rest2-migration-transaction.md) |
| `tests/test_v2_append_compatibility.py` | [20260831_rest2-v2-append-compatibility](journals/20260831_rest2-v2-append-compatibility.md) |
| `tests/test_rest2_edge_contracts.py` | [20260831_rest2-rrest2-edge-contracts](journals/20260831_rest2-rrest2-edge-contracts.md), [20260831_rest2-state-trajectories](journals/20260831_rest2-state-trajectories.md) |

## Contents

| instruction | journal |
|---|---|
| [20260828 finish the Amber-like OpenMM file interface](claudecode-instructions/20260828_finish_amber_like_openmm_file_interface.md) | [20260828 Amber-like OpenMM workflow](journals/20260828_amber_like_openmm_workflow.md) |
| [20260829 OpenMMTools-based REST2](claudecode-instructions/20260829_openmmtools-rest2.md) | [20260829 REST2 on OpenMMTools](journals/20260829_openmmtools_rest2.md) |
| [20260829 REST2 production correctness](claudecode-instructions/20260829_rest2-production-correctness.md) | [20260829 REST2 production correctness](journals/20260829_rest2-production-correctness.md) |
| [20260830 own REST2 and rREST2](claudecode-instructions/20260830_own-rest2-rrest2-openmm-md.md) | [20260830 own REST2 and rREST2](journals/20260830_own-rest2-rrest2-openmm-md.md) |
| [20260830 REST2/rREST2 correctness and phase space](claudecode-instructions/20260830_rest2-rrest2-correctness-phase-space.md) | [20260830 correctness and phase space](journals/20260830_rest2-rrest2-correctness-phase-space.md) |
| [20260831 REST2/rREST2 edge contracts](claudecode-instructions/20260831_rest2-rrest2-edge-contracts.md) | [20260831 edge contracts](journals/20260831_rest2-rrest2-edge-contracts.md) |
| [20260831 state trajectories, CMAP, rem.log](claudecode-instructions/20260831_rest2-state-trajectories-cmap-rem-log.md) | [20260831 state trajectories](journals/20260831_rest2-state-trajectories.md) |
| — | [20260831 migration provenance](journals/20260831_rest2-migration-provenance.md) |
| — | [20260831 migration transaction](journals/20260831_rest2-migration-transaction.md) |
| — | [20260831 pending validation](journals/20260831_rest2-pending-validation.md) |
| — | [20260831 v2 append compatibility](journals/20260831_rest2-v2-append-compatibility.md) |
| [20260901 standalone MD-tools, minimal MD-project](claudecode-instructions/20260901_md_tools_standalone_and_minimal_md_project.md) | — |

The last instruction is the one that created this repository as a standalone package. Its journal
was about the other repository and stayed there.

## Commit hashes before 2026-09-11

This repository's history was rewritten on 2026-09-11, before it was made public, to remove one
workstation's hostname and absolute paths from old files and commit messages, and to give every
commit one author address. Nothing else changed: the same commits, in the same order, with the
same dates, messages and final file tree. Every commit hash changed with it.

Records written before then cite the old hashes -- run records, dataset manifests and notes,
exported bundles, and release notes in this directory tree. [`commit-map.tsv`](commit-map.tsv)
maps every old hash to its new one. The ones those records cite most:

| cited as | now | what it is |
|---|---|---|
| `514e44b` | `6e94dbe` | the engine content the 2026-09 ALA campaign ran on |
| `ae6e994` | `d386899` | the exporter behind the campaign's REST2 bundles |
| `cd2b1d1` | `3cbe56b` | the 0.5.2 release commit before the final fixes |
| `c7e7265` | `3fc244f` | `main` at 0.5.1 |

The original history is kept in a private archive.

## Completed release instructions

Instructions that were carried out, moved here when their work landed. They are kept for the
reasoning and the evidence trail, not as a description of current behaviour. Each row says where
it came from, what shows it was done, and what replaced it.

| instruction | origin | completion evidence | replaced by |
|---|---|---|---|
| [20260913 GPU tests for 0.5.3](claudecode-instructions/20260913_gpu-tests-for-0.5.3.md) | the hpREST2 campaign's three findings on `fix/hprest2-findings` | [its results page](claudecode-instructions/20260913_gpu-tests-for-0.5.3-results.md), and `tests/test_hprest2_gpu_evidence.py` | — |
| [20260913 GPU test results](claudecode-instructions/20260913_gpu-tests-for-0.5.3-results.md) | the answer to the row above | itself: four CUDA tests on GPUs 7 and 8 | — |
| [20260913 PR body for 0.5.3](claudecode-instructions/20260913_pr-body-0.5.3.md) | the 0.5.3 release PR | [v0.5.3 release notes](../release-notes/v0.5.3.md) | — |
| [20260917 reusable ligands, PROPKA, concurrent CUDA, AIS](claudecode-instructions/20260917_next-release-reusable-ligands-propka-cuda-ais.md) | the user's next-release scope, 2026-09-17 | [v0.6.0 release notes](../release-notes/v0.6.0.md); 0.6.0 is still under the user's testing, so its *release* is not claimed | [20260918 parallel 0.6.1–0.7.2 development](../claudecode-instructions/20260918_parallel-0.6.1-0.7.x-development.md) |
| [20260917 AIS schedules and frames evidence](claudecode-instructions/20260917_evidence-ais-schedules-and-frames.md) | the companion evidence for two rows of the instruction above | itself | — |

The 20260917 instruction calls the release **v6.0.0**. The shipped numbering is **0.6.0**; the
label was never reconciled while it was active. Its wording is left as written, as everything in
this directory is. Active documents use 0.6.0, and the work it deferred — selected-residue REST2,
TI, FEP — is now carried by the branch aims under
[`docs/development/`](../development/README.md), not by that archived section 9.
