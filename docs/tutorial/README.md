# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

!!! note "Requires md-tools 0.5.4 or later"
    These pages describe 0.5.4, where REST2 states are built once, as files, by
    `md-openmm build-top --rest2-scaler`. The 0.5.3 tutorials are [archived](archived/README.md),
    with the reasons.

Run on NVIDIA RTX 3080 GPUs with CUDA and mixed
precision.

| tutorial | method | system | wall time |
|---|---|---|---|
| [paracetamol](cMD/paracetamol.md) | cMD | one small molecule, explicit TIP3P, 1800 atoms | ~2 min |
| [chignolin](cMD/chignolin.md) | cMD | 10-residue peptide from 1UAO, explicit TIP3P, 2553 atoms, 1 ns at 4 fs | ~1 min |
| [barnase–barstar](cMD/barnase-barstar.md) | cMD | protein–protein complex from 1BRS assembly 3, PROPKA protonation, explicit TIP3P, 29725 atoms, 10 ns (release after 0.5.4) | ~40 min |
| [paracetamol](REST2/paracetamol.md) | REST2 | explicit TIP3P, 4 states, 10 ns per state, 4 GPUs | ~12 min |
| [chignolin](REST2/chignolin.md) | REST2 | 10-residue peptide from 1UAO, explicit TIP3P, 6 states, 10 ns per state, 6 GPUs | ~8 min |
| [alanine dipeptide](AIS/alanine.md) | AIS | from a sequence, explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~12 min |
| [paracetamol](AIS/paracetamol.md) | AIS | explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~13 min |

Start with **cMD: paracetamol**. It explains each step; the others refer back to it.

## Before any of them

1. [Install md-tools](../install.md) with CUDA, and check `md-openmm --version`.
2. [Configure the machine](../machine-configuration.md). CUDA is the default and is mandatory:
   nothing falls back to the CPU on its own.

## The shape every tutorial follows

```text
md-openmm build-top   a structure       -> build/built.xml, built.pdb (the system)
md-openmm build-md    a configuration   -> a run directory with run.sh
./run.sh                                -> the stages, each an `md-openmm md-run`
```
