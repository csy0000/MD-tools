# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

!!! note "Requires md-tools 0.5.4 or later"
    These pages describe 0.5.4, where REST2 states are built once, as files, by
    `md-openmm build-top --rest2-scaler`. The 0.5.3 tutorials are [archived](archived/README.md),
    with the reasons.

Run on NVIDIA RTX 3080 GPUs (and one RTX A5000 for the Chinolin ladder) with CUDA and mixed
precision.

| tutorial | method | system | wall time |
|---|---|---|---|
| [paracetamol](cMD/paracetamol.md) | cMD | one small molecule, explicit TIP3P, 1800 atoms | ~2 min |
| [Chinolin](cMD/chinolin.md) | cMD | quinoline, explicit TIP3P, 1806 atoms | ~1 min |
| [paracetamol](REST2/paracetamol.md) | REST2 | explicit TIP3P, 4 states, 10 ns per state, 4 GPUs | ~12 min |
| [Chinolin](REST2/chinolin.md) | REST2 | explicit TIP3P, 4 states, 10 ns per state, 4 GPUs | ~11 min |
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
