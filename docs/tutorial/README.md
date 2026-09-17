# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

**Written and run with md-tools 0.5.3**, on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.
They describe 0.5.3 exactly, including where it needs a workaround; 0.5.4 will get tutorials of its
own rather than edits to these.

| tutorial | method | system | wall time |
|---|---|---|---|
| [paracetamol](cMD/paracetamol.md) | cMD | one small molecule, explicit TIP3P, 1800 atoms | ~2 min |
| [Chinolin](cMD/chinolin.md) | cMD | quinoline, explicit TIP3P, 1806 atoms | ~1 min |
| [paracetamol](REST2/paracetamol.md) | REST2 | the same molecule, a 4-state ladder, 1 ns per state | ~6 min |

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
