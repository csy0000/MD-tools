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
| [paracetamol](cMD/paracetamol.md) | cMD | **parameterise a molecule once**, then a box from that package: explicit TIP3P, 1800 atoms, 200 ps (release after 0.5.4) | ~1 min |
| [chignolin](cMD/chignolin.md) | cMD | 10-residue peptide from 1UAO, explicit TIP3P, 2553 atoms, 1 ns at 4 fs | ~1 min |
| [barnase–barstar](cMD/barnase-barstar.md) | cMD | protein–protein complex from 1BRS assembly 3, PROPKA protonation, explicit TIP3P, 29725 atoms, 10 ns (release after 0.5.4) | ~40 min |
| [bromodomain + paracetamol](cMD/bromodomain-paracetamol.md) | cMD | protein–ligand complex from 4A9K assembly 1, a reused paracetamol package, PROPKA protonation, explicit TIP3P, 24036 atoms, 10 ns (release after 0.5.4) | ~35 min |
| [paracetamol](REST2/paracetamol.md) | REST2 | the registered package from the cMD page, explicit TIP3P, 4 states, 10 ns per state, 1 GPU under MPS (release after 0.5.4) | ~20 min |
| [chignolin](REST2/chignolin.md) | REST2 | 10-residue peptide from 1UAO, explicit TIP3P, 6 states, 10 ns per state, 6 GPUs | ~8 min |
| [alanine dipeptide](AIS/alanine.md) | AIS | from a sequence, explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~12 min |
| [paracetamol](AIS/paracetamol.md) | AIS | explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~13 min |
| [TYK2 + ejm_31](protein-ligand-complex/REST2/README.md) | REST2 | **skeleton, not yet run**: protein–ligand complex from OpenFE's TYK2 benchmark, selective scaling of the ligand and of the ligand plus its pocket sidechains, 8 states, 4 GPUs | TO BE MEASURED |

Start with **cMD: paracetamol**. It explains each step, and it is where the ligand parameters the
other paracetamol pages reuse are made; the others refer back to it.

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
