# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

**Every page states the md-tools version it was executed against**, at the top. A page whose
version is older than the one you have installed has not been re-run for your release: the commands
may still be right, but nothing on this site claims they were checked against it.

At each major release every tutorial is re-run as written and adjusted where it no longer passes.
The 0.5.3 pages are [archived](archived/README.md), with the reasons.

Run on NVIDIA RTX 3080 GPUs with CUDA and mixed
precision.

| tutorial | method | system | wall time | tested against |
|---|---|---|---|---|
| [paracetamol](cMD/paracetamol.md) | cMD | **parameterise a molecule once**, then a box from that package: explicit TIP3P, 1800 atoms, 200 ps (release after 0.5.4) | ~1 min | `0.5.4` |
| [chignolin](cMD/chignolin.md) | cMD | 10-residue peptide from 1UAO, explicit TIP3P, 2553 atoms, 1 ns at 4 fs | ~1 min | `0.5.4` |
| [barnase–barstar](cMD/barnase-barstar.md) | cMD | protein–protein complex from 1BRS assembly 3, PROPKA protonation, explicit TIP3P, 29725 atoms, 10 ns (release after 0.5.4) | ~40 min | `0.5.4` |
| [bromodomain + paracetamol](cMD/bromodomain-paracetamol.md) | cMD | protein–ligand complex from 4A9K assembly 1, a reused paracetamol package, PROPKA protonation, explicit TIP3P, 24036 atoms, 10 ns (release after 0.5.4) | ~35 min | `0.5.4` |
| [paracetamol](REST2/paracetamol.md) | REST2 | the registered package from the cMD page, explicit TIP3P, 4 states, 10 ns per state, 1 GPU under MPS (release after 0.5.4) | ~20 min | `0.5.4` |
| [chignolin](REST2/chignolin.md) | REST2 | 10-residue peptide from 1UAO, explicit TIP3P, 6 states, 10 ns per state, 6 GPUs | ~8 min | `0.5.4` |
| [alanine dipeptide](AIS/alanine.md) | AIS | from a sequence, explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~12 min | `0.5.4` |
| [paracetamol](AIS/paracetamol.md) | AIS | explicit TIP3P, 2 ns source + 64 switching paths, 1 GPU | ~13 min | `0.5.4` |
| [TYK2 + ejm_31](protein-ligand-complex/REST2/README.md) | REST2 | protein–ligand complex from OpenFE's TYK2 benchmark, selective scaling of the ligand and of the ligand plus its pocket sidechains; four ladders run on CUDA, 8 and 12 states, 4 and 6 GPUs | 93.5 ns/day per state (8 rungs, 4 cards), 81.5 (12 rungs, 6 cards); two registered datasets | `0.6.1` |

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
