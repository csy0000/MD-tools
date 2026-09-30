# Alanine dipeptide

**Tested against md-tools `0.6.1`.** The smallest system in this set: two peptide bonds capped at
both ends, 22 atoms, and the standard toy for everything that depends on backbone torsions.

![The alanine dipeptide](images/ala.png)

*ACE–ALA–NME. Its two backbone torsions, φ and ψ, are the coordinates every method on this page
is measured against.*

## Why this system

φ and ψ have barriers of a few kT, so a plain simulation crosses them often enough to give a
reference distribution in minutes rather than days. That makes it the one system where an enhanced
method can be checked against the truth instead of against another enhanced method.

## Build it

The structure ships with the package as `tests/data/ALA.pdb`, and the build is one command. For an
explicit box:

```yaml
# build.config -- explicit solvent
solvent: {model: TIP3P, padding_nm: 1.0}
```

For implicit solvent, which for a system this small is often enough and has no box at all:

```yaml
# build.config -- implicit solvent
solvent: {model: GBn2}
```

```bash
md-openmm build-top -i ALA.pdb --config build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

An implicit build has no box, no barostat, no counter-ions and no NPT stage; the stages are
renamed accordingly rather than run at a pressure that means nothing.

## Where to go next

| method | what it is for | |
|---|---|---|
| [**cMD**](cMD.md) | the unbiased reference every method here is checked against, 10 ns | published |
| [**REST2**](REST2.md) | solute tempering, four states over τ 0 → 0.5 | published |
| [**AIS**](AIS.md) | annealed importance sampling between two Hamiltonians, with the torsion reweighting checked against the plain simulation | published |
| [**Umbrella sampling**](umbrella.md) | biasing φ, one window per run, towards a potential of mean force | commands published; the profile itself **not yet measured** |
| **Alchemical (TI, FEP)** | free energies by transformation | planned, 0.7.0 |

*The umbrella page's build, stages, generated tree and refusals were executed as written, on the
CPU; its per-window means, the window overlap and the PMF need CUDA and are marked as a gap on the
page rather than filled. The alchemical pages arrive in 0.7.0 — listed so the shape of the set is
visible, not written yet, and not linked to a page that does not exist.*
