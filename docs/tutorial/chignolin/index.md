# Chignolin

**Tested against md-tools `0.5.4`.** A ten-residue designed peptide that folds into a β-hairpin —
the smallest system in this set with a real folding equilibrium.

![Chignolin, 1UAO](images/chignolin.png)

*Chignolin (PDB 1UAO, first NMR model), coloured from N to C terminus. The hairpin is the fold;
the two termini sit side by side when it is formed.*

## Why this system

Folding and unfolding are slow compared with a torsion rotation but fast compared with a protein,
so chignolin is where an enhanced-sampling method can be shown to reach a state that plain MD
reaches only occasionally — and where the two can still be compared in a day.

## 1. The structure

```bash
mkdir -p CHI/build && cd CHI/build
curl -O https://files.rcsb.org/download/1UAO.pdb
awk '/^MODEL/{m++} m==1{print} /^ENDMDL/{if(m==1) exit}' 1UAO.pdb \
    | grep -E '^(ATOM|TER)' > chignolin.pdb
echo END >> chignolin.pdb
```

1UAO holds 18 NMR models; the build needs one. That leaves 138 atoms.

## 2. Build the system

`build-top.config`:

```yaml
solute:
  kind: peptide
solvent:
  model: TIP3P
  padding_nm: 1.5
hydrogen_mass_repartitioning:
  enabled: true
```

```bash
md-openmm build-top -i chignolin.pdb -os built.xml -op built.pdb \
    -log built.log --config build-top.config
```

Writes the System, the structure that matches it and the build record. From `built.log`:

```text
  atoms                       2553
  solute atoms                138
  waters                      803
  ions                        {'NA': 4, 'CL': 2}
  HMR                         applied, target 3.024 amu, recommend 4.0 fs
```

Four Na⁺ against two Cl⁻ because chignolin carries −2. What every key means:
[build-top](../../basics/build-top/index.md).

## Where to go next

| method | what it is for | |
|---|---|---|
| [**cMD**](cMD.md) | the unbiased reference, 1 ns at 4 fs with HMR | published |
| [**REST2**](REST2.md) | solute tempering over the whole peptide | published |
| **AIS** | annealed importance sampling | not written for this system |
| **Umbrella sampling** | a potential of mean force along an end-to-end distance | planned, 0.6.2 |
| **Alchemical (TI, FEP)** | free energies by transformation | planned, 0.7.0 |

*Umbrella sampling arrives in 0.6.2 and the alchemical pages in 0.7.0. They are listed here so the
shape of the set is visible; neither is written yet, and neither is linked to a page that does not
exist.*
