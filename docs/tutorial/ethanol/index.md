# Ethanol

**Tested against md-tools `0.6.4`.** Every command and every number on these pages comes from a run
executed as written, at that version, on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.

Nine heavy-atom-and-hydrogen atoms in a box of TIP3P water. Ethanol is here for one reason: it is
the smallest system on which an **alchemical free energy** can be computed, checked against an
experimental number, and checked again against a completely different alchemical route — all in
about an hour of GPU time.

## Why this system

Hydration free energy is the simplest free energy a simulation can be asked for. There is no
binding site, no pocket water, no conformational selection: a molecule is removed from water, and
the work of removing it is the answer. That makes it the right first alchemical calculation, and
the right place to learn what the machinery does before pointing it at a protein.

Ethanol specifically, rather than ethane or a noble gas:

* **Its hydration is dominated by ELECTROSTATICS.** Ethanol hydrogen-bonds; ethane does not. Of the
  two contributions FreeSolv separates for these molecules, ethane's is almost all cavity work and
  ethanol's is almost all charging. A ladder that is correct for ethane may barely exercise the
  part where electrostatic decoupling can go wrong, and that part is where the difficulty is.
* **It is in [FreeSolv](https://github.com/MobleyLab/FreeSolv) v0.52** as `mobley_2310185`, with an
  experimental hydration free energy of **−5.00 kcal/mol**. So there is an outside number.
* **It is small.** 1,869 particles. Twelve windows of 1 ns each finish in under an hour spread over
  four cards.

One warning about the FreeSolv comparison, because it is the easiest way to draw a wrong
conclusion from this tutorial. FreeSolv also publishes a *calculated* value for each molecule, and
that column is **GAFF/AM1-BCC**. These pages use **openff-2.2.1 (Sage)/AM1-BCC**. It is a different
force field, so that column is not the reference for anything here — agreeing with it would be
reassuring, not validating, and disagreeing with it would not be a defect.

## Build it

Exactly as for any other small molecule, and in two steps: a parameter package first, then a box
that reuses it. A small molecule has no library of residue templates, so the parameters have to be
produced before anything can be solvated.

`ethanol.smi`:

```text
CCO ethanol
```

```bash
mkdir -p EOH && cd EOH
md-openmm build-top --parameterize -i ethanol.smi --config param.config \
    -op build/parameter/EOH.pdb -os build/parameter/EOH.xml \
    -log build/parameterize.log --resname EOH
```

A `.smi` states the chemistry and no coordinates, so a conformer is embedded. Coordinates are not
part of a package's identity, so the same SMILES gives the same package whichever conformer comes
out of the embedding.

**Then the box, reusing that package rather than parameterising again.** `build.config`:

```yaml
# build.config -- ethanol in a cube of TIP3P, no salt
solute:
  kind: ligand
  residue_name: EOH
  parameters: ./build/parameter        # a PATH, relative to this file

solvent:
  model: TIP3P
  padding_nm: 1.35
  box_shape: cube
  ionic_strength_molar: 0.0
  cutoff_nm: 0.9

constraints:
  type: HBonds
  rigid_water: true
```

```bash
md-openmm build-top --config build.config -odir build
```

`param.config` is the parameterisation half — force field and charge method — and it is the same
file the [paracetamol page](../paracetamol/index.md) walks through key by key; nothing about it is
specific to this molecule.

**No salt, and that is deliberate.** Decoupling changes what is in the box, and a neutral molecule
in a neutral box keeps the net charge fixed at zero throughout — which is what lets this
calculation avoid a finite-size correction entirely. (A *charged* ligand does not, and
`combine-topology` refuses to build a decoupling plan for one, by name.)

## Where to go next

| method | what it is for | |
|---|---|---|
| [**Hydration free energy**](hydration.md) | ΔG of taking ethanol out of water, by alchemical decoupling | published |
| **cMD** | ordinary dynamics on this box | planned |
| **REST2** | nothing to enhance here — one rotatable bond | not planned |

*A row with no link is a page that does not exist yet. It is listed so the method is visible, not
to suggest it is written.*

## Related

* [What the alchemical runtime has been shown to compute](../../openmm_methods/alchemy/validation.md)
  — the nine-leg campaign these numbers are checked against, including the ethane → chloroethane
  cross-check between two different alchemical routes
* [Alchemical topology](../../openmm_methods/alchemy/README.md) — the plan builder and its refusals
