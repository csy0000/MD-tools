# Paracetamol

**Tested against md-tools `0.5.4`.** A single small molecule in water: 20 atoms, one rotatable
amide, one ring. The system to use when the question is about a LIGAND rather than about a protein.

![Paracetamol](images/paracetamol.png)

*Paracetamol (acetaminophen), embedded and minimised from its SMILES.*

## Why this system

A ligand needs parameters before it can be simulated, and paracetamol is small enough that
parameterising it takes about a minute — so it is the system where the parameter package, its
identity and its reuse can be shown end to end without waiting.

## Build it

Unlike a protein, a small molecule has no library of residue templates, so the first step produces
a **parameter package** of its own:

```bash
md-openmm build-top --parameterize -i paracetamol.sdf --config para.config \
    -op build/parameter/TYL.pdb -os build/parameter/TYL.xml \
    -log build/parameterize.log --resname TYL
```

Then the box, reusing that package rather than parameterising again:

```yaml
# build.config -- explicit solvent
solvent: {model: TIP3P, padding_nm: 1.0}
ligands:
  - select: {resname: TYL}
    parameters: <compound-id>/<parameter-id>
```

The package is identified by its chemical state and its parameters, not by the conformer it was
made from, so the same molecule from a different pose resolves to the same package.

## Where to go next

| method | what it is for | |
|---|---|---|
| [**cMD**](cMD.md) | parameterise once, then a plain simulation from that package | published |
| [**REST2**](REST2.md) | solute tempering of the molecule itself | published |
| [**AIS**](AIS.md) | annealed importance sampling, with torsion reweighting | published |
| **Umbrella sampling** | a potential of mean force along a chosen torsion | planned, 0.6.2 |
| **Alchemical (TI, FEP)** | hydration free energy | planned, 0.7.0 |

*Umbrella sampling arrives in 0.6.2 and the alchemical pages in 0.7.0. They are listed here so the
shape of the set is visible; neither is written yet, and neither is linked to a page that does not
exist.*
