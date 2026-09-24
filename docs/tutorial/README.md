# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

**Every page states the md-tools version it was executed against**, at the top. A page whose
version is older than the one you have installed has not been re-run for your release: the commands
may still be right, but nothing on this site claims they were checked against it.

At each major release every tutorial is re-run as written and adjusted where it no longer passes.
The 0.5.3 pages are [archived](archived/README.md), with the reasons.

## Start from a system

Each system page introduces the molecule, shows its structure, and builds it — explicit or implicit
solvent — before handing you to a method. The systems are ordered by what they are for, not by
size.

| system | what it is for | methods |
|---|---|---|
| [**Alanine dipeptide**](ALA/index.md) | the toy: two torsions with barriers a plain run crosses, so a method can be checked against the truth | AIS |
| [**Paracetamol**](paracetamol/index.md) | a ligand on its own: parameterise once, reuse everywhere | cMD, REST2, AIS |
| [**Chignolin**](chignolin/index.md) | a folding peptide: a real equilibrium, small enough to compare methods in a day | cMD, REST2 |
| [**Barnase + barstar**](barnase-barstar/index.md) | a protein–protein interface: the coordinate is between two molecules | cMD |
| [**TYK2 + ejm_31**](tyk2-ejm31/index.md) | a kinase with a real inhibitor: where a hot region is worth CHOOSING | selective REST2 |

## The methods, and where they are

| method | systems | state |
|---|---|---|
| **cMD** | paracetamol, chignolin, barnase–barstar | published |
| **REST2** | paracetamol, chignolin, TYK2 | published |
| **AIS** | alanine dipeptide, paracetamol | published |
| **Umbrella sampling** | alanine dipeptide (φ), protein–ligand and protein–protein (centre-of-mass distance) | planned, 0.6.2 |
| **Alchemical — TI and FEP** | every system | planned, 0.7.0 |

The two planned rows are listed so the shape of the set is visible. Neither is written, and neither
is linked to a page that does not exist.

Run on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.

| page | version it was run against |
|---|---|
| [paracetamol / cMD](paracetamol/cMD.md) | `0.5.4` |
| [paracetamol / REST2](paracetamol/REST2.md) | `0.5.4` |
| [paracetamol / AIS](paracetamol/AIS.md) | `0.5.4` |
| [chignolin / cMD](chignolin/cMD.md) | `0.5.4` |
| [chignolin / REST2](chignolin/REST2.md) | `0.5.4` |
| [alanine dipeptide / AIS](ALA/AIS.md) | `0.5.4` |
| [barnase + barstar / cMD](barnase-barstar/cMD.md) | `0.5.4` |
| [TYK2 + ejm_31 / selective REST2](tyk2-ejm31/REST2.md) | `0.6.1` |
