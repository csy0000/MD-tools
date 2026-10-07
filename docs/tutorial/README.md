# Tutorials

Complete runs, start to finish, each one executed exactly as written. The output shown on every
page is copied from the files that run produced, not composed for the page.

**Every page states the md-tools version it was executed against**, at the top. A page whose
version is older than the one you have installed has not been re-run for your release: the commands
may still be right, but nothing on this site claims they were checked against it.

At each major release every tutorial is re-run as written and adjusted where it no longer passes.
The 0.5.3 pages, and the 0.6.4 paracetamol clustering page that the symmetry-first workflow replaced,
are [archived](archived/README.md), with the reasons.

## Start from a system

Each system page introduces the molecule, shows its structure, and builds it — explicit or implicit
solvent — before handing you to a method. The systems are ordered by what they are for, not by
size.

| system | what it is for | methods |
|---|---|---|
| [**Alanine dipeptide**](ALA/index.md) | the toy: two torsions with barriers a plain run crosses, so a method can be checked against the truth | cMD, REST2, AIS |
| [**Paracetamol**](paracetamol/index.md) | a ligand on its own: parameterise once, reuse everywhere | cMD, REST2, AIS |
| [**Ethanol**](ethanol/index.md) | the smallest alchemical free energy there is, with an experimental number to check it against | hydration free energy |
| [**Ethane + chloroethane**](ethane-chloroethane/index.md) | one atom apart: the relative free energy pattern every protein-ligand campaign uses | relative hydration |
| [**Chignolin**](chignolin/index.md) | a folding peptide: a real equilibrium, small enough to compare methods in a day | cMD, REST2 |
| [**Barnase + barstar**](barnase-barstar/index.md) | a protein–protein interface: the coordinate is between two molecules | cMD |
| [**TYK2 + ejm_31**](tyk2-ejm31/index.md) | a kinase with a real inhibitor: where a hot region is worth CHOOSING | cMD, selective REST2, AIS |

## The methods, and where they are

| method | systems | state |
|---|---|---|
| **cMD** | alanine dipeptide, paracetamol, chignolin, barnase–barstar, TYK2 (1 µs) | published |
| **REST2** | alanine dipeptide, paracetamol, chignolin, TYK2 (selective) | published |
| **AIS** | alanine dipeptide, paracetamol; TYK2 as a measured LIMIT | published |
| **Umbrella sampling** | alanine dipeptide (φ), protein–ligand and protein–protein (centre-of-mass distance) | page planned, 0.6.5 |
| **Alchemical — decoupling** | [ethanol](ethanol/hydration.md) (hydration free energy) | published |
| **Alchemical — relative, ligand to ligand** | [ethane → chloroethane](ethane-chloroethane/relative.md); protein-ligand planned | published |

A planned row is listed so the shape of the set is visible; it is never linked to a page that does
not exist. `Alchemical — decoupling` removes a molecule from its surroundings and is published;
turning one ligand INTO another is a different plan and a different page, and it is not written.

Run on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.

| page | version it was run against |
|---|---|
| [ethanol / hydration free energy](ethanol/hydration.md) | `0.6.4` |
| [ethane → chloroethane / relative hydration](ethane-chloroethane/relative.md) | `0.6.4` |
| [paracetamol / cMD](paracetamol/cMD.md) | `0.6.1` |
| [paracetamol / REST2](paracetamol/REST2.md) | `0.6.1` |
| [paracetamol / AIS](paracetamol/AIS.md) | `0.6.1` |
| [chignolin / cMD](chignolin/cMD.md) | `0.6.1` |
| [chignolin / REST2](chignolin/REST2.md) | `0.6.1` |
| [alanine dipeptide / AIS](ALA/AIS.md) | `0.6.1` |
| [barnase + barstar / cMD](barnase-barstar/cMD.md) | `0.6.1` |
| [TYK2 + ejm_31 / selective REST2](tyk2-ejm31/REST2.md) | `0.6.1` |
| [alanine dipeptide / cMD](ALA/cMD.md) | `0.6.1` |
| [alanine dipeptide / REST2](ALA/REST2.md) | `0.6.1` |
| [TYK2 + ejm_31 / AIS](tyk2-ejm31/AIS.md) | `0.6.1` |
| [TYK2 + ejm_31 / cMD](tyk2-ejm31/cMD.md) | `0.6.1` |
| [chignolin / choosing τ_max](chignolin/choosing-tau.md) | `0.5.4`, not re-run |
