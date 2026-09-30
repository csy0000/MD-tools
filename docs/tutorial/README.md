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
| [**Alanine dipeptide**](ALA/index.md) | the toy: two torsions with barriers a plain run crosses, so a method can be checked against the truth | cMD, REST2, AIS, umbrella |
| [**Paracetamol**](paracetamol/index.md) | a ligand on its own: parameterise once, reuse everywhere | cMD, REST2, AIS |
| [**Chignolin**](chignolin/index.md) | a folding peptide: a real equilibrium, small enough to compare methods in a day | cMD, REST2 |
| [**Barnase + barstar**](barnase-barstar/index.md) | a protein–protein interface: the coordinate is between two molecules | cMD |
| [**TYK2 + ejm_31**](tyk2-ejm31/index.md) | a kinase with a real inhibitor: where a hot region is worth CHOOSING | cMD, selective REST2, AIS |

## The methods, and where they are

| method | systems | state |
|---|---|---|
| **cMD** | alanine dipeptide, paracetamol, chignolin, barnase–barstar, TYK2 (1 µs) | published |
| **REST2** | alanine dipeptide, paracetamol, chignolin, TYK2 (selective) | published |
| **AIS** | alanine dipeptide, paracetamol; TYK2 as a measured LIMIT | published |
| **Umbrella sampling** | [alanine dipeptide (φ)](ALA/umbrella.md); protein–ligand and protein–protein (centre-of-mass distance) still planned | **partly published** — see below |
| **Alchemical — TI and FEP** | every system | planned, 0.7.0 |

The alchemical row is listed so the shape of the set is visible. It is not written, and it is not
linked to a page that does not exist.

**The umbrella page is this set's one exception to "every number comes from a run."** Its build,
its resolved stages, its generated tree, its refusals and its window digests are measured — on the
CPU, which for a 22-atom implicit system is a legitimate way to check that commands work, and
which the page says plainly. Its per-window means, the window overlap and the PMF are **not**
measured: those need CUDA, and a CPU run is never offered here as CUDA evidence. The page marks
that gap in a section of its own instead of filling it, because a tutorial that quietly published
unmeasured numbers would be worse than one that says which half is missing.

Run on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.

| page | version it was run against |
|---|---|
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
