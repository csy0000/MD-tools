"""The selective-REST2 boundary fixture (version 1): small, explicit-solvent, and built in ~2 s.

What it has to be able to express (0.6.1 S1 assignment, first task, item 2):

  * TWO CHAINS, numbered straight through, so a mask index is never a per-chain number:

        1 ACE  2 ALA  3 SER  4 PHE  5 PRO  6 GLY  7 CYX  8 NME       chain A
        9 ACE 10 CYX 11 VAL 12 NME                                    chain B
       13 LGA 14 LGB 15 LGA                                           ligands, one chain each
       16...  water and neutralising ions

  * phi/psi across a residue boundary (every residue of chain A), chi1 reaching backbone N and CA
    (SER, PHE, CYX, VAL), a proline-like omega (SER-PHE-PRO), glycine (no sidechain), caps,
    termini of both chains, an aromatic ring (PHE), and a DISULFIDE between the chains (A7-B10);
  * two copies of ONE compound (LGA, N-methylacrylamide: a double bond, an amide, an improper, and
    two rotatable bonds) with a DIFFERENT compound (LGB, 1-propanol) between them, so the copies are
    not contiguous and one can be selected while its twin is not;
  * CMAP (ff19SB), with maps SHARED between residues of one type, so a partial selection has
    to split them.

The peptide coordinates are tleap's (`tests/data/selective_rest2/peptides.pdb`, ff19SB, chain B
translated). Chain B is moved here so the two cysteine SG sit 2.04 A apart. The geometry is
not relaxed, which does not matter for what this fixture is for: energies at FROZEN coordinates,
compared between Hamiltonians. It is not a starting structure for dynamics.

The ligand force field is GENERATED from the RDKit molecule (one atom type per atom, bonded
terms from the embedded geometry, two Fourier terms per proper torsion, an improper on every
trigonal centre). It is not GAFF and does not pretend to be: its job is to give every term class
the scaler must treat -- proper, improper, 1-4 exception, ordinary pair -- a non-zero value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIXTURE_VERSION = "selective-rest2-fixture/1"
DATA = Path(__file__).resolve().parent / "data" / "selective_rest2"

#: name -> (SMILES, residue name)
LIGANDS = {"LGA": "C=CC(=O)NC", "LGB": "CCCO"}

#: One-based topology residue indices, fixed by construction. Tests name residues by these.
RESIDUE = {"A.ACE": 1, "A.ALA": 2, "A.SER": 3, "A.PHE": 4, "A.PRO": 5, "A.GLY": 6, "A.CYX": 7,
           "A.NME": 8, "B.ACE": 9, "B.CYX": 10, "B.VAL": 11, "B.NME": 12,
           "LGA#1": 13, "LGB": 14, "LGA#2": 15}


def ligand_molecule(name: str):
    """The RDKit molecule, with hydrogens, one conformer and fixed atom names (`C1`, `H3`...)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(LIGANDS[name]))
    AllChem.EmbedMolecule(mol, randomSeed=20260919)
    AllChem.MMFFOptimizeMolecule(mol)
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        atom.SetProp("_Name", f"{symbol}{counts[symbol]}")
    mol.SetProp("_Name", name)
    return mol


def atom_names(mol) -> list[str]:
    return [atom.GetProp("_Name") for atom in mol.GetAtoms()]


_LJ = {"C": (0.34, 0.36), "N": (0.325, 0.71), "O": (0.296, 0.88), "H": (0.25, 0.066)}


def ligand_ffxml(name: str, mol) -> str:
    """An OpenMM force-field XML for one ligand. See the module docstring for what it is not."""
    from rdkit.Chem import AllChem

    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()])
    charges -= charges.mean()
    names = atom_names(mol)
    positions = mol.GetConformer().GetPositions() / 10.0
    cls = [f"{name}_{n}" for n in names]

    def dist(i, j):
        return float(np.linalg.norm(positions[i] - positions[j]))

    def angle(i, j, k):
        a, b = positions[i] - positions[j], positions[k] - positions[j]
        return float(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))

    neighbours = {a.GetIdx(): sorted(n.GetIdx() for n in a.GetNeighbors()) for a in mol.GetAtoms()}
    lines = ["<ForceField>", " <AtomTypes>"]
    for atom, c in zip(mol.GetAtoms(), cls):
        lines.append(f'  <Type name="{c}" class="{c}" element="{atom.GetSymbol()}" '
                     f'mass="{atom.GetMass():.4f}"/>')
    lines += [" </AtomTypes>", " <Residues>", f'  <Residue name="{name}">']
    for n, c, q in zip(names, cls, charges):
        lines.append(f'   <Atom name="{n}" type="{c}" charge="{q:.6f}"/>')
    for bond in mol.GetBonds():
        lines.append(f'   <Bond atomName1="{names[bond.GetBeginAtomIdx()]}" '
                     f'atomName2="{names[bond.GetEndAtomIdx()]}"/>')
    lines += ["  </Residue>", " </Residues>", " <HarmonicBondForce>"]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lines.append(f'  <Bond class1="{cls[i]}" class2="{cls[j]}" length="{dist(i, j):.5f}" '
                     f'k="300000.0"/>')
    lines += [" </HarmonicBondForce>", " <HarmonicAngleForce>"]
    for j in neighbours:
        for a, i in enumerate(neighbours[j]):
            for k in neighbours[j][a + 1:]:
                lines.append(f'  <Angle class1="{cls[i]}" class2="{cls[j]}" class3="{cls[k]}" '
                             f'angle="{angle(i, j, k):.5f}" k="400.0"/>')
    lines += [" </HarmonicAngleForce>", " <PeriodicTorsionForce>"]
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l in (j, i):
                    continue
                lines.append(f'  <Proper class1="{cls[i]}" class2="{cls[j]}" class3="{cls[k]}" '
                             f'class4="{cls[l]}" periodicity1="3" phase1="0.0" k1="0.65" '
                             f'periodicity2="2" phase2="3.14159265" k2="1.10"/>')
    for centre, around in neighbours.items():
        if len(around) == 3:
            a, b, c = around
            lines.append(f'  <Improper class1="{cls[centre]}" class2="{cls[a]}" '
                         f'class3="{cls[b]}" class4="{cls[c]}" periodicity1="2" '
                         f'phase1="3.14159265" k1="4.6"/>')
    lines += [" </PeriodicTorsionForce>",
              ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">',
              '  <UseAttributeFromResidue name="charge"/>']
    for atom, c in zip(mol.GetAtoms(), cls):
        sigma, epsilon = _LJ[atom.GetSymbol()]
        lines.append(f'  <Atom type="{c}" sigma="{sigma}" epsilon="{epsilon}"/>')
    lines += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(lines) + "\n"


@dataclass
class Fixture:
    topology: object
    positions: object          # openmm Quantity, nm
    system: object             # the unscaled built System
    molecules: dict            # ligand name -> RDKit molecule
    ffxml: dict                # ligand name -> XML text

    def residue(self, one_based: int):
        return list(self.topology.residues())[one_based - 1]

    def atom(self, one_based_residue: int, name: str) -> int:
        for atom in self.residue(one_based_residue).atoms():
            if atom.name == name:
                return atom.index
        raise KeyError(f"no atom {name} in residue {one_based_residue}")

    def write(self, directory: Path, *, sdfs: bool = True) -> Path:
        """`built.xml`, `built.pdb` and (optionally) `LGA.sdf`/`LGB.sdf`, as build-top lays them out."""
        from openmm import XmlSerializer, app
        from rdkit import Chem

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "built.xml").write_text(XmlSerializer.serialize(self.system),
                                             encoding="utf-8")
        with (directory / "built.pdb").open("w") as handle:
            app.PDBFile.writeFile(self.topology, self.positions, handle, keepIds=True)
        if sdfs:
            for name, mol in self.molecules.items():
                Chem.MolToMolFile(mol, str(directory / f"{name}.sdf"))
        return directory


def _place(positions_nm: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return positions_nm - positions_nm.mean(axis=0) + centre


#: Protein force fields the fixture can be built with. ff19SB carries CMAP and NO periodic
#: torsion across N-CA or CA-C (phi and psi live entirely in its CMAP); ff14SB has no CMAP and
#: does carry them. A backbone selection is only fully exercised by both.
FORCEFIELDS = {"ff19SB": ("amber19/protein.ff19SB.xml", "amber19/tip3p.xml"),
               "ff14SB": ("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")}


def build_fixture(*, padding_nm: float = 0.6, ligand_order=("LGA", "LGB", "LGA"),
                  reverse_lga_atoms: bool = False, forcefield: str = "ff19SB") -> Fixture:
    """Build the fixture from the committed peptide coordinates. Deterministic.

    `ligand_order` and `reverse_lga_atoms` build the REORDERED variants: the same chemistry with
    different residue order, or with every LGA's atoms in reverse order inside the residue. The
    names travel with the atoms, which is what a package-local identity relies on.
    """
    from openmm import app, unit
    from io import StringIO

    pdb = app.PDBFile(str(DATA / "peptides.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # The disulfide: CYS -> CYX without HG, chain B translated so SG-SG is 2.04 A, and the bond.
    cys = [r for r in modeller.topology.residues() if r.name == "CYS"]
    assert len(cys) == 2, "the committed peptides carry exactly two cysteines"
    modeller.delete([a for r in cys for a in r.atoms() if a.name == "HG"])
    xyz = np.array(modeller.positions.value_in_unit(unit.nanometer))
    cys = [r for r in modeller.topology.residues() if r.name == "CYS"]
    sg = [next(a.index for a in r.atoms() if a.name == "SG") for r in cys]
    cb = [next(a.index for a in r.atoms() if a.name == "CB") for r in cys]
    outward = xyz[sg[0]] - xyz[cb[0]]
    outward /= np.linalg.norm(outward)
    shift = xyz[sg[0]] + 0.204 * outward - xyz[sg[1]]
    chain_b = [a.index for a in list(modeller.topology.chains())[1].atoms()]
    xyz[chain_b] += shift
    for residue in cys:
        residue.name = "CYX"
    atoms = list(modeller.topology.atoms())
    modeller.topology.addBond(atoms[sg[0]], atoms[sg[1]])
    modeller.positions = unit.Quantity([tuple(p) for p in xyz], unit.nanometer)

    # The ligands, each its own chain: LGA, LGB, LGA.
    molecules = {name: ligand_molecule(name) for name in LIGANDS}
    if reverse_lga_atoms:
        from rdkit import Chem

        count = molecules["LGA"].GetNumAtoms()
        molecules["LGA"] = Chem.RenumberAtoms(molecules["LGA"], list(range(count - 1, -1, -1)))
    ffxml = {name: ligand_ffxml(name, mol) for name, mol in molecules.items()}
    centre = xyz.mean(axis=0)
    spots = [np.array([1.3, 0.0, 0.0]), np.array([0.0, -1.3, 0.0]), np.array([-1.3, 0.0, 0.6])]
    for name, spot in zip(ligand_order, spots):
        mol = molecules[name]
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue(name, chain)
        added = []
        for atom, atom_name in zip(mol.GetAtoms(), atom_names(mol)):
            added.append(top.addAtom(atom_name, app.Element.getBySymbol(atom.GetSymbol()),
                                     residue))
        for bond in mol.GetBonds():
            top.addBond(added[bond.GetBeginAtomIdx()], added[bond.GetEndAtomIdx()])
        where = _place(mol.GetConformer().GetPositions() / 10.0, centre + spot)
        modeller.add(top, unit.Quantity([tuple(p) for p in where], unit.nanometer))

    field = app.ForceField(*FORCEFIELDS[forcefield], *[StringIO(text) for text in ffxml.values()])
    modeller.addSolvent(field, model="tip3p", padding=padding_nm * unit.nanometer)
    system = field.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=0.9 * unit.nanometer,
                                     constraints=app.HBonds, rigidWater=True)
    return Fixture(topology=modeller.topology, positions=modeller.positions, system=system,
                   molecules=molecules, ffxml=ffxml)
