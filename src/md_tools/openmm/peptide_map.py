"""Read peptide chemistry off a molecular graph, for a solute built as one whole molecule.

WHY THIS EXISTS

    A cyclic peptide built from SMILES is parameterised as a single molecule -- one residue, one
    name, no residue evidence at all. That is correct for the force field: Sage sees a molecule
    and assigns valence and charges to it. It is wrong for everything that is keyed on RESIDUE
    identity, and the sharp case is `mbondi3`, whose Arg/Asp/Glu radius corrections are selected
    by residue and atom NAME. Against a single `RGD`/`UNL` residue those rules match nothing, the
    radii silently reduce to mbondi2, and the build still says "mbondi3".

    So the chemistry is read from the GRAPH instead: what a residue IS, not what it is called.

WHAT IS NOT DONE HERE

    No parameterisation. This module never selects a force field, never assigns a charge and
    never builds a System. It produces a MAP -- which atoms form which residue, in what
    protonation state, with what stereochemistry -- and callers use that map to do things that
    need to know. `peptide-like` is the ligand route plus this map, not a third route.

WHAT IS REFUSED

    Everything ambiguous. A partial map that names four residues out of five and shrugs at the
    fifth is worse than no map, because the corrections it drives would be applied to some of the
    molecule and silently not to the rest. So: an unambiguous head-to-tail alpha-peptide cycle,
    canonical side chains, resolved stereochemistry, or a refusal that says which residue and
    why. The generic `ligand` route remains available and is the honest answer for a solute this
    cannot describe.

L AND D ARE NOT S AND R

    For a residue built on a CH(N)(C=O)(R) centre the CIP code and the L/D label track each other
    -- L is S -- because the substituent priorities fall the same way every time. Cysteine is the
    exception: its sulfur outranks the carbonyl carbon, the priorities swap, and L-Cys is R. A
    mapper that reported a raw CIP letter as though it were an L/D label would call L-Cys a
    D-residue, and would be believed. So handedness is derived per residue, from a table.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

#: Backbone atoms every residue contributes, in the order they are recorded.
BACKBONE = ("N", "CA", "C", "O")

#: Canonical side chains, as the SMILES of the side-chain fragment with a `*` marking where it
#: attaches to the alpha carbon. Compared as CANONICAL SMILES of an assembled fragment, so the
#: comparison is on the chemical graph and cannot be fooled by atom order, atom names or the
#: order the input happened to be written in.
#:
#: Protonation is part of the identity, not an afterthought: `ASP` and `ASH` are different
#: entries because they are different molecules and only one of them gets the mbondi3 carboxylate
#: correction.
SIDE_CHAINS: dict[str, str] = {
    "GLY": "[H]*",
    "ALA": "C*",
    "VAL": "CC(C)*",
    "LEU": "CC(C)C*",
    "ILE": "CCC(C)*",
    "MET": "CSCC*",
    "PHE": "c1ccc(cc1)C*",
    "TRP": "c1ccc2c(c1)c(c[nH]2)C*",
    "SER": "OC*",
    "THR": "CC(O)*",
    "ASN": "NC(=O)C*",
    "GLN": "NC(=O)CC*",
    # Charged and neutral forms, kept apart deliberately.
    "ASP": "[O-]C(=O)C*",
    "ASH": "OC(=O)C*",
    "GLU": "[O-]C(=O)CC*",
    "GLH": "OC(=O)CC*",
    "LYS": "[NH3+]CCCC*",
    "LYN": "NCCCC*",
    "ARG": "NC(=[NH2+])NCCC*",
    "ARN": "NC(=N)NCCC*",
    "CYS": "SC*",
    "CYM": "[S-]C*",
    "TYR": "Oc1ccc(cc1)C*",
    "TYM": "[O-]c1ccc(cc1)C*",
    "HID": "c1c([nH]cn1)C*",
    "HIE": "c1c(nc[nH]1)C*",
    "HIP": "c1c([nH]c[nH+]1)C*",
}

#: Proline is not in the table above because its side chain closes back onto the backbone
#: nitrogen; it is identified structurally instead. Its ring makes the amide nitrogen tertiary,
#: which is exactly why the omega policy treats it differently.
PROLINE = "PRO"

#: Residues whose L form is R rather than S. Sulfur outranks the carbonyl carbon at the alpha
#: centre, so the CIP priorities swap while the geometry does not.
_L_IS_R = {"CYS", "CYM"}

#: Which residues carry the charged groups mbondi3 corrects, and how to find those atoms.
#: Keyed on the mapped identity, so a neutral variant is simply absent and cannot be corrected
#: by accident.
CARBOXYLATE_RESIDUES = ("ASP", "GLU")
GUANIDINIUM_RESIDUES = ("ARG",)


class PeptideMapError(ValueError):
    """This solute is not an unambiguous cyclic peptide. Says which residue, and why."""


@dataclass
class MappedResidue:
    """One residue, in the atom indices of the molecule that was mapped."""

    position: int
    residue: str
    handedness: str | None
    n: int
    ca: int
    c: int
    o: int
    side_chain: tuple[int, ...]
    #: Every atom of the residue, backbone and side chain, hydrogens included.
    atoms: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"position": self.position, "residue": self.residue,
                "handedness": self.handedness,
                "N": self.n, "CA": self.ca, "C": self.c, "O": self.o,
                "side_chain": list(self.side_chain), "atoms": list(self.atoms)}


@dataclass
class PeptideMap:
    """A validated head-to-tail cyclic peptide, expressed in one molecule's atom indices."""

    residues: tuple[MappedResidue, ...]
    #: `(carbonyl C, amide N)` for each backbone link, cyclically, closure included.
    links: tuple[tuple[int, int], ...]
    n_atoms: int
    formal_charge: int
    #: Atom indices whose GB radius mbondi3 would correct, by rule.
    carboxylate_oxygens: tuple[int, ...] = ()
    guanidinium_hydrogens: tuple[int, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(r.residue for r in self.residues)

    def digest(self) -> str:
        """A stable content digest of the map, for provenance.

        Over the chemistry and the atom sets, so two builds of the same molecule in the same atom
        order agree and any change to either shows up.
        """
        payload = json.dumps({
            "residues": [r.as_dict() for r in self.residues],
            "links": [list(link) for link in self.links],
            "n_atoms": self.n_atoms,
            "formal_charge": self.formal_charge,
            "carboxylate_oxygens": list(self.carboxylate_oxygens),
            "guanidinium_hydrogens": list(self.guanidinium_hydrogens),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "residues": [r.as_dict() for r in self.residues],
            "links": [list(link) for link in self.links],
            "n_atoms": self.n_atoms,
            "formal_charge": self.formal_charge,
            "carboxylate_oxygens": list(self.carboxylate_oxygens),
            "guanidinium_hydrogens": list(self.guanidinium_hydrogens),
            "digest": self.digest(),
            "notes": list(self.notes),
        }

    def torsions(self) -> list[dict[str, Any]]:
        """phi, psi and omega for every residue, cyclically indexed.

        With no chain terminus every torsion is defined, so a five-residue cycle yields fifteen.
        A linear peptide would lose phi of the first residue and psi of the last, and quietly
        reporting thirteen where fifteen were asked for is the error the cyclic indexing exists
        to prevent.
        """
        n = len(self.residues)
        out: list[dict[str, Any]] = []
        for i, residue in enumerate(self.residues):
            previous, following = self.residues[(i - 1) % n], self.residues[(i + 1) % n]
            name = f"{residue.residue}{i}"
            out.append({"name": f"phi_{name}", "kind": "phi",
                        "atoms": [previous.c, residue.n, residue.ca, residue.c]})
            out.append({"name": f"psi_{name}", "kind": "psi",
                        "atoms": [residue.n, residue.ca, residue.c, following.n]})
            out.append({"name": f"omega_{name}", "kind": "omega",
                        "atoms": [residue.ca, residue.c, following.n, following.ca]})
        return out


def _freeze(mol, atoms, attachment=None) -> str:
    """Canonical SMILES of a heavy-atom skeleton with hydrogen counts FROZEN onto it.

    One function for both sides of the comparison. The pattern table and the molecule being
    mapped are put through exactly this, so they cannot differ by a rendering convention -- an
    earlier version compared a bracketed `*[C]([C])[C]` against an implicit-hydrogen `*C(C)C` and
    matched nothing while both described isopropyl.

    Hydrogens are counted with `includeNeighbors=True`, because in a prepared SDF they are real
    ATOMS rather than a per-atom count; without it every heavy atom reports zero hydrogens and a
    carboxylate becomes indistinguishable from a carboxylic acid.
    """
    from rdkit import Chem

    heavy = [i for i in sorted(atoms) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    builder = Chem.RWMol()
    mapping: dict[int, int] = {}
    for index in heavy:
        source = mol.GetAtomWithIdx(index)
        atom = Chem.Atom(source.GetAtomicNum())
        atom.SetFormalCharge(source.GetFormalCharge())
        atom.SetNumExplicitHs(source.GetTotalNumHs(includeNeighbors=True))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(source.GetIsAromatic())
        mapping[index] = builder.AddAtom(atom)
    dummy = None
    if attachment is not None:
        dummy = builder.AddAtom(Chem.Atom(0))
    for a in heavy:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            b = bond.GetOtherAtomIdx(a)
            if b in mapping and b > a:
                builder.AddBond(mapping[a], mapping[b], bond.GetBondType())
            elif dummy is not None and b == attachment:
                builder.AddBond(mapping[a], dummy, Chem.BondType.SINGLE)
    fragment = builder.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except Exception as broken:                                    # noqa: BLE001
        raise PeptideMapError(f"a side chain could not be read as a molecule ({broken})") from None
    return Chem.MolToSmiles(fragment)


def _canonical(smiles: str) -> str:
    """A declared side-chain pattern, put through the same normalisation as a real fragment."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise PeptideMapError(f"internal: side-chain pattern {smiles!r} does not parse")
    mol = Chem.AddHs(mol)
    return _freeze(mol, range(mol.GetNumAtoms()))


def _side_chain_table() -> dict[str, str]:
    """Canonical SMILES -> residue name, built once from the declared patterns."""
    table: dict[str, str] = {}
    for name, smiles in SIDE_CHAINS.items():
        key = _canonical(smiles)
        if key in table:
            raise PeptideMapError(
                f"internal: {name} and {table[key]} have the same canonical side chain {key!r}")
        table[key] = name
    return table


def _fragment_smiles(mol, atoms, attachment) -> str:
    """The side chain as canonical SMILES, with a `*` where it joins the alpha carbon.

    The dummy matters: without it leucine and isoleucine are the same four carbons, and the
    attachment point is the only thing that tells them apart.
    """
    return _freeze(mol, atoms, attachment)


def _hydrogens_of(mol, index) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(index).GetNeighbors()
            if n.GetAtomicNum() == 1]


def map_cyclic_peptide(mol) -> PeptideMap:
    """Read a head-to-tail cyclic peptide off an RDKit molecule with explicit hydrogens.

    Raises `PeptideMapError` with a specific reason for anything it cannot describe completely.
    """
    from rdkit import Chem

    if mol is None:
        raise PeptideMapError("no molecule was given")
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception as broken:                                    # noqa: BLE001
        raise PeptideMapError(f"the molecule does not sanitise ({broken})") from None
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) is False:
        raise PeptideMapError("the molecule has no explicit hydrogens; the map needs them to "
                              "identify protonation states and the corrected hydrogens")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    if len(Chem.GetMolFrags(mol)) != 1:
        raise PeptideMapError(
            f"the input is {len(Chem.GetMolFrags(mol))} disconnected fragments; a cyclic peptide "
            f"is one connected molecule")

    # -- the backbone amide links -------------------------------------------------------------
    amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
    candidates = []
    for carbon, oxygen, nitrogen in mol.GetSubstructMatches(amide):
        candidates.append((carbon, oxygen, nitrogen))
    if not candidates:
        raise PeptideMapError("no amide bond was found; this is not a peptide")

    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]

    # -- alpha carbons --------------------------------------------------------------------------
    # An alpha carbon is bonded to an amide nitrogen AND to the carbonyl carbon of the NEXT
    # link. Identified from the graph, never from a name.
    carbonyls = {c: (o, n) for c, o, n in candidates}
    nitrogens = {n for _c, _o, n in candidates}

    alphas: dict[int, tuple[int, int]] = {}                # CA -> (N, C)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        index = atom.GetIdx()
        neighbour_n = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in nitrogens]
        neighbour_c = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in carbonyls]
        if len(neighbour_n) == 1 and len(neighbour_c) == 1:
            alphas[index] = (neighbour_n[0], neighbour_c[0])
    if not alphas:
        raise PeptideMapError(
            "no alpha carbon was found: no carbon lies between an amide nitrogen and an amide "
            "carbonyl carbon. A LINEAR peptide reaches this branch, because its terminal "
            "residues are bounded by a free amine and a free acid rather than by amide bonds on "
            "both sides. Only an unambiguous head-to-tail cycle is supported; use "
            "solute.kind: ligand for anything else.")

    # -- the backbone cycle ---------------------------------------------------------------------
    # Walk N -> CA -> C -> N. A head-to-tail cycle returns to its start having visited every
    # alpha carbon exactly once; anything else is refused rather than partially described.
    start = min(alphas)
    order: list[int] = []
    seen: set[int] = set()
    current = start
    while current not in seen:
        seen.add(current)
        order.append(current)
        _n, carbon = alphas[current]
        _oxygen, next_n = carbonyls[carbon]
        following = [a for a in alphas if alphas[a][0] == next_n]
        if len(following) != 1:
            raise PeptideMapError(
                f"the backbone does not continue unambiguously from the carbonyl of atom "
                f"{carbon}: {len(following)} alpha carbons follow amide nitrogen {next_n}")
        current = following[0]
    if current != start:
        raise PeptideMapError(
            "the backbone walk did not return to its start, so this is not a head-to-tail cycle. "
            "Linear peptides are not supported by this map; use solute.kind: ligand.")
    if len(order) != len(alphas):
        missing = sorted(set(alphas) - set(order))
        raise PeptideMapError(
            f"the backbone cycle covers {len(order)} of {len(alphas)} alpha carbons; "
            f"{missing} are not on it. An unambiguous single cycle is required.")
    if len(order) < 2:
        raise PeptideMapError("a cyclic peptide needs at least two residues")

    # -- per residue ----------------------------------------------------------------------------
    backbone_atoms: set[int] = set()
    for ca in order:
        n, c = alphas[ca]
        backbone_atoms.update({n, c, ca, carbonyls[c][0]})

    table = _side_chain_table()
    residues: list[MappedResidue] = []
    for position, ca in enumerate(order):
        n, c = alphas[ca]
        oxygen = carbonyls[c][0]

        # The side chain: everything reachable from the alpha carbon that is not backbone. The
        # walk is BLOCKED at backbone atoms, which is what keeps proline's ring -- whose CD is
        # bonded to the backbone nitrogen -- from wandering into the next residue.
        side: set[int] = set()
        stack = [x.GetIdx() for x in mol.GetAtomWithIdx(ca).GetNeighbors()
                 if x.GetIdx() not in backbone_atoms]
        while stack:
            index = stack.pop()
            if index in side or index in backbone_atoms:
                continue
            side.add(index)
            stack.extend(x.GetIdx() for x in mol.GetAtomWithIdx(index).GetNeighbors()
                         if x.GetIdx() not in side and x.GetIdx() not in backbone_atoms)

        heavy_side = {i for i in side if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
        # Proline: the side chain closes back onto this residue's own backbone nitrogen.
        closes_on_n = any(mol.GetBondBetweenAtoms(i, n) is not None for i in heavy_side)

        if closes_on_n:
            if len(heavy_side) != 3:
                raise PeptideMapError(
                    f"residue {position}: the side chain closes onto its own backbone nitrogen "
                    f"but has {len(heavy_side)} heavy atoms; only proline's three-carbon ring is "
                    f"supported")
            name = PROLINE
        else:
            # Glycine has no side chain at all: the alpha carbon carries two hydrogens and
            # nothing else, so there are no heavy atoms to build a fragment from.
            fragment = _canonical("[H]*") if not heavy_side \
                else _fragment_smiles(mol, side, ca)
            name = table.get(fragment)
            if name is None:
                raise PeptideMapError(
                    f"residue {position} (alpha carbon {ca}) has side chain {fragment!r}, which "
                    f"is not a canonical amino-acid side chain in a recognised protonation "
                    f"state. Modified or non-standard side chains are not supported by this map; "
                    f"use solute.kind: ligand for this solute.")

        # -- stereochemistry ---------------------------------------------------------------------
        cip = mol.GetAtomWithIdx(ca).GetPropsAsDict().get("_CIPCode")
        if name == "GLY":
            handedness = None
            if cip is not None:
                raise PeptideMapError(
                    f"residue {position} is glycine but its alpha carbon carries a CIP code "
                    f"{cip!r}; glycine has two hydrogens there and is achiral")
        else:
            if cip is None:
                raise PeptideMapError(
                    f"residue {position} ({name}, alpha carbon {ca}) has unspecified "
                    f"stereochemistry. A peptide-like solute must state it: the L and D forms "
                    f"are different molecules and the map will not guess.")
            if name in _L_IS_R:
                handedness = {"R": "L", "S": "D"}[cip]
            else:
                handedness = {"S": "L", "R": "D"}[cip]

        atoms = tuple(sorted({n, ca, c, oxygen} | side
                             | set(_hydrogens_of(mol, n)) | set(_hydrogens_of(mol, ca))))
        residues.append(MappedResidue(
            position=position, residue=name, handedness=handedness,
            n=n, ca=ca, c=c, o=oxygen,
            side_chain=tuple(sorted(side)), atoms=atoms))

    links = tuple((residues[i].c, residues[(i + 1) % len(residues)].n)
                  for i in range(len(residues)))
    for carbon, nitrogen in links:
        if mol.GetBondBetweenAtoms(carbon, nitrogen) is None:
            raise PeptideMapError(f"no bond between carbonyl {carbon} and amide nitrogen "
                                  f"{nitrogen}: the cycle is not closed")

    mapped = PeptideMap(
        residues=tuple(residues), links=links,
        n_atoms=mol.GetNumAtoms(),
        formal_charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()),
    )
    mapped.carboxylate_oxygens, mapped.guanidinium_hydrogens = _correctable_atoms(mol, mapped)
    return mapped


def _correctable_atoms(mol, mapped: PeptideMap) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The atoms mbondi3 corrects, selected by CHEMISTRY rather than by residue and atom name.

    ParmEd's `mbondi3` selects `OD*`/`OE*` in GLU/ASP/GL4/AS4 and `HH*`/`HE*` in ARG. Those names
    do not exist here -- the solute is one residue -- so the equivalent sets are found from the
    mapped chemistry: the two oxygens of a DEPROTONATED side-chain carboxylate, and the hydrogens
    on the three nitrogens of a PROTONATED guanidinium.

    Deliberately stricter than the name rule in one respect: ParmEd also corrects `AS4`/`GL4`,
    which are the protonated variants, because it matches on the name. Here `ASH`/`GLH` map to
    their own identities and are not corrected, which is what "do not apply charged-group
    corrections to neutral protonation variants" requires.
    """
    from rdkit import Chem

    oxygens: list[int] = []
    hydrogens: list[int] = []
    for residue in mapped.residues:
        if residue.residue in CARBOXYLATE_RESIDUES:
            side = set(residue.side_chain)
            found: list[int] = []
            for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=[OX1])[OX1-]")):
                carbon, double_o, single_o = match
                if {carbon, double_o, single_o} <= side:
                    found.extend((double_o, single_o))
            if len(found) != 2:
                raise PeptideMapError(
                    f"residue {residue.position} maps as {residue.residue} but its side chain "
                    f"presents {len(found)} carboxylate oxygens rather than two")
            oxygens.extend(sorted(found))
        if residue.residue in GUANIDINIUM_RESIDUES:
            side = set(residue.side_chain)
            found_h: set[int] = set()
            for match in mol.GetSubstructMatches(
                    Chem.MolFromSmarts("[NX3][CX3](=[NX3+])[NX3]")):
                if set(match) <= side:
                    for nitrogen in (match[0], match[2], match[3]):
                        found_h.update(_hydrogens_of(mol, nitrogen))
            if not found_h:
                raise PeptideMapError(
                    f"residue {residue.position} maps as {residue.residue} but no protonated "
                    f"guanidinium was found in its side chain")
            hydrogens.extend(sorted(found_h))
    return tuple(sorted(oxygens)), tuple(sorted(hydrogens))


def map_from_sdf(path) -> PeptideMap:
    """The map for a prepared SDF, which is where bond orders and formal charges survive."""
    from pathlib import Path

    from rdkit import Chem

    path = Path(path)
    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise PeptideMapError(f"{path}: RDKit could not read this SDF")
    if mol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(mol)
    return map_cyclic_peptide(mol)
