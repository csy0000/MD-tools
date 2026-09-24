"""4A9K tutorial analysis: protein CA RMSD, paracetamol heavy-atom RMSD and pocket contacts."""
import mdtraj as md
import numpy as np

traj = md.load("cMD-run1/solute_prod1.nc", top="build/built.solute.pdb")
ref = md.load("build/built.solute.pdb")
top = traj.topology
protein = top.select("protein and name CA")
ligand = [a.index for a in top.atoms if a.residue.name == "TYL" and a.element.symbol != "H"]
# The ligand can sit in a different periodic image from the protein: make molecules whole and put
# the ligand in the image nearest the protein before measuring anything between them.
prot_atoms = {a for a in top.atoms if a.residue.is_protein}
lig_atoms = {top.atom(i) for i in ligand}
traj.image_molecules(inplace=True, anchor_molecules=[prot_atoms], other_molecules=[lig_atoms],
                     make_whole=True)
traj.superpose(ref, atom_indices=protein)
ca = md.rmsd(traj, ref, atom_indices=protein) * 10
lig = np.sqrt(((traj.xyz[:, ligand] - ref.xyz[0, ligand]) ** 2).sum(axis=2).mean(axis=1)) * 10
print(f"frames {traj.n_frames}, time {traj.time[0]:.0f}-{traj.time[-1]:.0f} ps")
print(f"protein CA RMSD (A): mean {ca.mean():.2f}, last 1 ns {ca[-100:].mean():.2f}, max {ca.max():.2f}")
print(f"paracetamol heavy-atom RMSD after protein alignment (A): mean {lig.mean():.2f}, "
      f"last 1 ns {lig[-100:].mean():.2f}, max {lig.max():.2f}")
heavy = [a.index for a in top.atoms if a.residue.is_protein and a.element.symbol != "H"]
pairs = np.array([[i, j] for i in ligand for j in heavy])
d = md.compute_distances(traj, pairs)
close = d < 0.40
print(f"protein heavy atoms within 4 A of the ligand: start "
      f"{int((md.compute_distances(ref, pairs)[0] < 0.40).sum())}, mean {close.sum(axis=1).mean():.0f}, "
      f"last 1 ns {close.sum(axis=1)[-100:].mean():.0f}")
residues = {}
for k, (i, j) in enumerate(pairs):
    if close[:, k].mean() > 0.5:
        r = top.atom(j).residue
        residues[f"{r.name}{r.resSeq}"] = max(residues.get(f"{r.name}{r.resSeq}", 0), close[:, k].mean())
print("residues in contact for more than half the run:",
      ", ".join(f"{k} ({v:.0%})" for k, v in sorted(residues.items(), key=lambda x: -x[1])))
