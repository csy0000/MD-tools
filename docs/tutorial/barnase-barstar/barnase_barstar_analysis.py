#!/usr/bin/env python
"""1BRS tutorial analysis: complex and per-chain CA RMSD, and interface contacts, over 10 ns.

Run from the dataset root (the directory holding build/ and cMD-run1/). Needs mdtraj and numpy.
"""
import mdtraj as md
import numpy as np

traj = md.load("cMD-run1/solute_prod1.nc", top="build/built.solute.pdb")
ref = md.load("build/built.solute.pdb")
top = traj.topology
# The solute trajectory is written as the periodic simulation holds it: the two chains can sit in
# different periodic images. Make each molecule whole and put barstar in the image nearest barnase
# before measuring anything BETWEEN the chains.
proteins = [set(c.atoms) for c in top.chains if c.n_residues > 20]
traj.image_molecules(inplace=True, anchor_molecules=[proteins[0]], other_molecules=[proteins[1]],
                     make_whole=True)
ca = top.select("name CA")
chains = {c.index: [a.index for a in c.atoms if a.name == "CA"] for c in top.chains if c.n_residues > 20}
traj.superpose(ref, atom_indices=ca)
rmsd = md.rmsd(traj, ref, atom_indices=ca) * 10
print(f"frames {traj.n_frames}, time {traj.time[0]:.0f}-{traj.time[-1]:.0f} ps")
print(f"complex CA RMSD (A): mean {rmsd.mean():.2f}, last 1 ns {rmsd[-100:].mean():.2f}, max {rmsd.max():.2f}")
for i, idx in chains.items():
    t = traj[:]
    t.superpose(ref, atom_indices=idx)
    r = md.rmsd(t, ref, atom_indices=idx) * 10
    print(f"chain index {i} ({len(idx)} CA): CA RMSD mean {r.mean():.2f} A, last 1 ns {r[-100:].mean():.2f} A")
(a, b) = [c for c in top.chains if c.n_residues > 20][:2]
ha = [x.index for x in a.atoms if x.element.symbol != "H"]
hb = [x.index for x in b.atoms if x.element.symbol != "H"]
pairs = np.array([[i, j] for i in ha for j in hb])
d = md.compute_distances(traj, pairs)
contacts = (d < 0.45).sum(axis=1)
d0 = md.compute_distances(ref, pairs)[0]
native = d0 < 0.45
kept = ((d < 0.45) & native).sum(axis=1) / native.sum()
print(f"interface heavy-atom contacts <4.5 A: start {int((d0 < 0.45).sum())}, mean {contacts.mean():.0f}, last 1 ns {contacts[-100:].mean():.0f}")
print(f"fraction of initial contacts kept: mean {kept.mean():.2f}, last 1 ns {kept[-100:].mean():.2f}")
res_a = {top.atom(i).residue for i, j in pairs[native]}
print("interface barnase residues (start):", len(res_a))
