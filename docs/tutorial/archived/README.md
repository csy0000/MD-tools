# Archived tutorials

Tutorials for releases the current workflow has moved past. Each page was run exactly as written
against the release it names, and is kept unchanged as a record of that release: its commands and
outputs are true for that version, and may be refused by a later one.

**Use the [current tutorials](../README.md)** for the current package.

## 0.6.4

Archived because the default clustering workflow changed: `t_hdbscan` now enumerates the
molecule's symmetry BEFORE clustering and measures distances as the minimum over symmetry
relabellings, and frames are assigned by an absolute 18-of-20 vote. The page below clustered on the
plain torsion distance with the historical 15-neighbour margin vote and merged symmetry-related
clusters afterwards (`TorsionSymmetry`). Its saved outputs are those of commit `8516c12a` (tag
`v0.6.4`), not re-executed; only the helper import path and the install link were updated. On the current package it is
reproduced with `t_hdbscan(tors, symmetry=False, vote_rule="legacy-margin")`.

| tutorial | workflow |
|---|---|
| [paracetamol clustering](0.6.4/paracetamol/clustering.ipynb) | the historical cluster-then-merge workflow |

## 0.5.3

Archived because 0.5.4 changed how a scaled Hamiltonian comes to exist and how a run is told which
one to integrate:

* REST2 states are built once, as files, by `md-openmm build-top --rest2-scaler`, with a record
  (`scaler.yaml`) and a picture of what stays unscaled. In 0.5.3 every run scaled in memory.
* A stage and a ladder no longer scale anything: a hot cMD stage takes its saved state as `-s`, and
  a REST2 ladder reads `-s` only from its group file, one saved state per line.
* More torsions stay unscaled in every state: aromatic ring bonds, other double bonds and every
  improper, as well as the ordinary amide omega.

| tutorial | method |
|---|---|
| [paracetamol](0.5.3/cMD/paracetamol.md) | cMD |
| [Chinolin](0.5.3/cMD/chinolin.md) | cMD |
| [paracetamol](0.5.3/REST2/paracetamol.md) | REST2 -- including the 0.5.3 workaround for `run.sh` refusing the ladder |
