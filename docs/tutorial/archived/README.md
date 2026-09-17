# Archived tutorials

Tutorials for releases the current workflow has moved past. Each page was run exactly as written
against the release it names, and is kept unchanged as a record of that release: its commands and
outputs are true for that version, and may be refused by a later one.

**For md-tools 0.5.4 or later, use the [current tutorials](../README.md).**

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
