# Protein–protein complex — on hold

!!! warning "On hold since 2026-10-13"
    MD-tools is now dedicated to sampling the conformations of a **ligand**, in solvent or in a
    protein. A protein–protein complex, where the quantity of interest is the association of two
    folded chains, is outside that. Nothing here is part of the current release, and the tutorial was
    last run against **0.6.1**. See [On hold](../index.md) for what else is waiting.

## What exists

| part | what it is | last run against |
|---|---|---|
| [Barnase and barstar](tutorial/barnase-barstar/index.md) | the system: PDB 1BRS, biological assembly 3, built in explicit water | 0.6.1 |
| [cMD: barnase–barstar](tutorial/barnase-barstar/cMD.md) | 10 ns of production from the deposited crystal structure, with the analysis script | 0.6.1 |

## What stays in MD-tools

The code that builds and runs a protein complex is not removed: a protein is still built from a
`.pdb`, protonated by PROPKA, solvated and run. What is on hold is the *topic*: tutorials, planned
pages and release testing for protein–protein association. A protein carrying a ligand is the
supported case, and is not on hold.

## What would take it off hold

A decision that protein–protein sampling is back in scope, and the tutorial re-run on the
then-current release, because every number on it was produced by 0.6.1.
