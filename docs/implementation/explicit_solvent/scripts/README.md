# `scripts/` — the explicit-solvent stages

Specification and every scientific decision: **`../baseline_setups.md`**.
Implementation: **`md_templates.openmm`** — these scripts only parse arguments and
call it, per the project's code-boundary rule.

```bash
conda activate md-templates   # REQUIRED: AM1BCC needs AmberTools' sqm on PATH
```

| stage | script | in | out |
|---|---|---|---|
| — | `generate_config.py --all --system alanine\|macrocycle` | — | the four `*-config.json` |
| (a) | `simbox-setup.py --smiles "$SMILES" \| --pdb $pdb --out-suffix S --config simbox-config.json` | SMILES or PDB | `S_system.xml`, `S_topology.pdb`, `S_simbox.json` |
| (b) | `min-eq.py --p S_system.xml --c S_topology.pdb --out-suffix E --config min-eq-config.json` | box | `E_state.xml` (+ pdb, csv, json) |
| (c) | `md.py --p S_system.xml --c E_state.xml --out-suffix M --config md-cold-config.json` | box + state | `M/chunk_0000../`, `M_md.json` |
| (d) | `md_REST2.py --p S_system.xml --c E_state.xml --out-suffix R --config rest2-config.json` | box + state | `R/replica_00../`, `R_exchange_attempts.csv` |
| (e) | `md_cBAR.py`, `pREST2.py` | — | **not built yet** |

`run_all.sh <alanine|macrocycle> "<SMILES>" [structure.pdb]` runs (a) and (b), then launches
(c) cold, (c) hot and (d) one per GPU.

## The two flags

* **`--p`** names the parameterised `System` — the analogue of an Amber prmtop. A `System` XML
  carries parameters but no atom or residue names, so the topology is read from the sibling
  `<stem>_topology.pdb` that stage (a) wrote next to it, along with `<stem>_simbox.json` (solute
  atom count, omega bonds). Keep the three together.
* **`--c`** names coordinates: a `.pdb`, or a serialised OpenMM `State` `.xml`, which also carries
  velocities and the box. Stage (b) emits the latter, so (c) and (d) inherit equilibrated velocities
  rather than redrawing them.

Deliberately *not* a checkpoint: a checkpoint is only valid for the exact `System` that wrote it,
and the equilibration `System` carries a barostat and a restraint force that a production `System`
does not.

## Configs

Each stage's `--config` is a JSON of values that **differ** from
`md_templates.openmm.DEFAULTS`; `config_defaults.json` is that tree dumped in full.
All four files are slices of the same schema, so nothing drifts between stages, and an unknown key
raises rather than leaving a baseline value silently in force.

`generate_config.py -h` lists every switch. The one worth knowing:

```bash
generate_config.py --rest2 --s_cold 1 --s_hot 0.25 --N_rungs 4 --interp sqrt
```

`--system alanine` → **6 rungs**, `--system macrocycle` → **8 rungs** (unvalidated starting
ladder), `--system cyclo_rgdfv` → **10 rungs** (matched-pilot evidence, RGD only; enforces the
Sage 2.2 / AM1-BCC SMILES route and rejects a PDB invocation). `--all` writes everything, including
both `md-cold-config.json` and `md-hot-config.json`.

Stages (c) and (d) are resumable: re-running continues at the first chunk without a `done.json`,
and a finished run reports `already-complete`. **One process per GPU** — two processes sharing a
card without CUDA MPS run ~3.7× slower each.
