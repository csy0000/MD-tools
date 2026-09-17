# REST2: paracetamol in explicit water

!!! warning "Archived: written for md-tools 0.5.3"
    This page was run against 0.5.3 and is kept as a record of that release. 0.5.4 changed the
    workflow it describes: REST2 states are built once, as files, by
    `md-openmm build-top --rest2-scaler`; a stage or a ladder no longer scales anything at run
    time, so a hot stage takes its saved state as `-s` and a ladder reads `-s` only from its group
    file; and more torsions stay unscaled (aromatic rings, double bonds and impropers, not only the
    amide omega). Commands here may be refused by 0.5.4. For 0.5.4 or later, use the
    [current tutorials](../../../README.md).

A four-state REST2 ladder over paracetamol, run with md-tools **0.5.3**. Every command below was
run exactly as written, and every number is copied from that run (one NVIDIA RTX 3080, CUDA, mixed
precision).

!!! warning "In 0.5.3, `run.sh` refuses this ladder — use the direct command in step 5"
    Paracetamol's amide has to be classified from the molecule's bond orders, which only the
    `built.sdf` beside `built.xml` carries. The ladder step `build-md` writes into `run.sh` points
    at per-state files that have no SDF beside them, so it stops with

    ```text
    REST2: REST2: 1 amide candidate(s) could not be classified as ordinary or proline-like.
    ```

    That refusal is correct: the alternative is guessing. Step 5 runs the same ladder from
    `built.xml`, where the SDF is found. Fixed after 0.5.3: see [the 0.5.4 notes](#after-053).

If you have not run [cMD: paracetamol](../cMD/paracetamol.md), start there: this page does not
repeat how the system is built.

## What REST2 does here

Four copies of the system — **states** — run side by side at the same temperature, 300 K. They
differ only in how strongly the solute interacts: state *i* has its solute–solute terms scaled by
(1−τ)² and its solute–water terms by (1−τ), with τ = 0, 0.167, 0.333, 0.5. State 0 is the real,
unscaled molecule. Every 10 ps, neighbouring states try to swap configurations, so a conformation
found where the barriers are low can reach state 0. Bonds, angles and the ordinary amide ω torsion
are never scaled: a hot state that let the amide flip cis/trans would sample something state 0
never does. See [REST2](../../../../openmm_methods/REST2/README.md).

## 1. The dataset root, and why it is not the cMD one

```bash
mkdir -p PARA-REST2/build
cd PARA-REST2/build
```

Use a **new** root rather than adding REST2 beside `PARA/cMD-run1`. In 0.5.3 `input/` and `min/` are
shared by every run on a system, and the cMD run's `input/min.in` records its own stage lengths; a
REST2 configuration that differs is refused:

```text
build-md: input/min.in already exists and is not what this configuration resolves to.
```

## 2. Build the system

The same two input files as the cMD tutorial — `paracetamol.smi`:

```text
CC(=O)Nc1ccc(O)cc1 paracetamol
```

and `build-top.config`:

```yaml
solute:
  kind: ligand
```

```bash
md-openmm build-top -i paracetamol.smi \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

It writes `built.xml`, `built.pdb` and **`built.sdf`** — keep all three together. The SDF is what
step 5 reads the amide from.

## 3. Generate the ladder

`REST2.config` at the dataset root:

```yaml
protocol: REST2
solvent: explicit

rest2:
  number_of_replicas: 4
  tau_max: 0.5
  exchange_interval_steps: 5000
  number_of_exchanges: 100

stages:
  minimization_iterations: 1000
  restrained_nvt_steps: 5000
  restrained_npt_steps: 5000
  unrestrained_npt_steps: 5000

reporting:
  crd_printout_solute: 500
  info_printout: 5000
  checkpoint_printout: 5000
```

100 exchanges every 5000 steps is 500 000 steps, 1 ns, **per state**.

```bash
cd ..                                            # PARA-REST2/
md-openmm build-md -odir ./REST2-run1 --config REST2.config
```

## 4. Equilibrate with `run.sh`

```bash
cd REST2-run1
./run.sh
```

`run.sh` minimises and equilibrates once, at τ = 0 (`min`, `eq_1`, `eq_2`, `eq_3`; 26 s here), and
the box is fixed by the end of `eq_3`. Its last step, `== REST2 prod1 ==`, is the ladder, and in
0.5.3 it stops with the refusal in the warning above, once per MPI rank. The equilibrated state it
leaves, `eq/eq_3.xml`, is what the ladder starts from.

## 5. Run the ladder

```bash
python REST2.py -p ../build/built.pdb -s ../build/built.xml -c eq/eq_3.xml -log REST2.log
```

`-s ../build/built.xml` names the unscaled System; the ladder builds its four states from it at
start-up. `built.sdf` beside it is found automatically, and the amide is classified from it. One
process drives all four states on one GPU; choose it with `CUDA_VISIBLE_DEVICES`.

**Do not** point it at `--groupfile remd_groupfile.1`. That file names the per-state Systems
`build-md` wrote in `remd<n>/`, and in 0.5.3 those were scaled without the SDF: their
`build_states.log` lists `excluded_central_bonds: []` and `n_excluded_torsions: 0`, so the amide ω
in them is scaled with everything else.

It finished in 3 min 46 s. Check that the amide was protected — `solute.yaml`:

```text
rest2:
  omega_excluded_bonds:
  - - 1
    - 3
  omega_detection_route: ligand
  omega_ambiguous_candidates: []
  excluded_central_bonds:
  - - 1
    - 3
  n_excluded_torsions: 8
```

One amide C–N bond, atoms 1 and 3, with its 8 torsion terms left unscaled in every state, and no
candidate left unclassified.

## 6. Results

`REST2.out`:

```text
# steps completed       : 500000 of 500000
# exchanges             : 100 (every 5000 steps = 10.0 ps)
# solute frames         : 1000 (every 500 steps)
# production per replica: 1000.0 ps
# NEIGHBOURING-PAIR acceptance:
#   state 0 <-> state 1   12/50   0.240
#   state 1 <-> state 2   6/50   0.120
#   state 2 <-> state 3   6/50   0.120
#   overall               24/150   0.160
# REST2:
#   tau ladder            0, 0.166667, 0.333333, 0.5   (4 state(s), one temperature 300.0 K, NVT)
#   scaling               solute-solute (1-tau)^2, solute-environment 1-tau
#   left unscaled         bonds unscaled, angles unscaled, ordinary amide omega unscaled
#   solute region         20 atom(s), 1 omega bond(s) excluded
#   velocities on swap    never rescaled (one beta across the ladder)
# TIMINGS:
#   throughput            413.86 ns/day per replica, 1655.44 ns/day aggregate over 4 state(s)
run_status: completed
```

Each neighbouring pair exchanged 12–24 % of the time, so configurations do move along the ladder.
For a real study, lengthen the ladder (`number_of_exchanges`), and add states if a pair's acceptance
falls much lower.

What it wrote, in `REST2-run1/`:

| file | what |
|---|---|
| `solute_state<i>_prod1.nc` | the solute trajectory of **state** *i*, whichever copy held it at the time. `state0` is the physical ensemble |
| `whole_state<i>_prod1.nc` | every atom, per state |
| `exchange.csv`, `rem.log` | every exchange attempt and whether it was accepted |
| `REST2.nc` | the authoritative analysis store the summary is computed from |
| `restart.json`, `REST2.log` | the completion manifest and the machine record |
| `solute.yaml` | the scaled region and the omega decision above |

The files follow the thermodynamic state, not the copy: there is no demultiplexing step to run
before analysing `solute_state0_prod1.nc`.

## After 0.5.3 { #after-053 }

Fixed for 0.5.4: the omega evidence is chosen per residue, and every step that scales refuses an
amide it cannot classify instead of scaling it, including the per-state files `build-md` writes,
which then receive `built.sdf`. With that code `./run.sh` runs the ladder itself and step 5 is not
needed. These tutorials are pinned to 0.5.3, and 0.5.4 will have its own.
