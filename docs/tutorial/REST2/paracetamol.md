# REST2: paracetamol in explicit water, 10 ns per state

!!! note "Requires md-tools 0.5.4 or later"
    Earlier releases built REST2 states differently; their tutorials are
    [archived](../archived/README.md).

A four-state REST2 ladder over paracetamol, from a SMILES string to 10 ns of production **per state**,
with md-tools **0.5.4**. Every command below was run exactly as written, and every number is copied
from the files that run produced (md-tools at commit `e1a84e1`; four NVIDIA RTX 3080 GPUs, CUDA,
mixed precision).

The whole thing took about 12 minutes: 1.5 min to build, 1 s to scale, 11 min to equilibrate and run
the ladder.

!!! info "What changed since 0.5.3"
    In 0.5.3 the ladder scaled its states itself, at run time, and `run.sh` refused this molecule.
    In 0.5.4 the scaled states are **built once, as files**, by `md-openmm build-top --rest2-scaler`,
    and every run integrates exactly those files. You can look at them, and at a picture of what
    was left unscaled, before a single step runs. Compare [the archived 0.5.3 page](../archived/0.5.3/REST2/paracetamol.md).

## What REST2 does here

Four copies of the system — **states** — run side by side at 300 K. They differ only in how strongly
the solute interacts: state *i* scales solute–solute terms by (1−τ)² and solute–water terms by
(1−τ), with τ = 0, 0.167, 0.333, 0.5. State 0 is the real molecule. Every 2 ps, neighbouring states
try to swap configurations, so something found where barriers are low can reach state 0.

Some torsions are **never scaled**, because a hot state that bent them would sample geometries state
0 never visits: the ordinary amide ω, every aromatic ring bond, other double bonds, and every
improper. See [REST2](../../openmm_methods/REST2/README.md).

## 1. The dataset root

```bash
mkdir -p PARA/build
cd PARA/build
```

`paracetamol.smi`:

```text
CC(=O)Nc1ccc(O)cc1 paracetamol
```

`build-top.config`:

```yaml
solute:
  kind: ligand
```

## 2. Build the system

```bash
md-openmm build-top -i paracetamol.smi \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

```text
Counts
  atoms                       1800
  solute atoms                20
  waters                      592
  ions                        {'NA': 2, 'CL': 2}
```

It writes `built.xml`, `built.pdb` and `built.sdf`. **Keep `built.sdf`**: it holds the bond orders,
which is how the next step knows which bonds are aromatic and which C–N is an amide.

## 3. Build the scaled states

This is the new step. `scaler.config`, beside the built System:

```yaml
method: REST2
schedule:
  kind: linear
  n_states: 4
  tau_min: 0.0
  tau_max: 0.5
```

```bash
md-openmm build-top --rest2-scaler -s built.xml -p built.pdb --config scaler.config
```

It took 1 s and wrote `build/REST2/`:

```text
build/REST2/
├── system_state0.xml  system_state1.xml  system_state2.xml  system_state3.xml
├── scaler.yaml          what these files are and how they were made (sha256 of each)
├── scaler.log           the same, for a person
└── UNL-unscaled.png     what stays unscaled, drawn
```

From `scaler.log`:

```text
  schedule                    linear, 4 state(s), tau 0.0 .. 0.5
  solute                      20 atom(s)
  unscaled torsions           amide omega 1 bond(s), aromatic ring 6 bond(s), double bond 0 bond(s), impropers 24 term(s)
                              56 torsion term(s) unscaled, 18 scaled
  SDF for UNL                 built.sdf
```

and `UNL-unscaled.png` — every red bond keeps all its torsions unscaled in every state, and every red
atom is the centre of an unscaled improper; the numbers are the atom indices `scaler.yaml` uses:

![paracetamol: unscaled torsions in red](images/paracetamol-unscaled.png)

The amide (1–3) and the whole ring are protected. What REST2 *does* heat are the 18 torsion terms
left: the methyl rotation, the rotation of the ring about the N–C bond (3–4), and the O–H rotation.

## 4. Generate the ladder

`REST2.config` at the dataset root:

```yaml
protocol: REST2
solvent: explicit

rest2:
  number_of_replicas: 4          # must match build/REST2/scaler.yaml
  tau_max: 0.5                   # must match build/REST2/scaler.yaml
  exchange_interval_steps: 1000  # 2 ps between exchange attempts
  number_of_exchanges: 5000      # 5,000,000 steps = 10 ns per state
  equilibration_steps: 50000     # 100 ps per state at its own Hamiltonian, before production

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 50000    # 100 ps
  restrained_npt_steps: 50000    # 100 ps
  unrestrained_npt_steps: 100000 # 200 ps

reporting:
  crd_printout_solute: 1000      # a solute frame every 2 ps: 5000 frames per state
  crd_printout_whole: 50000      # every atom every 100 ps
  info_printout: 5000
  checkpoint_printout: 50000
```

`number_of_replicas` and `tau_max` do not scale anything any more — they are a **claim** about
`build/REST2/`, and `build-md` refuses if they disagree with it.

```bash
cd ..                                            # PARA/
md-openmm build-md -odir ./REST2-run1 --config REST2.config
```

The group file it writes names the saved states, one per line:

```text
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state0.xml -c eq/eq_3.xml --group-index 0
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state1.xml -c eq/eq_3.xml --group-index 1
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state2.xml -c eq/eq_3.xml --group-index 2
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state3.xml -c eq/eq_3.xml --group-index 3
```

## 5. Run it

```bash
cd REST2-run1
CUDA_VISIBLE_DEVICES=1,2,3,4 ./run.sh
```

`run.sh` minimises and equilibrates once on the unscaled system (`min`, `eq_1`, `eq_2`, `eq_3`),
then launches the ladder, one MPI rank per state and one GPU per rank:

```text
mpirun -n 4 md-openmm md-run -ng 4 -i ../input/REST2.in -p "${TOPOLOGY}" \
  --groupfile remd_groupfile.1 -odir . \
  -o remd_records/REST2_prod1.out -log remd_records/REST2_prod1.log -r remd_records/restart_prod1.json
```

!!! tip "Which GPU is which"
    CUDA numbers devices fastest first, `nvidia-smi` by PCI bus, so on a machine with mixed cards
    `CUDA_VISIBLE_DEVICES=1,2,3,4` need not be the GPUs `nvidia-smi` lists as 1–4 -- on the machine
    this ran on, it was not. Add `CUDA_DEVICE_ORDER=PCI_BUS_ID` in front of the command to make the
    two numberings agree.

It finished in 11 min and ended with `run.sh: all stages reported completion`.

## 6. Results

`remd_records/REST2_prod1.out`:

```text
# steps completed       : 5000000 of 5000000
# exchanges             : 5000 (every 1000 steps = 2.0 ps)
# solute frames         : 5000 (every 1000 steps)
# production per replica: 10000.0 ps
# NEIGHBOURING-PAIR acceptance:
#   state 0 <-> state 1   546/2500   0.218
#   state 1 <-> state 2   540/2500   0.216
#   state 2 <-> state 3   583/2500   0.233
#   overall               1669/7500   0.223
# REST2:
#   tau ladder            0, 0.166667, 0.333333, 0.5   (4 state(s), one temperature 300.0 K, NVT)
#   left unscaled         bonds unscaled, angles unscaled, torsions: ordinary amide omega, aromatic ring bonds, other double bonds, impropers
#   solute region         20 atom(s), 7 unscaled central bond(s), impropers unscaled
# TIMINGS:
#   throughput            1397.83 ns/day per replica, 5591.32 ns/day aggregate over 4 state(s)
run_status: completed
```

**The walkers really travel.** Counting, in `exchange.csv`, every full trip of a walker from state 0
to state 3 and back: 54, 51, 46 and 56 round trips for the four walkers.

**The unscaled amide stays planar, and the scaled rotation loosens.** Measured on
`solute_state<i>_prod1.nc`, 5000 frames each:

| torsion | scaled? | state 0 (τ = 0) | state 3 (τ = 0.5) |
|---|---|---|---|
| amide ω, atoms 0-1-3-4 | no | trans, circular sd 11.9°, no cis frame | trans, circular sd 12.0°, no cis frame |
| ring about N–C, atoms 1-3-4-5 | yes | circular sd 110° | circular sd 160° |

The amide looks the same in the hottest state as in the physical one, which is the point of leaving
it unscaled; the ring rotation, which REST2 is meant to help, is visibly freer at τ = 0.5.

The files follow the thermodynamic **state**: `solute_state0_prod1.nc` is the physical ensemble, and
no demultiplexing is needed before analysing it. `solute.yaml` records that the ladder integrated
the saved states (`detection_route: saved-state`, and the sha256 of each).

## Next

* the ordinary MD version of this molecule: [cMD: paracetamol](../cMD/paracetamol.md)
