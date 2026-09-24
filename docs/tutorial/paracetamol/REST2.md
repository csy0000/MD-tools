# REST2: paracetamol in explicit water, 10 ns per state

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version.

A four-state REST2 ladder over paracetamol, 10 ns of production **per state**. Every command below
was run exactly as written, and every number is copied from the files that run produced.

The molecule is **not parameterised here**: this page reuses the package
[cMD: paracetamol](../paracetamol/cMD.md) makes, so read that page first if you want to know where
ligand parameters come from. The charge calculation happens once, there.

The ladder took 25.4 min on one shared GPU, plus a few seconds to build and scale.

## What this runs

Four copies at 300 K, differing only in how strongly the solute interacts: state *i* scales
solute–solute terms by (1−τ)² and solute–water by (1−τ), τ = 0, 0.167, 0.333, 0.5. State 0 is the
real molecule, and neighbours swap configurations every 2 ps. Amide ω, aromatic rings, other double
bonds and impropers are never scaled. The method:
[REST2](../../openmm_methods/REST2/README.md).

## 1. The dataset root

A REST2 ladder gets its **own** dataset root, beside the cMD one rather than inside it: `input/` is
shared by every run on a system, and this ladder's equilibration stages are longer than the cMD
page's, so `build-md` refuses to overwrite them. The parameters are shared; the box is not.

```bash
mkdir -p REST2-PARA/build
cd REST2-PARA/build
cp "$MD_DATA/parameters/ligands/CHEMBL112/param_e932f4c4f371/TYL.sdf" paracetamol.sdf
```

## 2. Build the system, from the registered package

`build-top.config`:

```yaml
solute:
  kind: ligand
  residue_name: TYL
  parameters: CHEMBL112/param_e932f4c4f371
```

```bash
md-openmm build-top -i paracetamol.sdf \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

```text
Resolved configuration
  solute.parameters           CHEMBL112/param_e932f4c4f371   (set)
  residue name                TYL   (stated)
Preparation
  parameters   : CHEMBL112/param_e932f4c4f371 (reused (stated reference))
  solvation    : 592 waters, ions {'NA': 2, 'CL': 2}, box dodecahedron (19.1 nm^3)
Counts
  atoms                       1800
  solute atoms                20
```

`parameters` names the package **by identity**, with no path: it is resolved in the shared catalog
under `$MD_DATA/parameters/ligands`, where `md-openmm data-register --ligand-package` put it. A
package that is not registered is named by its path instead — see
[cMD: paracetamol](../paracetamol/cMD.md).

`reused (stated reference)` means no charge was generated here. This box and the cMD page's box
rest on the same numbers, which is the point: a comparison between the two runs is a comparison of
sampling, not of parameters.

It also writes `TYL.sdf` beside the System. **Keep it**: it holds the bond orders, which is how the
next step knows which bonds are aromatic and which C–N is an amide.

## 3. Build the scaled states

`scaler.config`, beside the built System:

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
└── TYL-unscaled.png     what stays unscaled, drawn
```

From `scaler.log`:

```text
  schedule                    linear, 4 state(s), tau 0.0 .. 0.5
  solute                      20 atom(s)
  unscaled torsions           amide omega 1 bond(s), aromatic ring 6 bond(s), double bond 0 bond(s), impropers 24 term(s)
                              56 torsion term(s) unscaled, 18 scaled
  SDF for TYL                 TYL.sdf
```

and `TYL-unscaled.png` — every red bond keeps all its torsions unscaled in every state, and every red
atom is the centre of an unscaled improper; the numbers are the atom indices `scaler.yaml` uses:

![paracetamol: unscaled torsions in red](images/paracetamol-rest2-unscaled.png)

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
cd ..                                            # REST2-PARA/
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

Four replicas need four workers. Give each one its own GPU if you have them:

```bash
cd REST2-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4 ./run.sh
```

This run had **one** card, so the four workers shared it — which md-tools allows only under
NVIDIA MPS, and refuses otherwise:

```text
md-run: REST2: this launch shares device 0 between 4 workers. NVIDIA MPS is not-a-client for this
process ... Without MPS, workers on one GPU are time-sliced: each waits for the others' kernels,
and a synchronous ladder runs at the pace of that shared GPU.
```

md-tools neither starts nor stops the daemon — that is the operator's decision, because an MPS
daemon outlives the run. Start one that belongs to you, run, and stop it:

```bash
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.cache/mps/pipe"    # keep this path SHORT: it holds a Unix
export CUDA_MPS_LOG_DIRECTORY="$HOME/.cache/mps/log"      # socket, and those cap at 108 characters
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./run.sh

echo quit | nvidia-cuda-mps-control
```

A private pipe directory matters: with the default one the daemon serves **every** card on the
host, so a shared machine's other GPUs end up behind your daemon.

`run.sh` minimises and equilibrates once on the unscaled system (`min`, `eq_1`, `eq_2`, `eq_3`),
then launches the ladder, one MPI rank per state:

```text
mpirun -n 4 md-openmm md-run -ng 4 -i ../input/REST2.in -p "${TOPOLOGY}" \
  --groupfile remd_groupfile.1 -odir . \
  -o remd_records/REST2_prod1.out -log remd_records/REST2_prod1.log -r remd_records/restart_prod1.json
```

!!! tip "Which GPU is which"
    CUDA numbers devices fastest first, `nvidia-smi` by PCI bus, so on a machine with mixed cards
    `CUDA_VISIBLE_DEVICES=1,2,3,4` need not be the GPUs `nvidia-smi` lists as 1–4. Put
    `CUDA_DEVICE_ORDER=PCI_BUS_ID` in front to make the two numberings agree.

It finished in 19.5 min and ended with `run.sh: all stages reported completion`. Stages that had
already completed were skipped, not repeated — only the ladder ran.

## 6. Results

`remd_records/REST2_prod1.out`:

```text
# steps completed       : 5000000 of 5000000
# exchanges             : 5000 (every 1000 steps = 2.0 ps)
# whole frames          : 100 (every 50000 steps)
# solute frames         : 5000 (every 1000 steps)
# production per replica: 10000.0 ps
# exchange rule         : neighbouring
# final state->walker   : [0, 3, 1, 2]
# NEIGHBOURING-PAIR acceptance:
#   basis: cumulative over every committed exchange row in the authoritative NetCDF, exchanges 0-4999
#   state 0 <-> state 1   544/2500   0.218
#   state 1 <-> state 2   553/2500   0.221
#   state 2 <-> state 3   565/2500   0.226
#   overall               1662/7500   0.222
# REST2:
#   tau ladder            0, 0.166667, 0.333333, 0.5   (4 state(s), one temperature 300.0 K, NVT)
#   scaling               solute-solute (1-tau)^2, solute-environment 1-tau
#   left unscaled         bonds unscaled, angles unscaled, torsions: ordinary amide omega, aromatic ring bonds, other double bonds, impropers
#   solute region         20 atom(s), 7 unscaled central bond(s), impropers unscaled
#   system sha256         52db01895dced19f084147c3fb2ce3168149fe6112b5d4859ce60b472dad5132
#   velocities on swap    never rescaled (one beta across the ladder)
# TIMINGS:
#   elapsed               1523.8 s
#   throughput            567.00 ns/day per replica, 2268.00 ns/day aggregate over 4 state(s)
#   per step              0.3048 ms
run_status: completed
```

An acceptance near 0.22 on every neighbouring pair is a usable ladder: high enough that
configurations move, low enough that the states are telling each other something.

Both measurements below come from [`paracetamol_analysis.py`](paracetamol_analysis.py), run from
the dataset root.

**The walkers really travel.** An acceptance ratio on its own does not show that: neighbours can
swap busily while nothing ever crosses the ladder. Counting, per walker, full journeys from the
physical state to the hottest and back:

```text
round trips (0 -> top -> 0), per walker: 56, 48, 54, 46
```

**The unscaled amide stays planar.** Measured on `solute_state<i>_prod1.nc`, 5000 frames each. An
angle needs a circular standard deviation — 179° and −179° are 2° apart, not 358°:

| torsion | scaled? | state 0 (τ = 0) | state 1 | state 2 | state 3 (τ = 0.5) |
|---|---|---|---|---|---|
| amide ω, atoms 0-1-3-4 | no | 12.2° | 12.3° | 12.0° | 11.7° |
| ring about N–C, atoms 1-3-4-5 | yes | 148.5° | 139.2° | 174.1° | 138.6° |

and **not one cis frame in any state** — 0 of 5000 with \|ω\| < 90°, state 0 to state 3. The amide
looks the same in the hottest state as in the physical one, which is the point of leaving it
unscaled: a hot state that isomerised it would sample a geometry state 0 never visits, and every
exchange would carry that geometry down the ladder. **That is the robust result on this page**: the
12° and the zero cis frames come back unchanged from one run to the next.

!!! warning "The ring row does not show what it looks like it should"
    A reader expects the scaled torsion to widen with τ, and an earlier run of this page obliged —
    126.7°, 138.6°, 160.2°, 161.1°. This one does not: 148.5°, 139.2°, 174.1°, 138.6°, with the
    hottest state *narrower* than the physical one. Nothing is wrong with either run. The ring has
    two equivalent faces, so its distribution is multi-modal, and a circular standard deviation
    over a multi-modal distribution is a weighted sum of population occupancies that 10 ns does
    not pin down. It is the wrong statistic for this torsion, and no number of extra digits fixes
    that. To show that REST2 frees the rotation, compare the *populations* — or the barrier
    crossing rate, which is what τ_max should be chosen by (see
    [choosing τ_max](../chignolin/choosing-tau.md)).

The files follow the thermodynamic **state**: `solute_state0_prod1.nc` is the physical ensemble, and
no demultiplexing is needed before analysing it. `solute.yaml` records that the ladder integrated
the saved states (`detection_route: saved-state`, and the sha256 of each).

!!! note "One shared GPU is slower, and samples the same"
    This run put all four workers on one RTX 3080 under MPS and got 567 ns/day per replica.
    Earlier runs of the same ladder reached about 1400 ns/day per replica on four separate RTX
    3080s and 738 on one shared A5000. The physics is identical across all of them — same Systems,
    same ladder, acceptance near 0.22, the same round-trip counts to within the scatter of a
    different random trajectory. A shared card costs wall time, not correctness.

## Next

* where the parameters came from: [cMD: paracetamol](../paracetamol/cMD.md)
* the same parameters in a protein pocket:
  [the TYK2 complex](../tyk2-ejm31/index.md)
* a ladder over a peptide: [REST2: chignolin](../chignolin/REST2.md)
* register the finished directory as a dataset:
  [Registering a finished run](../../basics/data-register/index.md)
