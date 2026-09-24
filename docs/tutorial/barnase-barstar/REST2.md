# Selective REST2: heating the barnase–barstar interface

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version, on four NVIDIA RTX 3080 cards.

A twelve-rung REST2 ladder over **only the interface** of the barnase–barstar complex: the 39
residues whose sidechains face across the two chains, heated to τ = 0.4, with the rest of both
proteins and all 8 913 waters at full strength. 10 ns of production per state, 2 hours wall clock.

**Starts from a built system.** Do [the system page](index.md) first: it expands biological
assembly 3 of 1BRS, builds the 44 missing side-chain atoms, protonates with PROPKA and writes
`1BRS/build/built.xml`.

## Why heat only the interface

A REST2 ladder's acceptance falls as the heated region grows, because the energy difference
between neighbouring rungs grows with it. Scaling the whole 3 132-atom solute of a protein–protein
complex would need roughly five times the rungs chignolin's 138 atoms need for the same τ span —
more replicas than most people have cards for.

The interface is also the part the question is about. Barnase–barstar associate through a small,
tightly packed, heavily charged patch; what moves on a 10 ns timescale is sidechain rotamers in
that patch, not the fold of either protein. Heating exactly that patch is the whole point of
selective REST2, and it is the same machinery [the TYK2 ladder](../tyk2-ejm31/REST2.md) uses for a
ligand pocket.

## 1. Which residues line the interface

Ask, read, paste. The helper prints; it never resolves an interface at run time, so the
configuration carries the explicit mask and the record says which residues were hot.

The mask is over **topology** residue indices, which are not the author numbering in the PDB, so
find each chain's range first:

```bash
python -c "
from openmm.app import PDBFile
for chain in PDBFile('build/built.pdb').topology.chains():
    rs = [r for r in chain.residues() if r.name not in ('HOH', 'NA', 'CL')]
    if rs: print(f'chain {chain.id}: :{rs[0].index + 1}-{rs[-1].index + 1}')
"
```

```text
chain A: :1-108
chain B: :109-197
```

Then ask for the interface between them:

```bash
python -m md_tools.rest2.pocket build/built.pdb \
    --int1 ":1-108" --int2 ":109-197" --cutoff-nm 0.5
```

```text
interface between --int1 (108 residue(s)) and --int2 (89 residue(s))
  criterion : md-tools-pocket-selection/2: heavy-atom minimum distance ACROSS the two regions,
              strictly < 0.5 nm, minimum image (the topology has a box)
  structure : built.pdb  topology_sha256 36acd984cdbf...

   side  index  chain  resid   name   min distance (nm)
   int1     25      A     27   LYS       0.310
   int1     33      A     35   TRP       0.389
   …
   int1     57      A     59   ARG       0.264
   …
   int2    146      B     39   ASP       0.266
   int2    184      B     76   GLU       0.264

  paste into the scaler configuration, unchanged:
    sidechain_scaling_list: ":25,33,35-36,54-60,71,80-83,85,99-102,104,135,137-139,141-144,146-147,150-154,181,184"
      --int1 side only: ":25,33,35-36,54-60,71,80-83,85,99-102,104"
      --int2 side only: ":135,137-139,141-144,146-147,150-154,181,184"
```

Both sides are protein sidechains here, so the **combined** mask is what goes into
`sidechain_scaling_list`. The per-side masks are printed because for a *ligand* site only one side
belongs there: the ligand goes in `ligand_scaling_dict`, and naming it in both would scale it
twice. `--ligand X` is the same call as `--int2 X` with `--int1` left out, where it defaults to
every non-solvent residue not in `--int2`.

**Only contacts ACROSS the two regions count.** A residue is never selected because of its own
neighbours in the same chain; if closeness to anything counted, the mask would be the whole
protein.

It selects **39 residues**: 22 on barnase and 17 on barstar.

| chain | residues |
|---|---|
| barnase (A) | Lys27, Trp35, Ala37, Ser38, Phe56, Ser57, Asn58, **Arg59**, Glu60, Gly61, Lys62, Glu73, Phe82, **Arg83**, Asn84, Ser85, **Arg87**, Asp101, **His102**, Tyr103, Gln104, Phe106 |
| barstar (B) | Pro27, Tyr29, Tyr30, Gly31, Asn33, Leu34, Asp35, Ala36, Trp38, **Asp39**, Thr42, Gly43, Trp44, Val45, Glu46, Val73, **Glu76** |

That is a recognisable interface rather than an arbitrary shell: barnase's Arg59, Arg83 and Arg87
against barstar's Asp39 and Glu76 are the salt bridges this complex is famous for, and His102 is
barnase's catalytic histidine, which barstar blocks.

!!! tip "Check the count the scaler reports back"
    Give the helper the wrong ranges — splitting a chain in the middle, say — and it returns a
    different mask that looks just as plausible. The check is in
    [step 2](#2-build-the-twelve-scaled-states): the scaler reports how many residues actually
    carry scaled atoms, and it must equal your residue count minus the glycines.

## 2. Build the twelve scaled states

`build/scaler.config`:

```yaml
method: REST2
schedule: {kind: linear, n_states: 12, tau_min: 0.0, tau_max: 0.4}
sidechain_scaling_list: ":25,33,35-36,54-60,71,80-83,85,99-102,104,135,137-139,141-144,146-147,150-154,181,184"
```

There is no `backbone_scaling_list`: a category you heat nothing of is one you **leave out**, and
an empty mask is refused rather than accepted as "nothing".

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
```

From `scaler.log`:

```text
  schedule                    linear, 12 state(s), tau 0.0 .. 0.4
  solute                      3132 atom(s)
  selection                   EXPLICIT (md-tools-selective-rest2/1): 400 nonbonded atom(s), 97 torsion central bond(s), 0 CMAP term(s) scaled
  unscaled torsions           amide omega 22 bond(s), aromatic ring 71 bond(s), double bond 9 bond(s), impropers 4 term(s)
                              412 torsion term(s) unscaled, 954 scaled
  picture of protein          not drawn: 329 heavy atoms in 36 residue(s), above 200: not drawn; scaler.yaml lists every unscaled bond
```

**Read `36 residue(s)` as the check on your mask.** Three of the 39 selected residues are glycines
— barnase Gly61, barstar Gly31 and Gly43 — which have no sidechain to heat. They stay in the mask
and contribute no atoms, so 39 − 3 = 36 is exactly right. A mask that is off by one would still
resolve, and this number is how you would notice.

400 nonbonded atoms are scaled, of which 329 are heavy. The other 2 732 solute atoms, every water
and every ion are untouched, and so are all 412 unscaled torsion terms — the amide ω, the aromatic
rings, the other double bonds and every improper.

The picture is not drawn above 200 heavy atoms; `scaler.yaml` lists every unscaled bond instead.

## 3. Generate the ladder

`REST2.config` at the dataset root:

```yaml
protocol: REST2
solvent: explicit

rest2:
  number_of_replicas: 12         # must match build/REST2/scaler.yaml
  tau_max: 0.4                   # must match build/REST2/scaler.yaml
  exchange_interval_steps: 1000  # 2 ps between exchange attempts at 2 fs
  number_of_exchanges: 5000      # 5,000,000 steps = 10 ns per state
  equilibration_steps: 50000     # 100 ps per state at its own Hamiltonian

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

```bash
cd ..                                            # 1BRS/
md-openmm build-md -odir ./REST2-run1 --config REST2.config
```

`number_of_replicas` and `tau_max` scale nothing — they are a **claim** about `build/REST2/`, and
`build-md` refuses if they disagree with it.

## 4. Run it

Twelve states need twelve workers. With fewer than twelve cards they share, which md-tools allows
only under NVIDIA MPS. This run used four:

```bash
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.cache/mps/pipe"   # keep it SHORT: a Unix socket caps at 108 bytes
export CUDA_MPS_LOG_DIRECTORY="$HOME/.cache/mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
CUDA_VISIBLE_DEVICES=<your four cards> nvidia-cuda-mps-control -d

cd REST2-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh   # the SERVER's numbering

echo quit | nvidia-cuda-mps-control
```

The client addresses the daemon's cards as `0..n-1`, **not** by their physical numbers — the same
trap [the paracetamol ladder](../paracetamol/REST2.md#5-run-it) describes in full.

The placement is measured, not assumed, and comes out even:

```text
# platform : CUDA device=0 (measured throughput (balanced) (rank 0 of 12, local rank 0, 3 worker(s) on this device)), precision mixed
```

three workers on each of the four cards.

## 5. Results

`remd_records/REST2_prod1.out`:

```text
# steps completed       : 5000000 of 5000000
# exchanges             : 5000 (every 1000 steps = 2.0 ps)
# solute frames         : 5000 (every 1000 steps)
# production per replica: 10000.0 ps
# final state->walker   : [6, 2, 1, 3, 7, 8, 0, 4, 10, 5, 11, 9]
# NEIGHBOURING-PAIR acceptance:
#   state 0 <-> state 1   558/2500   0.223
#   state 1 <-> state 2   554/2500   0.222
#   state 2 <-> state 3   553/2500   0.221
#   state 3 <-> state 4   562/2500   0.225
#   state 4 <-> state 5   563/2500   0.225
#   state 5 <-> state 6   525/2500   0.210
#   state 6 <-> state 7   457/2500   0.183
#   state 7 <-> state 8   486/2500   0.194
#   state 8 <-> state 9   404/2500   0.162
#   state 9 <-> state 10   405/2500   0.162
#   state 10 <-> state 11   322/2500   0.129
#   overall               5389/27500   0.196
# REST2:
#   tau ladder            0, 0.036364, ..., 0.4   (12 state(s), one temperature 300.0 K, NVT)
#   solute region         3132 atom(s), 84 unscaled central bond(s), impropers unscaled
# TIMINGS:
#   elapsed               7089.3 s
#   throughput            121.87 ns/day per replica, 1462.49 ns/day aggregate over 12 state(s)
```

**Acceptance is usable everywhere and thins at the top**, 0.225 at the cold end against 0.129 for
the last pair. That is the shape to expect when the rungs are evenly spaced in τ but the energy
gap is not: a further ladder on this system should space the top rungs more tightly rather than
add rungs at the bottom.

`final state->walker` is `[6, 2, 1, 3, 7, 8, 0, 4, 10, 5, 11, 9]` — **no walker ended where it
started**, so configurations travelled the whole ladder rather than rattling between neighbours.

Each state has its own trajectory, `solute_state<i>_prod1.nc`. The index is the **state's**, not
the walker's: `solute_state0_prod1.nc` is the unscaled ensemble, and it is the one to analyse. No
demultiplexing is needed.

## What it wrote

```text
1BRS/
├── build/          built.xml, built.pdb, assembly.json, REST2/ (12 states + scaler.yaml)
├── input/  min/    shared by every run on this system
└── REST2-run1/
    ├── remd_groupfile.1      one line per state, each naming its own saved System
    ├── remd_records/         REST2_prod1.out, .log, restart_prod1.json
    ├── solute_state*_prod1.nc   12 state trajectories, 5000 frames each
    ├── whole_state*_prod1.nc    every atom, every 100 ps
    ├── exchange.csv          every attempt, accepted or not
    └── solute.yaml           the scaler.yaml it read, and each state's sha256
```

## Next

* the same machinery on a ligand pocket, with the rung count measured rather than assumed:
  [TYK2 + ejm_31](../tyk2-ejm31/REST2.md) and [choosing the ladder](../tyk2-ejm31/choosing-the-ladder.md)
* the unbiased run this ladder is compared against: [cMD: barnase–barstar](cMD.md)
* what the scaled states are: [REST2](../../openmm_methods/REST2/README.md)
