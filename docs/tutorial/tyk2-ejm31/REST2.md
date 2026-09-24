# REST2 on a protein–ligand complex: heating the ligand, and the pocket around it

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a run
executed as written on four RTX 3080s. The settings are justified in
[Choosing τ_max and the rung count](choosing-the-ladder.md); you do not need it to follow this page.

**Two routes**, both starting from the same configuration with both selection lists EMPTY:

```yaml
backbone_scaling_list: []
sidechain_scaling_list: []
```

| | what you change | rungs | τ_max | what it buys |
|---|---|---|---|---|
| **Route A — the ligand alone** | nothing: leave both lists empty and name the ligand | 8 | 0.5 | the ligand's own conformers, one small hot region |
| **Route B — add the pocket** | paste the pocket finder's residues into `sidechain_scaling_list` | **12** | **0.4** | the pocket rearranges with the ligand |

**Route B needs MORE rungs and a LOWER τ_max than route A, and both follow from the same two
rules.** The hot region is six times larger, so at the same τ_max neighbouring states overlap
worse — recovering the acceptance takes more rungs, not a colder top rung. And τ_max is set by the
barriers you need crossed, which for this system are the ligand's: 0.4 crosses them, and the
pocket's χ torsions are along for the ride either way.

**`backbone_scaling_list` stays `[]` in both routes.** The pocket finder prints a
`sidechain_scaling_list` line specifically, and the backbone key — a separate supported key taking
the same mask grammar — is for a hinge or a loop you mean to reorganise. A pocket's χ torsions were
never the constraint here: going from τ 0.25 to 0.4 gained **6x on the ligand's torsions and 11% on
the χs**.

## What you need first

1. The prepared complex: `build/built.xml`, `build/built.pdb`, `build/built.log` and
   `build/ligand_mapping.json` from one `md-openmm build-top` of TYK2 with `ejm_31`, reusing the
   registered ligand package. **Not written here**: the preparation is done once and every method reuses it. Built here it is **53,030 particles**, 16,462 residues, ff14SB and
   TIP3P in a 1.2 nm cubic box.

2. **The ligand's SDF beside the System**, for now:

   ```bash
   cp <package>/L31.sdf build/L31.sdf
   ```

   !!! warning "A workaround with a date on it"
       `build-top`'s complex route does not yet write `<RESNAME>.sdf` beside the System, so
       `--rest2-scaler` cannot read the ligand's bond orders and refuses — correctly, since which
       of its torsions stay unscaled cannot be guessed. Recorded as a defect on 2026-09-21; the
       fix writes the file at build time from the package's own molecule, and this step then
       disappears. Until it lands, copy the file; do not skip the refusal.

3. **Four GPUs, and they must be the same model.** In a ladder every rank meets every other at the
   exchange barrier, so a faster card cannot make the ladder faster — it waits for the slowest
   rung — and it makes per-rung throughput incomparable between rungs. These runs used **four
   RTX 3080s**, and deliberately not the faster RTX A5000 in the same machine. Pick four identical
   cards; below they are written as `1,2,3,4` for the sake of a worked example.

4. **MPS**, because eight states over four cards is two ranks per card, and md-tools refuses to
   share a card without it (the ranks would be time-sliced, which is a different experiment rather
   than a slower one):

   ```bash
   export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/rest2-mps/pipe     # NOT under /tmp/<long path>
   export CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/rest2-mps/log
   mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
   CUDA_VISIBLE_DEVICES=<your four cards> nvidia-cuda-mps-control -d
   ```

   !!! warning "Two ways this goes wrong, both silent"
       **The pipe directory's path must be short.** A Unix socket path is limited to 108 bytes.
       Point `CUDA_MPS_PIPE_DIRECTORY` at a deep scratch directory and the daemon logs "Starting
       control daemon", exits immediately, and every later client reports MPS simply absent. Keep
       it in `$HOME/.cache`.

       **The client's device numbering is the SERVER's.** Say your four cards are `1,2,3,4`. A
       server started with `CUDA_VISIBLE_DEVICES=1,2,3,4` exposes them to its clients as
       `0,1,2,3`. A client that repeats `1,2,3,4` asks for a fifth card that does not exist, and
       every rank dies with "Illegal value for DeviceIndex: 3". Launch the ladder with the
       SERVER's numbering, which is always `0..n-1` for n cards:

       ```bash
       export CUDA_VISIBLE_DEVICES=0,1,2,3      # the server's four cards, in ITS numbering
       ```

       This has a useful consequence: a server bound to four chosen cards cannot reach any other
       card **at all**, so "do not use the fast one" stops depending on anybody remembering it.

## 1. Which residues line the pocket

Ask, read, paste. The helper prints; it never resolves a pocket at run time.

```bash
python -m md_tools.rest2.pocket build/built.pdb --ligand ":<ligand residue index>" --cutoff-nm 0.5
```

```text
pocket of L31 at topology residue 291 (chain 'B', id 1), 21 heavy atoms
  criterion : md-tools-pocket-selection/1: heavy-atom minimum distance, strictly < 0.5 nm,
              minimum image (the topology has a box)
  structure : built.pdb  topology_sha256 f2570f368d90...

  index  chain  resid   name   min distance (nm)
     15      A    903   LEU       0.354
     16      A    904   GLY       0.360
     17      A    905   GLU       0.315
     18      A    906   GLY       0.413
     23      A    911   VAL       0.333
     40      A    928   ALA       0.334
     72      A    960   ILE       0.371
     90      A    978   MET       0.415
     91      A    979   GLU       0.360
     92      A    980   TYR       0.330
     93      A    981   VAL       0.284
     94      A    982   PRO       0.322
     95      A    983   LEU       0.415
     96      A    984   GLY       0.351
    139      A   1027   ARG       0.364
    140      A   1028   ASN       0.339
    142      A   1030   LEU       0.355
    152      A   1040   GLY       0.391
    153      A   1041   ASP       0.361

  paste into the scaler configuration, unchanged:
    sidechain_scaling_list: ":15-18,23,40,72,90-96,139-140,142,152-153"

  NOT selected, within 0.05 nm beyond the cutoff (the boundary is strict):
       97 SER      0.507 nm
      141 VAL      0.526 nm
       13 ARG      0.532 nm
```

That is a recognisable ATP site rather than an arbitrary shell: the hinge (MET978, GLU979, TYR980,
VAL981), the glycine-rich loop (LEU903–GLY906), VAL911, ALA928, ILE960, ARG1027/ASN1028, and
GLY1040/ASP1041. Four of the nineteen are glycines, which have no sidechain; they were inside the
cutoff, they are kept in the mask, and the record says they contribute no atoms. Three more
residues sit 7–32 pm beyond the cutoff and are listed as not selected — whether SER97 at 0.507 nm
belongs in your pocket is your decision, and widening the cutoff is how you make it.

Read the table before pasting the mask. The criterion is heavy-atom minimum distance, strictly
inside the cutoff, and the report lists what lies just outside with its distance — if a residue
you expect in the site is 0.02 nm beyond, that is a decision for you, not for the tool.

## 2. The scaled states, one set per route

**Route A — the ligand alone.** Both selection lists empty; only the ligand is named:

```yaml
# build/scaler-ligand.config
method: REST2
schedule: {kind: linear, n_states: 8, tau_min: 0.0, tau_max: 0.5}
backbone_scaling_list: []
sidechain_scaling_list: []
ligand_scaling_dict:
  L01:
    mask: ":<ligand residue index>"
    torsion_exclusions: auto
```

**Route B — add the pocket.** The only edits are the sidechain list, the rung count and τ_max:

```yaml
# build/scaler-pocket.config
method: REST2
schedule: {kind: linear, n_states: 12, tau_min: 0.0, tau_max: 0.4}
backbone_scaling_list: []
sidechain_scaling_list: "<the mask step 1 printed>"
ligand_scaling_dict:
  L01:
    mask: ":<ligand residue index>"
    torsion_exclusions: auto
```

An empty list and an absent key are not the same thing to read, but they resolve the same way: a
category you name empty is a category you heat nothing of. Writing them explicitly is how the two
routes stay one file apart.

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb \
    --config build/scaler-ligand.config
```

`build/REST2/scaler.log` prints the resolved residue map — index, chain, author residue id, name
and role — and `scaler.yaml` records the selection, its digest and the exact arguments each state
was built with. Check the map against step 1's table before running anything.

```text
route A    selection   EXPLICIT (md-tools-selective-rest2/1):  32 nonbonded atom(s),
                       4 torsion central bond(s), 0 CMAP term(s) scaled
route B    selection   EXPLICIT (md-tools-selective-rest2/1): 193 nonbonded atom(s),
                       52 torsion central bond(s), 0 CMAP term(s) scaled
```

The whole solute is 4,701 atoms, so route A heats 0.7% of it and route B 4.1%. No CMAP term is
scaled in either: CMAP couples backbone φ and ψ, and neither ladder selects a backbone.

!!! note "Which sidechain bonds are actually scaled"
    Per residue, from the table in `md_tools.rest2.sidechains`: every chi, and never the aromatic
    rings, ARG's guanidinium or the ASN/GLN sidechain amide. `python -c "from
    md_tools.rest2.sidechains import describe_residue; print(describe_residue('TYR'))"` prints one
    residue.

## 3. Generate each ladder

The REST2 configuration *claims* the region; `build-md` checks the claim against the states and
refuses a disagreement, naming the rebuild command. Claiming nothing is also allowed: the record
is then authoritative and `build-md.log` prints the region it found.

```yaml
# REST2-ligand.config -- route A, exactly as it was run
protocol: REST2
solvent: explicit
dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 20260921}
stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000
  restrained_npt_steps: 25000
  unrestrained_npt_steps: 200000
rest2:
  number_of_replicas: 8
  tau_max: 0.5
  exchange_interval_steps: 1000
  number_of_exchanges: 2500
  state_trajectory: true
  rem_log: true
  neighbour_acceptance_report: true
  ligand_scaling_dict:
    L31: {mask: ":291"}
reporting: {crd_printout_solute: 5000, info_printout: 25000, checkpoint_printout: 50000}
```

2500 exchanges x 1000 steps x 2 fs is **5 ns per state, 40 ns aggregate**, and the exchange
interval is 2 ps. `L31` and `:291` are this complex's ligand label and residue index -- step 1
printed them; substitute your own.

For route B, the same file with three changes — the pocket, twelve rungs and τ_max 0.4:

```yaml
rest2:
  number_of_replicas: 12
  tau_max: 0.4
  sidechain_scaling_list: ":15-18,23,40,72,90-96,139-140,142,152-153"
```

```bash
md-openmm build-md -odir ./REST2-ligand-run1 --config REST2-ligand.config
```

## 4. Run them: route A on four cards, route B on six

```bash
cd REST2-ligand-run1
bash run.sh                 # or the mpirun line run.sh prints, -n 8 with MPS
```

Each rank prints where it landed. Eight ranks over four cards, two per card, MPS in front of
them:

```text
# platform           : CUDA device=0 (measured throughput (balanced) (rank 0 of 8, local rank 0 on <host>, 2 worker(s) on this device)), precision mixed
# this process drives: state(s) [0] of 8
# platform           : CUDA device=0 (... rank 1 of 8, local rank 1 ..., 2 worker(s) on this device), precision mixed
# platform           : CUDA device=1 (... rank 2 of 8 ...)      # ranks 2,3 -> device 1
# platform           : CUDA device=2 (... rank 4 of 8 ...)      # ranks 4,5 -> device 2
# platform           : CUDA device=3 (... rank 6 of 8 ...)      # ranks 6,7 -> device 3
```

**`device=N` is the MPS server's numbering, not nvidia-smi's**, and under MPS a client numbers the
server's device list 0..n-1 whatever its own `CUDA_DEVICE_ORDER` says. Set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` on the launch that starts the DAEMON -- without it, CUDA's default
FASTEST_FIRST order can map your `CUDA_VISIBLE_DEVICES` onto different physical cards than you
named -- and check with `nvidia-smi` which cards hold the server's memory before trusting any
record of where a run ran.

The tail of `remd_records/REST2_prod1.out` is the completion summary:

```text
#   overall               4802/8750   0.549
# REST2:
#   tau ladder            0, 0.071429, 0.142857, 0.214286, 0.285714, 0.357143, 0.428571, 0.5   (8 state(s), one temperature 300.0 K, NVT)
#   scaling               solute-solute (1-tau)^2, solute-environment 1-tau
#   left unscaled         bonds unscaled, angles unscaled, torsions: ordinary amide omega, aromatic ring bonds, other double bonds, impropers
#   solute region         4701 atom(s), 14 unscaled central bond(s), impropers unscaled
#   velocities on swap    never rescaled (one beta across the ladder)
# TIMINGS:
#   elapsed               4617.9 s
#   throughput            93.55 ns/day per replica, 748.39 ns/day aggregate over 8 state(s)
#   per step              1.8472 ms
# ---------------------------------------------------------------------------
run_status: completed
```

### Route B's group file, and why it is six cards

`build-md` writes one line per rung, so route B's group file has twelve rather than eight:

```text
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state0.xml  -c eq/eq_3.xml --group-index 0
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state1.xml  -c eq/eq_3.xml --group-index 1
...
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state11.xml -c eq/eq_3.xml --group-index 11
```

Twelve rungs at two ranks per card is **six cards**, and the launch is `run.sh`'s mpirun line with
`-n 12 ... -ng 12`. Keep two ranks per card: a third is worse and a fourth loses throughput
outright.

## 6. Checks worth running afterwards

```bash
md-openmm export-reference -idata REST2-ligand-run1 -odir bundle-ligand --stage REST2
python bundle-ligand/verify_rungs.py
```

`verify_rungs.py` rebuilds every rung from the built System with the bundled copy of md-tools' own
scaling code, and for a selective ladder it uses the region the record carries. It needs OpenMM
and nothing else.

```text
rung 0  tau 0          identical
rung 1  tau 0.071429   identical
rung 2  tau 0.142857   identical
rung 3  tau 0.214286   identical
rung 4  tau 0.285714   identical
rung 5  tau 0.357143   identical
rung 6  tau 0.428571   identical
rung 7  tau 0.5        identical
all 8 rungs rebuild identically from the built System
```

!!! warning "Two things `export-reference` needs that a finished run may not have beside it"
    Both were hit while producing the output above, and neither is reported until the export runs:

    1. **The stage log must be at `<run>/REST2.log`.** A ladder launched by `run.sh` writes
       `remd_records/REST2_prod1.log` instead, and the export stops with
       `REST2.log: cannot be read`. Copy it across first:
       `cp remd_records/REST2_prod1.log REST2.log`.
    2. **The source structure and the ligand package must still be beside the build** --
       `complex.pdb` next to `build/`, and `build/ligands/<id>/param_<id>/` intact with its
       `molecule.sdf`. The export verifies both by digest and refuses rather than guessing. A run
       tree that has been moved, or registered as a dataset without them, cannot be exported.

## Where these settings come from

τ_max and the rung counts above were established by running four ladders on this complex and
comparing them — including two measurements that look better than they are. That comparison, the
registered datasets, the throughput figures and the one check that did NOT pass are on their own
page, so they do not stand between a first-time reader and a running simulation:

[**Choosing τ_max and the rung count: what four ladders showed**](choosing-the-ladder.md)

