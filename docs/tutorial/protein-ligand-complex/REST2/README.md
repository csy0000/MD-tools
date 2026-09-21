# REST2 on a protein–ligand complex: heating the ligand, and the pocket around it

!!! danger "SKELETON — nothing here has been run"
    Every command on this page is written to be run exactly as it stands, but **no ladder has been
    run yet**, so every measured quantity is left blank and marked `TO BE MEASURED`. A tutorial
    number is copied from the files a run produced or it is not written at all. This page is
    finished when the two campaigns in
    [the protein–ligand campaign](../../../development/protein-ligand-campaign.md) have run on 4
    GPUs and their numbers are pasted in.

    Two prerequisites are not yet met: the prepared TYK2 complex (S2's, one `build-top` per
    ligand) does not exist yet, and no GPU has been allocated for these ladders.

Two ladders on TYK2 with `ejm_31` bound, both eight states at 300 K, differing only in what is
hot:

| | hot region | τ range | why |
|---|---|---|---|
| **A** | the ligand instance alone | 0 → 0.5 | the ligand's own conformers, at the price of one small hot region |
| **B** | the ligand **and** the sidechains lining the pocket | 0 → 0.25 | the pocket rearranges with the ligand; a smaller τ_max because the hot region is larger |

The τ ranges differ on purpose. τ_max is not a free knob: a wider hot region at the same τ_max
gives a worse overlap between neighbouring states, so B trades reach for acceptance. Whether 0.25
is the right trade for this pocket is one of the things these runs measure.

## What you need first

1. The prepared complex: `build/built.xml`, `build/built.pdb`, `build/built.log` and
   `build/ligand_mapping.json` from one `md-openmm build-top` of TYK2 with `ejm_31`, reusing the
   registered ligand package. **Not written here**: the preparation is done once, by one session,
   and every method reuses it. Built here it is **53,030 particles**, 16,462 residues, ff14SB and
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
   rung — and it makes per-rung throughput incomparable between rungs. These runs used cards 1–4
   (RTX 3080) and deliberately not card 0 (RTX A5000) on the same machine.

4. **MPS**, because eight states over four cards is two ranks per card, and md-tools refuses to
   share a card without it (the ranks would be time-sliced, which is a different experiment rather
   than a slower one):

   ```bash
   export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/rest2-mps/pipe     # NOT under /tmp/<long path>
   export CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/rest2-mps/log
   mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
   CUDA_VISIBLE_DEVICES=1,2,3,4 nvidia-cuda-mps-control -d
   ```

   !!! warning "Two ways this goes wrong, both silent"
       **The pipe directory's path must be short.** A Unix socket path is limited to 108 bytes.
       Point `CUDA_MPS_PIPE_DIRECTORY` at a deep scratch directory and the daemon logs "Starting
       control daemon", exits immediately, and every later client reports MPS simply absent. Keep
       it in `$HOME/.cache`.

       **The client's device numbering is the SERVER's.** A server started with
       `CUDA_VISIBLE_DEVICES=1,2,3,4` exposes those four cards to its clients as `0,1,2,3`. A
       client that repeats `1,2,3,4` asks for a card that does not exist and every rank dies with
       "Illegal value for DeviceIndex: 3". Launch the ladder with:

       ```bash
       export CUDA_VISIBLE_DEVICES=0,1,2,3      # the server's four cards, in ITS numbering
       ```

       This has a useful consequence: a server bound to cards 1–4 cannot reach card 0 **at all**,
       so the homogeneity rule above stops depending on anybody remembering it.

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

## 2. The two scaled-state sets

Ladder **A**, the ligand alone:

```yaml
# build/scaler-ligand.config
method: REST2
schedule: {kind: linear, n_states: 8, tau_min: 0.0, tau_max: 0.5}
ligand_scaling_dict:
  L01:
    mask: ":<ligand residue index>"
    torsion_exclusions: auto
```

Ladder **B**, the ligand and the pocket sidechains:

```yaml
# build/scaler-pocket.config
method: REST2
schedule: {kind: linear, n_states: 8, tau_min: 0.0, tau_max: 0.25}
sidechain_scaling_list: "<the mask step 1 printed>"
ligand_scaling_dict:
  L01:
    mask: ":<ligand residue index>"
    torsion_exclusions: auto
```

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb \
    --config build/scaler-ligand.config
```

`build/REST2/scaler.log` prints the resolved residue map — index, chain, author residue id, name
and role — and `scaler.yaml` records the selection, its digest and the exact arguments each state
was built with. Check the map against step 1's table before running anything.

```text
ladder A   selection   EXPLICIT (md-tools-selective-rest2/1):  32 nonbonded atom(s),
                       4 torsion central bond(s), 0 CMAP term(s) scaled
ladder B   selection   EXPLICIT (md-tools-selective-rest2/1): 193 nonbonded atom(s),
                       52 torsion central bond(s), 0 CMAP term(s) scaled
```

The whole solute is 4,701 atoms, so ladder A heats 0.7% of it and ladder B 4.1%. No CMAP term is
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
# REST2-ligand.config
protocol: REST2
solvent: explicit
rest2:
  number_of_replicas: 8
  tau_max: 0.5
  exchange_interval_steps: 5000
  number_of_exchanges: TO BE DECIDED
  ligand_scaling_dict:
    L01: {mask: ":<ligand residue index>"}
stages: {...}
reporting: {...}
```

```bash
md-openmm build-md -odir ./REST2-ligand-run1 --config REST2-ligand.config
```

## 4. Run them, eight states on four cards

```bash
cd REST2-ligand-run1
bash run.sh                 # or the mpirun line run.sh prints, -n 8 with MPS
```

```text
TO BE MEASURED: the platform lines, one per rank, and the completion summary
```

## 5. What these runs are for

Every number below is blank until the runs exist. None of it is a guess.

| quantity | ladder A | ladder B |
|---|---|---|
| ns/day per state | TO BE MEASURED | TO BE MEASURED |
| ns/day against rungs and cards | TO BE MEASURED | TO BE MEASURED |
| neighbouring-pair acceptance | TO BE MEASURED | TO BE MEASURED |
| walker round trips through the ladder | TO BE MEASURED | TO BE MEASURED |
| hot atoms / scaled torsion bonds | TO BE MEASURED | TO BE MEASURED |

The comparison the campaign asks for is between A and B: what heating the pocket as well as the
ligand costs in acceptance, and whether it buys sampling the ligand-only ladder does not reach.

## 6. Checks worth running afterwards

```bash
md-openmm export-reference -idata REST2-ligand-run1 -odir bundle-ligand --stage REST2
python bundle-ligand/verify_rungs.py
```

`verify_rungs.py` rebuilds every rung from the built System with the bundled copy of md-tools' own
scaling code, and for a selective ladder it uses the region the record carries. It needs OpenMM
and nothing else.

```text
TO BE MEASURED: its output
```
