# REST2 on a protein–ligand complex: heating the ligand, and the pocket around it

!!! note "Both ladders were run on 2026-09-21; every number here is copied from their files"
    Four RTX 3080s, 8 ranks, 2 per card under MPS, CUDA mixed precision, md-tools 0.6.1 on
    branch `work/0.6.1-selection`. The system is S2's prepared TYK2 fixture with `ejm_31`.

    **One check on this page did not pass**: the recorded exchange energies were recomputed
    against the saved states and came out at 0.077–0.121 kT, against a 0.05 kT tolerance that had
    been calibrated on a system thirty times smaller. The tolerance does not transfer between
    system sizes and is being replaced by a derived one; until that is measured, this check is
    **unvalidated on systems of this size**. It is recorded rather than quietly widened. See
    `docs/development/0.6.1/handoffs/S1.md`.

Two ladders on TYK2 with `ejm_31` bound, both eight states at 300 K, differing only in what is
hot:

| | hot region | τ range | why |
|---|---|---|---|
| **A** | the ligand instance alone | 0 → 0.5 | the ligand's own conformers, at the price of one small hot region |
| **B** | the ligand **and** the sidechains lining the pocket | 0 → 0.25 | the pocket rearranges with the ligand; a smaller τ_max because the hot region is larger |

The τ ranges differ on purpose. τ_max is not a free knob: a wider hot region at the same τ_max
gives a worse overlap between neighbouring states, so B trades reach for acceptance.

!!! warning "B's τ_max of 0.25 is NOT the right trade, and the measurements say so"
    Four ladders on this complex settle it. **Set τ_max by whether the hot rung crosses the
    barriers you care about; set acceptance by the number of rungs.** For this ligand:

    | ladder | τ_max | rungs | acceptance | ligand torsion crossings at the hot rung |
    |---|---|---|---|---|
    | A | 0.5 | 8 | 0.549 | 10.4 /ns |
    | A2 | 0.3 | 8 | 0.742 | **0.0 /ns** |
    | B | 0.25 | 8 | 0.388 | 3.8 /ns |
    | B3 | 0.4 | 12 | 0.373 | **23.1 /ns** |

    At τ 0.3 the ligand's aryl and amide torsions **never cross in 5 ns** — while acceptance and
    round trips both look their best of any ladder here. A ladder can transport configurations
    beautifully and have a hot rung that generates nothing new, and the acceptance figure cannot
    see it. (Counting every scaled bond would have hidden this too: a methyl on this ligand spins
    freely at any τ and supplies most of the raw count. Methyls and hydroxyls are excluded above.)

    Raising the pocket ladder to τ 0.4 over 12 rungs gives **6x the ligand crossings of B at the
    same rung spacing and the same acceptance** (0.373 against 0.388). The pocket sidechains were
    never the constraint: their χ torsions cross 238 /ns at τ 0.25 and 264 /ns at τ 0.4, a gain of
    11% where the ligand gained 6x. **τ_max is set by the ligand; the sidechains come along.**

    The cost is transport. Per-exchange diffusion is identical (0.171 against 0.170 states²
    /exchange), but a 12-rung ladder takes `N²` longer to traverse — 354 exchanges against 144 —
    so B3 completed 2 round trips where B completed 45. More rungs buy acceptance and pay for it
    in traversal, so use the fewest rungs that give workable acceptance at the τ_max your barriers
    demand.

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
# REST2-ligand.config -- ladder A, exactly as it was run
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

For ladder B, the same file with `tau_max: 0.25` and the pocket added:

```yaml
  sidechain_scaling_list: ":15-18,23,40,72,90-96,139-140,142,152-153"
```

```bash
md-openmm build-md -odir ./REST2-ligand-run1 --config REST2-ligand.config
```

## 4. Run them, eight states on four cards

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

### The 12-rung ladder the recommendation above is based on

Ladder B3 is ladder B's region at `tau_max: 0.4` over twelve rungs. Its scaler config:

```yaml
# build/scaler-pocket-12.config
method: REST2
schedule: {kind: linear, n_states: 12, tau_min: 0.0, tau_max: 0.4}
ligand_scaling_dict:
  L31: {mask: ":291", torsion_exclusions: auto}
sidechain_scaling_list: ":15-18,23,40,72,90-96,139-140,142,152-153"
```

and its protocol differs from ladder B's in two lines, `number_of_replicas: 12` and
`tau_max: 0.4`. `build-md` then writes a group file with twelve lines rather than eight:

```text
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state0.xml  -c eq/eq_3.xml --group-index 0
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state1.xml  -c eq/eq_3.xml --group-index 1
...
-i _protocol.py -p ../build/built.pdb -s ../build/REST2/system_state11.xml -c eq/eq_3.xml --group-index 11
```

Twelve rungs at two ranks per card is **six cards**, and the launch is `run.sh`'s mpirun line with
`-n 12 ... -ng 12`. Keep two ranks per card: a third is worse and a fourth loses throughput
outright.

## 5. What these runs measure

| quantity | ladder A | ladder B |
|---|---|---|
| hot atoms (of 4,701 solute atoms) | 32 | 193 |
| scaled torsion central bonds | 4 | 52 |
| scaled CMAP terms | 0 | 0 |
| ns/day per state | 93.5 | 96.3 |
| ns/day aggregate, 8 rungs on 4 cards | 748 | 771 |
| neighbouring-pair acceptance | 0.483–0.602, overall 0.549 | 0.358–0.408, overall 0.388 |
| walker round trips (8 walkers, 5 ns) | 105 | 45 |
| wall clock | 83 min | 81 min |

**The finding: six times the hot region, at half the τ_max, costs about a third of the acceptance
and half the mixing.** That is the trade, measured. Which side of it you want depends on whether
the pocket has to move with the ligand; neither ladder is "better".

!!! warning "B is not evidence that a bigger hot region is free"
    B's 96.3 ns/day against A's 93.5 is **not** a measurement that heating 193 atoms costs less
    than heating 32. The two ladders did not run against identical background load on a shared
    machine. What the numbers do support is that the per-step cost here is dominated by the
    53,030-atom box, which both ladders share: the hot region changes what is *sampled*, not what
    each step costs.

### Both runs are registered datasets

Every figure in the table above is read from these two, not retyped from a log:

```text
$MD_DATA/2026/tutorials/tyk2-ejm31-rest2-ligand-8x5ns          110 files, 545.8 MB   (ladder A)
$MD_DATA/2026/tutorials/tyk2-ejm31-rest2-ligand-pocket-8x5ns   117 files, 546.0 MB   (ladder B)
```

Each holds the whole tree a reader needs to check the claim -- `build/` with the scaled states and
`scaler.yaml`, the equilibration chain, and the ladder's own `REST2.nc`, `restart.json` and
per-state trajectories -- and each verifies against its own inventory:

```bash
md-openmm data-register --verify-only -idata <the dataset> \
    -project_name tutorials -data_name tyk2-ejm31-rest2-ligand-8x5ns -year 2026
```

!!! warning "Their energy check is a recorded FAIL, and the datasets say so"
    Gate G2-6 recomputes a ladder's recorded exchange energies against its saved states. Both runs
    FAIL it as originally written (0.077/0.093 kT and 0.106/0.121 kT against a 0.05 kT tolerance
    that had been calibrated on a system 30x smaller). Against the size-aware replacement
    `dU <= 3*K*sqrt(|u|)`, validated at a third system size, **ladder A is inside on both channels
    and ladder B is outside by 1.16x on the cross channel**. A deliberately wrong Hamiltonian is
    caught by four orders of magnitude more, so this is a question about a tolerance, not about
    the scaling. Each dataset's notes carry its own numbers; the full account is
    `docs/development/0.6.1/handoffs/S1.md`.

### How fast is eight rungs on four cards, really

**The naive expectation:** one rank alone does 330 ns/day on this system, two ranks share a card,
four cards — so something near 4 × 330 / 2 per rung.

**The measurement:** **93.5 ns/day per rung**, 748 ns/day aggregate. Each of the two ranks on a
card gets 57% of what one rank alone gets, so a shared card delivers about **1.13×** its
single-rank throughput, not 2×. Part of that is the sharing and part is the exchange barrier:
every rank waits for the slowest rung at every exchange, so a ladder runs at the pace of its
slowest card — which is also why the cards must be the same model.

Plan from the measured number, not from the single-rank one. Multiplying a single-rank figure by
the card count overestimates a ladder by about a factor of two.

**How the cost divides.** Three 100 ps probes, same system, same equilibrated start. **This table
is for comparing configurations with each other — do not size a campaign from its absolute
numbers; see the warning directly below it.**

| configuration | ranks per card | ns/day per rung | ns/day aggregate |
|---|---|---|---|
| one rank, no ladder at all | — | 330 | — |
| 4 rungs on 4 cards | 1 | 209.9 | 839.5 |
| 8 rungs on 4 cards | 2 | 117.6 | 940.8 |
| 8 rungs on 2 cards | 4 | 69.1 | 553.1 |

!!! warning "These probe numbers are 26% too high for planning"
    The 8-on-4 probe reads 117.6 ns/day per rung. The 5 ns production ladder of the same shape
    sustained **93.5**. A 100 ps probe does not pay the accumulated cost of a long run, so a
    campaign sized from this table comes out about a quarter short. Take the relative scaling from
    here and the absolute number from a production run.

The ladder machinery — exchange barriers, per-state trajectories, checkpoints — costs 36% before
any sharing (330 → 209.9). A second rank on a card gives back 1.12× that card's aggregate
throughput, not 2×.

**A fourth rank per card is a loss, not a smaller gain.** Eight rungs on two cards delivers 553.1
ns/day aggregate, *below* the 839.5 of four rungs on four cards and far below the 940.8 of eight
on four. Two ranks per card is the most this system rewards; three is untested and four is worse
than not sharing at all.

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
