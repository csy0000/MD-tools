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
   and every method reuses it.
2. Four GPUs, and the MPI/MPS rules in [placement](../../../machine-configuration.md): eight
   states over four cards is two ranks per card, which needs MPS.

## 1. Which residues line the pocket

Ask, read, paste. The helper prints; it never resolves a pocket at run time.

```bash
python -m md_tools.rest2.pocket build/built.pdb --ligand ":<ligand residue index>" --cutoff-nm 0.5
```

```text
TO BE MEASURED: the printed table and mask for TYK2 + ejm_31
```

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
TO BE MEASURED: the resolved residue map, and the number of hot atoms and scaled torsion bonds
```

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
