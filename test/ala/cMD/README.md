# Alanine dipeptide, conventional MD

Two worked conventional-MD protocols on the vetted ACE-ALA-NME solute, one in each supported solvent
mode. Both are 1 ns of production run as **two committed 500 ps segments**, which is the point: a
single uninterrupted 1 ns process would show that integration works but would test none of the
segment boundary, checkpoint restart, append behaviour, or monotonic runtime indices.

`duration_per_segment: "500 ps"` here is a worked-test override. The named profile default is 5 ns.

|  | `explicit/` | `implicit/` |
|---|---|---|
| solvent | ff19SB / OPC box, 0.15 M NaCl, PME | ff19SB / GBn2 with mbondi3 |
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1 | min, eq_nvt, cMD_1 |
| production ensemble | NPT, 300 K, 1 bar | NVT, 300 K |
| timestep | 4 fs, HMR to 3.024 amu | 2 fs, no HMR |
| steps for 1 ns | 250,000 | 500,000 |
| volume / density | recorded | **not applicable** and recorded as null |

## Running one

```bash
python MD_system_gen.py -i ace_ala_nme.pdb -o bundle \
    --config test/ala/cMD/implicit/system_config.json
python MD_input_gen.py --system bundle/system_manifest.json -o run \
    --config test/ala/cMD/implicit/md_config.json
cd run && CMD_NUMBER_OF_SEGMENTS=2 ./run_all.sh
```

Equilibration runs once; the driver then invokes the cMD stage the requested number of times. To add
segments later, invoke `cMD_1/cMD_1.sh` directly — see `*/extension/`.

Select a GPU explicitly and pin the device ordering, or the index you record will not be the device
you used:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID     # otherwise CUDA orders by FASTEST_FIRST
MD_DEVICES=0 CMD_NUMBER_OF_SEGMENTS=2 ./run_all.sh
```

## What a completed run should show

```
2 committed generations
absolute step 250,000 (explicit) or 500,000 (implicit)
absolute time 1000.0 ps
restart source: checkpoint
10 all-atom frames at the 100 ps cadence, 100 selected-atom frames at 10 ps
one log header, monotonic rows, no NaN
```

A run that reaches these numbers has executed correctly. **It has not been scientifically
validated**: 1 ns is not convergence for anything, and nothing in this directory supports a claim
about alanine dipeptide's conformational preferences.
