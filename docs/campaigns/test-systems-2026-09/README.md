# md-tools test datasets, 2026-09

Short runs of two small systems through every protocol md-tools has a reference exporter for, as
worked examples and as test data: something to open, analyse and export without committing to a
production-length campaign.

| system | input | route |
|---|---|---|
| ALA | `tests/data/ALA.pdb`: ACE-ALA-NME, exactly tleap's `sequence { ACE ALA NME }` | ff14SB |
| phenol | `phenol.smi`: `Oc1ccccc1` | Sage 2.2.1, AM1-BCC |

Each system in implicit (GBn2) and explicit (TIP3P, 0.15 M NaCl) solvent, and in each solvent:

| method | protocol | length |
|---|---|---|
| `cMD-cold` | cMD at τ = 0 | 1 ns |
| `cMD-hot` | cMD at τ = 0.5 | 1 ns |
| `REST2` | linear τ 0 → 0.5, exchange every 10 ps | 1 ns per state; 4 states implicit, 6 explicit |

2 fs, 300 K, HBonds, no hydrogen mass repartitioning. Equilibration follows the protocol the ALA
campaign settled on 2026-09-11: 20 ps restrained NVT in implicit solvent; 10 ps restrained NVT then
10 ps restrained NPT in explicit solvent; restraints 1 kcal/mol/Å² on the solute. Solute
coordinates every 5 ps; explicit whole-system coordinates every 10 ps.

## Things to know

* **An explicit cold run is NPT, and every other run is NVT**, including every REST2 state. The
  explicit cold run and the bottom state of the explicit ladder are therefore not the same
  ensemble.
* **The explicit hot run starts from the explicit cold run's box.** A run at τ > 0 is fixed-volume
  and cannot equilibrate its own box, and build-top's is below the equilibrium density.
  `run_tests.py` writes `cMD-hot/run_from_cold.sh`: build-md's `run.sh` with minimisation removed
  and the first stage continuing from the cold run's last pressure-coupled state. It is part of the
  registered dataset.
* **The SMILES first proposed for the ligand**, `OC1C2OOC1C([O])(O)C=C2`, is C6H7O5 with one
  unpaired electron -- a radical, which AM1-BCC cannot charge. Phenol was used instead.

## Running it

In the activated `openmm-env`, with no `PYTHONPATH`:

```bash
python run_tests.py prepare  --root <staging dir>
python run_tests.py run      --root <staging dir> --gpus <devices> --no-ladder-gpus <devices>
python run_tests.py export   --root <staging dir>
python run_tests.py register --root <staging dir> --project-repo <MD-tools checkout>
```

`--no-ladder-gpus` keeps a ladder off a device of a different card model, so no rung runs on
different hardware from the others. Devices already running something are skipped whoever started
it. `export` writes each run's OpenMM-only bundle into `bundle/` inside it; `register` moves each
method directory to

```text
$MD_DATA/2026/md-tools/<system>-test/<method>-<solvent>
```

## Files

| path | what |
|---|---|
| `configs/sys.*.config` | the `build-top` configurations, by route and solvent |
| `configs/{cold,hot,rest2}_{implicit,explicit}.config` | the `build-md` configurations |
| `phenol.smi` | the ligand |
| `run_tests.py` | prepare, run, export, register |
