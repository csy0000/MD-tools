# `docs/implementation/` — what this directory is

> **HISTORICAL — not current usage.**
>
> This describes the explicit-solvent baseline as it stood before the repository was reduced to the
> six-command CLI in `ca29fcd`. Its runnable scripts have been deleted: they were a second,
> unmaintained copy of the run scripts, and they never received the REST2 duration, equilibration,
> seed and trajectory corrections of 2026-08-25. Running them would reproduce those bugs.
>
> It is kept because `reports/explicit_solvent/` cites it, and a retained validation report needs
> the document that explains what was run. For current usage see the root `README.md`.


Setups that are **specified here and implemented in `src/`**: the runnable definition of how a
production calculation is prepared and launched, as opposed to `docs/theory/` (why an estimator is
correct), `docs/protocols/` (conventions that bind every run) or `docs/decisions/` (records of
choices that were made once).

A document here is authoritative for its setup. If it disagrees with
`docs/protocols/conditional_bar_conventions.md`, that file wins.

| document | scope | status |
|---|---|---|
| `baseline_setups.md` + `scripts/` | explicit-solvent REST2-REMD reference and free cold/hot walkers | **new, under review**, no production run yet |

---

## The explicit-solvent baseline — a reviewer's guide

Written 2026-08-14 in response to: *move to explicit solvent, this is the real run, REST2 and free
cold/hot walkers, not yet cBAR*, with eight specifications (print frequencies, exchange frequency,
chunk sizes, run lengths, force fields, box and cutoff, initial structure, and the file layout).

### Where things are

```
baseline_setups.md                the specification + every decision, with numbers
scripts/generate_config.py        writes the per-stage configs (-h lists everything)
scripts/simbox-setup.py           (a) SMILES or PDB -> parameterised solvated box
scripts/min-eq.py                 (b) minimise + equilibrate -> the state (c) and (d) start from
scripts/md.py                     (c) one free walker, cold or hot by config
scripts/md_REST2.py               (d) REST2-REMD reference
scripts/run_all.sh                (a)+(b) then (c)/(c)/(d), one per GPU
scripts/config_defaults.json      the whole parameter tree, dumped
src/md_tools/systems/explicit_baseline.py     the implementation (all of it)
```

Stage **(e)** — `md_cBAR.py` and `pREST2.py` — is not built. `generate_config.py --cbar/--prest2`
refuses and writes nothing rather than emitting a config for a script that cannot read it.

The two flags every stage after (a) takes: **`--p`** is the parameterised `System` (the analogue of
an Amber prmtop; its sibling `<stem>_topology.pdb` and `<stem>_simbox.json` are read alongside,
since a `System` XML carries no atom names), and **`--c`** is coordinates — a PDB, or a serialised
OpenMM `State` that also carries velocities and the box.

The scripts contain no scientific logic, per the project's code-boundary rule. Reviewing
`explicit_baseline.py` and `baseline_setups.md` covers everything.

### The REST2 ladder

Spaced evenly in `√s` between `s_cold = 1` and `s_hot = 0.25`:

| `--system` | rungs | ladder | `T_eff` on the solute (K) |
|---|---:|---|---|
| `alanine` | **6** | `[1.0, 0.81, 0.64, 0.49, 0.36, 0.25]` | 300, 370, 469, 612, 833, 1200 |
| `macrocycle` | **8** | `[1.0, 0.8622, 0.7347, 0.6173, 0.5102, 0.4133, 0.3265, 0.25]` | 300, 348, 408, 486, 588, 726, 919, 1200 |
| `cyclo_rgdfv` | **10** | `[1.0, 0.8920, 0.7901, 0.6944, 0.6049, 0.5216, 0.4444, 0.3735, 0.3086, 0.25]` | 300, 336, 380, 432, 496, 575, 675, 803, 972, 1200 |

The alanine 6-rung ladder is exactly the one every existing reference in this project used.

* The generic **8-rung `macrocycle` ladder is an UNVALIDATED starting ladder** — no pilot supports it.
* The **10-rung evidence applies specifically to cyclo-RGDfV** and is not transferable to other
  macrocycles. `--system cyclo_rgdfv` additionally *enforces* the Sage 2.2 / AM1-BCC ligand route:
  a PDB invocation fails before any force field is built.
* **RGD's 2 fs vs 4 fs comparison is an unresolved production gate**, not an optional confirmation.
  4 fs is not validated merely because HMR and the middle integrator are present.

Override with `--s_cold / --s_hot / --N_rungs / --interp`, or by setting
`production.remd.scale_factors` outright.

### Environment and versions

`environment.yml` in this directory pins what the baseline was validated against:

```bash
conda env create -f docs/implementation/explicit_solvent/environment.yml
conda activate md-tools
pip install -e . --no-deps            # from the repository root
```

The existing repo-root `environment.yml` (unpinned, `md-tools`) also works and is what the
measurements below were actually made in; the pinned file exists so a future reader can reproduce
them exactly.

| package | version | load-bearing for |
|---|---|---|
| python | 3.11.15 | |
| **openmm** | **8.5.1** | the engine. `Modeller.addSolvent(boxShape='dodecahedron')` and `DCDReporter(atomSubset=…, enforcePeriodicBox=…)` are both required, so an older OpenMM may simply not have them |
| openmmtools | 0.26.0 | NEQ integrators — not used by this template |
| **openmmforcefields** | **0.16.0** | `SMIRNOFFTemplateGenerator`, which supplies Sage 2.2 (`openff-2.2.0`) |
| openff-toolkit | 0.17.1 | `Molecule`, charge assignment, the toolkit registry |
| openff-interchange | 0.4.5 | SMIRNOFF → OpenMM conversion under the generator |
| **ambertools** | **24.8** | `sqm`, i.e. **AM1BCC**. Must be on `PATH` — see below |
| parmed | 4.3.1 | the enforced GBn2/mbondi3 implicit route (not used here) |
| **rdkit** | **2025.03.6** | ETKDGv3 embedding and MMFF94s minimisation |
| **pymbar** | **4.2.0** | the analysis that follows: MBAR + Zwanzig hot term, all-replica MBAR cold reference |
| mdtraj | 1.11.1 | reading trajectories and extracting CVs — **no longer used by the writers** |
| numpy / scipy / pandas | 2.4.6 / 1.17.1 / 2.3.3 | |
| matplotlib / scikit-learn | 3.10.9 / 1.9.0 | figures, clustering |
| cuda-version | 13.0 | OpenMM reports platforms `Reference, CPU, CUDA, OpenCL` |

Two things that bite:

* **AmberTools must be on `PATH`, so activate the environment.** The OpenFF toolkit discovers
  `AmberToolsToolkitWrapper` by looking for `sqm`/`antechamber` on `PATH`. In a bare shell they are
  not found even though they are installed, and AM1BCC silently becomes unavailable.
  `build_forcefield()` checks the registry and **fails loudly** rather than letting the charges fall
  back to another method while still being labelled AM1BCC.
* **pymbar 4.2 prints a JAX banner on import** and enables JAX's 64-bit mode when called. It is
  harmless here, but it is global: anything else in the same process using JAX in 32-bit mode will
  be affected.

`pymbar` has also been added to the repo-root `environment.yml`. It was previously present only as
a transitive dependency of `openmmtools`, which is not something the analysis should rely on.

### Read in this order

1. **`baseline_setups.md` §1**, the table mapping each of the eight specifications to what it
   resolves to — and then the four subsections **(a)–(d)** immediately under it. Those are the
   points where the specification as given could not be implemented literally, and they are the
   parts most worth disagreeing with.
2. **§4, equilibration.** Rewritten after review: the first version was the small-rigid-solute
   protocol and was wrong for a macrocycle.
3. **§7, what is verified and what is not.** Short, and the honest part.

### What is reused rather than rewritten

Everything carrying scientific meaning already existed and is called unchanged:

* `systems/openmm_system.build_rest2_scaled_system` — the REST2 Hamiltonian (solute charges × √s,
  epsilons × s, solute–solvent exceptions × √s, torsions and ff19SB CMAP × s);
* `systems/topology_prep.resolve_remd_scale_ladder` — ladder validation, `replica 0 == s = 1`;
* `methods/md_run.exchange_pairs` / `attempt_rest2_exchange` — the neighbour schedule and the
  Metropolis criterion, already box-vector aware.

So acceptance and round-trip statistics are comparable with the existing implicit-solvent
references, and `analysis/remd_reliability.py` reads the new `exchange_attempts.csv` with no
converter (checked).

This baseline deliberately does **not** go through `topology_prep.build_topology`: its explicit
route is tleap, OPC water only, isometric box only, and it cannot mix ff19SB residues with a
SMIRNOFF ligand. The OpenMM `ForceField`/`Modeller` route was required by the specification and
does all three.

### The four decisions to check first

| | issue | what was done |
|---|---|---|
| (a) | "MMFF99" does not exist in RDKit | **MMFF94s**, matching `systems/etkdg.py` |
| (b) | `padding = 1.2 nm` + `cutoff = 1.0 nm` **cannot run** as literally specified — a dodecahedron's minimum perpendicular height is `width/√2` = 1.697 nm, so the max legal cutoff is 0.85 nm. (Corrected 2026-08-24: the blocker is the *height*, not the solute–image gap, which is 1.456 nm — the shortest lattice translation is `width` for every OpenMM box shape.) | box width solved for the *requested* gap; both readings selectable, all numbers recorded |
| (c) | 4 fs + **leapfrog** Langevin — 4 fs is normally justified with the middle/BAOAB scheme | **resolved 2026-08-14: `langevin-middle` (BAOAB) is now the default**; leapfrog remains available for reproducing earlier runs |
| (d) | a peptide cannot come from SMILES (ff19SB matches by residue, RDKit gives one `UNL`) | `--smi` = ligand route (Sage 2.2 + AM1BCC), `--pdb` = peptide route (ff19SB) |

### Verified end to end (solvated alanine dipeptide, 1832 atoms, CPU, short times)

Both input routes build a System; dodecahedron geometry, `addHydrogens` at pH 7, 0.15 M NaCl; HMR
conserves total mass to 1e-6 amu and never touches water; omega central-bond detection returns the
two backbone amides; the staged equilibration runs all nine stages; the cold walker, the hot walker
at `s = 0.25`, and a 3-replica REMD with exchange all run; resume detection works; reporter frame
counts are exactly 10 ps / 2 ps.

### Not verified — do not read the above as more than it says

* **the REST2 ladder in explicit solvent.** The alanine 6-rung default is inherited from the implicit work,
  and REST2's solute–solvent cross term is far larger here. A ladder whose weakest neighbour pair
  does not exchange makes a 1 µs reference worthless. **Run a ~2 ns acceptance pilot first.**
* **the staged equilibration on an actual macrocycle** — alanine's solute barely moves (RMSD
  0.051 nm), which is not a test of the protocol's purpose.
* ~~**ff19SB + TIP3P-FB** — ff19SB was parameterised against OPC.~~ **Resolved 2026-08-21:** the
  default water now follows the solute — OPC where a peptide is present, TIP3P for a ligand-only
  system — so each force field gets the water it was validated with. The TIP3P-FB `-v1` profiles
  stay name-resolvable so existing runs remain reproducible.
* **GPU timings**, so the wall-clock cost of 1 µs is unknown; `done.json` records ns/day per chunk,
  so the first real chunk answers it.
* **the `"complex"` route** (ff19SB residues + SMIRNOFF ligand in one system).

### Known deviations from project convention

* Output goes to `<data_root>/explicit_baseline/<run.name>/`, **not** the standard
  `YYYYMMDDTHHMMSS__<system>__<method>__<config_hash>` run ID. Each step does write git commit,
  dirty flag, command, OpenMM version and the resolved config. If these become canonical production
  runs they should probably carry real run IDs.
* Storage has no budget set. Alanine measures ~2.3 GB per µs, so cold + 6 replicas ≈ 16 GB; a
  solvated macrocycle is 4–7 k atoms, i.e. **35–60 GB per system**.

### Corrections already folded in

Two claims made during development were wrong and are corrected in place, noted here so a reader of
the git history is not misled:

* wrapping the solute trajectory was said to fragment the molecule and corrupt CVs. It does not —
  OpenMM's `enforcePeriodicBox` wraps whole *molecules* (measured: max bonded distance 0.153 nm
  under both settings). The solute stream is still written unwrapped, but because wrapping teleports
  the molecule across the box between frames, which breaks continuous-trajectory analyses and
  viewing — not because bonds break. **No existing project trajectory is affected.**
* the first equilibration was minimise → NVT → NPT with no restraints or heating ramp. That is the
  small-rigid-solute protocol and is not appropriate for a macrocycle from a gas-phase conformer;
  it survives as `equilibration.protocol = "simple"`.
