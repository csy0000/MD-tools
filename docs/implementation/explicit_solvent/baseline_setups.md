# Explicit-solvent baseline setups

**Status:** authoritative for explicit solvent · **Written:** 2026-08-14 · **Scope:** REST2-REMD
references and free cold/hot walkers (conditional BAR is *not* part of this baseline yet)

Implementation: `src/md_templates/systems/explicit_baseline.py`.
Runnable stages: `scripts/` — `simbox-setup.py`, `min-eq.py`, `md.py`, `md_REST2.py`, with
`generate_config.py` writing their configs. Every parameter lives in one tree;
`scripts/config_defaults.json` is that tree dumped, and each stage's config carries only the values
that differ.

```bash
conda activate md-templates  # REQUIRED -- see "AmberTools must be on PATH" below
S=docs/implementation/explicit_solvent/scripts

# configs: 4 REST2 rungs for alanine, 6 for macrocycles
python $S/generate_config.py --all --system macrocycle

# (a) simulation box: SMILES (Sage 2.2) or PDB (ff19SB)
python $S/simbox-setup.py --smiles "$SMILES" --out-suffix simbox --config simbox-config.json
# (b) minimise + equilibrate
python $S/min-eq.py    --p simbox_system.xml --c simbox_topology.pdb \
                       --out-suffix eq    --config min-eq-config.json
# (c) free walkers: cold (1 us, s=1) and hot (200 ns, s=0.25)
python $S/md.py        --p simbox_system.xml --c eq_state.xml \
                       --out-suffix cold  --config md-cold-config.json
python $S/md.py        --p simbox_system.xml --c eq_state.xml \
                       --out-suffix hot   --config md-hot-config.json
# (d) REST2-REMD reference
python $S/md_REST2.py  --p simbox_system.xml --c eq_state.xml \
                       --out-suffix rest2 --config rest2-config.json
# (e) md_cBAR.py and pREST2.py -- future
```

`--p` is the parameterised `System` (the analogue of a prmtop; its sibling `<stem>_topology.pdb` and
`<stem>_simbox.json` are read alongside, since a `System` XML has no atom names). `--c` is
coordinates: a PDB, or a serialised OpenMM `State` that also carries velocities and the box.


Each step writes `manifest_<step>.json` (git commit + dirty flag, command line, OpenMM version,
host, timestamp) and a copy of the fully resolved config beside its outputs, so a run is
reproducible from its own directory and a later edit to the defaults cannot silently reinterpret it.
**Unknown config keys raise** rather than being ignored — a misspelled key would otherwise leave the
baseline value in force while the run's own `config.json` claimed otherwise.

---

## 1. The requested setup, and what it resolves to

| # | requested | implemented as | verified |
|---|---|---|---|
| 1 | print `f_all = 10 ps`, `f_solute = 2 ps` | `DCDReporter` every 2500 steps (all atoms, wrapped) + `DCDReporter(atomSubset=solute, enforcePeriodicBox=False)` every 500 steps — the solute trajectory is **unwrapped**, see §5 | ✔ frame counts + whole-molecule check |
| 2 | exchange every 10 ps | 2500 steps between neighbour-swap rounds | ✔ `exchange_attempts.csv` |
| 3 | chunks: 100 ns cold/REST2, 1 ns hot | one directory per chunk, own trajectories + `end.chk` | ✔ resume test |
| 4 | 1 µs REST2 + cold (10 chunks), 200 ns hot (200 chunks) | `production.{cold,remd}.total_ns = 1000`, `hot.total_ns = 200` | ✔ chunk arithmetic asserted |
| 5 | ff19SB / Sage 2.2 + AM1BCC / TIP3P-FB + 0.15 M NaCl / HMR + HBonds / 4 fs | `amber19/protein.ff19SB.xml`, `openff-2.2.0` via SMIRNOFF, `amber19/tip3pfb.xml`, manual solute-only HMR, `openmm.LangevinMiddleIntegrator` at 4 fs (see (c)) | ✔ ran |
| 6 | dodecahedron, 1.2 nm padding, 1.0 nm cutoff both terms, `addHydrogens` at pH 7 | `boxShape` solved for the requested gap (below), `PME`, `nonbondedCutoff = 1.0 nm` | ✔ ran |
| 7 | RDKit ETKDGv3 then MMFF, 1000 steps | `rdDistGeom.ETKDGv3` + `MMFFGetMoleculeForceField(...).Minimize(maxIts=1000)` | ✔ ran |
| 8 | one document, one script per stage, config by JSON | this file + `scripts/` (a–d; e not built) | ✔ ran |

### Four points where the specification needed a decision — read these

**(a) "MMFF99" does not exist; RDKit implements MMFF94 and MMFF94s.** The baseline uses
**MMFF94s** (`structure.mmff.variant`), the variant with the planarised sp³-nitrogen/amide
parameters, which is the right choice for peptides and macrocycles and is what
`systems/etkdg.py` already used for every conformer library in this project. `MMFF94` is available
by changing that one key. Nothing named MMFF99 is reachable — if a different force field was meant
(MMFF94s is not MM3, MMFF99 in some commercial codes means MMFF94s), say so and it is a one-line
change.

**(b) `padding = 1.2 nm` does not mean 1.2 nm of water — and it collides with the 1.0 nm cutoff.**
OpenMM's `Modeller.addSolvent(padding=p)` sets the box width to `max(2·radius + p, 2·p)`, and a
rhombic dodecahedron's *minimum image distance* is `width/√2`, not `width`. Measured on solvated
alanine dipeptide (bounding radius 0.472 nm):

| | raw OpenMM padding | this baseline (`solute-image-gap`) |
|---|---|---|
| box width | 2.400 nm | 3.033 nm |
| minimum image distance | 1.697 nm | 2.144 nm |
| **solute-to-image gap** | **0.753 nm** | **1.200 nm** |
| largest legal cutoff | 0.849 nm | 1.072 nm |
| box volume | 9.8 nm³ | 19.7 nm³ |
| waters | 285 | 602 |
| **1.0 nm cutoff** | **hard OpenMM error** | fits |

So the raw reading of the spec cannot run at all: `Context` construction dies with *"the cutoff
distance cannot be greater than half the periodic box size."*

#### What "solving the box width for the requested gap" means

A periodic box is one number — the **width** `w` — plus a shape. OpenMM's reduced vectors for a
rhombic dodecahedron are

```
a = (w,   0,   0)
b = (0,   w,   0)
c = (w/2, w/2, w/√2)
```

In a reduced triclinic cell the **minimum image distance** — the shortest distance from any point to
the nearest copy of itself — is the smallest diagonal element, `min(a_x, b_y, c_z)`. For a
dodecahedron that is `c_z = w/√2 ≈ 0.707·w`, *not* `w`. Write that fraction as `frac`: 1 for a cube,
`1/√2` for a dodecahedron, `√6/3 ≈ 0.816` for a truncated octahedron.

Now "the box is padded by 1.2 nm from the solute" means: along the tightest direction there should
be at least 1.2 nm of solvent between the solute and its nearest image. The solute spans `2r`, where
`r` is its bounding radius from the centroid, and the centre-to-image-centre distance is `frac·w`,
so the solvent gap is

```
gap  =  frac·w  −  2r
```

Setting `gap = padding` and solving for the unknown gives the width the baseline actually uses:

```
w  =  (2r + padding) / frac
```

For alanine dipeptide, `r = 0.472 nm` and `padding = 1.2 nm`:

```
w        = (2×0.472 + 1.2) × √2  = 3.033 nm
min image = 0.707 × 3.033        = 2.144 nm
gap       = 2.144 − 0.944        = 1.200 nm   <- the requested padding, delivered
max cutoff = 2.144 / 2           = 1.072 nm   <- the 1.0 nm cutoff fits
```

against OpenMM's own `w = max(2r + p, 2p) = max(2.144, 2.400) = 2.400 nm`, whose minimum image is
1.697 nm, whose gap is only 0.753 nm, and whose largest legal cutoff is 0.849 nm — below the 1.0 nm
requested, hence the hard error.

So "solved for the requested gap" is just: *treat the padding as the physical quantity it names, and
compute whatever box width delivers it, given the shape.* It is one line of algebra, and it happens
to make the 1.0 nm cutoff fit with room to spare. `solvation.padding_semantics = "openmm"` restores the
literal `addSolvent` behaviour; `solvation.cutoff_fit_policy` (`"grow"` default, or `"refuse"`)
decides what happens if a box still cannot hold the cutoff. Both the requested and realised
geometry are recorded in `solvation.json → geometry`. The cost is real: the box roughly doubles.

**(c) 4 fs runs on the middle (BAOAB) integrator, not leapfrog — decided 2026-08-14.** The
specification asked for leapfrog. The reason 4 fs is considered safe with HMR + HBonds is almost
always `openmm.LangevinMiddleIntegrator`, which places the position update between two half-kicks;
the legacy leapfrog Langevin integrator samples a slightly *hot* configurational distribution at
long timesteps, and this project's estimand is a free-energy difference between two conformational
cells — exactly the quantity a systematic sampling-temperature error shifts. The middle scheme costs
nothing per step. **`integrator.kind = "langevin-middle"` is therefore the default**;
`"leapfrog-langevin"` remains available and is what any run made before 2026-08-14 used.

With the middle integrator the 4 fs + HMR + HBonds + rigid-water combination is the standard
one — but **that is not validation**. For cyclo-RGDfV the 2 fs vs 4 fs comparison is an
**unresolved production gate**, not an optional confirmation, and it has not been run. Do not treat
4 fs as validated merely because HMR and the middle integrator are present.

**(d) A peptide must be supplied as a PDB; `--smi` is the ligand path.** ff19SB assigns parameters
by *residue template*, and an RDKit structure built from SMILES is a single residue named `UNL` —
ff19SB cannot see a peptide in it (verified: `No template found for residue 0 (UNL)`). So:

* `--smi "$SMILES"` → ETKDGv3 + MMFF94s → **Sage 2.2 + AM1BCC** (the ligand/macrocycle route).
* `--pdb $pdbfile` with proper residue names → **ff19SB** (the peptide route); ETKDG is skipped,
  because an experimental or already-prepared structure should not be re-embedded.

`system.solute_kind = "auto"` decides from the residue names and is recorded. A protein–ligand
complex (`"complex"`) gets ff19SB for the residues it recognises and SMIRNOFF for the rest, but no
such system exists in this project yet and that path is **untested**.

---

## 2. Force field

| component | choice | file / source |
|---|---|---|
| peptide | **ff19SB** | `amber19/protein.ff19SB.xml` (ships with OpenMM 8.5) |
| ligand / macrocycle | **Sage 2.2** (`openff-2.2.0`) | `SMIRNOFFTemplateGenerator`, openmmforcefields 0.16 |
| ligand charges | **AM1BCC** | AmberTools `sqm` via the OpenFF toolkit |
| water | **TIP3P-FB** | `amber19/tip3pfb.xml` |
| ions | Na⁺ / Cl⁻ at **0.15 M** + neutralising counterions | `addSolvent(ionicStrength=...)` |

**`solvation.water_model = "tip3p"` is not a typo.** That argument selects the 3-site water
*geometry template* Modeller packs into the box; the *parameters* come from whichever water XML the
force field loaded, here `tip3pfb.xml`. TIP3P-FB is a reparameterisation of TIP3P with identical
topology, so this is the documented way to build a TIP3P-FB box.

**AmberTools must be on `PATH`.** The OpenFF toolkit registry discovers `AmberToolsToolkitWrapper`
by looking for `sqm`/`antechamber` on `PATH`. In a bare shell they are not found even though they
are installed in the `md-templates` environment, and AM1BCC then silently becomes unavailable.
`build_forcefield` checks the registry and **fails loudly** rather than letting the charges fall back
to a different method under an AM1BCC label. Always `conda activate md-templates`.

**ff19SB was parameterised with OPC water, not TIP3P-FB.** The combination requested here is what
this baseline implements, and it is a common and defensible pairing (TIP3P-FB is markedly better
than TIP3P), but it is *not* the pairing the ff19SB paper validated. ff19SB's CMAP corrections were
fit against OPC, and helical/backbone propensities are the properties most sensitive to it. For a
dipeptide or a macrocycle whose estimand is a torsional-cell population this is a second-order
concern; it should be stated in any report built on this baseline rather than left implicit.
`forcefield.water` switches to `amber19/opc.xml` if that trade is ever preferred (OPC is 4-site, so
`solvation.water_model` must become `"tip4pew"` and the virtual-site guard in
`md_run.run_remd` becomes relevant).

**REST2 charge scaling and net charge.** REST2 scales solute charges by `√s`. If the solute has
zero net charge this preserves neutrality; if it does not, the scaled system is non-neutral and PME
silently applies a uniform background charge that differs between rungs. All current systems
(alanine dipeptide, cyclo-(RGDfV), cilengitide, rgd_redPheVal, cyclosporin) are net neutral, so this
does not bite today — but a charged solute needs the counterions folded into the scaled set or the
ladder is not a ladder of the same system. The net charge of a SMIRNOFF ligand is recorded in
`solvation.json → forcefield.ligand.net_charge_e`.

---

## 3. System construction

```
nonbondedMethod       PME
nonbondedCutoff       1.0 nm      (electrostatics AND Lennard-Jones)
switchDistance        None        -> hard vdW cutoff; set system_build.switch_distance_nm to soften
useDispersionCorrection  True     (analytic long-range LJ; matters for density)
ewaldErrorTolerance   5e-4
constraints           HBonds
rigidWater            True
removeCMMotion        True
```

### Hydrogen mass repartitioning

`m_H → 3.024 amu` (= 3 × 1.008), with the bonded heavy atom losing exactly what the hydrogen gains,
so the **total mass is conserved** (asserted; the function raises if it is not). With `HBonds`
removing the X–H stretches, the fastest remaining motions are the angle bends involving hydrogen;
lowering their frequency is what makes 4 fs viable.

**Water is never repartitioned**, at any `hmr_scope`. Rigid water is fully constrained, so its
hydrogen masses do not limit the timestep, and changing them would alter water's rotational
dynamics — and therefore its diffusion constant and dielectric relaxation — for no benefit.
`system_build.hmr_scope = "solute"` (default) repartitions only within the solute;
`"all"` extends to any non-water hydrogen, i.e. also a co-solute.

Measured on solvated alanine dipeptide: 12 hydrogens repartitioned, total mass 11106.281048 amu
before and after, 1832 particles / 1818 constraints / 3675 degrees of freedom.

### Omega-selective REST2

REST2 here is **omega-selective**, matching the rest of the project: the secondary-amide `C–N`
torsions the REMD ladder leaves unscaled must also be unscaled in the walled/scaled systems, or the
hot rungs isomerise cis/trans where the reference never does. The central bonds are found from the
topology (a carbon bonded to exactly one oxygen and to a nitrogen — a topology carries no bond
orders) and the detected list is written into `system_build.json → omega_central_bonds` so it can be
**audited rather than trusted**. On ACE-ALA-NME it finds `[(4, 6), (14, 16)]`, the two backbone
amides. Set `rest2.omega_exclusion = false` to scale everything.

The scaling itself is `openmm_system.build_rest2_scaled_system`, unchanged and shared with the
implicit-solvent path: solute charges × `√s`, solute epsilons × `s`, solute–solute exceptions × `s`,
solute–solvent exceptions × `√s`, solute torsions and CMAP × `s`. For a pairwise nonbonded force
that gives `U_s = s·U_solute + √s·U_solute-solvent + U_solvent`, the standard REST2 Hamiltonian.
At `s = 1` the function is bypassed entirely and the system is a plain deep copy.

---

## 4. Equilibration and the production ensemble

`equilibration.protocol = "staged"` (**default**) is the standard protocol for a flexible solute
dropped into a freshly packed water box — which is what an ETKDG/MMFF conformer of a macrocycle is:

| # | stage | restraint (kJ/mol/nm²) | length | dt |
|---|---|---|---|---|
| 1 | minimise, solute heavy atoms restrained | 4184 (≈10 kcal/mol/Å²) | to convergence | — |
| 2 | minimise, free | 0 | to convergence | — |
| 3 | heat 50 → 300 K in 25 windows, NVT | 4184 | 200 ps | **1 fs** |
| 4 | NPT, restrained | 4184 | 200 ps | 2 fs |
| 5 | NPT, restraint released in steps | 1046 → 418 → 0 | 3 × 200 ps | 2 fs |
| 6 | NPT, free — its tail sets the production box | 0 | 1000 ps | 2 fs |

**2.0 ns total**, negligible against 1 µs. Then production: NVT at that box, 4 fs.

Why each piece, since none of it is decoration:

* **Restrained minimisation first.** `Modeller.addSolvent` places water on a lattice-like
  arrangement, so the first shell is badly packed and carries large local forces. Minimising the
  whole system at once lets the *conformer* deform to accommodate a bad shell rather than the other
  way round. Restraining the solute heavy atoms puts the relaxation where it belongs, and the free
  minimisation afterwards removes any residual strain the restraint introduced.
* **The restraint reference is the minimised structure, not the input.** Otherwise stage 3 would
  pull the solute back toward a geometry that has already been improved.
* **1 fs during heating.** Instantaneous 300 K velocities on an unrelaxed box is the classic way to
  lose a run; the ramp plus a short timestep costs 200 ps.
* **Staged release, not a switch-off.** Going 10 → 0 kcal/mol/Å² in one step gives the solute the
  full accumulated restraint strain back at once.
* **`periodicdistance` in the restraint**, so an atom that crosses a box face is measured to its own
  reference point and not to an image of it.

`protocol = "simple"` (minimise → NVT → NPT, no restraints, no ramp) is retained and is appropriate
for a **small rigid solute such as alanine dipeptide** — and nothing larger.

**What equilibration does not do.** It does not converge the conformer. The starting structure is
one arbitrary point in the macrocycle's conformational space, and no equilibration protocol
equilibrates cis/trans amides, ring pucker or rotamers — that is what the REST2 ladder and the
production length are for. The solute heavy-atom RMSD from the minimised structure is therefore
recorded **per stage** in `equilibration.json`, so it is visible how far the conformer moved and
whether the restraints were doing anything. Measured on solvated alanine (short test):

```
initial            V = 19.72 nm^3   RMSD 0.000 nm
min_restrained     V = 19.72        RMSD 0.010
min_free           V = 19.72        RMSD 0.016
nvt_heat           V = 19.72        RMSD 0.015
npt_restrained     V = 18.58        RMSD 0.012      <- density collapses here
npt_release_k1046  V = 18.25        RMSD 0.009
npt_release_k418   V = 18.17        RMSD 0.023
npt_release_k0     V = 18.47        RMSD 0.044
npt_free           V = 18.89        RMSD 0.051
```

### The handoff to production goes through the State, not the checkpoint

The equilibration System carries two forces the production System does not — a
`MonteCarloBarostat` and the positional restraint. An OpenMM checkpoint is only valid for the exact
System that wrote it, so `loadCheckpoint(equilibrated.chk)` across that boundary is undefined
behaviour: it happened to work before the restraint force existed, which is worse than failing.
Production now reads `equilibrated_state.xml` and copies **positions, velocities and box vectors**
explicitly, which is System-agnostic and cannot silently mismatch. The checkpoint is still written,
for restarting equilibration itself.

**Why production is NVT.** Two reasons, both structural rather than preferential:

1. `attempt_rest2_exchange` — the exchange criterion this project's references were all produced
   with — has no `PV` term. An NPT ladder needs one, and adding it would make the new references
   incommensurable with the existing ones.
2. The cold free walker is compared directly against the REMD `s = 1` rung. If one were NPT and the
   other NVT they would be different ensembles and the comparison would carry an ensemble
   difference on top of the sampling difference it is meant to measure.

`production.ensemble` accepts only `"NVT"`; anything else raises rather than being quietly ignored.

**The box comes from an average, not a snapshot.** A single instantaneous NPT volume carries the
full volume fluctuation — measured on solvated alanine, 18.34 ± 0.12 nm³ over 20 samples, i.e.
±0.7 %, which would go straight into the production density. The mean over the last 500 ps is used
instead, and the sd is recorded so the choice is checkable.

---

## 5. Production runs

| | cold walker | hot walker | REST2-REMD |
|---|---|---|---|
| script | `md.py --config md-cold-config.json` | `md.py --config md-hot-config.json` | `md_REST2.py` |
| `s` | 1.0 | 0.25 (T_eff = 1200 K on the solute) | 6 alanine / 8 macrocycle / 10 cyclo_rgdfv, §6 |
| length | 1 µs | 200 ns | 1 µs per replica |
| chunk | 100 ns × 10 | **1 ns × 200** | 100 ns × 10 |
| exchange | — | — | every 10 ps |
| output | `cold/chunk_0000..0009/` | `hot/chunk_0000..0199/` | `rest2/replica_00../chunk_0000..0009/` |

Each chunk directory holds `traj_all.dcd` (10 ps), `traj_solute.dcd` (2 ps), `state.csv`, a mid-run
`mid.chk`, an end-of-chunk `end.chk` and `done.json` (wall time and ns/day). Per 100 ns chunk that
is 10 000 all-atom frames and 50 000 solute-only frames.

**The solute trajectory is written unwrapped.** OpenMM's `enforcePeriodicBox` wraps whole
*molecules*, not atoms, so the solute is never split across an image and torsions are safe under
either setting (verified: a solute deliberately straddling a box face returns a max bonded distance
of 0.153 nm both ways). What wrapping does is **teleport the whole solute across the box** whenever
its centre crosses a face — which breaks any analysis that reads the trajectory as continuous
(RMSD without re-imaging, diffusion, Cartesian TICA) and makes the structure jump around in a
viewer. `traj_solute.dcd` therefore uses `enforcePeriodicBox=False`; the molecule may drift far from
the origin, which nothing here cares about. `traj_all.dcd` stays wrapped, so the water box renders
as a box.

**Why the hot walker chunks at 1 ns.** It is a *seed source*, not a trajectory to be analysed on its
own: the originating project drew per-generation seeds from ~1 ns slices of the hot chain,
so the chunk boundary and the generation boundary should coincide. 200 chunk directories is the
point, not an accident.

**Resume semantics.** Re-running any production step picks up at the first chunk without a
`done.json` and continues from the previous chunk's `end.chk`. It does **not** re-minimise — the
project has been bitten by that before: `minimizeEnergy(maxIterations=0)` on a continuing walker
quenches the structure it had reached. A completed run reports `already-complete` and does nothing.

**Interval alignment is asserted.** `chunk_ns` must be a whole multiple of both reporter intervals
and (for REMD) of the exchange interval, and every interval must be a whole number of timesteps.
Otherwise frames drift relative to chunk boundaries and the first frame of each chunk is not at a
round time. A rounded reporter interval silently changes the sampling frequency, so it raises.

**ONE process per GPU.** All six replicas share one process and one CUDA context by default. Two
processes on one A5000 without CUDA MPS run ~3.7× slower each — running two jobs on one card is
slower than running them sequentially. Scale across devices with
`production.device_index`, one process per card.

### The REMD reference is checkable without more REMD

`remd/exchange_attempts.csv` is written in the schema `analysis/remd_reliability.py` reads (it
carries `replica_i`/`replica_j` as `md_run.run_remd` writes them, `i`/`j` aliases, and
`walker_at_replica_XX`), so the four probes in `docs/protocols/remd_reference_reliability.md` run on
it directly: per-neighbouring-pair exchange acceptance (the ladder is a chain — the weakest link,
not the mean), **round trips** replayed from the accepted swaps (the decisive one), time-halves
drift in kT, and the per-replica spread. Run them before using a new reference for anything.

The exchange RNG is re-seeded on resume. That is unbiased for Metropolis accept/reject, so a
continuation is faithful *distributionally* rather than bit-exact, and `resumed` runs are labelled
as such in the manifest.

---

## 6. Configuration and the REST2 ladder

Four config files, one per stage, each a **slice of the same schema**
(`md_templates.openmm.DEFAULTS`, dumped in full as `scripts/config_defaults.json`).
One schema means nothing can drift between stages; a config carries only what differs from the
baseline, and an unknown key **raises** rather than leaving a baseline value silently in force while
the run's own `config.json` claims otherwise.

```bash
python scripts/generate_config.py --all --system macrocycle     # all four, 8 rungs
python scripts/generate_config.py --all --system cyclo_rgdfv    # all four, 10 rungs (Sage 2.2 enforced)
python scripts/generate_config.py --all --system alanine        # all four, 6 rungs
python scripts/generate_config.py --md --walker hot             # just md-config.json
python scripts/generate_config.py --rest2 --s_cold 1 --s_hot 0.25 --N_rungs 4 --interp sqrt
```

`--all` also writes both `md-cold-config.json` and `md-hot-config.json`, since a campaign needs
each. `--cbar` / `--prest2` refuse and write nothing: stage (e) does not exist yet, and a config for
a script that cannot read it would only rot.

### The ladder

`rest2_ladder(s_cold, s_hot, n_rungs, interp)` builds it; `production.remd.scale_factors` overrides
it outright. `interp` fixes what is spaced evenly:

| `interp` | spacing | note |
|---|---|---|
| **`sqrt`** (default) | linear in `√s` | exchange acceptance is governed by the overlap of two rungs' energy distributions, whose width scales roughly as `√s`, so even spacing here gives roughly even acceptance along the chain |
| `linear` | linear in `s` | bunches rungs at the hot end |
| `geometric` | linear in `ln s` | the usual *parallel tempering* choice; REST2 scales only the solute, so that argument does not transfer unchanged |

At `s_hot = 0.25` the `sqrt` ladder reproduces the one every existing reference in this project
used, which is the reason it is the default:

| `--system` | rungs | ladder | `T_eff` on the solute (K) |
|---|---:|---|---|
| `alanine` | **6** | `[1.0, 0.81, 0.64, 0.49, 0.36, 0.25]` | 300, 370, 469, 612, 833, 1200 |
| `macrocycle` | **8** | `[1.0, 0.8622, ..., 0.3265, 0.25]` | 300, 348, 408, 486, 588, 726, 919, 1200 |
| `cyclo_rgdfv` | **10** | `[1.0, 0.8920, ..., 0.3086, 0.25]` | 300, 336, 380, 432, 496, 575, 675, 803, 972, 1200 |

The generic 8-rung `macrocycle` ladder is an **unvalidated starting ladder**. The 10-rung ladder is
supported by matched pilots **for cyclo-RGDfV specifically** and does not transfer to other
macrocycles.

Fewer rungs suffice for alanine because REST2's solute–solvent cross term grows with solute size: a
dipeptide's neighbouring rungs overlap far more readily than a macrocycle's.

**These are starting points, not verified ladders.** The ladder is a chain and is only as good as
its weakest neighbouring pair, so run a short pilot and read the *worst-pair* acceptance out of
`<suffix>_exchange_attempts.csv` before committing a long run. `generate_config.py` prints this
warning whenever it writes a REST2 config.

### Seeds and platform

`run.seed` is the master seed; any unset per-stage seed derives from it deterministically and
distinctly. Every integrator is seeded **before its `Context` is created** — `setRandomNumberSeed`
afterwards is silently ignored, which is a project-wide engine invariant and the reason
`make_integrator` returns an integrator rather than a Simulation. `production.precision` is set to
`mixed` **explicitly** on CUDA/OpenCL; the platform default is single, ~25 % faster, and silently
changes energies.

### Output layout

```
<out-dir>/
├── simbox_system.xml  simbox_topology.pdb  simbox_simbox.json   stage (a)
├── simbox_prep/                                                 (a) intermediates
├── eq_state.xml  eq_equilibrated.pdb  eq_minimized.pdb          stage (b)
│   eq_equilibration.csv  eq_min-eq.json
├── cold/chunk_0000../   cold_md.json                            stage (c)
├── hot/chunk_0000../    hot_md.json                             stage (c)
└── rest2/replica_00../chunk_0000../                             stage (d)
    rest2_exchange_attempts.csv  rest2_rest2.json
```

Each stage also writes `manifest_<suffix>_<stage>.json` (git commit + dirty flag, command line,
OpenMM version, host, timestamp) and a copy of its fully resolved config.

## 7. What has been verified, and what has not

Verified end to end on solvated alanine dipeptide (1832 atoms, CPU platform, short times):

* the peptide route (`--pdb` → ff19SB) and the ligand route (`--smi` → ETKDGv3 + MMFF94s → Sage 2.2
  + AM1BCC) both build a System;
* dodecahedron geometry, `addHydrogens` at pH 7, 0.15 M NaCl;
* HMR conserves total mass to 1e-6 amu and touches no water;
* omega central-bond detection returns the two backbone amides;
* the full staged equilibration (9 stages, restrained → released), box averaging over 20 samples,
  and the State-based handoff into both the cold walker and REMD;
* the 1.0 nm cutoff fits the corrected box (it does not fit the raw-padding box).

**Not verified:**

* **4 fs against 2 fs** on any system — see §1(c); with the middle integrator this is a
  confirmation, not a gate;
* **ff19SB + TIP3P-FB** against ff19SB + OPC on any observable;
* **a macrocycle** through the ligand route — alanine's SMILES exercised it, but a cyclic
  sage/openff topology has bitten this project before (`mdtraj.compute_phi/psi` finds nothing on
  one), and CV extraction from these trajectories must go through
  `systems/cv_definition.load_cv_definition`, never re-derived;
* **the `"complex"` route** (ff19SB residues + SMIRNOFF ligand in one system);
* **realised salt molarity is quantised at small box sizes** — alanine's 602-water box takes 2 ion
  pairs and lands at 0.184 M against the 0.15 M requested. `solvation.json` records what was
  actually built; a larger solute will land closer;
* **the staged protocol on an actual macrocycle** — it was exercised on alanine dipeptide, whose
  solute barely moves (RMSD 0.051 nm); a 200-atom ring from a gas-phase ETKDG conformer is the case
  the protocol exists for and is where the per-stage RMSD trace should first be read carefully;
* **timings on GPU**, and therefore the wall-clock cost of 1 µs. `done.json` records ns/day per
  chunk, so the first chunk of the first real run answers this.

---

## 8. Relationship to the existing implicit-solvent path

This baseline is deliberately *not* built on `systems/topology_prep.build_topology`, whose explicit
route goes through tleap and supports OPC water in an isometric box only — neither TIP3P-FB nor a
dodecahedron, and it cannot mix ff19SB residues with a SMIRNOFF ligand. The OpenMM
`ForceField`/`Modeller` route requested here does all three.

What *is* reused, unchanged, is everything that carries scientific meaning:
`openmm_system.build_rest2_scaled_system` (the REST2 Hamiltonian, including the CMAP scaling ff19SB
needs and the `√s` solute–solvent cross terms), `topology_prep.resolve_remd_scale_ladder` (ladder
validation and the `replica 0 == s = 1` convention), and `md_run.exchange_pairs` /
`md_run.attempt_rest2_exchange` (the neighbour schedule and the Metropolis criterion, already
box-vector aware). Acceptance and round-trip statistics are therefore comparable between the
implicit and explicit references.

Conditional BAR is out of scope here by request. When it comes, the pieces it will need from this
baseline are the hot walker's 1 ns chunks (seed source) and a flat-bottom wall built on the
solvated topology — `rectangular_flatbottom` acts on solute CVs only and is solvent-agnostic, but
the **no-restraint-in-the-physical-work** invariant and the interior-frames-only rule for reverse
seeds carry over untouched.
