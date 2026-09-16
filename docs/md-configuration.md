# Configuration reference

**Generated from the schemas by `md_tools.build.manual.render()`. Do not hand-edit:**
a test asserts this file and the schemas agree, so an edit here is a failure rather than
a change. The authoritative text for every key is the `Field(doc=...)` string in
`md_tools.build.top` (for `build-top`) and `md_tools.build.md` (for `build-md`).

Every configuration file is YAML despite the `.config` suffix. Unknown keys are refused
by name, so a misspelled key is an error and never silently becomes metadata. Every key
below has a default unless it says **required**, and the defaults here are the ones the
resolver actually applies -- the shipped examples in `configs/` are held to them by test.

Lengths are **integer step counts**, everywhere. A duration in picoseconds would have to
divide by a timestep the file may not have been written against; logs derive ps and ns
for the reader instead.

## Which keys apply to which protocol

A key that a protocol does not read is refused rather than ignored, so this table is a constraint and not a convenience.

| protocol | sections it reads |
|---|---|
| `cMD` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables` |
| `REST2` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `umbrella`, `rest2`, `reservoir` |
| `rREST2` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `umbrella`, `rest2`, `reservoir` |
| `AIS` | `top-level`, `dynamics`, `reporting`, `collective_variables`, `ais`, `ais_source` |
| `umbrella` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `umbrella` |

## `build-top`: topology and System construction

Topology and System construction for `md-openmm build-top`.

Resolved by `md-openmm build-top`. Unknown keys are refused by name rather than ignored, so a typo is an error and never silent metadata.

### `solute`

What the input is, and how it is parameterised.

#### `solute.kind`

type: string · default: `peptide` · one of `peptide`, `peptide-like`, `ligand`

What the solute IS, which decides how it is parameterised and what chemistry may be read from it. This is the authoritative classification. peptide       -- read -i as a peptide/protein PDB and parameterise it with the protein force field. Sage never touches it. ligand        -- read -i as a .smi or .sdf and parameterise the whole molecule with the small-molecule force field. One residue, no peptide chemistry is claimed or read. A .smi states the chemistry and the conformer is generated (ETKDGv3, then MMFF); a .sdf carries the coordinates too and they are used as given. peptide-like  -- the SAME whole-molecule route as `ligand`, with the same force field and the same charges, PLUS a validated peptide-chemistry map over the result. It exists for a head-to-tail cyclic peptide built from SMILES, whose residues are real amino acids but which a single-residue ligand representation cannot describe -- so residue-keyed corrections such as mbondi3's silently miss it. It never loads a protein force field and never replaces Sage's charges or bonded terms.

#### `solute.peptide`

type: boolean or null · default: `null`

RETIRED spelling of `kind`, kept so configurations written before `kind` existed -- including every `resolved.config` already on disk -- still read. true means kind: peptide, false means kind: ligand, and there is no boolean for peptide-like. Resolved into `kind` BEFORE defaults are applied, so a stated boolean is never compared against a `kind` nobody wrote. Stating both is accepted only when they agree exactly; anything else is refused with a migration message rather than silently preferring one.

#### `solute.ligand_forcefield`

type: string · default: `sage-2.2.1`

Small-molecule force field, used only when peptide is false. Two families are supported. `sage-2.2.1` (the default) is OpenFF Sage, applied through SMIRNOFF. `gaff2` selects the newest installed GAFF 2.x and is resolved to its exact version -- `gaff-2.2.20` here -- because GAFF2 has been distributed as several different parameter sets and the ambiguous label does not identify a Hamiltonian. An exact version such as `gaff-2.11` may be written instead; one that is not installed is refused, naming what is.

#### `solute.ligand_charge_method`

type: string · default: `am1bcc` · one of `am1bcc`, `am1bccelf10`, `gasteiger`, `nagl`

Partial-charge method for the small molecule. am1bcc is the validated default and runs on CPU; it is the slowest part of a ligand build.

#### `solute.residue_name`

type: string or null · default: `null`

Three-character residue name for a molecule read from .smi or .sdf. Left null, a deterministic name is assigned from the file and recorded, so the same input always produces the same residue identity.

### `forcefield`

Force-field selection. The resolved resource names are recorded, not these labels.

#### `forcefield.protein`

type: string · default: `ff14SB` · one of `ff14SB`, `ff19SB`

Protein force field. ff19SB is intended to be paired with OPC water; the pairing is checked.

### `solvent`

Solvent treatment. Under GBn2 every key here except `model` is inapplicable.

#### `solvent.model`

type: string · default: `TIP3P` · one of `TIP3P`, `OPC`, `GBn2`

TIP3P or OPC give an explicit, periodic, solvated system. GBn2 is IMPLICIT solvent: no water, no box, no ions, no barostat and no NPT stage anywhere downstream. Choosing GBn2 changes what the rest of this file may say.

#### `solvent.padding_nm`

type: number · default: `1.5` · minimum 0.5; maximum 5.0; unit: nm

Minimum distance from the solute to the box boundary. The built box may be grown beyond this if the nonbonded cutoff requires it; both the requested and achieved clearances are recorded.

#### `solvent.box_shape`

type: string · default: `dodecahedron` · one of `dodecahedron`, `cube`, `octahedron`

A rhombic dodecahedron holds ~71% of the water a cube needs for the same clearance, so it is the default. Explicit solvent only.

#### `solvent.ionic_strength_molar`

type: number · default: `0.15` · minimum 0.0; maximum 2.0; unit: mol/L

Salt added AFTER neutralising the solute charge, so the final ionic strength is this value and the box is neutral. 0.15 M is physiological.

#### `solvent.positive_ion`

type: string · default: `Na+` · one of `Na+`, `K+`, `Li+`, `Cs+`, `Rb+`

Cation used both to neutralise and to reach the ionic strength.

#### `solvent.negative_ion`

type: string · default: `Cl-` · one of `Cl-`, `Br-`, `F-`, `I-`

Anion used both to neutralise and to reach the ionic strength.

#### `solvent.cutoff_nm`

type: number · default: `1.0` · minimum 0.6; maximum 2.0; unit: nm

Nonbonded real-space cutoff. The box must be at least twice this in its smallest reduced height; if it is not, the box is grown and that is logged.

### `constraints`

Constraints. These change the serialised System.

#### `constraints.type`

type: string · default: `HBonds` · one of `HBonds`, `AllBonds`, `None`

HBonds constrains the LENGTH of every bond to a hydrogen, and permits the 2 fs default timestep. It constrains NO ANGLE -- not even H-X-H. That is the point: with the X-H stretches frozen the fastest remaining motions are the hydrogen bond-angle vibrations at ~10 fs, which 2 fs resolves. OpenMM's HAngles would constrain those too and is deliberately not offered. AllBonds additionally freezes heavy-atom bond lengths. HOW the constraints are solved is OpenMM's choice, not a setting here and not selectable: CCMA for the general case, which is every X-H constraint in an ordinary solute, and SETTLE for rigid three-site water. There is no SHAKE in OpenMM. See docs/scientific-defaults.md section 11.2.

#### `constraints.rigid_water`

type: boolean · default: `true`

Hold water rigid -- the case OpenMM solves with SETTLE, which needs a real rigid triangle (both O-H bonds and the H-H distance). Forced to false under implicit solvent, where there is no water to hold rigid, and recorded as the resolved value rather than the requested one; such a run therefore uses CCMA and nothing else.

### `hydrogen_mass_repartitioning`

Hydrogen mass repartitioning, stated explicitly rather than implied by a null.

#### `hydrogen_mass_repartitioning.enabled`

type: boolean · default: `false`

Whether to repartition hydrogen masses. FALSE by default: the serialised masses are then the force field's own. Repartitioning rewrites particle masses in built.xml, which is a property of the System -- it cannot be inferred later from a configuration that merely asks for a 4 fs timestep, which is exactly why it is stated here and verified from the System at run time.

#### `hydrogen_mass_repartitioning.hydrogen_mass_amu`

type: number · default: `3.024` · minimum 1.0; maximum 6.0; unit: amu

Target hydrogen mass when enabled. Mass is moved FROM the bonded heavy atom, so the total is conserved; water is never repartitioned, because rigid water's hydrogen masses do not limit the timestep. Ignored, and recorded as ignored, when enabled is false.

## `build-md`: protocol, stages and reporting

Protocol, stage lengths in steps, and reporting intervals for `md-openmm build-md`.

Resolved by `md-openmm build-md`. Unknown keys are refused by name rather than ignored, so a typo is an error and never silent metadata.

### Top-level keys

#### `protocol`

type: string · default: `cMD` · one of `cMD`, `REST2`, `rREST2`, `AIS`, `umbrella`

cMD is plain molecular dynamics. REST2 adds a replica-exchange ladder in which only the solute's Hamiltonian is scaled. rREST2 adds a Boltzmann reservoir refresh of the hottest rung. AIS runs non-equilibrium switching paths from an EXISTING equilibrium source ensemble -- it has no minimisation or equilibration chain of its own, because its input is a trajectory you have already produced.

#### `solvent`

type: string · default: `explicit` · one of `explicit`, `implicit`

Must match the System `build-top` produced. Under implicit solvent there is no box and no barostat, so the pressure-coupled equilibration stages are replaced by honest NVT equivalents with different file names -- they are NOT NPT stages with the pressure quietly ignored.

### `dynamics`

Physical constants and integrator settings. These carry real units.

#### `dynamics.timestep_fs`

type: number or string · default: `auto` · unit: fs

`auto` (the default) resolves the timestep from the masses serialised in built.xml when the run starts: 2.0 fs on ordinary hydrogens, 4.0 fs on repartitioned ones. `build-md` never opens built.xml, so it cannot know whether HMR was applied -- and a configuration CLAIMING it is not evidence. A number may be given instead: it is honoured up to 3.0 fs, and above that only if the System really was repartitioned. The resolved value and its basis are recorded in every log.

#### `dynamics.temperature_K`

type: number · default: `300.0` · minimum 1.0; maximum 1000.0; unit: K

Thermostat temperature. Under REST2 every replica runs at this same physical temperature; the ladder scales the Hamiltonian, not the bath.

#### `dynamics.pressure_bar`

type: number · default: `1.0` · minimum 0.0; maximum 1000.0; unit: bar

Barostat pressure. Ignored -- and refused if a stage claims NPT -- under implicit solvent, which has no volume.

#### `dynamics.friction_per_ps`

type: number · default: `1.0` · minimum 0.01; maximum 100.0; unit: 1/ps

LangevinMiddleIntegrator collision rate.

#### `dynamics.barostat_interval_steps`

type: integer · default: `25` · minimum 1; unit: steps

MonteCarloBarostat volume-move attempt interval, in STEPS. It is the frequency, not the barostat's presence, that makes a stage NPT.

#### `dynamics.restraint_kcal_per_mol_A2`

type: number · default: `1.0` · minimum 0.0; maximum 1000.0; unit: kcal/mol/A^2

Positional restraint on solute heavy atoms during the restrained equilibration stages. A physical constant with real units: the statement that stage LENGTHS are step counts does not make force constants unitless.

#### `dynamics.tau`

type: number · default: `0.0` · minimum 0.0; maximum 0.95

Fixed REST2 scaling for a cMD run. 0.0 (the default) is the unmodified physical Hamiltonian. A non-zero value runs cMD at ONE rung of the REST2 ladder -- the same scaling the ladder applies, held fixed -- which is how a Boltzmann reservoir for rREST2 is generated, and how a hot ensemble is produced without running an exchange. A scaled run is NVT by construction: it must sample the top rung's fixed-volume ensemble, so a barostat would sample the wrong distribution and is refused.

#### `dynamics.phase_space_printout`

type: integer · default: `0` · minimum 0; unit: steps

Write a phase-space stream (positions, VELOCITIES and box) every N steps. 0 disables it. A reservoir needs complete samples including velocities, which a trajectory does not carry, so this is what a reservoir source is generated with.

#### `dynamics.seed`

type: integer · default: `1` · minimum 0

Base random seed. Each stage derives its own from this plus the stage name, so stages are independent and the whole chain is reproducible.

### `stages`

Stage lengths, as exact integer step counts.

#### `stages.minimization_iterations`

type: integer · default: `1000` · minimum 0

Minimiser ITERATIONS, not a time interval: minimisation does not integrate and has no timestep. 0 skips minimisation.

#### `stages.restrained_nvt_steps`

type: integer · default: `5000` · minimum 0; unit: steps

Restrained NVT heating/settling. 5000 steps = 10 ps at 2 fs.

#### `stages.restrained_npt_steps`

type: integer · default: `5000` · minimum 0; unit: steps

Restrained NPT density equilibration. 5000 steps = 10 ps at 2 fs. Under implicit solvent this becomes a second restrained NVT stage instead.

#### `stages.unrestrained_npt_steps`

type: integer · default: `5000` · minimum 0; unit: steps

Unrestrained NPT, the last stage before production. 5000 steps = 10 ps at 2 fs. Under implicit solvent this becomes unrestrained NVT.

#### `stages.production_steps`

type: integer · default: `2500000` · minimum 0; unit: steps

Production length. 2,500,000 steps = 5 ns at 2 fs. This is the authoritative number; the log prints the derived ps and ns beside it.

#### `stages.number_of_segments`

type: integer · default: `1` · minimum 1

How many SEGMENTS the production run is written as. 1, the default, is a single `prod1` segment and the historical behaviour. It SPLITS the production total rather than multiplying it: `production_steps` stays the whole run and each segment gets `production_steps / number_of_segments`, so raising this re-divides the same trajectory and never lengthens it. A value that does not divide exactly is refused with the arithmetic that would fix it, because a final short segment would make the last chunk incomparable with the others. On a REST2/rREST2 ladder there is no `production_steps`: production is `number_of_exchanges * exchange_interval_steps`, so the split is of `number_of_exchanges` and an exchange is never allowed to straddle two segments. Each segment writes its own files -- `solute_state<i>_prod<N>.nc`, `cv_state<i>_prod<N>.dat`, `restart_state<i>_prod<N>.json` -- so segments cannot overwrite one another. That was a real defect: every chunk of a five-chunk reference run wrote `_prod1`.

### `reporting`

Output intervals, in steps.

#### `reporting.crd_printout_solute`

type: integer · default: `1000` · minimum 0; unit: steps

Trajectory output interval. 1000 steps = 2 ps at 2 fs.

#### `reporting.crd_printout_whole`

type: integer · default: `0` · minimum 0; unit: steps

COORDINATE output interval for the WHOLE system, in steps, written to `whole_prod<N>.nc` (or `whole_rep<i>_prod<N>.nc` per replica on a ladder). 0, the default, writes no whole-system trajectory. Say what you want kept: on a solvated system the whole stream is two orders of magnitude larger than the solute one -- 1796 atoms against 22 for alanine dipeptide in water -- so an unasked-for whole-system trajectory at a solute cadence is how a 26 MB run becomes 2.1 GB. Distinct from `crd_printout_solute`, which is the solute alone, and from `info_printout`, which is scalars and no coordinates at all. Those three were previously two, and the two were named for what a reader assumed rather than for what they wrote.

#### `reporting.info_printout`

type: integer · default: `10000` · minimum 0; unit: steps

Scalar state (energy, temperature, volume, density) interval, written to a CSV beside the log.

#### `reporting.checkpoint_printout`

type: integer · default: `10000` · minimum 0; unit: steps

Checkpoint interval. A checkpoint is what an interrupted stage resumes from, and it is written with a fingerprint of the configuration that produced it so it cannot be resumed under different settings.

### `collective_variables`

Torsion collective-variable reporting. Observation only: no Force is added and the Hamiltonian is unchanged. Both keys disable it by default, and supplying only one of them is an error rather than a guess.

#### `collective_variables.file`

type: string or null · default: `null`

Path to a cv.yaml defining the torsions to report, resolved relative to THIS configuration file. Null disables collective-variable reporting.

#### `collective_variables.interval_steps`

type: integer · default: `0` · minimum 0; unit: steps

Steps between collective-variable observations. Independent of the trajectory and state-data intervals, and may be more frequent than either. 0 disables collective-variable reporting.

### `rest2`

The REST2 ladder. Ignored when protocol is cMD. The Hamiltonian scaling itself -- bonds and angles unscaled, ordinary amide omega unscaled, eligible solute torsions and CMAP by (1-tau)^2, solute-solute nonbonded and 1-4 by (1-tau)^2, solute-environment by (1-tau), GB by (1-tau) -- is a property of the validated implementation and is not configurable here.

#### `rest2.number_of_replicas`

type: integer · default: `4` · minimum 2; maximum 64

States in the ladder. Every state runs at the same physical temperature.

#### `rest2.tau_max`

type: number · default: `0.5` · minimum 0.0; maximum 0.95

The hottest rung's tau. The ladder is linear from 0.0 to this value. tau = 0 is the unscaled physical Hamiltonian.

#### `rest2.exchange_interval_steps`

type: integer · default: `5000` · minimum 1; unit: steps

Steps of dynamics between exchange attempts. 5000 steps = 10 ps at 2 fs.

#### `rest2.number_of_exchanges`

type: integer · default: `500` · minimum 1

Exchange attempts. Total production per state is this times exchange_interval_steps.

#### `rest2.equilibration_steps`

type: integer · default: `0` · minimum 0; unit: steps

Relaxation run at EACH STATE'S OWN Hamiltonian before the first exchange attempt, and not counted as production. 0 (the default) starts exchanging immediately. This exists because the alternative is wrong in a way that is hard to see. A ladder takes ONE starting state -- the group file refuses per-rung coordinates, deliberately, since rungs must be states of the same system -- so without this every rung begins from a configuration equilibrated under tau = 0. The hot rungs then spend their opening exchanges relaxing out of a distribution that is not theirs, and those samples are production by every record that describes them. The driver has always relaxed each rung under its own scaled Hamiltonian; only the number was unreachable from a configuration file.

#### `rest2.equilibration_per_tau`

type: boolean · default: `false`

Run the equilibration stages on EVERY RUNG, under that rung's own tau, instead of once at tau = 0. REST2 and rREST2 only; refused for any other protocol. Off by default. When true, the tau = 0 chain stops early: at minimisation under implicit solvent, and after its NPT stages under explicit solvent, which run once at tau = 0 to fix the box every rung then shares. Every rung -- tau = 0 included -- then runs eq_nvt_posres (restrained_nvt_steps), eq_nvt_posres_2 (restrained_npt_steps) and eq_nvt_free (unrestrained_npt_steps), all at fixed volume, from the ladder's starting state, each stage with its own seed per rung. The restraint is the stage chain's, on the same atoms at the same strength; the rung Systems the ladder propagates never carry it. Order: these stages, then `equilibration_steps`, then the first exchange. Neither is production. Stages of 0 steps are skipped, and all three at 0 is refused.

#### `rest2.state_trajectory`

type: boolean · default: `true`

Write one trajectory per fixed thermodynamic STATE (solute_state<i>_prod<N>.nc, and whole_state<i>_prod<N>.nc when a whole-system cadence is set). A state trajectory follows a state, not a walker; the filename carries the state index and never the tau value.

#### `rest2.rem_log`

type: boolean · default: `true`

Write an Amber-style rem.log projection of the exchange history.

#### `rest2.neighbour_acceptance_report`

type: boolean · default: `true`

Report acceptance for each neighbouring pair. A single averaged acceptance hides a ladder with one impassable gap.

### `ais`

The switching path. Ignored unless protocol is AIS. Every length is an exact integer step count; nothing here is a duration that has to divide by a timestep.

#### `ais.number_of_paths`

type: integer · default: `100` · minimum 1

How many independent switching paths to run. Each gets its own directory, its own trajectory, and its own deterministic seeds.

#### `ais.tau_start`

type: number · default: `0.5` · minimum 0.0; maximum 0.95

The tau the path starts at. It must equal the tau of the source ensemble: the path begins in the ensemble it anneals away from, and the source's own record is checked against this rather than assumed.

#### `ais.tau_end`

type: number · default: `0.0` · minimum 0.0; maximum 0.95

The tau the path ends at. 0.0 is the unmodified physical Hamiltonian. tau_start and tau_end must differ, or the Hamiltonian never changes and every work value would be zero.

#### `ais.switching_steps`

type: integer · default: `50000` · minimum 1; unit: steps

The length of the switching path, as an exact step count. 50000 steps is 100 ps at 2 fs. WORK IS PATH-LENGTH DEPENDENT: a faster switch does more dissipative work, so this is a scientific choice and not a performance knob. The log states the derived ps.

#### `ais.observation_interval_steps`

type: integer · default: `2500` · minimum 1; unit: steps

How often a path is observed: one coordinate frame and one work row. switching_steps must divide by this exactly, so the last observation lands at tau_end. 50000/2500 gives 20 intervals and therefore 21 observations, counting both endpoints. AN OBSERVATION IS NOT A STEP.

#### `ais.parameter_update_interval_steps`

type: integer · default: `1` · minimum 1; unit: steps

How often tau moves. 1 changes the Hamiltonian every step -- 50000 parameter changes over the path above. observation_interval_steps must divide by this, or observations would not sit on the update grid.

#### `ais.work_measurement`

type: string · default: `work` · one of `work`, `components`

How the work increment at each parameter update is obtained, and what else is recorded with it. This is a scientific choice AND the dominant cost of an AIS run: everything else here controls what is written, this controls what is computed. work        -- TWO energy evaluations per update, U(tau_k) and U(tau_k+1). The work integral, and nothing more. This is the default because it is what a free energy for the schedule you actually ran needs. components  -- a THREE-point basis probe per update, at amplitudes (0, 0.5, 1), from which the work follows analytically. It also gives the potential as a FUNCTION of tau, which is what reweighting onto a different schedule, a different endpoint, or a Hummer-Szabo estimator evaluated at an unvisited tau requires. A single total work cannot produce that function, and re-running at another tau is not reweighting. Component columns are ABSENT from a `work` path rather than zero, so a reader expecting them fails instead of treating a missing measurement as a measured nought.

#### `ais.verify_every_updates`

type: integer · default: `0` · minimum 0

In `components` mode, how often the fitted work is checked against a directly measured U(tau_k+1) - U(tau_k). Costs two extra evaluations whenever it fires. 0 (the default) verifies the FIRST update of every path and no other. That is not a token check: the three-group identity is a property of the SYSTEM, not of the step -- a force carrying tau-dependence outside the basis is outside it at any coordinate -- so one verified update per path establishes the model the whole path relies on, for two evaluations rather than two thousand. N > 0 re-verifies every N updates as well, for the case the first update cannot cover: a force whose tau-dependence only switches on at some geometry the path reaches later. Ignored in `work` mode, where the work IS the direct measurement and there is nothing to cross-check it against.

### `ais_source`

Where the starting configurations come from. Ignored unless protocol is AIS.

#### `ais_source.trajectory`

type: string or null · default: `null`

REQUIRED for AIS. The equilibrium trajectory the paths start from, typically a fixed-tau cMD run at tau = ais.tau_start. Resolved relative to the directory run.sh is invoked from.

#### `ais_source.topology`

type: string or null · default: `null`

Topology for reading that trajectory. Null uses the -p topology the run was given, which is the usual case. The resolved choice is recorded.

#### `ais_source.first_frame`

type: integer · default: `0` · minimum 0

First eligible frame, INCLUSIVE, as a 0-based index into the trajectory file. A frame index, never a time: the two are interchangeable only when the frame interval is known, and it is not always. Use this to discard equilibration.

#### `ais_source.last_frame`

type: integer or null · default: `null` · minimum 0

Last eligible frame, INCLUSIVE. Null means the final frame in the file.

#### `ais_source.frame_stride`

type: integer · default: `1` · minimum 1

Take every Nth frame of the window as eligible. Consecutive frames of an MD trajectory are correlated, so drawing paths from every frame draws several of them from what is effectively one configuration. A stride is the honest way to say how far apart samples have to be; it does not make them independent, it stops them being obviously dependent.

#### `ais_source.selection`

type: string · default: `uniform_random` · one of `uniform_random`, `evenly_spaced`

How starting frames are drawn from the eligible window. uniform_random draws with the run's seed; evenly_spaced takes them at a fixed stride.

#### `ais_source.allow_repeated_frames`

type: boolean · default: `false`

Whether two paths may start from the SAME frame. False by default: two paths from one configuration are not two independent realisations, and treating them as such understates the spread of the work distribution.

### `umbrella`

Umbrella sampling: restrain named collective variables and report them. Producing the biased series is what this protocol does. Turning a set of windows into a free-energy profile is ANALYSIS and is deliberately not here: WHAM and MBAR belong to the project asking the question, not to the engine generating the samples. A window needs `collective_variables.file` and `interval_steps` set too. The restraint resolves its `cv` name against that same file, so the quantity that is biased and the quantity that is reported are the same object by construction -- a run cannot restrain one torsion and report another.

#### `umbrella.file`

type: string or null · default: `null`

Path to the restraint definition, resolved beside `resolved.config` -- the same rule `collective_variables.file` follows. A LIST of restraints does not fit a namelist `.in`, and inventing a packed-string encoding for one would make the most consequential line of an umbrella input the least readable. So the restraints live in their own YAML, referenced by path, exactly as the collective variables they name already do. Each entry names a CV from `collective_variables.file` and says how it is restrained -- see `md_tools.umbrella.load_umbrella_definition` for the schema and every way it is refused.

### `reservoir`

rREST2 reservoir. Ignored unless protocol is rREST2.

#### `reservoir.enabled`

type: boolean · default: `false`

rREST2 only. Refresh the hottest rung from a pre-generated Boltzmann reservoir instead of propagating it.

#### `reservoir.path`

type: string or null · default: `null`

Reservoir directory. Required when enabled.

#### `reservoir.refresh_interval_exchanges`

type: integer · default: `1` · minimum 1

How often the hottest rung is refreshed from the reservoir, in exchange attempts.

#### `reservoir.velocities`

type: string · default: `resample` · one of `resample`, `inherit`

Where a refreshed configuration's velocities come from. `resample` draws them from the Maxwell-Boltzmann distribution at the run temperature; `inherit` keeps the reservoir's own. Recorded explicitly because it is a provenance question, not a tuning knob.

## What a run writes

Nothing in this section is a setting. It is what the files you get MEAN, because a reader should
not have to run something to find out.

### The `.out` census

Every stage's `.out` opens with what it actually ran, read off the **serialised System** rather
than off the configuration that asked for it. That distinction is the point: a configuration
asking for a 1.0 nm cutoff and a System carrying 0.8 nm are different runs, and only one of them
integrates.

| section | what it reports |
|---|---|
| `System` | atoms, residues by name, net charge, degrees of freedom, box and volume |
| `Method` | nonbonded treatment and cutoff, Ewald tolerance, dispersion correction, switching, 1-4 exception count, constraints, barostat present, force inventory |
| `Selections` | the solute atom count, and which omega bonds are left unscaled |

The same facts enter the `.log` as structured fields, because prose is not a database. A box is
reported only when the System is genuinely **periodic**: an OpenMM System defaults to a 2 nm cube,
so an implicit run would otherwise print an invented box and an 8 nm³ volume in the same voice as
the measurement above it.

### The per-term energy decomposition

`energy_components_<stage>.csv` carries the potential energy term by term beside `mdout*.csv` --
except the production stage itself, which is `energy_components.csv` on its first segment and
`energy_components_prod<N>.csv` on later ones. `md_tools.md._stages.energy_components_name` is the
one authority, and only `cMD` and `umbrella` count as production stage names.

Where the System has usable force groups -- the implicit route, through ParmEd -- they are read
directly. Where it does not, which is every explicit-solvent System built through OpenMM's
`ForceField.createSystem`, a group-separated **copy** is probed instead: `built.xml`, the run's own
Context and every digest taken from them stay untouched, because a force group is part of the
serialised System and regrouping the integrated one would change `system_sha256` and invalidate
every checkpoint fingerprint in flight.

`EELEC` from `VDWAALS`, and the 1-4 terms from either, are **not** reachable that way -- every 1-4
pair is an exception inside the single `NonbondedForce`, which evaluates charge and dispersion in
one kernel -- so that split lives in post-hoc `md_tools.openmm.decomposition`.

### Fluctuations over a single sample

An rms fluctuation over one report reads `n/a (single sample)` rather than `0`. Over one sample
`sqrt(<x^2> - <x>^2)` is exactly zero, which reads as "this did not move" when it means "there was
nothing to compare it against" -- and a stage shorter than `info_printout` produces exactly one
row, so this is the ordinary case rather than the corner one.

### A ladder additionally writes

The Hamiltonian it integrated, so the ladder is checkable from its own output rather than from the
configuration that requested it: the tau ladder, the scaling laws applied and the terms left
unscaled, the solute region and its excluded omega bonds, `system_sha256`, that velocities are
never rescaled on a swap (one beta across the ladder), and a TIMINGS block with elapsed time,
per-replica and aggregate throughput, and cost per step.

Per-state and per-segment filenames are documented in `docs/run-layout.md`.
