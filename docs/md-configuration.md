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
| `REST2` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `umbrella`, `rest2` |
| `AIS` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `ais`, `ais_source` |
| `umbrella` | `top-level`, `dynamics`, `stages`, `reporting`, `collective_variables`, `umbrella` |

## `build-top`: topology and System construction

Topology and System construction for `md-openmm build-top`.

Resolved by `md-openmm build-top`. Unknown keys are refused by name rather than ignored, so a typo is an error and never silent metadata.

### Top-level keys

#### `ligands`

type: list · default: `[]`

The ligand instances of a `kind: complex` build, each mapped explicitly onto a reusable parameter package. One entry per instance: - select: {chain: B, resid: "201", insertion_code: ""} parameters: CHEMBL112/param_e932f4c4f371 `chain` is the chain id the input file carries (the AUTHOR chain for mmCIF); `resid` is a quoted string. Each selector must name exactly one residue. The residue's heavy atoms are matched to the package's chemical graph, its hydrogens come from the package, and its deposited pose is kept. A match that is ambiguous in a way that changes the chemistry (the two oxygens of a carboxylic acid) is refused unless the entry adds `atom_map: {deposited atom name: package atom name}` for every heavy atom. Repeated copies name the same package. Every residue that is not a standard protein residue, water or ion must be listed; nothing is guessed. Empty, and refused if set, for every other kind.

### `solute`

What the input is, and how it is parameterised.

#### `solute.kind`

type: string · default: `peptide` · one of `peptide`, `peptide-like`, `ligand`, `complex`

What the solute IS, which decides how it is parameterised and what chemistry may be read from it. This is the authoritative classification. peptide       -- read -i as a peptide/protein .pdb, or as a .seq holding one line of residue names that tleap's `sequence` builds (extended), and parameterise it with the protein force field. Sage never touches it. ligand        -- read -i as a .smi or .sdf and parameterise the whole molecule with the small-molecule force field. One residue, no peptide chemistry is claimed or read. A .smi states the chemistry and the conformer is generated (ETKDGv3, then MMFF); a .sdf carries the coordinates too and they are used as given. peptide-like  -- the SAME whole-molecule route as `ligand`, with the same force field and the same charges, PLUS a validated peptide-chemistry map over the result. It exists for a head-to-tail cyclic peptide built from SMILES, whose residues are real amino acids but which a single-residue ligand representation cannot describe -- so residue-keyed corrections such as mbondi3's silently miss it. It never loads a protein force field and never replaces Sage's charges or bonded terms. complex       -- read -i as a .pdb or .cif holding protein chains and ligand instances. The protein takes the protein force field; each ligand instance listed under `ligands` takes the parameters of an existing package, loaded as saved -- no charge is generated. Explicit solvent only.

#### `solute.peptide`

type: boolean or null · default: `null`

RETIRED spelling of `kind`, kept so configurations written before `kind` existed -- including every `resolved.config` already on disk -- still read. true means kind: peptide, false means kind: ligand, and there is no boolean for peptide-like. Resolved into `kind` BEFORE defaults are applied, so a stated boolean is never compared against a `kind` nobody wrote. Stating both is accepted only when they agree exactly; anything else is refused with a migration message rather than silently preferring one.

#### `solute.ligand_forcefield`

type: string · default: `sage-2.2.1`

Small-molecule force field, used only when peptide is false. Two families are supported. `sage-2.2.1` (the default) is OpenFF Sage, applied through SMIRNOFF. `gaff2` selects the newest installed GAFF 2.x and is resolved to its exact version -- `gaff-2.2.20` here -- because GAFF2 has been distributed as several different parameter sets and the ambiguous label does not identify a Hamiltonian. An exact version such as `gaff-2.11` may be written instead; one that is not installed is refused, naming what is.

#### `solute.ligand_charge_method`

type: string · default: `am1bcc` · one of `am1bcc`, `am1bccelf10`, `gasteiger`, `nagl`

Partial-charge method for the small molecule. am1bcc is the validated default and runs on CPU; it is the slowest part of a ligand build.

#### `solute.compound_id`

type: string or null · default: `null`

Catalog identity of the molecule for kind: ligand or peptide-like: a ChEMBL id (`CHEMBL112`) or `LOCAL-<first block of the standard InChIKey>`. The build writes the parameter package it creates under this compound. Left null, the LOCAL form is derived from the molecule and recorded.

#### `solute.aliases`

type: list · default: `[]`

Searchable names stored with a package this build creates: `[paracetamol, acetaminophen, TYL]`. Names, not identities.

#### `solute.parameters`

type: string or null · default: `search`

Where this molecule's parameters come from. Four kinds of value: search (the default) -- look in the catalog for a package whose declared criteria match this build: the molecule's topology, its protonation state, and the charge method INCLUDING the implementation that would run here (AM1-BCC through AmberTools' sqm is not AM1-BCC through OpenEye, and neither is NAGL's graph model of it), together with the small-molecule force field. Reuse on a match, parameterise on any difference; a near match is a difference. Which package matched, on what, and what else was considered, are recorded in built.log. <compound id>/param_<12 hex> -- reuse exactly that package from the catalogs, and search nothing. a PATH to a package directory -- reuse the package there, and search nothing. Absolute, or relative to this configuration file, so a build works from `build-top --parameterize` output beside it with no catalog configured at all: `parameters: ./parameter`. The directory is named whatever its build named it; `<compound>/param_<id>/` is what the CATALOG requires, and registration is what gives it that shape. generate -- parameterise the molecule whatever the catalog holds. On either kind of reuse the prepared molecule must be the package's exact chemical state (every hydrogen, charge and bond order, and the stereochemistry of its coordinates); its atoms are put into package order and no charge is generated. A reused package brings its own force field and charges, so ligand_forcefield and ligand_charge_method may not be stated beside a reference. Catalogs are searched in order: ligand_catalog.path, then $MD_DATA/parameters/ligands.

#### `solute.residue_name`

type: string or null · default: `null`

Three-character residue name for a molecule read from .smi or .sdf. It is APPLIED: the molecule's residue in built.pdb, built.solute.pdb and the topology carries it, and the prepared molecule is written beside the System as `<residue_name>.sdf`. Left null, a deterministic name is assigned from the file (the .smi name field, else the file stem) and recorded, so the same input always produces the same residue identity. Three letters or digits; a name that already means water, an ion or a protein residue is refused, because solvent selection and the omega classifier read residue names. Refused for kind: peptide, whose residues are named by the input.

### `input`

How the structure file is read.

#### `input.assembly`

type: string or null · default: `null`

Build this BIOLOGICAL ASSEMBLY of an mmCIF input (the `_pdbx_struct_assembly` id, quoted: "3"), not its asymmetric unit. They are different molecules: 1TYL's asymmetric unit is an insulin dimer, its assembly 3 the T3R3 hexamer. Every copy of a chain gets its own chain id (A, B, C ... in operator order), and build/assembly.json maps each back to its author chain, label_asym ids and operator. An ion or water every operator places on the same symmetry-axis position is kept once, and each dropped copy is recorded; coinciding protein or ligand atoms are refused. `ligands` selectors and `protonation.overrides` name the EXPANDED chain ids. Null builds the file as deposited. Only for a .cif input and kind: peptide or complex.

#### `input.missing_atoms`

type: string · default: `refuse` · one of `refuse`, `add`

What to do when a standard residue lacks heavy atoms -- a disordered surface side chain, a missing terminal OXT. `refuse` (the default) stops the build and lists every such residue and the atoms it lacks. `add` builds them with PDBFixer and records every added atom in built.log; they carry no crystallographic evidence. Missing RESIDUES inside a chain (a C-N break above 2 A) are always refused: that is loop modelling. Only for kind: peptide or complex built from a .pdb or .cif.

### `protonation`

Protonation of titratable protein residues.

#### `protonation.method`

type: string · default: `openmm` · one of `openmm`, `propka`

How titratable protein residues get their protonation states. openmm -- Modeller.addHydrogens(pH) chooses, as md-tools always did. propka -- PROPKA3 predicts pKa values on the prepared structure (ligands and ions kept), and md-tools assigns variants by one stated rule: ASP/GLU protonated (ASH/GLH) when pKa > pH, LYS neutral (LYN) when pKa < pH, HIS doubly protonated (HIP) when pKa > pH and otherwise NEUTRAL, with OpenMM's hydrogen-bond heuristic choosing HID or HIE -- PROPKA does not resolve that tautomer -- and CYX for disulfides. A predicted state no supported variant builds (deprotonated CYS, tyrosinate, neutral ARG, a changed terminus) is REPORTED and the standard state kept. PROPKA missing or failing is an error, never a fall back. Predictions within `near_ph_window` of the pH are flagged. Everything is recorded in build/protonation.json. Either way the states are held fixed for the run: this is not constant-pH MD.

#### `protonation.ph`

type: number · default: `7.0` · minimum 0.0; maximum 14.0

Target pH.

#### `protonation.overrides`

type: list · default: `[]`

Explicit per-residue variants, which win over any prediction and are reported when they do: - select: {chain: A, resid: "102", insertion_code: ""} variant: HIE One of ASP ASH GLU GLH HID HIE HIP LYS LYN CYS CYX, of the residue's own family. An unknown selector or another family's variant is refused.

#### `protonation.histidine_proximity_angstrom`

type: number · default: `5.0` · minimum 0.0; maximum 20.0; unit: A

A histidine with a heavy atom within this distance of a ligand or ion heavy atom is printed as a WARNING: chain/resid/icode, the neighbour, the distance, the predicted pKa, the final variant and where it came from, and for an ion the ND1-ion and NE2-ion distances separately. Screening only: proximity is not coordination, and no tautomer is imposed; set an override if it matters.

#### `protonation.near_ph_window`

type: number · default: `1.0` · minimum 0.0; maximum 7.0

A predicted pKa within this many units of the pH is flagged as near-pH, so its assigned state reads as uncertain.

### `ligand_catalog`

Where reusable ligand parameter packages are looked up.

#### `ligand_catalog.path`

type: string or null · default: `null`

A directory laid out as `<compound id>/<parameter id>/`, searched FIRST for `solute.parameters` and `ligands[].parameters` -- a registered catalog, or the `ligands/` directory of an earlier build. Relative to the configuration file. The shared catalog $MD_DATA/parameters/ligands is searched after it when $MD_DATA is known. A build only reads a catalog; registration writes to it.

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

type: string · default: `cMD` · one of `cMD`, `REST2`, `AIS`, `umbrella`

cMD is plain molecular dynamics. REST2 adds a replica-exchange ladder in which only the solute's Hamiltonian is scaled. AIS runs non-equilibrium switching paths from an EXISTING equilibrium source ensemble -- it has no minimisation or equilibration chain of its own, because its input is a trajectory you have already produced.

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

Fixed REST2 scaling for a cMD run. 0.0 (the default) is the unmodified physical Hamiltonian. A non-zero value runs cMD at ONE rung of the REST2 ladder -- the same scaling the ladder applies, held fixed -- which is how a hot ensemble is produced without running an exchange. A scaled run is NVT by construction: it must sample the top rung's fixed-volume ensemble, so a barostat would sample the wrong distribution and is refused.

#### `dynamics.phase_space_printout`

type: integer · default: `0` · minimum 0; unit: steps

Write a phase-space stream (positions, VELOCITIES and box) every N steps. 0 disables it. A complete sample needs velocities, which a trajectory does not carry.

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

How many SEGMENTS the production run is written as. 1, the default, is a single `prod1` segment and the historical behaviour. It SPLITS the production total rather than multiplying it: `production_steps` stays the whole run and each segment gets `production_steps / number_of_segments`, so raising this re-divides the same trajectory and never lengthens it. A value that does not divide exactly is refused with the arithmetic that would fix it, because a final short segment would make the last chunk incomparable with the others. On a REST2 ladder there is no `production_steps`: production is `number_of_exchanges * exchange_interval_steps`, so the split is of `number_of_exchanges` and an exchange is never allowed to straddle two segments. Each segment writes its own files -- `solute_state<i>_prod<N>.nc`, `cv_state<i>_prod<N>.dat`, `restart_state<i>_prod<N>.json` -- so segments cannot overwrite one another. That was a real defect: every chunk of a five-chunk reference run wrote `_prod1`.

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

Path to a cv.yaml defining the torsions to report, resolved relative to THIS configuration file. Null disables collective-variable reporting unless `generate` is set.

#### `collective_variables.generate`

type: string or null · default: `null` · one of `all_solute_torsions`

Have `build-md` WRITE the cv.yaml instead of naming one: `all_solute_torsions` lists every proper torsion of the solute (every chain of four bonded solute atoms, hydrogens included), by explicit atom selector, read from build/built.pdb and the bonds of build/built.xml. The generated file is copied in exactly as a named one would be, and `resolved.config` then names that copy in `file` -- so the run measures what an ordinary, readable cv.yaml says. Give `file` or `generate`, not both.

#### `collective_variables.interval_steps`

type: integer · default: `0` · minimum 0; unit: steps

Steps between collective-variable observations. Independent of the trajectory and state-data intervals, and may be more frequent than either. 0 disables collective-variable reporting.

### `rest2`

The REST2 ladder. Ignored when protocol is cMD. The Hamiltonian scaling itself -- bonds and angles unscaled, amide omega, aromatic ring, double bond and improper torsions unscaled, eligible solute torsions and CMAP by (1-tau)^2, solute-solute nonbonded and 1-4 by (1-tau)^2, solute-environment by (1-tau), GB by (1-tau) -- is a property of the validated implementation and is not configurable here.

#### `rest2.number_of_replicas`

type: integer · default: `4` · minimum 2; maximum 64

States in the ladder. Every state runs at the same physical temperature.

#### `rest2.tau_max`

type: number · default: `0.5` · minimum 0.0; maximum 0.95

The hottest rung's tau. The ladder is linear from 0.0 to this value. tau = 0 is the unscaled physical Hamiltonian.

#### `rest2.backbone_scaling_list`

type: string or null · default: `null`

Selective REST2: a CLAIM about the saved states, not a way to make them. The residues whose BACKBONE is hot, as a quoted AMBER residue mask of one-based topology residue indices (":45,46,59", ":45-50"). The region is chosen when `md-openmm build-top --rest2-scaler` builds the states; build-md resolves this claim and refuses it unless it is the region that build/REST2/scaler.yaml records, as it does for number_of_replicas and tau_max. Masks are compared as resolved regions, not as text. Leave all three selector keys out to accept whatever region the record holds; build-md.log then prints it. A claim is checked at generation and is not carried into resolved.config or the generated input: it is not a run setting, and `md-run` refuses it in an input.

#### `rest2.sidechain_scaling_list`

type: string or null · default: `null`

Selective REST2: a claim, as for backbone_scaling_list, naming the residues whose SIDECHAIN is hot. Chi1 belongs to the sidechain.

#### `rest2.ligand_scaling_dict`

type: mapping or null · default: `null`

Selective REST2: a claim naming hot ligand INSTANCES, as `{label: {mask: ":201", torsion_exclusions: <file> or auto}}`. The label is a name only; the instance is the residue the mask resolves to, and an exclusion file is compared by its contents, not its path (relative paths are read from this configuration's directory). The compact form `label: <file>` is refused: no instance name is recorded for it to resolve against yet.

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

Run the equilibration stages on EVERY RUNG, under that rung's own tau, instead of once at tau = 0. REST2 only; refused for any other protocol. Off by default. When true, the tau = 0 chain stops early: at minimisation under implicit solvent, and after its NPT stages under explicit solvent, which run once at tau = 0 to fix the box every rung then shares. Every rung -- tau = 0 included -- then runs eq_nvt_posres (restrained_nvt_steps), eq_nvt_posres_2 (restrained_npt_steps) and eq_nvt_free (unrestrained_npt_steps), all at fixed volume, from the ladder's starting state, each stage with its own seed per rung. The restraint is the stage chain's, on the same atoms at the same strength; the rung Systems the ladder propagates never carry it. Order: these stages, then `equilibration_steps`, then the first exchange. Neither is production. Stages of 0 steps are skipped, and all three at 0 is refused.

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

The switching path: lambda from 0 to 1, V(lambda) = (1 - lambda) V0 + lambda V1, where V0 is `-s`/`-p` (the state the source ensemble was sampled from) and V1 is `-s2`/`-p2`. Ignored unless protocol is AIS. Every length is an exact integer step count; nothing here is a duration that has to divide by a timestep.

#### `ais.number_of_paths`

type: integer · default: `100` · minimum 1

How many independent switching paths to run. Each gets its own directory, its own trajectory, and its own deterministic seeds.

#### `ais.switching_steps`

type: integer · default: `50000` · minimum 1; unit: steps

The length of the switching path, as an exact step count. 50000 steps is 100 ps at 2 fs. WORK IS PATH-LENGTH DEPENDENT: a faster switch does more dissipative work, so this is a scientific choice and not a performance knob. The log states the derived ps.

#### `ais.observation_interval_steps`

type: integer · default: `2500` · minimum 1; unit: steps

How often a path is observed: one coordinate frame and one work row. switching_steps must divide by this exactly, so the last observation lands at lambda = 1. 50000/2500 gives 20 intervals and therefore 21 observations, counting both endpoints. AN OBSERVATION IS NOT A STEP.

#### `ais.parameter_update_interval_steps`

type: integer · default: `1` · minimum 1; unit: steps

How often lambda moves. 1 changes the Hamiltonian every step -- 50000 parameter changes over the path above. observation_interval_steps must divide by this, or observations would not sit on the update grid.

#### `ais.lambda_schedule`

type: string · default: `linear` · one of `linear`, `tau-linear`

How lambda follows the switching progress t (0 -> 1). linear: lambda = t. tau-linear: lambda = [(1 - tau0 + tau0 t)^2 - (1 - tau0)^2] / [1 - (1 - tau0)^2], for V0 a saved REST2 state at tau0 and V1 its unscaled source: the mixture's solute-solute scaling then equals (1 - tau)^2 along a tau linear in t. Its solute-environment scaling does NOT equal (1 - tau); one lambda cannot follow both. The run refuses tau-linear unless -s is a saved state at tau0 whose record names -s2 as its source.

#### `ais.lambda_schedule_tau0`

type: number or null · default: `null` · minimum 0.0; maximum 0.95

V0's tau, for lambda_schedule: tau-linear, and refused with linear. When ais_source.generate is true it defaults to dynamics.tau and must equal it; build-md writes the value into resolved.config.

### `ais_source`

Where the starting configurations come from. Ignored unless protocol is AIS.

#### `ais_source.trajectory`

type: string or null · default: `null`

The equilibrium trajectory the paths start from, sampled from V0 -- typically a fixed-tau cMD run whose System is the -s given to AIS. A trajectory that records its System digest is checked against -s. Resolved relative to the directory run.sh is invoked from. Required unless `generate` is true, in which case build-md sets it to the source stage's whole-system trajectory.

#### `ais_source.generate`

type: boolean · default: `false`

Generate the source ensemble in this run instead of naming one. build-md then writes a stage chain before AIS: minimisation of the UNSCALED build/built.xml (the shared min/), the equilibration stages and a `source` production stage on V0, then the switching paths from that stage's whole-system trajectory. `run.sh` passes build/built.xml for minimisation and as V1, and the saved scaled state build/AIS/system_state0.xml (V0) for everything else. `stages.production_steps` is the source run's length and `reporting.crd_printout_whole` its frame interval, so both must be set. `dynamics.tau` must be V0's tau, which the stages check against the scaler.yaml beside the state and which makes the equilibration fixed-volume.

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
