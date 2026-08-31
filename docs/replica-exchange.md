# Replica exchange: REST2 and rREST2, through `openmm-md`

There is **one simulation executor**. A single system and a coordinated ladder are the same
command with different arguments, exactly as Amber has `pmemd` and `pmemd.MPI -groupfile`:

```bash
# one system
openmm-md -i eq/npt_free/npt_free.py -p input/topology.pdb -s input/system.xml \
          -c min/min.state.xml -o npt_free.out -x npt_free.dcd -r npt_free.state.xml

# a ladder
mpiexec -n 6 openmm-md -ng 6 --groupfile REST2/rest2.group \
          -o REST2/rest2.out -x REST2/rest2.nc -r REST2/restart.json \
          --checkpoint REST2/rest2_checkpoint.nc

# a ladder with a different transition rule
mpiexec -n 6 openmm-md -ng 6 --groupfile rREST2/rrest2.group \
          --exchange-rule rREST2/rrest2_exchange.py --reservoir rREST2/reservoir.yaml \
          -o rREST2/rrest2.out -x rREST2/rrest2.nc -r rREST2/restart.json \
          --checkpoint rREST2/rrest2_checkpoint.nc
```

A method is a protocol file plus, when the transitions differ, a rule file. It is never a new
executable, a new flag namespace and a new set of help text to keep consistent with this one.

## Who owns what

| owned by | what |
|---|---|
| **OpenMM** | System, Context, Integrator, State, and every energy evaluation |
| **MD-templates** | the REST2 Hamiltonian, the ladder, the exchange schedule and rules, the NetCDF schema, checkpointing, restart, validation, statistics |
| **MD-project** | which systems, which request, the campaign schedule, and where the analysis boundary is |

**OpenMMTools is not imported by generated production code.** It remains available as an optional
oracle in tests, and the production package and generated runtime work without it installed.

## The science

REST2 is Hamiltonian replica exchange at ONE physical thermostat temperature. tau is persisted;
everything else is derived:

```
s        = (1 - tau)^2      solute-solute terms, and solute torsions
sqrt(s)  = 1 - tau          solute-environment terms
1                           environment-environment terms
```

Peptide omega torsions are left **unscaled** -- this repository's omega-selective convention, not
an unmodified textbook REST2.

Because every rung shares one temperature and one beta, an exchange **never rescales velocities**.
That rescaling belongs to temperature REMD, where the rungs differ in beta; here it would inject or
remove energy at every accepted swap. A test asserts no runtime module does it outside the
reservoir path, where the velocity comes from the reservoir's own stored phase-space sample.

The runtime is **NVT**. It installs no barostat, and a requested pressure is refused: exchanging
complete configurations under NPT would also have to exchange volumes and carry the pV work, and
that is deliberately not implemented rather than approximated.

## Four schedules, no `segment_ps`

Exchange, whole-system output, solute output and checkpointing are **four independent schedules**.
Each is given in picoseconds, each converts to an exact whole number of integration steps, and an
interval that does not is refused rather than rounded. There is no `segment_ps`: it bundled all
four into one number, so a solute stream could never be denser than the exchange period. Asking
for it now raises an error that names what replaced it.

Propagation advances to the next event and never past it. When several events fall on the same
step they are handled in one fixed order -- `exchange, whole, solute, checkpoint` -- so a frame
written at an exchange step is the post-exchange state and the checkpoint describes everything
already written.

## Phase space, not configuration

An rREST2 reservoir stores **positions and velocities and boxes**. A DCD cannot store velocities,
so a DCD is not a phase-space reservoir and is refused as a source.

Both policies read the **same** source. The phase-space NetCDF is what carries positions, box,
absolute source step and time, the Hamiltonian identity, the atom identity and the completion
marker in one auditable file; the policy decides what is done with the recorded momenta and does
not widen the source format.

`velocity_policy: stored` is the default: the recorded momentum is installed unchanged, which is
what the probability-one acceptance assumes. It requires finite, correctly shaped, **nonzero**
velocities in every frame.

`maxwell` is an explicit alternative. It uses the source positions and box and **deliberately
ignores** the recorded velocity values, redrawing at the one common temperature from a seed
recorded per refresh -- so any single draw is reproducible from the storage alone. Under MPI only
the owning rank calls OpenMM's draw and the resulting array is shared, so serial and six ranks
install identical momenta.

`maxwell` is **not** DCD or coordinate-only support. A DCD carries no Hamiltonian identity, no
absolute source step and no completion marker, so it satisfies this contract under neither policy.
A coordinate-only source would need its own versioned, separately validated contract; there is
none.

There is no silent fallback. Velocities that are missing, the wrong shape, non-finite or
identically zero are a hard error when the source is opened, never a quiet switch to `maxwell`.

The reservoir is materialised **without replacement**, and asking for it is refused before a
directory exists. A repeated draw would give one configuration extra statistical weight, make
`frames` overstate the effective reservoir size, and produce a file this repository's own
strictly-increasing-step check rejects.

## One Hamiltonian, recomputed on both sides

The reservoir and the rung it refreshes must be the **same complete Hamiltonian**. The comparison
is a SHA-256 over the canonically serialized OpenMM System plus a digest of the enhanced-region
selection, and the current side is always **recomputed** -- two stored claims are never compared
with each other. tau and temperature agreeing is necessary and not sufficient: ff14SB/TIP3P and
ff19SB/OPC agree on both.

The system compared is the **top rung's scaled** system, not the unscaled reference the ladder is
built from. The reference is recorded with `tau: null`, which is a different claim from the rung at
`tau: 0.0`.

The fixed-tau source is generated **with** the ladder, so one input, one force field, one solvation
model and one equilibrated box are shared by construction rather than by two projects happening to
agree.

## The representation, chosen once

Contexts are **fixed to thermodynamic states**. Context *i* is built from the System scaled to
`tau[i]` and never changes Hamiltonian; an accepted exchange moves complete **configurations** --
positions, velocities and box together -- between two contexts.

The alternative, walkers fixed to contexts with state assignments moving, would need either a
System rebuild per accepted swap or every rank holding every scaled System. Copying coordinates is
cheaper than both.

Both views are reconstructable and are tested to invert each other exactly:

```
state_to_walker[i]   which walker's configuration currently sits in state i   (stored)
walker_to_state[w]   which state walker w's configuration sits in             (derived)
```

Every stored mapping row is a permutation, and the validator refuses storage where one is not.

## The group file

Plain text, one group per line, **parsed with `shlex` and never evaluated by a shell** -- a data
file that can run commands is a vulnerability, not a convenience. Blank lines and `#` comments are
ignored.

```text
-i REST2/rest2.py -p input/topology.pdb -s input/system.xml \
   -c eq/npt_free/npt_free.state.xml --solute input/solute.yaml --group-index 0
```

Group lines carry **inputs only**. `-o`, `-x`, `-r` and `--checkpoint` appear once on the outer
command, because they describe the coordinated run rather than one replica of it. Indices are
unique, contiguous and zero-based, and are stated rather than taken from line order so a reordered
file still means the same thing. Unknown fields, missing values, duplicates, run-level flags inside
a group line and index gaps are each refused **by line number**.

## The exchange-rule contract

A rule receives a bounded view -- iteration, exchange count, the mapping, a reduced-potential
lookup, a dedicated RNG, an optional reservoir, and its own persisted state -- and returns explicit
proposals and decisions. It never propagates, opens storage, touches MPI or parses a command line.

This is a contract, not a framework: no registry, no plugin discovery, no method database.
`--exchange-rule FILE.py` loads one file and takes the `rule` object it defines. A later
non-Boltzmann or kinetic reservoir becomes a new rule file and changes nothing in `openmm-md`.

### The default rule

Conventional neighbouring REST2: one sweep of adjacent pairs per exchange, with a **strictly
alternating** odd/even schedule. All four reduced potentials of a pair are evaluated
independently -- never inferred by scaling another energy, because the scaled Hamiltonians differ
by more than a single factor.

```
log(alpha) = [u_i(x_i) + u_j(x_j)] - [u_i(x_j) + u_j(x_i)]
```

The phase alternates on the **exchange-attempt count**, not the iteration number. Exchanges land
every `exchange_stride` iterations, so an even stride freezes the iteration parity and one phase
would run forever -- a three-state ladder proposed (1,2) six times and (0,1) not once before this
was corrected.

## rREST2: a Boltzmann reservoir

`rREST2` refreshes the configuration occupying the **hottest** rung from a finite, pre-generated
ensemble. Under the v1 contract -- Boltzmann-weighted, at exactly the top rung's tau, temperature,
Hamiltonian and fixed volume, holding complete configurations -- a refresh is accepted with
probability one, because the drawn configuration is a sample from the same distribution as the one
it replaces.

**That is the entire justification, so every clause is checked.** v1 refuses a non-Boltzmann,
clustered or kinetic reservoir; an NPT reservoir; solute-only insertion; and any mismatch of tau,
temperature, topology, atom order or box.

> Roitberg, Okur, Simmerling, *J. Phys. Chem. B* 2007, **111**, 2415 (doi:10.1021/jp068335b)
> introduced reservoir REMD and derived the Boltzmann acceptance rule.
> Kasavajhala, Lam, Simmerling, *J. Chem. Inf. Model.* 2020, **60**, 1218 (PMCID PMC7725893) show
> what non-Boltzmann reservoirs require instead, and that using the Boltzmann rule with a reservoir
> that is not Boltzmann-weighted biases **every** replica -- the ladder propagates the error down.

The **finite-reservoir approximation** is real and is written into every record: N configurations
are not the top state's full equilibrium distribution, and the assumption that the selected source
window represents it is an assumption, not a result.

### The order, chosen once

When an exchange iteration and a reservoir attempt coincide, the **neighbouring sweep happens
first** and the refresh second. A refresh installs a configuration this ladder has not propagated;
going first would let it be swapped down the ladder in the same iteration, having never been
propagated in the ladder at all. The order is recorded in the rule's `describe()` and in the
manifest, never left to scheduling.

### Where the reservoir comes from

A fixed-tau cMD run at the top rung's tau -- `cMD_tau0p5` when `tau_max` is 0.5. **The path is not
evidence.** tau, temperature, ensemble and the frame-time map come from that run's own
`resolved_run.yaml`, written where they were decided. A directory name can be renamed or copied,
and a reservoir drawn from the wrong ensemble runs to completion while being wrong. The prepared
manifest records `tau_from_directory_name: false`.

## The shared source reader

`source_ensemble.py` holds the rules for drawing configurations out of an equilibrium production
run. AIS established them; rREST2 needs the same ones; there is one implementation, because a
second is a second thing that can be wrong and its wrongness would be silent.

- never `mdtraj.load` a production trajectory -- bounded `iterload` only;
- never hash a production trajectory at run time; record bounded observations instead;
- a frame index is not a time and a DCD header is not the production clock;
- source tau is never a directory name, and a declared/recorded conflict is refused, not resolved;
- atom NAME is not identity -- full per-index identity and bond connectivity are compared;
- a DCD carries no velocities; momenta are redrawn at the common temperature from a recorded seed;
- box vectors come from recorded exact values, reduced for OpenMM;
- the selection is materialised **once** with a manifest, after which the source is never reopened;
- prepared inputs that disagree with their manifest are **refused**, never silently regenerated.

Two boxes are compared as **lattices**, not as numbers: reduction has a genuine boundary case at
`|c_x| = a_x/2` where `+a_x/2` and `-a_x/2` describe one rhombic dodecahedron. The check is an
integer change of basis, which is exact.

## Storage, restart and validation

Four records, four jobs:

| record | written | job |
|---|---|---|
| analysis NetCDF (`-x`) | every committed iteration | the authoritative history, plus the scientific identity written **before** propagation begins |
| checkpoint NetCDF | every `whole_output_stride` | configurations, mapping, RNG states, rule state, iteration and budget |
| `<stem>.runstate.json` | atomically, on transition | `initialized` → `running` → `completed`/`interrupted`/`failed`; never claims completion |
| `restart.json` (`-r`) | atomically, at the end | evidence of completion; **never** a precondition for resuming |

Each stream's marker -- `last_exchange`, `last_frame`, `last_solute_frame` -- is written **after**
every array for that row, so an interrupted write leaves the counter on the previous, complete row.
A marker can therefore lag its data but never lead it, and a marker ahead of the rows the file
holds is refused as a file that disagrees with itself.

A **resume continues from the last checkpoint**, not the last committed row: the exchange stream
commits every attempt while checkpoints are written less often, so an interruption leaves committed
rows with no configurations behind them. Those rows are rewound. Before anything is opened for
writing, the stored output is read and validated: markers within their rows, steps strictly
increasing in every stream, and the stored exchange steps equal to the **scheduled** ones -- a
dropped attempt in the middle leaves increasing steps and can still reach the budget, so nothing
else would see it.

`--resume` needs no `restart.json`. `--extend N` requires a run that reached its budget and adds
exactly N attempts **to the budget that run reached**, not to the one in the protocol file: a run
already extended once stores a larger budget than the protocol describes, and extending the
protocol's would produce a total below the steps already run. For the same reason the validator
treats the identity's `number_of_exchanges` as a **floor** -- an extended run legitimately holds
more rows than the protocol was created with.

Both flags are **grouped only**. The conventional single-protocol path builds a fresh run -- it
resets time and step state and creates its reporters from scratch -- so append-safe continuation
of a cMD stage is not implemented, and asking for it is refused in the validation pass before a
directory is created, a report is opened or the protocol is imported. A refused invocation leaves
every existing byte untouched. `--force` is a deliberate fresh-run replacement, not continuation.

An interruption is neither success nor failure and exits with its own status (130). The executor's
promise check -- "the protocol finished but did not write X" -- applies only to a run that claims
to have finished.

```bash
openmm-md --verify-only -x REST2/rest2.nc --checkpoint REST2/rest2_checkpoint.nc \
          -r REST2/restart.json
```

The validator **opens and reads** the files. A file that exists is what a crashed run leaves
behind, so existence is never treated as completion.

## Parallel policy

World size must be **1 or exactly the number of states**. Nothing in between: a policy that
silently packed several states onto a rank would make the device assignment and the timing
unreproducible, and "it ran" would stop meaning "it ran the way the record says".

Nothing binds ranks to devices automatically, so every rank binds explicitly and records rank,
host, visible devices, selected `DeviceIndex`, precision and the states it drives.

At each exchange the **full n x n** reduced-potential matrix is evaluated -- rank *r* computes row
*r*, which parallelises exactly. That is more work than a neighbouring sweep needs, and the reason
is that it makes the rule contract a pure lookup with no communication and leaves a complete matrix
in storage.

## Limitations

- NVT only. NPT exchange is refused, not approximated.
- v2 reservoirs are Boltzmann-weighted and complete-configuration only.
- `--extend` and `--resume` are coordinated across ranks at an event boundary; only rank 0 opens
  the analysis storage for writing, in any mode.
- Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
- Smoke runs are picoseconds and validate neither ladder quality nor convergence.
- The legacy `md-gen --method REST2` loop still exists for the contract-managed route and its
  existing datasets; it is marked legacy and points at `openmm-md`.
