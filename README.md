# MD-tools

Standalone OpenMM tooling: build a topology, generate readable MD run scripts, and register
finished data into managed storage. One executable, `md-openmm`, and three commands.

```bash
pip install md-tools
```

The scientific stack — OpenMM, the OpenFF toolkit, openmmforcefields, AmberTools/ParmEd/RDKit —
comes from your conda environment. Install this package into that environment with `--no-deps`, or
pip will resolve a second, pip-built OpenMM alongside the conda one.

---

## The three commands

```text
md-openmm build-top        one input structure  ->  built.xml + built.pdb + built.log
md-openmm build-md         a protocol config    ->  readable run scripts in ./md_script/
md-openmm data-register    a finished directory ->  a verified dataset under $MD_DATA
```

Nothing else is public. There is no `setup`, no `sys-gen`, no `md-gen`, no separate `openmm-md`
executor and no environment installer; what those did that is still needed is reached through
these three.

### A whole run, end to end

```bash
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log
md-openmm build-md  -odir ./md_script/
cd md_script && ./run.sh ../built.pdb ../built.xml
md-openmm data-register -idata . -project_name ALA -data_name ALA-cMD -year 2026
```

---

## `build-top`

```text
md-openmm build-top -i INPUT.pdb|INPUT.smi [-os built.xml] [-op built.pdb] [-log built.log]
                    [--config CONFIG] [--overwrite]
```

`-i` names an existing **file**: a `.pdb` for a peptide, or a `.smi` holding exactly one SMILES
record for a single molecule. An inline structure is not accepted, because the input is hashed into
the record and a string cannot be.

`built.pdb` and `built.xml` are a **pair**: the serialised System's particle order matches the PDB
exactly. The build refuses to write either if they disagree, and re-reads both after writing them.

Defaults: ff14SB, Sage 2.2.1 for small molecules, TIP3P, a rhombic dodecahedron with 1.5 nm of
requested solute padding, neutralised and then brought to 0.15 M with Na⁺/Cl⁻, HBonds constraints,
and **no** hydrogen mass repartitioning. OPC and GBn2 are the supported alternatives.

Combinations that are individually reasonable and jointly wrong are refused, not ignored:

* GBn2 is implicit solvent — no box, no periodic boundary, no ions, no barostat, no NPT stage
  anywhere downstream. Setting any box, salt or cutoff key alongside it is an error.
* ff19SB was parameterised against OPC; pairing it with TIP3P is refused.
* A `.smi` with `solute.peptide: true`, or a `.pdb` with `false`, is refused.

Sage is used only where a small molecule needs it. A peptide-only build records `ligand: null`,
and a ligand-only build records `protein: null` — what **resolved**, not what was requested.

The commented example is [`configs/sys/build-top.config`](configs/sys/build-top.config).

## `build-md`

```text
md-openmm build-md [-odir ./md_script/] [--config CONFIG] [--all-in-one] [--overwrite]
```

Every length is an **exact integer step count**. A step count is exact; a duration in picoseconds
depends on a timestep the file may not have been written against. Each stage log prints both the
count and the physical time it works out to, but the count is what is stored and what runs.

Default explicit-solvent cMD:

| stage | steps | at 2 fs |
|---|---:|---:|
| minimisation | 1000 iterations | not a time |
| restrained NVT | 5 000 | 10 ps |
| restrained NPT | 5 000 | 10 ps |
| unrestrained NPT | 5 000 | 10 ps |
| production | 2 500 000 | 5 ns |

emitted as `min.py`, `eq_nvt_posres.py`, `eq_npt_posres.py`, `eq_npt_free.py`, `cMD.py`, `run.sh`.

Under implicit solvent the two pressure-coupled stages are **renamed** to `eq_nvt_posres_2.py` and
`eq_nvt_free.py` and run NVT. They are not NPT stages with the pressure quietly ignored: there is
no box, so there is no volume to control and no barostat is added.

`--all-in-one` emits a single `md.py` running the identical stages with identical resolved
settings, boundaries, logs, checkpoints and restart semantics. Only the process count differs.

Generated scripts are small entry points that import the installed `md_tools` runtime. They contain
no absolute path and no reference to a source checkout, so the directory can be moved anywhere.
Each takes the same Amber-like flags:

```text
-p  topology PDB     -x  output trajectory    -log  the log, carrying the machine record
-s  System XML       -r  output final state   -chk  output checkpoint
-c  the previous stage's final state
```

`run.sh` chains them with explicit relative paths, `set -euo pipefail`, safe quoting and no `eval`.
A stage that already reports completion is **not** rerun. A partial stage resumes from its
checkpoint only if that checkpoint's fingerprint matches this stage's configuration; a checkpoint
written under a different ensemble, timestep, temperature, seed or System is refused. Asking for a
longer run is the one change that does not invalidate a checkpoint, because extension is legitimate.

Examples are [`configs/md/cMD.config`](configs/md/cMD.config), [`REST2.config`](configs/md/REST2.config) and [`rREST2.config`](configs/md/rREST2.config).

### REST2 and rREST2

`protocol: REST2` generates a replica-exchange ladder; `rREST2` adds a Boltzmann reservoir refresh
of the hottest rung. The scientific contract is a property of the implementation and is **not**
configurable:

* every replica runs at the **same physical temperature** — this is Hamiltonian scaling, not
  temperature REMD;
* bonds and angles are unscaled;
* eligible solute torsions and CMAP scale with `(1-tau)²`;
* ordinary amide omega torsions are left **unscaled**, so peptide bonds do not rotate at the hot
  rungs;
* solute–solute ordinary nonbonded and 1-4 terms scale with `(1-tau)²`;
* solute–environment nonbonded terms scale with `(1-tau)`;
* the generalised-Born contribution scales with `(1-tau)`;
* an exchange never rescales velocities;
* the runtime is NVT.

Each **thermodynamic state** gets its own trajectory, `remd0.nc … remdN.nc`. A state trajectory
follows the state, not the walker, and the filename carries the state index — never the tau value.
An Amber-style `rem.log` projection is written, and acceptance is reported for **each neighbouring
pair**, because a single averaged number hides a ladder with one impassable gap.

`dynamics.tau` runs cMD held at one rung of that ladder — the same scaling, fixed. That is how a
hot ensemble is produced without exchanges, and with `dynamics.phase_space_printout` it is how the
Boltzmann reservoir an rREST2 run refreshes from is generated. A scaled run is NVT by construction:
it must sample the fixed-volume ensemble of the rung it sits at.

## `data-register`

```text
md-openmm data-register -idata DIR -project_name NAME -data_name NAME -year YYYY
                        [--common-data] [--dry-run] [--verify-only]
md-openmm data-register --init
```

Canonical destinations, **contract v2** — there is no month segment:

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/{year}/common/{project_name}/{data_name}/     # --common-data
```

`year` is the year the data were **completed**, and it is checked against the records rather than
accepted as a label.

Registration, in order: resolve the storage root; validate the names as single safe path segments;
discover every structured record recursively; refuse anything missing, malformed, failed,
incomplete or still being written; validate lineage against exact input/output digests; inventory
every file with its relative path, size and SHA-256; derive the manifest and validate it against the
contract; compute the destination; refuse a collision unless the existing destination is provably
identical; stage transactionally beside the destination; re-read the destination and verify every
digest; remove the source **only then**; replace it with a symlink; and validate once more.

If any check fails the source is preserved, unchanged. Two datasets are never merged in place,
completed data are never overwritten, and `shutil.move` is never treated as proof that a
cross-filesystem transfer arrived intact.

**Completion is read from a structured record, never from prose.** Every log this package writes
carries one delimited, versioned YAML block, and a reader parses that block and nothing else. A log
with no block, two blocks, an unparseable block, or a `schema_version` this build does not know is
not evidence of anything. `status: completed` is written only after the outputs are flushed,
reopened and re-read.

### User configuration

`--init` writes your identity and storage root. It is never stored in the installed package or in
the source repository: those may be read-only, are shared between projects, and a wheel is not a
place to keep somebody's paths.

```text
1. --user-config PATH
2. $MD_TOOLS_CONFIG
3. ${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config
```

The storage root resolves separately, because it changes far more often than an identity:
`--md-data`, then `$MD_DATA`, then `machine.md_data`. Which source supplied it is always printed.
A commented example is [`configs/machine/user.config.example`](configs/machine/user.config.example).

---

## Configuration examples

Browse them here; they are ordinary files, not a symlink:

```text
configs/
├── machine/user.config.example     identity and storage root, for `data-register --init`
├── sys/build-top.config            force fields, solvent, box, ions, constraints, HMR
└── md/
    ├── cMD.config                  plain molecular dynamics
    ├── REST2.config                the replica-exchange ladder
    └── rREST2.config               REST2 with a Boltzmann reservoir
```

They install with the wheel as data files under `<prefix>/share/md-tools/configs/`, and
`md_tools.configs.example_root()` finds them there through the distribution's own metadata. There
is exactly one copy of each file: the one in this repository.

## Configuration files

Every configuration is YAML — despite the `.config` suffix, which says what the file is *for* —
is validated against a declared schema, and **rejects what it does not recognise**. An unknown key
is an error naming the nearest known key, not a value that is silently dropped:

```text
solvent: unknown key(s) 'padding_nn' (did you mean 'padding_nm'?)
```

That matters because a silently ignored setting produces a run that completes, logs cleanly, and
answers a different question than the one that was asked.

The shipped examples are **generated from the schema objects that enforce them**, so a comment
documenting a default and the default the code applies have one source and cannot drift. A test
regenerates them and compares.

## Tests

```bash
python -m pytest tests -m "not slow and not gpu"    # fast
python -m pytest tests -m "gpu or slow"             # builds systems and integrates them, on CUDA
```

GPU tests run on CUDA and are deselected rather than quietly passing on the wrong platform: a
CPU-only machine cannot produce evidence about the platform the work is actually done on.

## Where this sits

**MD-tools** owns topology construction, script generation, the OpenMM runtime those scripts use,
the run records, the dataset contract and its validators, and the registration transaction.

**MD-project** — or any project — is a consumer. It installs this package, keeps its own inputs,
configurations and analysis, and never vendors, pins, clones or imports a source checkout of it.

The dataset contract in `src/md_tools/data_contract/` is ported from
[MD-data](https://github.com/csy0000/MD-data), branch `protect-md-project-dev-test`, commit
`20d982eb`. `md_data` is not a runtime dependency of anything here.

## Requirements

Python ≥ 3.11, OpenMM ≥ 8.6, and the OpenFF stack for ligand parameterisation. `pyyaml`, `numpy`
and `pydantic` are the only declared dependencies.
