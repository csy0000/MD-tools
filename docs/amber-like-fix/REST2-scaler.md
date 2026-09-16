# REST2 scaling as a build step: `build-top --rest2-scaler`

2026-09-16. **Design decided by the user; no code written yet.** Owner: the `md-tools-omega-fix`
session. Touches the AIS redesign (`AIS-two-topology.md`, owned by `md-tools-ais`) and `build-md`
(where `MD-tools-main` is retiring `--all-in-one`), so both are named where they are affected.

---

## 1. Decisions

| question | decision (user, 2026-09-16) |
|---|---|
| a fifth command? | **No.** A mode of `build-top`: `md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config`. The four-command contract stands. |
| where the tau schedule lives | **`scaler.config`**: `kind: linear`, `n_states`, `tau_min`, `tau_max`. |
| how many scaled sets per dataset | **One per method.** `build/<method>/system_state<n>.xml`, described by `build/<method>/scaler.yaml`. Different methods may use different schedules. |
| `tau_min > 0` | **Allowed; defaults to 0.** |
| several non-standard residues | **Refuse a guess.** Each residue name `X` is read from `build/X.sdf` by default, or from an explicit `sdf_filelist` in `scaler.config`. |
| `omega_exclusion` | **Kept**, `true` by default; `omega_exclusion: false` in `scaler.config` turns it off. |
| the System a hot-cMD integrates, and AIS's V0 | **The saved state**, `build/<method>/system_state<n>.xml`. A fixed-tau stage does not write its own `system.xml`. |
| `build-md` for REST2 | **Scales nothing.** Checks the scaled states exist and verify; otherwise refuses, naming the command to run. |

rREST2 is being archived for 0.5.4 (user's decision, done by md-tools-ais after AIS), so only
REST2 and fixed-tau cMD are in scope here.

## 2. Why

Scaling is done today in three places with three lifetimes: `build-md` writes
`REST2-runN/remd<n>/build_state<n>.xml` once per RUN; the fixed-tau cMD preflight scales in memory
and saves nothing; AIS scales in memory through `TauSwitcher`. Each had to be taught the omega
refusal separately (`7a4ea5d` wired it into six call sites). A build step makes "which Hamiltonian"
one decision per dataset and method, made once, readable in `build/` before any run, and separate
from "how it is sampled".

## 3. The command

```text
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
```

* `-s` / `-p` have md-run's meanings: the serialised System and its topology. Both REQUIRED in this
  mode; `-i`, `-os`, `-op` are refused by name in it, and `-s`/`-p` are refused without it.
* `--config`, not `-config`: `build-top` already spells it that way and abbreviation is off.
* Nothing is rebuilt. No force field is loaded, no charges computed: the input System is scaled.
* Output goes to `<directory of -s>/<method>/`. `--check` creates nothing. An existing
  `build/<method>/` is refused unless `--overwrite` (§7 on what overwrite must not silently do).
* Every refusal runs before the first directory exists.

## 4. `scaler.config`

```yaml
# md-openmm build-top --rest2-scaler -- scaled Hamiltonians from a built System
method: REST2            # [REST2 | cMD] names the output directory build/<method>/
schedule:
  kind: linear           # the one implemented kind; anything else refused
  n_states: 4            # [int >= 1]
  tau_min: 0.0           # [float, default 0.0]
  tau_max: 0.5           # [float < 1]
omega_exclusion: true    # [bool, default true] false scales ordinary amide omegas too
sdf_filelist:            # [map, optional] residue NAME -> SDF, relative to this file
  MO1: MO1.sdf
  MO2: MO2.sdf
proline_like_residues: [PRO]   # names whose amide stays ELIGIBLE for scaling
max_proline_ring_size: 7
```

* `method: cMD` is the fixed-tau hot run: normally `n_states: 1` with `tau_min == tau_max`.
* `n_states >= 2` requires `tau_min < tau_max`; `n_states == 1` requires them equal.
* The schedule is computed by `remd.generated.tau_ladder`, the one implementation, extended to
  `tau_min`; never re-spelled (its rounding is why a second spelling once broke every CV resume).
* **`tau_min > 0` on a ladder** is accepted, and both `scaler.log` and the REST2 `.out` say, in
  words, that state 0 is NOT the physical Hamiltonian: REST2 recovers the physical ensemble only
  from an unscaled state, and a reader of state 0 must not assume otherwise.
* **`omega_exclusion: false`** is recorded and printed as "every torsion scaled, including ordinary
  amide omegas". No classification runs, so no unclassified refusal either.
* Paths in `sdf_filelist` resolve relative to `scaler.config`, the repository's convention; with the
  config in `build/`, `MO1.sdf` is `build/MO1.sdf`.
* `proline_like_residues` and `max_proline_ring_size` are the classifier's existing arguments,
  which no user configuration could reach (`_legacy_cfg` rebuilds `rest2` from `DEFAULTS`). This
  file is their home.

## 5. Which SDF describes which residue

For every non-standard **solute** residue name `X` (`solute_atom_indices`, so `HOH`, `NA`, `CL`
never count; a name in `proline_like_residues` needs no SDF), in order:

1. `sdf_filelist[X]` when given;
2. otherwise `build/X.sdf` beside `-s`;
3. otherwise, and only when `X` is the ONLY non-standard solute residue name, `built.sdf` beside
   `-s` -- what `build-top` writes for a `.smi`/`.sdf` input today;
4. otherwise **refused**, listing every name without an SDF and the paths looked for.

A residue name in `sdf_filelist` that is not in the solute is refused too: a map naming a residue
that is not there is describing a different system.

**Classifier change.** `classify_omega_bonds(..., ligand_sdf=)` maps ONE SDF onto ALL non-standard
residues together. It becomes `residue_sdfs={name: path}`, mapping each SDF onto that name's atoms
only (every instance of the residue). The per-candidate evidence rule (`35d216c`) is otherwise
unchanged, and `omega_exclusions` stays the one enforcing entry point.

## 6. Outputs

```text
build/<method>/system_state<n>.xml   one per state, n = 0 .. n_states-1
build/<method>/scaler.yaml           what these files are and how they were scaled (machine record)
build/<method>/scaler.log            the same, for a person
```

`scaler.yaml` records: the method; the schedule as configured and each state's tau, file name,
sha256, size and scale factors (`(1-tau)^2`, `(1-tau)`); the base System's and topology's sha256;
the solute atoms; `omega_exclusion`; every omega decision with its evidence and the SDF it came
from; the excluded torsion indices; the proline-like settings; the scaler convention and version;
the `scaler.config` sha256; and the md-tools version and commit. Written atomically, last, so a
directory without it is an incomplete build. Its schema is generated from a model and
drift-checked, like every other record.

The scaler writes `build_scaled_system`'s default form, which is what ladder rungs and a fixed-tau
stage at tau > 0 already integrate (`prepare_for_switching` only matters at tau = 0, and
`reparameterise_for_global_switching` is reached only through `TauSwitcher`, which md-tools-ais is
deleting).

## 7. Never scale twice, and never swap a Hamiltonian under a run

A scaled file given where an unscaled one is expected, with tau > 0 in a run config, takes
solute–solute to (1−τ)⁴ with entirely plausible numbers. `build/rungs.py` names this hazard; **no
refusal of it exists in `src/`**. Once `build/<method>/system_state<n>.xml` is an ordinary input:

* **one identity function**, callable from every preflight, returns
  `{"record": ".../scaler.yaml", "method", "state", "tau", "system_sha256"}` for a System some
  record beside it names, or `None`. md-tools-ais calls it to name V0 in the AIS run record;
* a **request to scale** a System that has an identity -- `tau > 0` in a stage config, a ladder built
  over it -- is refused. **Using one as an input is not**: AIS scales nothing, so V0 =
  `build/REST2/system_state<n>.xml` with V1 = `build/built.xml` is accepted;
* a stage's `dynamics.tau` becomes a claim checked against the record's tau, refused on mismatch;
* a file inside `build/<method>/` whose digest the record does not list is refused.

**`--overwrite` must not silently change a run that already used the old states.** Every run records
the sha256 of the state files it integrates; its preflight (and every resume) compares, and a
mismatch is refused by name. Overwriting `build/REST2/` then fails loudly on the runs it
invalidates rather than changing their Hamiltonian between segments.

## 8. What else changes

* **`build-md` REST2.** `write_rung_systems` is no longer called and is removed. The group file names
  `../build/REST2/system_state<n>.xml`. Refused before anything is written when
  `build/REST2/scaler.yaml` is missing ("run `md-openmm build-top --rest2-scaler` with
  `method: REST2` first"), when a state file fails its digest, or when a record is incomplete.
  `rest2.number_of_replicas` and `rest2.tau_max` leave `REST2.config`, refused with the migration
  message; the state count comes from the record. Sequenced after MD-tools-main's `--all-in-one`
  retirement, which is editing `build/md.py` and `cli/md_openmm.py`.
* **Fixed-tau cMD.** The hot stages' `-s` names `../build/cMD/system_state0.xml` and the stage scales
  nothing. `_prepare_stage` loses its scaling and its `omega_exclusions` call. The Hamiltonian is
  unchanged; only where it comes from moves.
* **AIS (md-tools-ais).** V0 for a REST2 switch is a saved state, V1 is `build/built.xml`. The
  "fixed-tau stage writes `system.xml`" proposal in `AIS-two-topology.md` §4 is dropped by the
  user's decision.
* **The AST guard** in `tests/test_omega_unclassified_is_refused.py` gains the scaler as the one
  scaling site; the ladder preflight's `solute_document` and the executor fallback shrink with it.
* **Runs generated before this** keep `REST2-runN/remd<n>/build_state<n>.xml`; the executor still
  reads them.
* **`data-register` / contract v2** must accept `build/<method>/`; checked before code.
* **Wheel data**: `configs/sys/scaler.config`, one copy, resolving through the real resolver to the
  model's defaults.
* **CLAUDE.md**: the `build-top` line in "The package", the configuration list, and the REST2, cMD
  and AIS invariants that say where scaling happens.

## 9. Order of work

1. Classifier: `residue_sdfs`, with failing tests first (`openmm/system.py`; no collision).
2. Scaler core, record, identity function and `scaler.config` schema (new module; no collision).
3. Runtime refusals of §7 (`run/preflight.py`, stage functions only; agree with md-tools-ais).
4. `build-top --rest2-scaler` CLI mode, then `build-md` REST2 and fixed-tau cMD (after the
   `--all-in-one` retirement lands).
5. Docs, example config, CLAUDE.md, `data-register` check, a slow end-to-end test (build-top →
   scaler → build-md → preflight), and GPU evidence for REST2 and hot-cMD on CUDA.

---

## 10. From "omega" to "unscaled torsions" (user decision, 2026-09-16)

**This changes a scientific invariant** in CLAUDE.md ("ordinary amide omega torsions unscaled;
eligible solute torsions and CMAP by (1-tau)²"). It is the user's decision, not a defect fix, and
it changes the Hamiltonian of every REST2, rREST2, hot-cMD and AIS state with a scaled solute --
proteins included.

### What stays unscaled

| class | identified by | scope |
|---|---|---|
| **ordinary amide ω** | as today: per candidate, residue name or SDF, proline-like exception (ring ≤ 7) | proteins and small molecules |
| **aromatic ring bonds** | small molecule: RDKit aromatic bond in the SDF. Protein: a table of ring bonds for PHE, TYR, TRP, HIS/HID/HIE/HIP | both |
| **other double bonds** | small molecule: non-aromatic bond order 2 (C=C, C=N, N=N ...). Protein: a table (ARG guanidinium NE–CZ, CZ–NH1, CZ–NH2) | both |
| **impropers** | a torsion term whose four atoms are NOT a bonded chain i–j–k–l, found from the topology's bonds (not from atom order: Amber puts the centre third, SMIRNOFF second) | every solute improper |

For the first three classes the unit is still the **central bond**: every proper torsion across it
is left unscaled. Impropers have no central bond, so they are a separate predicate.

The protein table is checked by a test against OpenMM's `residues.xml`, so an atom name that does
not exist fails the suite instead of silently exempting nothing.

### Consequences

* **Every non-standard residue now needs bond orders**, not only those holding an amide: a
  small molecule's aromatic and double bonds cannot be read from a topology. A non-standard
  residue with no SDF is refused, as an unclassified amide is today. Chinolin, which needed no
  SDF under the amide-only rule, needs one now (`build-top` writes it).
* **The convention changes version.** `REST2_IMPLEMENTATION` "rest2-no-bond-angle-omega" v2
  becomes a new name and v3, and `ordinary_amide_omega: unscaled` becomes a list of the classes.
  Records written under v2 remain readable.
* **Existing runs.** A ladder whose rungs are files (`build-md`'s, or `build/<method>/`) keeps its
  Hamiltonian. A run that scales at run time from `built.xml` would now build a DIFFERENT
  Hamiltonian on resume; its checkpoint fingerprint must refuse that rather than continue.
* **The pictures** colour every unscaled class red (bonds and improper centres).

### The rename

`omega_*` becomes `unscaled_torsion*` everywhere: `omega_exclusions` → `unscaled_torsions`,
`UnclassifiedOmegaError` → `UnclassifiedTorsionError`, the `scaler.config` key
`omega_exclusion` → `unscaled_torsions` (bool, default true), record keys, logs, and pictures.
Records written by 0.5.3 (`solute.yaml` `rest2.omega_excluded_bonds`, `restart.json`,
`build_states.log`) are still READ under their old keys, so existing runs resume and register.

### Open

1. Impropers: ALL solute impropers, or only those centred on an atom of an unscaled class?
2. ARG guanidinium as "double bonds": included, or aromatic/true double bonds only?
