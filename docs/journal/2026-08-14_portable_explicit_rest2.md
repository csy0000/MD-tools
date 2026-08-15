# Making explicit-solvent REST2 usable from another repository

**2026-08-14.** Status: **the portability layer works and is verified end to end.** A wheel built
from this checkout, installed into an environment that cannot see the source tree, runs
`prepare → rest2` from a temporary directory and produces a complete run. No scientific gate moved:
the ladder evidence, the 2 fs/4 fs question and the restart question are exactly where they were
yesterday.

## What was actually blocking reuse

The science was already in one place — `escort_ais.systems.explicit_baseline`, 2392 lines, driven
by a single configuration tree. What could not travel was everything around it.

Four things, and none of them were "missing documentation":

1. **The entry points were documentation scripts.** `docs/implementation/explicit_solvent/scripts/`
   held the stage CLIs. A consuming repository would have had to vendor a path inside this
   repository's `docs/` tree, which is not an interface.
2. **The force-field route was inferable rather than declared.** cyclo-(RGDfV) is a cyclic
   pentapeptide parameterised as a Sage/AM1-BCC *ligand*. "Peptide" is a defensible guess from the
   name and a wrong one; an ff19SB run of the same molecule is a different calculation that
   produces numbers under the same label. The old preset enforced this correctly, but only for
   invocations that went through the preset.
3. **The launcher encoded one machine.** `run_all.sh` hardcoded `CUDA_VISIBLE_DEVICES=0/1/2` and
   assumed three GPUs. On a one-GPU machine jobs two and three land back on device 0, where two
   REST2 processes sharing a card without CUDA MPS run **~3.7× slower each** — wrong, but not
   visibly wrong.
4. **There was no unit that could be handed to someone.** A prepared box existed as a scatter of
   `simbox_*.xml/pdb/json` files with no manifest tying them to a molecule, a force field or a
   toolchain.

## What was built

`src/escort_ais/explicit/` — a portability layer that calls the existing science and adds nothing
scientific of its own. One console script, `escort-explicit`, with six subcommands.

**Two versioned manifests, deliberately separate.** A *system* manifest is molecular identity plus
parameterisation; an *experiment* manifest is seeds, integrator, ladder, budget, precision. The
split lets one system run under several experiments without either file encoding the other's
choices.

The system manifest pins identity rather than describing it. `input.route` may never be `auto`; a
ligand route may not name a protein force field and a peptide route may not name a small-molecule
one; the molecule is fixed by its RDKit canonical isomeric SMILES *and* that string's SHA-256, so
an edited manifest fails to load rather than silently parameterising something else.

`cyclo_rgdfv` is pinned hardest:

```
canonical SMILES  CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)[C@H](CCCNC(N)=[NH2+])NC1=O
sha256            59d4422635f77292ed94c609a76808df5c97110be09cd9779819270363143159
formal charge     0        (the zwitterion: Asp deprotonated, Arg guanidinium protonated)
formula           C26H38N8O7, 4 stereocentres, all assigned
```

That name may not be attached to an arbitrary neutral SMILES, because "a neutral macrocycle called
RGD" is precisely what a copy-paste error produces.

**Portable bundles.** `prepare` writes a self-contained directory — System, topology, equilibrated
state, both manifests, the resolved config, and a `bundle_manifest.json` carrying a SHA-256 for
every file plus identity, formal charge, force fields, realized ion counts, atom/water/ion/
constraint/DOF counts, box vectors, omega exclusions, every seed, package version, git SHA and
dirty state, wheel SHA-256, the full toolchain, `sqm`'s path, hardware, the exact command, and a
UTC timestamp. `validate-bundle` recomputes all of it and additionally re-derives the config hash
from the manifests inside the bundle, so swapping a manifest and rehashing the file is still
caught.

**Immutable runs.** `RUN_ROOT/YYYYMMDDTHHMMSSZ__SYSTEM_ID__rest2__CONFIG_HASH/`, refused if it
exists. The config hash derives from the canonical serialisation of both manifests and excludes
the platform and device, so the same calculation on CPU and on CUDA has the same hash. Statuses
are `running` / `completed` / `failed` / `interrupted`, and `completed` requires the observed
exchange rounds — counted from `exchange_attempts.csv`, not from the driver's return value — to
reach the planned ones. `SIGTERM` is handled, so a preempted job records `interrupted` instead of
being left at `running` for ever.

## Four things the work exposed that were not in the plan

**A bundle needs `simbox.json`.** A System XML carries parameters but no atom names, nothing that
says which atoms are the solute, and nothing about which bonds are omega. The first end-to-end
attempt failed at `_load_bundle` because the mandated bundle layout uses bare filenames while the
driver addresses a System by the `<stem>_system.xml` convention. Both facts are now handled:
`simbox.json` is a bundle file, and the runner stages a `<stem>_`-named view of the bundle inside
the run directory (copies, not symlinks — the run must stay meaningful after the bundle moves).

**The box-growth policy leaves no margin.** The smoke aborted twice with "periodic box size has
decreased to less than twice the nonbonded cutoff". The cause is that when a requested box is
smaller than the minimum image the policy grows it to *exactly* twice the cutoff, so the first NPT
step is guaranteed to cross the limit. Increasing padding does not help while the result still
lands below the requirement — the padding has to be large enough that no growth happens at all.
The smoke now pairs 1.2 nm padding with a 0.7 nm cutoff (~34 % margin). **This is a latent trap for
any small solute**, not a smoke-specific quirk; a production system with a tight box and a 1.0 nm
cutoff can hit it.

**A package directory called `data/` is invisible to git here.** The shipped manifests were first
written to `src/escort_ais/explicit/data/`. `git add` then staged *nothing* from it, silently: this
repository gitignores `data/` at **any** depth, which `CLAUDE.md` records as a trap for report
bundles and which applies just as well to package data. Everything worked on this machine — the
wheel built, the manifests shipped, the smoke passed — while a fresh clone would have produced a
wheel with no shipped manifests and a `--system cyclo_rgdfv` that resolves to nothing. The
directory is now `manifests/`, and `test_shipped_manifests_are_not_hidden_from_git` asserts both
the name and that `git check-ignore` does not match, so the failure cannot recur quietly.

**`validate-env` earns its place immediately.** The first outside-checkout attempt failed at
`sqm: not on PATH` and exited 4 before doing any work — the venv had the wheel but not AmberTools.
That is the intended behaviour and it is worth stating: AM1-BCC silently becomes unavailable when
`sqm` is missing, because the OpenFF toolkit discovers it by PATH lookup.

## Verification

Everything below was run, not planned.

| check | result |
|---|---|
| `python -m build` | `escort_ais-0.1.0-py3-none-any.whl`, sha256 `d89c421568a7daf7242b7d9e30ea6bc2a34fbdb3f39ee4fe1665daec4e5268c5` |
| wheel contents | entry point + all five shipped manifests present under `explicit/manifests/` |
| `pytest test_rest2_scaler test_rest2_lambda_system test_explicit_baseline -v` | 76 passed, exit 0 |
| `pytest -m "not slow" -q` | 820 passed, 3 skipped, 2 deselected, exit 0 |
| `pytest tests/test_explicit_portable.py -m slow` | 1 passed (66 s), exit 0 |
| `git diff --check` | clean, exit 0 |
| wheel installed outside the checkout, `escort-explicit smoke --platform CPU` | `status: completed`, 4/4 exchange rounds, exit 0, ~67 s (re-run after the `manifests/` rename) |

`tests/test_explicit_portable.py` adds 49 behavioural tests. Two deserve mention because they
guard the guards: `test_developer_path_detector_actually_detects` fails if the machine-local-path
detector stops firing (otherwise the portability assertions are vacuous), and
`test_executable_source_filter_is_not_vacuous` checks the comment/string stripper used to scan for
hardcoded device indices — without it, the modules would have to delete the sentence explaining
what they replaced in order to pass.

No test asserts `hashlib.sha256(x).hexdigest()` is truthy. Every hash assertion either recomputes
the digest independently or checks that a real modification flips the verdict.

## What this does not establish

Stated plainly because the smoke is easy to over-read:

* The portable smoke demonstrates **installability and mechanical execution**. Nothing else.
* It **does not validate a macrocycle ladder**. Three rungs, two picoseconds, on cyclo-(Gly-Gly-Gly).
  The acceptance it reports is 0.000 and that number is meaningless at this budget.
* **Generic macrocycles need a new pair-resolved acceptance pilot.** `macrocycle_pilot_8rung` ships
  with `ladder_status: unvalidated` and says so in its own comments.
* **RGD keeps the ten-rung working ladder** — three matched 2 ns/replica pilots against the 8-rung
  control, median worst-pair acceptance 0.140 → 0.240, pooled minimum 0.173 → 0.277. One molecule
  is the entire evidence base.
* **The 2 fs/4 fs and restart gates still prevent long production.** Neither was touched here.

## Loose end worth recording

`systems/cyclo_rgdfv/system.yaml` records `forcefield: ff19SB`. That file is the scientific system
definition for the implicit-solvent AIS work and its REMD reference, and the explicit-solvent
portable manifest parameterises the same molecule as a Sage/AM1-BCC ligand. Both are correct for
their own calculation, and `test_rgd_system_yaml_is_not_consumed_by_the_explicit_workflow` already
pins that they are not interchangeable — but a reader who finds them by grep will see two force
fields for one molecule. The note is now in `PORTABLE_REST2.md` and in the shipped manifest's
header; it would be better still if the two files cross-referenced each other by path.
