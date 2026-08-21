# Claude Code instruction: close the remaining cMD acceptance gaps

Date: 2026-08-21  
Repository: `csy0000/MD-templates`  
Branch: `dev`  
Expected starting commit: `1bc29cdd7ccc40a5d64bc3a353744afc664989c1`, or a direct descendant containing this instruction

## Outcome and completion rule

Close the remaining issues found during review of
`claudecode-instructions/20260821_cmd-review-fixes-and-opc-default.md`, without reopening the
architecture or undoing the fixes that already pass.

This instruction supersedes only the old requirement that OPC be used for every explicit-solvent
route. The user has now approved force-field-dependent water defaults:

| solute parameterization | active default water |
|---|---|
| ff19SB peptide/protein | `amber19/opc.xml` with modeller model `opc` |
| Sage/OpenFF 2.2 ligand, including Sage-parameterized RGDfV | plain TIP3P with modeller model `tip3p` |
| ff19SB + Sage protein-ligand complex | OPC by default, explicitly documented as a mixed-force-field compatibility choice |

TIP3P-FB is not the Sage default. It remains only where an old named compatibility profile or an
explicit user override truthfully requests it. Explicit user water overrides remain supported and
must be recorded in provenance.

Return `PASS` only when every item below is implemented, tested, pushed, and verified on the exact
final remote `dev` SHA. If a blocking gate remains, return `BLOCKED` and name it. Do not create a
branch or pull request, do not merge into `main`, and do not revive PR #3.

## Establish the baseline

Before editing:

1. fetch `dev` and record its exact remote SHA and clean working-tree status;
2. read `CLAUDE.md`, both 2026-08-21 cMD instructions, the two associated journals, and the complete
   current implementations and tests for `stage`, `cmd_segments`, `runstate`, profiles, profile
   resolution, system preparation, persistence, and CI;
3. inspect the diff from `1bc29cd` and preserve unrelated user work;
4. run the existing focused cMD tests and record the baseline result.

## 1. Preserve and document the approved water-default policy

Do not change the current scientifically appropriate peptide/ligand split merely to satisfy the
superseded OPC-everywhere wording.

Required behavior:

- `explicit-md-peptide-v2` and `explicit-rest2-peptide-v2`: ff19SB + OPC;
- `explicit-md-ligand-v2` and `explicit-rest2-ligand-v2`: Sage/OpenFF 2.2 + plain TIP3P;
- protein-ligand complex default: ff19SB + Sage + OPC, with clear provenance and documentation that
  no single water model is the native fitting partner of both solute force fields;
- derive the default from the resolved solute force-field family wherever practical, rather than
  trusting only a broad input label such as `peptide` or `ligand`;
- reject or clearly diagnose an ambiguous custom mixed-force-field configuration rather than
  silently guessing when the normal protein-ligand rule cannot be applied;
- keep v1 profile IDs name-resolvable and nondefault; preserve the water model they historically
  represented;
- keep implicit profiles free of explicit water settings;
- use water-model-compatible ion parameters and record their source in the resolved provenance.

Update tests, README/configuration documentation, and journal wording to state that this is a
user-approved supersession of the earlier OPC-everywhere requirement. Do not rewrite the older
instruction; cross-link it as superseded on this one decision. Avoid unsupported claims: cite the
Sage/OpenFF primary publication or official OpenFF documentation for the TIP3P pairing.

Blocking tests must prove the four v2 defaults above, the complex rule, explicit override behavior,
v1 compatibility, implicit absence of water, and resolved water/ion provenance.

## 2. Complete the exact cMD continuity identity

The current contract hashes `system.xml` and the topology, but it records the predecessor State only
by pathname and carries only a partial parsed manifest identity. Complete the versioned continuity
contract with:

- SHA-256 of the exact first-segment predecessor State bytes, plus its declared path and producer;
- SHA-256 of the exact `system_manifest.json` bytes and its recorded configuration hash/identity;
- SHA-256 of the exact `forcefield.json` bytes, in addition to the human-readable force-field
  projection;
- the existing exact System/topology hashes, particle/constraint counts, periodicity, ordered atom
  identity, integrator, ensemble, barostat/pressure, restraint state/convention, segment length,
  reporting cadence, and output atom ordering;
- one deterministic hash over the canonical complete contract.

The first-segment predecessor is immutable provenance for how the cMD chain began. Once a generation
is committed, continuation still restores that committed generation; it must never switch back to
the predecessor State. If the on-disk cMD schema meaning changes, bump its version and refuse the
unaccepted older layout with regeneration guidance rather than silently reinterpreting it.

Add independent refusal-before-mutation tests for:

1. one changed byte in `system.xml`;
2. topology atom permutation with equal counts;
3. changed `system_manifest.json` identity/configuration hash;
4. changed `forcefield.json` or water-model provenance;
5. changed predecessor State bytes/path/producer for a fresh run;
6. equal-length selected-atom permutation;
7. temperature, timestep, ensemble, barostat/pressure, restraint convention, and every reporting
   cadence;
8. segment length changing while execution segment count remains extension-only and leaves the
   continuity hash unchanged.

Every refusal test must assert all existing output streams remain byte-for-byte unchanged.

## 3. Verify restart step/time before altering them

The current continuation path assigns `sim.currentStep` from `committed.json` before calling
`verify_restart_matches_commit()`. This makes a wrong checkpoint step appear correct and defeats
the check.

Required behavior:

- checkpoint path: load the checkpoint, capture and verify its actual loaded step and Context time
  against the commit, and only then proceed; do not overwrite either value to make verification
  pass;
- serialized-State fallback: document that an OpenMM State does not carry `currentStep`; verify the
  State's actual Context time against the commit first, then restore `currentStep` from the atomic
  commit and record that provenance explicitly;
- if the State time is inconsistent, refuse before opening or truncating output;
- announce and record State fallback as non-bitwise stochastic continuation;
- preserve checkpoint preference.

Add tests using a loadable but wrong checkpoint and a loadable State with wrong time. Both must be
refused before output mutation. Retain the corrupted-checkpoint/valid-State success test and assert
the exact starting and ending step/time provenance.

## 4. Make the five crash-boundary tests deterministic

The two existing time-based process kills are useful smoke tests, but they do not prove which
boundary was interrupted. Add narrowly scoped fault-injection hooks or an equivalent deterministic
test mechanism for exactly:

1. before reporters close;
2. after reporters close but before restart save;
3. after the checkpoint member is written but before the State member;
4. after both restart members but before atomic commit;
5. immediately after atomic commit but before `run_state.json` cache/status reconciliation.

The hooks must be inert in normal production and must not change scientific configuration or public
CLI behavior. For each boundary, use a real subprocess where practical and prove:

- the last committed physics remains authoritative;
- uncommitted tails or restart members are ignored/recovered safely;
- retry creates exactly one next generation and invocation;
- steps, time, frames, log rows, and invocation history remain monotonic without duplication;
- interruption after the atomic commit retains the new generation and reconstructs the stale cache
  from `committed.json`.

Keep the exclusive run-directory lock and the existing nondeterministic kill smoke test as additional
coverage, not as substitutes for these five tests.

## 5. Validation and exact-final-SHA CI evidence

After the corrections:

1. run the focused water-default, continuity, restart, DCD, output-integrity, and crash tests;
2. run all cMD/persistence/implicit/REST2/profile/public-generator tests;
3. run the complete non-slow suite and the complete suite including slow tests, recording exact
   pass/fail/skip/deselect counts and durations;
4. run `scripts/ci/fast_checks.sh` and `scripts/ci/integration_cpu.sh`;
5. build wheel and sdist, inspect packaged resources, and run installed-wheel commands outside the
   checkout without source-tree `PYTHONPATH`;
6. run bundle relocation/offline validation and scan tracked files for generated trajectories,
   checkpoints, States, environments, secrets, or large logs;
7. rerun the explicit ff19SB/OPC and implicit ff19SB/GBn2/mbondi3 alanine cMD validations from
   freshly generated deterministic bundles under the final code: two committed 500 ps segments,
   exactly 1 ns, expected frame/log counts, monotonic boundaries, checkpoint continuation, and no
   NaN/Inf; record current hashes and GPU/version provenance without committing simulation output;
8. push the final code and journal, locate both Actions workflows for that exact remote SHA, wait for
   terminal success, and record workflow names, run IDs, URLs, conclusions, and final SHA.

The previous journal's #31 URLs apply to `b1ab424`, not to later code. They cannot satisfy the
final-SHA gate. If Actions cannot be verified, return `BLOCKED: remote CI unverified`.

## Journal and final report

Add `docs/journal/2026-08-21_remaining-cmd-acceptance-gaps.md` containing:

- starting and final SHAs;
- the approved force-field-dependent water policy and its scientific sources;
- a requirement -> implementation -> test -> evidence matrix for every item above;
- schema compatibility/refusal behavior;
- exact commands and complete results;
- fresh 1 ns validation results/hashes;
- exact-final-SHA Actions run URLs and conclusions;
- limitations, without relabelling a failed or unrun gate as passed.

Before returning, inspect the final diff and explicitly confirm:

- [ ] ff19SB defaults to OPC; Sage defaults to plain TIP3P; complex behavior and overrides are documented and tested;
- [ ] predecessor, manifest/configuration, force-field file, System, topology, and ordered selection identities are bound by the cMD continuity hash;
- [ ] checkpoint step/time is verified before modification and State fallback semantics are explicit and tested;
- [ ] all five deterministic crash boundaries recover correctly;
- [ ] focused, full, packaging, integration, relocation, REST2, two 1 ns validations, artifact scan, and exact-final-SHA CI gates pass;
- [ ] documentation, journal, goldens, and compatibility guidance match the final code;
- [ ] no unrelated work or generated simulation data is committed.

The final report must begin with exactly one of:

- `PASS — all remaining cMD acceptance gaps are closed.`
- `BLOCKED — PASS is not claimed.`
