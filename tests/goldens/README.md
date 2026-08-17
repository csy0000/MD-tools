# Compatibility goldens

These fixtures freeze the behaviour that the multi-method migration must not disturb. They were
captured from the baseline **before** any structural change of PR 1, so a later PR that moves a
module, repackages a resource or edits a profile has to move a golden to do it — and moving a golden
is a reviewable diff with a scientific claim attached, not an invisible side effect.

## Regenerating

```console
$ python scripts/capture_goldens.py            # rewrite the fixtures
$ python scripts/capture_goldens.py --check    # compare only; non-zero exit on any difference
```

`tests/test_compat_goldens.py` consumes the committed files and **never** regenerates them. A test
run that rewrote its own expectations would pass by construction, so regeneration is only ever the
explicit command above.

Every input is a literal in `scripts/capture_goldens.py` or a resource shipped inside the package.
Nothing is read from a run directory, nothing is timestamped, and nothing depends on the working
tree, so two runs on two machines with the same package produce identical bytes.

## What is frozen, and why that and not bytes

| file | contract |
|---|---|
| `configuration_hashes.json` | canonical-JSON hash, all five projection hashes, selected profile and scientifically meaningful resolved values, for MD and REST2 on both declared routes |
| `profiles.json` | every packaged profile's schema version, route, method, `is_default` flag and hash, plus which profile the `default` alias selects for each route/method pair |
| `format_equivalence.json` | the same document written as YAML and as JSON canonicalises to identical bytes and identical hashes |
| `seed_derivation.json` | `master_seed + index in STAGE_ORDER`, with explicit per-stage overrides preserved |
| `bundle_contract.json` | bundle schema version, the logical-role to path map, the checksum and original-input locations, and the separated count fields |
| `runstate_contract.json` | run-state schema, restart/committed/quarantine locations, the continuity and extension path sets, and restart member naming |
| `rest2_defaults.json` | the shipped ladders, omega exclusion and proline classification — the settings whose change would be a scientific change |

Trajectories, checkpoints, serialized `System` XML, prepared bundles and run directories are
deliberately **not** frozen. Their bytes depend on the OpenMM build, the toolkit versions and the
machine, so a byte comparison would fail for reasons that have nothing to do with this repository's
contracts. Where an artifact is environment-dependent the fixture records the semantic projection or
an independently derived invariant instead — `bundle_contract.json` and `runstate_contract.json` are
the two places that applies, and each says so in its generator's docstring.

The bundle and restart contracts also have direct behavioural tests
(`tests/test_bundle_portability.py`, `tests/test_crash_recovery.py`, `tests/test_explicit_portable.py`).
The goldens complement those rather than duplicating them: the tests prove the behaviour works, the
goldens prove the shape did not silently change.
