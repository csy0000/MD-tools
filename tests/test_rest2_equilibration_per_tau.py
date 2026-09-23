"""`rest2.equilibration_per_tau`: the equilibration stages on every rung, under its own tau.

THE SETTING

    Off by default. When on, the tau = 0 chain stops at minimisation (implicit solvent) or after
    its NPT stages (explicit solvent, which fix the box every rung shares), and every rung -- tau = 0
    included -- then runs eq_nvt_posres, eq_nvt_posres_2 and eq_nvt_free under its own tau at
    fixed volume, then `equilibration_steps`, then the first exchange.

WHAT IS ASSERTED

    * Off leaves every generated script, `run.sh` and the `_protocol.py` helper byte-identical to
      what build-md wrote before the setting existed, and the scientific identity without the
      key; each `.in` and `resolved.config` gains exactly one line. A change to any of those is a
      change to every existing ladder directory, whose helpers are content-addressed.
    * Refused by name for a protocol with no ladder, and when there is nothing to run.
    * The per-rung stages ARE the fixed-volume stages a scaled run gets: the same stage dicts.
    * The driver runs them stage by stage across its rungs and stops between stages on an
      interruption, recording a phase that `--resume` refuses read-only.
    * A real implicit ladder leaves each rung at a state an independent reconstruction -- plain
      OpenMM, the documented seed formula and restraint -- reproduces bit for bit, distinct per
      rung; an explicit ladder keeps the box its NPT stages fixed on every rung.
PLATFORM_POLICY_EXEMPTION: the CPU ladders here compare the engine's per-tau end states with an
independent reconstruction bit for bit, which needs OPENMM_CPU_THREADS=1 on both sides. The CUDA
evidence, one rank per rung, is `test_rest2_equilibration_per_tau_cuda.py`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "md_tools"
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}
RUNGS = 4
STAGES = ("eq_nvt_posres", "eq_nvt_posres_2", "eq_nvt_free")

#: One worker for the module-scoped ladders below; see tests/test_reference_export_rest2.py.
pytestmark = [pytest.mark.xdist_group("rest2-per-tau")]


def _resolve(tmp_path, document):
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "per_tau.config"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return resolve_md_config(path)


def _build_md(root, document, odir="REST2-run1"):
    """Generate one run into a dataset root, and return the RUN directory.

    The dataset root is made first because a ladder's rungs are scaled from `build/built.xml` at
    build time now. Callers that need the shared `input/` or `min/` reach them through
    `root`, not through the returned run directory.
    """
    from .conftest import make_dataset_root

    make_dataset_root(root, solvent=str(document.get("solvent") or "implicit"))
    path = root / f"{odir}.config"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    from .conftest import make_states_for

    make_states_for(root, path)
    done = subprocess.run(CLI + ["build-md", "-odir", odir, "--config", str(path)], cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return root / odir


def _documented_seed(base, *parts):
    """`derive_seed`, written out from its definition rather than imported."""
    value = int(base)
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    return (value % (2 ** 31 - 1)) or 1


# --- resolution -----------------------------------------------------------------------------------

def test_it_is_off_by_default_and_then_plans_nothing(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages

    resolved = _resolve(tmp_path, {"protocol": "REST2"})
    assert resolved["rest2"]["equilibration_per_tau"] is False
    assert per_tau_equilibration_stages(resolved) == []


@pytest.mark.parametrize("document", [
    {"protocol": "cMD"},
    {"protocol": "umbrella", "umbrella": {"file": "umbrella.yaml"},
     "collective_variables": {"file": "cv.yaml", "interval_steps": 10}},
    {"protocol": "AIS", "ais_source": {"trajectory": "source.nc"}},
], ids=["cMD", "umbrella", "AIS"])
def test_a_protocol_without_a_ladder_is_refused_by_name(tmp_path, document):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError) as refusal:
        _resolve(tmp_path, dict(document, rest2={"equilibration_per_tau": True}))
    message = str(refusal.value)
    assert "rest2.equilibration_per_tau" in message and document["protocol"] in message, message


def test_a_ladder_with_nothing_to_run_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="all 0"):
        _resolve(tmp_path, {"protocol": "REST2", "rest2": {"equilibration_per_tau": True},
                            "stages": {"restrained_nvt_steps": 0, "restrained_npt_steps": 0,
                                       "unrestrained_npt_steps": 0}})


def test_the_rung_stages_are_the_stages_a_scaled_run_gets(tmp_path):
    """"The same equilibration" is the same stage dicts, not a second description of them."""
    from md_tools.build.md import per_tau_equilibration_stages, stage_plan

    stages = {"restrained_nvt_steps": 300, "restrained_npt_steps": 200,
              "unrestrained_npt_steps": 100}
    dynamics = {"restraint_kcal_per_mol_A2": 2.5}
    ladder = _resolve(tmp_path, {"protocol": "REST2", "solvent": "explicit", "stages": stages,
                                 "dynamics": dynamics, "rest2": {"equilibration_per_tau": True}})
    scaled = _resolve(tmp_path, {"protocol": "cMD", "solvent": "explicit", "stages": stages,
                                 "dynamics": dict(dynamics, tau=0.3)})
    expected = [{"name": stage["name"], "steps": stage["steps"],
                 "restraint_kcal_per_mol_A2": stage["restraint_kcal_per_mol_A2"]}
                for stage in stage_plan(scaled) if stage["name"].startswith("eq_")]
    assert [stage["name"] for stage in expected] == list(STAGES)
    assert per_tau_equilibration_stages(ladder) == expected
    assert [stage["restraint_kcal_per_mol_A2"] for stage in expected] == [2.5, 2.5, 0.0]


def test_a_stage_of_zero_steps_is_left_out(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages

    # REST2, not rREST2: rREST2 is archived (0.5.4), and the stage selection is the ladder's.
    resolved = _resolve(tmp_path, {"protocol": "REST2", "rest2": {"equilibration_per_tau": True},
                                   "stages": {"restrained_npt_steps": 0}})
    assert [stage["name"] for stage in per_tau_equilibration_stages(resolved)] == [
        "eq_nvt_posres", "eq_nvt_free"]


def test_the_stage_names_agree_between_the_resolver_and_the_runtime():
    from md_tools.build.md import PER_TAU_STAGE_NAMES as resolver
    from md_tools.remd.rung_equilibration import PER_TAU_STAGE_NAMES as runtime

    assert resolver == runtime == STAGES


# --- what build-md writes ---------------------------------------------------------------------------

def test_under_implicit_solvent_the_tau_zero_chain_is_minimisation_alone(tmp_path):
    """No equilibration stage belongs to the RUN: the rungs each equilibrate under their own tau.

    Implicit solvent has no box to fix first, so the tau = 0 chain is minimisation alone -- and
    minimisation is shared, at `<system>/min/`. So the run itself has no `eq/` at all, which is
    the assertion: an `eq/` here would mean a tau = 0 equilibration ran as well as the per-rung
    ones.
    """
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                               "rest2": {"equilibration_per_tau": True}})
    root = out.parent
    assert not (out / "eq").exists(), sorted(p.name for p in (out / "eq").iterdir())
    assert sorted(p.name for p in root.joinpath("input").iterdir()) == ["REST2.in", "min.in"]
    assert (root / "min" / "min.py").is_file()
    # The ladder starts from the shared minimised structure -- and that now lives in the GROUP
    # FILE rather than in run.sh. `-c` is a per-replica INPUT, and each line of a group file
    # carries its own; the executor call names only what describes the coordinated run.
    assert "-c ../min/min.xml" in (out / "remd_groupfile.1").read_text(encoding="utf-8")
    assert "per-tau equilibration" in (out / "build-md.log").read_text(encoding="utf-8")


def test_under_explicit_solvent_the_npt_stages_still_fix_the_box_first(tmp_path):
    """The NPT chain at tau = 0 decides the volume every rung shares, so it survives -- filed by
    position, as `eq_1/eq_2/eq_3`, with the ensemble in each stage's own header."""
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "explicit",
                               "rest2": {"equilibration_per_tau": True}})
    root = out.parent
    assert {"eq_1.in", "eq_2.in", "eq_3.in"} <= {p.name for p in (root / "input").iterdir()}
    assert {"eq_1.py", "eq_2.py", "eq_3.py"} <= {p.name for p in (out / "eq").iterdir()}
    assert "-c eq/eq_3.xml" in (out / "remd_groupfile.1").read_text(encoding="utf-8")


def test_every_generated_input_resolves_back_to_its_resolved_config(tmp_path):
    """The method input carries the ladder; the shared preparation inputs deliberately do not.

    `min.in` and `eq_<k>.in` serve every method on this system, so they carry no `protocol` and
    no `&remd` -- their `rest2` block is therefore the schema's defaults, not this run's ladder.
    Only `REST2.in` is checked for the ladder blocks, and the preparation inputs for the ones
    that genuinely are theirs.
    """
    from md_tools.build.md import resolve_md_config
    from md_tools.run.inputs import parse_run_input

    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                               "rest2": {"equilibration_per_tau": True}})
    shared = out.parent / "input"
    expected = resolve_md_config(out / "resolved.config")
    assert expected["rest2"]["equilibration_per_tau"] is True

    method = parse_run_input(shared / "REST2.in", run_config=out / "run.config")
    for block in ("rest2", "stages", "dynamics"):
        assert method.resolved[block] == expected[block], ("REST2.in", block)

    for path in sorted(shared.glob("*.in")):
        if path.name == "REST2.in":
            continue
        parsed = parse_run_input(path, run_config=out / "run.config")
        for block in ("stages", "dynamics"):
            assert parsed.resolved[block] == expected[block], (path.name, block)


#: What `build-md` generates for a default REST2 ladder, by sha256. Originally captured BEFORE
#: `rest2.equilibration_per_tau` existed, which is what makes it evidence: with the setting off,
#: every script, `run.sh` and the protocol helper must still be these bytes, and each `.in` and
#: `resolved.config` these bytes once the one added line is taken out again.
#:
#: THESE DIGESTS MOVE WHEN THE SCHEMA GAINS A FIELD, and that is expected rather than a
#: regression: a generated `.in` and `resolved.config` list every resolved key, so one new field
#: rewrites eleven of the files below. `stages.number_of_segments` did exactly that -- the six
#: `.in`/`resolved.config` entries per solvent changed and the `.py` files and `run.sh` did not,
#: which is itself the shape a schema-only change should have.
#:
#: HOW TO REFRESH THEM HONESTLY. Do not just paste the new hashes: that absorbs any unintended
#: drift sitting beside the intended change. Strip the NEW field's line out of the freshly
#: generated text and check the PINNED digest comes back. If it does, that line is the only
#: difference and the snapshot can be updated. If it does not, something else moved as well, and
#: that is the thing to look at rather than to overwrite.
#: REFRESHED FOR THE RUN LAYOUT, and the refresh is itself the evidence. Every path below moved
#: when `build/`, `min/` and `input/` became dataset-level and the run gained `eq/` and
#: `remd<n>/`, so the old keys could not survive. Rather than pasting new hashes -- which this
#: comment has always forbidden, because it absorbs unintended drift alongside the intended
#: change -- each old digest was checked against the file at its NEW path. The result, identically
#: for both solvents:
#:
#:   7 of 13 files are BIT-IDENTICAL under their new paths: every `.py` entry point
#:   (`min/min.py`, `eq/eq_{1,2,3}.py`, `REST2.py`), `run.config` and `resolved.config`. The move
#:   changed where they live and nothing about what they say.
#:
#:   6 changed, and each had to: the five `.in` files lost `protocol` and (for the preparation
#:   ones) the whole `&remd` block so that `min.in`/`eq_<k>.in` can be shared across METHODS, and
#:   gained the ensemble in their heading now that the filename can no longer carry it; `run.sh`
#:   gained `-odir` per stage and the `../build/` defaults.
#:
#: WHAT THE RUNG DIGESTS ARE NOT. `remd<n>/build_state<n>.xml` is identical between the explicit
#: and implicit tables because the fixture System is the same GBn2 one in both -- an md
#: configuration's `solvent` key does not change the built System. They pin that scaling is
#: deterministic, and they are NOT explicit-solvent evidence.
#: REFRESHED for 0.5.4 (REST2.py and run.sh only; checked by reverting each change in the generated
#: text and recovering the old digest): run.sh's header stops at the first flag, so
#: `./run.sh --cpu` no longer takes "--cpu" as the topology; REST2.py's usage shows the
#: group-file launch, since -s on the command line is refused.
#: REFRESHED for the rREST2 archive (0.5.4): every `resolved.config` and `input/REST2.in` lost the
#: `reservoir` section. Verified by appending `reservoir: {enabled: false, path: null,
#: refresh_interval_exchanges: 1, velocities: resample}` to each resolved.config, and
#: `reservoir_enabled = false, refresh_interval_exchanges = 1, reservoir_velocities = resample`
#: to the end of &remd, which recovers every previous digest exactly. Nothing else moved.
#:
#: REFRESHED AGAIN FOR 0.5.4 on the AIS branch, rebased onto the rREST2 archive: the same two
#: changes as below, re-verified against these digests -- removing `collective_variables.generate`
#: and `ais_source.generate` and re-inserting the four retired AIS lines reproduces every previous
#: digest, both solvents; nothing else moved.
#: REFRESHED AGAIN FOR 0.5.4, when the single-topology AIS keys were retired. The three
#: `resolved.config` digests per solvent moved, and nothing else: inserting exactly the four removed
#: lines (`tau_start: 0.5`, `tau_end: 0.0`, `work_measurement: work`, `verify_every_updates: 0`)
#: back into the freshly generated files at their schema positions reproduces each OLD pinned
#: digest, for both solvents. And once more in 0.5.4 for two ADDED fields,
#: `collective_variables.generate: null` and `ais_source.generate: false`: stripping exactly those
#: two lines from the fresh files reproduces the previous digests, for both solvents.
#: REFRESHED for the AIS lambda schedules: every `resolved.config` gained exactly
#: `lambda_schedule: linear` and `lambda_schedule_tau0: null` in its `ais` block. Generated from
#: dev 01e86a6 and from this branch into separate roots and diffed file by file: those two lines are
#: the whole difference, and the dev files reproduce the previous digests, for both solvents.
BEFORE = {
    "explicit": {
        "REST2-run1/REST2.py": "09af17815c1e151c6292e8fbeb8394fbabf8996c3b1d7d753ebc999ae1e93cce",
        "REST2-run1/eq/eq_1.py": "b5a332209934c06cbc1fe47f8cb780933bd672cf13de18f2d76214bb6cf89019",
        "REST2-run1/eq/eq_2.py": "a9a19c6c5e8f839a7a51e81a1ec554f89655bd04d581bc0c1babaa6aa07e6562",
        "REST2-run1/eq/eq_3.py": "6daed6d160528e34e730c67997fb015667e2a3fe5cf80677be6b3f1e68b3ccad",
        "REST2-run1/resolved.config":
            "41baeb581e97fae87a4f4ec40fac13bd54b08f8ed4d4d84c4226362b8911624d",
        # BYTE-IDENTICAL to the run root's, and the equality is the assertion: `eq/` holds
        # generated scripts, and a generated script reads the `resolved.config` strictly beside
        # itself, so the copy must be the same document rather than a second one.
        "REST2-run1/eq/resolved.config":
            "9373d2ae7c0cce005d01f3a7cba211d01205f5c97086340c03306dd3120d6ca0",
        # The SHARED minimisation's declaration: method-neutral, so it is the same file whichever
        # method generates it first. Its digest therefore differs from the run's by construction.
        "min/resolved.config":
            "9373d2ae7c0cce005d01f3a7cba211d01205f5c97086340c03306dd3120d6ca0",
        "REST2-run1/run.config":
            "0a421e80abb4cd6c47291af8b0304341f8dddd1828fbec077aa528f732c2e71c",
        "REST2-run1/run.sh": "583fa241e887760637a9f729d71cb8b85e40314b034c4e695ea952b92d1aff8f",
        "input/REST2.in": "b2cfa97173b178eb6243d7433b42014013e309e0e94f2e20d46848c64cda1fc8",
        "input/eq_1.in": "149c24d1d7536173034b74f8f8dfa035292b2710aebe1d62449fcc7d2a8cd4eb",
        "input/eq_2.in": "1e803fb37f931840ea037601df2255831815a0e4e2aea7676f35af5a3cf74cd3",
        "input/eq_3.in": "372e78ebdae42eac63ca85445c397739e73258f6f4353b05f20114503bc35586",
        "input/min.in": "23a1345e686f06ba7c5b440d4d75154a1a1c9b4637e27e5d54d7ab2ef25fc71b",
        "min/min.py": "c85b0c6bfd43627551e84f9fec6a0e76db4d16f089a640053d78fba522d9d601",
    },
    "implicit": {
        "REST2-run1/REST2.py": "09af17815c1e151c6292e8fbeb8394fbabf8996c3b1d7d753ebc999ae1e93cce",
        "REST2-run1/eq/eq_1.py": "b5a332209934c06cbc1fe47f8cb780933bd672cf13de18f2d76214bb6cf89019",
        "REST2-run1/eq/eq_2.py": "3054435667e24ebc079e4ecae1f1ddbc3de854035a6ee440f1a61e4f95c2cd35",
        "REST2-run1/eq/eq_3.py": "63f242cd9e3c1bf26ae98ff86e95792d76bbe4d94d06653bdaa6c1b140628f62",
        "REST2-run1/resolved.config":
            "1a185610b88b15c92884aed5a32f84eb08de5f208e09b8acce6fb677af70fd50",
        "REST2-run1/eq/resolved.config":
            "518dae1547bdd9691e8780b255943bc0685a0bd7c014ad6a0413a12dd1f2f23a",
        "min/resolved.config":
            "518dae1547bdd9691e8780b255943bc0685a0bd7c014ad6a0413a12dd1f2f23a",
        "REST2-run1/run.config":
            "0a421e80abb4cd6c47291af8b0304341f8dddd1828fbec077aa528f732c2e71c",
        "REST2-run1/run.sh": "583fa241e887760637a9f729d71cb8b85e40314b034c4e695ea952b92d1aff8f",
        "input/REST2.in": "2f34092f82e0031e7cf6a2ab2439661d4eb60b253d35d85336df22a05bd10525",
        "input/eq_1.in": "97f1e792f92394f75abb30a88161566cc617990a3ad32d075401afd421057e20",
        "input/eq_2.in": "42adbaf0eac18587f548d8b040d915f89b41078a94600e4d0972f9dd2d1756e8",
        "input/eq_3.in": "e4b9e3b368a7261715c33468a67e008257eee7c42227ff42f17990d18f1caecc",
        "input/min.in": "cc7b66db65f7edfb399d26856e5d1b74116c598a81e04fbac778bef95a817940",
        "min/min.py": "c85b0c6bfd43627551e84f9fec6a0e76db4d16f089a640053d78fba522d9d601",
    },
}
#: What this layout ADDED, pinned separately: these are new behaviour rather than a move, so they
#: have no "before" to be compared against and belong outside the table above. Every one of them
#: is genuinely solvent-independent: the group file names rungs rather than describing them, and
#: the two `run.config` files hold one number each.
#:
#: THE RUNG SYSTEMS are no longer generated at all: see the note after this table.
NEW_IN_THIS_LAYOUT = {
    # One group file per segment, naming one rung per line. New behaviour with the rungs.
    # REFRESHED: `-i` on every group line is `_protocol.py`, not `../input/REST2.in`.
    #
    # The executor's grouped mode imports a group line's `-i` as PYTHON and expects one `protocol`
    # object -- it is the module that describes the ladder. Naming the Amber-like input made every
    # grouped launch through `run.sh` die with "could not be loaded as a Python file", after the
    # whole equilibration chain had completed. `remd.generated._group_file_text` had always
    # written `_protocol.py`; the two group-file writers disagreed and the build-time one was the
    # wrong half.
    # REFRESHED for step 4, diffed against 02fe5b1's text: the four `-s` paths now name
    # `../build/REST2/system_state<n>.xml` in place of `remd<n>/build_state<n>.xml`, and the
    # header comment says so. Nothing else changed, identically for both solvents.
    "REST2-run1/remd_groupfile.1":
        "79260b058729e880f1cd971ff0af9552786f682114909ce3a5aa22a4917e0729",
    # THE MINIMISATION'S OWN SEED. `min/` is shared, so it cannot hold a run's seed -- and it does
    # not need to: minimisation draws no velocities, so no run's sampling descends from this
    # number. Every run on the system may carry a different one without disturbing it.
    "min/run.config":
        "7a4e8aa29726c2975dea4be7fc2162d1f24a91964ea1b4504336821657bfcde9",
    # The RUN's seed, beside the equilibration scripts. `md-run` layers `run.config` from `-odir`,
    # so a stage run into `eq/` needs one there or it resolves the seed to the schema default and
    # refuses against `eq/resolved.config`. Same bytes whichever solvent, since it holds one number.
    "REST2-run1/eq/run.config":
        "09847c0816635b3e559e754923710c189bb044551c0b9fae61af85120821f30b",
}
#: THE RUNG SYSTEMS ARE NOT GENERATED ANY MORE (step 4 of docs/amber-like-fix/REST2-scaler.md):
#: `build-md` no longer scales `remd<n>/build_state<n>.xml` into the run, and the group file names
#: the SAVED states under `build/REST2/`, which are an INPUT to generation and excluded above like
#: the rest of `build/`. Their bytes are pinned where they are made, in tests/test_rest2_scaler.py
#: and tests/test_ladder_on_saved_states.py. The per-solvent `RUNGS` digest table that stood here
#: also shadowed the integer `RUNGS` the GPU tests below read, and is gone with them.
#: The `_protocol.py` a default four-state ladder materialises at 2 fs, before this setting.
#: REFRESHED for convention v3: diffed against 6cc67c8's text, the only change is the docstring
#: line "omega left unscaled" becoming "amide omega, aromatic ring, double bond and improper
#: torsions left unscaled".
PROTOCOL_HELPER_BEFORE = "e764ac0f085c49108b6fef9c6daa8580b662078bacc6d93e0255a742cbe86041"


@pytest.mark.parametrize("solvent", ["explicit", "implicit"])
def test_off_leaves_every_generated_file_as_it_was(tmp_path, solvent):
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": solvent})
    root = out.parent
    generated = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    # `build/` is the INPUT to this generation rather than a product of it, and the two configs
    # are the test's own scaffolding.
    generated -= {name for name in generated
                  if name.startswith("build/") or name.endswith(".config")
                  and "/" not in name}
    generated -= {"REST2-run1/build-md.log", "REST2-run1/build_states.log"}
    assert generated == (set(BEFORE[solvent]) | set(NEW_IN_THIS_LAYOUT)), sorted(generated)

    for name, digest in {**BEFORE[solvent], **NEW_IN_THIS_LAYOUT}.items():
        text = (root / name).read_text(encoding="utf-8")
        # The `.in` files and `resolved.config` gain exactly one line when the setting exists but
        # is off. The preparation inputs are the exception: they carry no `&remd` at all now, so
        # there is no `equilibration_per_tau` line in them to take out.
        if name.endswith("resolved.config"):
            # 0.6.1 added the three selective-REST2 claim keys to `rest2:`. Off, they are null,
            # and they are the ONLY other new lines: stripping them must give 0.6.0's bytes.
            selectors = ("backbone_scaling_list:", "sidechain_scaling_list:",
                         "ligand_scaling_dict:")
            claimed = [line for line in text.splitlines() if line.strip().startswith(selectors)]
            assert len(claimed) == 3 and all(line.rstrip().endswith("null")
                                             for line in claimed), (name, claimed)
            text = "".join(line for line in text.splitlines(keepends=True)
                           if not line.strip().startswith(selectors))
        if name.endswith("resolved.config") or name in ("input/REST2.in",):
            added = [line for line in text.splitlines() if "equilibration_per_tau" in line]
            assert len(added) == 1 and "false" in added[0], (name, added)
            text = "".join(line for line in text.splitlines(keepends=True)
                           if "equilibration_per_tau" not in line)
        elif name.endswith(".in"):
            assert "equilibration_per_tau" not in text, (
                f"{name} is a SHARED preparation input and must carry no ladder setting")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, name


@pytest.mark.parametrize("solvent", ["explicit", "implicit"])
def test_off_leaves_the_protocol_helper_and_the_identity_as_they_were(tmp_path, solvent):
    from md_tools.remd.generated import ladder_from_resolved, protocol_file_text

    ladder = ladder_from_resolved(_resolve(tmp_path, {"protocol": "REST2", "solvent": solvent}),
                                  "REST2")
    assert "per_tau_equilibration" not in ladder
    ladder["dynamics"]["timestep_fs"] = 2.0     # what the runtime resolves `auto` to here
    text = protocol_file_text(ladder)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PROTOCOL_HELPER_BEFORE
    namespace = {}
    exec(text, namespace)
    assert "per_tau_equilibration" not in namespace["protocol"].describe()


def test_on_the_stages_reach_the_protocol_and_its_identity(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages
    from md_tools.remd.generated import ladder_from_resolved, protocol_file_text

    resolved = _resolve(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                                   "rest2": {"equilibration_per_tau": True}})
    ladder = ladder_from_resolved(resolved, "REST2")
    assert ladder["per_tau_equilibration"] == per_tau_equilibration_stages(resolved)
    ladder["dynamics"]["timestep_fs"] = 2.0
    namespace = {}
    exec(protocol_file_text(ladder), namespace)
    protocol = namespace["protocol"]
    assert protocol.per_tau_equilibration == ladder["per_tau_equilibration"]
    assert protocol.describe()["per_tau_equilibration"]["stages"] == ladder["per_tau_equilibration"]


@pytest.mark.parametrize("plan, words", [
    ([{"name": "eq_npt_free", "steps": 10}], "not one of"),
    ([{"name": "eq_nvt_posres", "steps": 0}], "positive integer"),
    ([{"name": "eq_nvt_free", "steps": 5}, {"name": "eq_nvt_posres", "steps": 5}], "order"),
    ([{"name": "eq_nvt_posres", "steps": 5, "restraint_kcal_per_mol_A2": -1.0}], ">= 0"),
])
def test_a_malformed_plan_is_refused_by_the_protocol(plan, words):
    from md_tools.remd.protocol import ProtocolError, REST2Protocol

    with pytest.raises(ProtocolError, match=words):
        REST2Protocol(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=2.0,
                      exchange_interval_ps=0.1, number_of_exchanges=2,
                      per_tau_equilibration=plan)


def test_the_handoff_files_are_owned_outputs_only_when_the_setting_is_on(tmp_path):
    from md_tools.run.preflight import _ladder_inventory

    common = dict(protocol="REST2", replicas=3, output=tmp_path / "REST2.out",
                  log=tmp_path / "REST2.log", trajectory=tmp_path / "REST2.nc",
                  restart=tmp_path / "restart.json",
                  checkpoint=tmp_path / "REST2_checkpoint.nc", groupfile=None)
    off = _ladder_inventory(**common).roles
    on = _ladder_inventory(**common, per_tau=True).roles
    assert set(on) - set(off) == {"per_tau_state_0", "per_tau_state_1", "per_tau_state_2",
                                  "per_tau_record"}
    assert on["per_tau_state_2"].name == "per_tau_state2.xml"


# --- the runtime module -------------------------------------------------------------------------------

def test_the_rung_equilibration_module_imports_nothing_from_md_tools():
    """It is copied into reference bundles, which run with md_tools absent."""
    tree = ast.parse((SRC / "remd" / "rung_equilibration.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in {"openmm", "numpy"} for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.module in {"engine", "core"}, node.module
            else:
                assert node.module.split(".")[0] in {"openmm", "numpy"}, node.module


def test_the_seeds_are_the_documented_derivation_and_all_distinct():
    from md_tools.md._stages import derive_seed as stage_chain_seed
    from md_tools.remd.rung_equilibration import derive_seed, stage_seed

    assert stage_chain_seed is derive_seed
    seeds = {(stage, index): stage_seed(7, stage, index) for stage in STAGES for index in range(8)}
    for (stage, index), seed in seeds.items():
        assert seed == _documented_seed(7, stage, f"state{index}")
    assert len(set(seeds.values())) == len(seeds)


class _Interruption:
    requested = False
    signal = None


class _Coordinator:
    rank, size, is_root = 0, 1, True

    @staticmethod
    def any_true(value):
        return bool(value)

    def barrier(self):
        pass


def _stub_driver(tmp_path, monkeypatch, *, interrupt_after=None):
    """A driver with its OpenMM work stubbed out: only the orchestration is under test here."""
    from md_tools.remd import rung_equilibration
    from md_tools.remd.driver import ReplicaRun

    calls = []
    interruption = _Interruption()

    def fake_stage(restrained, configuration, stage, **settings):
        calls.append((stage["name"], settings["state_index"]))
        if interrupt_after is not None and len(calls) >= interrupt_after:
            interruption.requested = True
        return configuration, {"stage": stage["name"], "seed": 1}, "<State/>"

    monkeypatch.setattr(rung_equilibration, "run_stage", fake_stage)
    monkeypatch.setattr(rung_equilibration, "restrained_clone", lambda system, *_: system)
    start = tmp_path / "min.xml"
    start.write_text("<State/>", encoding="utf-8")

    driver = ReplicaRun.__new__(ReplicaRun)
    driver.protocol = SimpleNamespace(
        per_tau_equilibration=[
            {"name": "eq_nvt_posres", "steps": 10, "restraint_kcal_per_mol_A2": 1.0},
            {"name": "eq_nvt_free", "steps": 10, "restraint_kcal_per_mol_A2": 0.0}],
        random_seed=7, temperature_k=300.0, friction_per_ps=1.0, timestep_fs=2.0,
        constraint_tolerance=1e-8, total_steps=100, n_states=2, tau=[0.0, 0.5])
    driver.files = SimpleNamespace(topology=str(ALA), trajectory=str(tmp_path / "REST2.nc"),
                                   coordinates=str(start))
    driver.owned = [0, 1]
    driver.coordinator = _Coordinator()
    driver.solute_indices = list(range(22))
    driver._interruption = interruption
    driver._platform, driver._properties, driver._fault_crossings = None, {}, {}
    driver.trajectories = driver.solute_trajectories = driver.reporter = None
    driver.engine = SimpleNamespace(set_configuration=lambda index, configuration: None)
    driver._gather_configurations = lambda state: [SimpleNamespace(box=None)] * 2
    return driver, calls, interruption


def test_every_rung_runs_a_stage_before_any_rung_runs_the_next(tmp_path, monkeypatch):
    driver, calls, _interruption = _stub_driver(tmp_path, monkeypatch)
    state = {"configurations": ["walker0", "walker1"], "state_to_walker": [0, 1]}
    assert driver._equilibrate_per_tau(state, systems=["rung0", "rung1"]) is False
    assert calls == [("eq_nvt_posres", 0), ("eq_nvt_posres", 1),
                     ("eq_nvt_free", 0), ("eq_nvt_free", 1)]
    record = json.loads((tmp_path / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert [entry["file"] for entry in record["states"]] == ["per_tau_state0.xml",
                                                             "per_tau_state1.xml"]
    assert [len(entry["stages"]) for entry in record["states"]] == [2, 2]


def test_an_interruption_stops_between_stages_and_names_the_phase(tmp_path, monkeypatch):
    """Every rung finishes the stage it is in; none starts the next. The record says where."""
    from md_tools.remd.storage import read_run_state

    driver, calls, _interruption = _stub_driver(tmp_path, monkeypatch, interrupt_after=2)
    state = {"configurations": ["walker0", "walker1"], "state_to_walker": [0, 1]}
    assert driver._equilibrate_per_tau(state, systems=["rung0", "rung1"]) is True
    assert calls == [("eq_nvt_posres", 0), ("eq_nvt_posres", 1)]
    assert state["per_tau_stage"] == "eq_nvt_free"
    assert not (tmp_path / "per_tau_equilibration.json").exists()

    result = driver._record_per_tau_interruption(state, identity={})
    assert result["run_status"] == "interrupted"
    recorded = read_run_state(tmp_path / "REST2.nc")
    assert recorded["status"] == "interrupted"
    assert recorded["phase"] == "per_tau_equilibration"
    assert recorded["stage"] == "eq_nvt_free"
    assert "--overwrite" in recorded["note"]


# --- real ladders, CPU ----------------------------------------------------------------------------------

def _build_top(root, config_text):
    """The REAL build-top, into the dataset's `build/`.

    Into `build/` because that is where a run reads its System from, and because
    `make_dataset_root` -- which `_build_md` calls -- now declines to overwrite an existing built
    System. The two therefore agree instead of racing: whichever runs first provides the System,
    and these tests want this one, built by tleap under the stated solvent model.
    """
    (root / "sys.config").write_text(config_text, encoding="utf-8")
    (root / "build").mkdir(exist_ok=True)
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


#: Stage name -> (shared input, `-odir` relative to the run, the state it leaves behind).
#:
#: A stage is GENERATED under its physics name (`eq_nvt_posres`) and FILED by position (`eq_1`),
#: and the two are different things: the first decides what runs, the second where it lands.
#: Minimisation is shared at `<system>/min/`, equilibration is per run in `<run>/eq/`.
_LAYOUT = {
    # The third element is the restart the stage LEAVES, and it is FILED BY POSITION -- `eq_1.xml`
    # for the first equilibration stage whatever that stage is called. The stage name decides the
    # physics; the filing key decides the filename, and `stage_plan` stamps it so the layout and
    # the runtime cannot disagree.
    #
    # They did disagree: the runtime named outputs from the stage, so `run.sh` chained
    # `-c eq/eq_1.xml` against a stage that had written `eq/eq_nvt_posres.xml` and no chain could
    # complete through `run.sh` at all.
    "min": ("../input/min.in", "../min", "../min/min.xml"),
    "eq_nvt_posres": ("../input/eq_1.in", "eq", "eq/eq_1.xml"),
    "eq_nvt_posres_2": ("../input/eq_2.in", "eq", "eq/eq_2.xml"),
    "eq_npt_posres": ("../input/eq_2.in", "eq", "eq/eq_2.xml"),
    "eq_nvt_free": ("../input/eq_3.in", "eq", "eq/eq_3.xml"),
    "eq_npt_free": ("../input/eq_3.in", "eq", "eq/eq_3.xml"),
}


def _stage(run, name, parent=None):
    """Run one preparation stage the way `run.sh` does, and return the state it wrote.

    NO `-r`, `-chk`, `-o` or `-log`. md-run names all four inside `-odir`, and a value that IS
    given is taken verbatim against the WORKING directory -- so spelling them here would put the
    minimisation's restart beside the run root instead of in `min/`, which is the defect the
    per-stage `-odir` exists to remove.
    """
    source, odir, produced = _LAYOUT[name]
    argv = CLI + ["md-run", "-i", source, "-p", "../build/built.pdb", "-s", "../build/built.xml",
                  "-odir", odir, "--cpu"]
    if parent:
        argv += ["-c", parent]
    done = subprocess.run(argv, cwd=run, capture_output=True, text=True, timeout=1800,
                          env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return produced


def _ladder_argv(run, start, *extra):
    """The ladder, driven as `run.sh` drives it: through the group file `build-md` wrote.

    NO `-s` AND NO `-c` (0.5.4). A ladder reads both only from its group file, one line per
    state: `-s` is that state's saved scaled System under `build/REST2/`, and `-c` the state the
    ladder continues from. An earlier revision required `-s` here because these tests drove
    `md-run` with no group file and the ladder scaled from the built System; that path is gone.
    `start` is what the calling test prepared, and every line must name it -- otherwise the
    ladder would begin from a file this test never wrote.
    """
    group = Path(run) / "remd_groupfile.1"
    lines = [line for line in group.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert len(lines) == RUNGS and all(f" -c {start} " in f" {line} " for line in lines), lines
    return ["mpirun", "-n", str(RUNGS), *CLI, "md-run", "-ng", str(RUNGS),
            "-i", "../input/REST2.in", "-p", "../build/built.pdb",
            "--groupfile", group.name,
            "-x", "REST2.nc", "-r", "restart.json", "-o", "REST2.out", "-log", "REST2.log",
            "--cpu", *extra]


def _needs_mpirun():
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")


IMPLICIT = {
    "protocol": "REST2", "solvent": "implicit",
    "dynamics": {"timestep_fs": 2.0, "seed": 11},
    "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 100,
               "restrained_npt_steps": 0, "unrestrained_npt_steps": 100, "production_steps": 0},
    "reporting": {"crd_printout_solute": 50, "info_printout": 50, "checkpoint_printout": 100},
    "rest2": {"number_of_replicas": RUNGS, "tau_max": 0.5, "exchange_interval_steps": 50,
              "number_of_exchanges": 4, "equilibration_steps": 50,
              "equilibration_per_tau": True},
}


@pytest.fixture(scope="module")
def implicit_ladder(tmp_path_factory):
    _needs_mpirun()
    root = tmp_path_factory.mktemp("per-tau-implicit")
    _build_top(root, "solvent:\n  model: GBn2\n")
    run = _build_md(root, IMPLICIT, odir="run")
    start = _stage(run, "min")
    done = subprocess.run(_ladder_argv(run, start), cwd=run, capture_output=True, text=True,
                          timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root, run, done.stdout


def _positions(path):
    from openmm import XmlSerializer, unit

    state = XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
    return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.slow
def test_every_rung_ends_its_equilibration_at_its_own_state(implicit_ladder):
    _root, run, _stdout = implicit_ladder
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert [stage["name"] for stage in record["stages"]] == ["eq_nvt_posres", "eq_nvt_free"]
    assert [entry["state_index"] for entry in record["states"]] == list(range(RUNGS))
    seeds = [stage["seed"] for entry in record["states"] for stage in entry["stages"]]
    assert len(set(seeds)) == len(seeds) == 2 * RUNGS

    start = _positions(run.parent / "min" / "min.xml")
    ends = []
    for entry in record["states"]:
        assert _sha256(run / entry["file"]) == entry["sha256"]
        ends.append(_positions(run / entry["file"]))
    for index, end in enumerate(ends):
        assert not np.array_equal(end, start), f"rung {index} never left the starting state"
        for other in ends[index + 1:]:
            assert not np.array_equal(end, other), "two rungs ended in the same configuration"

    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["per_tau_equilibration"]["states"] == record["states"]
    assert manifest["per_tau_equilibration"]["record_sha256"] == _sha256(
        run / "per_tau_equilibration.json")


@pytest.mark.slow
def test_a_rungs_end_state_is_what_the_documented_procedure_gives(implicit_ladder):
    """Rebuilt with plain OpenMM: the rung's scaling, the documented restraint and seed formula.

    Not the module the driver calls -- that would compare the code with itself. Only the rung's
    scaled System comes from md_tools, because the scaling is not what is under test here.
    """
    from openmm import (Context, CustomExternalForce, LangevinMiddleIntegrator, Platform,
                        XmlSerializer, unit)
    from openmm.app import PDBFile

    from md_tools.rest2 import build_scaled_system

    root, run, _stdout = implicit_ladder
    solute = yaml.safe_load((run / "solute.yaml").read_text(encoding="utf-8"))
    atoms = list(range(int(solute["n_solute_atoms"])))
    excluded = [tuple(bond) for bond in solute["rest2"]["unscaled_central_bonds"]]
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    rung = 2
    tau = record["states"][rung]["tau"]
    system = build_scaled_system(
        XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8")),
        atoms, tau, excluded_bonds=excluded)

    force = CustomExternalForce("0.5*restraint_k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("restraint_k", 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    reference = PDBFile(str(root / "build" / "built.pdb")).positions.value_in_unit(unit.nanometer)
    for index in atoms:
        force.addParticle(index, list(reference[index]))
    system.addForce(force)

    start = XmlSerializer.deserialize((run.parent / "min" / "min.xml").read_text(encoding="utf-8"))
    positions = start.getPositions(asNumpy=True)
    velocities = start.getVelocities(asNumpy=True)
    for name, strength in (("eq_nvt_posres", 1.0), ("eq_nvt_free", 0.0)):
        integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                              2.0 * unit.femtosecond)
        integrator.setConstraintTolerance(1e-8)
        integrator.setRandomNumberSeed(_documented_seed(11, name, f"state{rung}"))
        context = Context(system, integrator, Platform.getPlatformByName("CPU"),
                          {"Threads": "1"})
        context.setPositions(positions)
        context.setVelocities(velocities)
        context.setParameter("restraint_k", strength * 418.4)
        integrator.step(100)
        state = context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        del context, integrator

    engine = _positions(run / f"per_tau_state{rung}.xml")
    assert np.array_equal(np.asarray(positions.value_in_unit(unit.nanometer)), engine), (
        "the engine's per-tau end state is not what the documented procedure produces; largest "
        f"difference {np.abs(np.asarray(positions.value_in_unit(unit.nanometer)) - engine).max()}"
        " nm")


@pytest.mark.slow
def test_the_ladder_says_what_it_did(implicit_ladder):
    from md_tools.build.record import read_record

    _root, run, stdout = implicit_ladder
    assert "per_tau_equilibration=[" in (run / "_protocol.py").read_text(encoding="utf-8")
    record = read_record(run / "REST2.log")
    assert [s["name"] for s in record["ladder"]["per_tau_equilibration"]] == [
        "eq_nvt_posres", "eq_nvt_free"]
    readable = stdout + "".join(p.read_text(encoding="utf-8", errors="replace")
                                for p in run.glob("REST2*.out"))
    assert "# per-tau equilibration" in readable


def _tree(path):
    return {p.relative_to(path).as_posix(): _sha256(p)
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


@pytest.mark.slow
def test_resume_after_an_interruption_during_it_is_refused_and_writes_nothing(implicit_ladder,
                                                                              tmp_path):
    """No exchange step was taken, so there is no checkpoint: say so, and touch nothing."""
    root, _run, _stdout = implicit_ladder
    copy = tmp_path / "root"
    shutil.copytree(root, copy)
    run = copy / "run"
    state_path = run / "REST2.runstate.json"
    document = json.loads(state_path.read_text(encoding="utf-8"))
    document.update(status="interrupted", phase="per_tau_equilibration", stage="eq_nvt_free")
    state_path.write_text(json.dumps(document), encoding="utf-8")
    before = _tree(copy)

    done = subprocess.run(_ladder_argv(run, "../min/min.xml", "--resume"), cwd=run,
                          capture_output=True, text=True, timeout=1800, env=ONE_THREAD)
    assert done.returncode != 0, done.stdout[-2000:]
    assert "per-tau equilibration" in done.stderr and "--overwrite" in done.stderr, done.stderr
    assert _tree(copy) == before, "a refused continuation changed the directory"


@pytest.mark.slow
def test_a_failure_during_it_stops_the_whole_ladder(implicit_ladder, tmp_path):
    """A rank that fails in per-tau equilibration aborts every rank; nothing claims completion."""
    root, _run, _stdout = implicit_ladder
    (tmp_path / "build").mkdir(exist_ok=True)
    for name in ("build/built.xml", "build/built.pdb", "build/built.log"):
        shutil.copy2(root / name, tmp_path / name)
    run = _build_md(tmp_path, IMPLICIT, odir="run")
    start = _stage(run, "min")
    done = subprocess.run(
        _ladder_argv(run, start), cwd=run, capture_output=True, text=True, timeout=1800,
        env={**ONE_THREAD, "MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "2",
             "MD_TOOLS_FAIL_LADDER_AT": "per-tau-equilibration"})
    assert done.returncode != 0
    assert "per-tau-equilibration" in done.stdout + done.stderr
    assert not (run / "restart.json").exists()
    assert not (run / "per_tau_equilibration.json").exists()


@pytest.fixture(scope="module")
def explicit_ladder(tmp_path_factory):
    _needs_mpirun()
    root = tmp_path_factory.mktemp("per-tau-explicit")
    _build_top(root, "solvent:\n  model: TIP3P\n")
    run = _build_md(root, {
        "protocol": "REST2", "solvent": "explicit",
        "dynamics": {"timestep_fs": 2.0, "seed": 5},
        "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 0},
        "reporting": {"crd_printout_solute": 50, "info_printout": 50,
                      "checkpoint_printout": 100},
        "rest2": {"number_of_replicas": RUNGS, "tau_max": 0.5, "exchange_interval_steps": 50,
                  "number_of_exchanges": 2, "equilibration_per_tau": True},
    }, odir="run")
    parent = None
    for name in ("min", "eq_nvt_posres", "eq_npt_posres", "eq_npt_free"):
        # The parent is the LAYOUT path of the state the previous stage wrote, not `<name>.xml`
        # at the run root: minimisation leaves `../min/min.xml` and equilibration `eq/eq_<k>.xml`.
        parent = _stage(run, name, parent)
    done = subprocess.run(_ladder_argv(run, parent), cwd=run, capture_output=True,
                          text=True, timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root, run


@pytest.mark.slow
def test_explicit_rungs_share_the_box_the_npt_stages_fixed(explicit_ladder):
    from openmm import XmlSerializer, unit

    _root, run = explicit_ladder

    def box(path):
        state = XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
        return np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))

    fixed = box(run / "eq" / "eq_3.xml")
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert record["box_shared"] is True
    assert [s["name"] for s in record["stages"]] == list(STAGES)
    for entry in record["states"]:
        assert np.array_equal(box(run / entry["file"]), fixed), entry["file"]

    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    # The rungs the ladder propagated carry neither the restraint nor a barostat.
    forces = json.dumps(manifest["hamiltonian"])
    assert "CustomExternalForce" not in forces and "Barostat" not in forces, forces

