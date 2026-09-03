"""Whether a cMD stage that CLAIMS to be complete may be believed.

WHY THIS IS A MODULE

    "Completed" is a field in a file. The previous check read that field, compared the stage
    dictionary the log carried against the one this invocation resolved, and asked whether the
    final restart and (for dynamics) the trajectory EXIST. Everything else it appeared to check,
    it did not:

      the recorded `fingerprint` was read and tested for presence, and then never recomputed or
      compared -- so a log whose fingerprint belonged to another System passed, as long as its
      copy of the stage dictionary matched;

      `exists()` is not `is the file we wrote`. A truncated DCD, a state CSV cut in half by a
      full disk, a checkpoint binary edited since it was committed and a restart from a different
      molecule all exist;

      the checkpoint was not consulted at all, so a stage could be "complete" with a committed
      generation that stopped a thousand steps short;

      the state CSV was not even in the record, so no later check could have looked at it.

    A stage that is wrongly skipped is worse than one wrongly rerun: the NEXT stage continues from
    its restart, and the error propagates through the chain wearing the right file names.

    Everything here is read-only. A verifier that repairs what it finds cannot be run twice with
    the same answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CompletionProblem(str):
    """One reason a completion claim was not believed. A string with a name."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_records(role: str, path: Path) -> int | None:
    """How many records this stream holds, by the SAME reader the commit counted with.

    Delegating to `_count_stream` rather than reimplementing it is the point: a verifier that
    counts differently from the committer disagrees with a healthy run, and the disagreement
    looks like corruption.
    """
    from .stage import _count_stream

    return _count_stream(role, Path(path))


def verify_completed_stage(
    previous: dict[str, Any],
    *,
    stage: dict[str, Any],
    name: str,
    restart: Path,
    trajectory: Path,
    log_path: Path,
    fingerprint: str,
    particles: int | None = None,
    checkpoints: Path | None = None,
    inventory=None,
    streams: dict[str, Path] | None = None,
) -> list[str]:
    """Every reason this stage may not be skipped. Empty list means it is genuinely done.

    `fingerprint` is the one THIS invocation computed, from the resolved scientific configuration
    and the current topology and System digests. Passing it in rather than recomputing it here
    keeps one implementation of the fingerprint and makes the comparison the point of this
    function rather than a side effect of it.
    """
    from ..openmm.checkpoint import POINTER_NAME, CheckpointError, read_committed

    problems: list[str] = []
    recorded_stage = previous.get("stage")
    recorded_print = previous.get("fingerprint")

    if recorded_print is None or not isinstance(recorded_stage, dict):
        return ["it carries no configuration fingerprint, so nothing about it can be matched to "
                "this stage"]

    # -- 1. the fingerprint, RECOMPUTED and compared -------------------------------------------
    #
    # The check this replaces read this field and asserted it was not None. A recorded
    # fingerprint that is never compared is a receipt nobody reads: a `resolved.config` edited
    # since, a rebuilt `built.xml`, a topology from another structure all leave it in place and
    # wrong.
    if str(recorded_print) != str(fingerprint):
        problems.append(
            f"its fingerprint does not match this configuration "
            f"(recorded {str(recorded_print)[:16]}..., this run computes {fingerprint[:16]}...). "
            f"The resolved configuration, the topology or the System has changed since it ran")

    # Kept for the diagnostic: the fingerprint says THAT something differs, the field comparison
    # says WHICH, and a person reading the refusal needs the second.
    from .stage import EXTENDABLE_FIELDS, NON_SCIENTIFIC_STAGE_KEYS

    wanted = {k: v for k, v in sorted(stage.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    theirs = {k: v for k, v in sorted(recorded_stage.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    differing = sorted(k for k in set(wanted) | set(theirs) if wanted.get(k) != theirs.get(k))
    if differing:
        problems.append(f"it records a different configuration for this stage "
                        f"({', '.join(differing)} differ)")

    # -- 2. the recorded output manifest -------------------------------------------------------
    outputs = previous.get("outputs")
    if not isinstance(outputs, dict) or "final_state" not in outputs:
        # A log from a build that did not record an output manifest cannot be verified by this
        # one. Saying "verified" about it would be a claim nothing supports.
        problems.append(
            "it records no verifiable output manifest (an older log format). Nothing about the "
            "outputs it claims can be checked, so it cannot be accepted as proof of completion")
        return problems

    steps = int(stage.get("steps") or 0)
    dynamics = steps > 0
    expected_roles = {"final_state"}
    if dynamics and int(stage.get("trajectory_interval_steps") or 0) > 0:
        expected_roles.add("trajectory")
    if dynamics and int(stage.get("state_interval_steps") or 0) > 0:
        expected_roles.add("state_csv")
    if dynamics and int(stage.get("phase_space_interval_steps") or 0) > 0:
        expected_roles.add("phase_space")
    if dynamics and int((stage.get("collective_variables") or {}).get("interval_steps") or 0) > 0:
        expected_roles.add("collective_variables")

    missing_roles = sorted(expected_roles - set(outputs))
    if missing_roles:
        problems.append(
            f"this configuration produces {', '.join(missing_roles)}, which the record does not "
            f"claim. It was written by a run of a different shape")

    # -- 3. every recorded output, by digest and size ------------------------------------------
    #
    # "checkpoint" and "checkpoint_pointer" are recorded by basename (`file_facts` with no
    # `relative_to`), because they live under `<stem>.checkpoints/`, not beside the log --
    # joining them to `log_path.parent` would check the wrong file. Item 5 verifies them
    # correctly, through `read_committed`, which already checks the binary against its sidecar's
    # digest; re-deriving a second, wrong path here would be a second, worse check of the same
    # fact.
    directory = Path(log_path).parent
    for role, facts in sorted(outputs.items()):
        if role in ("checkpoint", "checkpoint_pointer"):
            continue
        if not isinstance(facts, dict) or "sha256" not in facts:
            problems.append(f"the record for {role} carries no digest, so it cannot be verified")
            continue
        path = Path(facts.get("path") or "")
        if not path.is_absolute():
            path = directory / path
        if not path.exists():
            problems.append(f"the {role} it claims is missing: {path}")
            continue
        size = path.stat().st_size
        if "bytes" in facts and int(facts["bytes"]) != size:
            problems.append(
                f"{path.name} is {size} bytes; the record says {facts['bytes']}. It has been "
                f"truncated, appended to or replaced since the run finished")
            continue
        if _sha256(path) != facts["sha256"]:
            problems.append(
                f"{path.name} does not match the sha256 recorded for it. The file has changed "
                f"since the run that claims it")

    # -- 4. the final state describes THIS System ----------------------------------------------
    restart = Path(restart)
    if particles is not None and restart.is_file():
        try:
            from openmm import XmlSerializer

            reread = XmlSerializer.deserialize(restart.read_text(encoding="utf-8"))
            written = reread.getPositions(asNumpy=True).shape[0]
        except Exception as exc:                                  # noqa: BLE001 - reported
            problems.append(f"{restart.name} could not be read back as a serialised State: {exc}")
        else:
            if int(written) != int(particles):
                problems.append(
                    f"{restart.name} holds {written} particles and this System has {particles}. "
                    f"The restart belongs to a different molecule")

    # -- 5. the committed checkpoint ------------------------------------------------------------
    if checkpoints is not None:
        checkpoints = Path(checkpoints)
        pointer = checkpoints / POINTER_NAME
        if not pointer.is_file():
            problems.append(
                f"no committed checkpoint pointer at {pointer}. A completed stage commits a final "
                f"generation, so its absence means the run did not finish the way the record says")
        else:
            try:
                committed = read_committed(checkpoints)
            except CheckpointError as refusal:
                # read_committed already verifies the sidecar, the binary and its digest.
                problems.append(f"the committed checkpoint is not usable: {refusal}")
                committed = None
            if committed is not None:
                state = committed.get("state") or {}
                if str(state.get("fingerprint")) != str(fingerprint):
                    problems.append(
                        "the committed checkpoint was taken under a different configuration "
                        "fingerprint than this stage resolves to")
                done = int(state.get("steps_done", -1))
                if done != steps:
                    problems.append(
                        f"the committed checkpoint stops at step {done} and this stage asks for "
                        f"{steps}. It was interrupted, not completed")
                if state.get("stage") not in (None, name):
                    problems.append(
                        f"the committed checkpoint belongs to stage {state.get('stage')!r}, "
                        f"not {name!r}")
                # NOT "seed": the checkpoint records the derived RUNTIME seed
                # (`checked.seed`), which is drawn fresh when the configuration does not pin one
                # -- comparing it to the raw `stage.get("seed")` config field compares two
                # different things and would refuse a healthy, unpinned-seed run for disagreeing
                # with itself. The fingerprint already binds the resolved configuration; the seed
                # POLICY is part of that, the drawn value is not.
                wanted_ensemble = stage.get("ensemble")
                if wanted_ensemble is not None and "ensemble" in state:
                    if str(state.get("ensemble")) != str(wanted_ensemble):
                        problems.append(
                            f"the committed checkpoint records ensemble={state.get('ensemble')!r} "
                            f"and this stage resolves ensemble={wanted_ensemble!r}")
                # -- 6. the committed stream counts against the files on disk -----------------
                for role, count in sorted((state.get("streams") or {}).items()):
                    path = (streams or {}).get(role)
                    if path is None or not Path(path).exists():
                        problems.append(
                            f"the checkpoint vouches for {count} {role} records, and {role} is "
                            f"not present to be checked")
                        continue
                    actual = _count_records(role, Path(path))
                    if actual is None:
                        problems.append(
                            f"{Path(path).name} could not be read as {role}; a completed stage's "
                            f"outputs must be readable")
                    elif int(actual) != int(count):
                        problems.append(
                            f"{Path(path).name} holds {actual} records and the committed "
                            f"checkpoint vouches for {count}")

    # -- 7. the inventory: nothing this run WOULD HAVE PRODUCED may be missing ------------------
    #
    # Not every inventory role -- the inventory also names paths for streams this configuration
    # never opens, such as a trajectory for a minimisation (`steps == 0`) or a CV series when
    # reporting is off. Requiring all of them made a healthy, idempotently-skippable minimisation
    # fail verification because it had never written a trajectory it was never going to write.
    if inventory is not None:
        always_expected = {"out", "log", "final_state"} | expected_roles
        for role, path in sorted((getattr(inventory, "roles", None) or {}).items()):
            if role in getattr(inventory, "resumable", frozenset()):
                continue
            if role not in always_expected:
                continue
            if not Path(path).exists():
                problems.append(f"the {role} this stage owns is missing: {path}")

    return problems
