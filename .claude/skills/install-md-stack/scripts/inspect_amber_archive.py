#!/usr/bin/env python3
"""Inspect a user-provided Amber archive without extracting or exposing its contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unsafe_reason(member: tarfile.TarInfo) -> str | None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        return "absolute path or parent traversal"
    if member.isdev() or member.isfifo():
        return "device node or FIFO"
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute():
            return "absolute link target"
        combined = path.parent.joinpath(target)
        depth = 0
        for part in combined.parts:
            depth += -1 if part == ".." else (0 if part in ("", ".") else 1)
            if depth < 0:
                return "link target escapes archive root"
    return None


def classify(names: list[str]) -> dict[str, Any]:
    lowered = [name.lower() for name in names]
    signals = {
        "release26_name": any("amber26" in name or "ambertools26" in name for name in lowered),
        "ambertools26": any(
            marker in name
            for name in lowered
            for marker in ("ambertools", "antechamber", "tleap", "cpptraj")
        ),
        "amber26_or_pmemd": any("pmemd" in name for name in lowered),
        "run_cmake": any(name.endswith("run_cmake") for name in lowered),
        "install_docs": [name for name in names
                         if Path(name).name.lower().startswith(("readme", "install"))][:30],
    }
    if signals["ambertools26"] and signals["amber26_or_pmemd"]:
        kind = "combined-or-overlay-layout"
    elif signals["amber26_or_pmemd"]:
        kind = "amber26-or-pmemd"
    elif signals["ambertools26"]:
        kind = "ambertools26"
    else:
        kind = "unrecognized"
    return {"inferred_kind": kind, **signals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to a user-provided Amber tar archive")
    parser.add_argument("--expected", choices=("auto", "ambertools26", "amber26-or-pmemd"),
                        default="auto")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    archive = Path(args.archive).expanduser()
    errors: list[str] = []
    if not archive.exists():
        errors.append("archive does not exist")
    elif archive.is_symlink():
        errors.append("archive is a symbolic link; provide the resolved regular file")
    elif not archive.is_file():
        errors.append("archive is not a regular file")

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive.resolve()) if archive.exists() else str(archive),
        "expected": args.expected,
        "errors": errors,
    }
    if not errors:
        report["size_bytes"] = archive.stat().st_size
        report["sha256"] = sha256_file(archive)
        try:
            with tarfile.open(archive, mode="r:*") as tar:
                members = tar.getmembers()
                unsafe = [{"path": member.name, "reason": reason}
                          for member in members
                          if (reason := unsafe_reason(member))]
                names = [member.name for member in members if member.isfile()]
                roots = sorted({PurePosixPath(member.name).parts[0]
                                for member in members if PurePosixPath(member.name).parts})
                report.update({
                    "member_count": len(members),
                    "regular_file_count": len(names),
                    "top_level_entries": roots[:30],
                    "unsafe_members": unsafe[:50],
                    "classification": classify(names),
                })
                if unsafe:
                    errors.append("archive contains unsafe members and must not be extracted")
        except (tarfile.TarError, OSError) as exc:
            errors.append(f"archive could not be read: {type(exc).__name__}: {exc}")

    kind = (report.get("classification") or {}).get("inferred_kind")
    if args.expected != "auto" and kind == "unrecognized":
        report.setdefault("warnings", []).append(
            "archive type could not be inferred from installation-relevant paths; "
            "verify it against the official release description before extraction"
        )
    elif args.expected != "auto" and kind not in (args.expected, "combined-or-overlay-layout"):
        errors.append(f"archive classification {kind!r} does not match {args.expected!r}")
    report["ok"] = not errors
    report["errors"] = errors

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
