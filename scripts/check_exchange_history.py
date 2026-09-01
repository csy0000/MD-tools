#!/usr/bin/env python
"""Assert a REST2 exchange history is one continuous run of `rounds` rounds.

    python scripts/check_exchange_history.py md_script 6

A resumed ladder that restarted its attempt indices looks healthy in every other artifact it
writes: the replicas move, the trajectories grow, the log says "segment complete". This is the
check that separates one trajectory from several.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def check(directory: Path, rounds: int | None = None) -> int:
    path = Path(directory) / "exchange_attempts.csv"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; the ladder wrote no exchange history")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} has a header and no attempts")

    index = sorted({int(row["attempt_index"]) for row in rows})
    if index != list(range(len(index))):
        raise SystemExit(f"the exchange history is not contiguous, so this is more than one run: "
                         f"{index}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps):
        raise SystemExit(f"step counts are not monotonic across the history: {steps}")
    if rounds is not None and len(index) != rounds:
        raise SystemExit(f"expected {rounds} exchange rounds, found {len(index)}")

    print(f"exchange history continuous over {len(index)} round(s), "
          f"{len(rows)} attempt(s), through step {max(steps)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
    raise SystemExit(check(Path(sys.argv[1]), expected))
