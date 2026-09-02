"""The `.out` file: what a person reads while a run is going.

MD-tools writes two files about every run, and they are for two different readers.

    -log    the provenance record. Resolved configuration identity, input and output hashes,
            software and hardware, warnings, status, MD-data-contract fields. A machine reads it;
            `read_record` parses it; registration decides completion from it.
    -o      this. What the run is, how far it has got, what the energy and temperature are doing,
            whether it resumed, and how it ended.

They were briefly one file. That was wrong in a specific way: it made a person open a machine
record and read past a hundred lines of hashes to find out whether a simulation was making
progress. Amber has kept `mdout` and a logfile separate for the same reason, and anyone coming
from `pmemd` expects `-o` to be the thing they `tail`.

Deliberately plain text with no parser on the other end. Nothing downstream reads it -- the moment
something did, it would become a second machine record that could disagree with the first, and the
separation this file exists for would collapse the other way.
"""
from __future__ import annotations

import datetime as _datetime
from pathlib import Path
from typing import Any

__all__ = ["SimulationOutput"]

_RULE = "-" * 78


class SimulationOutput:
    """A readable running account of one simulation.

    Opened before anything integrates and closed on the way out, success or failure, so a run that
    died still leaves a file saying what it was doing when it died. That is the case the reader
    most needs it for.
    """

    def __init__(self, path: str | Path, *, title: str, log_path: str | Path | None = None,
                 echo: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", buffering=1)   # line-buffered: tailable
        self.echo = echo
        self._started = _datetime.datetime.now(_datetime.timezone.utc)

        self.write(f"md-openmm  |  {title}")
        self.write("=" * 78)
        self.write(f"started              {self._started.isoformat(timespec='seconds')}")
        if log_path is not None:
            # The reciprocal reference. Either file names the other, so whichever one a reader
            # opens first tells them where the rest of the story is.
            self.write(f"provenance record    {log_path}")
        self.write("")

    # -- writing ------------------------------------------------------------------------------

    def write(self, line: str = "") -> None:
        self.handle.write(line + "\n")
        if self.echo:
            print(line)

    def heading(self, text: str) -> None:
        self.write("")
        self.write(text)
        self.write(_RULE)

    def field(self, name: str, value: Any) -> None:
        self.write(f"  {name:<28} {value}")

    def table_header(self, columns: tuple[str, ...], widths: tuple[int, ...]) -> None:
        self.write("  " + "".join(f"{c:>{w}}" for c, w in zip(columns, widths)))
        self.write("  " + "".join("-" * (w - 1) + " " for w in widths))

    def table_row(self, values: tuple[Any, ...], widths: tuple[int, ...]) -> None:
        cells = []
        for value, width in zip(values, widths):
            cells.append(f"{value:>{width}.4f}" if isinstance(value, float)
                         else f"{value:>{width}}")
        self.write("  " + "".join(cells))

    # -- the ends -----------------------------------------------------------------------------

    def completed(self, summary: str) -> None:
        self.heading("Summary")
        self.write(f"  {summary}")
        self._finish("completed")

    def failed(self, message: str) -> None:
        self.heading("Failure")
        for line in str(message).splitlines():
            self.write(f"  {line}")
        self._finish("failed")

    def _finish(self, status: str) -> None:
        finished = _datetime.datetime.now(_datetime.timezone.utc)
        elapsed = (finished - self._started).total_seconds()
        self.write("")
        self.write(f"finished             {finished.isoformat(timespec='seconds')}")
        self.write(f"elapsed              {elapsed:.1f} s")
        self.write(f"status               {status}")
        self.close()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> "SimulationOutput":
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if not self.handle.closed:
            # An exception on the way out still has to leave a readable file: a run that crashed
            # is exactly the one whose `.out` somebody will open.
            if value is not None:
                self.failed(f"{type(value).__name__}: {value}")
            else:
                self.close()
