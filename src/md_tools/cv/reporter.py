"""Writing the collective-variable series: the CSV, the sidecar, and the cost accounting.

WHAT THIS DELIBERATELY DOES NOT DO

    It does not touch the System, add a Force, register a force group, or ask the Context for
    energies. A CV observation retrieves POSITIONS -- which the reporting point generally already
    needed -- and does arithmetic on them. That is what keeps a CV-enabled run the same experiment
    as a CV-disabled one, and it is what makes "enabling CVs changes no energy and no coordinate"
    a property that can actually be tested rather than hoped for.

    The evaluation cost is recorded SEPARATELY from energy evaluations, for the same reason. A
    position-only torsion is not an energy evaluation, and counting it as one would corrupt the
    one number that says how expensive the Hamiltonian is -- which is the number used to compare
    protocols and to size a machine allocation.

CSV, NOT SOMETHING CLEVERER

    UTF-8, one header line, no comment lines, stable column order. A comment line is the specific
    thing that breaks a naive reader, and a CV series is read by exactly those: a person's pandas
    one-liner, a plotting script, a spreadsheet. The provenance a comment would have carried goes
    in the sidecar instead, where it can be complete rather than squeezed into a `#` line.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .torsion import torsion_degrees


class CVReportError(RuntimeError):
    """A collective-variable series could not be written.

    A simulation failure, deliberately -- never a reason to carry on with reporting disabled. A
    run that quietly stopped writing the observable it exists to measure is worse than one that
    stopped, because it finishes and looks complete.
    """


#: Written beside the CSV. Everything needed to interpret the series without the topology, the
#: configuration, or the cv.yaml that produced it.
SIDECAR_SUFFIX = ".cv.json"


class CVSeries:
    """One CSV of torsion values, plus its sidecar. Append-only, flushed per row.

    Flushed per row on purpose: an interrupted run's CV series must be readable up to the last
    committed observation, and must agree with the count the checkpoint vouches for. A buffered
    writer would leave the file short of what the checkpoint claims and make a resumable
    interruption look like corruption.
    """

    def __init__(self, path, definition, *, extra_columns=(), sidecar_extra=None):
        self.path = Path(path)
        self.definition = definition
        self.extra_columns = tuple(extra_columns)
        self.rows_written = 0
        #: Cost accounting, kept apart from any energy-evaluation counter. See the module note.
        self.evaluations = 0
        self.seconds = 0.0
        self._handle = None
        self._sidecar_extra = dict(sidecar_extra or {})

    # -- lifecycle ---------------------------------------------------------------------------

    def open(self, *, append_from: int = 0):
        """Create or reopen the CSV. `append_from` truncates to that many committed rows first.

        Truncation is what makes a resume produce no duplicate and no gap: the checkpoint says how
        many observations it committed, and anything after that belongs to steps whose dynamics
        are about to be repeated. Keeping them would double-count; deleting the file would lose
        the committed ones.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if append_from and self.path.is_file():
                self._truncate_to(append_from)
                self._handle = self.path.open("a", encoding="utf-8", newline="")
                self.rows_written = append_from
            else:
                self._handle = self.path.open("w", encoding="utf-8", newline="")
                self._handle.write(",".join(self.columns()) + "\n")
                self._handle.flush()
                self.rows_written = 0
        except OSError as broken:
            raise CVReportError(
                f"{self.path} could not be opened for collective-variable output: {broken}") \
                from None
        self._write_sidecar()
        return self

    def columns(self) -> list[str]:
        return [*self.extra_columns, *self.definition.names]

    def _truncate_to(self, rows: int) -> None:
        """Keep the header and exactly `rows` data lines."""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise CVReportError(
                f"{self.path} is empty but a continuation expected {rows} committed "
                f"observation(s) in it")
        kept = lines[:1 + int(rows)]
        if len(kept) < 1 + int(rows):
            raise CVReportError(
                f"{self.path} holds {len(lines) - 1} observation(s) and the checkpoint vouches "
                f"for {rows}. The series is shorter than the run it belongs to claims; it cannot "
                f"be continued without inventing the missing rows")
        self.path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def _write_sidecar(self) -> None:
        body = dict(self.definition.resolved())
        body["column_order"] = self.columns()
        body.update(self._sidecar_extra)
        # `<stem>.cv.csv` -> `<stem>.cv.json`. Spelled by string rather than `with_suffix`,
        # which would have to be applied twice for a two-part suffix and reads as a bug either way.
        sidecar = Path(str(self.path)[:-len(".csv")] + ".json") \
            if str(self.path).endswith(".cv.csv") else self.path.with_suffix(SIDECAR_SUFFIX)
        try:
            sidecar.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n",
                               encoding="utf-8")
        except OSError as broken:
            raise CVReportError(f"{sidecar} could not be written: {broken}") from None
        self.sidecar = sidecar

    def close(self):
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            finally:
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    # -- observation -------------------------------------------------------------------------

    def evaluate(self, positions, box=None) -> list[float]:
        """Every torsion, in column order, from positions this caller already had.

        Timed and counted here so the cost is attributable and is never added to an
        energy-evaluation total.
        """
        started = time.perf_counter()
        try:
            values = [torsion_degrees(positions, cv.indices, box)
                      for cv in self.definition.variables]
        except Exception as broken:
            raise CVReportError(
                f"a collective variable could not be evaluated: "
                f"{type(broken).__name__}: {broken}") from None
        self.seconds += time.perf_counter() - started
        self.evaluations += 1
        return values

    def write(self, leading, values) -> None:
        """One row: the leading protocol columns, then the CV values.

        `None` in `leading` becomes an empty field -- which is how "there is no trajectory frame
        at this step" is spelled, and it must be empty rather than 0 or -1, both of which are
        valid frame indices.
        """
        if self._handle is None:
            raise CVReportError(f"{self.path} is not open for writing")
        if len(values) != len(self.definition.variables):
            raise CVReportError(
                f"{self.path}: {len(values)} value(s) for "
                f"{len(self.definition.variables)} collective variable(s)")
        fields = [_field(value) for value in leading] + [f"{float(v):.6f}" for v in values]
        if len(fields) != len(self.columns()):
            raise CVReportError(
                f"{self.path}: a row of {len(fields)} field(s) does not match the "
                f"{len(self.columns())} declared column(s)")
        try:
            self._handle.write(",".join(fields) + "\n")
            self._handle.flush()
        except OSError as broken:
            raise CVReportError(
                f"{self.path}: a collective-variable row could not be written: {broken}") from None
        self.rows_written += 1

    def cost(self) -> dict[str, float]:
        """What the reporting cost, for the run record. Never merged into energy evaluations."""
        return {"cv_evaluations": int(self.evaluations),
                "cv_seconds": round(float(self.seconds), 6),
                "cv_rows": int(self.rows_written)}


def _field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
