"""Where every file of a run lives. One authority, so no caller invents a path.

THE LAYOUT, as `docs/basics/run-layout.md` specifies it:

    <dataset>/                    one dataset is a system and every run on it
      build/                      the built system: shared by every method and every repeat
      min/                        the minimised structure: shared for the same reason
      input/                      every .in: shared, because an input is not per-repeat
      <method>-run<N>/
        run.config                the seed -- the only per-run declaration
        resolved.config           authoritative, resolved per run
        eq/                       this run's equilibration
        remd<n>/                  one directory per thermodynamic STATE
        remd_records/             the ladder's own records, one set per segment
        rank/                     per-PROCESS reports
        bundles/                  reproducibility, with or without MD-tools

WHAT IS SHARED FOLLOWS FROM THE PHYSICS. `build/`, `min/` and `input/` sit at the dataset root
because every run on one system starts from the same built System, the same minimised coordinates
and the same instructions -- minimisation draws no velocities and has no seeded stochastic
element, so a second copy could only drift from the first. `eq/` is per run because equilibration
draws Maxwell velocities from that run's own seed: two repeats are SUPPOSED to diverge there.

WHY A MODULE RATHER THAN STRING JOINS AT SEVEN CALL SITES. `reservoir.py`, `run/continuation.py`,
`ais/run.py`, `openmm/checkpoint.py`, `remd/state_trajectories.py`, `run/overwrite.py` and
`md/stage.py` each assembled output paths themselves, which is seven places to disagree about
where a file goes -- and a reader of a finished run has no way to tell which one was right.

THE PER-STATE FILE NAMES ARE NOT REDEFINED HERE. `md_tools.remd.amber_trajectory` owns them and
is imported; this module owns the DIRECTORIES and the run-level record names. Two modules naming
the same file is the defect this one exists to remove.
"""
from __future__ import annotations

from pathlib import Path

from .remd.amber_trajectory import (RANK_DIRECTORY, state_cv_name, state_directory,
                                    state_restart_name, state_system_name, state_trajectory_name)

__all__ = [
    "BUILD", "MIN", "INPUT", "EQ", "RECORDS", "RANK", "BUNDLES",
    "RUN_ROOT_FILES", "DatasetLayout", "RunLayout",
    "ledger_name", "record_name", "eq_name", "eq_stage_key",
    # re-exported so a caller needs one import, not two
    "RANK_DIRECTORY", "state_directory", "state_system_name", "state_cv_name",
    "state_restart_name", "state_trajectory_name",
]

#: Directory names. Strings rather than methods, because a test should be able to assert the
#: literal a finished run on disk will carry.
BUILD = "build"
MIN = "min"
INPUT = "input"
EQ = "eq"
RECORDS = "remd_records"
RANK = RANK_DIRECTORY
BUNDLES = "bundles"

#: What stays at the run root: the per-run declaration, the resolved document it produced, and
#: the generated helpers whose identity is bound to this run. Everything else is in a directory.
RUN_ROOT_FILES = ("run.config", "resolved.config", "build-md.log", "build-md.out")

#: Ladder-wide records, per segment. Indexed by segment ALONE, which is why they share one
#: directory with the segment in each filename, rather than a directory per segment.
_RECORDS = {
    "ledger": "ledger_prod{segment}.nc",
    "ledger_solute": "ledger_prod{segment}.solute.nc",
    "checkpoint": "checkpoint_prod{segment}.nc",
    "restart": "restart_prod{segment}.json",
    "runstate": "runstate_prod{segment}.json",
    "rem_log": "rem_prod{segment}.log",
    "exchange": "exchange_prod{segment}.csv",
}


def record_name(kind: str, *, segment: int = 1, protocol: str = "REST2") -> str:
    """One ladder-wide record's filename, for `remd_records/`.

    `out`, `log`, `group` and `stage_stdout` carry the PROTOCOL in the name because that is what
    the run is called in its own reports (`REST2.out`, `rem.log`); the rest are named for what
    they are, since a reader looking for the exchange ledger should not have to know which
    protocol wrote it.
    """
    index = int(segment)
    if index < 1:
        raise ValueError(f"a segment number starts at 1; got {segment}")
    if kind in _RECORDS:
        return _RECORDS[kind].format(segment=index)
    suffixes = {"out": "out", "log": "log", "group": "group", "stage_stdout": "stage.stdout"}
    if kind in suffixes:
        return f"{protocol}_prod{index}.{suffixes[kind]}"
    raise ValueError(f"unknown record kind {kind!r}; known: "
                     f"{sorted(set(_RECORDS) | set(suffixes))}")


def ledger_name(*, segment: int = 1, solute: bool = False) -> str:
    """`ledger_prod<x>.nc` -- the exchange ledger, which is NOT a trajectory.

    It carries no coordinates and no `Conventions` attribute, so cpptraj will not read it: its
    variables are the exchange history (`u`, `u_evaluated`, `proposed`, `accepted`, each
    `(exchange, state, state)`) plus `state_to_walker` and the frame bookkeeping. It was called
    `REST2.nc`, which read as the run's trajectory and is how a reader came to expect coordinates
    in it. `exchange_prod<x>.csv` beside it is the flat per-(exchange, state) view of the same
    thing: NetCDF for the full matrix, CSV for the flat view.
    """
    return record_name("ledger_solute" if solute else "ledger", segment=segment)


def eq_stage_key(order: int) -> str:
    """`eq_1`, `eq_2`, ... -- an equilibration stage by its POSITION in the chain.

    THE ENSEMBLE IS DELIBERATELY NOT IN THE NAME, and that has a consequence worth stating.
    The stages are generated as `eq_nvt_posres`, `eq_npt_posres`, `eq_npt_free` under explicit
    solvent and RENAMED under implicit (`eq_nvt_posres`, `eq_nvt_posres_2`, `eq_nvt_free`),
    precisely so a pressure-coupled name never appears on a boxless run. Under `eq_<k>` that
    distinction cannot live in the filename, so it lives in each stage's `.out` header and its
    resolved configuration, which state the ensemble explicitly. The invariant survives; the
    evidence for it moves.
    """
    index = int(order)
    if index < 1:
        raise ValueError(f"an equilibration stage position starts at 1; got {order}")
    return f"eq_{index}"


def eq_name(order: int, what: str) -> str:
    """One equilibration artefact's filename.

    `what` is a suffix (`xml`, `out`, `log`, `py`, `dcd`) or a compound
    (`whole.nc`, `solute.nc`, `energy_components.csv`, `mdout.csv`). `checkpoints` gives the
    directory name.

    THE TWO COORDINATE STREAMS ARE SOLVENT-DEPENDENT: explicit solvent writes `whole.nc` and
    `solute.nc`, implicit writes `whole.nc` and a `dcd`. That is not an inconsistency -- ordinary
    MD writes genuine DCD, and OpenMM has no native NetCDF writer -- so the pair is whatever the
    stage actually produced rather than a fixed two.
    """
    return f"{eq_stage_key(order)}.{what}"


class DatasetLayout:
    """The dataset root: the system, its minimised structure, and the inputs every run shares."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    # -- the three shared directories -----------------------------------------------------------

    @property
    def build(self) -> Path:
        return self.root / BUILD

    @property
    def min(self) -> Path:
        return self.root / MIN

    @property
    def input(self) -> Path:
        return self.root / INPUT

    # -- the files inside them ------------------------------------------------------------------

    def built(self, what: str = "xml") -> Path:
        """`build/built.xml`, `built.pdb`, `built.solute.pdb`, `built.log`.

        The names are `build-top`'s own: `-os` defaults to `built.xml`, `-op` to `built.pdb` and
        `-log` to `built.log`, so the layout does not rename the tool's output. Which files exist
        varies -- an implicit system has no `build-top.out`, a ligand system carries its prepared
        molecule -- so a caller asks for one rather than assuming a set.

        That molecule is `<RESNAME>.sdf` (`build/TYL.sdf`), named for the solute residue, since
        0.5.4, and `built.sdf` before it. `built("sdf")` still names the old spelling, and
        `run.preflight._ligand_sdf_beside` is the one place that finds whichever exists; ask it,
        not this method, for the molecule.
        """
        return self.build / (what if what.startswith("built") or "." in what
                             else f"built.{what}")

    def minimised(self, what: str = "xml") -> Path:
        return self.min / f"min.{what}"

    def stage_input(self, name: str) -> Path:
        """`input/min.in`, `input/eq_1.in`, `input/REST2.in`."""
        return self.input / (name if name.endswith(".in") else f"{name}.in")

    def run(self, method: str, index: int) -> "RunLayout":
        return RunLayout(self.root / f"{method}-run{int(index)}", dataset=self)

    def existing_runs(self, method: str) -> list[int]:
        """Every `<method>-run<N>` index already present, ascending.

        A suffix that is not a number is IGNORED rather than refused: a directory someone named
        `REST2-run-old` is not this tool's business, and failing the scan over it would make an
        unrelated directory block every new run.
        """
        import re

        if not self.root.is_dir():
            return []
        pattern = re.compile(rf"^{re.escape(method)}-run(\d+)$")
        found = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            matched = pattern.match(path.name)
            if matched:
                found.append(int(matched.group(1)))
        return sorted(found)

    def next_run_index(self, method: str) -> int:
        """The index a new run of `method` takes: one past the highest that exists.

        GAPS ARE NOT REUSED, and that is the point rather than an oversight. With `run1` and
        `run3` present the answer is 4, never 2. A gap means a run was moved, archived or
        registered elsewhere -- `data-register` takes the source away and leaves a symlink -- and
        handing its number to a new run would make two different experiments share an identity
        that a manifest somewhere already refers to. Counting the directories present would do
        exactly that.
        """
        existing = self.existing_runs(method)
        return (existing[-1] + 1) if existing else 1

    def next_run(self, method: str) -> "RunLayout":
        """The layout for the next run of `method`. Creates nothing."""
        return self.run(method, self.next_run_index(method))


class RunLayout:
    """One run: its own equilibration, its per-state output, and its ladder records."""

    def __init__(self, root, *, dataset: DatasetLayout | None = None,
                 protocol: str = "REST2") -> None:
        self.root = Path(root)
        self.dataset = dataset if dataset is not None else DatasetLayout(self.root.parent)
        self.protocol = protocol

    # -- directories ----------------------------------------------------------------------------

    @property
    def eq(self) -> Path:
        return self.root / EQ

    @property
    def records(self) -> Path:
        return self.root / RECORDS

    @property
    def rank(self) -> Path:
        return self.root / RANK

    @property
    def bundles(self) -> Path:
        return self.root / BUNDLES

    def state(self, index: int) -> Path:
        """`remd<n>/` -- everything belonging to one thermodynamic state."""
        return self.root / state_directory(index)

    # -- per-run files --------------------------------------------------------------------------

    @property
    def run_config(self) -> Path:
        """The seed, and nothing that is not genuinely per-run."""
        return self.root / "run.config"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved.config"

    # -- per-state files ------------------------------------------------------------------------

    def state_system(self, index: int) -> Path:
        """The rung Hamiltonian, serialised. No segment: it does not change between segments."""
        return self.state(index) / state_system_name(index)

    def state_trajectory(self, index: int, *, content: str = "whole", segment: int = 1) -> Path:
        name = state_trajectory_name(index, content=content, segment=segment)
        # The whole-system stream is `remd_state<n>_prod<x>.nc`; `state_trajectory_name` still
        # spells it `whole_...`, so the one rename happens here rather than in two places.
        if content == "whole":
            name = name.replace("whole_state", "remd_state", 1)
        return self.state(index) / name

    def state_cv(self, index: int, *, segment: int = 1, suffix: str = "dat") -> Path:
        return self.state(index) / state_cv_name(index, segment=segment, suffix=suffix)

    def state_restart(self, index: int, *, segment: int = 1, suffix: str = "json") -> Path:
        return self.state(index) / state_restart_name(index, segment=segment, suffix=suffix)

    # -- ladder records and rank reports --------------------------------------------------------

    def record(self, kind: str, *, segment: int = 1) -> Path:
        return self.records / record_name(kind, segment=segment, protocol=self.protocol)

    def ledger(self, *, segment: int = 1, solute: bool = False) -> Path:
        return self.records / ledger_name(segment=segment, solute=solute)

    def rank_report(self, rank: int | str, *, segment: int = 1, what: str = "out") -> Path:
        """`rank/REST2_prod<x>.out.rank<r>`.

        NOT under `remd<n>/`: a rank is a PROCESS and a state is a thermodynamic state, and they
        are not in bijection -- the reference ladders ran 6 states on 5 ranks and 4 states on 3.
        Filing a rank report under a state directory would assert a correspondence that does not
        exist.
        """
        label = rank if isinstance(rank, str) else f"{int(rank):02d}"
        return self.rank / f"{self.protocol}_prod{int(segment)}.{what}.rank{label}"

    # -- equilibration --------------------------------------------------------------------------

    def eq_artifact(self, order: int, what: str) -> Path:
        return self.eq / eq_name(order, what)

    def eq_checkpoints(self, order: int) -> Path:
        return self.eq / f"{eq_stage_key(order)}.checkpoints"
