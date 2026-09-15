"""The N per-state Amber trajectories, written and committed as one logical frame.

One frame event produces N rows -- one per fixed thermodynamic state -- spread across N files. That
is N chances to be interrupted half way, so the set is treated as a single commit and `exchange.nc`
holds the one marker that says how much of it is real:

    write the row to every state trajectory
    sync every state trajectory
    advance the committed-frame marker in the authoritative record
    sync the authoritative record

A crash anywhere before the marker moves leaves rows on disk that the marker does not count. Those
rows are UNCOMMITTED: a continuation ignores them and overwrites them from the marker onward, which
is the documented policy. It is not "remove them" because an Amber NetCDF cannot shrink, and it is
not "trust them" because nothing says they are all there.

A state file that is SHORTER than the marker, or missing, is corruption rather than an incomplete
commit -- the marker only advances once every file has the row. That is refused, never padded: a
fabricated frame is worse than a refusal because it looks like data.
"""
from __future__ import annotations

from pathlib import Path

from .amber_trajectory import AmberTrajectoryWriter, read_frames, state_trajectory_name

#: Where the per-state trajectories live, relative to the run directory. A subdirectory keeps the
#: N files from crowding the record they belong to.
DEFAULT_SUBDIRECTORY = "."


class StateTrajectoryError(RuntimeError):
    """The set of state trajectories is not in a state that may be written or continued."""


class StateTrajectorySet:
    """One Amber trajectory per fixed thermodynamic state, committed together."""

    def __init__(self, writers, *, directory, atom_indices=None, content="whole"):
        self.writers = list(writers)
        self.directory = Path(directory)
        #: When set, each frame is the SUBSET at these indices. The solute set uses it; the whole
        #: set leaves it None and writes every atom.
        self.atom_indices = None if atom_indices is None else list(int(i) for i in atom_indices)
        self.content = content

    @property
    def n_states(self):
        return len(self.writers)

    @classmethod
    def create(cls, directory, *, taus, n_atoms, temperature_k, periodic,
               program="md-tools", program_version="0", atom_indices=None, content="whole",
               segment=1):
        """A fresh set. Refuses to overwrite files it did not just create.

        `atom_indices` makes this the SOLUTE set: one AMBER trajectory per state holding the
        solute alone. It used to be a single file with a `walker` dimension --
        `solute_positions(frame, walker, atom, xyz)` -- which no standard reader opens, because
        the AMBER convention has no walker axis and mdtraj looks for `coordinates`. One file per
        state is the same data in a format cpptraj and mdtraj read without being told anything.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if atom_indices is not None:
            n_atoms = len(atom_indices)
        name = lambda i: state_trajectory_name(i, content=content, segment=segment)  # noqa: E731
        existing = [name(i) for i in range(len(taus)) if (directory / name(i)).exists()]
        if existing:
            raise StateTrajectoryError(
                f"{len(existing)} state trajectory/ies already exist in {directory} "
                f"({', '.join(existing[:4])}). A new run does not write into them; "
                f"pass --overwrite to "
                f"replace a run deliberately, or continue the existing one with --resume.\n"
                f"  (--overwrite, NOT --force: this message is reached from `md-run`, which "
                f"defines no --force. The generated ladder script accepts either.)")
        writers = [
            AmberTrajectoryWriter(
                directory / name(index), n_atoms=n_atoms, state_index=index,
                tau=float(tau), temperature_k=temperature_k, periodic=periodic,
                program=program, program_version=program_version)
            for index, tau in enumerate(taus)]
        return cls(writers, directory=directory, atom_indices=atom_indices, content=content)

    @classmethod
    def continue_from(cls, directory, *, taus, n_atoms, committed_frames,
                      atom_indices=None, content="whole", segment=1):
        """Reopen an existing set to continue it, at the committed-frame marker.

        Inspection comes FIRST and the files are opened read-only for it. Nothing is opened for
        writing until every state file has been shown to be present, to belong to this ladder, and
        to be at least as long as the marker. A continuation that discovered the corruption half
        way through would already have appended to the healthy files.
        """
        directory = Path(directory)
        report = cls.inspect(directory, n_states=len(taus), expect_frames=int(committed_frames),
                             expect_taus=list(taus), content=content, segment=segment)
        if report["problems"]:
            raise StateTrajectoryError(
                "the state trajectories in {} cannot be continued:\n  - {}".format(
                    directory, "\n  - ".join(report["problems"])))
        if atom_indices is not None:
            n_atoms = len(atom_indices)
        writers = [
            AmberTrajectoryWriter.open_existing(
                directory / state_trajectory_name(index, content=content, segment=segment),
                n_atoms=n_atoms, state_index=index,
                tau=float(tau), from_frame=int(committed_frames))
            for index, tau in enumerate(taus)]
        return cls(writers, directory=directory, atom_indices=atom_indices, content=content)

    # -- the commit ------------------------------------------------------------------------------

    def write_frame(self, *, step, time_ps, state_to_walker, configurations):
        """One row per state, from the configuration that now occupies it.

        This is where a fixed-state trajectory differs from a walker trajectory: after an accepted
        exchange the configuration in state 2 is a different walker's, and it is that one which is
        written to `whole_state2_prod1.nc`.
        """
        if len(state_to_walker) != self.n_states:
            raise StateTrajectoryError(
                f"the mapping names {len(state_to_walker)} states but this set has "
                f"{self.n_states}")
        for state_index, writer in enumerate(self.writers):
            walker = int(state_to_walker[state_index])
            configuration = configurations[walker]
            positions = (configuration.positions if self.atom_indices is None
                         else configuration.positions[self.atom_indices])
            writer.append(positions, time_ps=time_ps,
                          box_nm=configuration.box)
        return self.n_states

    def sync(self):
        """Every file, before the marker that counts them is allowed to move."""
        for writer in self.writers:
            writer.sync()

    def close(self):
        for writer in self.writers:
            writer.close()

    # -- continuation ----------------------------------------------------------------------------

    @staticmethod
    def inspect(directory, *, n_states, expect_frames=None, expect_taus=None,
                content="whole", segment=1):
        """READ-ONLY. Are the N files present, coherent with each other, and long enough?

        `expect_frames` is the committed-frame marker from the authoritative record. A file with
        MORE rows than that has uncommitted rows, which is ordinary after a crash. A file with
        FEWER is corruption.
        """
        directory = Path(directory)
        problems, facts = [], {}
        frames = {}
        for index in range(n_states):
            path = directory / state_trajectory_name(index, content=content, segment=segment)
            if not path.is_file():
                problems.append(
                    f"{path.name} is missing. Every state owns one trajectory, and a "
                    f"state without "
                    f"one cannot be reconstructed: refusing rather than fabricating its frames.")
                continue
            try:
                seen = read_frames(path)
            except Exception as failure:                # noqa: BLE001 - reported, not raised
                problems.append(f"{path.name} could not be read ({type(failure).__name__})")
                continue
            frames[index] = seen
            if seen["state_index"] != index:
                problems.append(
                    f"{path.name} records state_index {seen['state_index']}, not {index}. The "
                    f"filename and the metadata disagree about which state this is.")
            if expect_taus is not None and abs(seen["tau"] - float(expect_taus[index])) > 1e-12:
                problems.append(
                    f"{path.name} records tau {seen['tau']} but this ladder puts "
                    f"{expect_taus[index]} at state {index}")

        if frames:
            counts = {index: seen["n_frames"] for index, seen in frames.items()}
            facts["frames_per_state"] = counts
            if expect_frames is not None:
                short = {i: n for i, n in counts.items() if n < expect_frames}
                if short:
                    problems.append(
                        f"state trajectory/ies {sorted(short)} hold fewer frames than the "
                        f"committed marker says exist ({short} < {expect_frames}). The "
                        f"marker only "
                        f"advances once every file has the row, so this is corruption, not an "
                        f"interrupted commit.")
                facts["uncommitted_rows"] = {i: n - expect_frames for i, n in counts.items()
                                             if n > expect_frames}
            # the committed prefix must agree across files
            common = min(counts.values()) if counts else 0
            limit = common if expect_frames is None else min(common, expect_frames)
            times = {index: tuple(seen["times_ps"][:limit]) for index, seen in frames.items()}
            if len(set(times.values())) > 1:
                problems.append(
                    "the state trajectories disagree about their committed frame times; they were "
                    "not written by one coordinated run")
        return {"problems": problems, "facts": facts}
