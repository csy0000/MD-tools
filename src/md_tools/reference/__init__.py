"""Export a finished run as a bundle that needs OpenMM and nothing from this package.

WHY THIS EXISTS

    A registered reference simulation outlives the tool that made it. Someone reading it in two
    years should be able to run it, or check their own implementation against it, with OpenMM
    alone -- not by installing a pinned MD-tools and hoping it still builds.

    The generated entry points in a run directory are the opposite of that. They are two lines:

        from md_tools.md import run_generated_stage
        raise SystemExit(run_generated_stage(__file__, "cMD"))

    which is right for a working directory -- a correction reaches every generated directory at
    once -- and useless as a durable artefact.

WHAT MAKES IT HONEST

    The physics is not re-derived. `system.xml` is the serialised OpenMM `System` the run actually
    integrated, copied byte for byte, so the Hamiltonian in the bundle IS the one that produced the
    data. What the exported script contains is only the part that was never in the file: the
    integrator, the reporters and the loop.

    Every number in it is read from the run's own machine record rather than recomputed, so a
    bundle cannot claim a timestep or a seed the run did not use.
"""
from .export import export_reference

__all__ = ["export_reference"]
