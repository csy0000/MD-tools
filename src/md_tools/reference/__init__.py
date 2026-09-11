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

    The physics is not re-derived. `system.xml` is the System the stage INTEGRATED -- produced by
    the engine's own preparation step, not by a second construction of it, and not the build file
    named on the command line, which carries neither the solute scaling nor the restraint force.
    What the exported script contains is only the part that was never a file: the integrator, the
    reporters and the loop.

    Every number in it is read from the run's own machine record rather than recomputed, so a
    bundle cannot claim a timestep or a seed the run did not use.

TWO EXPORTS, BECAUSE A LADDER IS NOT A STAGE

    `export_reference` writes a cMD stage: one Context, a fixed number of steps, a loop short
    enough that writing it out is honest.

    `export_rest2_reference` writes a REST2 ladder, and does NOT write the loop out. The modules
    that decide what happens -- the acceptance criterion, the odd/even sweep, the reduced
    potential, the seed derivation -- are copied verbatim into the bundle, because they already
    import nothing from this package. A reimplementation would be a second implementation of the
    physics, and the cMD export is on record as demonstrating what those do.
"""
from .export import export_reference
from .rest2_export import export_rest2_reference

__all__ = ["export_reference", "export_rest2_reference"]
