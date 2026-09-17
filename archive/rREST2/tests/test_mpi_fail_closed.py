"""ARCHIVED with rREST2 (0.5.4). Moved out of tests/test_mpi_fail_closed.py unchanged; these depended on the
reservoir refresh and do not run against the current tree. See archive/rREST2/README.md.
"""


def test_the_reservoir_declaration_is_written_by_rank_zero_alone():
    """Every rank used to write `reservoir.yaml`, then every rank read it.

    Eight processes truncating and rewriting one small YAML file while others parse it is an
    intermittent failure that looks like a corrupt configuration. It goes through the same
    rank-0-writes / everyone-verifies-the-digest path as the other helpers now, and the function
    that builds it returns TEXT so it cannot write anything by itself.
    """
    import inspect

    from md_tools.remd import generated

    source = inspect.getsource(generated.reservoir_declaration_text)
    for forbidden in ("write_text", "open(", "os.replace"):
        assert forbidden not in source, (
            f"reservoir_declaration_text writes to the filesystem ({forbidden}); it must return "
            f"text and leave the writing to the rank-0 helper path")
    assert "reservoir_file" in inspect.getsource(generated.replica_main)
