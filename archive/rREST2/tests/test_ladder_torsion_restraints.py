"""ARCHIVED with rREST2 (0.5.4). Moved out of tests/test_ladder_torsion_restraints.py unchanged; these depended on the
reservoir refresh and do not run against the current tree. See archive/rREST2/README.md.
"""


@pytest.mark.parametrize("protocol, reservoir, refused",
                         [("rREST2", True, True), ("REST2", False, False)])
def test_restraints_with_a_reservoir_are_refused(tmp_path, protocol, reservoir, refused):
    """An rREST2 reservoir sample carries no bias, so it cannot enter a restrained ladder.

    The refresh replaces the top rung's configuration with one drawn from a distribution generated
    WITHOUT the restraint. The rung then samples neither the biased nor the unbiased ensemble, and
    every output still looks well formed -- so the combination is refused rather than documented.
    """
    from md_tools.build.md import ConfigError, resolve_md_config

    # The unrestrained case must be REST2, not rREST2 with the reservoir off: rREST2 IS the
    # reservoir variant and refuses `enabled: false` for its own reason, which would make this
    # parameter pass for nothing to do with restraints.
    config = tmp_path / f"{protocol}.config"
    config.write_text(
        f"protocol: {protocol}\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 1}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 100, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 0}\n"
        "reporting: {crd_printout_solute: 50, info_printout: 50, checkpoint_printout: 50}\n"
        "collective_variables: {file: cv.yaml, interval_steps: 50}\n"
        "umbrella: {file: umbrella.yaml}\n"
        f"reservoir: {{enabled: {'true' if reservoir else 'false'}, path: reservoir}}\n"
        "rest2: {number_of_replicas: 2, tau_max: 0.5, exchange_interval_steps: 50, "
        "number_of_exchanges: 4}\n", encoding="utf-8")
    if refused:
        with pytest.raises(ConfigError, match="drawn from a distribution generated WITHOUT"):
            resolve_md_config(config)
    else:
        resolve_md_config(config)
