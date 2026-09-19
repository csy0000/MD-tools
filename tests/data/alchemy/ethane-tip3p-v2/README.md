# ethane in TIP3P, version 2

`../v1/ethane-tip3p/` is a 1.9 nm cube at a 0.9 nm cutoff, 2.8% above twice the cutoff. In 1 ns
NPT windows (S4's M2 campaign, on CUDA) its box fluctuated to a 1.808 nm edge and OpenMM stopped
every water repeat ("periodic box size has decreased to less than twice the nonbonded cutoff").
v1 is unchanged; this is a new version with room to breathe.

Same ethane package (`../v1/packages/LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4`), TIP3P, PME with a
0.9 nm cutoff, dispersion correction, HBonds, rigid water, no ions, cube -- with
`padding_nm: 1.35`, which gives a **2.7 nm edge**: 2 x cutoff + 0.9 nm, above the required
2 x cutoff + 0.8 nm. 1868 particles (620 waters). `test_the_v2_box_leaves_room_for_npt_fluctuation`
asserts the margin from the System itself, so it cannot regress silently.

Made on 2026-09-19 at `0b831fc` with `MD_DATA` an empty temporary root (still empty
afterwards), the ethane package reused from a path:

```bash
md-openmm build-top -i input/ETA.sdf --config build/build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

`built.log` is the build's record with `environment.hostname`, `environment.user` and the
invoked `md_openmm.py` path replaced by `<redacted>`; nothing else differs, and its `outputs`
sha256 match `built.xml` and `built.pdb`.
