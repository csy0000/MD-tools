# Gate 4 (cross-engine): the softcore Hamiltonian against pmemd, at fixed coordinates

Produced 2026-09-19 by `scripts/s3_amber_crossengine.py` on work/0.7.0-hamiltonian. Development
evidence for milestone A2, **CPU**: this is AMBER cross-engine evidence and not CUDA evidence.

```bash
python scripts/s3_amber_crossengine.py --amberhome $AMBERHOME --out <dir> [--boundary unscaled] [--tail]
```

| | |
|---|---|
| AMBER | Amber 26 PMEMD, CPU build (`$AMBERHOME/bin/pmemd`), double precision; source `pmemd26.tar.bz2` sha256 `0478ccce892f3525e995e9c85458d552c6060b73dd28acd03c366e61ecf23a14` |
| OpenMM | 8.6.0.dev-c6173db, Reference platform (float64) |
| fixture | `tests/alchemy_s3_fixture.py`, `build(True, dispersion=False[, tail=True])`: H -> F on an ethane-like core, Cl-, Na+, a probe, 30 flexible TIP3P-like waters, 2.4 nm box |
| pmemd TI | `icfe=1, ifsc=1`, `scalpha=0.5, scbeta=12.0`, `gti_add_sc` = 1 (`scaled`) or 0 (`unscaled`), `timask`/`scmask` over a V0 and a V1 copy of the molecule, `ifmbar=1` at lambda 0, 0.25, 0.5, 0.75, 1 |
| path | the Amber18 diagonal, lambda_electrostatics = lambda_sterics = lambda_bonded = clambda |

**Matched, deliberately:** PME alpha (OpenMM's, 4.02498 nm^-1), a 100^3 grid on both (pmemd needs
2,3,5 factors; OpenMM had chosen 103), spline order 5, cutoff 0.9 nm, no long-range LJ correction
(`vdwmeth=0`, OpenMM dispersion off), no constraints, no net-force removal.
**Intentional differences, reported rather than hidden:** pmemd evaluates erfc from a table
(`eedmeth=1`, `eedtbdns=20000`; pmemd refuses exact erfc under PME), and reports the unscaled
terms on softcore atoms ("Softcore part", SC_EPtot) outside EPtot and the MBAR energies; the
driver adds them back for the absolute comparison.

**The acceptance rule** is `verdict()` in the driver. Calibration comes first: pmemd and OpenMM on
the ordinary single-copy end states, whose difference is the engines' shared discrepancy. A TI
difference is then held to the lambda-DEPENDENT part of that discrepancy, |cal_B - cal_A| plus two
pmemd print resolutions (1e-4 kcal/mol = 4.18e-4 kJ/mol), for dU/dlambda and for the shape of
U(lambda). A constant offset cancels in every free-energy quantity and is reported separately.
The `scaled` numbers were inspected once before the rule was written down. The `unscaled`,
`scaled_tail` and `unscaled_tail` runs were first compared under the written rule. The rule is
derived from print resolution and the plain end-state calibration, not from any TI number.

All energies are in kJ/mol.

| run | calibration A / B | worst dU/dlambda | worst U(lambda) shape | tolerance | worst absolute (tol) | verdict |
|---|---|---|---|---|---|---|
| `scaled` | 6.96e-03 / 7.05e-03 | 4.42e-04 | 3.09e-04 | 9.27e-04 | 7.03e-03 (7.89e-03) | PASS |
| `unscaled` | 6.96e-03 / 7.05e-03 | 8.74e-04 | 7.82e-04 | 9.27e-04 | 7.55e-03 (7.89e-03) | PASS |
| `scaled_tail` | 6.96e-03 / 9.87e-03 | 2.51e-03 | 2.28e-03 | 3.75e-03 | 9.85e-03 (1.07e-02) | PASS |
| `unscaled_tail` | 6.96e-03 / 9.87e-03 | 1.04e-03 | 6.18e-04 | 3.75e-03 | 9.64e-03 (1.07e-02) | PASS |

ParmEd writer round trip (single-copy end states through OpenMM's AmberPrmtopFile): A 1.0e-06, B 1.3e-06 kJ/mol.

What the tail runs show: pmemd's treatment of the appearing group's INTERNAL pairs and 1-4s agrees
with the Hamiltonian's (unscaled, the Ewald share removed from the weighted sum) within
calibration. A different convention there would shift U(lambda) by kJ/mol, not by 1e-3.

Margins: `unscaled` passes its slope check at 8.7e-4 against 9.3e-4, the thinnest margin, and
within pmemd's own print resolution of 4.2e-4. It is not evidence of agreement below about 1e-3
kJ/mol, and neither is any row here.

**Not covered here:** pmemd.cuda (the Amber18 GPU TI of R4), which needs a card that is not
allocated (BLOCKED); a dispersion correction; a complex leg; any sampling.

Files per run: `report.json` (every parsed number), the `.mdin` inputs, and the mdout at
clambda = 0.5 and of the end-state-A calibration. `scaled/` and `scaled_tail/` also hold the TI
`ti.parm7`/`ti.rst7`. Machine paths in the mdouts are replaced by `<run-dir>`.
