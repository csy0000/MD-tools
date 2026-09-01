# Test fixtures

| file | what it is |
|---|---|
| `ALA.pdb` | alanine dipeptide, the smallest system every route can build |
| `amber26_h_rem.log.save` | **genuine Amber output**, copied verbatim from AmberTools26's own test suite (`test/h_rem/rem.log.save`) |

`amber26_h_rem.log.save` is the grammar reference for `templates/rem_log.py`: the H-REMD log
format is asserted against Amber's own bytes rather than against a format inferred from
documentation, because a log cpptraj cannot parse is a log nobody can use.

It is vendored here rather than read from an AmberTools installation. It used to be read from an
absolute path on one workstation, which meant the test that decides whether our `rem.log` is
well formed silently skipped everywhere else -- including in CI, where AmberTools *is* installed
but its test suite is not laid out the same way. A 3.6 KB reference file is worth committing to
make that test run wherever the suite runs.
