# 0.6.1 worker handoffs

One report per worker session, owned by that session. The coordinator reads them and does not
edit them; a worker writes only its own.

A handoff records, for the work it covers:

- the base and contract SHA it worked from, and its current commit;
- the actual files changed, and any API or configuration change;
- the tests run, with the exact commands, the fixtures, the platform and precision, pass or fail,
  and where the evidence is;
- remaining failures, evidence that could not be obtained and why it is BLOCKED rather than
  passed, and the next dependent task.

A claim with no command behind it is not evidence. A skipped lane is not a pass, and a CPU run is
not CUDA evidence.
