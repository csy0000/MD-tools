# MD-tools documentation

Small on purpose. These documents are authoritative for the current package; anything else is
history.

| document | what it answers |
|---|---|
| [`data-contract.md`](data-contract.md) | what a dataset is, where it goes, and who owns which part of the FAIR boundary |
| [`replica-exchange.md`](replica-exchange.md) | the REST2 / rREST2 scientific contract |
| [`scientific-defaults.md`](scientific-defaults.md) | why each default is what it is, with the evidence |
| [`scientific-defaults.bib`](scientific-defaults.bib) | the references that document cites |
| [`support-matrix.md`](support-matrix.md) | versions and combinations this package is tested against |
| [`release-notes/`](release-notes/) | what changed in each release, and what was verified |

The commands themselves, the configuration layout and the three-command workflow are in the
[root README](../README.md). The shipped configuration examples are in
[`configs/`](../configs/) and are meant to be read.

## For agents

**Do not search Git history to answer a question about how this package works today.** These
documents and the code are the answer. History contains a large amount of superseded material —
retired commands (`sys-gen`, `md-gen`, `sys-config`, `setup`, `openmm-md`), a v1 dataset contract
with a `{namespace}/{yyyy-mm}/{dataset_name}` path, and an environment installer — none of which
exist any more. Reading it as current guidance produces confidently wrong instructions.

Search history only when the question is explicitly historical: what a released version did, or
why something was changed. The v0.5.0 migration and everything it removed are summarised in
[`release-notes/v0.5.0.md`](release-notes/v0.5.0.md), and the state immediately before that cleanup
is preserved at the tag `pre-v0.5-doc-cleanup`.
