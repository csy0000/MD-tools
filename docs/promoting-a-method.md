# Promoting a method to MD-tools

A project owes MD-tools nothing while it is being built. This document is about the one moment
when that stops being true: when a method that works here is put forward as something MD-tools
should carry.

Everything below is a gate, not a style guide. Each item exists because its absence has produced
a result that looked finished and was not.

## Before anything else: is it a technique at all?

Three kinds of thing live in this stack and only the first belongs upstream.

> **a BASIC TECHNIQUE** — how to sample at all, and nothing above it can build the same thing by
> composing what already exists. cMD, umbrella sampling, REST2, AIS. **MD-tools.**
>
> **a METHOD** — a way of sampling assembled FROM those basics. pREST2, LREX. **Stays here**, in
> `src/<method>/`, with the estimators it needs.
>
> **a STUDY** — what we sampled and what we concluded. **Stays here.**

Most work is the second kind. A method that can be expressed by composing existing techniques
should be, and the composition belongs in the project that needs it.

## The gate

### 1. The Hamiltonian must be recoverable as an object, not as a description

MD-tools' durability story rests on serialising the OpenMM `System` that was actually integrated.
If a method builds its Hamiltonian by mutating a Context at run time, or depends on state that
exists only inside a running process, it cannot be exported and it cannot be reproduced by anyone
who does not have your code.

The test is blunt: can the thing that was integrated be written to a file and deserialised
somewhere else into the same energies?

### 2. Its decision-making must be separable from its plumbing

An exported bundle carries the modules that DECIDE what happens — the acceptance rule, the
schedule, the seed derivation — copied byte for byte, and carries none of the plumbing: resume,
checkpointing, MPI coordination, provenance, failure handling.

That split only works if the deciding code imports nothing from the plumbing. In practice this
means the decision modules depend on the standard library, numpy and OpenMM, and on nothing else.
If your acceptance rule needs the driver in order to run, the bundle will have to reimplement it,
and a reimplementation is a second implementation that drifts silently.

### 3. It must be checked against the engine, not against itself

A second implementation of a sampler runs perfectly and samples something else. This is not
hypothetical: a reference exporter shipped carrying the *build* System instead of the integrated
one — an entirely different Hamiltonian at non-zero tau — and its test did not catch it, because
the test compared the export against a Context built in the test file, which shared every one of
the export's assumptions.

So the equivalence test compares against the engine's own output, and it compares the quantity
that cannot be accidentally right:

* for a single-Context method, the serialised final state, element for element;
* for a ladder, the state-to-walker mapping after every exchange.

Not an acceptance *rate*. A runner that swaps the wrong pairs produces a plausible rate.

Pin the thread count (`OPENMM_CPU_THREADS=1`) on both sides so the assertion can be equality
rather than a tolerance nobody can justify.

### 4. Every number must come from a record, never from a recomputation

An exported bundle reads its timestep, its seed and its step count from the run's own machine
record. Anything recomputed at export time is a second chance to be wrong — and the derived
integrator seed is `derive_seed(config_seed, stage_name)`, not the number in the config, which is
exactly the kind of difference that produces a plausible, reproducible, wrong trajectory.

### 5. It must refuse what it cannot carry

A bundle that quietly omits a bias, a reservoir or a CV definition still runs and samples
something else. Refusals happen **before anything is written**: a refusal that leaves a partial
directory behind is not a refusal.

### 6. The data must register

`data-register` must accept a finished run: machine records reporting completion, stage handoffs
verifiable by digest, and an inventory that re-reads. Register the whole run directory rather than
a hand-assembled subset — the lineage check is the reason, and a staged subset reports
`0 stage handoff(s) verified`.

## What to bring

| | |
|---|---|
| the technique | in `src/md_tools/...`, with the decision/plumbing split above |
| an equivalence test | against the engine, asserting the sharp quantity |
| an exporter | or a stated reason the method cannot have one yet |
| one finished run | registered, with its bundle, as evidence the whole path works |
| a release note | what it does, and what it deliberately does not do |

## What does NOT need to be true before this point

Everything else. While the method is being explored in `src/<method>/` it may ignore the dataset
contract, invent file formats, skip machine records and change signatures freely. The freedom is
the point: binding a project to an engine's contract means every experiment needs a contract
change before anyone finds out whether the idea was any good.
