# MD-tools

OpenMM tools for building systems, generating MD workflows, running them on CUDA, and registering
the result as a verified dataset.

One executable, `md-openmm`, and five commands:

```text
md-openmm build-top         a structure       -> built.xml + built.pdb + built.log
md-openmm build-md          a protocol config -> a <method>-run<N>/ run beside build/, min/ and input/
md-openmm md-run            an Amber-like .in -> a stage, a ladder, or AIS switching paths
md-openmm data-register     a finished tree   -> a verified dataset under $MD_DATA
md-openmm export-reference  a finished run    -> a bundle that runs on OpenMM alone
```

**Status:** `0.6.0`. Not on PyPI — install from the repository.

## Start here

<div class="grid cards" markdown>

* **[Installing](install.md)** — the NVIDIA driver, the conda environment, the package. Step 0
  needs root; the rest is a normal user install.

* **[Machine configuration](machine-configuration.md)** — one file per machine: which platform it
  runs on, and where registered data goes. CUDA is the default and there is no silent fallback.

* **[Methods](openmm_methods/README.md)** — cMD, REST2, AIS and umbrella sampling. Each
  page carries a runnable `example.config` and the commands that use it.

* **[Running](md-run.md)** — the Amber-like flags, the `.in` language, the MPI rules, and what
  each protocol writes.

</div>

## The shape of a run

Two commands to prepare, one script to run:

```bash
md-openmm build-top -i ALA.pdb \
    -os build/built.xml -op build/built.pdb -log build/built.log
md-openmm build-md  -odir ./cMD-run1 --config configs/md/cMD.config
cd cMD-run1 && ./run.sh
```

`build-top` decides the **physics** — force field, charges, radii, constraints — and writes
`built.xml`, which is the Hamiltonian. `build-md` decides the **experiment** — stage lengths,
intervals, the ladder. It never opens `built.xml`: the timestep is resolved at run time from the
masses actually serialised there, because a configuration that claims HMR is a request and the
System is the fact.

`build-top` accepts a `.pdb` or a `.seq` (one line of residue names, built by tleap) for a peptide,
or a `.smi` or `.sdf` for a single small molecule. A `.smi` is embedded with ETKDGv3 and
MMFF-minimised; a `.sdf` supplies its own coordinates and they are used **as given**, so a docked
or crystallographic pose survives.

Where the files land, and what is shared between runs on one system, is
[the run layout](run-layout.md).

## `md-tools` and `md-openmm`

| | what it is |
|---|---|
| **`md-tools`** | the **package**. `import md_tools` — inspectable implementations of the sampling techniques over OpenMM: Hamiltonian scaling, exchange, switching, restraints, reporting, the dataset contract. |
| **`md-openmm`** | the **executable**. An Amber-like front end that mimics `pmemd.cuda` and calls those modules. It adds no behaviour of its own. |

The simulation logic lives in `md-tools`; `md-openmm` is how you type it. A project building a new
sampling method imports the package.

## Reference

* [Configuration reference](md-configuration.md) — every key, generated from the schemas.
* [Scientific defaults](scientific-defaults.md) — every consequential default, its evidence, and
  the limits of that evidence.
* [Support matrix](support-matrix.md) — supported, experimental and unsupported combinations.
* [The dataset contract](data-contract.md) — the schema-level authority for records.

Nothing in this repository has been validated against experiment; every benchmark cited in the
rationale was run by someone else, on their systems, with their protocol.
