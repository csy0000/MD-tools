# Basics

What each command does, what each configuration key means, and what a finished run looks like on
disk. The [tutorials](../tutorial/README.md) are the steps; these pages are the explanations behind
them, and every tutorial links here rather than repeating them.

| | |
|---|---|
| [**build-top**](build-top/index.md) | one structure → a serialised System, the structure it matches, and a build record |
| [**Parameterising a ligand**](build-top/parameterization.md) | a molecule with no residue template → a reusable parameter package |
| [**build-md**](build-md/index.md) | a protocol configuration → the run scripts, the `.in` files and `run.sh` |
| [**Every configuration key**](build-md/configuration.md) | the generated reference for every key `build-md` accepts |
| [**md-run**](md-run.md) | the Amber-like surface: flags, what each one means, and what is refused |
| [**The run layout**](run-layout.md) | what a run writes, where, and which files belong to the system rather than the run |
| [**Collective variables**](collective-variables.md) | torsion reporting: the `cv.yaml` schema, cadences and outputs |
| [**Registration**](data-register/index.md) | turning a finished tree into a verified dataset |
| [**Ligand parameter packages**](ligand-packages.md) | what a package is, how it is identified, and how it is reused |
