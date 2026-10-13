# Bespoke torsion fit: installing BespokeFit and MACE-OMOL, and fitting paracetamol

**Tested against md-tools `0.6.5` (development branch).** This page is the part the
[ligand-parameterization tutorial](index.md) leaves outside `build-top`: how to install the tools
that fit a molecule's torsions, and how to run the fit that produces the `.offxml` `build-top`
consumes. The fit ran on two GPUs; nothing else on this page needs one.

!!! warning "What this page does and does not claim"
    * `build-top` **never runs a fit** and never imports BespokeFit, QCEngine or MACE. They live in
      an environment you install and run yourself, which is why they are not in this package's
      `environment.yml`.
    * The install recipe below is the one the environment used for these runs was built from. **It
      was not re-created from scratch for this page**, and the MACE weights were already in the
      cache, so the download step is described from the library's behaviour and not re-observed.
    * Only the **MACE-OMOL-0** reference was run. `--reference mace-off23-*` is accepted by the
      script and untested; Psi4 is not offered here, because the environment's Psi4 crashed even on
      a trivial calculation when the proof of concept tried it.

## 1. Install the environment

The recipe is [`bespoke/environment.yml`](bespoke/environment.yml): OpenFF BespokeFit, QCEngine,
ForceBalance, geomeTRIC, TorsionDrive, AmberTools (for the AM1-BCC charges), PyTorch with CUDA, and
the pieces `mace-torch` needs.

```bash
conda env create -p <prefix>/bespokefit -f bespoke/environment.yml
<prefix>/bespokefit/bin/pip install --no-deps mace-torch==0.3.16
```

Three pins matter, and each one is a failure you would otherwise meet in the middle of a fit:

| pin | why |
|---|---|
| `qcengine` and `qcelemental` below 0.50 | BespokeFit 0.4.3 uses the v1 QCSchema models |
| `e3nn=0.4.4` | required exactly by `mace-torch` 0.3.16 |
| `openff-forcefields>=2024.09` | provides `openff_unconstrained-2.2.1`, the Sage 2.2.1 file BespokeFit fits against |

`mace-torch` is installed with `--no-deps` so that `pip` cannot replace the numpy that conda chose.
The versions in the environment these runs used:

```text
openff-bespokefit 0.4.3   openff-toolkit 0.16.5   qcengine 0.34.2   qcelemental 0.29.0
mace-torch 0.3.16         torch 2.5.1             e3nn 0.4.4        forcebalance 1.9.6
torsiondrive 1.2.0        geometric 1.1.1         openff-qcsubmit 0.57.0
```

### The MACE models

`mace-torch` downloads a model into `~/.cache/mace/` the first time it is asked for it. For
MACE-OMOL-0 that is a 422 MB file; the checkpoint these runs used has this sha256, which you can
compare against yours:

```text
9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469  MACE-omol-0-extra-large-1024.model
```

!!! note "Licences"
    The MACE-OFF23 weights are distributed under the Academic Software License, which does not
    permit commercial use. Read the licence of every model you download; this package does not
    redistribute any of them.

**MACE-OMOL-0 takes a charge and a spin, so it can evaluate a charged molecule. MACE-OFF23 cannot**
and the script refuses a molecule of net charge for it, naming the alternative.

## 2. The scripts

They are in [`bespoke/`](bespoke/fit_bespoke.py) beside this page, and run **under the environment's
python, not MD-tools'**; none of them imports `md_tools`.

| file | what it is |
|---|---|
| `fit_bespoke.py` | the driver: reads an MD-tools package, pins its charges, runs BespokeFit, writes the `.offxml` |
| `mace_adapter.py` | QCEngine adapters for MACE-OFF23 and MACE-OMOL |
| `qc_worker.py` | BespokeFit's QC worker with those adapters registered |
| `coordinator_once.py` | a BespokeFit coordinator that runs each stage once, and has a long HTTP timeout |

The adapters exist because QCEngine 0.34.2's own `mace` program fails against `mace-torch` 0.3.16
(`Configuration.__init__() missing 2 required positional arguments`). They evaluate through
`mace-torch`'s public ASE calculator, in float64, and put the workers on the GPUs when asked.
`coordinator_once.py` exists because, with the installed FastAPI, BespokeFit's coordinator starts
twice and submits every scan twice; the answer is unchanged and the cost is doubled. No installed
file is edited.

## 3. Run the fit

The input is the package `build-top --parameterize` wrote
([ligand-parameterization, section 2](index.md#2-a-ligand-paracetamol-with-sages-torsions)):

```bash
export PATH=<prefix>/bespokefit/bin:$PATH        # BespokeFit launches `redis-server`
python bespoke/fit_bespoke.py \
    --package parameter/ --reference mace-omol --out fit/ \
    --grid 15 --qc-workers 4 --qc-cores 2 --device cuda:auto
```

`--device cuda:auto` spreads the QC workers over the cards the process can see, taking the card with
the fewest workers first. `--device cpu` works, and is far slower. `--out` must not exist.

What it does, in order:

1. **Reads the package**: `molecule.sdf` for the molecule and `metadata.json` for the AM1-BCC
   charges, so the fit uses the numbers the package already has.
2. **Pins the charges.** BespokeFit's ForceBalance engine rebuilds every system without the
   molecule's charges, which with plain Sage means AM1-BCC is recomputed at every evaluation. The
   script writes the charges into the starting force field as a whole-molecule `LibraryCharges`
   entry, averaged over graph-symmetry classes (a whole-molecule SMIRKS matches once per
   automorphism), and **checks that the force field assigns exactly those numbers** before
   starting. For paracetamol the check gave a worst difference of `0.0` e.
3. **Scans.** A 1-D TorsionDrive on every non-terminal rotatable bond, 15° grid, each point relaxed
   by geomeTRIC with MACE-OMOL as the energy and gradient. The molecule is not fragmented.
4. **Fits.** ForceBalance, L1 prior of width 6.0, TorsionProfile targets attenuated above
   1 kcal/mol and cut at 10 kcal/mol, at most 10 iterations.

The result for paracetamol, on two cards with four workers:

```text
[✓] fragmentation successful
[✓] qc-generation successful
[✓] optimization successful
{"status": "success", "wall_time_s": 1397.3}
```

**23 minutes.** The proof of concept this script was adapted from took 8.3 minutes on three cards
with six workers, and with a patch that makes ForceBalance rebuild its system once per parameter set
and not once per grid point; that patch is not shipped here, so expect this to be slower.

### The fit reproduces the proof of concept

The script wrote `fit/bespoke.offxml`. It holds Sage 2.2.1 unchanged plus six added proper torsions
(`t182`–`t187`), and their force constants agree with the proof of concept's fit of the same
molecule to within 0.003 kcal/mol:

| torsion | proof of concept, k₁…k₄ (kcal/mol) | this script |
|---|---|---|
| t182 | 0.242, 2.101, 0.405, 0.0 | 0.242, 2.101, 0.407, 0.0 |
| t185 | 1.225, 1.295, 0.444, 0.0 | 1.225, 1.295, 0.442, 0.0 |
| t186 | 0.0, 1.003, 0.0, 0.134 | 0.0, 1.003, 0.0, 0.134 |

The two are independent fits and differ in the third decimal. BespokeFit's results depend on hash
ordering, and its workers are forked, so `fit_bespoke.py` re-executes itself with
`PYTHONHASHSEED=0` to make a repeat of the same run as reproducible as the tool allows; this page
did not repeat the run, so it does not measure that.

## 4. Use it

Hand the file to `build-top`, exactly as in the
[tutorial](index.md#3-the-same-ligand-with-mace-omol-refitted-torsions); it verifies the file before
it uses it:

```yaml
solute:
  kind: ligand
  parameters: generate
  torsion_refinement: mace-omol-bespokefit
  refinement_offxml: fit/bespoke.offxml
```

```bash
md-openmm build-top --parameterize -i paracetamol.smi --config para-bespoke.config \
    -op parameter-bespoke/TYL.pdb -os parameter-bespoke/TYL.xml -log parameterize-bespoke.log --resname TYL
```

The file must hold every Sage 2.2.1 proper torsion unchanged and add at least one. The package
records the method and the file's sha256, and a search never confuses it with a stock package.

## When the fit does not succeed

The script exits with status 2 and prints the status and error. The common cause is ForceBalance
reaching its iteration limit before the objective converges: BespokeFit then reports "errored" and
writes **no** force field, even if the objective had nearly stopped changing. Paracetamol converged
well inside the limit; a larger molecule can hit it. The proof of concept has a separate
`fit-continue` command that restarts ForceBalance from its own restart file (tested on folic acid);
it is not shipped here, so a fit that stops at the limit has to be continued with the proof of
concept's tooling or refitted with more iterations.

A scan that changes the molecule's bonding (a proton transfer, say) is a different problem: its
profile is not a torsion profile. The proof of concept detects it and refits with that bond left at
Sage's parameters; this script does not, and would fit to the broken scan.

## What was not checked

* Re-creating the environment from the recipe on a clean machine.
* Downloading the weights from nothing: they were in the cache already.
* `--reference mace-off23-*`, and any reference other than MACE-OMOL-0.
* A molecule other than paracetamol, and a charged one.
* The fit's quality. The held-out validation reported on the
  [tutorial page](index.md#3-the-same-ligand-with-mace-omol-refitted-torsions) was run on the proof of
  concept's fit with its validation tooling, which is not shipped; this script's fit has the same
  parameters, and was not validated separately.
