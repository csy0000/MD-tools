# 2026-08-27 — evidence-based scientific defaults, and the rationale that carries them

Instruction: `claudecode-instructions/20260827_evidence-based-scientific-defaults.md`.
Branch: `dev`. Version: `0.4.0.dev0` (unchanged).

## What this was

Six of the repository's public defaults were changed or made public, and every consequential
default was given a written, classified evidence base. No architecture changed: the six commands,
`sysgen.generate_system()` and `mdgen.generate_md()` are where they were, and nothing resembling a
profile registry, a schema layer or an inheritance system was introduced.

---

## 1. The audit, before anything changed

`show-default`/`sys-config` → resolved YAML → builder → serialized `System` → provenance was traced
for every value in the instruction. What that turned up:

### Duplicate and stale declarations

| where | what | disposition |
|---|---|---|
| `defaults.py` | `SOLVENTS`, `EXPLICIT_PROTEIN_FORCEFIELD`, ligand label, `padding_nm: 2.0`, `sys_defaults(solvent="OPC")`, `md_defaults(solvent="OPC")` | **the canonical public declaration**; changed here |
| `builder_defaults.py` `DEFAULTS` | `forcefield.protein/water/ligand`, `solvation.water_model`, `solvation.padding_nm: 2.0` | second copies of public values → now **imported** from `defaults.py` (`EXPLICIT_COMBINATIONS`, `DEFAULT_PADDING_NM`) |
| `builder_defaults.py` `DEFAULTS['equilibration']` | `barostat_interval: 50`, `timestep_fs`, `pressure_bar`, `nvt_ps`, `npt_ps`, release schedule, … | **dead** — nothing has read `cfg["equilibration"]` since the stage chain replaced the workflow manager. A stale `barostat_interval: 50` sitting beside the live 25 is exactly the second declaration this task is about. **Deleted.** |
| `builder_defaults.py` `DEFAULTS['integrator']` | `kind`, `timestep_fs`, `friction_per_ps`, `temperature_k` | **dead** (grep: no `cfg["integrator"]` anywhere). **Deleted.** |
| `builder_defaults.py` `DEFAULTS['production']` | platform, precision, report intervals, md/remd chunking | **dead**. **Deleted.** |
| `sysgen._legacy_cfg` | inline fallbacks `"OPC"`, `2.0`, `1.0` | now `DEFAULT_SOLVENT`, `DEFAULT_PADDING_NM` from `defaults.py` |
| `templates/md_stages.py` | `PRODUCTION_BAROSTAT_FREQUENCY = 25` | a copy of a public default living inside generated projects. **Deleted**; the value is now required from `stage.yaml` / `md.config.yaml` |
| `cli/md_openmm.py` | `--solvent default="OPC"` | now `D.DEFAULT_SOLVENT` |
| `README.md`, `docs/support-matrix.md`, `CLAUDE.md` | prose defaults | updated |

### Defects found while tracing, not asked for

1. **`forcefield.json` recorded `"method": "unknown"` for every explicit system.**
   `_nonbonded_record` read `reported.get("nonbonded")`, but `reported` is `build_forcefield`'s
   report (`xml`, `protein_forcefield`, `water`, `ligand`) while the nonbonded description lives one
   level up in `build_system`'s own record. Every explicit `forcefield.json` this repository has
   ever written says the electrostatics treatment is unknown. Fixed, and the function now **raises**
   rather than writing `"unknown"` for a System that certainly has a `NonbondedForce`.
2. **`forcefield.json` recorded a null HMR block for a repartitioned System.** Same shape of bug:
   `hmr` was looked for at the top level of the explicit record and in the force-field report, but
   the explicit route leaves it inside `record["omega"]`, which is `build_system`'s info dict.
   Reproduced by building with `hydrogen_mass_amu: 3.024` and reading the record. Fixed, and the
   record now refuses to be written at all if a repartitioning was requested and the builder
   reported none — the exact failure mode the repository already had once.
3. **`_water_xml` prefixed any unqualified water label with `amber19/`.** `tip3p.xml` would have
   become `amber19/tip3p.xml`, which OpenMM does not ship. Replaced with an explicit table that
   refuses an unknown label instead of inventing a resource name.
4. **`stage_run.py` imported `PRODUCTION_BAROSTAT_FREQUENCY` and never used it.**
5. `build_system` recorded `NonbondedForce.getNonbondedMethod()` as a **bare integer** and never
   recorded the switching distance.

---

## 2. What changed

### Explicit default, and the alternative

`defaults.EXPLICIT_COMBINATIONS` — a two-entry lookup, not a registry — pairs the protein force
field with the water model, because each pair is internally consistent and mixing across them
discards the reason either works:

| `--solvent` | protein | water |
|---|---|---|
| `TIP3P` (default) | `amber14-all.xml` → `amber14/protein.ff14SB.xml` | `amber14/tip3p.xml` |
| `OPC` | `amber19-all.xml` → `amber19/protein.ff19SB.xml` | `amber19/opc.xml` |

Ligand: `sage-2.2.1` → `openff-2.2.1`, standard AM1-BCC through AmberTools `sqm`, unchanged in
mechanism. The written YAML now carries the **qualified** water resource rather than a short label
that a magic prefix had to repair.

`amber14/tip3p.xml` was checked for the ion templates `addSolvent` places (`NA`, `CL`, `HOH` — all
present), which is why the qualified copy and not the top-level `tip3p.xml` is the default.

### Box

`DEFAULT_PADDING_NM = 1.5`, `CONSERVATIVE_PADDING_NM = 2.0`. Nothing about the geometry code
changed: the four distances were already computed and recorded separately, and the
cutoff/minimum-image gate in `solvation._resolve_box` is untouched. What changed is which number is
requested by default, and that the whole `geometry` dict now reaches `forcefield.json` and
`provenance.yaml` instead of stopping at `solvation.json` inside the deleted `_work/` tree.

### Barostat frequency

`common.barostat_frequency_steps: 25` is public, validated in `resolve_md_config` (positive whole
number of steps; refused as null under explicit solvent, written as null under implicit), carried
into every `stage.yaml` by `stage_plan`, and read by `stage_run.py`, `cmd_run.py`, `rest2_run.py`
and `rest2_equilibrate.py`. The generated scripts have **no fallback**: a project without the key
fails with a message naming it, rather than silently constructing a barostat at an interval no
record mentions.

### HMR

`check_timestep_against_masses` now refuses 4 fs for three reasons instead of one: no
repartitioning, unconstrained hydrogen bonds, or flexible water under explicit solvent. And
`verify_hmr_group_masses` builds the **same `createSystem` call with `hydrogenMass` omitted** and
compares each heavy-atom/hydrogen group's total mass. That reference build is the only way
conservation is actually checkable — a force field assigns its own masses, so the pre-HMR total
cannot be reconstructed from the topology afterwards. It runs only when HMR is requested.

### Implicit ligand scope

`implicit.gb_parameter_coverage` reads the built `CustomGBForce`'s per-particle parameters back and
counts the atoms carrying ParmEd's unfitted `(α, β, γ) = (1.0, 0.8, 4.85)` fallback, and checks
whether any residue mbondi3 adjusts is present at all. `forcefield_record._implicit_support_status`
turns that into `support_status`, `support_note` and `amber_igb8_parity_claimed`.

The two facts this rests on were read out of the installed libraries, not assumed:

* `parmed.structure.Structure._get_gb_parameters` (ParmEd 4.3.1) assigns fitted GB-Neck2 screen/α/β/γ
  for elements **{H, C, N, O, S}** and its own comment calls the rest "non-optimized values as
  defaults".
* `parmed.tools.changeradii.mbondi3` is `mbondi2` plus adjustments keyed on the residue names
  GLU/ASP/GL4/AS4/ARG and the atom name `OXT`.

Measured on a chlorinated benzamide (`Clc1ccc(cc1)C(=O)NC`): 18 of 19 atoms covered, chlorine not,
`mbondi3_reduces_to_mbondi2: true`, `support_status: "experimental"`,
`amber_igb8_parity_claimed: false`, and a warning printed by `sys-gen`. Measured on ACE-ALA-NME:
22 of 22 covered, `support_status: "supported"`, parity claimed.

### Records

`forcefield.json` gained: `protein.openmm_resource_includes` (a wrapper like `amber14-all.xml` is a
manifest, and naming it alone does not say which file carried the protein parameters), the full
nonbonded block, `explicit_solvent.box_geometry` (the four distances), `explicit_solvent.salt`, the
water packing substitution, `constraints.hmr_group_conservation`, and the implicit support status.
`inputs/provenance.yaml` gained `forcefield_summary` and the System's force classes and constraint
count. `MD/provenance.yaml` gained a `protocol` block. `resolved_stage.yaml` gained
`barostat_frequency_steps` and `barostat_interval_ps`; `resolved_run.yaml` gained `integrator` and
`pressure_coupling`, and each REST2 replica record gained its own barostat counts.

---

## 3. Verification

### Traced end to end, at the default

`sys-config` (no flags) → `sys.config.yaml` → `sys-gen` → `system.xml` + `forcefield.json` +
`provenance.yaml` → `md-gen` → `stage.yaml` × 4 + `MD/provenance.yaml` → CUDA run →
`resolved_stage.yaml` / `resolved_run.yaml`.

Read back off the **serialized System** and compared to the record:

```
nonbonded method code : 4  (NonbondedForce.PME = 4)      record: "PME"          agree
cutoff nm             : 1.0                              record: 1.0            agree
switching             : False                            record: false          agree
dispersion correction : True                             record: true           agree
Ewald tolerance       : 0.0005                           record: 0.0005         agree
box volume nm3        : 19.091883092036785               record: 19.09188309…   agree
barostats in System   : none (sys-gen adds none)
solute H masses       : {1.008}                          (no HMR at the default)
```

### Box clearance at the default padding

ALA, 1.5 nm requested:

```
padding_nm_requested          1.5
box_width_nm                  3.0       ( = max(2R + padding, 2*padding), R = 0.47329 )
shortest_lattice_translation  3.0
solute_image_clearance_nm     2.05343
min_reduced_box_height_nm     2.12132   ( = width / sqrt(2) )
required_cutoff_height_nm     2.1       ( = 2*1.0 + 0.1 margin )
grown_for_cutoff              false     -> the gate was reached and passed without growing
max_legal_cutoff_nm           1.06066
```

### Barostat, on CUDA

```
minimization    barostats_in_system 1  active 0  freq 0   interval_ps None
eq/nvt_1kcal    barostats_in_system 1  active 0  freq 0   interval_ps None
eq/npt_1kcal    barostats_in_system 1  active 1  freq 25  interval_ps 0.05
eq/npt_free     barostats_in_system 1  active 1  freq 25  interval_ps 0.05
cMD             pressure_coupling {frequency_steps: 25, interval_ps: 0.05}
REST2           pressure_coupling {frequency_steps: 25, interval_ps: 0.05}, each replica active 1
implicit        barostat_frequency_steps: null in every stage; no barostat Force in the System
```

### HMR, on CUDA

`docs/examples/hmr-4fs.yaml` applied to the default combination:

```
solute masses      {3.024, 5.962, 9.994, 11.994, 12.01, 16.0}   (hydrogens at the target)
water H masses     {1.0079}                                     (water untouched)
total mass amu     10854.066512 before == 10854.066512 after
groups checked     6, max group mass change 0.0 amu
lightest heavy     5.962 amu  (> 0)
4 fs chain         minimization + 3 eq stages on CUDA, timestep_fs 4.0, barostat interval 0.10 ps
```

### Tests

`196 passed in 157 s` on `CUDA_VISIBLE_DEVICES=0,1,2,3` (RTX A5000 + RTX 3080), OpenMM 8.6.0.dev-c6173db.
`120 passed, 76 deselected` for `-m "not gpu and not slow"`.

New tests: `tests/test_scientific_defaults.py` (14) plus additions to
`tests/test_config_generation.py` and `tests/test_fair_provenance.py`. Every test that minimises or
integrates carries `gpu` and runs on CUDA; the box-geometry tests build no System and are unmarked.

---

## 4. The rationale document

`docs/md-defaults-scientific-rationale.md`, `.pdf`, and `docs/md-defaults-references.bib`.

**The `academic-research` skill named in the instruction is not installed in this environment.**
`~/.claude/plugins` carries only `claude-mem@thedotmack`, and no plugin in the official marketplace
catalogue provides an `academic-research` skill. Rather than invent citations, the literature work
was done directly and every step is reproducible:

* **DOI verification:** every DOI resolved against the **Crossref REST API**
  (`https://api.crossref.org/works/<doi>`) on 2026-08-27; the title, author list, journal, volume,
  issue and pages in the `.bib` are the values Crossref returned. Where Crossref's `issued` date is
  the ACS ASAP posting date rather than the issue date (ff19SB, GB-Neck, OpenMM 8, the ParmEd
  paper, the 2013 Leimkuhler AMRX paper), the `.bib` carries the **issue** year and a `note`
  recording the discrepancy. The Zenodo dataset DOI is not in Crossref and was verified against
  **DataCite** (`https://api.datacite.org/dois/…`), which reports it as resourceType `Dataset`.
* **Link resolution:** 26 DOIs and 3 URLs checked. 24 DOIs followed through to a live publisher
  page (many answering `403` to a HEAD from a script, which still proves the DOI resolved and to
  where); the two Elsevier DOIs could not be followed because `linkinghub.elsevier.com` does not
  resolve from this machine's DNS, so they were confirmed instead against the **DOI handle registry**
  (`https://doi.org/api/handles/<doi>`, `responseCode 1`, correct target). All 3 URLs returned 200.
* **Primary-source reading:** the Sage 2.0.0, ff14SB, ff19SB and GB-Neck2 papers were read in full
  text (Europe PMC `fullTextXML` for Sage and for Hahn 2024; PMC for ff14SB, ff19SB and GBn2) and
  the claims in §3, §5 and §6 are quoted from them. The Sage benchmark's protein force field is
  quoted verbatim, which is what the instruction asked to get right.
* **Implementation facts** — OpenMM's barostat default frequency, the LFMiddle/BAOAB relationship,
  ParmEd's GB parameter table, the mbondi3 rules, the Sage 2.2.0/2.2.1 release notes and the
  installed offxml date — were read from the **installed libraries** on this machine, not from a
  web page.
* **Citation completeness:** 29 bib entries, 29 distinct keys cited, 74 citation markers / 91 cite
  tokens in the Markdown. Nothing cited is missing from the bibliography and nothing in the
  bibliography is uncited (checked by script).

### Corrections the document makes explicitly

* The Sage 2.0 protein–ligand benchmark used **`ff99SB*-ILDN` with TIP3P**, quoted from the paper —
  not ff14SB and not ff19SB.
* Sage was **not** fitted to protein-binding affinities: binding free energies appear as a benchmark
  "to ensure the refit did not adversely affect performance", while the valence parameters were fit
  to QCArchive QM and the LJ parameters to condensed-phase densities and enthalpies of mixing.
* Sage's **complete parameter set was not fitted against TIP3P**: TIP3P entered only through the
  aqueous subset of the physical-property training data, the water model was not refit, and the
  valence parameters involve no water model at all.

### Rendering, and how the PDF was inspected

No pandoc, no LaTeX, no HTML-to-PDF engine exists on this machine. Installing a documentation
toolchain to render one document would add a dependency the six commands do not need, so the PDF is
produced by `scripts/render_markdown_pdf.py` — a ~330-line renderer over **ReportLab 5.0.0**, which
is already in the scientific environment.

```
python scripts/render_markdown_pdf.py \
    docs/md-defaults-scientific-rationale.md docs/md-defaults-scientific-rationale.pdf
```

Tool versions: Python 3.12.13, reportlab 5.0.0, DejaVu Sans from
`/usr/share/fonts/truetype/dejavu`, poppler `pdftoppm`/`pdftotext`/`pdfinfo` 24.02.0.

**Rendering defects found by inspecting the rasterised pages, and fixed:**

1. **Dropped glyphs.** ReportLab's built-in Helvetica is WinAnsi-encoded and *silently drops* a
   character outside that encoding — the `→` in the defaults table vanished, and the PDF looked
   correct until it was read beside the Markdown. Fixed twice over: a Unicode TrueType family is
   registered when one is on the machine, and `check_glyph_coverage` now consults the font's own
   cmap for every character in the document and **fails the render** rather than producing a PDF
   with missing text. It caught α, β, γ, √ and ⁴ on the first run, which is how the first fix was
   found to be incomplete (only `DejaVuSans-Oblique.ttf` was missing from this packaging, and the
   original all-or-nothing registration had therefore fallen back to Helvetica entirely).
2. **Mid-word column breaks.** `electrostatics` rendered as `electrosta / tics`. Column widths were
   proportional to content length with no floor for the longest *unbreakable token*; now each
   column's floor is measured in points from its longest word with `stringWidth`, capped at 30% of
   the frame so one long path cannot starve four other columns.
3. **Backslash escapes rendered literally.** `HF/6-31G\*` came out as `HF/6-31G\`. The renderer now
   honours Markdown backslash escapes, and the source uses code spans for the QM basis sets.
4. **A bare dark stripe** where a `| | |` metadata table had an empty header row. An empty header is
   now dropped rather than styled.
5. Ordered-list markers lacked their period.

**Visual inspection:** all 12 pages rendered to PNG at 110 dpi with `pdftoppm -r 110 -png` and each
image examined individually — not sampled. Checked for: text running past the frame, missing
glyphs, blank pages, tables overflowing the page width or breaking mid-word, citation markers
rendering as raw `[@key]`, and headings orphaned from their section. Text extraction with
`pdftotext -layout` was compared against the Markdown as a second check that nothing was dropped
silently. Final state: **12 pages** (≈11 of body, §15 sources on the last), no overflow, no blank
page, every glyph covered.

---

## 5. Limitations that remain

1. **The specific triple ff14SB + Sage 2.2.1 + TIP3P has no published benchmark under its own
   version numbers.** §3.3 and §12.1 of the rationale label this an extrapolation and say from what.
2. **ff19SB + Sage 2.2.1 + OPC has no joint benchmark at all.** It is offered as two internally
   consistent halves, and the document says so.
3. **The implicit ligand route stays experimental.** It is not disabled: it builds, it is recorded
   as experimental, the unfitted atoms are counted and named, and no `igb=8` parity is claimed. A
   future task could add a numerical comparison against explicit solvent on a small ligand set,
   which is the evidence that would change the label.
4. **1.5 nm is not a guarantee about a conformation that has not happened yet.** The enforced
   guarantee is the built-box cutoff gate, which is a different and weaker statement.
5. **HMR + 4 fs is validated here only for stability**, over picoseconds. Nothing about kinetics.
6. **Two ff14SB conversions exist** — OpenMM's `amber14/protein.ff14SB.xml` (used here, through
   `amber14-all.xml`) and openmmforcefields' `amber/ff14SB.xml` (used by OpenFE). They were not
   compared parameter-by-parameter and no identity is claimed.
7. **`scripts/render_markdown_pdf.py` needs ReportLab**, which is not a declared dependency of this
   package. It is a `scripts/` tool, it fails with a message naming the package, and the Markdown
   stays authoritative — but a checkout without ReportLab cannot rebuild the PDF.
8. **The `academic-research` skill was unavailable**, so the literature work was done by direct
   Crossref/DataCite verification and full-text reading. Every step is recorded above and is
   repeatable, but it is not the workflow the instruction named.
9. **`--error-on-skip` was not used** for the acceptance run; the suite was run with a working CUDA
   platform and reported 196 passed, 0 skipped.

---

## 6. Not done, deliberately

* No nanosecond run. No cMD or REST2 campaign was repeated: this was a defaults and documentation
  task, and the instruction says so.
* No new public command, no profile registry, no schema layer, no inheritance.
* No retrofit of 0.3.x data. Two tests assert that a 0.3.x tree recording ff19SB/OPC at 2.0 nm comes
  back out saying exactly that, and that a tree recording nothing comes back `unknown` rather than
  acquiring a 0.4 default.
* `docs/implementation/explicit_solvent/` still describes ff19SB/OPC/Sage 2.2 as defaults. Those
  files already carry a **HISTORICAL — not current usage** banner and describe the pre-`ca29fcd`
  architecture; rewriting them would be rewriting history.
