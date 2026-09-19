"""`md-openmm combine-topology`: two ligand parameter packages and one environment -> a topology plan.

UNDER CONSTRUCTION (0.7.0). This is the command surface only. Everything it does is done by the
callable layer in `md_tools.alchemy.topology` and `md_tools.alchemy.topology_mapping`, and every
refusal there happens before anything is written:

    resolve both packages      ligands.catalog.resolve_package (the one package resolver)
    read the environment       alchemy.topology.Environment.from_files
    the atom map               AtomMap.from_pairs / AtomMap.from_record, or propose_map
    build and check the plan   alchemy.topology.build_topology_plan
    write it                   TopologyPlan.write (a NEW directory, staged and renamed)

The configuration says WHAT to combine; `-odir` on the command line says where the plan goes, as
it does for `build-md`. `--check` resolves and builds the plan in memory and creates nothing.

A map PROPOSED by `map: {automatic: true}` is written for review BESIDE the plan, as
`<odir>.map.yaml`, because a plan directory holds exactly the files its record names. That record,
given back as `map: {file: ...}`, reproduces the same plan (same `plan_sha256`).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .strict import ConfigError, Field, Schema, Section

__all__ = ["COMBINE_FORMAT", "COMBINE_SCHEMA", "MODES", "combine_topology"]

COMBINE_FORMAT = "md-tools-combine-topology/1"
MODES = ("single", "hybrid", "dual")


def _check(document: dict[str, Any]) -> None:
    mode = document["mode"]
    chosen = [key for key in ("file", "automatic")
              if document["map"][key] not in (None, False)]
    if len(chosen) != 1:
        raise ConfigError(
            "map: give exactly one of `file` (explicit pairs, or a stored map record) or "
            "`automatic: true` (a proposal, written beside the plan for review); got "
            f"{chosen or 'neither'}.")
    if document["map"]["automatic"] and mode == "single":
        raise ConfigError(
            "map.automatic is refused for mode: single. A single-topology map decides which atoms "
            "BECOME which, and that is stated, not proposed: give map.file.")
    if document["dual"]["restraint_k_kj_mol_nm2"] is not None and mode != "dual":
        raise ConfigError(
            f"dual.restraint_k_kj_mol_nm2 is set but mode is {mode}. It is the restraint that "
            f"keeps the two dual-topology ligands together, and no other mode has one.")


def _refuse_separated(document: Any) -> None:
    if isinstance(document, dict) and document.get("mode") == "separated":
        raise ConfigError(
            "mode: separated is not implemented. Separated topology is deferred beyond 0.7.0 "
            "and is not partially supported; use single, hybrid or dual.")


COMBINE_SCHEMA = Schema(
    "combine-topology.config",
    doc="Two registered ligand parameter packages, one environment and an atom map, combined into "
        "an alchemical topology plan by `md-openmm combine-topology`. UNDER CONSTRUCTION (0.7.0).",
    fields=[
        Field("format", str, enum=(COMBINE_FORMAT,),
              doc=f"Must be `{COMBINE_FORMAT}`."),
        Field("mode", str, enum=MODES,
              doc="How the two endpoints are represented: `single` (one evolving atom set), "
                  "`hybrid` (a mapped core plus endpoint-unique atoms) or `dual` (both ligands "
                  "whole, mutually excluded, held together by a restraint). `separated` is "
                  "deferred and refused by name."),
        Field("b_pose", str, default=None, nullable=True,
              doc="An .sdf holding endpoint B's pose. It must be B's chemical state; its atoms "
                  "are put into package order by the ligand module's own matcher, which refuses "
                  "anything else. Default: B's reference conformer superposed on the mapped A "
                  "atoms, with the fit's RMSD recorded. Relative to this file."),
    ],
    sections=[
        Section("endpoints", [
            Field("A", dict, doc="`{parameters: <compound>/param_<id> or a package path}`: the "
                                 "endpoint the environment already holds."),
            Field("B", dict, doc="`{parameters: ...}`: the endpoint it becomes."),
        ], required=True, doc="The two ligand parameter packages. Parameters are used exactly as "
                              "the packages record them; nothing is reparameterised."),
        Section("environment", [
            Field("system", str, doc="The built System holding endpoint A, e.g. build/built.xml."),
            Field("topology", str, doc="Its topology, e.g. build/built.pdb."),
            Field("ligand", dict, doc="A ligand selector naming exactly ONE residue: `{resname}` "
                                      "or `{chain, resid, insertion_code}`."),
        ], required=True, doc="ONE matched environment. Endpoint B never has its own box: two "
                              "independently solvated boxes are not atom-matched."),
        Section("map", [
            Field("file", str, default=None, nullable=True,
                  doc="Explicit pairs (`pairs: [[a, b], ...]` by package atom name or index), or "
                      "an atom-map record as `combine-topology` writes it, re-verified against "
                      "both packages."),
            Field("automatic", bool, default=False,
                  doc="Propose the map (`rdkit-fmcs-heavy/1`). A chemically ambiguous proposal is "
                      "refused, listing the alternatives. Refused for mode: single."),
        ], doc="The atom map from A to B. Exactly one of `file` or `automatic: true`."),
        Section("dual", [
            Field("restraint_k_kj_mol_nm2", (int, float), default=None, nullable=True,
                  minimum=0.0, unit="kJ/mol/nm^2",
                  doc="The restraint holding the two dual-topology ligands together. mode: dual "
                      "only; its default is the builder's."),
        ], doc="Dual-topology settings."),
        Section("ligand_catalog", [
            Field("path", str, default=None, nullable=True,
                  doc="A catalog directory searched FIRST for `<compound>/param_<id>` references, "
                      "before $MD_DATA/parameters/ligands. Relative to this file."),
        ], doc="Where package references are looked up, exactly as for build-top."),
    ],
    checks=[_check],
)


def _load(config_path: Path) -> dict[str, Any]:
    from .strict import load_yaml_strictly

    document = load_yaml_strictly(Path(config_path).read_text(encoding="utf-8"),
                                  source=str(config_path))
    _refuse_separated(document)
    return COMBINE_SCHEMA.resolve(document)


def _relative(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _b_pose_nm(path: Path, package):
    """Endpoint B's pose in package atom order, in nm -- only if the file IS the package's state.

    `ligands.build.order_like_package` is the one place a molecule is matched to a package's atom
    order; it refuses a different chemical state rather than guessing a correspondence.
    """
    from rdkit import Chem

    from ..ligands.build import order_like_package

    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None:
        raise ConfigError(f"b_pose: {path} could not be read as an SDF molecule")
    coordinates_angstrom, _ = order_like_package(mol, package, where="b_pose")
    return coordinates_angstrom / 10.0


def combine_topology(*, config_path: Path, out_dir: Path, check: bool = False,
                     echo=print) -> dict[str, Any]:
    """Resolve, build and (unless *check*) write one topology plan. Returns a summary.

    Nothing is written until the plan has been built and every check in the callable layer has
    passed; `check=True` stops there and creates nothing, not even the output's parent.
    """
    from ..alchemy.topology import Environment, TopologyError, build_topology_plan
    from ..alchemy.topology_mapping import AtomMap, MapError, propose_map
    from ..ligands.catalog import resolve_package
    from ..ligands.mapping import LigandSelector, MappingError
    from ..ligands.package import PackageError
    from .strict import load_yaml_strictly
    from .top import catalog_roots

    config_path = Path(config_path)
    out_dir = Path(out_dir)
    base = config_path.parent
    resolved = _load(config_path)
    sidecar = out_dir.parent / f"{out_dir.name}.map.yaml"
    if out_dir.exists():
        raise ConfigError(f"{out_dir} already exists. A plan is written once, into a new "
                          f"directory; choose another -odir.")
    if resolved["map"]["automatic"] and sidecar.exists():
        raise ConfigError(f"{sidecar} already exists; the proposed map is written there for "
                          f"review, and an existing file is never overwritten.")

    roots = catalog_roots(resolved, config_path)
    try:
        packages = {}
        for side in ("A", "B"):
            block = resolved["endpoints"][side]
            unknown = sorted(set(block) - {"parameters"})
            if unknown or "parameters" not in block:
                raise ConfigError(f"endpoints.{side}: exactly one key, `parameters`, is accepted; "
                                  f"got {sorted(block)}")
            packages[side] = resolve_package(str(block["parameters"]), roots=roots, base_dir=base)
        selector = LigandSelector.from_mapping(resolved["environment"]["ligand"],
                                               where="environment.ligand")
        environment = Environment.from_files(
            _relative(base, resolved["environment"]["system"]),
            _relative(base, resolved["environment"]["topology"]), selector)
        mode = resolved["mode"]
        report = None
        if resolved["map"]["automatic"]:
            atom_map, report = propose_map(packages["A"], packages["B"], mode)
        else:
            map_path = _relative(base, resolved["map"]["file"])
            document = load_yaml_strictly(map_path.read_text(encoding="utf-8"),
                                          source=str(map_path))
            if not isinstance(document, dict):
                raise ConfigError(f"{map_path}: expected a mapping")
            if set(document) == {"pairs"}:
                atom_map = AtomMap.from_pairs(packages["A"], packages["B"], document["pairs"])
            elif set(document) == {"map", "proposal"}:
                # A proposal this command wrote for review: the record is re-verified in full.
                atom_map = AtomMap.from_record(document["map"], packages["A"], packages["B"])
            else:
                atom_map = AtomMap.from_record(document, packages["A"], packages["B"])
        b_positions = None
        if resolved["b_pose"] is not None:
            b_positions = _b_pose_nm(_relative(base, resolved["b_pose"]), packages["B"])
        extra = ({"dual_restraint_k": float(resolved["dual"]["restraint_k_kj_mol_nm2"])}
                 if resolved["dual"]["restraint_k_kj_mol_nm2"] is not None else {})
        plan = build_topology_plan(packages["A"], packages["B"], atom_map, environment,
                                   mode=mode, b_positions_nm=b_positions, **extra)
    except (PackageError, MappingError, MapError, TopologyError, FileNotFoundError,
            ValueError) as refusal:
        if isinstance(refusal, ConfigError):
            raise
        raise ConfigError(f"combine-topology: {refusal}") from None

    summary = {"mode": mode, "package_a": packages["A"].reference,
               "package_b": packages["B"].reference, "plan_sha256": plan.sha256,
               "n_pairs": len(atom_map.pairs), "check": check, "out_dir": str(out_dir),
               "proposed_map": str(sidecar) if report is not None else None}
    echo(f"combine-topology: {mode} plan {packages['A'].reference} -> "
         f"{packages['B'].reference}, {len(atom_map.pairs)} mapped atom pairs, "
         f"plan_sha256 {plan.sha256[:16]}...")
    if check:
        echo("combine-topology: --check -- the plan builds and passes every check; nothing was "
             "written.")
        return summary

    plan.write(out_dir)
    if report is not None:
        record = {"map": atom_map.record(packages["A"], packages["B"]), "proposal": report}
        handle, staged = tempfile.mkstemp(prefix=f".{sidecar.name}-", dir=sidecar.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(f"# The atom map `combine-topology` PROPOSED for {out_dir.name}/.\n"
                         f"# Review it. Given back as `map: {{file: {sidecar.name}}}` it "
                         f"reproduces the same plan.\n")
            yaml.safe_dump(record, stream, sort_keys=False)
        os.replace(staged, sidecar)
        echo(f"combine-topology: proposed map written for review: {sidecar}")
    echo(f"combine-topology: plan written: {out_dir}")
    return summary
