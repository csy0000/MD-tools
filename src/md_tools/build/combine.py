"""`md-openmm combine-topology`: ligand packages and one environment -> a topology plan.

This is the command surface only. Everything it does is done by the callable layer in
`md_tools.alchemy.topology` and `md_tools.alchemy.topology_mapping`, and every refusal there
happens before anything is written:

    resolve the package(s)     ligands.catalog.resolve_package (the one package resolver)
    read the environment       alchemy.topology.Environment.from_files
    the atom map               AtomMap.from_pairs / AtomMap.from_record, or propose_map
                               -- a TRANSFORMATION only; `decoupling` has no second endpoint
    build and check the plan   alchemy.topology.build_topology_plan, or
                               build_decoupling_plan, which is where the net-charge refusal lives
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
#: `decoupling` is the only mode with ONE endpoint. Endpoint B is the ligand ABSENT -- not another
#: molecule -- so it takes no `endpoints.B`, no atom map and no `b_pose`, and each of those is
#: refused BY NAME rather than ignored. `md_tools.alchemy.topology.build_decoupling_plan` has
#: supported it since the plan builder existed; until 0.6.4 it was reachable only from Python, so
#: absolute hydration -- the simplest alchemical calculation there is -- could not be run from the
#: command line at all.
MODES = ("single", "hybrid", "dual", "decoupling")

#: The modes that transform one ligand into another, and so need a second package and a map.
PAIRED_MODES = ("single", "hybrid", "dual")


def _check(document: dict[str, Any]) -> None:
    mode = document["mode"]
    if mode == "decoupling":
        _check_decoupling(document)
        return
    if document["endpoints"].get("B") is None:
        raise ConfigError(
            f"endpoints.B is not set and mode is {mode}, which transforms one ligand into "
            f"another and so needs the ligand it becomes. Only mode: decoupling has one endpoint.")
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


def _check_decoupling(document: dict[str, Any]) -> None:
    """Every key that presupposes a second ligand is refused BY NAME, not ignored.

    Endpoint B here is the ligand ABSENT. A configuration that names a second package, an atom
    map or a B pose is not a decoupling with harmless extras: it is a mutation someone wrote and
    then set the wrong mode on, or a decoupling someone believes will transform something. Both
    are answered by saying which key does not belong, because a setting that is accepted and inert
    is worse than one refused.
    """
    for key, what in (("endpoints.B", "a second ligand package"),
                      ("b_pose", "a pose for a second ligand"),
                      ("map.file", "an atom map"),
                      ("map.automatic", "a proposed atom map"),
                      ("dual.restraint_k_kj_mol_nm2", "the dual-topology restraint")):
        section, _, field = key.partition(".")
        value = document[section].get(field) if field else document[section]
        if value not in (None, False):
            raise ConfigError(
                f"{key} is set but mode is decoupling, which names {what} nowhere: endpoint B is "
                f"the ligand ABSENT, not another molecule. Its own bonded terms, internal "
                f"exceptions and internal pairs stay physical at both ends, and the whole ligand "
                f"is the unique region. Remove {key}, or choose a mode that transforms one ligand "
                f"into another ({', '.join(PAIRED_MODES)}).")


def _refuse_separated(document: Any) -> None:
    if isinstance(document, dict) and document.get("mode") == "separated":
        raise ConfigError(
            "mode: separated is not implemented. Separated topology is deferred beyond 0.7.0 "
            "and is not partially supported; use single, hybrid, dual or decoupling.")


COMBINE_SCHEMA = Schema(
    "combine-topology.config",
    doc="A ligand parameter package and the environment holding it -- plus, for a transformation, a "
        "second package and an atom map -- combined into an alchemical topology plan by "
        "`md-openmm combine-topology`. Under `mode: decoupling` the second endpoint is the same "
        "ligand ABSENT, and there is no second package and no map.",
    fields=[
        Field("format", str, enum=(COMBINE_FORMAT,),
              doc=f"Must be `{COMBINE_FORMAT}`."),
        Field("mode", str, enum=MODES,
              doc="How the two endpoints are represented: `single` (one evolving atom set), "
                  "`hybrid` (a mapped core plus endpoint-unique atoms), `dual` (both ligands "
                  "whole, mutually excluded, held together by a restraint) or `decoupling` "
                  "(endpoint B is the ligand ABSENT -- no second package, no map, no pose). "
                  "`separated` is deferred and refused by name."),
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
            Field("B", dict, default=None, nullable=True,
                  doc="`{parameters: ...}`: the endpoint it becomes. Required by every mode that "
                      "transforms one ligand into another, and REFUSED by `decoupling`, whose "
                      "endpoint B is the ligand absent."),
        ], required=True, doc="The ligand parameter packages. Parameters are used exactly as "
                              "the packages record them; nothing is reparameterised."),
        Section("environment", [
            Field("system", str, doc="The built System holding endpoint A, e.g. build/built.xml."),
            Field("topology", str, doc="Its topology, e.g. build/built.pdb."),
            Field("record", str, doc="The build-top record of that build, e.g. build/built.log. "
                                     "It must describe these two files (its outputs sha256 are "
                                     "checked) and state the 1-4 scales the System applies to "
                                     "the ligand (nonbonded_compatibility), which are never "
                                     "inferred."),
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
    from ..alchemy.topology import (Environment, TopologyError, build_decoupling_plan,
                                    build_topology_plan)
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
        decoupling = resolved["mode"] == "decoupling"
        packages: dict[str, Any] = {"B": None}
        for side in ("A",) if decoupling else ("A", "B"):
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
            _relative(base, resolved["environment"]["topology"]), selector,
            record=_relative(base, resolved["environment"]["record"]))
        mode = resolved["mode"]
        report = None
        atom_map = None
        if decoupling:
            # ONE call, and it is the library's own entry point rather than build_topology_plan
            # with mode="decoupling" spelled out here: `build_decoupling_plan` is where the net
            # formal charge refusal lives, and reaching past it would lose that check.
            plan = build_decoupling_plan(packages["A"], environment)
        elif resolved["map"]["automatic"]:
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
        if not decoupling:
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

    # `package_b` and `n_pairs` are None for a decoupling, not "" and not 0. Zero mapped pairs is
    # a true statement about a mutation that shares nothing; a decoupling has no map to count, and
    # writing 0 would let a reader compare the two as though they were the same measurement.
    summary = {"mode": mode, "package_a": packages["A"].reference,
               "package_b": packages["B"].reference if packages["B"] is not None else None,
               "plan_sha256": plan.sha256,
               "n_pairs": len(atom_map.pairs) if atom_map is not None else None,
               "check": check, "out_dir": str(out_dir),
               "proposed_map": str(sidecar) if report is not None else None}
    if decoupling:
        echo(f"combine-topology: decoupling plan for {packages['A'].reference} -- endpoint B is "
             f"the ligand absent, plan_sha256 {plan.sha256[:16]}...")
    else:
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
