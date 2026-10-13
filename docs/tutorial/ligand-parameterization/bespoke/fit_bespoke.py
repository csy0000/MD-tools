#!/usr/bin/env python
"""Fit one molecule's torsions with BespokeFit, against a MACE reference.

Run it with the python of the environment built from `environment.yml`, with that environment's
`bin` first on PATH (BespokeFit launches `redis-server`):

    PATH=<env>/bin:$PATH <env>/bin/python fit_bespoke.py \
        --package parameter/ --reference mace-omol --out fit/ --device cuda:auto

`--package` is an MD-tools ligand package (the directory `build-top --parameterize` writes). Its
molecule and its AM1-BCC charges are read from `molecule.sdf` and `metadata.json`; this script never
imports md_tools. The charges are pinned into the starting force field as a whole-molecule
LibraryCharges entry, so every evaluation, scan and fit uses exactly the package's numbers.

Writes `<out>/bespoke.offxml` (the starting force field plus the fitted torsions), `<out>/run.json`
and `<out>/summary.json`. Exit status: 0 success; 2 the fit did not succeed (BespokeFit writes no
force field when ForceBalance stops at its iteration limit before converging).

Settings follow the BespokeFit workflow of Horton et al., JCIM 2022 where this script does not state
otherwise: 1-D TorsionDrive on a 15-degree grid, geomeTRIC relaxation at every point, ForceBalance
with an L1 prior of width 6.0, TorsionProfile targets attenuated above 1 kcal/mol and cut at
10 kcal/mol. The molecule is not fragmented.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_FORCEFIELD = "openff_unconstrained-2.2.1.offxml"
GEOMETRIC = dict(coordsys="dlc", enforce=0.1, reset=True, qccnv=True, epsilon=0.0,
                 convergence_set="GAU", maxiter=300, trust=0.1, tmax=0.3)
REFERENCES = {
    "mace-omol": {"program": "mace-omol", "method": "extra_large", "label": "MACE-OMOL-0-extra_large"},
    "mace-off23-small": {"program": "mace", "method": "small", "label": "MACE-OFF23-small"},
    "mace-off23-medium": {"program": "mace", "method": "medium", "label": "MACE-OFF23-medium"},
    "mace-off23-large": {"program": "mace", "method": "large", "label": "MACE-OFF23-large"},
}


def _hash_seed():
    """BespokeFit's workers are forked, so the hash seed must be set before the interpreter starts."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def read_package(directory: Path):
    """The package's molecule (atom order = package order) and its partial charges."""
    from rdkit import Chem

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    mol = Chem.MolFromMolFile(str(directory / "molecule.sdf"), removeHs=False)
    if mol is None:
        raise SystemExit(f"{directory}/molecule.sdf could not be read")
    charges = [float(a["partial_charge_e"]) for a in metadata["atoms"]]
    if len(charges) != mol.GetNumAtoms():
        raise SystemExit("the package's charges and molecule disagree on the atom count")
    return mol, charges, metadata


def symmetrise(mol, charges):
    """Average the charges over graph-symmetry classes.

    A whole-molecule SMIRKS matches once per graph automorphism, so charges that differ between
    symmetry-equivalent atoms would make the assignment depend on match order. Raw AM1-BCC charges
    from one conformer are not exactly symmetric; the differences are numerical noise.
    """
    from collections import defaultdict

    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    groups = defaultdict(list)
    for index, rank in enumerate(ranks):
        groups[rank].append(index)
    out = list(charges)
    for members in groups.values():
        mean = sum(charges[i] for i in members) / len(members)
        for i in members:
            out[i] = mean
    return out, max(abs(a - b) for a, b in zip(out, charges))


def starting_force_field(mol, charges, name: str, out: Path):
    """Sage 2.2.1 (unconstrained) plus the molecule's charges as one LibraryCharges entry."""
    import numpy as np
    from openff.toolkit import ForceField, Molecule, unit

    offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    mapped = offmol.to_smiles(mapped=True)
    ff = ForceField(BASE_FORCEFIELD)
    ff.get_parameter_handler("LibraryCharges").add_parameter(
        {"smirks": mapped, "charge": [q * unit.elementary_charge for q in charges], "id": f"lib-{name}"})
    path = out / "start.offxml"
    ff.to_file(str(path))
    # The check that makes "pinned" a fact: the force field really assigns these numbers.
    import openmm

    system = ForceField(str(path)).create_openmm_system(offmol.to_topology())
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    got = np.array([nb.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
                    for i in range(nb.getNumParticles())])
    worst = float(np.max(np.abs(got - np.asarray(charges))))
    if worst > 1e-8:
        raise SystemExit(f"the starting force field does not assign the package's charges "
                         f"(worst difference {worst:.2e} e)")
    return path, mapped, worst


def build_factory(start: Path, reference: dict, grid: int, n_conformers: int):
    from openff.bespokefit.schema.optimizers import ForceBalanceSchema
    from openff.bespokefit.schema.smirnoff import ProperTorsionHyperparameters
    from openff.bespokefit.schema.targets import TorsionProfileTargetSchema
    from openff.bespokefit.schema.tasks import Torsion1DTaskSpec
    from openff.bespokefit.utilities.smirks import SMIRKSettings
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.qcsubmit.common_structures import QCSpec
    from openff.qcsubmit.procedures import GeometricProcedure
    from qcelemental.models.common_models import Model

    task = Torsion1DTaskSpec(
        program=reference["program"], model=Model(method=reference["method"], basis=None),
        grid_spacing=grid, scan_range=None, n_conformers=n_conformers,
        optimization_spec=GeometricProcedure(**GEOMETRIC))
    return BespokeWorkflowFactory(
        initial_force_field=str(start),
        fragmentation_engine=None,                    # the whole molecule is scanned and fitted
        default_qc_specs=[QCSpec(program=reference["program"], method=reference["method"], basis=None,
                                 spec_name=reference["label"].lower(),
                                 spec_description=f"{reference['label']} relaxed torsion scans",
                                 scf_properties=[])],
        optimizer=ForceBalanceSchema(max_iterations=10, penalty_type="L1"),
        parameter_hyperparameters=[ProperTorsionHyperparameters(priors={"k": 6.0})],
        target_templates=[TorsionProfileTargetSchema(
            calculation_specification=task, attenuate_weights=True,
            energy_denominator=1.0, energy_cutoff=10.0)],
        smirk_settings=SMIRKSettings(expand_torsion_terms=True, generate_bespoke_terms=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, type=Path, help="an MD-tools ligand package directory")
    parser.add_argument("--reference", choices=sorted(REFERENCES), default="mace-omol")
    parser.add_argument("--out", required=True, type=Path, help="a directory that does not exist yet")
    parser.add_argument("--grid", type=int, default=15, help="torsion grid spacing in degrees")
    parser.add_argument("--n-conformers", type=int, default=4)
    parser.add_argument("--qc-workers", type=int, default=2)
    parser.add_argument("--qc-cores", type=int, default=2, help="CPU cores per QC worker")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:N, or cuda:auto (spread the workers)")
    parser.add_argument("--port-offset", type=int, default=0,
                        help="added to BespokeFit's gateway and redis ports; keeps two fits apart")
    args = parser.parse_args()
    _hash_seed()

    reference = REFERENCES[args.reference]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mol, raw_charges, metadata = read_package(args.package)
    charges, symmetrisation = symmetrise(mol, raw_charges)
    net = round(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    if args.reference.startswith("mace-off23") and net != 0:
        raise SystemExit(f"MACE-OFF23 has no charge input and cannot evaluate a molecule of net charge "
                         f"{net:+d}; use --reference mace-omol")
    start, mapped, charge_check = starting_force_field(mol, charges, metadata["compound"]["id"], out)

    os.environ["BEFLOW_QC_COMPUTE_WORKER"] = "qc_worker"
    os.environ["BEFLOW_COORDINATOR_ROUTER"] = "coordinator_once:router"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])
    os.environ["BEFLOW_KEEP_TMP_FILES"] = "true"
    os.environ["BEFLOW_GATEWAY_PORT"] = str(8110 + args.port_offset)
    os.environ["BEFLOW_REDIS_PORT"] = str(6410 + args.port_offset)
    os.environ["BESPOKE_MACE_DEVICE"] = args.device
    if args.device == "cuda:auto":
        os.environ["MDT_BESPOKE_GPU_SLOTS"] = str(out / "gpu_slots")

    from openff.bespokefit.executor import BespokeExecutor, BespokeWorkerConfig, wait_until_complete
    from openff.toolkit import Molecule

    molecule = Molecule.from_mapped_smiles(mapped, allow_undefined_stereo=True)
    factory = build_factory(start, reference, args.grid, args.n_conformers)
    factory.to_file(str(out / "workflow-factory.json"))
    schema = factory.optimization_schema_from_molecule(molecule, index=metadata["compound"]["id"])

    began = time.time()
    with BespokeExecutor(
            n_fragmenter_workers=1, n_qc_compute_workers=args.qc_workers,
            qc_compute_worker_config=BespokeWorkerConfig(n_cores=args.qc_cores, max_memory=2.0),
            n_optimizer_workers=1, directory=str(out / "executor"), launch_redis_if_unavailable=True):
        result = wait_until_complete(BespokeExecutor.submit(schema), frequency=10)
    wall = time.time() - began

    (out / "output.json").write_text(result.json(indent=2))
    record = {
        "package": str(args.package.resolve()), "parameter_id": metadata["parameter_id"],
        "reference": reference, "base_forcefield": BASE_FORCEFIELD, "grid_degrees": args.grid,
        "status": str(result.status), "error": str(result.error) if result.error else None,
        "wall_time_s": round(wall, 1), "charges_symmetrisation_max_change_e": symmetrisation,
        "charge_check_max_abs_diff_e": charge_check, "device": args.device,
        "stages": [{"type": s.type, "status": str(s.status), "error": s.error} for s in result.stages]}
    (out / "run.json").write_text(json.dumps(record, indent=2))
    if result.bespoke_force_field is None:
        print(f"fit did not succeed: status {result.status}; error {result.error}", file=sys.stderr)
        return 2
    result.bespoke_force_field.to_file(str(out / "bespoke.offxml"))
    print(json.dumps({k: record[k] for k in ("status", "wall_time_s")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
