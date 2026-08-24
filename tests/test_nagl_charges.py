"""``ligand_charge_method='am1bcc_nagl'`` -- AM1-BCC charges from the graph, not from sqm.

AmberTools' ``am1bcc`` is a single-conformer AM1 calculation: about forty minutes for a 79-atom
macrocycle, and its answer depends on which conformer ETKDG happened to produce. ``am1bccelf10``
removes that dependence by averaging ten conformers, at ten times the cost. OpenFF NAGL reproduces
AM1-BCC ELF10 from the molecular graph alone -- no conformer is involved at any point -- in about a
second.

The thing worth testing is not that NAGL runs. It is that the *record* of what ran is complete: a
trained model can be upgraded underneath an unchanged configuration file, so the method name alone
does not identify a Hamiltonian. These tests pin the resolution rule (newest production model, never
a release candidate) and require the model file and its digest to reach the recorded provenance.
"""

from __future__ import annotations

import json

import pytest

from md_templates.openmm.config import DEFAULTS
from md_templates.openmm.system import NAGL_AM1BCC_METHODS, build_forcefield

pytest.importorskip("openff.nagl_models", reason="openff-nagl-models is not installed")

# Ethanol: large enough to have distinct chemical environments, small enough that the whole module
# runs in seconds. The properties under test are about resolution and provenance, not about size.
ETHANOL_SMILES = "CCO"


@pytest.fixture(scope="module")
def ligand_sdf(tmp_path_factory):
    """One conformer on disk, because `build_forcefield` takes a file."""
    rdkit = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = rdkit.AddHs(rdkit.MolFromSmiles(ETHANOL_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260824
    assert AllChem.EmbedMolecule(mol, params) == 0
    path = tmp_path_factory.mktemp("nagl") / "ethanol.sdf"
    writer = rdkit.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _config(method):
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg["forcefield"]["ligand_charge_method"] = method
    return cfg


# -------------------------------------------------------------------------------------------
# Model resolution
# -------------------------------------------------------------------------------------------
def test_the_resolved_model_is_a_production_release_not_a_candidate():
    """The bug this replaces: a hard-coded `openff-gnn-am1bcc-0.1.0-rc.3.pt`.

    `openff-nagl-models` ships an alpha and three release candidates alongside the release. Picking
    the newest file overall would prefer a pre-release the moment one is published, so the resolver
    asks for production models specifically.
    """
    from md_templates.openmm.system import resolve_nagl_am1bcc_model

    model = resolve_nagl_am1bcc_model()
    assert model["name"].endswith(".pt")
    assert "rc" not in model["name"] and "alpha" not in model["name"], model["name"]
    assert len(model["sha256"]) == 64


def test_the_resolver_returns_the_newest_production_model():
    """Whatever is installed, the resolver agrees with the package's own production listing."""
    from openff.nagl_models import get_models_by_type

    from md_templates.openmm.system import resolve_nagl_am1bcc_model

    expected = list(get_models_by_type("am1bcc", production_only=True))
    assert expected, "no production model installed; the resolver should have raised"
    assert resolve_nagl_am1bcc_model()["name"] == str(expected[-1]).rsplit("/", 1)[-1]


def test_the_digest_matches_the_file_on_disk():
    """The digest is the reproducibility claim, so it is checked against the bytes it describes."""
    from md_templates.openmm.hashing import sha256_file
    from md_templates.openmm.system import resolve_nagl_am1bcc_model

    model = resolve_nagl_am1bcc_model()
    assert model["sha256"] == sha256_file(model["path"])


# -------------------------------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", NAGL_AM1BCC_METHODS)
def test_both_spellings_are_accepted_and_record_the_model(method, ligand_sdf):
    """`am1bcc_nagl` is the preferred name; `nagl` is the older one and must keep working."""
    _, info = build_forcefield(_config(method), ligand_sdf=ligand_sdf, route="ligand")
    ligand = info["ligand"]
    assert ligand["charge_method"] == method
    assert ligand["nagl_model_file"].endswith(".pt")
    assert len(ligand["nagl_model_sha256"]) == 64


def test_the_two_spellings_resolve_to_the_same_model(ligand_sdf):
    """An alias that quietly meant a different model is exactly the bug being fixed."""
    records = [build_forcefield(_config(m), ligand_sdf=ligand_sdf, route="ligand")[1]["ligand"]
               for m in NAGL_AM1BCC_METHODS]
    assert len({r["nagl_model_sha256"] for r in records}) == 1


def test_the_charges_are_neutral_for_a_neutral_molecule(ligand_sdf):
    """NAGL's output is charge-constrained; a drifting sum would mean the wrong code path ran."""
    _, info = build_forcefield(_config("am1bcc_nagl"), ligand_sdf=ligand_sdf, route="ligand")
    assert info["ligand"]["formal_charge"] == 0
    assert abs(info["ligand"]["net_charge_e"]) < 1e-6


def test_am1bcc_does_not_claim_a_nagl_model(ligand_sdf):
    """The NAGL fields are specific to NAGL. Finding them on an sqm run would misdescribe it."""
    pytest.importorskip("openff.toolkit")
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

    names = [t.__class__.__name__ for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]
    if "AmberToolsToolkitWrapper" not in names:
        pytest.skip("AmberTools is not on PATH")
    _, info = build_forcefield(_config("am1bcc"), ligand_sdf=ligand_sdf, route="ligand")
    assert "nagl_model_file" not in info["ligand"]
    assert info["ligand"]["charge_scheme"] in ("am1bcc", "am1bccelf10")


def test_an_unsupported_method_names_the_supported_ones(ligand_sdf):
    """A rejection that does not say what to use instead makes the user guess."""
    with pytest.raises(ValueError) as excinfo:
        build_forcefield(_config("gasteiger"), ligand_sdf=ligand_sdf, route="ligand")
    message = str(excinfo.value)
    assert "gasteiger" in message
    for method in ("am1bcc", *NAGL_AM1BCC_METHODS):
        assert method in message, message
