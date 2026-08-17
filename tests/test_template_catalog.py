"""The root registry and the template descriptors.

Every negative case builds its own catalog in a temporary directory. Nothing here reads or mutates
the developer's checkout, so the suite means the same thing on a clean tree and a dirty one.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_templates.core import (
    REGISTRY_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION,
    RegistryError,
    TemplateError,
    load_catalog,
    parse_descriptor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_ID = "conventional-md/openmm/explicit-water"
REST2_ID = "rest2/openmm/explicit-water"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    """A byte-for-byte copy of the shipped catalog, safe to mutate.

    Every path the descriptors reference is recreated as an empty stand-in of the same kind, so the
    registry's reference-existence check has something to find without the fixture copying the
    repository. The stand-ins are empty by design: this suite tests the catalog, not the contents of
    what it points at.
    """
    (tmp_path / "registry.yaml").write_bytes((REPO_ROOT / "registry.yaml").read_bytes())
    for tid in (MD_ID, REST2_ID):
        src = REPO_ROOT / "templates" / tid / "template.yaml"
        dst = tmp_path / "templates" / tid / "template.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        for ref in read_yaml(src).get("repository_references", []):
            target = tmp_path / ref
            if (REPO_ROOT / ref).is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
    return tmp_path


def mutate_registry(root: Path, fn) -> None:
    doc = read_yaml(root / "registry.yaml")
    fn(doc)
    (root / "registry.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def mutate_descriptor(root: Path, template_id: str, fn) -> None:
    path = root / "templates" / template_id / "template.yaml"
    doc = read_yaml(path)
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


# ================================================================================================
# Registry
# ================================================================================================

def test_1_valid_registry_loads(catalog_dir):
    catalog = load_catalog(catalog_dir)
    assert catalog.registry.schema_version == REGISTRY_SCHEMA_VERSION
    assert catalog.canonical_url == "https://github.com/csy0000/MD-templates"
    assert sorted(e.template_id for e in catalog.entries) == sorted([MD_ID, REST2_ID])
    assert catalog.get(MD_ID).method_aliases == ["md"]
    assert catalog.get("templates/rest2/openmm/explicit-water/template.yaml").engine == "openmm"


def test_shipped_catalog_in_the_real_repository_validates():
    catalog = load_catalog(REPO_ROOT)
    assert sorted(catalog.descriptors) == sorted([MD_ID, REST2_ID])


def test_registry_records_no_commit_sha():
    """A SHA here would be stale the instant it was committed."""
    text = (REPO_ROOT / "registry.yaml").read_text(encoding="utf-8")
    import re

    assert not re.search(r"\b[0-9a-fA-F]{40}\b", text)
    assert not re.search(r"\b(commit_sha|sha|revision)\s*:", text)


def test_2a_unknown_root_field_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d.update({"maintainer": "someone"}))
    with pytest.raises(RegistryError, match="maintainer"):
        load_catalog(catalog_dir)


def test_2b_unknown_nested_field_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][0].update({"engine_version": 2}))
    with pytest.raises(RegistryError, match="engine_version"):
        load_catalog(catalog_dir)


def test_2c_unknown_field_inside_repository_block_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["repository"].update({"mirror": "https://x.example"}))
    with pytest.raises(RegistryError, match="mirror"):
        load_catalog(catalog_dir)


@pytest.mark.parametrize("version", [0, 2, 99, "1", 1.0, True, None])
def test_3_unsupported_schema_version_fails(catalog_dir, version):
    mutate_registry(catalog_dir, lambda d: d.update({"schema_version": version}))
    with pytest.raises(RegistryError, match="schema_version"):
        load_catalog(catalog_dir)


def test_4_duplicate_template_id_fails(catalog_dir):
    def dup(d):
        clone = copy.deepcopy(d["templates"][0])
        clone["template_path"] = "templates/conventional-md/openmm/other/template.yaml"
        d["templates"].append(clone)

    mutate_registry(catalog_dir, dup)
    with pytest.raises(RegistryError, match="duplicates"):
        load_catalog(catalog_dir)


def test_5_duplicate_template_path_fails(catalog_dir):
    def dup(d):
        clone = copy.deepcopy(d["templates"][0])
        clone["template_id"] = "conventional-md/openmm/other"
        d["templates"].append(clone)

    mutate_registry(catalog_dir, dup)
    with pytest.raises(RegistryError, match="duplicates"):
        load_catalog(catalog_dir)


def test_6_missing_descriptor_fails(catalog_dir):
    (catalog_dir / "templates" / REST2_ID / "template.yaml").unlink()
    with pytest.raises(RegistryError, match="does not exist"):
        load_catalog(catalog_dir)


@pytest.mark.parametrize("bad,reason", [
    ("/templates/rest2/openmm/explicit-water/template.yaml", "absolute"),
    ("C:/templates/rest2/openmm/explicit-water/template.yaml", "absolute"),
    ("templates/rest2/openmm/../openmm/explicit-water/template.yaml", "escapes"),
    ("../templates/rest2/openmm/explicit-water/template.yaml", "escapes"),
    ("templates/./rest2/openmm/explicit-water/template.yaml", "'.' component"),
    ("templates//rest2/openmm/explicit-water/template.yaml", "empty path component"),
    ("templates\\rest2\\openmm\\explicit-water\\template.yaml", "backslash"),
    ("templates/rest2/openmm/explicit-water/template.yaml/", "trailing separator"),
])
def test_7_to_10_path_rules(catalog_dir, bad, reason):
    """Absolute, escaping, dot, empty-component, Windows-separator and trailing-slash paths."""
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update({"template_path": bad}))
    with pytest.raises(RegistryError, match=reason.replace(".", r"\.").replace("'", "'")):
        load_catalog(catalog_dir)


def test_path_must_stay_under_templates(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update(
        {"template_path": "src/rest2/openmm/explicit-water/template.yaml"}))
    with pytest.raises(RegistryError, match="must live under templates/"):
        load_catalog(catalog_dir)


def test_descriptor_filename_must_be_template_yaml(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update(
        {"template_path": "templates/rest2/openmm/explicit-water/descriptor.yaml"}))
    with pytest.raises(RegistryError, match="must end in template.yaml"):
        load_catalog(catalog_dir)


def test_11_template_id_path_mismatch_fails(catalog_dir):
    """The ID must equal the descriptor's directory relative to templates/."""
    root = catalog_dir / "templates" / "rest2" / "openmm" / "elsewhere"
    root.mkdir(parents=True)
    (root / "template.yaml").write_bytes(
        (catalog_dir / "templates" / REST2_ID / "template.yaml").read_bytes())
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update(
        {"template_path": "templates/rest2/openmm/elsewhere/template.yaml"}))
    with pytest.raises(RegistryError, match="must equal the descriptor directory"):
        load_catalog(catalog_dir)


def test_12_method_path_mismatch_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update({"method": "conventional-md"}))
    with pytest.raises(RegistryError, match="declared method"):
        load_catalog(catalog_dir)


def test_13_engine_path_mismatch_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update({"engine": "gromacs"}))
    with pytest.raises(RegistryError, match="declared engine"):
        load_catalog(catalog_dir)


def test_registry_and_descriptor_must_agree(catalog_dir):
    """An internally consistent descriptor that contradicts its registry entry is still refused."""
    def relabel(doc):
        doc["method"]["id"] = "conventional-md"
        doc["template_id"] = "conventional-md/openmm/explicit-water"

    mutate_descriptor(catalog_dir, REST2_ID, relabel)
    with pytest.raises(RegistryError, match="disagrees"):
        load_catalog(catalog_dir)


def test_a_self_inconsistent_descriptor_is_refused_on_its_own_terms(catalog_dir):
    mutate_descriptor(catalog_dir, REST2_ID, lambda d: d["method"].update({"id": "conventional-md"}))
    with pytest.raises(RegistryError, match="must begin with"):
        load_catalog(catalog_dir)


@pytest.mark.parametrize("field,value", [
    ("method_aliases", ["MD"]),
    ("method_aliases", [" md"]),
    ("method_aliases", ["md", "md"]),
    ("tags", ["Explicit-Water"]),
    ("tags", [""]),
    ("tags", ["-leading"]),
])
def test_aliases_and_tags_must_be_normalised_identifiers(catalog_dir, field, value):
    mutate_registry(catalog_dir, lambda d: d["templates"][0].update({field: value}))
    with pytest.raises(RegistryError):
        load_catalog(catalog_dir)


def test_empty_template_list_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d.update({"templates": []}))
    with pytest.raises(RegistryError, match="at least one template"):
        load_catalog(catalog_dir)


def test_unsupported_identity_scheme_fails(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["identity"].update({"scheme": "semantic-version"}))
    with pytest.raises(RegistryError, match="not supported"):
        load_catalog(catalog_dir)


def test_missing_registry_is_a_clear_error(tmp_path):
    with pytest.raises(RegistryError, match="registry not found"):
        load_catalog(tmp_path)


def test_registry_errors_name_the_field_and_the_value(catalog_dir):
    mutate_registry(catalog_dir, lambda d: d["templates"][1].update(
        {"template_path": "/absolute/template.yaml"}))
    with pytest.raises(RegistryError) as exc:
        load_catalog(catalog_dir)
    assert "template_path" in str(exc.value) and "/absolute/template.yaml" in str(exc.value)


# ================================================================================================
# Template descriptors
# ================================================================================================

def test_14_both_shipped_descriptors_validate():
    catalog = load_catalog(REPO_ROOT)
    for tid in (MD_ID, REST2_ID):
        d = catalog.descriptor(tid)
        assert d.schema_version == TEMPLATE_SCHEMA_VERSION
        assert d.template_id == tid
        assert d.identity.mode == "git-commit-plus-template-path"
        assert {r.id for r in d.input_routes} == {"smiles", "pdb"}


def test_descriptors_state_that_dispatch_is_not_active():
    """PR 1 is catalog-only, and the descriptor has to say so itself."""
    catalog = load_catalog(REPO_ROOT)
    for tid in (MD_ID, REST2_ID):
        impl = catalog.descriptor(tid).implementation
        assert impl.dispatch == "legacy-direct"
        assert impl.binding == "current-openmm-implementation"
        assert impl.python_namespace == "md_templates.openmm"
        assert impl.cli_command == "md-openmm"


def test_descriptors_do_not_claim_template_local_profiles():
    """Profiles are a provider reference; PR 1 moves no packaged resource."""
    catalog = load_catalog(REPO_ROOT)
    for tid in (MD_ID, REST2_ID):
        profiles = catalog.descriptor(tid).profiles
        assert profiles.template_local is False
        assert profiles.provider == "current-openmm-implementation"
        assert profiles.python_resource == "md_templates.openmm.spec.profiles"


@pytest.mark.parametrize("path,key", [
    ([], "maintainer"),
    (["method"], "flavour"),
    (["engine"], "build"),
    (["implementation"], "entry_point"),
    (["restart"], "always_works"),
    (["profiles"], "path"),
    (["validation"], "score"),
    (["bundle_schema"], "deprecated"),
])
def test_15_unknown_fields_fail_at_every_modelled_level(catalog_dir, path, key):
    def mutate(doc):
        node = doc
        for part in path:
            node = node[part]
        node[key] = "x"

    mutate_descriptor(catalog_dir, REST2_ID, mutate)
    with pytest.raises(RegistryError, match=key):
        load_catalog(catalog_dir)


def test_15b_unknown_field_in_a_list_element_fails(catalog_dir):
    mutate_descriptor(catalog_dir, REST2_ID,
                      lambda d: d["input_routes"][0].update({"forcefield": "ff19SB"}))
    with pytest.raises(RegistryError, match="forcefield"):
        load_catalog(catalog_dir)


def test_15c_unknown_field_in_evidence_fails(catalog_dir):
    mutate_descriptor(catalog_dir, REST2_ID,
                      lambda d: d["validation"]["evidence"][0].update({"weight": 10}))
    with pytest.raises(RegistryError, match="weight"):
        load_catalog(catalog_dir)


@pytest.mark.parametrize("version", [0, 2, 99])
def test_16_unsupported_template_schema_version_fails(catalog_dir, version):
    mutate_descriptor(catalog_dir, MD_ID, lambda d: d.update({"schema_version": version}))
    with pytest.raises(RegistryError, match="schema_version"):
        load_catalog(catalog_dir)


def test_16b_template_schema_version_is_independent_of_the_others():
    """A descriptor version must not be tied to the bundle, config or run-state versions."""
    from md_templates.openmm import bundlev2, runstate
    from md_templates.openmm.spec import models

    assert TEMPLATE_SCHEMA_VERSION == 1
    # independence is a source property, not a numeric one: the constant is defined in the catalog
    # package and nothing there imports the runtime's schema constants.
    source = (REPO_ROOT / "src" / "md_templates" / "core" / "template.py").read_text()
    for name in ("BUNDLE_SCHEMA_VERSION", "RUN_STATE_SCHEMA", "SYSTEM_SCHEMA_VERSION"):
        assert name not in source
    assert (bundlev2.BUNDLE_SCHEMA_VERSION, runstate.RUN_STATE_SCHEMA,
            models.SYSTEM_SCHEMA_VERSION) == (2, 1, 1)


@pytest.mark.parametrize("field,value", [
    ("implementation_status", "great"),
    ("implementation_status", "validated"),
    ("scientific_status", "works"),
    ("scientific_status", "implemented-and-tested"),
])
def test_17_invalid_status_enum_fails(catalog_dir, field, value):
    mutate_descriptor(catalog_dir, MD_ID, lambda d: d["validation"].update({field: value}))
    with pytest.raises(RegistryError, match=field):
        load_catalog(catalog_dir)


@pytest.mark.parametrize("bad", [
    "../outside/thing.txt",
    "/etc/passwd",
    "src/../../escape",
    "src\\md_templates",
    "src/./md_templates",
])
def test_18_repository_reference_escape_fails(catalog_dir, bad):
    mutate_descriptor(catalog_dir, MD_ID,
                      lambda d: d["repository_references"].append(bad))
    with pytest.raises(RegistryError):
        load_catalog(catalog_dir)


def test_18b_repository_reference_must_exist(catalog_dir):
    mutate_descriptor(catalog_dir, MD_ID,
                      lambda d: d.update({"repository_references": ["docs/does-not-exist.md"]}))
    with pytest.raises(RegistryError, match="does not exist"):
        load_catalog(catalog_dir)


def test_declared_profile_ids_exist_and_belong_to_the_declared_method():
    """Cross-checked in the test, not in the model.

    The catalog package may not import the OpenMM implementation, so the descriptor cannot validate
    its own profile references against the shipped files. A test can, and this is the pairing that
    keeps the dependency boundary from turning into an unchecked claim.
    """
    from md_templates.openmm.spec import resolve

    shipped = {d["profile_id"]: d for d in resolve.list_profiles()}
    for tid, method in ((MD_ID, "md"), (REST2_ID, "rest2")):
        declared = load_catalog(REPO_ROOT).descriptor(tid).profiles.profile_ids
        assert declared, tid
        for pid in declared:
            assert pid in shipped, f"{tid} names profile {pid!r}, which is not shipped"
            assert shipped[pid]["method"] == method, (
                f"{tid} names profile {pid!r}, which is a {shipped[pid]['method']!r} profile"
            )


def test_19_rest2_declares_omega_exclusion_enabled_by_default():
    d = load_catalog(REPO_ROOT).descriptor(REST2_ID)
    feature = d.feature("omega-exclusion")
    assert feature is not None
    assert feature.supported is True
    assert feature.default_enabled is True
    assert feature.disableable is True


def test_19b_the_descriptor_records_the_runtime_default_without_changing_it():
    """Metadata must agree with the runtime, and the runtime is the authority."""
    from md_templates.openmm import DEFAULTS
    from md_templates.openmm.spec import resolve

    declared = load_catalog(REPO_ROOT).descriptor(REST2_ID).feature("omega-exclusion")
    assert DEFAULTS["rest2"]["omega_exclusion"] is declared.default_enabled
    for pid in ("explicit-rest2-ligand-v1", "explicit-rest2-peptide-v1"):
        profile = resolve.load_profile(pid)
        assert profile["defaults"]["protocol"]["production"]["omega_exclusion"] is True


def test_20_smoke_evidence_cannot_be_marked_scientifically_validated(catalog_dir):
    def elevate(doc):
        for entry in doc["validation"]["evidence"]:
            if entry["kind"] == "cpu-smoke":
                entry["establishes_general_scientific_validation"] = True

    mutate_descriptor(catalog_dir, REST2_ID, elevate)
    with pytest.raises(RegistryError, match="prohibited as scientific evidence"):
        load_catalog(catalog_dir)


def test_20b_regression_evidence_cannot_establish_validation(catalog_dir):
    def elevate(doc):
        for entry in doc["validation"]["evidence"]:
            if entry["kind"] == "regression":
                entry["establishes_general_scientific_validation"] = True

    mutate_descriptor(catalog_dir, MD_ID, elevate)
    with pytest.raises(RegistryError, match="not scientific validation"):
        load_catalog(catalog_dir)


def test_21_rgd_evidence_cannot_elevate_the_general_rest2_status(catalog_dir):
    def elevate(doc):
        for entry in doc["validation"]["evidence"]:
            if entry["id"] == "rgd-10rung-pilot":
                entry["establishes_general_scientific_validation"] = True

    mutate_descriptor(catalog_dir, REST2_ID, elevate)
    with pytest.raises(RegistryError, match="cannot establish general validation"):
        load_catalog(catalog_dir)


def test_21b_validated_status_without_qualifying_evidence_fails(catalog_dir):
    mutate_descriptor(catalog_dir, REST2_ID,
                      lambda d: d["validation"].update({"scientific_status": "validated"}))
    with pytest.raises(RegistryError, match="Running is not validation"):
        load_catalog(catalog_dir)


def test_21c_shipped_rest2_status_is_unvalidated_and_rgd_is_system_specific():
    d = load_catalog(REPO_ROOT).descriptor(REST2_ID)
    assert d.validation.scientific_status == "unvalidated"
    assert d.validation.implementation_status == "implemented-and-tested"
    rgd = next(e for e in d.validation.evidence if e.id == "rgd-10rung-pilot")
    assert rgd.kind == "pilot" and rgd.scope == "system-specific"
    assert rgd.establishes_general_scientific_validation is False
    assert not any(e.establishes_general_scientific_validation for e in d.validation.evidence)


def test_21d_conventional_md_is_not_validated_merely_because_it_runs():
    d = load_catalog(REPO_ROOT).descriptor(MD_ID)
    assert d.validation.scientific_status == "unvalidated"
    assert any("2 fs" in gate for gate in d.validation.open_gates)


def test_bundle_schema_must_be_readable_where_writable(catalog_dir):
    mutate_descriptor(catalog_dir, MD_ID,
                      lambda d: d["bundle_schema"].update({"readable": [1], "writable": [2]}))
    with pytest.raises(RegistryError, match="writable but not readable"):
        load_catalog(catalog_dir)


def test_descriptor_template_id_must_match_declared_method_and_engine():
    with pytest.raises(TemplateError, match="must begin with"):
        parse_descriptor({
            **read_yaml(REPO_ROOT / "templates" / REST2_ID / "template.yaml"),
            "template_id": "conventional-md/openmm/explicit-water",
        })


def test_parse_descriptor_rejects_a_non_mapping():
    with pytest.raises(TemplateError, match="expected a mapping"):
        parse_descriptor(["not", "a", "mapping"])


# ================================================================================================
# review finding 2: catalog loading must not follow symlinks
#
# `is_file()`, `read_text()` and `exists()` all follow symlinks silently. A committed template.yaml
# symlink can point at mutable bytes outside the checkout while Git still reports the tree clean --
# the link itself is unchanged. SHA + path would then name content the commit does not contain.
# ================================================================================================

def test_descriptor_symlink_pointing_outside_the_repository_is_refused(catalog_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "template.yaml"
    external.write_bytes((catalog_dir / "templates" / REST2_ID / "template.yaml").read_bytes())

    descriptor = catalog_dir / "templates" / REST2_ID / "template.yaml"
    descriptor.unlink()
    descriptor.symlink_to(external)

    # the decisive detail: the descriptor still PARSES, so only the symlink rule catches this
    assert read_yaml(descriptor)["template_id"] == REST2_ID

    with pytest.raises(RegistryError, match="symlink"):
        load_catalog(catalog_dir)


def test_symlinked_directory_component_is_refused(catalog_dir, tmp_path):
    """A symlinked directory redirects everything beneath it, so components are checked too."""
    outside = tmp_path / "outside-engine"
    outside.mkdir(parents=True)
    variant = outside / "explicit-water"
    variant.mkdir()
    variant.joinpath("template.yaml").write_bytes(
        (catalog_dir / "templates" / REST2_ID / "template.yaml").read_bytes())

    engine_dir = catalog_dir / "templates" / "rest2" / "openmm"
    shutil.rmtree(engine_dir)
    engine_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RegistryError, match="symlink"):
        load_catalog(catalog_dir)


def test_repository_reference_symlink_pointing_outside_is_refused(catalog_dir, tmp_path):
    outside = tmp_path / "outside-docs"
    outside.mkdir()
    (outside / "note.md").write_text("external\n", encoding="utf-8")

    link = catalog_dir / "docs" / "external-link.md"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside / "note.md")
    mutate_descriptor(catalog_dir, MD_ID,
                      lambda d: d["repository_references"].append("docs/external-link.md"))

    assert link.exists(), "exists() follows the link -- which is exactly the problem"
    with pytest.raises(RegistryError, match="symlink"):
        load_catalog(catalog_dir)


def test_a_symlink_that_stays_inside_the_repository_is_also_refused(catalog_dir):
    """Refused outright rather than resolved-and-permitted.

    An internal link is representable in the commit, so this case is arguably safe. It is still
    refused: permitting it would mean deciding per link whether the target is both inside the tree
    and covered by the same commit, which is easy to get subtly wrong and buys the catalog nothing.
    """
    real = catalog_dir / "templates" / REST2_ID / "template.yaml"
    inside = catalog_dir / "templates" / "rest2" / "shared-template.yaml"
    inside.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(inside)

    with pytest.raises(RegistryError, match="symlink"):
        load_catalog(catalog_dir)


def test_registry_symlink_is_refused(catalog_dir, tmp_path):
    """The index is read under the same rule as everything it points at."""
    outside = tmp_path / "outside-registry"
    outside.mkdir()
    external = outside / "registry.yaml"
    external.write_bytes((catalog_dir / "registry.yaml").read_bytes())

    (catalog_dir / "registry.yaml").unlink()
    (catalog_dir / "registry.yaml").symlink_to(external)

    with pytest.raises(RegistryError, match="symlink"):
        load_catalog(catalog_dir)


def test_the_real_repository_contains_no_symlinked_catalog_paths():
    """The shipped catalog satisfies the rule it enforces."""
    catalog = load_catalog(REPO_ROOT)
    for entry in catalog.entries:
        current = REPO_ROOT
        for part in entry.template_path.split("/"):
            current = current / part
            assert not current.is_symlink(), current
        for ref in catalog.descriptor(entry.template_id).repository_references:
            current = REPO_ROOT
            for part in ref.split("/"):
                current = current / part
                assert not current.is_symlink(), current


# ================================================================================================
# 36. dependency boundary
# ================================================================================================

def test_36_importing_catalog_modules_does_not_import_openmm():
    """Listing and validating templates must work where nothing can be simulated.

    Run in a subprocess: this test process has almost certainly imported OpenMM already through
    another module, so an in-process check would prove nothing.
    """
    code = (
        "import sys\n"
        "import md_templates.core as core\n"
        "from md_templates.core import load_catalog\n"
        "load_catalog(sys.argv[1])\n"
        "banned = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'openmm', 'openff', 'rdkit', 'mdtraj', 'parmed', 'openmmtools', 'numpy', 'scipy'})\n"
        "print(';'.join(banned))\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT)],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"catalog import pulled in {result.stdout.strip()}"


def test_36b_catalog_source_names_no_engine_dependency():
    core_dir = REPO_ROOT / "src" / "md_templates" / "core"
    for path in sorted(core_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for banned in ("import openmm", "import openff", "import rdkit", "import mdtraj",
                       "import numpy", "from openmm", "from openff", "from rdkit"):
            assert banned not in text, f"{path.name} references {banned}"
