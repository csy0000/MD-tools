"""The examples in `docs/` are validated by the same code that validates a user's file.

A documentation example that does not resolve is worse than no example: it is copied, it fails, and
the reader concludes the tool is broken. These do not check that the files LOOK right -- they run
the real strict resolvers and the real contract models over them.

The method examples are deliberately NOT byte-for-byte copies of the comprehensive `configs/md/`
files. Those document every accepted key; these show the few a reader normally sets. Both must
resolve, and a test asserts they have not collapsed into each other.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from md_tools.build.md import resolve_md_config
from md_tools.build.strict import ConfigError

DOCS = Path(__file__).resolve().parents[1] / "docs"
METHODS = DOCS / "openmm_methods"
REGISTER = DOCS / "data_register"
PROTOCOLS = ("cMD", "REST2", "rREST2", "AIS")


# --- the method examples --------------------------------------------------------------------

def test_every_method_has_a_page_and_an_example():
    for protocol in PROTOCOLS:
        assert (METHODS / protocol / "README.md").is_file(), f"{protocol} has no page"
        assert (METHODS / protocol / "example.config").is_file(), f"{protocol} has no example"
    assert (METHODS / "README.md").is_file()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_each_documented_example_resolves_and_selects_its_protocol(protocol):
    resolved = resolve_md_config(METHODS / protocol / "example.config")
    assert resolved["protocol"] == protocol


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_documented_example_is_not_a_copy_of_the_comprehensive_one(protocol):
    """Two files with the same content are one file and a maintenance burden.

    The root example is the reference for every key; the documented one is a task-sized starting
    point. If they ever become identical, one of them has lost its reason to exist.
    """
    documented = (METHODS / protocol / "example.config").read_text(encoding="utf-8")
    comprehensive = (DOCS.parent / "configs" / "md" / f"{protocol}.config").read_text(encoding="utf-8")
    assert documented != comprehensive
    assert len(documented) < len(comprehensive), (
        "the documented example is no longer the smaller of the two")


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_documented_example_states_every_key_it_needs_to_be_runnable(protocol):
    """`protocol` and `solvent` are the two a reader cannot infer, so they must be written out."""
    document = yaml.safe_load((METHODS / protocol / "example.config").read_text(encoding="utf-8"))
    assert document["protocol"] == protocol
    assert document["solvent"] in ("explicit", "implicit")


def test_an_example_with_an_unknown_key_would_still_be_refused():
    """The examples pass because they are correct, not because validation is lenient here."""
    broken = yaml.safe_load((METHODS / "cMD" / "example.config").read_text(encoding="utf-8"))
    broken["stages"]["producton_steps"] = 10          # deliberate typo
    import tempfile

    path = Path(tempfile.mkdtemp()) / "broken.config"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ConfigError, match="producton_steps"):
        resolve_md_config(path)


# --- the registration examples --------------------------------------------------------------

def test_the_documented_user_config_is_the_shipped_one():
    """One file, documented in two places, is one file. A drifted copy teaches the wrong format."""
    documented = (REGISTER / "user.config.example").read_text(encoding="utf-8")
    shipped = (DOCS.parent / "configs" / "machine" / "user.config.example").read_text(encoding="utf-8")
    assert documented == shipped


def test_the_documented_user_config_carries_no_real_identity_or_path():
    document = yaml.safe_load((REGISTER / "user.config.example").read_text(encoding="utf-8"))
    assert document["user"]["person_id"] == "your-name-here"
    assert document["machine"]["md_data"].startswith("/absolute/path")
    text = (REGISTER / "user.config.example").read_text(encoding="utf-8")
    for leak in ("/home/", "/data3/", "@", "orcid.org/0000-0002"):
        assert leak not in text, f"the example leaks {leak!r}"


def test_the_documented_dataset_manifest_validates_against_the_model():
    from md_tools.data_contract.model import Dataset

    document = yaml.safe_load((REGISTER / "dataset.yaml.example").read_text(encoding="utf-8"))
    dataset = Dataset.model_validate(document)
    assert dataset.schema_version == "2.0"
    # The contract's shape, not just a parse: year-first, no month segment.
    assert dataset.path == "2026/ALA/ALA-cMD"
    assert "-" not in dataset.year


def test_the_documented_extension_validates_against_the_model():
    from md_tools.data_contract.extension import Extension

    document = yaml.safe_load((REGISTER / "extension.yaml.example").read_text(encoding="utf-8"))
    extension = Extension.model_validate(document)
    assert extension.schema_version == "2.0"
    # The digest is what makes "this continues that" checkable rather than claimed.
    assert extension.target.source_checkpoint_sha256


def test_the_registration_page_links_the_contract_rather_than_restating_it():
    """`docs/data-contract.md` stays the schema-level authority; this page is how to use it."""
    page = (REGISTER / "README.md").read_text(encoding="utf-8")
    assert "data-contract.md" in page


# --- the documentation tree -------------------------------------------------------------------

def test_no_internal_documentation_link_is_dangling():
    """A link to a page that was consolidated away is how a reader ends up in Git history."""
    import re

    def prose(text):
        """The page with code removed, because a link inside code is not a link.

        Chemistry notation collides with the link syntax exactly: a SMILES such as
        `[C@@H](Cc2ccccc2)N` and a SMARTS such as `[CX3](=[OX1])[NX3]` both contain `](...)`,
        and scanning the raw text reported them as links to pages named `Cc2ccccc2` and
        `=[OX1]`. Fenced blocks first, then inline spans, so a fence containing backticks is
        not half-stripped.
        """
        text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)
        return re.sub(r"`[^`\n]*`", "", text)

    broken = []
    for page in sorted(DOCS.rglob("*.md")) + [DOCS.parent / "README.md"]:
        for target in re.findall(r"\]\(([^)#:]+?)(?:#[^)]*)?\)",
                                 prose(page.read_text(encoding="utf-8"))):
            if target.startswith(("http", "mailto")):
                continue
            if not (page.parent / target).exists():
                broken.append(f"{page.relative_to(DOCS.parent)} -> {target}")
    assert not broken, "dangling documentation links:\n  " + "\n  ".join(broken)


def test_the_retired_replica_exchange_page_is_gone_and_not_referenced():
    """Its content moved into the REST2 and rREST2 pages; two authorities is the failure."""
    assert not (DOCS / "replica-exchange.md").exists()
    for page in sorted(DOCS.rglob("*.md")) + [DOCS.parent / "README.md"]:
        if page.parts[-2:] == ("release-notes", "v0.5.0.md"):
            continue                                   # history may name what was consolidated
        assert "replica-exchange.md" not in page.read_text(encoding="utf-8"), page


def test_the_link_checker_ignores_code_and_still_catches_a_real_break(tmp_path):
    """The stripping must not blind the check -- that would be a checker that passes always."""
    import re

    def prose(text):
        text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)
        return re.sub(r"`[^`\n]*`", "", text)

    page = (
        "See [the contract](data-contract.md).\n\n"
        "```\n"
        "CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])N\n"
        "```\n\n"
        "Inline SMARTS `[CX3](=[OX1])[NX3]` too.\n\n"
        "And [a page that went away](retired-page.md).\n"
    )
    targets = re.findall(r"\]\(([^)#:]+?)(?:#[^)]*)?\)", prose(page))
    assert "Cc2ccccc2" not in targets, "a SMILES inside a fence was read as a link"
    assert "=[OX1]" not in targets, "a SMARTS inside inline code was read as a link"
    assert targets == ["data-contract.md", "retired-page.md"], targets
