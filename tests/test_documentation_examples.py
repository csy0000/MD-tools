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
REGISTER = DOCS / "basics" / "data-register"
PROTOCOLS = ("cMD", "REST2", "AIS")  # rREST2 is archived (0.5.4)


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


#: The least a document may state and still resolve, per protocol. Anything absent here is a
#: default, which is what makes "the canonical example states nothing else" checkable.
_REQUIRED = {
    "cMD": "",
    "REST2": "",
    "AIS": "ais_source:\n  trajectory: source.nc\n",
    "umbrella": ("umbrella:\n  file: windows.yaml\n"
                 "collective_variables:\n  file: cv.yaml\n  interval_steps: 1000\n"),
}


def _bare_document(tmp_path, protocol):
    """A document stating ONLY `protocol` plus whatever the resolver refuses to do without."""
    path = tmp_path / f"bare-{protocol}.config"
    path.write_text(f"protocol: {protocol}\n" + _REQUIRED[protocol], encoding="utf-8")
    return path


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_documented_example_is_not_a_copy_of_the_shipped_one(protocol, tmp_path):
    """Two files with the same content are one file and a maintenance burden.

    THE DISTINCTION IS PURPOSE, NOT SIZE. It used to be asserted as "the documented one is
    smaller", which worked only while `configs/md/*.config` carried every key's documentation
    inline and ran to 18-20 KB. That documentation now lives once, generated, in
    `docs/basics/build-md/configuration.md`, so both files are short and a byte count no longer distinguishes
    them -- AIS's quick-start is in fact the larger of the two.

    What separates them is what they are FOR, and that is directly checkable:

      * `configs/md/<protocol>.config` is canonical -- it resolves to the model's own defaults,
        so it shows what the package does when you ask for the least;
      * `docs/openmm_methods/<protocol>/example.config` is task-shaped -- it sets small step
        counts so a run finishes while you watch, and therefore does NOT resolve to the defaults.

    A file that satisfied both descriptions would be one of them wearing the other's name.
    """
    from md_tools.build.md import resolve_md_config

    documented_path = METHODS / protocol / "example.config"
    shipped_path = DOCS.parent / "configs" / "md" / f"{protocol}.config"
    assert documented_path.read_text(encoding="utf-8") != shipped_path.read_text(encoding="utf-8")

    # THE BASELINE IS PER PROTOCOL, and that is not a detail. `resolve_md_config(None)` resolves
    # as cMD, and some reporting defaults depend on the protocol -- a bare AIS document resolves
    # `reporting.info_printout` to 500 where cMD gives 10000. Comparing AIS against cMD's
    # defaults therefore reports the canonical AIS example as non-canonical, which is a bug in
    # the test rather than in the file.
    defaults = resolve_md_config(_bare_document(tmp_path, protocol))
    shipped = resolve_md_config(shipped_path)
    documented = resolve_md_config(documented_path)

    KEYS = (("stages", "production_steps"),
            ("reporting", "crd_printout_solute"),
            ("reporting", "info_printout"))

    # The shipped one states nothing beyond what its protocol requires, so every length and
    # cadence it leaves out must land on that protocol's own default.
    for section, key in KEYS:
        assert shipped[section][key] == defaults[section][key], (
            f"configs/md/{protocol}.config resolves {section}.{key} to "
            f"{shipped[section][key]} where a bare {protocol} document gives "
            f"{defaults[section][key]}, so it is no longer the canonical example")

    # The task-shaped one differs somewhere, or it has no reason to exist beside the other.
    differs = any(documented[section][key] != defaults[section][key]
                  for section, key in KEYS + (("stages", "minimization_iterations"),))
    assert differs, (
        f"docs/openmm_methods/{protocol}/example.config resolves to the same values as a bare "
        f"{protocol} document, which makes it a second copy of configs/md/{protocol}.config "
        f"rather than a task-shaped starting point")


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
    """`docs/structure/project-data.md` stays the schema-level authority; this page is how to use it."""
    page = (REGISTER / "index.md").read_text(encoding="utf-8")
    assert "project-data.md" in page


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
    """Its content moved into the method pages; two authorities is the failure."""
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
