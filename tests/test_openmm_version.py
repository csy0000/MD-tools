"""OpenMM version identity: which string is the identity, and which one is a build stamp.

Three different strings describe one OpenMM installation, and they disagree:

===================  ==================  ====================
attribute            8.5.2 reports       8.6.0 reports
===================  ==================  ====================
``__version__``      ``8.5.2``           ``8.6``
``short_version``    ``8.5.2``           ``8.6.0``
``full_version``     ``8.5.2.dev-...``   ``8.6.0.dev-...``
===================  ==================  ====================

Two conclusions follow, and both are load-bearing.

``__version__`` changed meaning between releases -- full version to major.minor -- so an equality
check against it is a latent break at the next upgrade regardless of what it is pinned to. That is
not hypothetical: the installer's validator compared against it and would have failed on 8.6.0.

``full_version`` carries a ``.dev-<sha>`` suffix on the *official* conda-forge and PyPI builds
because they ship ``release = False``. Reading that and concluding "this is a development snapshot"
is wrong -- the sha is the release's own tag commit. Recording it as ``8.6.0`` would be equally
wrong, in the other direction: a manifest would then claim a version the installation never stated.

So identity is ``short_version`` (exact stable string) plus ``git_revision`` (the tag commit), and
``full_version`` is recorded verbatim.
"""

from __future__ import annotations

import pytest

from md_templates.openmm import provenance

openmm = pytest.importorskip("openmm")

TAG_COMMIT_8_6_0 = "c6173db6e8edd705eb59172bd21e9ce69c572405"


# -------------------------------------------------------------------------------------------
# What the constants claim
# -------------------------------------------------------------------------------------------
def test_the_supported_version_is_an_exact_stable_string():
    """No range, no wildcard, no ``.dev`` -- the thing a human reads off a release page."""
    assert provenance.SUPPORTED_OPENMM_VERSION == "8.6.0"
    assert ".dev" not in provenance.SUPPORTED_OPENMM_VERSION


def test_the_supported_revision_is_a_full_40_character_sha():
    """A short sha collides; the tag is pinned by its full commit."""
    revision = provenance.SUPPORTED_OPENMM_GIT_REVISION
    assert revision == TAG_COMMIT_8_6_0
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


# -------------------------------------------------------------------------------------------
# The identity block
# -------------------------------------------------------------------------------------------
def test_identity_reports_every_string_separately():
    """Collapsing these into one field is what loses the distinction."""
    identity = provenance.openmm_identity()
    assert identity["available"] is True
    for key in ("short_version", "version", "full_version", "git_revision",
                "release_flag", "dist_metadata_version", "platforms"):
        assert key in identity, key


def test_full_version_is_recorded_verbatim_not_normalised():
    """The manifest must state what the installation stated, suffix and all."""
    identity = provenance.openmm_identity()
    assert identity["full_version"] == openmm.version.full_version


def test_a_dev_suffix_on_an_official_release_is_not_a_failure():
    """The property that makes the whole distinction necessary.

    If this installation is the validated one, it passes the check *even though* ``full_version``
    carries ``.dev-``. A checker that rejected on the suffix would reject the real release.
    """
    identity = provenance.openmm_identity()
    ok, message = provenance.check_openmm_version()
    if identity["git_revision"] == TAG_COMMIT_8_6_0:
        assert ok, message
        if ".dev" in (identity["full_version"] or ""):
            assert identity["release_flag"] is False


def test_short_version_never_carries_the_dev_suffix():
    """Which is why it, and not ``full_version``, is the string compared against."""
    identity = provenance.openmm_identity()
    assert ".dev" not in (identity["short_version"] or "")


# -------------------------------------------------------------------------------------------
# The check itself
# -------------------------------------------------------------------------------------------
def test_the_installed_build_is_the_validated_one():
    ok, message = provenance.check_openmm_version()
    assert ok, message


def test_a_wrong_version_is_rejected_and_says_which_half_failed():
    ok, message = provenance.check_openmm_version(expected_version="0.0.0")
    assert not ok
    assert "short_version" in message and "0.0.0" in message


def test_a_wrong_revision_is_rejected_even_when_the_version_string_matches():
    """The case a plain string comparison cannot catch: a rebuild calling itself 8.6.0."""
    ok, message = provenance.check_openmm_version(expected_revision="0" * 40)
    assert not ok
    assert "git_revision" in message


def test_the_rejection_message_explains_the_dev_suffix():
    """A failure that leaves the reader thinking `.dev-` was the problem has misdiagnosed it."""
    _, message = provenance.check_openmm_version(expected_version="0.0.0")
    assert ".dev" in message and "not itself a failure" in message


def test_dunder_version_is_not_used_as_the_identity():
    """Regression guard for the defect this replaces.

    8.5.2's ``__version__`` was ``8.5.2`` and 8.6.0's is ``8.6``. Any check that compares
    ``__version__`` to a full version string breaks on upgrade. Assert the two genuinely differ
    here, so the guard is anchored to observed behaviour rather than to a remembered claim.
    """
    identity = provenance.openmm_identity()
    assert identity["short_version"] == openmm.version.short_version
    if openmm.version.short_version == "8.6.0":
        assert openmm.__version__ != openmm.version.short_version, (
            "8.6.0 is expected to report __version__ == '8.6'; if upstream changed this, the "
            "installer guidance in installation/README.md needs revisiting")


# -------------------------------------------------------------------------------------------
# Provenance wiring
# -------------------------------------------------------------------------------------------
def test_every_manifest_carries_the_openmm_identity():
    """Recorded once, centrally, so no writer can forget it."""
    block = provenance.environment_block()
    assert "openmm" in block
    assert block["openmm"]["short_version"] == openmm.version.short_version
    assert block["openmm"]["git_revision"] == openmm.version.git_revision


def test_validate_env_reports_the_version():
    from md_templates.openmm import envcheck

    names = [c.name for c in envcheck.run_checks(route="pdb")]
    assert "openmm version" in names
