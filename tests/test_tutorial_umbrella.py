"""The umbrella tutorial's configurations resolve, and its prose agrees with them.

`docs/tutorial/ALA/umbrella.md` carries YAML the reader is told to copy, and derived claims about
what that YAML means -- step counts as times, a stage list, atom names for four indices. Both rot
in the same silent way: the schema gains a field, a default moves, a length is edited on the page
and not in the sentence beneath it, and the page goes on looking exactly as authoritative.

So this test lifts the page's own blocks out and puts them through the REAL resolvers, and
cross-checks each derived sentence against the configuration it claims to describe. It is the only
thing standing between the page and drift: no other test compares prose to code.

WHAT IS NOT CHECKED HERE. The page's measured behaviour -- per-window distributions, overlap, the
PMF -- is not measured yet and the page says so. When it is, its numbers belong in a test that
reads the run's own output, not this one.

PLATFORM_POLICY_EXEMPTION: resolves configurations and inspects a topology. Nothing is integrated,
so no platform is selected and this is not offered as evidence for any platform.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "tutorial" / "ALA" / "umbrella.md"
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.skipif(not PAGE.is_file(), reason="the umbrella tutorial is not present")


def _blocks(language: str):
    """Every fenced block of one language on the page, in order."""
    text = PAGE.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)


def _documents():
    """The page's YAML blocks that are whole documents, by the `# filename` they declare.

    A fragment -- a `restraints:` list shown on its own, or the heredoc inside the loop -- is not
    a standalone document and is skipped rather than failed: the page is allowed to show a piece
    of a file. Every block that DOES name a file must parse and resolve.
    """
    found = {}
    for block in _blocks("yaml"):
        first = block.splitlines()[0].strip() if block.strip() else ""
        match = re.match(r"^#\s*([\w.\-]+\.(?:config|yaml))\b", first)
        if match:
            found[match.group(1)] = block
    return found


def test_the_page_declares_the_files_it_tells_you_to_write():
    documents = _documents()
    assert {"build.config", "cv.yaml", "umbrella.yaml", "umbrella.config"} <= set(documents), (
        f"the page names {sorted(documents)}; a reader cannot follow it without all four")


def test_every_declared_document_is_valid_yaml():
    for name, block in _documents().items():
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError as broken:                  # pragma: no cover - the failure message
            pytest.fail(f"{name} on the page is not valid YAML: {broken}")
        assert isinstance(parsed, dict), f"{name} should be a mapping, got {type(parsed).__name__}"


def test_the_umbrella_configuration_resolves_through_the_real_resolver(tmp_path):
    """`md_tools.build.md` is the ONE authority for MD workflow configuration.

    Through the resolver that `build-md` itself calls, on a file holding exactly the page's bytes
    -- not a dict assembled here, which would skip the strict loader that refuses unknown and
    duplicate keys.
    """
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "umbrella.config"
    path.write_text(_documents()["umbrella.config"], encoding="utf-8")
    resolved = resolve_md_config(path)
    assert resolved["protocol"] == "umbrella"
    assert resolved["solvent"] == "implicit"
    # The umbrella and CV sections are both present, because this protocol requires them
    # together: a window that biased a variable and never recorded it is unreweightable.
    assert (resolved.get("umbrella") or {}).get("file")
    assert int((resolved.get("collective_variables") or {}).get("interval_steps") or 0) > 0


def test_the_build_configuration_resolves_through_the_real_resolver():
    from md_tools.openmm.system_config import resolve_sys_config

    document = yaml.safe_load(_documents()["build.config"])
    resolved = resolve_sys_config(document)
    # `solvation` is the resolved fact the page depends on: implicit means no box, no barostat,
    # no ions and no NPT stage anywhere downstream, which is what its stage list shows.
    assert resolved["solvation"] == "implicit", resolved


def test_the_cv_definition_parses_and_names_phi_and_psi():
    from md_tools.cv import parse_cv_definition

    definition = parse_cv_definition(_documents()["cv.yaml"], source="the tutorial's cv.yaml")
    assert set(definition.names) == {"phi_ALA", "psi_ALA"}, definition.names


def test_the_restraint_names_a_variable_the_cv_file_defines():
    """A restraint resolves its `cv` BY NAME, so a name with no definition is unresolvable."""
    cv_document = yaml.safe_load(_documents()["cv.yaml"])
    defined = {entry["name"] for entry in cv_document["collective_variables"]}
    restraints = yaml.safe_load(_documents()["umbrella.yaml"])["restraints"]
    assert restraints, "the page's umbrella.yaml declares no restraint"
    for restraint in restraints:
        assert restraint["cv"] in defined, (
            f"the page biases {restraint['cv']!r}, which its own cv.yaml does not define")


def test_the_page_quotes_the_stage_lengths_its_configuration_sets():
    """Each `# NN ps` annotation beside a step count must be that count at that timestep."""
    document = yaml.safe_load(_documents()["umbrella.config"])
    timestep_fs = float(document["dynamics"]["timestep_fs"])
    block = _documents()["umbrella.config"]

    checked = 0
    for line in block.splitlines():
        match = re.match(r"\s*(\w+):\s*(\d+)\s*#\s*([\d.]+)\s*(ps|ns)\b", line)
        if not match:
            continue
        _key, steps, quoted, unit = match.groups()
        picoseconds = int(steps) * timestep_fs / 1000.0
        expected = picoseconds if unit == "ps" else picoseconds / 1000.0
        assert abs(expected - float(quoted)) < 1e-9, (
            f"the page annotates {line.strip()!r}, but {steps} steps at {timestep_fs} fs is "
            f"{picoseconds:g} ps")
        checked += 1
    assert checked >= 3, f"only {checked} annotated length(s) found; the page should have several"


def test_the_cv_interval_divides_every_stage_the_page_configures():
    """The page claims this, and `cv/schedule.py` enforces it -- on the real numbers."""
    from md_tools.cv import check_divides

    document = yaml.safe_load(_documents()["umbrella.config"])
    interval = int(document["collective_variables"]["interval_steps"])
    stages = document["stages"]
    for key, span in stages.items():
        if key == "minimization_iterations" or not int(span):
            continue
        check_divides(interval, int(span), where=f"the tutorial's {key}", what=key)


def test_the_page_names_the_atoms_its_indices_actually_select():
    """The page prints atom names for its four-index quartets. Against the real topology."""
    pytest.importorskip("openmm")
    from openmm.app import PDBFile

    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    atoms = list(PDBFile(str(ALA)).topology.atoms())
    cv_document = yaml.safe_load(_documents()["cv.yaml"])
    indices = {entry["name"]: entry["atom_indices"]
               for entry in cv_document["collective_variables"]}

    # The page's own table, as it is written there.
    for name, expected in (("phi_ALA", ["ACE:C", "ALA:N", "ALA:CA", "ALA:C"]),
                           ("psi_ALA", ["ALA:N", "ALA:CA", "ALA:C", "NME:N"])):
        resolved = [f"{atoms[i].residue.name}:{atoms[i].name}" for i in indices[name]]
        assert resolved == expected, (
            f"the page says {name} is {' '.join(expected)}; those indices select "
            f"{' '.join(resolved)}")
        assert " ".join(expected) in PAGE.read_text(encoding="utf-8").replace("  ", " "), (
            f"the page no longer prints the quartet {' '.join(expected)} for {name}")


def test_the_profile_loop_gives_each_window_its_own_restraint_file():
    """The defect this page exists partly to correct.

    An earlier version of the method README edited the restraint into a file the configuration
    did not name, so every window resolved the same definition and the profile was N copies of
    one window. The loop on the page must write a restraint file that the configuration it then
    passes to `build-md` actually reads.
    """
    loops = [block for block in _blocks("bash") if "for centre in" in block]
    assert loops, "the page shows no window loop"
    loop = loops[0]
    assert "umbrella.yaml" in loop, "the loop writes no umbrella.yaml"
    # The heredoc must land on the file the configuration names, inside the window's own root.
    assert re.search(r"cat >\s*w\$\{centre\}/umbrella\.yaml", loop), (
        "the loop must write each window's restraint to w${centre}/umbrella.yaml -- the file "
        "umbrella.config names -- not to a scratch name nothing reads")
    umbrella_file = yaml.safe_load(_documents()["umbrella.config"])["umbrella"]["file"]
    assert umbrella_file == "umbrella.yaml", (
        f"the page's configuration reads {umbrella_file!r} while its loop writes umbrella.yaml")


def test_the_page_does_not_offer_a_different_hamiltonian_as_its_reference():
    """The error this page nearly shipped, pinned so it cannot come back.

    An earlier draft promised to check an IMPLICIT GBn2 profile against this system's 1 microsecond
    EXPLICIT TIP3P run, because both are "the ALA phi distribution". They are two different
    Hamiltonians with different free-energy surfaces: agreement would have been luck and
    disagreement uninterpretable, so the check could not have failed for the right reason. A check
    that cannot fail correctly is worse than none, because it reads as validation.

    The page may still LINK the explicit run -- it is the right thing to point at for what a short
    unbiased run misses -- but it must not present it as this profile's reference, and it must say
    which Hamiltonian the reference it does use was run in.
    """
    import re

    text = PAGE.read_text(encoding="utf-8")
    # Emphasis markers removed before matching: the page is free to bold a word inside a phrase
    # ("the **same** Hamiltonian") and a test that breaks on formatting teaches authors to write
    # for the test instead of the reader.
    lowered = re.sub(r"[*_`]", "", text.lower())

    assert "do not compare this against the explicit-solvent runs" in lowered, (
        "the page must warn against comparing an implicit profile with the explicit runs")
    assert "same hamiltonian" in lowered, (
        "the page must say its reference is the SAME Hamiltonian, which is the whole point")
    assert "implicit gbn2" in lowered

    # The reference must be described as UNBIASED. A profile checked against another biased
    # calculation shares whatever the bias got wrong.
    assert "unbiased" in lowered


def test_the_page_quotes_the_agreement_it_claims_consistently():
    """The three checks are quoted in the header and again in section 6; they must match.

    A number repeated in two places in one document is two statements of one fact, and the second
    one rots. These are small enough to compare directly.
    """
    import re

    text = PAGE.read_text(encoding="utf-8")
    rms = sorted(set(re.findall(r"rms(?: deviation)?\s+(?:\*\*)?([0-9.]+)(?:\*\*)? kJ/mol",
                                text)))
    assert "0.23" in rms, f"the PSI agreement -- the method's error -- should appear; found {rms}"
    assert "0.15" in rms, f"the phi consistency check should appear as 0.15; found {rms}"
    assert "0.13" in rms, f"the MBAR agreement should appear as 0.13; found {rms}"

    # AND THE ACCOUNTING, not just the numbers. A phi marginal from phi-biased windows is nearly
    # circular, so a page that leads with it flatters the method. The page must say which number
    # is the error and which is the consistency check.
    lowered = text.lower()
    assert "the method's error" in lowered, "the page must name which number is the method's error"
    assert "consistency check" in lowered, (
        "the page must label the phi agreement as a consistency check rather than a prediction")
    assert "not computed here" in lowered, (
        "the page must say the 2D surface is not computed, since two marginals are weaker")

    # kT must be quoted correctly, since every agreement is judged against it.
    assert "2.49" in text, "kT at 300 K is 2.49 kJ/mol and the page compares its errors to it"


def test_the_page_explains_why_some_windows_are_stiffer():
    """The slide is slope/k, and a reader following this page on a steep barrier will hit it."""
    text = PAGE.read_text(encoding="utf-8")
    assert "slope / k" in text or "slope/k" in text, (
        "the page must give the displacement of a window on a slope, which is what sent two "
        "windows 30 degrees off their centres")
    assert "Energy is NaN" in text, (
        "the page must say that a stiff window launched from an unbiased equilibration blows up, "
        "because 7 of 12 did")
