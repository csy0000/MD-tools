"""Hamiltonian identity under selective REST2 (0.6.1 S1-F; shared contract §3, 2026-09-19).

Everything 0.6.1 writes is `md-tools-hamiltonian-identity/v3`, whose `selection_sha256` hashes the
Hamiltonian-determining projection of the selection, never its provenance. A 0.6.0 (v2) identity is accepted through ONE named compatibility branch,
and only for a legacy selection with every v2 field matching. The three tests the contract names
are the first three here.

PLATFORM_POLICY_EXEMPTION: identity records over serialised Systems; no Context is created.
"""
from __future__ import annotations

import hashlib

import pytest

openmm = pytest.importorskip("openmm")

from md_tools.rest2.identity import (FINGERPRINT_FORMAT, LEGACY_FINGERPRINT_FORMAT,  # noqa: E402
                                     HamiltonianMismatch, force_summary,
                                     hamiltonian_identities_agree, identity_record,
                                     require_same_hamiltonian, system_fingerprint)

from .selective_rest2_fixture import build_fixture                               # noqa: E402

SOLUTE = list(range(0, 40))
EXCLUDED = [(4, 6), (10, 12)]


@pytest.fixture(scope="module")
def system():
    return build_fixture().system


def _v2_record(system, **overrides):
    """A v2 identity written exactly as 0.6.0's `identity_record` wrote it."""
    selection = hashlib.sha256(repr(({
        "solute": SOLUTE, "unscaled_central_bonds": sorted([a, b] for a, b in EXCLUDED),
        "unscaled_impropers": True})).encode("utf-8")).hexdigest()
    record = {"format": LEGACY_FINGERPRINT_FORMAT, "system_sha256": system_fingerprint(system),
              "selection_sha256": selection, "n_solute_atoms": len(SOLUTE),
              "n_unscaled_central_bonds": len(EXCLUDED), "unscaled_impropers": True,
              "tau": 0.3, "temperature_k": 300.0, "ensemble": "NVT",
              "summary": force_summary(system)}
    record.update(overrides)
    return record


def _current(system, selection=None):
    return identity_record(system, tau=0.3, temperature_k=300.0, ensemble="NVT",
                           solute_indices=SOLUTE, excluded_bonds=EXCLUDED, selection=selection)


EXPLICIT = {"format": "md-tools-solute-selection/2.0", "selection_mode": "explicit",
            "selected_nonbonded_atoms": SOLUTE, "selected_torsion_central_bonds": [[1, 2]],
            "scaled_cmap_terms": [], "excluded_central_bonds": [list(b) for b in EXCLUDED]}


# --- the three tests the contract requires ---------------------------------------------------------

def test_a_legacy_v2_identity_resumes_under_0_6_1(system):
    current = _current(system)
    assert current["format"] == FINGERPRINT_FORMAT
    assert current["selection_mode"] == "legacy-full-solute"
    assert require_same_hamiltonian(_v2_record(system), current) is True


def test_a_v2_identity_meeting_an_explicit_selection_is_refused_by_name(system):
    with pytest.raises(HamiltonianMismatch, match="SELECTIVE REST2 region"):
        require_same_hamiltonian(_v2_record(system), _current(system, EXPLICIT))


@pytest.mark.parametrize("field, value", [
    ("system_sha256", "0" * 64), ("selection_sha256", "1" * 64), ("tau", 0.31),
    ("temperature_k", 310.0), ("ensemble", "NPT"), ("n_solute_atoms", 39),
    ("n_unscaled_central_bonds", 3), ("unscaled_impropers", False),
])
def test_a_v2_identity_with_one_field_changed_is_refused(system, field, value):
    with pytest.raises(HamiltonianMismatch, match=field):
        require_same_hamiltonian(_v2_record(system, **{field: value}), _current(system))


# --- v3 itself -------------------------------------------------------------------------------------

def test_v3_distinguishes_two_selections_with_the_same_nonbonded_atoms(system):
    other = dict(EXPLICIT, selected_torsion_central_bonds=[[1, 2], [2, 3]])
    first, second = _current(system, EXPLICIT), _current(system, other)
    assert first["n_solute_atoms"] == second["n_solute_atoms"]
    assert first["v2_selection_sha256"] == second["v2_selection_sha256"], (
        "the v2 hash cannot see the difference -- which is why v3 exists")
    assert first["selection_sha256"] != second["selection_sha256"]
    with pytest.raises(HamiltonianMismatch, match="selection_sha256"):
        require_same_hamiltonian(first, second)
    assert require_same_hamiltonian(first, _current(system, dict(EXPLICIT)))


def test_a_legacy_v3_record_never_equals_an_explicit_one(system):
    with pytest.raises(HamiltonianMismatch, match="selection_mode"):
        require_same_hamiltonian(_current(system), _current(system, EXPLICIT))


def test_an_unknown_format_is_still_refused(system):
    with pytest.raises(HamiltonianMismatch, match="not"):
        require_same_hamiltonian(_v2_record(system, format="md-tools-hamiltonian-identity/v1"),
                                 _current(system))


def test_the_boolean_form_uses_the_same_rule(system):
    assert hamiltonian_identities_agree(_v2_record(system), _current(system))
    assert not hamiltonian_identities_agree(_v2_record(system), _current(system, EXPLICIT))
    assert not hamiltonian_identities_agree(_v2_record(system, tau=0.5), _current(system))


# --- identity is the Hamiltonian, not its provenance (S0 ruling, 2026-09-19) ------------------------

@pytest.fixture(scope="module")
def selections(tmp_path_factory):
    """Real explicit selections of the boundary fixture, resolved as the scaler resolves them."""
    from .selective_rest2_fixture import build_fixture
    from .test_selective_rest2_selection import _select

    fx = build_fixture()
    where = tmp_path_factory.mktemp("identity-selections")

    def select(config):
        return _select(fx, config, where)[1]
    return fx, select


BASE = {"backbone_scaling_list": ":2-3", "ligand_scaling_dict": {"L01": {"mask": ":13"}}}


def _identity(fx, selection):
    arguments = selection.as_scaler_arguments()
    return identity_record(fx.system, tau=0.3, temperature_k=300.0, ensemble="NVT",
                           solute_indices=arguments["solute_indices"],
                           excluded_bonds=arguments["excluded_bonds"],
                           selection=selection.to_document())


@pytest.mark.parametrize("variant", [
    dict(BASE, ligand_scaling_dict={"renamed": {"mask": ":13"}}),        # a relabelled instance
    dict(BASE, backbone_scaling_list=":2,3"),                            # a respelled mask
])
def test_provenance_alone_resumes(selections, variant):
    fx, select = selections
    first, second = select(BASE), select(variant)
    assert first.provenance_digest() != second.provenance_digest(), "the records DO differ"
    assert require_same_hamiltonian(_identity(fx, first), _identity(fx, second))


def test_one_more_hot_atom_is_refused(selections):
    fx, select = selections
    base = select(BASE)
    wider = select(dict(BASE, sidechain_scaling_list=":2"))
    assert set(wider.solute_atoms) > set(base.solute_atoms)
    with pytest.raises(HamiltonianMismatch, match="selection_sha256"):
        require_same_hamiltonian(_identity(fx, base), _identity(fx, wider))


def test_one_torsion_decision_flipped_is_refused(selections):
    import dataclasses

    fx, select = selections
    base = select(BASE)
    moved = base.torsion_bonds[0]
    flipped = dataclasses.replace(base, torsion_bonds=base.torsion_bonds[1:],
                                  excluded_bonds=tuple(sorted(base.excluded_bonds + (moved,))))
    with pytest.raises(HamiltonianMismatch, match="selection_sha256"):
        require_same_hamiltonian(_identity(fx, base), _identity(fx, flipped))


def test_one_cmap_decision_flipped_is_refused(selections):
    import dataclasses

    fx, select = selections
    base = select(BASE)
    document = dict(base.details)
    decisions = [dict(d) for d in document["cmap_decisions"]]
    decisions[0]["scaled"] = not decisions[0]["scaled"]
    details = tuple((k, decisions if k == "cmap_decisions" else v) for k, v in base.details)
    flipped = dataclasses.replace(base, details=details)
    with pytest.raises(HamiltonianMismatch, match="selection_sha256"):
        require_same_hamiltonian(_identity(fx, base), _identity(fx, flipped))


def test_an_exclusion_files_bytes_are_not_its_identity():
    from md_tools.rest2.identity import hamiltonian_selection_projection

    entry = {"residue_key": ["C", "1", ""], "package": {"parameter_id": "param_0123456789ab"},
             "label": "L01", "torsion_exclusions": {
                 "mode": "file", "file": "a/L01.yaml", "sha256": "1" * 64,
                 "contents": "format: x\n", "parameters": "LOCAL-X/param_0123456789ab",
                 "central_bonds": [["C2", "C3"]]}}
    other = dict(entry, label="L02", torsion_exclusions=dict(
        entry["torsion_exclusions"], file="b/other.yaml", sha256="2" * 64,
        contents="format: x\n# a comment\n"))
    document = {"format": "md-tools-solute-selection/2.0", "selection_mode": "explicit"}
    assert hamiltonian_selection_projection(dict(document, ligand_instances=[entry])) == \
        hamiltonian_selection_projection(dict(document, ligand_instances=[other]))


# --- v3 -> v4: the detector stamp left the projection ---------------------------------------------

def _v3_record(system, selection=None, **overrides):
    """A v3 identity: the v4 projection with `detector_policy_version` put back, as v3 hashed it."""
    from md_tools.rest2.identity import V3_FINGERPRINT_FORMAT, _v3_selection_sha256_candidates

    document = selection or _legacy_document()
    record = dict(_current(system, selection),
                  format=V3_FINGERPRINT_FORMAT,
                  selection_sha256=_v3_selection_sha256_candidates(document)[0])
    record.pop("v3_selection_sha256", None)
    record.update(overrides)
    return record


def _legacy_document():
    from md_tools.rest2.identity import _legacy_selection_document

    return _legacy_selection_document(SOLUTE, EXCLUDED, True)


def test_the_detector_stamp_is_not_in_the_projection(system):
    """It named which algorithm classified, which is provenance -- and this projection's contract
    is that it hashes what determines the Hamiltonian and NOT provenance."""
    from md_tools.rest2.identity import hamiltonian_selection_projection

    assert "detector_policy_version" not in hamiltonian_selection_projection(EXPLICIT)
    # And a document that still carries it (read from a 0.6.1 file) does not smuggle it back in.
    carried = dict(EXPLICIT, detector_policy_version=2)
    assert hamiltonian_selection_projection(carried) == hamiltonian_selection_projection(EXPLICIT)


def test_a_v3_identity_resumes_under_v4(system):
    """A v3 run's Hamiltonian is byte-identical under v4: only the digest's inputs changed.

    Refusing it would strand every in-flight 0.6.1/0.6.2 ladder for a difference that describes no
    physical change.
    """
    current = _current(system)
    assert current["format"] == FINGERPRINT_FORMAT
    assert require_same_hamiltonian(_v3_record(system), current) is True


def test_a_v3_identity_of_an_explicit_selection_resumes_too(system):
    """Unlike v2, v3 COULD describe a selective region, so this branch must accept one."""
    assert require_same_hamiltonian(_v3_record(system, EXPLICIT),
                                    _current(system, EXPLICIT)) is True


@pytest.mark.parametrize("field, value", [
    ("system_sha256", "0" * 64), ("selection_sha256", "1" * 64), ("tau", 0.31),
    ("temperature_k", 310.0), ("ensemble", "NPT"), ("n_solute_atoms", 39),
    ("n_unscaled_central_bonds", 3), ("unscaled_impropers", False),
])
def test_a_v3_identity_with_one_field_changed_is_still_refused(system, field, value):
    """The branch forgives the dropped stamp and nothing else."""
    with pytest.raises(HamiltonianMismatch, match=field):
        require_same_hamiltonian(_v3_record(system, **{field: value}), _current(system))


def test_a_v3_identity_of_a_DIFFERENT_selection_is_refused(system):
    """The stamp is forgiven; a genuinely different Hamiltonian is not."""
    other = dict(EXPLICIT, selected_torsion_central_bonds=[[1, 2], [2, 3]])
    with pytest.raises(HamiltonianMismatch, match="selection_sha256"):
        require_same_hamiltonian(_v3_record(system, EXPLICIT), _current(system, other))


def test_the_v4_digest_really_differs_from_every_STAMPED_v3_one(system):
    """If these were equal the stamped reconstruction would be untested dead code.

    The candidate list OPENS with the v4 digest itself, deliberately: a selection document that
    carried no `detector_policy_version` -- every legacy-full-solute one -- was hashed by v3
    without it, so v3's digest and v4's are the same string. That candidate is therefore expected
    and is not what this test is about. What must differ is every RECONSTRUCTED stamped digest,
    because those are the ones the branch exists to recognise.
    """
    from md_tools.rest2.identity import (_v3_selection_sha256_candidates,
                                         selection_identity_sha256)

    candidates = _v3_selection_sha256_candidates(EXPLICIT)
    assert candidates[0] == selection_identity_sha256(EXPLICIT)
    assert selection_identity_sha256(EXPLICIT) not in candidates[1:]
    assert len(set(candidates)) == len(candidates), "two reconstructions must not collide"
