"""Selective REST2 Hamiltonians (0.6.1 S1-C, S1-E): the energy the code computes, term by term.

Effective-temperature intuition is not evidence. What is tested here is the scaled System itself:

  * tau = 0 is the unscaled System, byte for byte, for every selection;
  * no selector reproduces 0.6.0 EXACTLY -- compared with the 0.6.0 `hamiltonian.py` frozen in
    `tests/data/selective_rest2/hamiltonian_0_6_0.py`, not with a description of it;
  * every particle and every exception carries exactly its S-S / S-E / E-E factor;
  * the nonbonded energy with PME is exactly `a^2 E_SS + a E_SE + E_EE` (a = 1 - tau), where the
    three components are evaluated INDEPENDENTLY on the unscaled System by zeroing one group;
  * every torsion term is scaled by exactly `a^2` iff the ownership rule says so, decided here
    from the topology rather than by asking the implementation;
  * a CMAP map shared with an unselected residue is duplicated, not scaled in place.

PLATFORM_POLICY_EXEMPTION: single-point energies at frozen coordinates compared to 1e-9 relative;
Reference is the platform that can carry that comparison. No propagation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import (CMAPTorsionForce, Context, NonbondedForce, PeriodicTorsionForce, Platform,  # noqa: E402
                    VerletIntegrator, XmlSerializer, unit)

from md_tools.md.stage import solute_atom_indices                               # noqa: E402
from md_tools.openmm.system import unscaled_torsions                            # noqa: E402
from md_tools.rest2.hamiltonian import build_scaled_system                      # noqa: E402
from md_tools.rest2.regions import atom_category, explicit_selection, resolve_region  # noqa: E402

from .selective_rest2_fixture import RESIDUE, build_fixture                      # noqa: E402

TAUS = (0.2, 0.45, 0.7)
FROZEN = Path(__file__).resolve().parent / "data" / "selective_rest2" / "hamiltonian_0_6_0.py"

COMBINED = {"backbone_scaling_list": ":2-4,11", "sidechain_scaling_list": ":3-4,7",
            "ligand_scaling_dict": {"L01": {"mask": ":13"}}}
CONFIGS = {
    "backbone": {"backbone_scaling_list": ":3"},
    "sidechain": {"sidechain_scaling_list": ":3-4"},
    "ligand": {"ligand_scaling_dict": {"L01": {"mask": ":13"}}},
    "combined": COMBINED,
}


@pytest.fixture(scope="module")
def fx():
    return build_fixture()


@pytest.fixture(scope="module")
def sdf_dir(fx, tmp_path_factory):
    return fx.write(tmp_path_factory.mktemp("built"))


def _selection(fx, config, sdf_dir):
    region = resolve_region(fx.topology, config, config_dir=sdf_dir)
    wanted = set(region["classification_atoms"])
    names = {a.residue.name for a in fx.topology.atoms() if a.index in wanted}
    sdfs = {n: sdf_dir / f"{n}.sdf" for n in ("LGA", "LGB") if n in names}
    classified = unscaled_torsions(fx.topology, region["classification_atoms"], residue_sdfs=sdfs)
    return explicit_selection(fx.topology, fx.system, region, classified)


def _scaled(fx, selection, tau):
    arguments = dict(selection.as_scaler_arguments())
    return build_scaled_system(fx.system, arguments.pop("solute_indices"), tau, **arguments)


def _frozen_module():
    spec = importlib.util.spec_from_file_location("hamiltonian_0_6_0", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- identities ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_tau_zero_is_the_unscaled_system_for_every_selection(fx, sdf_dir, name):
    selection = _selection(fx, CONFIGS[name], sdf_dir)
    assert XmlSerializer.serialize(_scaled(fx, selection, 0.0)) == \
        XmlSerializer.serialize(fx.system)


@pytest.mark.parametrize("tau", (0.0,) + TAUS)
def test_no_selector_is_byte_identical_to_0_6_0(fx, sdf_dir, tau):
    frozen = _frozen_module()
    solute = solute_atom_indices(fx.topology)
    classified = unscaled_torsions(fx.topology, solute,
                                   residue_sdfs={n: sdf_dir / f"{n}.sdf" for n in ("LGA", "LGB")})
    excluded = classified["unscaled_central_bonds"]
    then = frozen.build_scaled_system(fx.system, solute, tau, excluded_bonds=excluded)
    now = build_scaled_system(fx.system, solute, tau, excluded_bonds=excluded)
    assert XmlSerializer.serialize(now) == XmlSerializer.serialize(then)


def test_an_explicit_region_naming_the_whole_solute_is_the_legacy_hamiltonian(fx, sdf_dir):
    """Two different routes to the same region must reach the same System."""
    protein = ":1-12"
    config = {"backbone_scaling_list": protein, "sidechain_scaling_list": protein,
              "ligand_scaling_dict": {"a": {"mask": ":13"}, "b": {"mask": ":14"},
                                      "c": {"mask": ":15"}}}
    selection = _selection(fx, config, sdf_dir)
    solute = solute_atom_indices(fx.topology)
    assert list(selection.solute_atoms) == solute
    classified = unscaled_torsions(fx.topology, solute,
                                   residue_sdfs={n: sdf_dir / f"{n}.sdf" for n in ("LGA", "LGB")})
    legacy = build_scaled_system(fx.system, solute, 0.4,
                                 excluded_bonds=classified["unscaled_central_bonds"])
    assert XmlSerializer.serialize(_scaled(fx, selection, 0.4)) == \
        XmlSerializer.serialize(legacy)


# --- nonbonded: per parameter ----------------------------------------------------------------------

def _nonbonded(system):
    return next(f for f in system.getForces() if isinstance(f, NonbondedForce))


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("tau", TAUS)
def test_every_particle_and_exception_carries_its_factor(fx, sdf_dir, name, tau):
    selection = _selection(fx, CONFIGS[name], sdf_dir)
    hot = set(selection.solute_atoms)
    a = 1.0 - tau
    # Hold the scaled System: a Force outlives nothing, and reading one whose System was freed
    # returns whatever the memory now holds (it did: a charge product of 4.7e-310).
    scaled = _scaled(fx, selection, tau)
    before, after = _nonbonded(fx.system), _nonbonded(scaled)
    for i in range(before.getNumParticles()):
        q0, s0, e0 = before.getParticleParameters(i)
        q1, s1, e1 = after.getParticleParameters(i)
        fq, fe = (a, a * a) if i in hot else (1.0, 1.0)
        assert q1._value == pytest.approx(q0._value * fq, rel=1e-12, abs=0)
        assert e1._value == pytest.approx(e0._value * fe, rel=1e-12, abs=0)
        assert s1 == s0
    counts = {0: 0, 1: 0, 2: 0}
    for k in range(before.getNumExceptions()):
        i, j, qq0, s0, e0 = before.getExceptionParameters(k)
        _, _, qq1, s1, e1 = after.getExceptionParameters(k)
        n = int(i in hot) + int(j in hot)
        counts[n] += 1
        factor = {0: 1.0, 1: a, 2: a * a}[n]
        assert qq1._value == pytest.approx(qq0._value * factor, rel=1e-12, abs=0)
        assert e1._value == pytest.approx(e0._value * factor, rel=1e-12, abs=0)
        assert s1 == s0
    # A partial region must actually exercise every kind it can have, or the check is vacuous. A
    # ligand-only region has no S-E exception: a ligand shares no 1-4 pair with anything else.
    assert counts[2] > 0 and counts[0] > 0
    assert counts[1] > 0 or name == "ligand"


# --- nonbonded: the energy, with PME, against an independent decomposition ------------------------

def _nonbonded_energy(system, positions):
    system = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(1 if isinstance(force, NonbondedForce) else 0)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
    return energy.value_in_unit(unit.kilojoule_per_mole)


def _zeroed(system, keep):
    """The unscaled System with every particle outside *keep* (and every exception touching one)
    carrying no charge and no LJ: the energy of the *keep* group alone, PME background included."""
    copy = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    force = _nonbonded(copy)
    for i in range(force.getNumParticles()):
        if i not in keep:
            _, sigma, _ = force.getParticleParameters(i)
            force.setParticleParameters(i, 0.0, sigma, 0.0)
    for k in range(force.getNumExceptions()):
        i, j, _, sigma, _ = force.getExceptionParameters(k)
        if i not in keep or j not in keep:
            force.setExceptionParameters(k, i, j, 0.0, sigma, 0.0)
    return copy


@pytest.mark.parametrize("name", ["combined", "sidechain"])
def test_pme_energy_is_exactly_the_three_component_polynomial(fx, sdf_dir, name):
    selection = _selection(fx, CONFIGS[name], sdf_dir)
    hot = set(selection.solute_atoms)
    everyone = set(range(fx.system.getNumParticles()))
    full = _nonbonded_energy(fx.system, fx.positions)
    e_ss = _nonbonded_energy(_zeroed(fx.system, hot), fx.positions)
    e_ee = _nonbonded_energy(_zeroed(fx.system, everyone - hot), fx.positions)
    e_se = full - e_ss - e_ee
    assert abs(e_se) > 1.0 and abs(e_ss) > 1.0, "a vacuous decomposition proves nothing"
    for tau in TAUS:
        a = 1.0 - tau
        expected = a * a * e_ss + a * e_se + e_ee
        actual = _nonbonded_energy(_scaled(fx, selection, tau), fx.positions)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-6), (
            f"tau {tau}: {actual} vs a^2 E_SS + a E_SE + E_EE = {expected}")


# --- torsions: every term, against the ownership rule decided HERE ---------------------------------

def _expected_torsion_bonds(fx, config, selection):
    """Which central bonds SHOULD be scaled, from the topology and the stated rules alone."""
    residues = list(fx.topology.residues())
    parts = set()
    from md_tools.rest2.masks import parse_residue_mask

    for key, category in (("backbone_scaling_list", "backbone"),
                          ("sidechain_scaling_list", "sidechain")):
        if config.get(key):
            parts |= {(n - 1, category) for n in parse_residue_mask(config[key])}
    for entry in (config.get("ligand_scaling_dict") or {}).values():
        parts |= {(n - 1, "ligand") for n in parse_residue_mask(entry["mask"])}

    expected = set()
    for bond in fx.topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.residue.index == b.residue.index:
            r = a.residue.index
            if residues[r].name in ("LGA", "LGB"):
                owners = {(r, "ligand")}
            else:
                sidechain = "sidechain" in (atom_category(a), atom_category(b))
                owners = {(r, "sidechain" if sidechain else "backbone")}
        elif {a.name, b.name} == {"C", "N"}:
            carbon = a if a.name == "C" else b
            owners = {(carbon.residue.index, "backbone")}
        elif {a.name, b.name} == {"SG"}:
            owners = {(a.residue.index, "sidechain"), (b.residue.index, "sidechain")}
        else:
            continue
        if owners <= parts:
            expected.add(frozenset((a.index, b.index)))
    return expected - {frozenset(b) for b in selection.excluded_bonds}


@pytest.fixture(scope="module")
def fx14():
    return build_fixture(forcefield="ff14SB")


@pytest.fixture(scope="module")
def sdf_dir14(fx14, tmp_path_factory):
    return fx14.write(tmp_path_factory.mktemp("built14"))


@pytest.mark.parametrize("forcefield", ["ff19SB", "ff14SB"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_torsion_term_is_scaled_iff_its_central_bond_is_owned_by_the_selection(
        request, forcefield, name):
    fx = request.getfixturevalue("fx" if forcefield == "ff19SB" else "fx14")
    sdf_dir = request.getfixturevalue("sdf_dir" if forcefield == "ff19SB" else "sdf_dir14")
    tau = 0.45
    selection = _selection(fx, CONFIGS[name], sdf_dir)
    expected = _expected_torsion_bonds(fx, CONFIGS[name], selection)
    bonded = {frozenset((b.atom1.index, b.atom2.index)) for b in fx.topology.bonds()}
    before = next(f for f in fx.system.getForces() if isinstance(f, PeriodicTorsionForce))
    scaled = _scaled(fx, selection, tau)
    after = next(f for f in scaled.getForces() if isinstance(f, PeriodicTorsionForce))
    scaled_count = 0
    for k in range(before.getNumTorsions()):
        i, j, kk, l, n0, p0, k0 = before.getTorsionParameters(k)
        *_, k1 = after.getTorsionParameters(k)
        proper = all(frozenset(p) in bonded for p in ((i, j), (j, kk), (kk, l)))
        should = proper and frozenset((j, kk)) in expected
        factor = (1 - tau) ** 2 if should else 1.0
        assert k1._value == pytest.approx(k0._value * factor, rel=1e-12, abs=0), (
            f"torsion {k} {i}-{j}-{kk}-{l}: expected factor {factor}")
        scaled_count += should
    if forcefield == "ff19SB" and name == "backbone":
        # ff19SB has no periodic term across N-CA or CA-C: a backbone region's torsional
        # scaling is its CMAP term, and there must be one.
        assert scaled_count == 0 and selection.cmap_terms
    else:
        assert scaled_count > 0


def test_ff14sb_backbone_selection_scales_phi_and_psi_torsions(fx14, sdf_dir14):
    ser = RESIDUE["A.SER"]
    selection = _selection(fx14, {"backbone_scaling_list": f":{ser}"}, sdf_dir14)
    n, ca, c = (fx14.atom(ser, x) for x in ("N", "CA", "C"))
    before = next(f for f in fx14.system.getForces() if isinstance(f, PeriodicTorsionForce))
    scaled = _scaled(fx14, selection, 0.5)
    after = next(f for f in scaled.getForces() if isinstance(f, PeriodicTorsionForce))
    bonded = {frozenset((b.atom1.index, b.atom2.index)) for b in fx14.topology.bonds()}
    seen = {"phi": 0, "psi": 0, "improper": 0}
    for k in range(before.getNumTorsions()):
        i, j, kk, l, *_ = before.getTorsionParameters(k)
        central = {j, kk}
        which = "phi" if central == {n, ca} else "psi" if central == {ca, c} else None
        if which is None:
            continue
        if not all(frozenset(p) in bonded for p in ((i, j), (j, kk), (kk, l))):
            # An improper can put N-CA in its middle two slots (C-CA-N-H, centred on N). It stays
            # unscaled: impropers are never scaled.
            seen["improper"] += 1
            assert after.getTorsionParameters(k)[6] == before.getTorsionParameters(k)[6]
            continue
        seen[which] += 1
        assert after.getTorsionParameters(k)[6]._value == pytest.approx(
            0.25 * before.getTorsionParameters(k)[6]._value, rel=1e-12)
    assert seen["phi"] and seen["psi"], seen


def test_chi1_is_scaled_although_n_and_ca_are_not_hot(fx, sdf_dir):
    ser = RESIDUE["A.SER"]
    selection = _selection(fx, {"sidechain_scaling_list": f":{ser}"}, sdf_dir)
    chi1 = [fx.atom(ser, n) for n in ("N", "CA", "CB", "OG")]
    before = next(f for f in fx.system.getForces() if isinstance(f, PeriodicTorsionForce))
    scaled = _scaled(fx, selection, 0.5)
    after = next(f for f in scaled.getForces() if isinstance(f, PeriodicTorsionForce))
    terms = [k for k in range(before.getNumTorsions())
             if list(before.getTorsionParameters(k)[:4]) in (chi1, chi1[::-1])]
    assert terms, "the fixture has a chi1 torsion"
    for k in terms:
        assert after.getTorsionParameters(k)[6]._value == pytest.approx(
            0.25 * before.getTorsionParameters(k)[6]._value, rel=1e-12)
    assert chi1[0] not in selection.solute_atoms and chi1[1] not in selection.solute_atoms


# --- CMAP ------------------------------------------------------------------------------------------

def _cmap(system):
    return next(f for f in system.getForces() if isinstance(f, CMAPTorsionForce))


def _map_energies(force, index):
    return np.array([e.value_in_unit(unit.kilojoule_per_mole)
                     for e in force.getMapParameters(index)[1]])


def test_a_shared_cmap_map_is_duplicated_for_the_selected_residue_only(fx, sdf_dir):
    base = _cmap(fx.system)
    users: dict[int, list[int]] = {}
    for term in range(base.getNumTorsions()):
        users.setdefault(base.getTorsionParameters(term)[0], []).append(term)
    shared_map, terms = next((m, t) for m, t in users.items() if len(t) > 1)
    chosen = terms[0]
    # The residue that CMAP term belongs to: the residue of its phi's CA (atom 3 of 8).
    ca = base.getTorsionParameters(chosen)[3]
    residue = list(fx.topology.atoms())[ca].residue.index + 1
    selection = _selection(fx, {"backbone_scaling_list": f":{residue}"}, sdf_dir)
    tau = 0.3
    scaled = _scaled(fx, selection, tau)
    after = _cmap(scaled)
    assert after.getNumMaps() == base.getNumMaps() + 1, "exactly one duplicate"
    new_map = after.getTorsionParameters(chosen)[0]
    assert new_map == base.getNumMaps()
    original = _map_energies(base, shared_map)
    assert np.allclose(_map_energies(after, new_map), original * (1 - tau) ** 2, rtol=1e-12)
    assert np.array_equal(_map_energies(after, shared_map), original), "the map others use"
    for term in terms[1:]:
        assert after.getTorsionParameters(term)[0] == shared_map


def _cmap_energy(system, positions, only=None):
    system = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    for force in system.getForces():
        force.setForceGroup(1 if isinstance(force, CMAPTorsionForce) else 0)
    if only is not None:
        force = _cmap(system)
        for term in range(force.getNumTorsions()):
            if term not in only:
                p = list(force.getTorsionParameters(term))
                # Point unwanted terms at an all-zero map, appended once.
                p[0] = force.getNumMaps() - 1
                force.setTorsionParameters(term, *p)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    return context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_cmap_energy_is_a2_on_the_selected_terms_and_unchanged_on_the_rest(fx, sdf_dir):
    selection = _selection(fx, COMBINED, sdf_dir)
    scaled_terms = set(selection.cmap_terms)
    everyone = set(range(_cmap(fx.system).getNumTorsions()))
    assert scaled_terms and everyone - scaled_terms

    with_zero_map = XmlSerializer.deserialize(XmlSerializer.serialize(fx.system))
    size = _cmap(with_zero_map).getMapParameters(0)[0]
    _cmap(with_zero_map).addMap(size, [0.0] * size * size)
    e_sel = _cmap_energy(with_zero_map, fx.positions, only=scaled_terms)
    e_rest = _cmap_energy(with_zero_map, fx.positions, only=everyone - scaled_terms)
    assert _cmap_energy(fx.system, fx.positions) == pytest.approx(e_sel + e_rest, rel=1e-10)
    tau = 0.45
    actual = _cmap_energy(_scaled(fx, selection, tau), fx.positions)
    assert actual == pytest.approx((1 - tau) ** 2 * e_sel + e_rest, rel=1e-9, abs=1e-8)


# --- refusals at construction ----------------------------------------------------------------------

def test_a_selective_region_refuses_scaled_impropers_and_half_a_specification(fx):
    with pytest.raises(ValueError, match="improper"):
        build_scaled_system(fx.system, [0], 0.3, unscaled_impropers=False,
                            torsion_central_bonds=[], cmap_terms=[])
    with pytest.raises(ValueError, match="BOTH"):
        build_scaled_system(fx.system, [0], 0.3, torsion_central_bonds=[])
