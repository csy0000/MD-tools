"""The parameter dict the validated builders take.

`system.py`, `solvation.py` and `implicit.py` were written against this shape and read dozens of
keys from it. `sysgen.py` maps the user's two YAML files onto it -- see `sysgen._legacy_cfg` -- so
the chemistry is driven by the user's configuration without rewriting the builders' signatures,
which would risk the science for a cosmetic gain.

This is the BASE only. Every value a user can set is overwritten from their YAML before a build,
and the public defaults live in `defaults.py`, which is the single place they are declared. The
values here exist so a builder key is never missing; where one of them shadows a public default it
is imported from `defaults.py` rather than spelled again.

Extracted from the previous configuration module; the rest of that module -- resolution, schema
migration, manifest writing -- is gone. So are its `integrator`, `equilibration` and `production`
blocks: nothing read them after the stage chain replaced the old workflow manager, and a stale
`barostat_interval: 50` sitting beside the live `barostat_frequency_steps: 25` is exactly the kind
of second declaration this module is not allowed to keep.
"""
from .system_defaults import DEFAULT_PADDING_NM, DEFAULT_SOLVENT, EXPLICIT_COMBINATIONS

#: Which combination the base carries is `defaults.DEFAULT_SOLVENT`, not a second spelling of it.
#: Every key here is overwritten from the user's YAML before a build; this only decides what a
#: builder sees if a configuration somehow omits the block entirely.
_EXPLICIT = EXPLICIT_COMBINATIONS[DEFAULT_SOLVENT]

DEFAULTS = {'run': {'name': None, 'root': None, 'seed': 20260814},
 'system': {'slug': None, 'solute_kind': 'auto', 'require_input_route': None},
 'structure': {'etkdg': {'version': 'ETKDGv3',
                         'n_conformers': 10,
                         'seed': None,
                         'use_random_coords': False,
                         'prune_rms_thresh': 0.5,
                         'num_threads': 0},
               'mmff': {'variant': 'MMFF94s',
                        'max_iterations': 1000,
                        'energy_tolerance': 1e-06,
                        'force_tolerance': 0.0001}},
 'protonation': {'ph': 7.0,
                 'method': 'openmm',
                 'overrides': [],
                 'histidine_proximity_angstrom': 5.0,
                 'near_ph_window': 1.0,
                 'delete_existing_hydrogens': True,
                 'variants': None,
                 'skip_for_ligand': True},
 'forcefield': {'protein': _EXPLICIT['protein'],
                'water': _EXPLICIT['water'],
                'ligand': 'openff-2.2.1',
                'ligand_charge_method': 'am1bcc',
                'extra_xml': []},
 'solvation': {'water_model': DEFAULT_SOLVENT.lower(),
               'box_shape': 'dodecahedron',
               'padding_nm': DEFAULT_PADDING_NM,
               'padding_semantics': 'openmm',
               'cutoff_fit_policy': 'grow',
               'ionic_strength_molar': 0.15,
               'positive_ion': 'Na+',
               'negative_ion': 'Cl-',
               'neutralize': True},
 'system_build': {'nonbonded_method': 'PME',
                  'nonbonded_cutoff_nm': 1.0,
                  'minimum_image_margin_nm': 0.1,
                  'switch_distance_nm': None,
                  'use_dispersion_correction': True,
                  'ewald_error_tolerance': 0.0005,
                  'constraints': 'HBonds',
                  'rigid_water': True,
                  'hydrogen_mass_amu': None,
                  'hmr_scope': 'none',
                  'remove_cm_motion': True},
 'rest2': {'unscaled_torsions': True,
           'proline_like_residues': ['PRO'],
           'max_proline_ring_size': 7,
           'ladder': {'s_cold': 1.0, 's_hot': 0.25, 'n_rungs': 6, 'interp': 'sqrt'}}}
