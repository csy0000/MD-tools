"""Standard test molecules, chosen for what they exercise per second of `sqm`.

AM1-BCC cost scales hard with size, measured on this machine with `openff-toolkit`:

    ethanol              9 atoms     0.3 s
    phenol              13 atoms     0.6 s
    cyclo-triglycine    21 atoms     5.8 s
    cyclo-tetraglycine  28 atoms    26.5 s
    cyclo-(RGDfV)       ~70 atoms   ~40 min

So a test that is about the CHARGE METHOD -- that am1bcc is requested, runs, and is recorded --
must not pay for a macrocycle. Use `PHENOL`. It is 0.6 s and still exercises an aromatic ring,
ring-aware bond perception and a hydroxyl, which a linear alkyl like ethanol does not.

Reach for a bigger molecule ONLY when the molecule is the thing under test:

* `CYCLO_TETRAGLYCINE` -- a 12-membered ring with four backbone amides, for the omega ring-size
  bound. Nothing smaller has the property being asserted. (Its uses today are RDKit-only bond
  perception and pay no charge cost at all.)
* cyclo-(RGDfV) -- the vetted production molecule. It is pinned to NAGL charges everywhere,
  which is ~1 s rather than ~40 min, and that pin is why the RGD fixtures are affordable.
"""

#: The default small molecule for anything testing charge derivation.
PHENOL = "c1ccc(cc1)O"
PHENOL_N_ATOMS = 13

#: Cheapest possible molecule; use when even bond perception is incidental.
ETHANOL = "CCO"

#: A 12-ring with four backbone amides. Use only where the ring size IS the assertion.
CYCLO_TETRAGLYCINE = "O=C1CNC(=O)CNC(=O)CNC(=O)CN1"

#: Above this, a molecule is too expensive to charge with am1bcc in a routine test.
#: cyclo-triglycine (21 atoms, 5.8 s) is the smallest thing that exceeds it.
MAX_ATOMS_FOR_ROUTINE_AM1BCC = 16
