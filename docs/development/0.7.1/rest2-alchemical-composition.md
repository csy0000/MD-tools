# REST2 composed with the Amber18 softcore Hamiltonian (0.7.1, milestone 1)

**Design, on paper. Nothing here is implemented, and nothing on 0.7.1 is authorised for
implementation.** Written by S3 at the user's direction (2026-09-20), as the basis for
REST2-TI / REST2-FEP on the TYK2 campaign
([protein-ligand-campaign.md](../protein-ligand-campaign.md)).

Method reference: Wang, Berne and Friesner, *On achieving high accuracy and reliability in the
calculation of relative protein-ligand binding affinities*, PNAS 109(6):1937-1942 (2012),
<https://doi.org/10.1073/pnas.1114017109> (FEP/REST). The two pieces being composed are
0.6.1's selective REST2 (`md_tools.rest2.hamiltonian`, convention `rest2-unscaled-torsions` v3) and
0.7.0's Amber18 softcore Hamiltonian
([softcore-hamiltonian.md](../0.7.0/softcore-hamiltonian.md)).

## 0. The one design decision everything else follows from

REST2 scales **energies of a region**, through per-atom parameters: charges by `a = 1 - tau`,
epsilons by `a^2`, eligible solute torsions and solute CMAP by `a^2` (v3). The Lorentz-Berthelot
rule then produces `a^2` for hot-hot pairs and `a` for hot-environment pairs without any custom
force. The alchemical Hamiltonian mixes **two end states' energies** at fixed coordinates, and its
softcore terms soften a *distance* with `alpha lambda_s` and `beta lambda_e`.

So the composition is:

> **Scale each end state by REST2 at `tau` FIRST, then compose the two scaled end states with the
> Amber18 softcore at `lambda`.**
>
>     U(lambda, tau) = Alch_lambda{ REST2_tau(System A), REST2_tau(System B) }

Both orders are not equivalent, and the other one is wrong. Scaling the composed System would have
to decide what `(1-tau)` means for a mixing weight and for a softening argument; this order never
asks. `tau` reaches only parameters that carry energy, `lambda` only the mixing and the softening,
and neither touches the other's quantities.

Two consequences, and they are the reason to prefer this order:

- **Reduction 1 is by construction.** At `tau = 0`, `REST2_tau` returns an untouched clone
  (`build_scaled_system` returns the clone when the scale is 1), so `U(lambda, 0)` is 0.7.0's
  Hamiltonian over the same plan, force for force.
- **Reduction 2 is by construction.** At `lambda_e = lambda_s = lambda_b = 0` the alchemical
  Hamiltonian is end state A plus the B region's unscaled internal terms (contract section 4), so
  `U(0, tau)` is exactly `REST2_tau(System A)`: 0.6.1's Hamiltonian for that System and that
  selection.

The remaining work is not "multiply by `(1-tau)`": it is saying, per term, which factor the
existing REST2 convention gives it, and refusing the combinations where the answer is not defined.

## 1. The composed Hamiltonian, term by term

Notation. `a = 1 - tau`. `R` is the hot region: 0.6.1's selection record (format
`md-tools-solute-selection/2.0`), for TYK2 the ligand plus the pocket sidechains. `C`, `A-only`,
`B-only` are the topology plan's particle sets; `E` is everything outside `R`. A particle carries
`a_i = a` if it is in `R`, else 1. A pair factor is written `a_i a_j`, which is `a^2` hot-hot, `a`
hot-cold, 1 cold-cold — exactly REST2's split.

| term | 0.7.0 alone | composed, at `tau` | why |
|---|---|---|---|
| end state X's charges | `q_i^X` | `a_i q_i^X` | REST2 v3: charges by `a`. The pair factor follows |
| end state X's LJ epsilon | `eps_i^X` | `a_i^2 eps_i^X` | REST2 v3: epsilon by `a^2`; LB gives `a^2` hot-hot and `a` hot-cold |
| C-C nonbonded, C-C exceptions | energy-mixed `(1-l_e) El_A + l_e El_B` | same, over the scaled charges and epsilons | scaling is inside each end state, mixing is unchanged |
| **region-C softcore LJ** (eq. 21.5/21.6) | `w 4 eps_ij [1/(alpha l + x6)^2 - 1/(alpha l + x6)]` | `w (a_i a_j)^... 4 eps_ij [...]`, i.e. the SAME expression evaluated with the scaled `eps_i` | `eps` is an energy prefactor, so REST2's factor multiplies the whole pair energy. `sigma` and therefore `x6 = (r/sigma)^6` are untouched |
| **region-C softcore electrostatics** (eq. 21.7) | `w q_i q_j erfc(kappa r)/sqrt(r^2 + beta l)` | same expression with the scaled `q_i` | charges carry REST2's factor; the screened, softened kernel is unchanged |
| **the softening arguments `alpha lambda_s`, `beta lambda_e`** | — | **NOT scaled by `tau`** | they are a distance softening on the switching coordinate, not an energy. Scaling them would change the path rather than the region's effective temperature, and it would make `U(lambda, tau)` at fixed `lambda` describe a different transformation than `U(lambda, 0)` does |
| the mixing weights `(1-l)`, `l` | `lambda` only | unchanged | `tau` is not a mixing coordinate |
| boundary 1-4 exceptions (softcore atom to core), `sc_boundary_14 = scaled` | mixed with `lambda_e`, `lambda_s` | the end states' exception values carry `a_i a_j` before mixing | REST2 scales solute 1-4 by `a^2`; here the two factors compose |
| **a unique group's internal pairs and its own 1-4s** (unscaled in `lambda`, contract section 4) | full strength at every `lambda` | scaled by `a_i a_j`, which is `a^2` when the group is in `R` | they are solute-solute terms of the hot region. They stay `lambda`-independent and become `tau`-dependent: that is what makes reduction 2 hold at BOTH end points |
| bonded terms that differ between end states | mixed with `lambda_bonded` | each end state's terms REST2-scaled first (eligible solute torsions by `a^2`; bonds, angles, unscaled torsions and impropers untouched) | REST2 v3's torsion rules apply per end state |
| bonded terms on a softcore atom | unscaled in `lambda` | scaled by `a^2` if eligible under v3, else untouched | the classifier decides, not the alchemy |
| **CMAP of a hot sidechain** | static force, identical in both end states | scaled by `a^2` exactly as 0.6.1 decides it, through the selection record's `cmap_decisions` | CMAP has no `lambda` dependence. A CMAP term whose atoms include an alchemical unique atom is **refused** (see section 5) |
| dispersion correction | `(1-l_s) D_A(V) + l_s D_B(V)` | `D_X` computed from the SCALED epsilons | the coefficient is built from the end state's own parameters, so it follows automatically |
| PME alpha, grid; cutoff; exclusions | fixed | unchanged | scaling touches no geometry |
| environment-environment terms (water, protein outside `R`) | untouched | untouched | REST2 leaves `E` alone |

**What the table means operationally:** the composed Hamiltonian needs *no new force type and no new
expression*. `build_hamiltonian` already reads every per-particle and per-exception parameter from
the end-state Systems it is given. Handing it REST2-scaled end states produces every row above,
including the softcore rows, because the softcore forces take `q`, `sigma`, `eps` from those
Systems. The new work is the plumbing, the identity, and the refusals — not the potential.

## 2. `tau` is built; `lambda` is a Context parameter

CLAUDE.md: *a scaled Hamiltonian is built once, as a file, and never re-derived at run time*. That
rule stays. So a state `(lambda_j, tau_j)` is:

- `tau_j` **baked into a System** by `build-top --rest2-scaler` applied to each end state, recorded
  in `scaler.yaml` with the source digest and the selection record;
- `lambda_j` a **Context parameter** of the composed System, as in 0.7.0.

One composed System per distinct `tau`, and `lambda` moves freely inside it. This is a real cost
and it drives section 3: a cross-state energy at a different `tau` is a different System, not a
parameter change. `md_tools.alchemy.windows.CrossStateEvaluator` holds ONE evaluation Context
today; composed states need one per distinct `tau`. That is a small number (a ladder has a handful
of `tau` levels), but it is an interface change S4 must agree to, not an implementation detail.

The alternative — making `tau` a live Context parameter through custom forces — is rejected: it
would re-derive a scaled Hamiltonian at run time, it would put `tau` and `lambda` in the same
parameter namespace that contract section 5 separates, and it would make a saved state's meaning
depend on a parameter nobody recorded.

## 3. The exchange topology

**A 1-D ladder along a path through the 2-D `(lambda, tau)` space, with `tau = 0` at both ends.**
That is FEP/REST as published: the hot region's effective temperature rises in the middle of the
alchemical path and returns to the reference at both physical end points.

    state index j:      0      1      2      3      4      5      6   ...   K-1
    lambda_j:          0.00   0.08   0.20   0.35   0.50   0.65   0.80  ...  1.00
    tau_j:             0.00   0.10   0.20   0.25   0.25   0.25   0.20  ...  0.00

Why not the alternatives:

- **1-D in `tau` at fixed `lambda`** samples one window better and tells you nothing about the
  others; the alchemical path is what needs sampling.
- **A full 2-D grid** (`K_lambda x K_tau` replicas) is the general answer and the expensive one:
  it multiplies the state count, and most of its states are never the bottleneck. It is worth
  keeping as an option for a diagnostic, not as the default.
- **The tent path** keeps the state count of a plain alchemical leg, needs no reweighting at the
  end points (they are physical: `tau = 0`), and puts the extra sampling exactly where the
  alchemical intermediate is least physical. The end points are the states whose ensembles the free
  energy is about, so they must be unheated.

**Exchanges** are nearest-neighbour swaps in `j`, through the existing MPI and exchange authorities
(`md_tools.remd`), as 0.6.1's ladder does. An attempt between `j` and `j+1` needs the four reduced
potentials `u_j(x_j)`, `u_j(x_{j+1})`, `u_{j+1}(x_j)`, `u_{j+1}(x_{j+1})`. Each is an energy of one
state's System at one configuration, so with `tau` baked, each rank must evaluate its neighbour's
System as well as its own. Two ways, both already in the package's vocabulary: exchange the
CONFIGURATIONS (each rank keeps its System and swaps coordinates, which is what the REST2 ladder
does and what makes its trajectories per state) or hold a neighbour Context. The first is preferred
because it preserves 0.6.1's per-state trajectory contract.

**For the estimators**, MBAR needs `u_k(x_n)` at every state `k`, not only neighbours. With `tau`
baked, that is one evaluation per distinct `tau` level per sample, with `lambda` swept as parameters
inside each. The sample record (`md-tools-alchemical-samples/1`) does not change; `state_id`
already identifies a state, and `components` gains `tau` beside the lambdas — as contract section 5
already provides for.

**TI** consumes `dU/d lambda_k` at fixed `tau`, which is 0.7.0's derivative over scaled end states:
no new derivative. There is no `dU/d tau` requirement, because `tau` is a sampling coordinate here,
not an integration variable — the free energy is still the integral along `lambda` at the ends where
`tau = 0`.

## 4. The two reductions, as numerical tests

Both are exact identities, so they are tested as equalities at float64 on the Reference platform,
not as tolerances on a sampled quantity.

**Reduction 1 — `tau = 0` is 0.7.0.**
Build the composed Hamiltonian from `REST2_0(A)`, `REST2_0(B)` and the same plan, and 0.7.0's from
`A`, `B`. Then:
1. the two Systems' serialised forces agree force for force (same count, class, group and
   parameters);
2. at 9 states (the 5 diagonal lambdas and 4 off-diagonal), the total energies and all three
   `dU/d lambda_k` agree to `1e-9 kJ/mol` on the S3 fixture and on S2's TYK2-shaped plan;
3. the identity record's `alchemical` half is bit-identical, and its REST2 half records `tau = 0`.

**Reduction 2 — the physical end points are 0.6.1's ensemble.**
At `lambda = (0, 0, 0)`, the composed energy equals the energy of `REST2_tau(System A)` — the System
0.6.1's own scaler writes — at the same coordinates, to `1e-9 kJ/mol`, for `tau` in
`{0, 0.1, 0.25, 0.5}`; and at `lambda = (1, 1, 1)` it equals `REST2_tau(System B)` likewise. This is
the test that catches a wrong factor on a unique group's internal terms: those terms are present at
both end points, and only the `a^2` factor makes the end point equal a REST2 System.
A stronger check, worth having because it catches a right answer obtained for the wrong reason: the
per-force-group decomposition must match term family by term family, not only in total.

**Neither reduction is a sampling statement.** The campaign's requirement that REST2-FEP's ΔΔG agree
with 0.7.0's within combined uncertainty is a separate, sampled acceptance row.

## 5. What must be refused

- A hot region that contains **some but not all** of a unique group: the group's internal terms
  would carry mixed factors, and "solute" would mean two different sets in two places.
- A CMAP term whose atoms include an alchemical unique particle. CMAP is a protein backbone term
  and the alchemical region is a ligand; if a future mutation makes them overlap, the composition
  has to be derived again rather than assumed.
- `tau > 0` on a System that is not a saved scaled state, and a `tau` that disagrees with the
  `scaler.yaml` beside it — 0.6.0's existing rule, unchanged.
- More than one connected unique group per side, until 0.7.0 defines how pairs between two groups
  switch (the same refusal 0.7.0 carries).
- A selection record older than `md-tools-solute-selection/2.0`, or an alchemical plan whose
  topology digest does not match the selection's.

## 6. Identity

A composed state is identified by BOTH halves, recorded together and hashed as one:

    composed identity = { rest2: {implementation: rest2-unscaled-torsions v3, tau,
                                  selection_sha256 (the v2 record), scaler.yaml digest},
                          alchemical: {topology plan digest, softcore settings incl.
                                       sc_boundary_14, PME alpha and grid, force-group map} }

`require_same_hamiltonian` already refuses a REST2 mismatch; the composed check refuses either half.
A run that continues with the same `tau` but a different `sc_boundary_14` is a different experiment,
and a fingerprint that records only the REST2 half would not notice.

## 7. What it costs, and what it must buy

Per TYK2 RBFE edge, complex leg, as the campaign describes it (the numbers are the plan's, not
measurements):

| | 0.7.0 plain RBFE | 0.7.1 REST2-FEP, tent path | full 2-D grid |
|---|---|---|---|
| states, complex leg | ~20 lambda windows | ~20 states, `tau` varying along the same path | 20 x 4 `tau` levels = 80 |
| distinct Systems to hold | 1 | one per distinct `tau` (~4) | ~4 |
| extra per-sample evaluation for MBAR | K parameter sets in 1 Context | K states over ~4 Contexts | 80 states over ~4 Contexts |
| exchange traffic | none | nearest-neighbour, as 0.6.1's ladder | 2-D neighbours |
| GPU shape | independent windows | one ladder, 4 cards x 2 ranks (0.6.1's MPS rules) | 80 replicas: out of scope |

So the tent path costs **no extra states**, one composed System per `tau` level, the exchange
machinery 0.6.1 already has, and a cross-state evaluator that can hold a few Contexts.

What it must buy, stated before anything is run, because "it ran" is not a result:

1. **A sampling problem solved that plain FEP cannot.** The campaign's edge `ejm_31 -> ejm_43` is
   the candidate: if a pocket sidechain must rearrange, plain FEP's per-window trajectories stay in
   whichever rotamer they started in, and the ΔΔG depends on the starting structure. The test is
   two plain-FEP runs from different sidechain rotamers giving different ΔΔG, and REST2-FEP giving
   the same ΔΔG from both. That is a falsifiable claim, and it is the whole reason for the method.
2. **Better overlap at equal cost**, measured as the MBAR overlap matrix's nearest-neighbour
   entries and the number of windows needed for a target uncertainty.
3. **Round trips.** Each walker must traverse the path end to end; a ladder whose walkers do not
   round-trip is a ladder in name only. Report round-trip time and per-pair acceptance, as 0.6.1
   reports them.

If none of the three shows, the honest conclusion is that this edge did not need FEP/REST, and the
method's cost is not justified for it — which is a result worth recording, not a failure to hide.

## 8. Open questions for whoever implements this

1. **The torsion classifier must become plan-aware.** `unscaled_torsions` resolves bond orders from
   an SDF on disk, by residue name. A hybrid topology's unique atoms belong to a parameter package,
   not a residue in a file. Shared contracts already flag this seam for S1; composing REST2 with a
   hybrid topology is what forces it.
2. **Which end state's classification wins** when a torsion exists in both end states with different
   parameters but the same atoms. Proposal: classify per end state, refuse if the two disagree about
   whether the central bond is unscaled — a torsion that is eligible at one end and not at the other
   has no single `a^2`.
3. **Does the hot region include the dummies of the OTHER end state?** Proposal: yes, always, since
   they are part of the ligand; and the refusal in section 5 makes it explicit rather than implied.
4. **S4 interface:** one evaluation Context per `tau`, and `tau` in the state's `components`.
5. Whether `sc_boundary_14 = unscaled` is even admissible under composition: those 1-4s are
   `lambda`-independent but `tau`-dependent, which is representable, but the combination has never
   been tested and should be refused until it is.
