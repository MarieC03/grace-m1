# The neutrino degeneracy η_ν in the M1 rates: which η belongs where

**Status:** discussion note, 21 September 2026. Nothing described here is implemented.
GRACE currently follows FIL in every point below.

## The idea in short

The rates use one number per species, the neutrino degeneracy η_ν = μ_ν/T, in several
places. For ν_e and ν̄_e this number is multiplied by (1 − e^−τ), so it goes to zero where
neutrinos escape. η_ν plays two different physical roles in the code:

1. it describes **the neutrinos that are present** (blocking factors, spectral shape);
2. it enters **Kirchhoff's law**, which turns an absorption opacity into an emissivity.

The suppression is meant for role 1. The proposal is to keep it there and to use the
unsuppressed, matter-defined η in role 2, for ν_e and ν̄_e only.

## How η_ν is built today

`include/grace/physics/eas_neutrino_rates_analytic.hh`, `make_fugacity_state`:

- lines 525–529: η from the matter, μ_νe = μ_e + μ_p − μ_n − Q_np, η(ν̄_e) = −η(ν_e),
  muon flavours from μ_μ, heavy-lepton species η = 0;
- lines 565–566: the muon flavours are clamped to ±5; the electron flavours are not clamped;
- lines 590–596: `eta_nu[NUE] *= 1 - exp(-tau_n[NUE])`, the same for `NUEBAR`.
  GRACE does not suppress the muon flavours.

FIL does the same for the electron flavours (`Margherita_EOS/src/M1/fugacities.hh:165–170`)
and also suppresses the muon flavours with its own rules (lines 180–197).

## Where η_ν is used

Production path with the Weakhub opacity table (`compute_all_species_weakhub`):

| # | Code | Use | Role |
|---|---|---|---|
| 1 | lines 982–986, `Q = kappa_a * B_E(T, η)`, `R = kappa_n * B_N(T, η)`, all species | Kirchhoff: table opacity → emissivity | 2 |
| 2 | `add_kirchhoff_absorption_opacity_from_QR` (line 903, called at 1003) | Kirchhoff the other way: pair / plasmon / bremsstrahlung emissivity → their absorption opacity | 2 |
| 3 | `add_pair_process_emission` (733), `add_plasmon_decay_emission` (804) | final-state blocking `1 + exp(η − …)` | 1 |
| 4 | neutrino-temperature correction (1038; 1209 in the analytic path) | T_ν = F₂/F₃(η) · ⟨ε⟩ of the radiation field | 1 |
| 5 | beta-equilibrium closure, `eas_policies.hh:481`, `:490` | trapped-neutrino number and energy at a trial state, ∝ T³F₂(η), T⁴F₃(η) | acts only in trapped matter, factor ≈ 1 |
| 6 | output `eta_nu1…5`, `eas_policies.hh:904–913` | diagnostics | — |

The analytic path without the table (not used in production) adds: blocking and energy
averages of the charged-current absorption (656–667), the energy average of the scattering
opacity (688), blocking of the charged-current emission (706–707), and its own Kirchhoff
step (`apply_kirchhoff`, 881).

FIL uses the suppressed η in its blackbody too: `black_body_mev` (`M1.hh:187–198`) reads
`F.eta[i]`, and `calc_Kirchoff_emission` (`M1.hh:1087`) uses it. A comment at `M1.hh:432`
already notes that "later PP could use different fugacities".

## The argument

The emission of a ν_e is the reaction e⁻ + p → n + ν_e. Its rate is fixed by the matter:
T, μ_e, μ_p, μ_n. It does not depend on whether the neutrino will later escape.

Kirchhoff's law is the detailed-balance relation between that emission and the inverse
absorption. Per neutrino energy ε, with stimulated absorption included,

    j(ε) = κ*(ε) · f_eq(ε),    f_eq(ε) = 1 / (exp((ε − μ_ν^eq)/T) + 1),
    μ_ν^eq = μ_e + μ_p − μ_n − Q_np.

μ_ν^eq is a property of the matter. The relation holds whether or not a neutrino gas in
equilibrium exists; it is what makes κ·B equal to the capture rate. With any other η in B,
the product is no longer the emission rate of that matter.

Where the suppressed η is appropriate: everything that depends on the neutrinos actually
present. In transparent matter there is no equilibrium neutrino gas, so nothing blocks the
final state and the spectrum is not a Fermi–Dirac distribution with μ_ν^eq. This is how the
prescription arose in leakage schemes (Ruffert et al.), where the emission is computed
directly from capture formulas and the suppressed η enters only through the blocking factor.
In the Weakhub path the emission is reconstructed from κ, and η takes the other role.

In trapped matter both choices coincide, because the factor is 1.

## Size of the effect

For η_eq = −4 (typical of the neutron-rich ejecta) and η_used = 0:

| | number (F₂) | energy (F₃) |
|---|---|---|
| ν_e emission too high by F(0)/F(−4) | 49 | 52 |
| ν̄_e emission too low by F(+4)/F(0) | 19 | 27 |

ν_e emission lowers Y_e, ν̄_e emission raises it, so transparent neutron-rich matter is
pushed down in Y_e.

Measured in the head-on run `work_full_fix` (5 species, Weakhub, eikonal τ):

- Outside the τ(ν_e) = 1 surface η_eq is −2 … −8 and η_used is 0; inside both are
  −0.35 … −0.6. Share of matter outside (ρ × cell area on the xy plane): 2 % at it 4608,
  13 % at it 5376, 75 % at it 6400.
- In the affected cells the ν_e emission is 34–76× higher and the ν̄_e emission 13–23×
  lower than with η_eq. With η_eq in the emission the sign of the net Y_e source flips in
  94–99 % of the cells sitting at the Y_e floor.
- 76 % of the ejecta mass ends at the table's Y_e floor (0.01).
- The ejecta are out of beta equilibrium because they expand faster than weak reactions
  act: time to change Y_e by 0.1 at the actual rates vs expansion time r/|v_r| is
  940 vs 49 M☉ at it 5376 and 390 vs 135 M☉ at it 6400.
- The run before the optical-depth fix looked less affected because τ was stuck at zero
  everywhere: the dense matter then over-produced ν_e as well, and the extra ν_e absorption
  in the ejecta hid the inflated emission.

Figure: `~/Graveyard/Headon/headon_eta_suppression_work_full_fix.png`
(script `~/Graveyard/m1_scratch/headon_analysis/plot_headon_eta_suppression.py`).

## What would change in the code

- Keep the unsuppressed value as a second array in `fugacity_state` (`eta_eq[s]`), filled
  before lines 590–596.
- Use `eta_eq` in the blackbody of row 1 for `NUE` and `NUEBAR` only. Rows 3–6 unchanged.
- Behind a compile switch that is off by default, so FIL parity is the default behaviour.
- Bounds: the change acts only where τ < ~3. In the ejecta η_eq spans −6 … −0.3 (1st–99th
  percentile), and the largest per-step ΔY_e it produces in `work_full_fix` is 0.11, against
  0.09 today. Cold matter is the exception: see point 7 below.

## Points to discuss

1. Was the suppressed η in FIL's blackbody a deliberate choice, or inherited from the
   leakage prescription where emission is computed directly? (See the comment at
   `M1.hh:432`.)
2. How does the Weakhub table define `kappa_a`: with stimulated absorption (κ*) evaluated at
   the matter's μ_ν^eq? If so, Kirchhoff with η_eq is its consistent partner. Not checked.
3. Row 2 goes the other way (κ = Q/B). For consistency it should use the same η as row 1.
   For ν_e and ν̄_e only the pair process enters there, so the effect is small.
4. Which η should set the spectral shape in the neutrino-temperature correction (row 4)?
   It describes the radiation field, so the suppressed value looks right.
5. Independent test, not yet done: for a few ejecta cells compute the electron- and
   positron-capture emissivities directly (Bruenn 1985; Ruffert et al. B1–B2) and compare
   with κ·B(η_eq) and κ·B(0).
6. Consequence for comparisons: FIL results share the bias, so GRACE–FIL parity in ejecta
   Y_e would be lost with the switch on.
7. Cold tail: η = μ/T grows when T is small. In the transition layer at it 4608
   (T ≈ 0.9 MeV) the lowest 5 % of η_eq reach −22, which would give F₂(+22)/F₂(0) ≈ 2000 for
   the ν̄_e blackbody (times a small T³). Would the electron flavours then need the ±5 clamp
   the muon flavours already have?

## What is and is not established

Established: where η is used; the size and sign of the bias in one production run; that
the two choices agree in trapped matter. Not established: agreement of κ·B(η_eq) with a
direct capture-rate calculation (point 5), and the definition of the table opacity
(point 2).
