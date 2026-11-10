# GRACE M1 — the complete map (14 October 2026)

This is the one document that says what the M1 sector of GRACE does, in what order, with which
assumptions, what has been verified against what, what is open, and what has to be done next.
It replaces nothing: the detailed notes it points to stay where they are. Read it top to bottom once;
afterwards use sections 12 and 13 as the running list.

Status tags used below: **[OK]** verified (test, parity run, or bitwise comparison), **[CHOICE]** a
deliberate physics or numerics decision that departs from the reference and must not be "fixed" by
accident, **[OPEN]** known defect or question without a decision, **[DECISION]** needs the user to choose.

The reference implementation throughout is FIL (`frankfurt_m1` + Margherita EOS, `~/Codes/FIL`).
Where GRACE departs from FIL on purpose the tag is [CHOICE] and the reason is given.

---

## 1. The system in one page

Grey (energy-integrated) two-moment transport. Species: 1 (νe), 3 (νe, ν̄e, νx = all four heavy) or
5 (νe, ν̄e, νμ, ν̄μ, νx = ντ+ν̄τ), selected at build time by `GRACE_M1_NU_SPECIES`. Each species evolves
five lab-frame, densitised variables (√γ E, √γ F_i, √γ Γ n), closed with the Minerbo closure evaluated
in the fluid frame. Time stepping is ARS-IMEX (`imex232` in production): transport explicit, collisions
implicit, matter coupling ("backreaction") applied after each implicit stage from the realised exchange.

Rates come from one Weakhub grey table (absorption and scattering opacities per species on a
(ρ, T, Yₑ, Y_μ) grid, no emissivities) plus analytic thermal extras for the heavy flavours (e⁺e⁻ pairs,
plasmon, NN bremsstrahlung, Ruffert+96 forms) and Kirchhoff emission at the matter temperature with the
current neutrino chemical potential, interpolated between trapped and free-streaming limits by an eikonal
optical depth. A spectral correction (T_ν/T)² and per-cell beta-equilibrium policies sit on top.

Composition (Yₑ, Y_μ) and fluid energy change only through the backreaction; there is no projection
of the fluid onto equilibrium. The "beta-equilibrium policies" change the state the *rates* see, not the
fluid.

On top of this standard path, Marie's branch (merged 13 Oct) adds optional modes: spectral kernels for the
heavy-flavour pair channels with the partner either at LTE (`equilibrium`) or taken from the actual
radiation (`evolved`), and a muon decay / inverse decay channel. They are off by default; the standard
path is bitwise unchanged by their presence.

---

## 2. One time step, in order

`imex232` (production: HOT-TOV, head-on): explicit(γ) → implicit(γ) → explicit(1−δ) → implicit(γ) →
explicit(γ, last). Every stage ends with an auxiliaries pass; the implicit ones only when M1 is active.
So a step performs **five full auxiliaries + EAS passes** (3 explicit, 2 implicit) — see 12.B.1.
`imex222` is cheaper but its explicit part is RK2: Z4c is unstable at cfl 0.25 on the finest level
(use cfl ≤ 0.15, or imex232) [OK, measured].

### 2.1 Explicit substep (`advance_substep`, `src/evolution/evolve.cpp`)

1. `compute_fluxes`: GRMHD fluxes, then (if M1 active) `compute_M1_{x,y,z}_flux` via `getflux` with the
   closure; HLL-type flux with FIL's damping factor built from √(κ_L κ_R) [OK, HOT-TOV symmetric to 1e-12].
2. `compute_emfs`.
3. `flag_fofc_cells`: hydro criteria (c2p floored, DMP), and for M1 a per-species test of the tentative
   update against the radial atmosphere floors (E or N below floor → tag the six faces with bit s+1).
4. `apply_fofc_correction`: direct tagged sweep (no atomics in the list order) recomputes tagged faces
   with donor-cell fluxes; M1 faces through `compute_fofc_flux<dir,sp>` (Marie's fix: the restored
   conserved state is fed un-normalised; before it, 82 % of faces were tagged and the halo had E/N 3–10×
   wrong) [OK, curved beam A/B].
5. Refluxing, `add_fluxes_and_source_terms` (`compute_sources_M1` = geometric sources), CT, FD.
6. `update_m1_optical_depth` (only `tau_policy: eikonal`): Neilsen-type min-path relaxation of τ for the
   electron flavours from the lagged κ, neighbours-only sweep (the self term made τ monotone, fixed 21 Sep)
   [OK]. τ is then ghost-exchanged before the rates read it.
7. Boundary conditions, stage hook, algebraic constraints, auxiliaries pass (2.2).

### 2.2 Auxiliaries pass (`compute_auxiliary_quantities`, `src/evolution/auxiliaries.cpp`)

1. `conservs_to_prims` (hydro c2p; leptonic EOS; entropy backup; atmosphere floors) and E_center.
2. `m1_get_auxiliaries` → `m1_equations_system_t::compute_auxiliaries<s>` per species: closure (fluid-frame
   J, H, χ, Γ from z^i; the z→v re-conversion bug fixed 20 Sep), atmosphere (FIL's: radial E/N floors
   E_fl, N_fl with (r_damping/r)² fall-off, F zeroed on reset, ε bounds), excision (lapse or radius,
   `m1.excision.*`).
3. `set_m1_eas`: the rates pass (section 4) for every cell; floors for transparent cells; diagnostics.
   While the activation trigger is idle only a store-only pass runs that floors the EAS fields.

### 2.3 Implicit substep (`advance_implicit_substep`, `evolve.cpp:1993`)

1. new := old.
2. Only with `pair_treatment: evolved`: launch `m1_pair_sources` (small tiles, one spectral kernel built
   per cell and launch): species (νμ, ν̄μ) as a coupled pair solve `compute_pair_implicit_update<2,3>`
   (or, with `muon_decay`, the four-species `compute_leptonic_implicit_update`), then the symmetrised
   kernel for the τ aggregate `<4,4,2>`; 3 species: `<2,2,4>`. Failures go to the `[PAIR_FAILURE]` buffer
   (`check_pair_failures`).
3. `m1_implicit_sources`: per species not handled above, `compute_implicit_update<s>` — Newton on
   (E, F_i) with the lagged matter state and the stage's rates (section 6); then, if backreaction is
   enabled and t ≥ t_start, `add_backreaction` (section 7), which with decay active dispatches to
   `add_leptonic_backreaction`.
4. Auxiliaries pass (2.2) with `clamp_to_atmo=false`.

### 2.4 Per-step reports

`report_m1_implicit_failures` (bits per species, sticky in `m1_implicit_err`), `report_betaeq_failures`
(BETAEQ_* bits, partial-E rescues counted separately), `check_pair_failures` (her modes), the FOFC and
c2p counters, NaN termination (writes a NaN checkpoint — restart from an earlier one, 12.B.5).

---

## 3. Variables

**State** (per species s = 1..5): `ERADs_`, `NRADs_`, `FRADX/Y/Zs_` (densitised lab frame);
`YESTAR_`, `YMUSTAR_` (D·Yₑ, D·Y_μ; `YMUSTAR_` sits next to `YESTAR_` since the muon split, so pre-Aug-2026
5-species checkpoints are refused). With `GRACE_M1_OPTICAL_DEPTH`: `OPTD1_`, `OPTD2_` (electron flavours
only; OPTD3–5 were never read and are gone). FIL writes primitives (E lab, n fluid frame), GRACE writes
these conservatives — always convert when comparing [OK, documented].

**Aux, rates group** (per species): `KAPPAAs_` (κ_a, energy), `KAPPAANs_` (κ_n, number), `KAPPASs_`,
`ETAs_` (Q), `ETANs_` (R); `BETAEQ_ERR_`; only in `GRACE_M1_PAIR_KERNELS` builds (≥ 3 species) the 12
`PAIR_*` slots (`PAIR_MUE_ … PAIR_RES_`, zero in the standard mode; `PAIR_ACTIVE_` bit 8 = decay active).

**Aux, physics output** (`GRACE_M1_PHYSICS_OUTPUT`, default ON): output group "fugacity" = `ETANUs_` of the
lepton flavours (η_ν as used), `MUE_ MUMU_ MUP_ MUN_` (`MUMU_` with muons); group "backreaction" =
`M1_HEATCOOL_`, `M1_LEPTON_SOURCE_`, `M1_MUON_SOURCE_`, `M1_BR_REJECT_` (1 full reject, 8 muon partial,
16 electron partial, 32 connected-decay limiting).
**Aux, debugging diagnostics** (`GRACE_M1_DIAGNOSTICS`, implies the physics output; output group
"diagnostics" = both physics groups plus these): `ETANU` of νx (identically 0), `MUDELTA_NPE_/NPMU_`,
`XN_ XP_ XA_ XH_ ABAR_ ZBAR_`, `BETAEQ_TSCALE_` (τ_β/dt; 1e30 = never), `BETAEQ_T/YE/YMU_EQ_` (closure
target; −1 = not applied), `M1_MUON_SOURCE_RAW_`, `M1_IMPLICIT_ERR_` (5 bits per species: roundoff,
small step, max iter, linear fallback, non-physical), `M1_IMPLICIT_RES_`, `M1_EXPLICIT_STEP_`.
All of these are written straight from the fugacity state and the solves, never recomputed [CHOICE: FIL
recomputes].

**Scalars / surfaces**: luminosity detectors (species, index bug fixed), outflow bins
(`Mdot_ye` 30 linear bins over `outflows.ye_bin_min/max`, `Mdot_ymu` 24 log bins, `Mdot_yp` on the Yₑ
edges; they sum to `Mdot_unbound_bern`), c2p error bitmask (documented bit numbers off by one at 5
species, 12.B.6).

---

## 4. The rates pass (EAS) — the standard path

File: `include/grace/physics/eas_neutrino_rates_analytic.hh` (kernels) and `eas_policies.hh` (driver,
policies, Marie's hooks). Production uses `kinds: ["neutrino_weakhub"]` → `compute_all_species_weakhub`;
`neutrino_analytic` → `compute_all_species` (same structure, Ruffert CC instead of the table).

### 4.1 Fugacity state (`make_fugacity_state`)

From the EOS at the evolved (ρ, T, Yₑ, Y_μ): μₑ, μ_μ, μ_p, μ_n, X_n, X_p (free nucleons), X_a, X_h,
Ā, Z̄; n_b = ρ_cgs·N_A. Neutrino chemical potentials **with Qnp subtracted** (FIL parity, verified by
the beta-equilibrium Yₑ 0.06014):

    μ_νe = μₑ + μ_p − μ_n − Qnp,   μ_νμ = μ_μ + μ_p − μ_n − Qnp,   μ_νx = 0,   η_ν = μ_ν/T.

All chemical potentials include rest masses (leptonic EOS convention). The leptonic EOS "ye" axis is
really Y_p; with the no-electron baryon table and `add_ele_contribution=true`, Y_p = Yₑ + Y_μ [OK].

### 4.2 Three protections (standard path only — see 8.3 for what Marie's modes do with them)

1. **τ interpolation** (Foucart/Bollig): η_νe, η_ν̄e → η·(1 − e^{−τ}), τ from `tau_policy`
   (`eikonal` in production; `none` = thin limit, η = 0; `analytic_density` = cold fit, wrong for hot
   matter; `local_kappa` = κ·(r_outer − r); `local_spherical` = deprecated alias of the cold fit).
   Electron flavours only.
2. **±5 clamp** on η_νμ, η_ν̄μ (`eas_neutrino_rates_analytic.hh:565`). Load-bearing: without it the
   Kirchhoff denominators underflow (κ_n(ν̄μ) → 1e31–1e43) and the beta-eq solve diverges [OK, measured
   twice, with and without eikonal τ]. The source of large |η_νμ| is twofold: cold neutron-rich matter
   ((μ_p − μ_n − Qnp)/T ≈ 84 at the T floor) and floor muons (μ_μ ≈ 96–104 MeV, section 8).
3. **Muon rate gate** S(ρ, T): sigmoid in log ρ and T (T₀ = 2.5 MeV) that scales the muonic CC
   (Q, κ_a) and (R, κ_n) pairs together, so the equilibrium they encode is unchanged and only the
   relaxation rate vanishes in cold dilute matter.

### 4.3 Table and Kirchhoff

`weakhub.lookup(ρ, T, Yₑ, Y_μ)` returns κ_a^E, κ_a^N, κ_s per table species; GRACE's 5-slot layout maps
a 6-species table onto νx by **averaging** the heavy opacities (w = ½ at 5 species, ¼ at 3) while summing
emissivities [CHOICE: FIL sums, which is 1.34× too high]. Below the table's ρ_min the lookup floors
instead of clamping (a clamp produced 1000× opacities in the halo) [OK, root cause of the halo failures].
Outputs are floored at 1e-60 so log plots work.

Emission for every species is Kirchhoff at the matter temperature and the current η_ν:

    Q_s = κ_a,s · B_E(T, η_ν,s; g_s),   R_s = κ_n,s · B_N(T, η_ν,s; g_s),   g = (1,1,1,1,2) [5 sp], (1,1,4) [3 sp].

### 4.4 Thermal extras (heavy flavours)

`add_pair_process_emission` (Ruffert e⁺e⁻; νμ/ν̄μ pair with 1/36, νx with 1/18 (5 sp) or 1/9 (3 sp)),
`add_plasmon_decay_emission` (Ruffert transverse plasmon), `add_brems_emission` (analytic fit;
νμ, ν̄μ, 2× or 4× for νx). Electron-flavour pair and plasmon terms are **off in both paths** (FIL parity:
they overproduce without final-state blocking; the comment above `add_plasmon_decay_emission` claiming
otherwise is stale) [CHOICE]. Their Kirchhoff inversion κ = Q/B(T, η_ν) is added to the table's κ; the
emission is used directly. Note the inconsistency for the muon flavour: the pair term is inverted with the
*charged-current* η_νμ although pairs have μ_ν + μ_ν̄ = 0 (see `docs/m1-neutrino-degeneracy-roles.md`,
12.A.4). Checked against Harry's spectral kernels (section 11): pairs within 10 %, plasmon model 7–15×
below Harry's γPP [OPEN], brems fit 1.2–2.6× above HR98 at the states tested [OPEN].

In the Weakhub path the pair and brems extras are switched off automatically at startup when the table
already carries them (`weakhub_table_content` resolving to `thermal`, read from the table's companion
`_tableinfo.log`; one startup line says so) — nothing is counted twice.

**Vacuum muon decay (`muon_decay: true` with `pair_treatment: standard`, 14 Oct, `add_muon_decay_emission`).**
μ⁻ → e⁻ ν̄ₑ νμ and the conjugate as grey emission only, added after the gate and the spectral correction
in both rate paths: the muon excess over the leptonic EOS's dilute band (Y_μ ≤ 6e-4 is muon-free there,
so that residual stays and the backreaction never meets the table floor) decays with the free lifetime
2.197 μs, blocked by the electron Fermi sea (Michel average on 16 Gauss nodes: exponentially strong in
cold dense matter, 1 in ejecta) and by the emitted neutrinos jointly in the Ruffert form with F's η
(τ-interpolated electron flavours, clamped muon flavours), and only where the muonic charged-current
reactions are gated off — the factor 1 − S of the muon rate gate: inside the gate the standard path
treats the muon sector as a trapped equilibrium, in which the decay is balanced by its inverse (an
emission-only sink there would drain the core on ~100 μs; the semi-transparent shell, where the seas are
only partly depleted, is approximated as balanced — the honest treatment needs the actual partner
populations, Marie's direction); Michel means 7/20 and 3/10 m_μ; no inverse
opacity, no Kirchhoff inversion; the baryon number is ρ/m_p so one emitted pair is exactly one decayed
muon; the rate (1−e^{−Γdt})/dt on the full step cannot overshoot the excess with imex232. The electron's
7/20 m_μ stays in the fluid through the composition change (the lepton table's muon energy includes the
rest mass). Tests: rate kernel, blocking quadrature, placement, and a 20-stage backreaction loop
(Yₑ+Y_μ and τ+E_rad exact). Agrees with Marie's decay kernel at LTE to ≤ 15 % in dilute matter
(`[decay][vacuum]`, kernel builds). Startup checks: 5 species with muons, leptonic EOS with
`dilute_muon_suppression: true`, table floor below the band, no `GRACE_FREEZE_HYDRO`, table without decay.

### 4.5 Spectral correction

`temperature_correction` (on): T_ν from the fluid-frame mean energy ε = J Γ/N of the actual radiation
(closure), via the FD ratio; when T_ν > T, κ_a, κ_n, κ_s of νe/ν̄e are scaled by (T_ν/T)²; νx only κ_s;
muon flavours nothing (the table carries their spectral information). Q and R are **not** scaled
(`temperature_correction_emission: false`) [CHOICE, 20 Sep: scaling Q, R pinned the ejecta at the Yₑ
floor; FIL scales them; tested as the FIL–GRACE Yₑ driver and refuted]. The factor is 1 when there is no
radiation, in floor cells, and for ε ≥ 0.5 m_p (energised cells near a forming BH hit 1e4–1e10 without
the cut) [OK]. Never cap the correction itself [working agreement].

### 4.6 Beta-equilibrium policies (`betaeq_policy`)

- `off`: rates at the evolved state. (Head-on parfile originally.)
- `chemical`: Yₑ reset to μ_νe = 0 before the rates.
- `timescale` (FIL `use_grrt_beta_eq`; HOT-TOV production): τ_β = 1/√(κ_a(κ_a+κ_s)) of the fastest
  lepton-carrying species; if τ_β/dt < 1 and all lepton-carrying N are above 1e-16, solve the Perego+19
  trapped equilibrium (Y_le, Y_lμ and total energy conserved, equilibrium FD gas at the trial state,
  damped Newton with FD Jacobian, step-cap backtracking; `partial_e_fallback` retries (Yₑ, T) at fixed
  Y_μ, optionally held at the table floor) and evaluate the rates at (T, Yₑ, Y_μ)_eq — fully for
  τ_β/dt < 0.5, linearly blended up to 1. **Rates only, no grid write** [OK, FIL parity]. Failures are
  table-floor cells (Y_μ 5e-4, Yₑ 0.01) and neutrino-flooded cells whose lepton target lies outside the
  table; they hold 34–49 % of the backreaction rejections in the head-on [OPEN, 12.A.11].
- `gieg`: per-flavour two-timescale variant (Gieg+ 2026); redundant with eikonal τ, use `tau_policy: none`.

### 4.7 What is LTE and what is not

Evolved (allowed off equilibrium): E, N of every species; the mean energy ε (only acknowledged by 4.5);
fluid Yₑ, Y_μ, T (out of beta equilibrium wherever τ_β > dt); the angular distribution (closure).
Assumed LTE: the emission spectrum (black body at T with the current, τ-interpolated, clamped η_ν);
the grey opacities (table averages over that spectrum); pair partners (annihilation against an LTE
partner, κ = Q/B); one matter temperature; muon abundance from the EOS at (Y_μ, T); no muon decay.
Under `timescale`, stiff cells have their rates at full trapped equilibrium.

---

## 5. Transport details (pointers)

- Closure: Minerbo; fluid-frame quantities through z^i; `update_closure` solves a bracketed Brent per
  Newton evaluation (its ζ seed is dead — perf item). The fluid-frame closure is **not hyperbolic** for
  fluid moving across radiation (complex eigenvalues from v = 0.1, speeds > 1 at v = 0.5): wavespeeds are
  capped at 1 [CHOICE]; the F·v lower clamp stays removed (FIL has none; cost: a transparent beam across a
  fast oblique flow spreads in z) [CHOICE].
- Number flux: FIL's form with Γ, the extra √γ removed (0436b904) [OK: polar Yₑ-floor ejecta gone].
- FOFC: per-species bit tag, donor recompute, non-destructive floor; `PPLIM` disabled (per-face plus a
  dt←t mis-wire); the emf out-of-range read in `flag_fofc_cells`' ghost layer is unfixed [OPEN, 12.B.2].
- Atmosphere: FIL's treatment (`E_fl`, `N_fl` independent, `r_damping`, F zeroed). Remaining defects:
  `eps_fl` 2600× too large for its purpose; the FOFC floor test is blind in a band [OPEN, 12.A.8].
- Excision: `m1.excision.excision_lapse/radius` (negative = hydro value, 0 off); collisions and
  backreaction still run in excised cells (a sink if the M1 region exceeds the hydro one) [OK].
  Head-on: `excision_lapse 0.1` flickers 2×2 cells at the puncture near t ≈ 350; 0.15 makes it permanent.
- Activation trigger: `m1.trigger` from the co_tracker separation (FIL uses half the separation — set
  30 in GRACE for FIL's 15) [OK]; the startup check must also run on restart [OK].
- Initial data: `zero` (atmosphere: N = N_floor(r), Marie's fix), HOT-TOV (whole TOV solution on the
  grid, atmosphere from the first auxiliaries pass; no surface cut), test IDs (straight/curved beams,
  crossed beams, sphere wave, plane wave, scattering, emitting sphere, blast wave) used by the campaign.

---

## 6. The implicit collision step

`compute_implicit_update<s>`: Newton on (E, F_i) of one species with the matter frozen, unscaled ½|F|²
Armijo line search, **absolute** residual target 1e-15 (FIL: GSL hybridsj, scaled dogleg, relative
step) [CHOICE, keep-partial-iterate tried and reverted — it broke Hunter]. The initial guess relaxes over
the proper time α dt/W (1f6a92a3) [OK]. A thick-closure retry exists (bad at W ~ 5). Explicit thin
branch: when 4αW³κ dt < 0.1 the source is taken explicitly (`GRACE_M1_EXPLICIT_THIN`, bitwise except
near the limit) [OK]. Diagnostics: `m1_implicit_err`, `m1_implicit_res`, per-step log line.

Known limits: SUCCESS is only clean at E ~ 1 (absolute test: 0.6 % spurious failures at E = 1e-8, 15 %
at 1e-15); the 26.285 % failure rate in the halo is real but not the halo bug (Jacobian matches FIL;
guess/tolerance/linesearch/dogleg/clamp and FIL's no-deposit policy all tried and disproved or reverted
— read `m1-implicit-guess-tolerance-bug` before touching it); the v0 = 0.9 trapped pulse is 60–78 %
SMALLSTEP [OPEN, 12.A.12]. Halo c2p failures are |S|²/(2Dτ) ≈ 1 (no energy left for ε), reproduced by
`halo_c2p_test` [OK].

---

## 7. Matter coupling (backreaction)

`add_backreaction` (default `GRACE_M1_BACKREACT_HARDSTOP=1`): the realised exchange per species
(ΔE, ΔS_i, ΔN) of the implicit step is applied to (τ, S_i, D·Yₑ, D·Y_μ):

    dYₑ = (ΔN_νe − ΔN_ν̄e)/D,   dY_μ = (ΔN_νμ − ΔN_ν̄μ)/D,   dτ = Σ_s ΔE_s.

Lepton number is conserved flavour by flavour, so any new channel (pairs, decay) is bookkept correctly
without special code. Limiter: lepton pairs first (a pair that would push Yₑ or Y_μ off the EOS table is
reverted whole, or partially accepted with `muon_partial_at_bound` / `electron_partial_at_bound` — bits
8/16), then one summed energy decision (τ + ΣΔE > 0); the EOS bounds come from the host-fetched loaded
EOS (a default-constructed `eos_t` on the device gave uninitialised bounds — never do that) [OK].
`GRACE_M1_BACKREACT_HARDSTOP=0` is the scaled variant (throttle both sides by one factor). `rho_min`:
do not use. With `muon_decay` the connected `add_leptonic_backreaction` applies one θ to everything
(bit 32).

Known limits: neither variant limits the energy gain per baryon; the matter update is **lagged** behind
the implicit radiation step, so where the composition gain |dY_ν,eq/dY_l| > 1 the pair (matter, radiation)
2-cycles (salt-and-pepper Yₑ/Y_μ; dt-independent), and imex232 amplifies the lagged change 2.41× into the
second stage (unstable for gain > 0.41). The `timescale` policy removes most of it; the structural fixes
(Strang split, stage-2 rates from the stage-1 state, one Backward-Euler collision step as FIL does) are not
implemented [OPEN, 12.A.10].

---

## 8. The muon sector

### 8.1 Two regimes (measured in the head-on, Harry's picture)

Regime 1, dense and hot: muonisation happens and is **clipped at the lepton table's Y_μ ceiling 0.2**.
Regime 2, dilute and cold: demuonisation does not exist in the standard path (gate closed, clamp, no decay),
so 56 % of the unbound mass leaves with Y_μ > 0.01 and Yₑ underestimated by ≈ Y_μ [OPEN, 12.A.1].

### 8.2 The EOS side M1 depends on

Leptonic 4D EOS: baryon table **without electrons** (`sfho_compose_noele_0-5ye.h5`) + lepton table
with `add_ele_contribution=true` (the with-electrons table double counts) [OK]; log-pressure
interpolation ln(P_b + P_e) [OK]; temperature floor = table minimum (a cold star on T_min made every c2p
rewrite τ/S_i) [OK]; `keep` + `hermite_free_energy` switches cut the TOV core heating to 0.35× (FE 1.5–3.1×
slower per step, cause open) [OK]. Lepton table: the old Bollig table has a ×1.5(e)/×5(μ) P/ε bug, v2 has
a μ-grid defect for cold dilute muons, **v3** (`_ken_v3.h5`, 8 Oct, exact μ per point) fixes it — the
parfiles still point at v2 [OPEN, 12.B.4]. Cold-T muon interpolation is log-in-μ (the atmosphere c2p
errors) [OK]. Dilute-muon suppression (`eos.leptonic.dilute_muon_suppression`, default true): μ_μ →
μ_μ^tab·tanh((Y_μ − 6e-4)/5e-5), exactly 0 at Y_μ ≤ 6e-4, muon P/ε/s zeroed there.

Why it exists: at the Y_μ floor in cold matter the exact table gives μ_μ ≈ m_μ − T ln(n_Q/n_μ) ≈ 104 MeV,
so η_νμ ≈ +800. That is the EOS saying, correctly, that floor muons in cold dilute matter are far out of
equilibrium and should decay; the standard path has no channel for it and must hide the signal.
The suppression, the ±5 clamp and the rate gate are three stand-ins for the missing decay.

### 8.3 What Marie's modes change here — read this before running them

All of this exists only in `GRACE_M1_PAIR_KERNELS` builds (§9); production never sees it.
In `equilibrium`/`evolved` mode the EAS driver (`eas_policies.hh`, the `launch` lambda) **removes the ±5 clamp from
η_νμ, η_ν̄μ for the whole rates pass**, i.e. also for the table's charged-current Kirchhoff emission
Q = κ_a B(T, η_νμ) — not only for her pair partners ("both pair and charged-current fixed points must
agree"). With `muon_decay` it additionally **replaces η_νe, η_ν̄e by their unsuppressed chemical values**
(the τ interpolation of 4.2.1 is undone for the electron flavours) and **disables the muon rate gate**
(`apply_muon_gate = !muon_decay`), and the thermal extras of 4.4 are switched off in favour of her kernels.
Her decay also requires the suppression off (the ramp hides the muons from the decay kernel) and
`add_ele_contribution=true`. Consequences: (i) a decay run changes the electron-flavour emission physics in
free-streaming regions (η_νe no longer → 0 in the thin limit) — this is a side effect, not a documented
feature; (ii) the clamp that was load-bearing for κ_n(ν̄μ) and the beta-eq solve is gone in her modes;
both are untested beyond the 32-iteration Mac HOT-TOV runs [OPEN, 12.A.2]. (iii) `betaeq_policy` must be
`off` in her modes, so the stiff-cell protection of 4.6 is lost as well.

---

## 9. Marie's modes (`pair_treatment: equilibrium | evolved`, `muon_decay`)

**Fenced (14 Oct):** everything in this section exists only in builds configured with
`-DGRACE_M1_PAIR_KERNELS=ON` (default OFF; needs ≥ 3 species and the `extern/bns_nurates` submodule).
Production builds carry neither the code, the hooks in the rates driver nor the 12 aux slots, and are
bitwise the pre-fence standard path; a parfile asking for these modes in such a build stops at startup
and names the option. In a kernel build `muon_decay` with `standard` is still the vacuum sink of §4.4.

Files: `neutrino_pair_collision.hh` (bns_nurates kernels: Bruenn e⁺e⁻ with cached Fermi integrals,
HR98 brems hoisted per cell, transverse plasmon; Gauss–Legendre grid with a per-cell scale),
`neutrino_pair_update.hh` (coupled two-species implicit solve, relative Newton tolerances),
`neutrino_leptonic_update.hh` (four-species solve with decay), `neutrino_muon_decay.hh` (finite-mass
decay kernel R0, detailed balance), `neutrino_pair_diagnostics.hh` (failure records), hooks in
`eas_policies.hh::prepare_pairs/prepare_pair_transport` (writes `PAIR_*`, LTE rates for `equilibrium`)
and the launch in 2.3. Docs: `pair_processes.md`, `muon_decay.rst`, `muon_pair_processes.rst` (audit),
`pair_implementation_plan.md`, `muon_decay_plan.rst`.

What they add that the standard path lacks: (a) pair annihilation against the **actual** partner
population and blocking by the real distribution (evolved; the only part a thermodynamic table cannot
hold); (b) a **muon decay / inverse decay** channel; (c) HR98 brems with degeneracy at LTE; (d) no
Kirchhoff inversion for the pair channels. What they do not add: better e⁺e⁻ pairs (Ruffert is within 3 %
of her kernel and 10 % of Harry's), a better plasmon (same model as Ruffert), angular dependence (Φ₀ only,
so beam–beam annihilation geometry is not captured).

Requirements enforced at startup: 3/5 species, `GRACE_HAVE_BNS_NURATES`, analytic or Weakhub EAS with
`weakhub_table_content` resolving to `none` (a withPP table would double count), `betaeq_policy: off`,
`pair_plasmon_model: transverse_mass` if `plasmon_decay`; decay: 5 species, leptonic EOS,
`dilute_muon_suppression=false`, `add_ele_contribution=true`, a table without inverse lepton decay.

Cost (Mac 64³ HOT-TOV, star in 5 % of the cells, 32 its): standard 283 s, equilibrium 334 s, evolved 407 s
(after the 13 Oct speed-ups: keyword `standard`, aux 16→12, FDI caching, brems hoisting, one kernel
build per cell and launch; all bitwise except the shared quadrature grid in evolved mode, ≤ 2.3e-4 in
νμ, ≤ 5e-3 in τ, below the order-16→24 change) [OK].

Verification: `m1_pair_test` (kernel bitwise vs the library, LTE, stiff HOT-TOV core, two-moment vs
spectral relaxation ≤ 0.9 %/0.35 % from vacuum, physics vs GRACE fits), `muon_decay_test`,
`muon_decay_setup_rejection`; Harry's spectral tables (section 11). Her two crashes at iteration 0 are
fixed (underflowing FD weights → shifted Boltzmann weights; Newton stall at x ~ 1e4 → relative
tolerances).

Open: (D) `muon_decay` + `evolved` aborts at iteration 1 with non-Fermi-realisable νμ moments
(J < ¾ ε_F n) in 24 cells at the hot-slice edge (ρ 8e-6, T 17.8, μ_μ 46 MeV) — a modelling choice is
needed (project to the realisable boundary for the transport opacity/seed, or reject the step)
[DECISION]; decay + `equilibrium` never run; 8.3 (i)–(iii); the equilibrium partner uses the unclamped
η_νμ, i.e. a fictitious sea in cold dilute cells (harmless at 6.7e3 s⁻¹, not physics); plasmon vs Harry.

---

## 10. Build matrix and parameter map

Build: `GRACE_ENABLE_M1` (derived, cache-internal), `GRACE_M1_NU_SPECIES` 0/1/3/5, `GRACE_ENABLE_MUONS`
(forced on at 5), `GRACE_M1_OPTICAL_DEPTH`, `GRACE_M1_PHYSICS_OUTPUT` (default ON), `GRACE_M1_DIAGNOSTICS`, `GRACE_M1_PHOTONS`,
`GRACE_M1_PAIR_KERNELS` (default OFF: Marie's kernel modes, §9), `GRACE_HAVE_BNS_NURATES` (submodule
present), `GRACE_ENABLE_FOFC`, `GRACE_FREEZE_HYDRO`; defines
`GRACE_M1_EXPLICIT_THIN`, `GRACE_M1_BACKREACT_HARDSTOP`, `GRACE_M1_IMPLICIT_LB`. Mac: Release only,
`OMP_NUM_THREADS=4`; Kokkos 5.1.1 needs explicit 0 extents for label-only Views.

`parameters/m1.yaml` (731 lines). Production values are those of `~/Graveyard/HOTTOV/leptonic_hottov.yaml`
(= the Hunter HOT-TOV and, with the noted exceptions, the head-on).

| key | meaning | production |
|---|---|---|
| `bc_kind`, `id_type` | boundary kind; initial data (zero, hot_tov, test IDs) | zero / hot_tov |
| `atmosphere.E_fl, N_fl, r_damping, E/N_scaling, eps_fl/min/max/scaling` | FIL-style radial floors | 1e-16, 1e-14, 50 |
| `excision.excision_lapse/radius, E_excision, eps_excision` | M1 excision | hydro values (head-on 0.1) |
| `eas.kind / kinds` | provider(s): test, photon_rates, neutrino_analytic, neutrino_weakhub, bns_nurates | `["neutrino_weakhub"]` |
| `eas.weakhub_table` | the grey table | `SFHocompose_grey_npemu_Bollig_suppress_noPP_advopac_20240818` |
| `eas.beta_decay` | analytic CC (analytic path only) | true |
| `eas.pair_annihilation, plasmon_decay, bremsstrahlung` | thermal extras (heavy flavours), both paths | true |
| `eas.temperature_correction(_emission)` | (T_ν/T)² on κ (and on Q, R) | true / false |
| `eas.tau_policy, tau_outer_radius_code` | optical depth for the η interpolation | eikonal |
| `eas.betaeq_policy, betaeq_partial_e_fallback, betaeq_fallback_ymu_floor` | 4.6 | timescale, true, false |
| `eas.pair_treatment, pair_quadrature_order, pair_trace_iterations` | Marie's pair modes (kernel builds only) | standard, 16, 0 |
| `eas.muon_decay, muon_decay_kernel_order` | muon decay: the vacuum sink in `standard` (§4.4), her kernel in kernel builds | false, 24 |
| `eas.weakhub_table_content` | what the table carries: auto (companion `_tableinfo.log`), none, thermal, thermal+decay; thermal switches the analytic extras off, decay forbids `muon_decay` | auto (repo parfiles: none) |
| `eas.pair_plasmon_model` | kernel builds only | none |
| `backreaction.enabled, t_start, rho_min, muon/electron_partial_at_bound` | section 7 | true, −1, 0, false |
| `trigger.enabled, separation` | activation from the co_tracker | head-on: true, 30 |
| `*_test` blocks | test-ID parameters (beams, waves, scattering, emitting sphere) | campaign |
| `time_stepper`, `cfl_factor` (elsewhere) | imex232, 0.2 | |

EOS keys that M1 depends on: `eos.leptonic.add_ele_contribution`, `dilute_muon_suppression`,
`linear_pressure`, `keep`, `hermite_free_energy`, the two table paths.

---

## 11. Verification inventory

**Tests** (never run bare `ctest` on the Mac; use `ctest -L fast` plus names): `m1_pair_test`,
`muon_decay_test`, `muon_decay_setup_rejection`, `m1_analytic_rates_test` (the two sigmoid-gate failures
are gone since the merge), `m1_backreaction_test`, `m1_number_flux_test`, `m1_curved_beam_number_test`,
`m1_test_id_test`, `m1_optical_depth_test`, `betaeq_fallback_test`, `weakhub_table_test` (species clump,
1e-60 floor), `weakhub_golden_test`, `halo_c2p_test`, the leptonic c2p/free-energy/log-P suite, the
campaign runs (`m1;integration`). Tests need `configs/basic_config.yaml`: run from `build-*/test`.

**Reference runs**: Mac HOT-TOV template (`~/Graveyard/HOTTOV/mac_verify`, 64³ octant, 32 its ≈ 5 min,
`compare_runs.py` bitwise plane comparison; binaries kept per experiment); curved beam A/B
(`~/Graveyard/m1_scratch/fofc_ab`); the 29-test campaign (`~/Graveyard/m1_scratch/campaign`); production
on Hunter (user compiles and submits; `build-optd-cur`).

**FIL parity established** [OK]: beta-equilibrium Yₑ (0.06014), Qnp convention, flux A-factor (symmetric
to 1e-12), number flux, atmosphere floors, output conventions, luminosity balance after the √γ fix, the
same Weakhub table (npe 0906 / npemu 0818), the FIL lepton-table ln-axis trap (GRACE row-aligns, immune).
Deliberate departures [CHOICE]: νx opacities averaged; Q, R not spectrally scaled; electron pairs off;
absolute implicit residual; F·v clamp removed; wavespeed cap; atomic hard-stop backreaction.

**Against Harry's spectral Weakhub kernels** (iota, read-only, 14 Oct; scripts in
`~/Graveyard/m1_scratch/pair_check`): e⁺e⁻ pairs 1.01–1.09 (2024 SFHo PP table) and 0.96–1.00 (12-Oct
DD2 muonic table) of Harry, electron and heavy — so GRACE's Ruffert term is fine and the 22 Sep
"1.2–2.2× too high" note was wrong; plasmon 0.07–0.15× Harry's γPP [OPEN]; muon decay with GMUNU's ½
prefactor: μ⁻ 0.99–1.19, μ⁺ 1.13–1.20, mean energies agree. Harry's grey tables with pairs + brems folded
in exist (`…fixedBBforPP_advopac_20240809`), none has plasmon or decay.

**Known symmetry facts**: HOT-TOV is a mirror-symmetry test for ~20–50 iterations only (roundoff in the
surface blast front); the head-on asymmetry is hydro-led (entropy-backup c2p prime suspect; the T_ν floor
amplifier accepted by the user); the default build is not bitwise mirror-symmetric in a full-cube TOV
(1e-7, seed unlocated).

---

## 12. Open problems

### A. Physics and correctness

| # | item | status / what is known | decision needed |
|---|---|---|---|
| 1 | **Demuonisation** (regime 2): 56 % of unbound mass with Y_μ > 0.01 | IMPLEMENTED 14 Oct as the vacuum decay sink (`muon_decay: true`, §4.4): per decay ~37 MeV to e⁻ (1.85 MeV/baryon at Y_μ = 0.05), ~70 MeV to ν_μ + ν̄_e; Y_μ e-folds in 2.2 μs. Mac HOT-TOV 32 its (`~/Graveyard/HOTTOV/mac_verify/G_sink_gated` vs `F_standard`, `sink_diff.py`): surface/floor cells with leftover muons (Y_μ up to 1.0e-2) decay to the dilute band 6.6e-4, Yₑ +1.1e-2, T +1 %; shell max ΔY_μ 5.7e-3; core untouched to 5e-7 in Y_μ (ungated it drifted 1e-6 per 32 its, i.e. a slow drain — hence the gate) | validate: one head-on on Hunter with the outflow `Mdot_ymu`/`Mdot_ye` bins (user submits) |
| 2 | Marie's modes drop the ±5 clamp, the τ interpolation of η_νe (decay) and the muon gate (decay) | RESOLVED for production by the fence (kernel builds only); documented in §8.3 for whoever runs the kernel modes | none |
| 3 | (D) decay + evolved realisability abort (hot-slice edge) | transport/realisability, 24 cells; needs projection or rejection policy | yes |
| 4 | Pair channels inverted with the CC η_νμ (Kirchhoff) | `docs/m1-neutrino-degeneracy-roles.md`; ~40 lines; FIL does the same | yes (no code without a go) |
| 5 | Yₑ-floor ejecta from τ-suppressed η_ν in the Kirchhoff emission | open discussion in the same doc; `work_full_fix` run | yes |
| 6 | Core filling (E/N falls inward) and drained-N cells in the merged core | driven by numerical outer-layer heating (first-step surface energy error, 150–300× less at dx 0.125); not kick/dt/atmo/T-floor/refill | hydro track |
| 7 | GRACE heats cold stars more than FIL; pre-M1 surface-layer Yₑ differs; pre-contact ν̄e excess | 2nd order in dx in FV and FD4 (not smooth truncation error); surface-noise + floor-rectifier hypothesis with 4 cheap tests proposed | hydro track |
| 8 | M1 atmosphere: `eps_fl` 2600× too big, FOFC floor test blind band | catalogued, not fixed (r_damping now covers the radial fall-off) | low |
| 9 | Plasmon model 7–15× below Harry's γPP; brems fit 1.2–2.6× above HR98 | subdominant to brems at high density; brems only through a table with emissivities or her kernels | ask Harry |
| 10 | Composition 2-cycle / IMEX stage amplification (gain > 0.41) | timescale policy mitigates; Strang / stage-2 rates / one BE step not implemented | medium |
| 11 | Beta-eq failures at table floors and in flooded cells (34–49 % of rejections) | partial-E fallback exists; active-set fallback proposed | medium |
| 12 | Implicit solver: 26 % failure rate in the halo, SUCCESS only clean at E ~ 1, v0 = 0.9 trapped pulse | all simple variants tried; read the memory note first | low unless it bites |
| 13 | Lepton table Y_μ ceiling 0.2 (clips muonisation) and floor 5e-4 | table regeneration (v3 generator) | with 1 |
| 14 | Head-on start-up beta-eq failures (3924 then 0, symmetric); excision flicker at 0.1 | cosmetic / known | none |
| 15 | Muon-sector Kirchhoff with a table that has no emissivities | a withPP table would re-emit pair terms at the CC η (worse than today) — do not switch tables without emissivity support | design note |

### B. Numerics and infrastructure

| # | item | status |
|---|---|---|
| 1 | Performance: M1 = 64 % of a step (implicit 31 %); five EAS passes per step; dead ζ seed in `update_closure`; designed fixes 2a/2b/4/1 not implemented; evolved-mode FD Jacobian, cold-start reconstructions | roadmap exists (`m1-perf-roadmap`) |
| 2 | FOFC: emf out-of-range read in the ghost layer; ~35-cell 5-ULP residual (left on purpose) | open / accepted |
| 3 | Hunter GPU: 3 GPU-only test failures the Mac cannot reproduce; ROCm 7.0.2 miscompile (runs at −O1, ticket filed) | external |
| 4 | Lepton table v2 (cold-dilute μ defect) → v3 | DONE on the Mac 14 Oct (configs, parfiles, measured: cold dilute muon sector only); Hunter needs the v3 file copied |
| 5 | NaN termination checkpoint trap (restart loads the NaN checkpoint) | known procedure |
| 6 | c2p error bitmask documentation off by one at 5 species | doc fix |
| 7 | 16 local commits on `fix/leptonic-fil-parity` unpushed (pair fixes, speed-ups, physics tests, dumps, fence, sink, table content) | push when asked |
| 8 | Marie's realizability-dependent features have no Hunter run yet (kernel builds) | research track |
| 9 | Photon-only builds (`GRACE_M1_NU_SPECIES=0`, photons ON) did not compile since the M1 FOFC tagging and the trigger reset | FIXED 14 Oct (species guards) |
| 10 | The `dens_integral` scalar's last digits follow the OpenMP thread layout (reduction order), not the code; plane fields are bitwise | compare scalars at equal launch settings |

---

## 13. What has to be done — proposed order

Each item names its verification. Nothing here is started without a go.

**Phase 0 — housekeeping (days).** Push the commits (20 local as of 14 Oct). v3 lepton table: DONE on
the Mac (test configs, `HotTOV_leptonic*.yaml`, mac_verify parfiles; Mac HOT-TOV 32 its v3 vs v2: core
unchanged to 6e-7, shell max ΔY_μ 2.2e-4 / ΔT 0.4 %, floor ΔY_μ 1.9e-4, the cold-dilute muon rates
(κ_a, η of νμ) up to 31× — the v2 defect; same beta-eq fallback counts; the sink's effect identical on
v3). Still to do: copy `sfho_leptons_noele_0-5ye_ken_v3.h5` to Hunter's `EOS/4D/` (the parfiles already
name it). Fix the c2p bitmask doc. Keep this guide current.

**Phase 1 — the muon-sector science gap. DONE in code 14 Oct** (fence C, vacuum decay sink D, table
content E, this guide). What remains is the measurement: one head-on on Hunter with `muon_decay: true`
(user submits) with the outflow `Mdot_ymu`/`Mdot_ye` bins and the species luminosities as the result,
then the lepton table bounds (12.A.13) if they turn out to matter. Her decay stays the research route in
kernel builds (12.A.3: realisability policy; 8.3: the clamp and the τ interpolation). In both cases
regenerate the lepton table with a lower floor / higher ceiling (12.A.13) if the measurement shows the
bounds matter.

**Phase 2 — standard-path physics debts (FIL-parity relevant, each a small, testable change).**
12.A.4 (η roles in the muon flavour, ~40 lines, HOT-TOV), 12.A.5 (Yₑ-floor ejecta and the Kirchhoff
emission, needs the decision first), 12.A.9 (plasmon and brems: ask Harry; brems via a table with
emissivities would need ~50 lines of reader + a switch to drop the analytic extras).

**Phase 3 — numerics.** 12.A.10 (composition 2-cycle: stage-2 rates from the stage-1 state is the
cheapest; Strang if not enough; measure on the head-on with the salt-and-pepper diagnostic), 12.B.1
(perf: fewer EAS passes per step first — it is the largest and bitwise-checkable; then the closure seed;
then the evolved-mode Jacobian), 12.A.11 (active-set fallback for the beta-eq solve).

**Phase 4 — hydro-side items that limit M1 comparisons.** 12.A.6/7 (outer-layer heating and the
FIL–GRACE Yₑ gap before contact; the four cheap tests; resolution study at dx 0.125), the head-on
asymmetry. These are not M1 changes but they set what an M1 comparison with FIL can show.

**How to verify any change in this sector**: (1) the fast test subset plus the named M1 tests;
(2) Mac HOT-TOV 32 its, bitwise vs the kept reference binary where the change is meant to be inert,
otherwise the per-field relative change table (`evolved_diff.py`, `compare_runs.py`); (3) the curved
beam for anything touching transport or FOFC; (4) one Hunter run with the outflow bins and the species
luminosities for anything touching the muon sector or the backreaction. Report the numbers, including
the failures.

---

## 14. Where things live

| area | files |
|---|---|
| step driver | `src/evolution/evolve.cpp` (explicit/implicit substeps, FOFC), `src/evolution/auxiliaries.cpp` |
| M1 equations | `include/grace/physics/m1.hh` (closure aux, fluxes, sources, implicit update, backreaction), `m1_helpers.hh`, `m1_subexpressions.hh`, `m1_number_flux.hh`, `m1_trigger.hh`, `src/physics/m1.cpp` (EAS driver, ID, reports) |
| rates | `eas_kinds.hh` (selection, parameter checks), `eas_neutrino_rates_analytic.hh` (fugacity state, kernels, Kirchhoff, weakhub combination), `eas_policies.hh` (driver, τ, beta-eq policies, Marie's hooks, diagnostics), `eas_optical_depth.hh/.cpp`, `grace_weakhub_table.hh` |
| Marie's modes | `neutrino_pair_collision.hh`, `neutrino_pair_update.hh`, `neutrino_leptonic_update.hh`, `neutrino_muon_decay.hh`, `neutrino_pair_diagnostics.hh`, `extern/bns_nurates` |
| EOS | `include/grace/physics/eos/leptonic_eos_4d.hh`, `scripts/gen_lepton_table.py` |
| initial data | `include/grace/physics/id/m1_initial_data.hh`, `hot_tov.hh` |
| output | `src/IO/output_diagnostics.cpp`, `src/IO/diagnostics/outflow_diagnostics.cpp` |
| parameters | `parameters/m1.yaml`, `parameters/outflows*`, EOS yaml |
| tests | `test/test_m1_*.cpp`, `test_muon_decay.cpp`, `test_weakhub_table.cpp`, `test/CMakeLists.txt` |

Other documents: `docs/m1-neutrino-degeneracy-roles.md` (which η belongs where), `docs/m1-muon-sector-plan.md`
(muonisation/demuonisation plan, steps A–C), `docs/leptonic-eos-fil-parity.md`, `doc/developer_guide/pair_processes.md`,
`muon_decay.rst`, `muon_pair_processes.rst` (audit of the muonic pair channels), `pair_implementation_plan.md`,
`muon_decay_plan.rst`, `docs/hunter-rocm-7.0.2-miscompile-report.md`. Scratch and analysis:
`~/Graveyard/m1_scratch` (campaign, pair_check, fofc_ab, headon/hottov scripts; use
`/Users/miler/Codes/py-work/bin/python`), `~/Graveyard/HOTTOV/mac_verify`, `~/Graveyard/Headon`.
