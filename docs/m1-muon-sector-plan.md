# Plan: muonization and demuonization in GRACE's muonic sector

**Status:** plan only (22 September 2026), nothing implemented. Written after Harry's
remark that both regimes — muonization where ρ and T rise, demuonization where matter
cools and expands — must be right, and after measuring both in the head-on run
`work_full_fix` (see the summary at the end).

## 0. What is right and what is missing today

| Regime | Mechanism in GRACE | State |
|---|---|---|
| Muonization (hot, dense, μ_e > m_μ) | ν_μ/ν̄_μ absorption from the Weakhub κ_a, emission by Kirchhoff, β-eq "timescale" closure in trapped matter | works, but Y_μ is clipped at the table ceiling 0.2 (both tables) |
| Demuonization by capture μ⁻ + p → n + ν_μ (dense, Y_μ above equilibrium) | Kirchhoff emission κ_a(ν_μ)·B(T, η_νμ) | throttled: η_νμ clamped at ±5, and the rate gate closes below 10¹¹ g cm⁻³ / 2.5 MeV |
| Demuonization by decay μ⁻ → e⁻ + ν̄_e + ν_μ (cooling, expanding, μ_e < m_μ) | none | missing entirely: Y_μ is frozen in the ejecta |
| Dilute-Y_μ floor regularisation (GMUNU note) | `block_mumu`, `muons_resolved`, `muon_rate_gate` | present; keep |

The order below puts the self-contained physics first and the delicate rate changes
last.

## 1. Measure before changing (no code)

1. **Offline equilibrium muon fraction.** A script on the Mac that solves μ_μ(ρ,T,Y_e,Y_μ) = μ_e(ρ,T,Y_e,Y_μ)
   on the two tables (no trapped neutrinos) and, for trapped matter, the joint condition
   μ_μ − μ_νμ = μ_e − μ_νe with the closure's neutrino numbers. Output: Y_μ,eq(ρ,T,Y_e).
2. **Compare with the runs.** In `work_full_fix` and the HOT-TOV data, plot Y_μ / Y_μ,eq
   against τ_β/t_dyn. Cells with τ_β ≪ t_dyn must sit at Y_μ,eq: that is the muonization
   test. Cells that left the hot region show what demuonization has to remove.
3. **Map the handover.** From the tables, μ_e along the gate surface
   S(ρ,T) = 0.5. Decay must be unblocked (μ_e < m_μ) wherever the gate is closed;
   the numbers say it is (μ_e ≈ 10–20 MeV at 10¹¹ g cm⁻³), but check.

## 2. Step A: the muon-decay channel (regime 2)

**Physics.** μ⁻ → e⁻ + ν̄_e + ν_μ, τ_μ = 2.197 μs in vacuum. Per muon:

    Γ = (1/τ_μ) · (m_μ/E_μ) · [1 − f_e(ε_e)],   f_e = 1/(1 + exp((ε_e − μ_e)/T)),   ε_e ≈ m_μ/3,

with E_μ ≈ m_μ below a few MeV. The electron blocking switches the channel off where
muons belong (μ_e > m_μ) and on where they must go (μ_e ≪ m_μ). Inverse decay and
μ⁺ decay are neglected (they need a neutrino bath / are thermal-pair bookkeeping).

**Sources per decay** (electron-lepton number: e⁻ +1, ν̄_e −1 = 0; muon-lepton number:
μ⁻ −1, ν_μ +1 = 0, so GRACE's Y_le and Y_lμ bookkeeping is unchanged):

    dY_μ/dt = −Γ Y_μ            dY_e/dt = +Γ Y_μ
    R(ν_μ) = R(ν̄_e) = Γ n_μ     Q(ν_μ) = Q(ν̄_e) = Γ n_μ · (m_μ − m_e)/3 ≈ Γ n_μ · 35 MeV
    fluid energy: −Γ n_μ · 2(m_μ − m_e)/3 ≈ −70 MeV per muon

The fluid loses the neutrino energy; the rest-mass difference m_μ − m_e and the
electron's kinetic energy are handled by the EOS once (D, τ, Y_e, Y_μ) are updated,
because the leptonic EOS carries the lepton rest masses.

**Time integration.** Γ dt is 0.14 at dx = 0.25 and up to 9 on the coarsest level, so
update Y_μ analytically, Y_μ → Y_μ e^{−Γdt}, and give the decayed amount ΔY_μ to Y_e and
to the two radiation species as an explicit isotropic source (the emission machinery
of the collision term, η only). Non-stiff by construction: nothing to add to Newton.

**Where.** Compute Γ in `set_m1_eas` (the fugacity state has ρ, T, μ_e, Y_μ) and store
it in a new aux `m1_mu_decay` (diagnostics group). Apply it in `add_backreaction`
(`include/grace/physics/m1.hh`), inside the existing atomic hard-stop structure, so a
rejected step rejects matter and radiation together and lepton conservation stays
exact. Y_μ never goes below the table floor.

**Parameters** (`parameters/m1.yaml`): `m1.muon_decay.enabled` (default false),
`tau_mu` (2.197e-6 s), `eps_e_fraction` (1/3). Compile switch not needed.

**Unit tests** (`test/test_m1_backreaction.cpp`):
- cold dilute cell (T = 1 MeV, 10⁸ g cm⁻³, Y_μ = 0.02): Y_μ decays on τ_μ, Y_e rises by
  0.02, radiation gains 70 MeV per muon split equally, fluid loses the same;
- degenerate cell (μ_e = 200 MeV): Γ = 0 to machine precision;
- lepton numbers Y_le, Y_lμ and total energy conserved per cell to round-off;
- hot cell (T = 30 MeV, Y_μ at equilibrium): Γ_decay ≪ the capture rates, so the
  channel is invisible where the table physics rules.

**Production check.** Head-on M = 1 + 1 with M1: the Y_μ ejecta bins at DET_100 must
collapse into the floor bin and the Y_e bins shift up by about the ejecta's former Y_μ
(≈ 0.02); ν_μ and ν̄_e luminosities gain a contribution from the ejecta. Compare with
the run without the channel.

Effort: about 150 lines plus tests; one Hunter run.

## 3. Step B: the two roles of η in the muon flavours (regime 2, dense; regime 1 drive)

**Change.** Keep the unsuppressed, unclamped matter value η_eq in `fugacity_state`
(same array as proposed in `docs/m1-neutrino-degeneracy-roles.md`) and:
- row 1, table κ → emission (`Q = κ_a B(T,η)`): use η_eq for ν_μ and ν̄_μ. This is the
  capture channel; with B ∝ (Tη)⁴ = μ_ν⁴ in the degenerate limit it gives the full
  μ⁻ + p → n + ν_μ rate that the clamp currently caps at ~100× thermal;
- row 2, pair/plasmon/bremsstrahlung Q → κ (`add_kirchhoff_absorption_opacity_from_QR`):
  use η = 0 for all species. These processes are flavour-symmetric and their blocking is
  already inside Q; dividing by B(η_matter) is what produced κ_n(ν̄_μ) ≈ 10⁴³;
- rows 4 and 5 (T_ν correction, β-eq closure) keep a bounded η.

**Why this is not the rejected clamp removal.** Both earlier removals failed through
row 2 (κ_n blow-up) and the closure; with row 2 at η = 0 the first cannot happen, and
the closure keeps its bound. It has to be re-tested on the same production
discriminator (5 species, eikonal τ, HOT-TOV): κ_n(ν̄_μ) must stay at the table scale
and the β-eq failure count must not rise.

Effort: about 40 lines; one HOT-TOV run.

## 4. Step C: the Y_μ ceiling (regime 1)

Both tables end at Y_μ = 0.2 (lepton table `ymumax`, Weakhub `logymu` axis). The hot
interface of the 2×1.24 head-on reached it at μ_e ≈ 350 MeV and it may be reached in
any hot remnant.

- Regenerate the lepton table with `gen_lepton_table.py` to `ymumax = 0.4`
  (log axis, a few more points); validate as before against SFHo.
- Ask Harry for a Weakhub table with the same axis. Until then the hard stop keeps
  Y_μ ≤ 0.2 and `m1_br_reject` counts the clipped cells — report that number.
- After Step 1 says what Y_μ,eq is at the interface, decide whether 0.4 is enough.

## 5. Step D: the gate and the dilute prescription (keep, make testable)

- Expose ρ₀ and T₀ of `muon_rate_gate` and Y_μ,0, ΔY_μ of `block_mumu` as parameters
  (defaults unchanged: 10¹¹ g cm⁻³, 2.5 MeV, 6×10⁻⁴, 5×10⁻⁵).
- Sensitivity pair on Hunter (HOT-TOV or the head-on): Y_μ,0 = 3×10⁻⁴ / 1.2×10⁻³ and
  ρ₀, T₀ halved / doubled. The GMUNU note asks for exactly this and it has never been
  done. Results should not move; where they do, the prescription is doing physics.
- Known caveat to document, not fix: zeroing the muon P/ε/s below Y_μ,0 drops real
  thermal μ⁺μ⁻ pairs where T ≳ 20 MeV. Rarely bites (hot matter has Y_μ ≫ Y_μ,0).

## 6. Step E (later, optional): flavour exchange

μ⁻ + ν_e → e⁻ + ν_μ and μ⁻ + ν̄_μ → e⁻ + ν̄_e convert muons using the neutrino
fields. They are absorption opacities of ν_e / ν̄_μ proportional to n_μ and could be
added as extra κ_a terms once the two channels above are in. Second order for the
ejecta; not needed for the two regimes Harry named.

## 7. Order, and what each step needs from whom

1. Step 1 (offline, Mac, no code) — establishes the target Y_μ,eq and the handover map.
2. Step A (decay) — self-contained, addresses the missing regime; unit tests first,
   then one head-on run.
3. Step C's lepton-table regeneration can run in parallel (Mac); the Weakhub half
   needs Harry.
4. Step B — after A is validated, so a change in the ejecta can be attributed.
5. Step D — the sensitivity pair, once A and B are in.

Departures from FIL: A and B both leave FIL parity in the muon sector (FIL freezes
Y_μ below 10¹¹ g cm⁻³ and has no decay). GRACE's muon treatment is already the newer
one; this extends it.

## 8. The measurements this plan rests on (head-on `work_full_fix`, 22 Sep 2026)

- Y_μ reaches exactly 0.2000 (the table ceiling) from iteration 4864; the β-eq
  "bound-limited" failures and Y_μ hard-stop rejections are this ceiling.
- Pre-collapse cores (T 4–5 MeV, μ_e > m_μ in ~87 % of cells): median Y_μ 0.019 → 0.021
  over ~500 M1-active iterations; equilibrium value unknown (Step 1).
- Transparent ejecta (T < 1 MeV, ρ ≈ 6×10⁷ g cm⁻³): Y_μ up to 0.16 at it 6400 and 0.06 at
  it 10112; the rate gate is 10⁻¹⁴…10⁻²² there; `m1_muon_source` is zero.
- Unbound mass at DET_100: 56 % leaves with Y_μ > 0.01, peaking at 0.017–0.027, the
  stars' original value. Physically these muons decay in 2.2 μs and give Y_e ≈ +0.02.

## 9. Implementation design for Step A and Step B (22 Sep 2026, discussed, not executed)

### The rates pipeline as it is (Weakhub path, per cell, `set_m1_eas`)

1. `make_fugacity_state` (eas_neutrino_rates_analytic.hh:~460): EOS → μ_e, μ_μ (dilute-blocked), μ_p, μ_n,
   X_i; η_e, η̂, η_ν per species (μ_νe = μ_e + μ_p − μ_n − Q_np, μ_νμ = μ_μ + μ_p − μ_n − Q_np, η_νx = 0);
   muon flavours clamped to ±5 (:565); τ per species from the policy; η_νe, η_ν̄e multiplied by (1 − e^−τ) (:590).
2. β-eq "timescale" closure (eas_policies.hh:1077): if τ_β/dt < 1 for a lepton species, solve the joint
   (T, Y_e, Y_μ) equilibrium with the trapped neutrino numbers, blend, and re-evaluate ALL rates at the
   equilibrated state — rates only, no grid function written.
3. `compute_all_species_weakhub` (:955): table κ_a(E), κ_a(N), κ_s per species at (ρ, T, Y_e, Y_μ);
   Kirchhoff emission Q = κ_a B_E(T, η_ν), R = κ_n B_N(T, η_ν) for every species (row 1);
   analytic pair (heavy flavours only; the electron-flavour lines are commented out), plasmon (heavy only),
   bremsstrahlung (heavy only) accumulated in `extra` and turned into opacities by Q/B(T, η_ν) (row 2);
   muon gate S(ρ, T) on the (Q, κ_a) and (R, κ_n) pairs of ν_μ, ν̄_μ; `tau_post` feeds the eikonal sweep;
   (T_ν/T)² on the κ's of ν_e, ν̄_e (κ_s only for ν_x, nothing for the muon flavours), T_ν from
   F₂/F₃(η_ν)·⟨ε⟩ of the ACTUAL field; floor 10⁻⁶⁰; conversion to code units.
4. Collision update (m1.hh:447–): E, F implicit with κ_a, κ_s, η; N with κ_n/Γ and η_n
   (m1_helpers.hh:419); explicit branch when 4αW³(κ_a+κ_s)dt < 0.1.
5. `add_backreaction` (m1.hh:648): old − new per species; Y_e from N(ν_e) − N(ν̄_e), Y_μ from
   N(ν_μ) − N(ν̄_μ); hard stop at the EOS bounds; one energy decision; rejected species reverted.

### Observations that shape the design

- **The backreaction already converts any emitted neutrino into a composition change.** A ν_μ that appears
  in the radiation lowers Y_μ by one per baryon; a ν̄_e that appears raises Y_e by one; the fluid loses the
  emitted energy. So muon decay needs NO new kernel: if it enters the rates as an emissivity
  (η_N, η_E) of ν_μ and ν̄_e with no absorption partner, the existing collision update emits it and the
  existing backreaction does Y_μ → Y_e, the energy loss, the hard stop and the diagnostics, conservatively
  and atomically. The EOS then accounts for the rest-mass difference m_μ − m_e by itself, because the leptonic
  EOS carries the lepton rest masses. Total energy (fluid + radiation) is conserved to round-off.
- **It must be added AFTER `add_kirchhoff_absorption_opacity_from_QR`.** Everything in `extra` is inverted
  into an absorption opacity; a decay has no inverse in the radiation field (the inverse needs a ν_μ AND a
  ν̄_e, and is negligible where the decay is open). Adding it to `extra` would create a fake κ.
- **It must be added AFTER the muon gate**, or the gate kills it exactly where it is needed. (GMUNU exempts
  ILD from the Bollig suppression for the same reason.)
- **B(T, η) is a polynomial for η ≫ 0, not an exponential.** B_E ∝ T⁴F₃(η) → (Tη)⁴/4 = μ_ν⁴/4: the degenerate
  capture limit. The measured η_νμ ≈ 84 at the T floor gives μ_ν ≈ 8.5 MeV, i.e. a finite, physical rate.
  The blow-up that motivated the ±5 clamp lives only in row 2 (Q/B with η ≪ 0). So row 1 can use the
  unclamped η once row 2 no longer uses η.
- **Energy/number consistency of the emission:** ⟨ε⟩_emitted = Q/R = (κ_a/κ_n)·B_E/B_N. The table's κ_a and
  κ_n are spectrum-averaged by Weakhub at SOME spectrum — to be confirmed with Harry which η that spectrum
  used; if it is the matter's η_eq, row 1 with η_eq is the consistent partner and row 1 with η = 0 is not.
- **E and N are relaxed independently** (κ_a, η vs κ_n/Γ, η_n), so ⟨ε⟩ = E/N is unconstrained after the
  solve (we have seen 0.5 MeV and 415 MeV). A mean-energy limiter (clamp ⟨ε⟩ to [ε_min, ε_max] by
  adjusting N) is a separate, small item.
- **Check item:** `weakhub.lookup(rho_code, T, F.ye, F.ymu)` passes the electron fraction; confirm the
  table's "ye" axis is Y_e and not Y_p (GMUNU: "if no muonic table, ye = yp").

### Step A in code

- `parameters/m1.yaml`, `m1.eas.muon_decay`: `enabled` (false), `tau_s` (2.197e-6; GMUNU's conservative
  2.2e-5 as an option), `eps_frac_numu` (7/20), `eps_frac_nuebar` (3/10).
- `fugacity_state` unchanged. New POD `muon_decay_params_t { bool enabled; double tau_s, ymu_min,
  f_numu, f_nuebar, dt_code; }` filled in the EAS policy constructor (ymu_min from the loaded EOS,
  dt per call — the policy already has dt for the β-eq gate).
- New function in eas_neutrino_rates_analytic.hh, called in `compute_all_species_weakhub` after the gate
  and before `tau_post`:

      add_muon_decay_emission(F, p, rates):
        if (!p.enabled) return;
        n_avail = F.nb * max(F.ymu - p.ymu_min, 0)              // cm^-3
        eps_e   = 0.35 m_mu (+ m_e)                              // MeV, total electron energy
        block   = 1 - 1/(1 + exp((eps_e - F.mu_e)/T))            // GRACE mu_e includes the rest mass
        Gamma   = block / p.tau_s                                // s^-1
        dt_s    = p.dt_code converted to seconds
        R       = n_avail * (1 - exp(-Gamma dt_s)) / dt_s         // cm^-3 s^-1, cannot overshoot the floor
        rates.R[NUMU]    += R;  rates.Q[NUMU]    += R * p.f_numu   * m_mu
        rates.R[NUEBAR]  += R;  rates.Q[NUEBAR]  += R * p.f_nuebar * m_mu

  Units match `rates_accum` (R in cm⁻³ s⁻¹, Q in MeV cm⁻³ s⁻¹); the existing conversion to code units
  applies. The lab-frame transformation of the isotropic fluid-frame emission is done by `m1_source`.
- Diagnostic aux `m1_mu_decay` (the R above in code units) under `GRACE_M1_DIAGNOSTICS`.
- Tests (`test_m1_analytic_rates.cpp`, tag `[mudecay]`): cold dilute cell (T = 1, 10⁸ g cm⁻³, Y_μ = 0.02)
  gives R = n_μ/τ to 1e-12 and Q/R = 0.35 m_μ, 0.30 m_μ; degenerate cell (μ_e = 200 MeV) gives R = 0;
  Y_μ one floor-step above the minimum never overshoots; `[backreaction]`: one full collision + backreaction
  step on that cell gives ΔY_μ = −ΔN/D, ΔY_e = +ΔN/D, Δτ = −(emitted energy) to round-off.
- Production check: head-on M = 1 + 1, Y_μ ejecta bins → floor bin, Y_e bins +≈0.02, ν_μ/ν̄_e luminosity
  from the ejecta.

### Step B in code

- `fugacity_state`: add `eta_eq[NUMSPECIES]` = the raw matter values, stored before the clamp and the
  τ-suppression; `eta_nu` keeps its present meaning (suppressed/clamped, "neutrinos present").
- Row 1 (`compute_all_species_weakhub`, the Q = κ_a B loop): use `F.eta_eq[s]` for NUMU and NUMUBAR
  (compile switch `GRACE_M1_MUON_EMISSION_ETA_EQ`, default on after the HOT-TOV re-test); for NUE/NUEBAR keep
  `eta_nu` (the open discussion of docs/m1-neutrino-degeneracy-roles.md, its own switch, default off).
- Row 2 (`add_kirchhoff_absorption_opacity_from_QR`): pass η = 0 for all species (new bool argument);
  the analytic Q's already carry their blocking factors, and the processes are flavour-symmetric.
- Rows 3–5 unchanged (blocking factors, T_ν correction, β-eq closure keep the bounded `eta_nu`).
- Re-test on the 5-species HOT-TOV with eikonal τ (the discriminator of the two rejected clamp removals):
  κ_n(ν̄_μ) must stay at the table scale, β-eq failure counts must not rise, m1_muon_source must show
  capture-driven demuonization in hot cells with Y_μ above equilibrium.
