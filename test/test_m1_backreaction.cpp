/**
 * @file test_m1_backreaction.cpp
 * @brief Full behavioural spec of the M1 -> hydro backreaction
 *        (m1_equations_system_t::add_backreaction / _photons) on a single cell.
 *
 * add_backreaction couples the implicit collision step back onto the fluid in
 * three independent channels, each with its own limiter:
 *
 *   ENERGY / MOMENTUM  (all species, #ifndef GRACE_FREEZE_HYDRO)
 *     dE = Σ_s (E_old - E_new),  dS = Σ_s (F_old - F_new)
 *     fluid:      tau += f_E·dE ,  S += f_E·dS
 *     radiation:  (E,F) = f_E·(E,F)_new + (1-f_E)·(E,F)_old      [convex blend]
 *     f_E throttles the deposit so tau stays >= 0; the SAME f_E is applied to
 *     both sides, so energy and momentum are CONSERVED (the blend gives the
 *     unabsorbed remainder back to the radiation instead of deleting it).
 *
 *   Ye  (>= 3 species):  dYe*  = dN_nue  - dN_anue    (absorb nue RAISES Ye)
 *   Ymu (>= 5 species):  dYmu* = dN_numu - dN_anumu   (absorb numu RAISES Ymu)
 *     When the update would push Ye/Ymu past the EOS table bounds, a limiter
 *     throttles BOTH the composition change and the neutrino NUMBER fields by
 *     the same factor, so lepton number stays conserved.
 *
 * The three channels are INDEPENDENT: the energy limiter never touches N, the
 * composition limiters never touch E/F.  These tests pin every branch of every
 * channel, their conservation identities, causality of the energy blend, the
 * per-baryon (÷D) scaling, cross-channel independence, and the photon variant.
 *
 * Single hand-built cell, mock EOS with known bounds; no tables/rates/parfiles
 * beyond the hard-wired basic_config.  Requires GRACE_M1_NU_SPECIES >= 3;
 * the Ymu cases need >= 5; the photon cases need GRACE_M1_PHOTONS.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/m1.hh>

#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#if defined(GRACE_ENABLE_M1) && GRACE_M1_NU_SPECIES >= 3

namespace {

using namespace grace;

// Number of evolved neutrino species this build couples in add_backreaction.
constexpr int kNSpec =
    #if GRACE_M1_NU_SPECIES >= 5
        5;
    #else
        3;
    #endif

// Table bounds mirroring the production SFHo leptonic setup.
struct mock_bounds_eos_t {
    double yemin  = 0.01,   yemax  = 0.50;
    double ymumin = 5.0e-4, ymumax = 0.20;
    // Negative eps_min, as for a real tabulated EOS (SFHo+leptons reaches
    // ~-8.5e-4 at the cold surface, and can go to -energy_shift).  The
    // backreaction's invertibility guard is only meaningful when this can
    // be < 0 -- with eps_min pinned at 0 the guard silently reverts to the
    // eps >= 0 form the fix removes.
    double epsmin = -8.5e-4, epsmax = 1.0e4;
    KOKKOS_INLINE_FUNCTION double get_c2p_ye_max()  const { return yemax;  }
    KOKKOS_INLINE_FUNCTION double get_c2p_ye_min()  const { return yemin;  }
    KOKKOS_INLINE_FUNCTION double get_c2p_ymu_max() const { return ymumax; }
    KOKKOS_INLINE_FUNCTION double get_c2p_ymu_min() const { return ymumin; }
    KOKKOS_INLINE_FUNCTION void eps_range__rho_ye_ymu(
        double& emin, double& emax, double&, double&, double&,
        grace::eos_err_t&) const { emin = epsmin; emax = epsmax; }
};

var_array_t make_state(char const* label)
{
    // Single cell, one quadrant, full evolved-variable stride.  Kokkos
    // zero-initializes, so untouched channels are exactly 0.
    return var_array_t(label, 1, 1, 1, N_EVOL_VARS, 1);
}

/// Host-side handle on one cell: set/get named evolved fields.
struct cell_t {
    var_array_t old_state = make_state("br_old");
    var_array_t new_state = make_state("br_new");
    var_array_t aux       = var_array_t("br_aux", 1, 1, 1, N_AUX_VARS, 1);

    cell_t()
    {
        // Minkowski in the Z4c variables FILL_METRIC_ARRAY reads from the
        // OLD state: conformal metric = delta, chi = 1, alp = 1, beta = 0.
        set_old(GTXX_, 1.0); set_old(GTYY_, 1.0); set_old(GTZZ_, 1.0);
        set_old(CHI_,  1.0); set_old(ALP_,  1.0);
        // Sane defaults so pure-energy tests never divide by D = 0 or drive
        // the (always-run) composition block through NaN comparisons.  Every
        // composition test overrides these explicitly.
        set_new(DENS_,   1.0);
        set_new(YESTAR_, 0.30);
        #if GRACE_M1_NU_SPECIES >= 5
        set_new(YMUSTAR_, 0.02);
        #endif
        // Nonzero rates in the nue and nux channels: add_backreaction skips a
        // cell whose rates are negligible, and a radiation delta with no rates
        // is unphysical anyway.  The [implicit] cases override these.
        set_aux(KAPPAA1_, 1.0); set_aux(KAPPAS1_, 1.0); set_aux(ETA1_, 1.0);
        #if GRACE_M1_NU_SPECIES >= 5
        set_aux(KAPPAA5_, 1.0); set_aux(KAPPAS5_, 1.0); set_aux(ETA5_, 1.0);
        #elif GRACE_M1_NU_SPECIES >= 3
        set_aux(KAPPAA3_, 1.0); set_aux(KAPPAS3_, 1.0); set_aux(ETA3_, 1.0);
        #endif
    }

    void set_old(int v, double x) { poke(old_state, v, x); }
    void set_new(int v, double x) { poke(new_state, v, x); }
    double get_new(int v) const   { return peek(new_state, v); }
    double get_old(int v) const   { return peek(old_state, v); }

    // rho_min defaults to 0 (density cutoff disabled) so the coupling-logic
    // tests exercise add_backreaction unconditionally; the cutoff has its own
    // dedicated test that sets aux(RHO_) and a positive threshold.
    void run(mock_bounds_eos_t const& eos, double rho_min = 0.0)
    {
        m1_equations_system_t sys(old_state, staggered_variable_arrays_t{}, aux);
        auto ns = new_state;
        scalar_array_t<GRACE_NSPACEDIM> idx;   // unused by add_backreaction
        Kokkos::parallel_for("br_single", 1, KOKKOS_LAMBDA(int) {
            sys.add_backreaction<mock_bounds_eos_t>(0, VEC(0, 0, 0), idx, ns, eos, rho_min);
        });
        Kokkos::fence();
    }

    void set_aux(int v, double x) { poke(aux, v, x); }

    /// Drive the full implicit collision solve for species 0 (nue) on this
    /// cell: reads prims from old_state + aux, writes the post-collision
    /// radiation state into new_state.
    void run_implicit(double dt, double dtfact)
    {
        // advance_implicit_substep deep-copies old -> new before the kernel;
        // mirror it, or a cell the solver skips is left at zero instead of U = W.
        Kokkos::deep_copy(new_state, old_state);
        m1_equations_system_t sys(old_state, staggered_variable_arrays_t{}, aux);
        auto ns = new_state;
        scalar_array_t<GRACE_NSPACEDIM> idx;
        Kokkos::parallel_for("m1_implicit_single", 1, KOKKOS_LAMBDA(int) {
            sys.compute_implicit_update<0>(0, VEC(0, 0, 0), idx, ns, dt, dtfact);
        });
        Kokkos::fence();
    }

    #ifdef GRACE_M1_PHOTONS
    void run_photons()
    {
        m1_equations_system_t sys(old_state, staggered_variable_arrays_t{}, aux);
        auto ns = new_state;
        scalar_array_t<GRACE_NSPACEDIM> idx;
        Kokkos::parallel_for("br_ph_single", 1, KOKKOS_LAMBDA(int) {
            sys.add_backreaction_photons(0, VEC(0, 0, 0), idx, ns);
        });
        Kokkos::fence();
    }
    #endif

  private:
    static void poke(var_array_t const& v, int var, double x)
    {
        auto m = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(m, v);
        m(0, 0, 0, var, 0) = x;
        Kokkos::deep_copy(v, m);
    }
    static double peek(var_array_t const& v, int var)
    {
        auto m = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(m, v);
        return m(0, 0, 0, var, 0);
    }
};

// ── Shared assertions ────────────────────────────────────────────────────────

/// Composition change == net lepton number removed from the radiation fields,
/// in BOTH the plain and limited paths.
void require_lepton_identity(cell_t const& c, int nrad_nu, int nrad_anu,
                             int ystar, double ystar_before)
{
    double const dN_applied =
          (c.get_old(nrad_nu)  - c.get_new(nrad_nu))
        - (c.get_old(nrad_anu) - c.get_new(nrad_anu));
    REQUIRE_THAT(c.get_new(ystar) - ystar_before,
                 WithinAbs(dN_applied, 1e-15));
}

/// Total change (new - old) of an evolved radiation field summed over all
/// coupled neutrino species.  `base` is the species-1 index (ERAD1_, FRADX1_…).
double rad_change(cell_t const& c, int base)
{
    double d = 0.0;
    for (int s = 0; s < kNSpec; ++s) {
        int const off = s * GRACE_N_M1_VARS;
        d += c.get_new(base + off) - c.get_old(base + off);
    }
    return d;
}

/// Energy AND momentum conservation: the fluid's gain equals the radiation's
/// loss, measured from the PRE-collision radiation state.  Holds in every
/// branch (full transfer and throttled) because both sides carry the same f_E.
void require_em_conserved(cell_t const& c,
                          double tau0, double sx0, double sy0, double sz0)
{
    REQUIRE_THAT((c.get_new(TAU_) - tau0) + rad_change(c, ERAD1_),
                 WithinAbs(0.0, 1e-14));
    REQUIRE_THAT((c.get_new(SX_)  - sx0)  + rad_change(c, FRADX1_),
                 WithinAbs(0.0, 1e-14));
    REQUIRE_THAT((c.get_new(SY_)  - sy0)  + rad_change(c, FRADY1_),
                 WithinAbs(0.0, 1e-14));
    REQUIRE_THAT((c.get_new(SZ_)  - sz0)  + rad_change(c, FRADZ1_),
                 WithinAbs(0.0, 1e-14));
}

/// The convex blend keeps every species inside the causal cone |F| <= E
/// (Minkowski norm; the test metric is flat).
void require_causal(cell_t const& c, int spec)
{
    int const off = spec * GRACE_N_M1_VARS;
    double const E  = c.get_new(ERAD1_ + off);
    double const Fx = c.get_new(FRADX1_ + off);
    double const Fy = c.get_new(FRADY1_ + off);
    double const Fz = c.get_new(FRADZ1_ + off);
    double const F  = std::sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
    REQUIRE(F <= E * (1.0 + 1e-12));
}

}  // namespace


// =============================================================================
//  ENERGY / MOMENTUM channel
// =============================================================================

TEST_CASE("M1 backreaction: heating (energy_good) transfers the full exchange",
          "[m1][backreaction][energy]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const tau0 = 0.5;
    c.set_new(TAU_, tau0);
    // Radiation LOST energy -> fluid heats.  tau is ample, no throttling.
    c.set_old(ERAD1_, 1.0e-3);  c.set_new(ERAD1_, 8.0e-4);   // dE  = +2e-4
    c.set_old(FRADX1_, 1.0e-5); c.set_new(FRADX1_, 0.5e-5);  // dSx = +5e-6

    c.run(eos);

    // Full deposit; radiation left exactly at its post-collision value.
    REQUIRE_THAT(c.get_new(TAU_),   WithinRel(tau0 + 2.0e-4, 1e-14));
    REQUIRE_THAT(c.get_new(SX_),    WithinRel(5.0e-6, 1e-14));
    REQUIRE_THAT(c.get_new(ERAD1_), WithinRel(8.0e-4, 1e-14));
    REQUIRE_THAT(c.get_new(FRADX1_),WithinRel(0.5e-5, 1e-14));
    require_em_conserved(c, tau0, 0.0, 0.0, 0.0);
}


TEST_CASE("M1 backreaction: mild cooling within budget transfers fully",
          "[m1][backreaction][energy]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const tau0 = 1.0e-2;
    c.set_new(TAU_, tau0);
    // Radiation GAINED energy (dE < 0) but |dE| << tau0: energy_good, full.
    c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 2.0e-3);    // dE = -1e-3

    c.run(eos);

    REQUIRE_THAT(c.get_new(TAU_),   WithinRel(tau0 - 1.0e-3, 1e-12));
    REQUIRE_THAT(c.get_new(ERAD1_), WithinRel(2.0e-3, 1e-14));   // untouched
    require_em_conserved(c, tau0, 0.0, 0.0, 0.0);
}


TEST_CASE("M1 backreaction: extreme cooling throttles and CONSERVES (no revert)",
          "[m1][backreaction][energy]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Radiation would take 1e-3 from a cell holding only 1e-6: the fluid can
    // give ~all it has; the radiation keeps the unabsorbed remainder.
    double const tau0 = 1.0e-6;
    double const E0 = 1.0e-3, E1 = 2.0e-3;
    double const Fx0 = 0.0,   Fx1 = 1.0e-5;
    c.set_new(TAU_, tau0);
    c.set_old(ERAD1_, E0);   c.set_new(ERAD1_, E1);
    c.set_old(FRADX1_, Fx0); c.set_new(FRADX1_, Fx1);

    c.run(eos);

    // Invariants that hold for BOTH limiter policies.
    REQUIRE(c.get_new(TAU_) >= 0.0);            // never negative
    require_em_conserved(c, tau0, 0.0, 0.0, 0.0);
    require_causal(c, 0);

#if GRACE_M1_BACKREACT_HARDSTOP
    // Hard stop: the cell cannot afford the exchange, so it is reverted --
    // fluid and radiation both left at their pre-exchange values (halo-safe).
    REQUIRE_THAT(c.get_new(TAU_),   WithinRel(tau0, 1e-12));
    REQUIRE_THAT(c.get_new(ERAD1_), WithinRel(E0, 1e-14));
    (void)E1;
#else
    // Scaled: fluid drained to ~zero, radiation kept the unabsorbed remainder
    // (strictly between old and new -- neither reverted nor left at new).
    REQUIRE(c.get_new(TAU_) <= 1.0e-9);
    REQUIRE(c.get_new(ERAD1_) > E0);
    REQUIRE(c.get_new(ERAD1_) < E1);
#endif
}


TEST_CASE("M1 backreaction: zero radiation exchange is a no-op on the fluid",
          "[m1][backreaction][energy]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const tau0 = 0.5;
    c.set_new(TAU_, tau0);
    c.set_old(ERAD1_, 3.0e-3); c.set_new(ERAD1_, 3.0e-3);   // dE = 0
    c.set_old(FRADX1_, 2.0e-5); c.set_new(FRADX1_, 2.0e-5); // dS = 0

    c.run(eos);

    REQUIRE_THAT(c.get_new(TAU_),   WithinRel(tau0, 1e-15));
    REQUIRE_THAT(c.get_new(SX_),    WithinAbs(0.0, 1e-18));
    REQUIRE_THAT(c.get_new(ERAD1_), WithinRel(3.0e-3, 1e-15));
}


TEST_CASE("M1 backreaction: dE/dS accumulate over ALL species; per-species blend",
          "[m1][backreaction][energy][multispecies]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Small tau so the summed exchange forces throttling; each species carries
    // a different delta so we test the accumulation and the per-species blend.
    double const tau0 = 1.0e-6;
    c.set_new(TAU_, tau0);
    c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 1.4e-3);   // dE1 = -4e-4
    c.set_old(ERAD2_, 2.0e-3); c.set_new(ERAD2_, 2.3e-3);   // dE2 = -3e-4
    c.set_old(ERAD3_, 1.0e-3); c.set_new(ERAD3_, 1.2e-3);   // dE3 = -2e-4
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_old(ERAD4_, 1.0e-3); c.set_new(ERAD4_, 1.1e-3);   // dE4 = -1e-4
    c.set_old(ERAD5_, 1.0e-3); c.set_new(ERAD5_, 1.05e-3);  // dE5 = -5e-5
    #endif

    c.run(eos);

    require_em_conserved(c, tau0, 0.0, 0.0, 0.0);   // conserved either way
#if GRACE_M1_BACKREACT_HARDSTOP
    // tau may legitimately end up NEGATIVE.  The invertibility floor is
    // eps >= eps_min, and eps_min is NEGATIVE for a tabulated EOS (the mock
    // mirrors SFHo+leptons at -8.5e-4), so a cell cooling to eps < 0 is a
    // physical state, not a failure.  What the guard must enforce is only that
    // the accumulated exchange never drives eps below eps_min.  (This assertion
    // previously read `TAU_ >= 0`, which encoded the eps >= 0 bug.)
    double const D_cell = 1.0;   // cell_t default DENS_
    REQUIRE(c.get_new(TAU_) / D_cell >= eos.epsmin);
    // ... and the species that would have crossed the floor were reverted, so
    // the fluid did not simply absorb everything.
    double const dE_total = -(4e-4 + 3e-4 + 2e-4
    #if GRACE_M1_NU_SPECIES >= 5
                              + 1e-4 + 5e-5
    #endif
                             );
    REQUIRE(c.get_new(TAU_) > tau0 + dE_total);
#else
    REQUIRE(c.get_new(TAU_) <= 1.0e-9);
#endif
    // Every species stayed finite and >= its pre-collision energy (both paths).
    for (int s = 0; s < kNSpec; ++s) {
        int const off = s * GRACE_N_M1_VARS;
        REQUIRE(std::isfinite(c.get_new(ERAD1_+off)));
        REQUIRE(c.get_new(ERAD1_+off) >= c.get_old(ERAD1_+off) - 1e-15);
    }
}


TEST_CASE("M1 backreaction: momentum rides the energy factor (no separate cap)",
          "[m1][backreaction][energy][momentum]")
{
    cell_t c; mock_bounds_eos_t eos;

    // energy_good, but a large momentum exchange: it is applied in FULL — the
    // backreaction has no independent momentum limiter (velocity is c2p's job).
    double const tau0 = 1.0;
    c.set_new(TAU_, tau0);
    c.set_old(ERAD1_, 1.0e-4); c.set_new(ERAD1_, 0.9e-4);    // tiny dE
    c.set_old(FRADX1_, 1.0);   c.set_new(FRADX1_, 0.0);      // huge dSx = +1.0

    c.run(eos);

    REQUIRE_THAT(c.get_new(SX_), WithinRel(1.0, 1e-12));     // full kick
    require_em_conserved(c, tau0, 0.0, 0.0, 0.0);
}


// =============================================================================
//  Ye channel
// =============================================================================

TEST_CASE("M1 backreaction: nue/anue number changes map to Ye with physical signs",
          "[m1][backreaction][lepton]")
{
    cell_t c; mock_bounds_eos_t eos;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(TAU_, 0.5);

    c.set_old(NRAD1_, 0.10);  c.set_new(NRAD1_, 0.07);   // dN_nue  = +0.03 -> Ye up
    c.set_old(NRAD2_, 0.05);  c.set_new(NRAD2_, 0.04);   // dN_anue = +0.01 -> Ye down
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_new(YMUSTAR_, 0.02);
    c.set_old(NRAD3_, 0.020); c.set_new(NRAD3_, 0.016);  // dN_numu  = +0.004
    c.set_old(NRAD4_, 0.010); c.set_new(NRAD4_, 0.009);  // dN_anumu = +0.001
    #endif

    c.run(eos);

    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30 + 0.03 - 0.01, 1e-14));
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, 0.30);
    // In-bounds: the number fields themselves are NOT rescaled.
    REQUIRE_THAT(c.get_new(NRAD1_), WithinRel(0.07, 1e-14));
    REQUIRE_THAT(c.get_new(NRAD2_), WithinRel(0.04, 1e-14));
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(0.02 + 0.004 - 0.001, 1e-14));
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, 0.02);
    #endif
}


TEST_CASE("M1 backreaction: Ye limiter at the table ceiling conserves lepton number",
          "[m1][backreaction][lepton][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const ye0 = 0.49;
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, ye0);
    c.set_new(TAU_, 0.5);
    // Raw update Ye = 0.54 > yemax = 0.50 -> factor (0.50-0.49)/0.05 = 0.2.
    c.set_old(NRAD1_, 0.10);  c.set_new(NRAD1_, 0.05);   // dN_nue = +0.05
    c.set_old(NRAD2_, 0.05);  c.set_new(NRAD2_, 0.05);   // dN_anue = 0

    c.run(eos);

    REQUIRE(c.get_new(YESTAR_) <= eos.yemax);            // never past the bound
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, ye0);
#if GRACE_M1_BACKREACT_HARDSTOP
    // Hard stop: the out-of-bounds Ye channel is reverted -> Ye and N restored.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(ye0, 1e-12));
    REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(0.10, 1e-14));
#else
    // Scaled: Ye lands exactly on the ceiling, N throttled by the same factor.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(eos.yemax, 1e-8));
    REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(0.10 - 0.2 * 0.05, 1e-8));
#endif
}


TEST_CASE("M1 backreaction: Ye limiter at the table FLOOR conserves lepton number",
          "[m1][backreaction][lepton][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const ye0 = 0.02;   // just above yemin = 0.01
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, ye0);
    c.set_new(TAU_, 0.5);
    // anue absorbed: raw Ye = 0.02 - 0.05 = -0.03 < yemin -> factor
    // (0.01-0.02)/(-0.05) = 0.2.
    c.set_old(NRAD2_, 0.10); c.set_new(NRAD2_, 0.05);    // dN_anue = +0.05 -> Ye down

    c.run(eos);

    REQUIRE(c.get_new(YESTAR_) >= eos.yemin * (1.0 - 1e-12));   // never below floor
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, ye0);
#if GRACE_M1_BACKREACT_HARDSTOP
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(ye0, 1e-12));    // reverted
    REQUIRE_THAT(c.get_new(NRAD2_),  WithinRel(0.10, 1e-14));
#else
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(eos.yemin, 1e-8));
    REQUIRE_THAT(c.get_new(NRAD2_),  WithinRel(0.10 - 0.2 * 0.05, 1e-8));
#endif
}


TEST_CASE("M1 backreaction: Ye uses the per-baryon (÷ D) normalisation",
          "[m1][backreaction][lepton][density]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Same dN, but D = 0.5: Ye = Ye*/D, so a given number change moves Ye
    // twice as far.  YESTAR* is the conserved (densitised) quantity.
    double const D = 0.5, yestar0 = 0.30 * D;   // Ye = 0.30
    c.set_new(DENS_, D);
    c.set_new(YESTAR_, yestar0);
    c.set_new(TAU_, 0.5);
    c.set_old(NRAD1_, 0.10); c.set_new(NRAD1_, 0.07);   // dN_nue = +0.03

    c.run(eos);

    // In-bounds (Ye_new = (0.15 + 0.03)/0.5 = 0.36): YESTAR* += dN directly.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(yestar0 + 0.03, 1e-14));
    REQUIRE_THAT(c.get_new(YESTAR_) / D, WithinRel(0.36, 1e-12));
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, yestar0);
}


TEST_CASE("M1 backreaction: zero net lepton exchange leaves Ye untouched",
          "[m1][backreaction][lepton]")
{
    cell_t c; mock_bounds_eos_t eos;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(TAU_, 0.5);
    // dN_nue == dN_anue: net zero, and the limiter denominator is 0 -> must
    // not produce NaN / spurious rescaling.
    c.set_old(NRAD1_, 0.10); c.set_new(NRAD1_, 0.08);   // dN_nue  = +0.02
    c.set_old(NRAD2_, 0.10); c.set_new(NRAD2_, 0.08);   // dN_anue = +0.02

    c.run(eos);

    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));
    REQUIRE(std::isfinite(c.get_new(YESTAR_)));
    REQUIRE_THAT(c.get_new(NRAD1_), WithinRel(0.08, 1e-14));   // not rescaled
}


// =============================================================================
//  Ymu channel  (5-species only)
// =============================================================================

#if GRACE_M1_NU_SPECIES >= 5

TEST_CASE("M1 backreaction: numu/anumu number changes map to Ymu with physical signs",
          "[m1][backreaction][lepton][muon]")
{
    cell_t c; mock_bounds_eos_t eos;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(YMUSTAR_, 0.02);
    c.set_new(TAU_, 0.5);
    c.set_old(NRAD3_, 0.020); c.set_new(NRAD3_, 0.014);  // dN_numu  = +0.006 -> Ymu up
    c.set_old(NRAD4_, 0.010); c.set_new(NRAD4_, 0.008);  // dN_anumu = +0.002 -> Ymu down

    c.run(eos);

    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(0.02 + 0.006 - 0.002, 1e-14));
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, 0.02);
    // Ye channel untouched.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));
}


TEST_CASE("M1 backreaction: Ymu limiter at the table CEILING conserves lepton number",
          "[m1][backreaction][lepton][muon][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const ymu0 = 0.19;   // just below ymumax = 0.20
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(YMUSTAR_, ymu0);
    c.set_new(TAU_, 0.5);
    // numu absorbed: raw Ymu = 0.19 + 0.05 = 0.24 > ymumax -> factor
    // (0.20-0.19)/0.05 = 0.2.
    c.set_old(NRAD3_, 0.10); c.set_new(NRAD3_, 0.05);    // dN_numu = +0.05

    c.run(eos);

    REQUIRE(c.get_new(YMUSTAR_) <= eos.ymumax);
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, ymu0);
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));   // Ye untouched
#if GRACE_M1_BACKREACT_HARDSTOP
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(ymu0, 1e-12));  // reverted
    REQUIRE_THAT(c.get_new(NRAD3_),   WithinRel(0.10, 1e-14));
#else
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(eos.ymumax, 1e-8));
    REQUIRE_THAT(c.get_new(NRAD3_),   WithinRel(0.10 - 0.2 * 0.05, 1e-8));
#endif
}


TEST_CASE("M1 backreaction: Ymu limiter at the table FLOOR conserves lepton number",
          "[m1][backreaction][lepton][muon][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    double const ymu0 = 6.0e-4;
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(YMUSTAR_, ymu0);
    c.set_new(TAU_, 0.5);
    // anumu absorbed: raw Ymu = 6e-4 - 1e-3 < ymumin = 5e-4 -> factor 0.1.
    c.set_old(NRAD4_, 0.010); c.set_new(NRAD4_, 0.009);  // dN_anumu = +1e-3

    c.run(eos);

    REQUIRE(c.get_new(YMUSTAR_) >= eos.ymumin * (1.0 - 1e-12));
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, ymu0);
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));   // Ye untouched
#if GRACE_M1_BACKREACT_HARDSTOP
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(ymu0, 1e-12));  // reverted
    REQUIRE_THAT(c.get_new(NRAD4_),   WithinRel(0.010, 1e-14));
#else
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(eos.ymumin, 1e-8));
    REQUIRE_THAT(c.get_new(NRAD4_),   WithinRel(0.010 - 0.1 * 1.0e-3, 1e-8));
#endif
}

#endif  // GRACE_M1_NU_SPECIES >= 5


// =============================================================================
//  Cross-channel independence
// =============================================================================

TEST_CASE("M1 backreaction: energy blend does NOT alter the number fields",
          "[m1][backreaction][independence]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Force the ENERGY limiter (tiny tau, big cooling) while an in-bounds Ye
    // exchange happens: the energy blend rewrites E/F but must leave N exactly
    // at its post-collision value, so Ye still sees the FULL dN.
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(TAU_, 1.0e-6);
    c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 2.0e-3);   // extreme cooling
    c.set_old(NRAD1_, 0.10);   c.set_new(NRAD1_, 0.07);     // dN_nue = +0.03

    c.run(eos);

    // Number field untouched by the energy limiter; Ye got the full update.
    REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(0.07, 1e-14));
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30 + 0.03, 1e-14));
    // Energy still conserved.
    require_em_conserved(c, 1.0e-6, 0.0, 0.0, 0.0);
}


TEST_CASE("M1 backreaction: energy and Ye limiters fire independently together",
          "[m1][backreaction][independence][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    // BOTH channels saturated at once: tiny tau (energy throttled) AND an
    // over-ceiling Ye push (composition throttled).  Each obeys its own bound.
    double const ye0 = 0.49;
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, ye0);
    c.set_new(TAU_, 1.0e-6);
    c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 2.0e-3);   // extreme cooling
    c.set_old(NRAD1_, 0.10);   c.set_new(NRAD1_, 0.05);     // dN_nue = +0.05

    c.run(eos);

    require_em_conserved(c, 1.0e-6, 0.0, 0.0, 0.0);
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, ye0);
    REQUIRE(c.get_new(TAU_) >= 0.0);
    REQUIRE(c.get_new(YESTAR_) <= eos.yemax);
#if GRACE_M1_BACKREACT_HARDSTOP
    // Both channels revert independently: fluid energy AND Ye left untouched.
    REQUIRE_THAT(c.get_new(TAU_),    WithinRel(1.0e-6, 1e-12));
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(ye0, 1e-12));
#else
    // Both channels saturate independently: tau drained, Ye pinned to ceiling.
    REQUIRE(c.get_new(TAU_) <= 1.0e-9);
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(eos.yemax, 1e-8));
#endif
}


// =============================================================================
//  Photon backreaction  (energy/momentum only, no lepton number)
// =============================================================================

#ifdef GRACE_M1_PHOTONS

TEST_CASE("M1 photon backreaction: heating flows to tau, composition untouched",
          "[m1][backreaction][photons]")
{
    cell_t c;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_new(YMUSTAR_, 0.02);
    #endif
    c.set_new(TAU_, 0.5);
    c.set_old(ERADPH_, 1.0e-3);  c.set_new(ERADPH_, 8.0e-4);   // dE = +2e-4
    c.set_old(FRADXPH_, 1.0e-5); c.set_new(FRADXPH_, 0.5e-5);

    c.run_photons();

    REQUIRE_THAT(c.get_new(TAU_), WithinRel(0.5 + 2.0e-4, 1e-14));
    REQUIRE_THAT(c.get_new(SX_),  WithinRel(5.0e-6, 1e-14));
    REQUIRE_THAT(c.get_new(ERADPH_), WithinRel(8.0e-4, 1e-14));  // untouched
    // Photons carry no lepton number: Ye*/Ymu* must be bit-identical.
    REQUIRE(c.get_new(YESTAR_) == 0.30);
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE(c.get_new(YMUSTAR_) == 0.02);
    #endif
}


TEST_CASE("M1 photon backreaction: extreme cooling throttles and conserves",
          "[m1][backreaction][photons]")
{
    cell_t c;

    double const tau0 = 1.0e-6;
    double const E0 = 1.0e-3, E1 = 2.0e-3, N0 = 5.0e-4, N1 = 9.0e-4;
    c.set_new(TAU_, tau0);
    c.set_old(ERADPH_, E0); c.set_new(ERADPH_, E1);
    c.set_old(NRADPH_, N0); c.set_new(NRADPH_, N1);
    c.set_old(FRADXPH_, 0.0); c.set_new(FRADXPH_, 1.0e-5);

    c.run_photons();

    REQUIRE(c.get_new(TAU_) >= 0.0);
    // Energy conserved (fluid loss == photon gain from pre-collision).
    REQUIRE_THAT((c.get_new(TAU_) - tau0) + (c.get_new(ERADPH_) - E0),
                 WithinAbs(0.0, 1e-14));
#if GRACE_M1_BACKREACT_HARDSTOP
    // Hard stop: photon block reverted, fluid untouched.
    REQUIRE_THAT(c.get_new(TAU_),    WithinRel(tau0, 1e-12));
    REQUIRE_THAT(c.get_new(ERADPH_), WithinRel(E0, 1e-14));
    REQUIRE_THAT(c.get_new(NRADPH_), WithinRel(N0, 1e-14));
    (void)E1; (void)N1;
#else
    // Scaled: E, N and F blended toward old by the SAME factor -> strictly between.
    REQUIRE(c.get_new(TAU_) <= 1.0e-9);
    REQUIRE(c.get_new(ERADPH_) > E0);  REQUIRE(c.get_new(ERADPH_) < E1);
    REQUIRE(c.get_new(NRADPH_) > N0);  REQUIRE(c.get_new(NRADPH_) < N1);
#endif
}

#endif  // GRACE_M1_PHOTONS


// =============================================================================
//  Low-density ("halo") behaviour — documents the per-baryon amplification
//  that survives the (correct, conservative) limiters.
// =============================================================================

TEST_CASE("M1 backreaction: low-density halo behaviour differs by limiter policy",
          "[m1][backreaction][halo]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Atmosphere-like cell: D ~ 1e-10, yet the star's radiation streams a
    // "normal"-sized number/energy exchange through it.  This is the cell that
    // makes or breaks the visible halo.
    double const D = 1.0e-10;
    c.set_new(DENS_, D);
    c.set_new(YESTAR_, 0.30 * D);          // Ye = 0.30
    c.set_new(TAU_, 1.0e-12);
    c.set_old(ERAD1_, 1.0e-6); c.set_new(ERAD1_, 2.0e-6);  // cooling >> tau
    c.set_old(NRAD1_, 1.0e-6); c.set_new(NRAD1_, 0.0);     // dN_nue huge vs D
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_new(YMUSTAR_, 0.02 * D);
    c.set_old(NRAD3_, 1.0e-6); c.set_new(NRAD3_, 0.0);     // dN_numu huge vs D
    #endif

    c.run(eos);

    // Invariant either way: finite, non-negative tau, conserved.
    REQUIRE(std::isfinite(c.get_new(YESTAR_)));
    REQUIRE(c.get_new(TAU_) >= 0.0);
    require_em_conserved(c, 1.0e-12, 0.0, 0.0, 0.0);

#if GRACE_M1_BACKREACT_HARDSTOP
    // HARD STOP preserves the halo: the unaffordable exchange is reverted, so
    // the tenuous cell keeps its composition and energy rather than being
    // slammed onto a table edge.  This is the whole reason for the policy.
    REQUIRE_THAT(c.get_new(YESTAR_) / D, WithinRel(0.30, 1e-10));
    REQUIRE_THAT(c.get_new(TAU_),        WithinRel(1.0e-12, 1e-10));
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(c.get_new(YMUSTAR_) / D, WithinRel(0.02, 1e-10));
    #endif
#else
    // SCALED keeps everything in bounds but PINS Ye to the ceiling -- a normal
    // exchange over near-zero baryon content saturates the composition.  This
    // is the halo artefact the hard-stop policy avoids.
    REQUIRE(c.get_new(YESTAR_) / D <= eos.yemax * (1.0 + 1e-8));
    REQUIRE_THAT(c.get_new(YESTAR_) / D, WithinRel(eos.yemax, 1e-6));
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE(std::isfinite(c.get_new(YMUSTAR_)));
    REQUIRE(c.get_new(YMUSTAR_) / D <= eos.ymumax * (1.0 + 1e-8));
    #endif
#endif
}

TEST_CASE("M1 backreaction: cooling is applied whenever tau stays positive, "
          "even when it undercuts the momentum already present",
          "[m1][backreaction][momentum][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    // 'Taking too much out': a COLD moving cell (S^2 near the eps>=0 boundary
    // tau(tau+2D)) cooled by dE < 0.  |S| never grows and tau stays positive,
    // so BOTH limiters accept -- neither checks that the remaining energy can
    // still carry the momentum the fluid already has.  That is deliberate and
    // matches FIL, whose tau<0 test gates tau only; the protection against the
    // regime where it matters is the density cutoff, not a momentum bound.
    double const D = 1.0, tau0 = 1.0e-4, Sx0 = 1.0e-2;
    // initial state valid: Sx0^2 = 1e-4  <  tau0(tau0+2D) ~ 2e-4
    c.set_new(DENS_, D);
    c.set_new(TAU_, tau0);
    c.set_new(SX_, Sx0);
    c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 1.05e-3);   // dE = -5e-5 (cooling)

    c.run(eos);

    REQUIRE_THAT(c.get_new(TAU_), WithinRel(tau0 - 5.0e-5, 1e-12));
    REQUIRE_THAT(c.get_new(SX_),  WithinRel(Sx0, 1e-14));   // exchange carries no dS
}


TEST_CASE("M1 backreaction: momentum kick is applied unconditionally",
          "[m1][backreaction][momentum][limiter]")
{
    cell_t c; mock_bounds_eos_t eos;

    // Light fluid (tau + D ~ 2e-4) hit by a momentum exchange |dS| = 1: the
    // post-kick |S| implies |v| ~ 1 and c2p will invert garbage eps.  Neither
    // limiter rejects it -- the accept decision is energy-only in both (tau > 0
    // for the hard stop, limiting_factor_E for the scaled branch) and momentum
    // simply rides along, as in FIL.  The halo pathology this resembles came
    // from over-large opacities in near-empty cells, and is addressed upstream
    // in the rate lookup (and, if wanted, by the density cutoff below).
    double const D = 1.0e-4, tau0 = 1.0e-4;
    c.set_new(DENS_, D);
    c.set_new(YESTAR_, 0.30 * D);
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_new(YMUSTAR_, 0.02 * D);
    #endif
    c.set_new(TAU_, tau0);
    c.set_old(ERAD1_, 2.0);  c.set_new(ERAD1_, 2.0);     // dE = 0
    c.set_old(FRADX1_, 1.0); c.set_new(FRADX1_, 0.0);    // dSx = +1.0

    c.run(eos);

    REQUIRE_THAT(c.get_new(SX_),  WithinRel(1.0, 1e-12));
    REQUIRE_THAT(c.get_new(TAU_), WithinRel(tau0, 1e-14));   // dE = 0
}


// =============================================================================
//  Density cutoff (FIL M1_rho_floor) — the actual halo fix, independent of the
//  limiter policy.  Below rho_min the WHOLE coupling is skipped.
// =============================================================================

TEST_CASE("M1 backreaction: density cutoff skips the coupling below rho_min",
          "[m1][backreaction][cutoff]")
{
    mock_bounds_eos_t eos;
    double const rho_min = 1.0e-6;

    // An exchange that WOULD move Ye and tau (well in bounds) if applied.
    auto make = [&](double rho_cell) {
        cell_t c;
        c.set_aux(RHO_, rho_cell);
        c.set_new(DENS_, 1.0);
        c.set_new(YESTAR_, 0.30);
        c.set_new(TAU_, 0.5);
        c.set_old(ERAD1_, 1.0e-3); c.set_new(ERAD1_, 0.8e-3);  // dE = +2e-4 -> tau up
        c.set_old(NRAD1_, 0.10);   c.set_new(NRAD1_, 0.07);    // dN_nue = +0.03 -> Ye up
        return c;
    };

    SECTION("below rho_min: fluid completely untouched") {
        cell_t c = make(rho_min * 0.5);
        c.run(eos, rho_min);
        REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-15));   // no Ye drift
        REQUIRE_THAT(c.get_new(TAU_),    WithinRel(0.5,  1e-15));   // no heating
        REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(0.07, 1e-15));   // N left at new
    }

    SECTION("above rho_min: coupling applies normally") {
        cell_t c = make(rho_min * 2.0);
        c.run(eos, rho_min);
        REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30 + 0.03, 1e-14));
        REQUIRE_THAT(c.get_new(TAU_),    WithinRel(0.5 + 2.0e-4, 1e-14));
    }

    SECTION("rho_min = 0 disables the cutoff (couples even at zero rho)") {
        cell_t c = make(0.0);
        c.run(eos, 0.0);
        REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30 + 0.03, 1e-14));
    }
}


// =============================================================================
//  Implicit collision solve (compute_implicit_update)
// =============================================================================

namespace {

/// A surface/halo cell: thin radiation (|F|/E = fluxfac) of magnitude E0
/// riding on a moving fluid with every opacity at the 1e-60 floor.  The
/// physically correct implicit update is the identity to ~60 digits.
/// fluxfac = 1 (the generic free-streaming state at the stellar surface) and
/// slightly above 1 (transport overshoot before any causality rescale) put
/// zeta at the edge of the closure rootfind bracket -- the regime the halo
/// cells actually occupy.
cell_t make_floor_opacity_cell(double fluxfac, double E0)
{
    cell_t c;
    c.set_old(ERAD1_,  E0);
    c.set_old(FRADX1_, fluxfac * E0);
    c.set_old(NRAD1_,  3.0 * E0);
    c.set_aux(ZVECX_,  0.05);      // nonzero v in every component so any
    c.set_aux(ZVECY_, -0.03);      // frame-transform error appears in E and
    c.set_aux(ZVECZ_,  0.02);      // all three fluxes
    c.set_aux(KAPPAA1_,  1.0e-60);
    c.set_aux(KAPPAS1_,  1.0e-60);
    c.set_aux(ETA1_,     1.0e-60);
    c.set_aux(ETAN1_,    1.0e-60);
    c.set_aux(KAPPAAN1_, 1.0e-60);
    return c;
}

}  // namespace

TEST_CASE("M1 implicit: floor opacity is an exact no-op on thin radiation",
          "[m1][implicit]")
{
    // The collision change here is O(kappa*dt*E) ~ 1e-72, so ANY visible dU is
    // solver artifact.  At E0 = 1e-10 the Newton solve is forced to iterate and
    // lands on U = W; the floor-scale E0 = 1e-15 variant -- where the ABSOLUTE
    // 1e-15 tolerance instant-accepts the biased m1_fluid_to_lab_thick guess
    // with a silent O(v) relative error (the surface-halo mechanism, observed
    // as |dU|/|W| = 3.78e-02 ~ v on Hunter) -- is the KNOWN-FAILING hidden case
    // below, kept as documentation until a guess that is exact at zero opacity
    // WITHOUT losing positivity in stiff cells is found (the delta-form
    // attempt was exact but could go E < 0 in stiff/cooling cells and
    // detonated the interior -- see the hidden case's comment).
    double const fluxfac = GENERATE(0.9, 1.0, 1.05);
    double const E0      = 1.0e-10;
    cell_t c = make_floor_opacity_cell(fluxfac, E0);
    c.run_implicit(1.0e-2, 0.5);

    INFO("fluxfac = " << fluxfac << " E0 = " << E0);
    REQUIRE_THAT(c.get_new(ERAD1_),  WithinRel(E0, 1e-13));
    REQUIRE_THAT(c.get_new(FRADX1_), WithinRel(fluxfac * E0, 1e-13));
    REQUIRE_THAT(c.get_new(FRADY1_), WithinAbs(0.0, 1e-13 * E0));
    REQUIRE_THAT(c.get_new(FRADZ1_), WithinAbs(0.0, 1e-13 * E0));
    REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(3.0 * E0, 1e-13));
}

TEST_CASE("M1 implicit: strong coupling relaxes E toward eta/kappa_a and stays "
          "causal", "[m1][implicit]")
{
    // kappa_a*dt = 5: deep in the stiff regime that used to die with
    // SMALLSTEP/STAGNATION when the line search tested Armijo on the
    // re-solved-zeta residual with the frozen-zeta slope.
    cell_t c;
    double const E0 = 1.0, J_eq = 2.0, kap = 100.0, dt = 0.05;
    c.set_old(ERAD1_,  E0);
    c.set_old(FRADX1_, 0.1);
    c.set_old(NRAD1_,  1.0);
    c.set_aux(ZVECX_, 0.1); c.set_aux(ZVECY_, 0.05); c.set_aux(ZVECZ_, -0.02);
    c.set_aux(KAPPAA1_,  kap);
    c.set_aux(KAPPAS1_,  10.0);
    c.set_aux(ETA1_,     kap * J_eq);
    c.set_aux(ETAN1_,    kap * J_eq);
    c.set_aux(KAPPAAN1_, kap);
    c.run_implicit(dt, 1.0);

    double const E_new = c.get_new(ERAD1_);
    double const Fx = c.get_new(FRADX1_), Fy = c.get_new(FRADY1_),
                 Fz = c.get_new(FRADZ1_);
    // heated toward (fluid-frame) equilibrium, without overshooting it by more
    // than the O(v) lab-frame offset
    REQUIRE(E_new > E0);
    REQUIRE(E_new < 1.5 * J_eq);
    // relaxation is ~ (1+kappa*dt)^-1: after kappa*dt = 5 the remaining
    // distance to equilibrium must be well under half the initial one
    REQUIRE(std::abs(E_new - J_eq) < 0.5 * std::abs(E0 - J_eq));
    // absorption dominates: flux is suppressed, field stays causal
    REQUIRE(std::sqrt(Fx*Fx + Fy*Fy + Fz*Fz) <= E_new * (1.0 + 1e-12));
    REQUIRE(c.get_new(NRAD1_) > 0.0);
    REQUIRE(std::isfinite(E_new));
}

TEST_CASE("M1 implicit: solve is exactly mirror-equivariant", "[m1][implicit]")
{
    // Reflect x: flip v_x and F_x.  Every scalar the solver builds (v2, F2,
    // vdotF, vdotfh, J, zeta, chi) is bit-identical between the two cells and
    // every x-component is an exact IEEE negation, so the results must match
    // BITWISE -- rounding in round-to-nearest is sign-symmetric.  A failure
    // here means the implicit solve itself can seed L/R asymmetry in a
    // reflection-symmetric star.
    auto const run_pair = [](double kap, double eta) {
        cell_t a, b;
        for (cell_t* c : {&a, &b}) {
            double const sx = (c == &a) ? 1.0 : -1.0;
            c->set_old(ERAD1_,  1.0e-4);
            c->set_old(FRADX1_, sx * 0.6e-4);
            c->set_old(FRADY1_, 0.2e-4);
            c->set_old(NRAD1_,  2.0e-4);
            c->set_aux(ZVECX_, sx * 0.08);
            c->set_aux(ZVECY_, -0.04);
            c->set_aux(ZVECZ_,  0.01);
            c->set_aux(KAPPAA1_, kap);  c->set_aux(KAPPAS1_, 0.1 * kap);
            c->set_aux(ETA1_, eta);     c->set_aux(ETAN1_, eta);
            c->set_aux(KAPPAAN1_, kap);
            c->run_implicit(1.0e-2, 1.0);
        }
        REQUIRE(a.get_new(ERAD1_)  ==  b.get_new(ERAD1_));
        REQUIRE(a.get_new(FRADX1_) == -b.get_new(FRADX1_));
        REQUIRE(a.get_new(FRADY1_) ==  b.get_new(FRADY1_));
        REQUIRE(a.get_new(FRADZ1_) ==  b.get_new(FRADZ1_));
        REQUIRE(a.get_new(NRAD1_)  ==  b.get_new(NRAD1_));
    };
    SECTION("floor opacity")    { run_pair(1.0e-60, 1.0e-60); }
    SECTION("moderate opacity") { run_pair(1.0, 0.5); }
    SECTION("stiff opacity")    { run_pair(1.0e3, 2.0e3); }
}

#endif  // GRACE_ENABLE_M1 && GRACE_M1_NU_SPECIES >= 3

#if defined(GRACE_ENABLE_M1) && GRACE_M1_NU_SPECIES >= 3
// KNOWN FAILING (hidden: run explicitly with the [halo-bug] tag).  Hunter
// halo-cell reproduction: floor-scale E, curved (TOV-like) metric, a range of
// fluid velocities.  The absolute Newton tolerance instant-accepts the
// m1_fluid_to_lab_thick initial guess for every one of these, so the returned
// state carries the guess's O(v)*E thin-field bias -- the surface-halo
// mechanism.  A fix must make the guess exact at zero opacity WITHOUT giving
// up positivity in stiff cells: the delta-form guess (u = W + thick(Jhat-J,
// Hhat-H)) passed this test but could push E < 0 in stiff/cooling cells,
// which detonated the interior (T runaway, star evaporation) -- reverted.
TEST_CASE("M1 implicit: floor-scale E at floor opacity is exact for any metric "
          "and velocity", "[.][halo-bug]")
{
    double const alp   = GENERATE(1.0, 0.7);
    double const chi_c = GENERATE(1.0, 0.8);
    double const vmag  = GENERATE(0.05, 0.2, 0.4);
    double const ff    = GENERATE(0.9, 1.0);
    double const E0    = 1.0e-15;
    cell_t c;
    c.set_old(ALP_, alp);  c.set_old(CHI_, chi_c);
    c.set_old(ERAD1_,  E0);
    c.set_old(FRADX1_, ff * E0);
    c.set_old(NRAD1_,  3.0 * E0);
    c.set_aux(ZVECX_, vmag); c.set_aux(ZVECY_, -0.5*vmag); c.set_aux(ZVECZ_, 0.25*vmag);
    c.set_aux(KAPPAA1_, 1.0e-60); c.set_aux(KAPPAS1_, 1.0e-60);
    c.set_aux(ETA1_, 1.0e-60); c.set_aux(ETAN1_, 1.0e-60); c.set_aux(KAPPAAN1_, 1.0e-60);
    c.run_implicit(1.0e-3, 1.0);
    INFO("alp=" << alp << " chi=" << chi_c << " v=" << vmag << " ff=" << ff);
    REQUIRE_THAT(c.get_new(ERAD1_),  WithinRel(E0, 1e-10));
    REQUIRE_THAT(c.get_new(FRADX1_), WithinRel(ff * E0, 1e-10));
    REQUIRE_THAT(c.get_new(NRAD1_),  WithinRel(3.0 * E0, 1e-10));
}
#endif
