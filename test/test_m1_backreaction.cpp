/**
 * @file test_m1_backreaction.cpp
 * @brief Lepton-number bookkeeping of the M1 -> hydro backreaction
 *        (m1_equations_system_t::add_backreaction) on a single cell.
 *
 * The backreaction converts the implicit collision step's change in the
 * neutrino NUMBER fields into changes of the advected composition:
 *
 *     dYe*  = (N_nue_old  - N_nue_new)  - (N_anue_old  - N_anue_new)
 *     dYmu* = (N_numu_old - N_numu_new) - (N_anumu_old - N_anumu_new)
 *
 * i.e. absorbing nue RAISES Ye, absorbing anue LOWERS it (same pattern for
 * the muon pair into Ymu), and the energy/momentum deltas of ALL species go
 * into (tau, S_i).  When the update would push Ye/Ymu outside the EOS table
 * bounds, a limiter throttles BOTH the composition change and the neutrino
 * number fields by the same factor, so lepton number stays conserved --
 * that identity is asserted explicitly here.
 *
 * This is a wiring/sign/limiter test on hand-built single-cell state views
 * with a mock EOS carrying known bounds; no tables, parfiles beyond the
 * hard-wired basic_config, or rate providers are involved.  It exists
 * because the sign conventions here are invisible in production until Ymu
 * drifts the wrong way over milliseconds -- and because the limiter bounds
 * historically came from a default-constructed (uninitialized) EOS, a bug
 * fixed by making add_backreaction take the EOS as a parameter, which is
 * also what makes this test possible.
 *
 * Requires GRACE_M1_NU_SPECIES >= 3 (Ye coupling); the Ymu cases need >= 5.
 */
#include <catch2/catch_test_macros.hpp>
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

// Table bounds mirroring the production SFHo leptonic setup.
struct mock_bounds_eos_t {
    double yemin  = 0.01,   yemax  = 0.50;
    double ymumin = 5.0e-4, ymumax = 0.20;
    KOKKOS_INLINE_FUNCTION double get_c2p_ye_max()  const { return yemax;  }
    KOKKOS_INLINE_FUNCTION double get_c2p_ye_min()  const { return yemin;  }
    KOKKOS_INLINE_FUNCTION double get_c2p_ymu_max() const { return ymumax; }
    KOKKOS_INLINE_FUNCTION double get_c2p_ymu_min() const { return ymumin; }
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
    }

    void set_old(int v, double x) { poke(old_state, v, x); }
    void set_new(int v, double x) { poke(new_state, v, x); }
    double get_new(int v) const   { return peek(new_state, v); }
    double get_old(int v) const   { return peek(old_state, v); }

    void run(mock_bounds_eos_t const& eos)
    {
        m1_equations_system_t sys(old_state, staggered_variable_arrays_t{}, aux);
        auto ns = new_state;
        scalar_array_t<GRACE_NSPACEDIM> idx;   // unused by add_backreaction
        Kokkos::parallel_for("br_single", 1, KOKKOS_LAMBDA(int) {
            sys.add_backreaction<mock_bounds_eos_t>(0, VEC(0, 0, 0), idx, ns, eos);
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

/// The conservation identity: composition change == net lepton number
/// removed from the radiation fields, in BOTH the plain and limited paths.
void require_lepton_identity(cell_t const& c, int nrad_nu, int nrad_anu,
                             int ystar, double ystar_before)
{
    double const dN_applied =
          (c.get_old(nrad_nu)  - c.get_new(nrad_nu))
        - (c.get_old(nrad_anu) - c.get_new(nrad_anu));
    REQUIRE_THAT(c.get_new(ystar) - ystar_before,
                 WithinAbs(dN_applied, 1e-15));
}

}  // namespace


TEST_CASE("M1 backreaction: nue/anue number changes map to Ye with the "
          "physical signs", "[m1][backreaction][lepton]")
{
    cell_t c;
    mock_bounds_eos_t eos;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    c.set_new(TAU_, 0.5);

    // Implicit step absorbed neutrinos: numbers DROP from old to new.
    c.set_old(NRAD1_, 0.10);  c.set_new(NRAD1_, 0.07);   // dN_nue  = +0.03
    c.set_old(NRAD2_, 0.05);  c.set_new(NRAD2_, 0.04);   // dN_anue = +0.01
    #if GRACE_M1_NU_SPECIES >= 5
    c.set_new(YMUSTAR_, 0.02);
    c.set_old(NRAD3_, 0.020); c.set_new(NRAD3_, 0.016);  // dN_numu  = +0.004
    c.set_old(NRAD4_, 0.010); c.set_new(NRAD4_, 0.009);  // dN_anumu = +0.001
    #endif

    // Energy/momentum exchange (species 0 only): radiation LOST energy.
    c.set_old(ERAD1_, 1.0e-3);  c.set_new(ERAD1_, 8.0e-4);   // dE  = +2e-4
    c.set_old(FRADX1_, 1.0e-5); c.set_new(FRADX1_, 0.5e-5);  // dSx = +5e-6

    c.run(eos);

    // nue absorbed => Ye UP by dN_nue; anue absorbed => Ye DOWN by dN_anue.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30 + 0.03 - 0.01, 1e-14));
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, 0.30);
    // In-bounds: the number fields themselves are NOT rescaled.
    REQUIRE_THAT(c.get_new(NRAD1_), WithinRel(0.07, 1e-14));
    REQUIRE_THAT(c.get_new(NRAD2_), WithinRel(0.04, 1e-14));

    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(0.02 + 0.004 - 0.001, 1e-14));
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, 0.02);
    #endif

    // Energy lost by radiation lands in tau, momentum in S_i.
    REQUIRE_THAT(c.get_new(TAU_), WithinRel(0.5 + 2.0e-4, 1e-14));
    REQUIRE_THAT(c.get_new(SX_),  WithinRel(5.0e-6, 1e-14));
}


TEST_CASE("M1 backreaction: Ye limiter at the table ceiling conserves lepton "
          "number", "[m1][backreaction][lepton][limiter]")
{
    cell_t c;
    mock_bounds_eos_t eos;

    double const ye0 = 0.49;
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, ye0);
    c.set_new(TAU_, 0.5);
    // Raw update would give Ye = 0.54 > yemax = 0.50 -> limiter factor 0.2.
    c.set_old(NRAD1_, 0.10);  c.set_new(NRAD1_, 0.05);   // dN_nue = +0.05
    c.set_old(NRAD2_, 0.05);  c.set_new(NRAD2_, 0.05);   // dN_anue = 0

    c.run(eos);

    // Ye lands (just barely) at the ceiling, never beyond it.
    REQUIRE(c.get_new(YESTAR_) <= eos.yemax);
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(eos.yemax, 1e-8));
    // The neutrino number field was throttled by the SAME factor ...
    REQUIRE_THAT(c.get_new(NRAD1_), WithinRel(0.10 - 0.2 * 0.05, 1e-8));
    // ... so lepton number is exactly conserved under limiting.
    require_lepton_identity(c, NRAD1_, NRAD2_, YESTAR_, ye0);
}


#if GRACE_M1_NU_SPECIES >= 5
TEST_CASE("M1 backreaction: Ymu limiter at the table floor conserves lepton "
          "number", "[m1][backreaction][lepton][limiter]")
{
    cell_t c;
    mock_bounds_eos_t eos;

    double const ymu0 = 6.0e-4;
    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);       // Ye channel neutral (no NRAD1/2 change)
    c.set_new(YMUSTAR_, ymu0);
    c.set_new(TAU_, 0.5);
    // anumu absorbed: raw Ymu = 6e-4 - 1e-3 < ymumin = 5e-4 -> factor 0.1.
    c.set_old(NRAD4_, 0.010); c.set_new(NRAD4_, 0.009);  // dN_anumu = +1e-3

    c.run(eos);

    REQUIRE(c.get_new(YMUSTAR_) >= eos.ymumin * (1.0 - 1e-12));
    REQUIRE_THAT(c.get_new(YMUSTAR_), WithinRel(eos.ymumin, 1e-8));
    REQUIRE_THAT(c.get_new(NRAD4_), WithinRel(0.010 - 0.1 * 1.0e-3, 1e-8));
    require_lepton_identity(c, NRAD3_, NRAD4_, YMUSTAR_, ymu0);
    // Ye channel untouched.
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));
}
#endif


TEST_CASE("M1 backreaction: energy-positivity limiter reverts the radiation "
          "fields", "[m1][backreaction][energy]")
{
    cell_t c;
    mock_bounds_eos_t eos;

    c.set_new(DENS_, 1.0);
    c.set_new(YESTAR_, 0.30);
    // Radiation GAINED energy (dE = -1e-3) from a cell with tiny tau: the
    // full exchange would drive tau negative.
    c.set_new(TAU_, 1.0e-6);
    c.set_old(ERAD1_, 1.0e-3);  c.set_new(ERAD1_, 2.0e-3);
    c.set_old(FRADX1_, 0.0);    c.set_new(FRADX1_, 1.0e-5);

    c.run(eos);

    // tau throttled to (just above) zero, radiation reverted to pre-collision.
    REQUIRE(c.get_new(TAU_) >= 0.0);
    REQUIRE(c.get_new(TAU_) <= 1.0e-9);
    REQUIRE_THAT(c.get_new(ERAD1_),  WithinRel(1.0e-3, 1e-14));
    REQUIRE_THAT(c.get_new(FRADX1_), WithinAbs(0.0, 1e-18));
    // Ye untouched (no number-field changes were set).
    REQUIRE_THAT(c.get_new(YESTAR_), WithinRel(0.30, 1e-14));
}


#ifdef GRACE_M1_PHOTONS
TEST_CASE("M1 photon backreaction: energy flows to tau, composition untouched",
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
    // Photons carry no lepton number: Ye*/Ymu* must be bit-identical.
    REQUIRE(c.get_new(YESTAR_) == 0.30);
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE(c.get_new(YMUSTAR_) == 0.02);
    #endif
}
#endif

#endif  // GRACE_ENABLE_M1 && GRACE_M1_NU_SPECIES >= 3
