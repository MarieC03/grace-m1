/**
 * @file test_weakhub_table.cpp
 * @brief Unit tests for the Weakhub opacity-table lookup
 *        (grace_weakhub_table.hh) on synthetic in-memory tables.
 *
 * Coverage:
 *   [species]  Table-slot -> M1-species mapping for 3-, 5- and 6-species
 *              tables under the current GRACE_M1_NU_SPECIES build.  The key
 *              regression: a 6-species table consumed by a <=3-species build
 *              must SUM table slots 2..5 into NUX (slot 4).  The historic bug
 *              mapped numu/anumu into the dead slots 2,3 (g_nu = 0 there) and
 *              set NUX = nutau + anutau only, halving the heavy-lepton
 *              opacity -> under-cooled matter -> T runaway to the table
 *              ceiling -> NaN.  The build-independent invariant is that the
 *              total heavy-lepton opacity budget (slots 2+3+4 of the output)
 *              equals the sum of table slots 2..5.
 *   [interp]   Multilinear interpolation reproduces multilinear data exactly,
 *              including at axis endpoints.
 *   [clamp]    Out-of-range (rho, T, ye, ymu) queries clamp to the table
 *              bounds; nymu == 1 collapses the muon axis; an invalid handle
 *              returns zeros.
 *
 * All tables are built in memory with hand-chosen values -- no HDF5 file or
 * parfile-side weakhub configuration is needed.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/grace_weakhub_table.hh>

#include <array>
#include <cmath>

using Catch::Matchers::WithinRel;

namespace {

using namespace grace;

constexpr double kTol = 1e-13;

// lookup() floors every non-positive / non-finite opacity to 1e-60 before
// returning (positivity guard for downstream divisions), so "empty" output
// slots carry kFloor, never 0.0, and all synthetic table data must be
// strictly positive for exactness checks.
constexpr double kFloor = 1.0e-60;

// Synthetic axis bounds (log space where the production loader logs them).
constexpr double kLR[2] = {-4.0,  0.0};   // log(rho_code)
constexpr double kLT[2] = { 0.0,  3.0};   // log(T_MeV)
constexpr double kYE[2] = { 0.05, 0.55};  // Y_e (linear axis)
constexpr double kLM[2] = {-9.0, -2.0};   // log(Y_mu)

/// Build a device_handle around synthetic 2x2x2x{1,2} tables.
/// `value(table, s, lrho, ltemp, ye, lymu)` supplies the corner values;
/// table = 0/1/2 selects kappa_a_en / kappa_a_num / kappa_s.
template <typename F>
weakhub::device_handle make_handle(int nspec, int nymu, F value)
{
    weakhub::device_handle h;
    h.n_species_table = nspec;
    h.nrho = 2; h.ntemp = 2; h.nye = 2; h.nymu = nymu;
    h.logrho_min  = kLR[0]; h.logrho_max  = kLR[1];
    h.logtemp_min = kLT[0]; h.logtemp_max = kLT[1];
    h.ye_min      = kYE[0]; h.ye_max      = kYE[1];
    h.logymu_min  = kLM[0];
    h.logymu_max  = (nymu > 1 ? kLM[1] : kLM[0]);

    h.logrho_axis  = Kokkos::View<double*>("wh_lrho",  2);
    h.logtemp_axis = Kokkos::View<double*>("wh_ltemp", 2);
    h.ye_axis      = Kokkos::View<double*>("wh_ye",    2);
    h.logymu_axis  = Kokkos::View<double*>("wh_lymu",  nymu);
    {
        auto lr = Kokkos::create_mirror_view(h.logrho_axis);
        auto lt = Kokkos::create_mirror_view(h.logtemp_axis);
        auto ye = Kokkos::create_mirror_view(h.ye_axis);
        auto lm = Kokkos::create_mirror_view(h.logymu_axis);
        for (int i = 0; i < 2; ++i) { lr(i) = kLR[i]; lt(i) = kLT[i]; ye(i) = kYE[i]; }
        for (int i = 0; i < nymu; ++i) lm(i) = kLM[i];
        Kokkos::deep_copy(h.logrho_axis,  lr);
        Kokkos::deep_copy(h.logtemp_axis, lt);
        Kokkos::deep_copy(h.ye_axis,      ye);
        Kokkos::deep_copy(h.logymu_axis,  lm);
    }

    const int ntot = nspec * 2 * 2 * 2 * nymu;
    h.kappa_a_en_table  = Kokkos::View<double*>("wh_ka_en",  ntot);
    h.kappa_a_num_table = Kokkos::View<double*>("wh_ka_num", ntot);
    h.kappa_s_table     = Kokkos::View<double*>("wh_ks",     ntot);
    {
        auto en = Kokkos::create_mirror_view(h.kappa_a_en_table);
        auto nm = Kokkos::create_mirror_view(h.kappa_a_num_table);
        auto ks = Kokkos::create_mirror_view(h.kappa_s_table);
        for (int im = 0; im < nymu; ++im)
        for (int iy = 0; iy < 2; ++iy)
        for (int it = 0; it < 2; ++it)
        for (int ir = 0; ir < 2; ++ir)
        for (int s  = 0; s  < nspec; ++s) {
            // Same flattening as device_handle::flat_index.
            const int idx = s + nspec * (ir + 2 * (it + 2 * (iy + 2 * im)));
            en(idx) = value(0, s, kLR[ir], kLT[it], kYE[iy], kLM[im]);
            nm(idx) = value(1, s, kLR[ir], kLT[it], kYE[iy], kLM[im]);
            ks(idx) = value(2, s, kLR[ir], kLT[it], kYE[iy], kLM[im]);
        }
        Kokkos::deep_copy(h.kappa_a_en_table,  en);
        Kokkos::deep_copy(h.kappa_a_num_table, nm);
        Kokkos::deep_copy(h.kappa_s_table,     ks);
    }
    h.valid = true;
    return h;
}

/// Species-tagged constant tables: table t, species s -> (s+1) * 10^t.
/// Constant per species, so any interpolation point returns the tag exactly
/// and the species-slot mapping can be read off unambiguously.
struct species_tag_fill {
    double operator()(int t, int s, double, double, double, double) const {
        const double mult = (t == 0 ? 1.0 : (t == 1 ? 10.0 : 100.0));
        return (s + 1) * mult;
    }
};

/// Multilinear-in-axes fill (species-independent): reproduced exactly by
/// the multilinear interpolation at any interior or boundary point.  The
/// constant term keeps every corner value strictly positive so the output
/// positivity floor never engages.
constexpr double kLinC[5] = {200.0, 3.0, 5.0, 7.0, 11.0};
double linear_value(double lr, double lt, double ye, double lm)
{
    return kLinC[0] + kLinC[1]*lr + kLinC[2]*lt + kLinC[3]*ye + kLinC[4]*lm;
}
struct linear_fill {
    double operator()(int, int, double lr, double lt, double ye, double lm) const {
        return linear_value(lr, lt, ye, lm);
    }
};

/// Run device_handle::lookup inside a Kokkos kernel (the production call
/// site is device code) and copy the 3x5 outputs back to the host.
std::array<double, 15> device_lookup(
    weakhub::device_handle const& h,
    double rho_code, double temp_mev, double yle, double ymu)
{
    Kokkos::View<double*> out("wh_out", 15);
    Kokkos::parallel_for("wh_lookup", 1, KOKKOS_LAMBDA(int) {
        const weakhub::interp_outputs r = h.lookup(rho_code, temp_mev, yle, ymu);
        for (int s = 0; s < 5; ++s) {
            out(s)      = r.kappa_a_en[s];
            out(5 + s)  = r.kappa_a_num[s];
            out(10 + s) = r.kappa_s[s];
        }
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(m, out);
    std::array<double, 15> a;
    for (int i = 0; i < 15; ++i) a[i] = m(i);
    return a;
}

// Interior query point (mid-domain in every axis, in physical units).
constexpr double kRho  = 0.13533528323661270;  // exp(-2)
constexpr double kTemp = 4.4816890703380645;   // exp(1.5)
constexpr double kYe   = 0.30;
constexpr double kYmu  = 0.0040867714384640666; // exp(-5.5)

}  // namespace

// ===========================================================================
// Species-slot mapping
// ===========================================================================

TEST_CASE("6-species table: heavy-lepton slots map per build, total budget "
          "conserved", "[weakhub][species]")
{
    auto h = make_handle(6, 2, species_tag_fill{});
    const auto r = device_lookup(h, kRho, kTemp, kYe, kYmu);

    // nue / anue map straight through in every build.
    REQUIRE_THAT(r[0], WithinRel(1.0, kTol));
    REQUIRE_THAT(r[1], WithinRel(2.0, kTol));

    // Build-independent invariant -- THE regression for the T-runaway NaN:
    // the heavy-lepton opacity budget across output slots 2..4 must equal
    // the sum of table slots 2..5 (tags 3+4+5+6 = 18).  The broken mapping
    // gave 3+4 dead in slots 2,3 and only 5+6=11 in NUX.
    for (int t = 0; t < 3; ++t) {
        const double mult = (t == 0 ? 1.0 : (t == 1 ? 10.0 : 100.0));
        const double heavy = r[5*t + 2] + r[5*t + 3] + r[5*t + 4];
        INFO("table " << t << " (0=ka_en, 1=ka_num, 2=ks)");
        REQUIRE_THAT(heavy, WithinRel(18.0 * mult, kTol));
    }

#if GRACE_M1_NU_SPECIES <= 3
    // <=3-species build: the single NUX field represents ALL four heavy
    // species; slots 2,3 have g_nu = 0 and stay at the positivity floor.
    REQUIRE(r[2] == kFloor);
    REQUIRE(r[3] == kFloor);
    REQUIRE_THAT(r[4], WithinRel(18.0, kTol));
    REQUIRE_THAT(r[9],  WithinRel(180.0, kTol));
    REQUIRE_THAT(r[14], WithinRel(1800.0, kTol));
#else
    // 5-species build: numu, anumu live in slots 2,3; NUX = nutau + anutau.
    REQUIRE_THAT(r[2], WithinRel(3.0, kTol));
    REQUIRE_THAT(r[3], WithinRel(4.0, kTol));
    REQUIRE_THAT(r[4], WithinRel(11.0, kTol));
    REQUIRE_THAT(r[9],  WithinRel(110.0, kTol));
    REQUIRE_THAT(r[14], WithinRel(1100.0, kTol));
#endif
}

TEST_CASE("3-species table: slot 2 lands in NUX, muon slots stay zero",
          "[weakhub][species]")
{
    auto h = make_handle(3, 2, species_tag_fill{});
    const auto r = device_lookup(h, kRho, kTemp, kYe, kYmu);

    REQUIRE_THAT(r[0], WithinRel(1.0, kTol));
    REQUIRE_THAT(r[1], WithinRel(2.0, kTol));
    REQUIRE(r[2] == kFloor);
    REQUIRE(r[3] == kFloor);
    REQUIRE_THAT(r[4],  WithinRel(3.0,   kTol));
    REQUIRE_THAT(r[9],  WithinRel(30.0,  kTol));
    REQUIRE_THAT(r[14], WithinRel(300.0, kTol));
}

TEST_CASE("5-species table: identity slot mapping", "[weakhub][species]")
{
    auto h = make_handle(5, 2, species_tag_fill{});
    const auto r = device_lookup(h, kRho, kTemp, kYe, kYmu);

    for (int s = 0; s < 5; ++s) {
        REQUIRE_THAT(r[s],      WithinRel(double(s + 1),         kTol));
        REQUIRE_THAT(r[5 + s],  WithinRel(double(s + 1) * 10.0,  kTol));
        REQUIRE_THAT(r[10 + s], WithinRel(double(s + 1) * 100.0, kTol));
    }
}

TEST_CASE("Unrecognized species count yields floor opacity (silent -- "
          "documented)", "[weakhub][species]")
{
    // n_species_table = 4 hits the (commented-out ERROR) else branch and
    // silently yields the 1e-60 floor opacity everywhere.  This test pins
    // the current behaviour; if the ERROR is ever re-enabled this test
    // should be updated to expect the abort instead.
    auto h = make_handle(4, 2, species_tag_fill{});
    const auto r = device_lookup(h, kRho, kTemp, kYe, kYmu);
    for (int i = 0; i < 15; ++i) REQUIRE(r[i] == kFloor);
}

// ===========================================================================
// Interpolation
// ===========================================================================

TEST_CASE("Multilinear interpolation is exact for multilinear data",
          "[weakhub][interp]")
{
    auto h = make_handle(3, 2, linear_fill{});

    struct pt { double lr, lt, ye, lm; };
    // Interior points plus the exact lower and upper table corners
    // (find_bracket endpoint paths).
    const pt pts[] = {
        {-2.0, 1.5, 0.30, -5.5},
        {-3.2, 0.4, 0.10, -8.0},
        {-0.5, 2.7, 0.50, -2.5},
        {kLR[0], kLT[0], kYE[0], kLM[0]},
        {kLR[1], kLT[1], kYE[1], kLM[1]},
    };
    for (const auto& p : pts) {
        const auto r = device_lookup(h, std::exp(p.lr), std::exp(p.lt),
                                     p.ye, std::exp(p.lm));
        const double expect = linear_value(p.lr, p.lt, p.ye, p.lm);
        INFO("lrho=" << p.lr << " ltemp=" << p.lt
             << " ye=" << p.ye << " lymu=" << p.lm);
        // nue slot carries the interpolant directly in a 3-species table.
        REQUIRE_THAT(r[0],  WithinRel(expect, 1e-12));
        REQUIRE_THAT(r[5],  WithinRel(expect, 1e-12));
        REQUIRE_THAT(r[10], WithinRel(expect, 1e-12));
    }
}

// ===========================================================================
// Clamping and degenerate axes
// ===========================================================================

TEST_CASE("Out-of-range queries clamp to the table bounds", "[weakhub][clamp]")
{
    auto h = make_handle(3, 2, linear_fill{});

    // rho far above the table: clamps to logrho_max; everything else interior.
    {
        const auto r = device_lookup(h, std::exp(kLR[1] + 5.0), kTemp, kYe, kYmu);
        REQUIRE_THAT(r[0], WithinRel(
            linear_value(kLR[1], 1.5, kYe, -5.5), 1e-12));
    }
    // T far below the table: clamps to logtemp_min.
    {
        const auto r = device_lookup(h, kRho, std::exp(kLT[0] - 5.0), kYe, kYmu);
        REQUIRE_THAT(r[0], WithinRel(
            linear_value(-2.0, kLT[0], kYe, -5.5), 1e-12));
    }
    // ye above the table: clamps to ye_max.
    {
        const auto r = device_lookup(h, kRho, kTemp, 0.9, kYmu);
        REQUIRE_THAT(r[0], WithinRel(
            linear_value(-2.0, 1.5, kYE[1], -5.5), 1e-12));
    }
    // ymu below the table (including ymu = 0 -> log guarded): clamps to
    // logymu_min.
    for (double ymu : {std::exp(kLM[0] - 4.0), 0.0}) {
        const auto r = device_lookup(h, kRho, kTemp, kYe, ymu);
        REQUIRE_THAT(r[0], WithinRel(
            linear_value(-2.0, 1.5, kYe, kLM[0]), 1e-12));
    }
}

TEST_CASE("nymu == 1 collapses the muon axis", "[weakhub][clamp]")
{
    // Fill independent of lymu so the expected value is unambiguous.
    auto h = make_handle(3, 1, [](int, int, double lr, double lt, double ye,
                                  double) {
        return linear_value(lr, lt, ye, 0.0) ;
    });

    // Any ymu (huge, tiny, zero) must give the same answer: clamp_state
    // forces ymu = 0 and interp skips the muon axis entirely.
    const double ref = linear_value(-2.0, 1.5, kYe, 0.0);
    for (double ymu : {0.0, 1e-30, 0.05, 10.0}) {
        const auto r = device_lookup(h, kRho, kTemp, kYe, ymu);
        REQUIRE_THAT(r[0], WithinRel(ref, 1e-12));
        REQUIRE(std::isfinite(r[0]));
    }
}

TEST_CASE("Invalid handle returns all zeros", "[weakhub][clamp]")
{
    weakhub::device_handle h;  // valid = false, views unallocated
    REQUIRE_FALSE(h.valid);
    // lookup must early-out before touching any view.
    Kokkos::View<double*> out("wh_out_invalid", 15);
    Kokkos::parallel_for("wh_lookup_invalid", 1, KOKKOS_LAMBDA(int) {
        const weakhub::interp_outputs r = h.lookup(1.0, 1.0, 0.1, 0.0);
        for (int s = 0; s < 5; ++s) {
            out(s)      = r.kappa_a_en[s];
            out(5 + s)  = r.kappa_a_num[s];
            out(10 + s) = r.kappa_s[s];
        }
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(m, out);
    for (int i = 0; i < 15; ++i) REQUIRE(m(i) == 0.0);
}
