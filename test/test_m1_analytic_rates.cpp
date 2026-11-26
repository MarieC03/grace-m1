/**
 * @file test_m1_analytic_rates.cpp
 * @brief Unit tests for the analytic neutrino emissivity/opacity rates
 *        (eas_neutrino_rates_analytic.hh).
 *
 * Coverage:
 *   [fermi]       Fermi-Dirac integral fits FD2..FD5 against exact limits
 *                 (non-degenerate, Sommerfeld-degenerate, eta=0), branch
 *                 continuity and the FDR ratio forms.
 *   [helpers]     ipow / safe_pos / finite_pos / safe_inv_pos_finite.
 *   [units]       Code-unit conversion factors and their cross-consistency.
 *   [blackbody]   Equilibrium blackbody number/energy densities.
 *   [tau]         Optical-depth fit and tau policies.
 *   [kernels]     Individual rate kernels on hand-built fugacity states:
 *                 charged-current emission/absorption, pair, plasmon,
 *                 bremsstrahlung, scattering, Kirchhoff inversion.
 *   [integration] compute_all_species end-to-end through a mock EOS.
 *
 * All kernels take a fugacity_state by value, so no EOS table is needed:
 * states are constructed by hand with representative hot-NS matter values,
 * and the end-to-end path uses a minimal mock EOS.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <grace/physics/eas_policies.hh>

#include <array>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

using namespace grace;

namespace {

constexpr double kExactTol = 1e-12;  // identities that share a code path
constexpr double kFitTol   = 1e-5;   // Takahashi fit constants are 5-digit
constexpr double kPi       = nu_constants::pi;

// Exact values of the relativistic Fermi integrals F_k(0) = (1-2^{-k}) k! zeta(k+1)
constexpr double kZeta3 = 1.2020569031595943;
constexpr double kZeta5 = 1.0369277551433699;
constexpr double kF2_0  = 1.5 * kZeta3;                              // 1.80308...
const     double kF3_0  = 7.0 * std::pow(kPi, 4) / 120.0;            // 5.68220...
constexpr double kF4_0  = 22.5 * kZeta5;                             // 23.3309...
const     double kF5_0  = 31.0 * std::pow(kPi, 6) / 252.0;           // 118.266...

// Exact Sommerfeld polynomials (degenerate limit, corrections are O(e^-eta))
double F2_degenerate(double eta) {
    return eta*eta*eta/3.0 + kPi*kPi*eta/3.0;
}
double F3_degenerate(double eta) {
    double e2 = eta*eta;
    return e2*e2/4.0 + kPi*kPi*e2/2.0 + 7.0*std::pow(kPi,4)/60.0;
}
double F4_degenerate(double eta) {
    double e2 = eta*eta;
    return e2*e2*eta/5.0 + 2.0*kPi*kPi*e2*eta/3.0 + 7.0*std::pow(kPi,4)*eta/15.0;
}
double F5_degenerate(double eta) {
    double e2 = eta*eta;
    return e2*e2*e2/6.0 + 5.0*kPi*kPi*e2*e2/6.0
         + 7.0*std::pow(kPi,4)*e2/6.0 + 31.0*std::pow(kPi,6)/126.0;
}

// Representative hot dense NS matter, fields set directly (no EOS involved).
fugacity_state make_hot_state() {
    fugacity_state F;
    F.rho_cgs  = 1.0e13;                              // g/cm^3
    F.temp_mev = 10.0;                                // MeV
    F.nb       = F.rho_cgs * nu_constants::avogadro;  // cm^-3
    F.Xn = 0.7; F.Xp = 0.3; F.Xa = 0.0; F.Xh = 0.0;
    F.Abar = 1.0; F.Zbar = 1.0;
    F.mu_p = 5.0; F.mu_n = 20.0;
    F.eta_e   = 2.0;
    F.eta_hat = (F.mu_n - F.mu_p - nu_constants::Qnp) / F.temp_mev;
    F.eta_nu  = {{0.5, -0.5, 0.0, 0.0, 0.0}};
    F.eta_pn  = 0.3 * F.nb;
    F.eta_np  = 0.7 * F.nb;
    return F;
}

// Minimal mock EOS: only the single hook make_fugacity_state calls.
struct mock_eos_t {
    double mu_e{10.0}, mu_mu{0.0}, mu_p{5.0}, mu_n{20.0};
    double Xa{0.05}, Xh{0.05}, Xn{0.6}, Xp{0.3};
    double Abar{20.0}, Zbar{8.0};

    double mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(
        double& mumu, double& mup, double& mun,
        double& Xa_o, double& Xh_o, double& Xn_o, double& Xp_o,
        double& Abar_o, double& Zbar_o,
        double& /*temp*/, double& /*rho*/, double& /*ye*/, double& /*ymu*/,
        eos_err_t& /*err*/) const
    {
        mumu = mu_mu; mup = mu_p; mun = mu_n;
        Xa_o = Xa; Xh_o = Xh; Xn_o = Xn; Xp_o = Xp;
        Abar_o = Abar; Zbar_o = Zbar;
        return mu_e;
    }
};

constexpr double kXyzOrigin[3] = {0.0, 0.0, 0.0};

}  // namespace

// ===========================================================================
// Fermi-Dirac integral fits
// ===========================================================================

TEST_CASE("FD fits reproduce the exact values at eta = 0", "[m1rates][fermi]") {
    // eta = 0 falls in the non-degenerate branch (eta_min = 1e-3); the
    // Takahashi fits are good to a few times 1e-4 there.
    REQUIRE_THAT(fermi::FD<2>::get(0.0), WithinRel(kF2_0, 1e-3));
    REQUIRE_THAT(fermi::FD<3>::get(0.0), WithinRel(kF3_0, 1e-3));
    REQUIRE_THAT(fermi::FD<4>::get(0.0), WithinRel(kF4_0, 1e-3));
    REQUIRE_THAT(fermi::FD<5>::get(0.0), WithinRel(kF5_0, 1e-3));
}

TEST_CASE("FD fits approach the non-degenerate limit k! e^eta", "[m1rates][fermi]") {
    const double eta = -20.0;
    const double e   = std::exp(eta);
    REQUIRE_THAT(fermi::FD<2>::get(eta), WithinRel(  2.0 * e, 1e-6));
    REQUIRE_THAT(fermi::FD<3>::get(eta), WithinRel(  6.0 * e, 1e-6));
    REQUIRE_THAT(fermi::FD<4>::get(eta), WithinRel( 24.0 * e, 1e-6));
    REQUIRE_THAT(fermi::FD<5>::get(eta), WithinRel(120.0 * e, 1e-6));
}

TEST_CASE("FD fits match the Sommerfeld expansion in the degenerate limit",
          "[m1rates][fermi]") {
    // For eta >> 1 the exact integrals equal the Sommerfeld polynomials up to
    // exponentially small terms; the fit numerators ARE those polynomials with
    // 5-digit coefficients, so agreement is limited only by coefficient rounding.
    for (double eta : {10.0, 20.0, 30.0}) {
        REQUIRE_THAT(fermi::FD<2>::get(eta), WithinRel(F2_degenerate(eta), kFitTol));
        REQUIRE_THAT(fermi::FD<3>::get(eta), WithinRel(F3_degenerate(eta), kFitTol));
        REQUIRE_THAT(fermi::FD<4>::get(eta), WithinRel(F4_degenerate(eta), kFitTol));
        REQUIRE_THAT(fermi::FD<5>::get(eta), WithinRel(F5_degenerate(eta), kFitTol));
    }
}

TEST_CASE("FD fits are continuous across the eta_min branch switch",
          "[m1rates][fermi]") {
    const double lo = fermi::eta_min() * 0.999;
    const double hi = fermi::eta_min() * 1.001;
    REQUIRE_THAT(fermi::FD<2>::get(hi), WithinRel(fermi::FD<2>::get(lo), 1e-2));
    REQUIRE_THAT(fermi::FD<3>::get(hi), WithinRel(fermi::FD<3>::get(lo), 1e-2));
    REQUIRE_THAT(fermi::FD<4>::get(hi), WithinRel(fermi::FD<4>::get(lo), 1e-2));
    REQUIRE_THAT(fermi::FD<5>::get(hi), WithinRel(fermi::FD<5>::get(lo), 1e-2));
}

TEST_CASE("FD fits are positive and monotonically increasing", "[m1rates][fermi]") {
    double prev2 = 0, prev3 = 0, prev4 = 0, prev5 = 0;
    for (double eta = -30.0; eta <= 30.0; eta += 0.5) {
        const double f2 = fermi::FD<2>::get(eta);
        const double f3 = fermi::FD<3>::get(eta);
        const double f4 = fermi::FD<4>::get(eta);
        const double f5 = fermi::FD<5>::get(eta);
        REQUIRE(f2 > prev2);
        REQUIRE(f3 > prev3);
        REQUIRE(f4 > prev4);
        REQUIRE(f5 > prev5);
        prev2 = f2; prev3 = f3; prev4 = f4; prev5 = f5;
    }
}

TEST_CASE("FDR ratio forms agree with the direct FD ratio", "[m1rates][fermi]") {
    // Above eta_min FDR is literally FD<N>/FD<M>.  Below, the exp(eta)
    // factors cancel algebraically in the no_exp_inv forms, so the two
    // expressions are identical up to roundoff — including at eta so negative
    // that FD itself underflows.
    for (double eta : {-5.0, -1.0, 0.0, 0.5, 2.0, 10.0}) {
        const double direct54 = fermi::FD<5>::get(eta) / fermi::FD<4>::get(eta);
        const double direct42 = fermi::FD<4>::get(eta) / fermi::FD<2>::get(eta);
        const double direct53 = fermi::FD<5>::get(eta) / fermi::FD<3>::get(eta);
        REQUIRE_THAT((fermi::FDR<5,4>::get(eta)), WithinRel(direct54, kExactTol));
        REQUIRE_THAT((fermi::FDR<4,2>::get(eta)), WithinRel(direct42, kExactTol));
        REQUIRE_THAT((fermi::FDR<5,3>::get(eta)), WithinRel(direct53, kExactTol));
    }
    // Deep non-degenerate: still finite and ~ N!/M! even where exp(eta) underflows.
    REQUIRE_THAT((fermi::FDR<5,4>::get(-700.0)), WithinRel(5.0, 1e-6));
}

// ===========================================================================
// Small helpers
// ===========================================================================

TEST_CASE("ipow matches std::pow for integer exponents", "[m1rates][helpers]") {
    for (double x : {0.3, 1.0, 2.7, 13.6}) {
        REQUIRE_THAT(ipow<0>(x), WithinRel(1.0,                kExactTol));
        REQUIRE_THAT(ipow<1>(x), WithinRel(x,                  kExactTol));
        REQUIRE_THAT(ipow<2>(x), WithinRel(std::pow(x, 2.0),   kExactTol));
        REQUIRE_THAT(ipow<3>(x), WithinRel(std::pow(x, 3.0),   kExactTol));
        REQUIRE_THAT(ipow<4>(x), WithinRel(std::pow(x, 4.0),   kExactTol));
        REQUIRE_THAT(ipow<5>(x), WithinRel(std::pow(x, 5.0),   kExactTol));
        REQUIRE_THAT(ipow<6>(x), WithinRel(std::pow(x, 6.0),   kExactTol));
        REQUIRE_THAT(ipow<8>(x), WithinRel(std::pow(x, 8.0), 1e-14));
    }
}

TEST_CASE("safe_pos / finite_pos / safe_inv_pos_finite handle edge cases",
          "[m1rates][helpers]") {
    REQUIRE(safe_pos(5.0)   == 5.0);
    REQUIRE(safe_pos(0.0)   == 1e-80);
    REQUIRE(safe_pos(-3.0)  == 1e-80);

    REQUIRE(finite_pos(1.0));
    REQUIRE_FALSE(finite_pos(0.0));
    REQUIRE_FALSE(finite_pos(-1.0));
    REQUIRE_FALSE(finite_pos(std::numeric_limits<double>::infinity()));
    REQUIRE_FALSE(finite_pos(std::numeric_limits<double>::quiet_NaN()));

    REQUIRE_THAT(safe_inv_pos_finite(4.0), WithinRel(0.25, kExactTol));
    REQUIRE(safe_inv_pos_finite(0.0)  == 0.0);
    REQUIRE(safe_inv_pos_finite(-2.0) == 0.0);
    REQUIRE(safe_inv_pos_finite(std::numeric_limits<double>::infinity()) == 0.0);
}

// ===========================================================================
// Unit conversions
// ===========================================================================

TEST_CASE("Unit conversion constants are mutually consistent", "[m1rates][units]") {
    using namespace nu_constants;
    // Within the Cactus-GF family the relations are exact:
    // 1 s = c cm, so TIMEGF = LENGTHGF * c.
    REQUIRE_THAT(TIMEGF, WithinRel(LENGTHGF * clight, 1e-10));

    // The header carries TWO unit conventions that must stay close:
    //  - legacy Cactus GF constants (LENGTHGF/TIMEGF/RHOGF, used by the
    //    Q/R/kappa code-unit conversions), and
    //  - CODATA-2018-derived solar constants (Msun_to_cm = G Msun / c^2,
    //    used by code_length_to_cm and the tau path-length geometry).
    // They differ by ~1.1e-4 relative (different G and Msun vintages).
    // This test pins that drift: if either family is edited, agreement
    // must stay at the 2e-4 level or the mixing becomes inconsistent.
    REQUIRE_THAT(LENGTHGF * Msun_to_cm, WithinRel(1.0, 2e-4));
    REQUIRE_THAT(TIMEGF * Msun_to_s,    WithinRel(1.0, 2e-4));
    // Density compounds the length offset cubed -> wider bound.
    REQUIRE_THAT(RHOGF * Msun_cgs / (Msun_to_cm*Msun_to_cm*Msun_to_cm),
                 WithinRel(1.0, 5e-4));
}

TEST_CASE("Forward and inverse density conversions round-trip", "[m1rates][units]") {
    using namespace nu_constants;
    // 1e13 g/cm^3 -> code -> back
    const double rho_cgs  = 1.0e13;
    const double rho_code = rho_cgs * RHOGF;
    REQUIRE_THAT(rho_code_to_cgs(rho_code, 1.0), WithinRel(rho_cgs, kExactTol));
    // kappa: 1/cm -> 1/CU is division by LENGTHGF
    REQUIRE_THAT(kappa_to_code(1.0, 1.0), WithinRel(1.0 / LENGTHGF, kExactTol));
}

TEST_CASE("Q and R conversions preserve the mean energy per emitted neutrino",
          "[m1rates][units]") {
    using namespace nu_constants;
    // Q/R [MeV] must map to (Q_code/R_code) * mnuc_cgs/(mev_to_erg*EPSGF) [MeV]:
    // both conversions share RHOGF/TIMEGF, so the ratio depends only on the
    // energy-per-baryon factor.
    const double Q_mev = 3.0e30;  // MeV cm^-3 s^-1
    const double R_cgs = 1.0e29;  // cm^-3 s^-1
    const double mean_E_mev = Q_mev / R_cgs;
    const double ratio_code = Q_mev_to_code(Q_mev, 1.0) / R_to_code(R_cgs, 1.0);
    REQUIRE_THAT(ratio_code * mnuc_cgs / (mev_to_erg * EPSGF),
                 WithinRel(mean_E_mev, 1e-10));
}

// ===========================================================================
// Blackbody equilibrium densities
// ===========================================================================

TEST_CASE("Blackbody mean energy is T * F3/F2", "[m1rates][blackbody]") {
    for (double T : {1.0, 5.0, 20.0}) {
        for (double eta : {-2.0, 0.0, 3.0}) {
            const double Be = black_body_energy(2.0, T, eta);
            const double Bn = black_body_number(2.0, T, eta);
            const double expected = T * fermi::FD<3>::get(eta) / fermi::FD<2>::get(eta);
            REQUIRE_THAT(Be / Bn, WithinRel(expected, kExactTol));
        }
    }
    // At eta=0 the FD mean energy is ~3.15 T.
    const double mean = black_body_energy(2.0, 1.0, 0.0) / black_body_number(2.0, 1.0, 0.0);
    REQUIRE_THAT(mean, WithinRel(kF3_0 / kF2_0, 1e-3));
}

TEST_CASE("Blackbody densities obey T^3 / T^4 scaling and g linearity",
          "[m1rates][blackbody]") {
    const double eta = 1.0;
    REQUIRE_THAT(black_body_number(2.0, 8.0, eta),
                 WithinRel( 8.0 * black_body_number(2.0, 4.0, eta), kExactTol));
    REQUIRE_THAT(black_body_energy(2.0, 8.0, eta),
                 WithinRel(16.0 * black_body_energy(2.0, 4.0, eta), kExactTol));
    REQUIRE_THAT(black_body_number(4.0, 5.0, eta),
                 WithinRel( 2.0 * black_body_number(2.0, 5.0, eta), kExactTol));
    REQUIRE_THAT(black_body_energy(4.0, 5.0, eta),
                 WithinRel( 2.0 * black_body_energy(2.0, 5.0, eta), kExactTol));
}

// ===========================================================================
// Optical depth fit and tau policies
// ===========================================================================

TEST_CASE("Analytic tau fit anchors and monotonicity", "[m1rates][tau]") {
    // Deaton+ fit: log10(tau) = 0.96 (log10 rho - 11.7) -> tau(10^11.7) = 1.
    REQUIRE_THAT(compute_analytic_tau_from_rho_cgs(std::pow(10.0, 11.7)),
                 WithinRel(1.0, 1e-10));
    REQUIRE_THAT(compute_analytic_tau_from_rho_cgs(std::pow(10.0, 12.7)),
                 WithinRel(std::pow(10.0, 0.96), 1e-10));
    // Monotonic in rho, always >= 0 and finite.
    double prev = 0.0;
    for (double lg = 8.0; lg <= 15.0; lg += 0.5) {
        const double tau = compute_analytic_tau_from_rho_cgs(std::pow(10.0, lg));
        REQUIRE(std::isfinite(tau));
        REQUIRE(tau > prev);
        prev = tau;
    }
}

TEST_CASE("tau policies: none returns zero, local_spherical does geometry",
          "[m1rates][tau]") {
    tau_policy_none none;
    REQUIRE(none.tau_init(1e-5, kXyzOrigin, 1.0, NUE, 1e13) == 0.0);
    REQUIRE(none.tau_post(1e2, 1e-5, kXyzOrigin, 1.0, NUE, 1e13) == 0.0);

    tau_policy_local_spherical sph;
    sph.r_outer_code = 10.0;
    // tau_post = kappa_cgs * (r_out - r) * code_length_to_cm
    const double xyz[3] = {3.0, 0.0, 0.0};
    const double kappa_cgs = 1.0e-6;
    const double expected = kappa_cgs * 7.0 * code_length_to_cm(1.0);
    REQUIRE_THAT(sph.tau_post(kappa_cgs, 0.0, xyz, 1.0, NUE, 0.0),
                 WithinRel(expected, 1e-12));
    // Outside the sphere: no path length, tau = 0.
    const double far[3] = {20.0, 0.0, 0.0};
    REQUIRE(sph.tau_post(kappa_cgs, 0.0, far, 1.0, NUE, 0.0) == 0.0);
}

// ===========================================================================
// Rate kernels on hand-built fugacity states
// ===========================================================================

TEST_CASE("Charged-current emission: mean energy is T * F5/F4 of eta_e",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    rates_accum out{};
    add_charged_current_emission(F, out);

    REQUIRE(out.R[NUE]    > 0.0);
    REQUIRE(out.R[NUEBAR] > 0.0);
    // Q and R share the same blocking factor, so the ratio is exact.
    REQUIRE_THAT(out.Q[NUE] / out.R[NUE],
                 WithinRel(F.temp_mev * fermi::FD<5>::get( F.eta_e)
                                      / fermi::FD<4>::get( F.eta_e), kExactTol));
    REQUIRE_THAT(out.Q[NUEBAR] / out.R[NUEBAR],
                 WithinRel(F.temp_mev * fermi::FD<5>::get(-F.eta_e)
                                      / fermi::FD<4>::get(-F.eta_e), kExactTol));
    // Heavy species receive no charged-current emission.
    REQUIRE(out.R[NUMU]    == 0.0);
    REQUIRE(out.R[NUMUBAR] == 0.0);
    REQUIRE(out.R[NUX]     == 0.0);
}

TEST_CASE("Charged-current emission: nue/nuebar symmetry under eta_e -> -eta_e",
          "[m1rates][kernels]") {
    // e- + p -> n + nue uses eta_pn; e+ + n -> p + nuebar uses eta_np.
    // Flipping eta_e and swapping (eta_pn <-> eta_np, eta_nue <-> eta_nuebar)
    // must exchange the two channels exactly.
    fugacity_state F = make_hot_state();
    rates_accum out{};
    add_charged_current_emission(F, out);

    fugacity_state G = F;
    G.eta_e = -F.eta_e;
    G.eta_pn = F.eta_np;
    G.eta_np = F.eta_pn;
    G.eta_nu[NUE]    = F.eta_nu[NUEBAR];
    G.eta_nu[NUEBAR] = F.eta_nu[NUE];
    rates_accum mirrored{};
    add_charged_current_emission(G, mirrored);

    REQUIRE_THAT(mirrored.R[NUEBAR], WithinRel(out.R[NUE],    kExactTol));
    REQUIRE_THAT(mirrored.R[NUE],    WithinRel(out.R[NUEBAR], kExactTol));
    REQUIRE_THAT(mirrored.Q[NUEBAR], WithinRel(out.Q[NUE],    kExactTol));
    REQUIRE_THAT(mirrored.Q[NUE],    WithinRel(out.Q[NUEBAR], kExactTol));
}

TEST_CASE("Charged-current emission scales linearly with nucleon availability",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    rates_accum out1{};
    add_charged_current_emission(F, out1);

    fugacity_state F2x = F;
    F2x.eta_pn *= 2.0;
    F2x.eta_np *= 2.0;
    rates_accum out2{};
    add_charged_current_emission(F2x, out2);

    REQUIRE_THAT(out2.R[NUE],    WithinRel(2.0 * out1.R[NUE],    kExactTol));
    REQUIRE_THAT(out2.R[NUEBAR], WithinRel(2.0 * out1.R[NUEBAR], kExactTol));
    REQUIRE_THAT(out2.Q[NUE],    WithinRel(2.0 * out1.Q[NUE],    kExactTol));
}

TEST_CASE("Charged-current absorption opacity: electron flavors only, "
          "zero without free nucleons", "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    rates_accum out{};
    add_charged_current_absorption_opacity(F, out);

    REQUIRE(out.kappa_a[NUE]    > 0.0);
    REQUIRE(out.kappa_a[NUEBAR] > 0.0);
    REQUIRE(out.kappa_n[NUE]    > 0.0);
    // NUX is excluded from the charged-current loop.
    REQUIRE(out.kappa_a[NUX] == 0.0);
    REQUIRE(out.kappa_n[NUX] == 0.0);

    // No free nucleons -> no absorption.
    fugacity_state empty = F;
    empty.eta_pn = 0.0;
    empty.eta_np = 0.0;
    rates_accum out0{};
    add_charged_current_absorption_opacity(empty, out0);
    REQUIRE(out0.kappa_a[NUE]    == 0.0);
    REQUIRE(out0.kappa_a[NUEBAR] == 0.0);
}

TEST_CASE("Charged-current absorption uses the incident species' nucleon target",
          "[m1rates][kernels][cc-equilibrium]") {
    auto F = make_hot_state();
    rates_accum base{};
    add_charged_current_absorption_opacity(F, base);
    for (int target : {int(NUE), int(NUEBAR)}) {
        for (double factor : {0.0, 2.0}) {
            INFO("target " << target << " availability factor " << factor);
            auto G = F;
            if (target == NUE) G.eta_np *= factor;
            else G.eta_pn *= factor;
            rates_accum changed{};
            add_charged_current_absorption_opacity(G, changed);
            for (int s : {int(NUE), int(NUEBAR)}) {
                double const scale = s == target ? factor : 1.0;
                REQUIRE_THAT(changed.kappa_a[s], WithinAbs(scale * base.kappa_a[s], 1e-12 * base.kappa_a[s]));
                REQUIRE_THAT(changed.kappa_n[s], WithinAbs(scale * base.kappa_n[s], 1e-12 * base.kappa_n[s]));
            }
        }
    }
}

TEST_CASE("Analytic charged-current energy and number sources vanish at equilibrium",
          "[m1rates][integration][cc-equilibrium]") {
    using namespace nu_constants;
    for (double eta : {-10.0, -2.0, 0.0, 2.0, 10.0}) {
        INFO("eta_nue " << eta);
        auto F = make_hot_state();
        F.mass_scale = 1.0;
        F.rho_code = F.rho_cgs * RHOGF;
        F.eta_nu[NUE] = eta;
        F.eta_nu[NUEBAR] = -eta;
        auto const rates = compute_all_species(F, true, false, false, false,
                                               kXyzOrigin, tau_policy_none{}, false);
        for (int s : {int(NUE), int(NUEBAR)}) {
            auto const& r = rates.out[s];
            double const E = Q_mev_to_code(black_body_energy(1, F.temp_mev, F.eta_nu[s]), 1)
                           / kappa_to_code(1, 1);
            double const N = R_to_code(black_body_number(1, F.temp_mev, F.eta_nu[s]), 1)
                           / kappa_to_code(1, 1);
            REQUIRE(r.eta_E > 100 * weakhub::kappa_floor_code);
            REQUIRE(r.eta_N > 100 * weakhub::kappa_floor_code);
            REQUIRE_THAT(r.eta_E - r.kappa_a * E, WithinAbs(0.0, 1e-12 * r.eta_E));
            REQUIRE_THAT(r.eta_N - r.kappa_n * N, WithinAbs(0.0, 1e-12 * r.eta_N));
            REQUIRE(r.eta_E - r.kappa_a * (1.01 * E) < 0.0);
            REQUIRE(r.eta_N - r.kappa_n * (0.99 * N) > 0.0);
        }
    }
}

TEST_CASE("Pair process: species splits and mean energy", "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    F.eta_nu = {{0, 0, 0, 0, 0}};  // equal blocking for all species
    F.eta_e  = 0.0;                // symmetric e-/e+ gas
    rates_accum out{};
    add_pair_process_emission(F, out);

    REQUIRE(out.R[NUX] > 0.0);
    const double eta_m = fermi::FDR<4,3>::get(0.0);
    const double eps_fraction = 0.5 * F.temp_mev * (eta_m + eta_m);
    REQUIRE_THAT(out.Q[NUX] / out.R[NUX], WithinRel(eps_fraction, kExactTol));

#if GRACE_M1_NU_SPECIES >= 5
    // NUX carries tau + anti-tau (1/18) = twice the mu-channel (1/36).
    REQUIRE(out.R[NUMU] > 0.0);
    REQUIRE_THAT(out.R[NUMUBAR], WithinRel(out.R[NUMU],       kExactTol));
    REQUIRE_THAT(out.R[NUX],     WithinRel(2.0 * out.R[NUMU], kExactTol));
#else
    // 3-species mode: all heavy leptons live in NUX.
    REQUIRE(out.R[NUMU]    == 0.0);
    REQUIRE(out.R[NUMUBAR] == 0.0);
#endif
}

TEST_CASE("Plasmon decay: no electron-flavour emission, heavy-lepton split",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    F.eta_nu = {{0, 0, 0, 0, 0}};
    rates_accum out{};
    add_plasmon_decay_emission(F, out, /*include_electron_flavour=*/false);

    // Weakhub variant: electron-flavour channel restored (legacy behaviour).
    rates_accum out_wh{};
    add_plasmon_decay_emission(F, out_wh, /*include_electron_flavour=*/true);
    REQUIRE(out_wh.R[NUE] > 0.0);
    REQUIRE_THAT(out_wh.R[NUEBAR], WithinRel(out_wh.R[NUE], kExactTol));
    REQUIRE_THAT(out_wh.R[NUX],    WithinRel(out.R[NUX],    kExactTol));

    // Electron flavours are deliberately excluded (reference behaviour:
    // plasmon/pair additions overproduce nue/nuebar; their emission comes
    // from Kirchhoff inversion of the beta opacity instead).
    REQUIRE(out.R[NUE]    == 0.0);
    REQUIRE(out.R[NUEBAR] == 0.0);
    REQUIRE(out.R[NUX] > 0.0);

#if GRACE_M1_NU_SPECIES >= 5
    // Degeneracy 2 for NUX (tau pair) vs 1 per muon channel.
    REQUIRE(out.R[NUMU] > 0.0);
    REQUIRE_THAT(out.R[NUMUBAR], WithinRel(out.R[NUMU],       kExactTol));
    REQUIRE_THAT(out.R[NUX],     WithinRel(2.0 * out.R[NUMU], kExactTol));
#else
    REQUIRE(out.R[NUMU]    == 0.0);
    REQUIRE(out.R[NUMUBAR] == 0.0);
#endif
}

TEST_CASE("Bremsstrahlung: heavy-lepton split sums to four pair channels",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    rates_accum out{};
    add_brems_emission(F, out);

#if GRACE_M1_NU_SPECIES >= 5
    REQUIRE(out.R[NUMU] > 0.0);
    REQUIRE_THAT(out.R[NUMUBAR], WithinRel(out.R[NUMU],       kExactTol));
    REQUIRE_THAT(out.R[NUX],     WithinRel(2.0 * out.R[NUMU], kExactTol));
    const double total = out.R[NUMU] + out.R[NUMUBAR] + out.R[NUX];
    REQUIRE_THAT(total, WithinRel(4.0 * out.R[NUMU], kExactTol));
#else
    REQUIRE(out.R[NUX] > 0.0);
    REQUIRE(out.R[NUMU]    == 0.0);
    REQUIRE(out.R[NUMUBAR] == 0.0);
#endif
    // Electron flavors get no bremsstrahlung in this implementation.
    REQUIRE(out.R[NUE]    == 0.0);
    REQUIRE(out.R[NUEBAR] == 0.0);

    // factorY is symmetric under Xn <-> Xp.
    fugacity_state G = F;
    std::swap(G.Xn, G.Xp);
    rates_accum swapped{};
    add_brems_emission(G, swapped);
#if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(swapped.R[NUMU], WithinRel(out.R[NUMU], kExactTol));
#else
    REQUIRE_THAT(swapped.R[NUX], WithinRel(out.R[NUX], kExactTol));
#endif
}

TEST_CASE("Scattering opacity: flavor-blind at equal fugacities, linear in nb",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    F.eta_nu = {{0.7, 0.7, 0.7, 0.7, 0.7}};
    rates_accum out{};
    add_scattering_opacity(F, out);

    REQUIRE(out.kappa_s[NUE] > 0.0);
    for (int s = NUEBAR; s <= NUX; ++s) {
        REQUIRE_THAT(out.kappa_s[s], WithinRel(out.kappa_s[NUE], kExactTol));
    }

    fugacity_state F2x = F;
    F2x.nb *= 2.0;
    rates_accum out2{};
    add_scattering_opacity(F2x, out2);
    REQUIRE_THAT(out2.kappa_s[NUE], WithinRel(2.0 * out.kappa_s[NUE], kExactTol));
}

TEST_CASE("Heavy-flavour thermal fits: power laws in T and rho and the brems composition factor",
          "[m1rates][kernels][scaling]") {
    // Structure of the three grey fits with blocking off (eta_nu = -50): pairs R ~ T^8, Q ~ T^9 and no rho
    // dependence at fixed eta_e; plasmon R ~ T^8, Q ~ T^9 at fixed eta_e; brems R ~ rho^2 T^4.5, Q ~ rho^2 T^5.5
    // with the (X_n^2 + X_p^2 + 28/3 X_n X_p) factor.  Pins the exponents the B1 kernel comparison relies on.
    auto state = [](double T, double rho, double Xn, double Xp) {
        fugacity_state F = make_hot_state();
        F.temp_mev = T; F.rho_cgs = rho; F.nb = rho * nu_constants::avogadro; F.Xn = Xn; F.Xp = Xp;
        F.eta_e = 0.0; F.eta_nu = {{-50.0, -50.0, -50.0, -50.0, -50.0}};
        return F;
    };
    auto rates = [](const fugacity_state& F, int which) {
        rates_accum r{};
        if (which == 0) add_pair_process_emission(F, r);
        if (which == 1) add_plasmon_decay_emission(F, r, false);
        if (which == 2) add_brems_emission(F, r);
        return r;
    };
    const double T0 = 10.0, rho0 = 1.0e13;
    const rates_accum ref[3] = {rates(state(T0, rho0, 0.7, 0.3), 0), rates(state(T0, rho0, 0.7, 0.3), 1),
                                rates(state(T0, rho0, 0.7, 0.3), 2)};
    const double pR[3] = {8.0, 8.0, 4.5}, pQ[3] = {9.0, 9.0, 5.5}, prho[3] = {0.0, 0.0, 2.0};
    for (int w = 0; w < 3; ++w) {
        INFO("process " << w << " (0 pairs, 1 plasmon, 2 brems)");
        REQUIRE(ref[w].R[NUX] > 0.0);
        for (double T : {5.0, 20.0, 40.0}) {
            const rates_accum r = rates(state(T, rho0, 0.7, 0.3), w);
            REQUIRE_THAT(r.R[NUX] / ref[w].R[NUX], WithinRel(std::pow(T / T0, pR[w]), 1e-10));
            REQUIRE_THAT(r.Q[NUX] / ref[w].Q[NUX], WithinRel(std::pow(T / T0, pQ[w]), 1e-10));
        }
        for (double rho : {1.0e11, 1.0e14}) {
            const rates_accum r = rates(state(T0, rho, 0.7, 0.3), w);
            REQUIRE_THAT(r.R[NUX] / ref[w].R[NUX], WithinRel(std::pow(rho / rho0, prho[w]), 1e-10));
            REQUIRE_THAT(r.Q[NUX] / ref[w].Q[NUX], WithinRel(std::pow(rho / rho0, prho[w]), 1e-10));
        }
    }
    const rates_accum pure_n = rates(state(T0, rho0, 1.0, 0.0), 2), mixed = rates(state(T0, rho0, 0.5, 0.5), 2);
    REQUIRE_THAT(mixed.R[NUX] / pure_n.R[NUX], WithinRel(0.25 + 0.25 + (28.0 / 3.0) * 0.25, 1e-12));
    REQUIRE_THAT(mixed.Q[NUX] / pure_n.Q[NUX], WithinRel(0.25 + 0.25 + (28.0 / 3.0) * 0.25, 1e-12));
}

TEST_CASE("Kirchhoff inversion round-trips kappa -> (Q,R) -> kappa",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    std::array<double, NUMSPECIES> g_nu{{1, 1, 0, 0, 4}};
#if GRACE_M1_NU_SPECIES >= 5
    g_nu = {{1, 1, 1, 1, 2}};
#endif
    std::array<double, NUMSPECIES> kappa_a{{1e-4, 2e-4, 3e-4, 4e-4, 5e-4}};
    std::array<double, NUMSPECIES> kappa_n{{2e-4, 3e-4, 4e-4, 5e-4, 6e-4}};

    std::array<double, NUMSPECIES> Q{}, R{};
    add_kirchhoff_emission_from_absorption_opacity(F, g_nu, kappa_a, kappa_n, Q, R);

    std::array<double, NUMSPECIES> kappa_a_back{}, kappa_n_back{};
    add_kirchhoff_absorption_opacity_from_QR(F, g_nu, Q, R, kappa_a_back, kappa_n_back);

    for (int s = 0; s < NUMSPECIES; ++s) {
        if (!(g_nu[s] > 0.0)) {
            // Zero-degeneracy species are zeroed by the emission step.
            REQUIRE(Q[s] == 0.0);
            REQUIRE(R[s] == 0.0);
            continue;
        }
        REQUIRE_THAT(kappa_a_back[s], WithinRel(kappa_a[s], kExactTol));
        REQUIRE_THAT(kappa_n_back[s], WithinRel(kappa_n[s], kExactTol));
    }
}

// ===========================================================================
// make_fugacity_state + compute_all_species through a mock EOS
// ===========================================================================

TEST_CASE("make_fugacity_state: degeneracies, nucleon availability, "
          "tau suppression of fugacities", "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    const double rho_code = 1.0e13 * RHOGF;   // 1e13 g/cc in code units
    const double T_mev    = 10.0;

    SECTION("thin limit (tau policy none) zeroes electron-type fugacities") {
        tau_policy_none none;
        fugacity_state F = make_fugacity_state(
            eos, rho_code, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, none);

        REQUIRE_THAT(F.rho_cgs, WithinRel(1.0e13, 1e-10));
        // eta_e = mu_e / T before suppression (electron eta is not suppressed).
        REQUIRE_THAT(F.eta_e, WithinRel(eos.mu_e / T_mev, 1e-12));
        // Suppression factor 1 - exp(-0) = 0 kills the neutrino fugacities.
        REQUIRE(F.eta_nu[NUE]    == 0.0);
        REQUIRE(F.eta_nu[NUEBAR] == 0.0);
        // Nucleon availabilities are non-negative and finite.
        REQUIRE(F.eta_pn >= 0.0);
        REQUIRE(F.eta_np >= 0.0);
        REQUIRE(std::isfinite(F.eta_pn));
        REQUIRE(std::isfinite(F.eta_np));
    }

    SECTION("thick limit (analytic tau at high rho) preserves fugacities") {
        // tau(1e13 g/cc) ~ 18 >> 1, so 1 - exp(-tau) ~ 1.
        tau_policy_analytic_density thick;
        fugacity_state F = make_fugacity_state(
            eos, rho_code, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, thick);

        // FIL/Margherita parity: the neutrino chemical potential subtracts Qnp.
        const double mu_nue = eos.mu_e + eos.mu_p - eos.mu_n - Qnp;
        REQUIRE_THAT(F.eta_nu[NUE], WithinRel(mu_nue / T_mev, 1e-6));
        REQUIRE_THAT(F.eta_nu[NUEBAR], WithinRel(-mu_nue / T_mev, 1e-6));
    }

    SECTION("low-density branch uses free nucleon counting") {
        // rho < 2e11 g/cc: eta_pn = nb*Yp, eta_np = nb*Yn.
        tau_policy_none none;
        const double rho_low_cgs  = 1.0e10;
        const double rho_low_code = rho_low_cgs * RHOGF;
        fugacity_state F = make_fugacity_state(
            eos, rho_low_code, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, none);
        const double nb = rho_low_cgs * avogadro;
        REQUIRE_THAT(F.eta_pn, WithinRel(nb * eos.Xp, 1e-10));
        REQUIRE_THAT(F.eta_np, WithinRel(nb * eos.Xn, 1e-10));
    }
}

TEST_CASE("compute_all_species: outputs finite and non-negative over a state grid",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;

    for (double rho_cgs : {1.0e9, 1.0e11, 1.0e13, 1.0e14}) {
        for (double T_mev : {0.5, 2.0, 10.0, 30.0}) {
            const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, rho_cgs * RHOGF, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, tau),
                /*beta=*/true, /*plasmon=*/true, /*brems=*/true, /*pair=*/true,
                kXyzOrigin, tau, /*temp_corr=*/true);

            for (int s = 0; s < NUMSPECIES; ++s) {
                const auto& o = all.out[s];
                REQUIRE(std::isfinite(o.eta_E));
                REQUIRE(std::isfinite(o.eta_N));
                REQUIRE(std::isfinite(o.kappa_a));
                REQUIRE(std::isfinite(o.kappa_n));
                REQUIRE(std::isfinite(o.kappa_s));
                REQUIRE(o.eta_E   >= 0.0);
                REQUIRE(o.eta_N   >= 0.0);
                REQUIRE(o.kappa_a >= 0.0);
                REQUIRE(o.kappa_n >= 0.0);
                REQUIRE(o.kappa_s >= 0.0);
            }
            // Hot dense matter must actually emit electron-type neutrinos.
            if (rho_cgs >= 1.0e13 && T_mev >= 10.0) {
                REQUIRE(all.out[NUE].eta_E    > 0.0);
                REQUIRE(all.out[NUEBAR].eta_E > 0.0);
            }
        }
    }
}

TEST_CASE("compute_all_species: process flags gate the rates",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
    const double rho_code = 1.0e13 * RHOGF;
    const double T_mev    = 10.0;

    SECTION("all processes off leaves only scattering") {
        const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, rho_code, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, tau),
                false, false, false, false,
                kXyzOrigin, tau, false);
        for (int s = 0; s < NUMSPECIES; ++s) {
            REQUIRE(all.out[s].eta_E   == 0.0);
            REQUIRE(all.out[s].eta_N   == 0.0);
            REQUIRE(all.out[s].kappa_a == 0.0);
            REQUIRE(all.out[s].kappa_n == 0.0);
        }
        // Scattering is unconditional.
        REQUIRE(all.out[NUE].kappa_s > 0.0);
        REQUIRE(all.out[NUX].kappa_s > 0.0);
    }

    SECTION("beta-only: electron flavors emit, NUX does not") {
        const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, rho_code, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, tau),
                true, false, false, false,
                kXyzOrigin, tau, false);
        REQUIRE(all.out[NUE].eta_E      > 0.0);
        REQUIRE(all.out[NUEBAR].eta_E   > 0.0);
        REQUIRE(all.out[NUE].kappa_a    > 0.0);
        REQUIRE(all.out[NUX].eta_E     == 0.0);
        REQUIRE(all.out[NUX].kappa_a   == 0.0);
    }
}

#if GRACE_M1_NU_SPECIES >= 5
TEST_CASE("compute_all_species: muon species suppressed below threshold",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;

    // The gate is smooth, not a hard cutoff. Verify its actual factor
    // for emission AND absorption; kappa_s intentionally survives.
    SECTION("cold matter suppresses muon emission") {
        auto const F=make_fugacity_state(eos,1.0e13*RHOGF,1.0,0.1,0.05,1.0,kXyzOrigin,tau);
        const nu_rates_all_out all = compute_all_species(
                F,
                true, true, true, true,
                kXyzOrigin, tau, false);
        rates_accum raw{};
        add_pair_process_emission(F,raw);
        add_brems_emission(F,raw);
        add_plasmon_decay_emission(F,raw,false);
        apply_kirchhoff(F,{{1,1,1,1,2}},raw);
        double const gate=muon_rate_gate(F.rho_cgs,F.temp_mev);
        REQUIRE(gate>0); REQUIRE(gate<0.01);
        for (int s : {static_cast<int>(NUMU), static_cast<int>(NUMUBAR)}) {
            REQUIRE_THAT(all.out[s].eta_E,WithinRel(Q_mev_to_code(gate*raw.Q[s],1),1e-12));
            REQUIRE_THAT(all.out[s].eta_N,WithinRel(R_to_code(gate*raw.R[s],1),1e-12));
            REQUIRE_THAT(all.out[s].kappa_a,WithinRel(kappa_to_code(gate*raw.kappa_a[s],1),1e-12));
            REQUIRE_THAT(all.out[s].kappa_n,WithinRel(kappa_to_code(gate*raw.kappa_n[s],1),1e-12));
            // the same S on (Q, kappa_a) and (R, kappa_n): E_eq = Q/kappa_a and n_eq = R/kappa_n are not gated
            REQUIRE_THAT(all.out[s].eta_E / all.out[s].kappa_a,
                         WithinRel(Q_mev_to_code(raw.Q[s],1) / kappa_to_code(raw.kappa_a[s],1), 1e-10));
            REQUIRE_THAT(all.out[s].eta_N / all.out[s].kappa_n,
                         WithinRel(R_to_code(raw.R[s],1) / kappa_to_code(raw.kappa_n[s],1), 1e-10));
        }
    }

    SECTION("the gate function: bounded, monotone, continuous, 1/2 at its two scales, open in hot dense matter") {
        // Scales 1e11 g/cc (power 5) and 2.5 MeV (power 6).  A1 (31 Oct 2026): the npe-mu Weakhub table itself
        // carries a density factor of the same form with rho0 = 1e10 g/cc; this gate acts on top of it.
        REQUIRE_THAT(muon_rate_gate(1.0e11, 1.0e3), WithinRel(0.5, 1e-12));
        REQUIRE_THAT(muon_rate_gate(1.0e16, 2.5),   WithinRel(0.5, 1e-12));
        REQUIRE(muon_rate_gate(1.0e13, 10.0) > 0.999);
        REQUIRE(muon_rate_gate(1.0e9, 1.0) < 1.0e-12);
        double prev = 0.0;
        for (int i = 0; i <= 140; ++i) {                         // rho 1e8..1e15 g/cc at T = 50 MeV
            const double rho = std::pow(10.0, 8.0 + 0.05 * i), S = muon_rate_gate(rho, 50.0);
            REQUIRE(S > 0.0); REQUIRE(S <= 1.0); REQUIRE(S >= prev);
            REQUIRE(std::abs(muon_rate_gate(rho * (1.0 + 1.0e-8), 50.0) - S) < 1.0e-7);
            prev = S;
        }
        prev = 0.0;
        for (int i = 0; i <= 100; ++i) {                         // T 0.3..30 MeV at rho = 1e14 g/cc
            const double T = 0.3 * std::pow(100.0, 0.01 * i), S = muon_rate_gate(1.0e14, T);
            REQUIRE(S > 0.0); REQUIRE(S <= 1.0); REQUIRE(S >= prev);
            REQUIRE(std::abs(muon_rate_gate(1.0e14, T * (1.0 + 1.0e-8)) - S) < 1.0e-7);
            prev = S;
        }
    }

    SECTION("hot dense matter keeps muon channels alive") {
        const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, 1.0e13 * RHOGF, /*T=*/10.0, 0.1, 0.05, 1.0, kXyzOrigin, tau),
                true, true, true, true,
                kXyzOrigin, tau, false);
        REQUIRE(all.out[NUMU].eta_E    > 0.0);
        REQUIRE(all.out[NUMUBAR].eta_E > 0.0);
    }
}
#endif

TEST_CASE("compute_all_species: heavy-lepton emissivities equal the thermal "
          "processes exactly once (no double counting)",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
    const double rho_code = 1.0e13 * RHOGF;
    const double T_mev    = 10.0;

    // Pipeline result with bremsstrahlung only (cleanest thermal process).
    const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, rho_code, T_mev, 0.1, 0.05, 1.0, kXyzOrigin, tau),
                /*beta=*/false, /*plasmon=*/false, /*brems=*/true, /*pair=*/false,
                kXyzOrigin, tau, /*temp_corr=*/false);

    // Direct kernel evaluation on the identical fugacity state.
    fugacity_state F = make_fugacity_state(
        eos, rho_code, T_mev, 0.1, 0.05, 1.0, kXyzOrigin, tau);
    rates_accum thermal{};
    add_brems_emission(F, thermal);

    REQUIRE(thermal.Q[NUX] > 0.0);
    REQUIRE_THAT(all.out[NUX].eta_E,
                 WithinRel(Q_mev_to_code(thermal.Q[NUX], 1.0), 1e-12));
    REQUIRE_THAT(all.out[NUX].eta_N,
                 WithinRel(R_to_code(thermal.R[NUX], 1.0), 1e-12));
#if GRACE_M1_NU_SPECIES >= 5
    // Exactly one copy of brems, multiplied by the existing smooth gate.
    double const gate=muon_rate_gate(F.rho_cgs,F.temp_mev);
    REQUIRE_THAT(all.out[NUMU].eta_E,
                 WithinRel(Q_mev_to_code(gate*thermal.Q[NUMU], 1.0), 1e-12));
    REQUIRE_THAT(all.out[NUMUBAR].eta_N,
                 WithinRel(R_to_code(gate*thermal.R[NUMUBAR], 1.0), 1e-12));
#endif
    // Electron flavours get nothing: brems is heavy-only and beta is off.
    REQUIRE(all.out[NUE].eta_E    == 0.0);
    REQUIRE(all.out[NUEBAR].eta_E == 0.0);
}

TEST_CASE("compute_all_species: the neutrino-temperature factor scales the opacities "
          "and leaves the emission alone", "[m1rates][integration][tnu]") {
    // Hot neutrinos in colder matter are absorbed more strongly (neutrino heating), but
    // what the matter emits depends on the matter alone.  Scaling Q and R as well made
    // cold matter emit as if hot and pinned Ye at the table bound in ejecta.
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
    const double T_mev = 5.0;
    const fugacity_state F = make_fugacity_state(
        eos, 1.0e11 * RHOGF, T_mev, 0.1, 0.0, 1.0, kXyzOrigin, tau);

    const nu_rates_all_out off = compute_all_species(
        F, true, true, true, true, kXyzOrigin, tau, false);

    // Mean energies [MeV] that give T_nu = 3 T for nue and 10 T for nuebar, and a
    // radiation field COLDER than the matter for nux (factor must stay 1).
    double eps[NUMSPECIES] = {0.0};
    eps[NUE]    =  3.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[NUE]);
    eps[NUEBAR] = 10.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[NUEBAR]);
    eps[NUX]    =  0.5 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[NUX]);
    const nu_rates_all_out on = compute_all_species(
        F, true, true, true, true, kXyzOrigin, tau, true, eps);

    const double fact[2] = {9.0, 100.0};
    for (int s : {int(NUE), int(NUEBAR)}) {
        INFO("species " << s);
        REQUIRE(off.out[s].eta_E > 1.0e-50);             // a real rate, not the floor
        REQUIRE(on.out[s].eta_E == off.out[s].eta_E);    // emission untouched, bitwise
        REQUIRE(on.out[s].eta_N == off.out[s].eta_N);
        REQUIRE_THAT(on.out[s].kappa_a, WithinRel(fact[s] * off.out[s].kappa_a, 1e-12));
        REQUIRE_THAT(on.out[s].kappa_n, WithinRel(fact[s] * off.out[s].kappa_n, 1e-12));
        REQUIRE_THAT(on.out[s].kappa_s, WithinRel(fact[s] * off.out[s].kappa_s, 1e-12));
    }
    // colder radiation never lowers an opacity
    REQUIRE(on.out[NUX].kappa_s == off.out[NUX].kappa_s);
    REQUIRE(on.out[NUX].eta_E   == off.out[NUX].eta_E);

    SECTION("hot nux radiation scales kappa_s only, the muon flavours are never corrected, eps = 0 is inert") {
        double hot[NUMSPECIES] = {0.0};
        for (int s = 0; s < NUMSPECIES; ++s) hot[s] = 3.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[s]);
        const nu_rates_all_out h = compute_all_species(F, true, true, true, true, kXyzOrigin, tau, true, hot);
        REQUIRE(off.out[NUX].kappa_s > 1.0e-50);
        REQUIRE_THAT(h.out[NUX].kappa_s, WithinRel(9.0 * off.out[NUX].kappa_s, 1e-12));
        REQUIRE(h.out[NUX].kappa_a == off.out[NUX].kappa_a);
        REQUIRE(h.out[NUX].kappa_n == off.out[NUX].kappa_n);
        REQUIRE(h.out[NUX].eta_E   == off.out[NUX].eta_E);
        REQUIRE(h.out[NUX].eta_N   == off.out[NUX].eta_N);
        for (int s : {int(NUMU), int(NUMUBAR)}) {
            INFO("species " << s);
            REQUIRE(h.out[s].eta_E   == off.out[s].eta_E);
            REQUIRE(h.out[s].eta_N   == off.out[s].eta_N);
            REQUIRE(h.out[s].kappa_a == off.out[s].kappa_a);
            REQUIRE(h.out[s].kappa_n == off.out[s].kappa_n);
            REQUIRE(h.out[s].kappa_s == off.out[s].kappa_s);
        }
        const double zero[NUMSPECIES] = {0.0, 0.0, 0.0, 0.0, 0.0};
        const nu_rates_all_out z = compute_all_species(F, true, true, true, true, kXyzOrigin, tau, true, zero);
        for (int s = 0; s < NUMSPECIES; ++s) {
            INFO("species " << s);
            REQUIRE(z.out[s].eta_E   == off.out[s].eta_E);
            REQUIRE(z.out[s].eta_N   == off.out[s].eta_N);
            REQUIRE(z.out[s].kappa_a == off.out[s].kappa_a);
            REQUIRE(z.out[s].kappa_n == off.out[s].kappa_n);
            REQUIRE(z.out[s].kappa_s == off.out[s].kappa_s);
        }
    }

    SECTION("temperature_correction_emission also scales Q and R (FIL 'corrected' routine)") {
        const nu_rates_all_out emis = compute_all_species(
            F, true, true, true, true, kXyzOrigin, tau, true, eps, /*temp_correction_emission=*/true);
        for (int s : {int(NUE), int(NUEBAR)}) {
            INFO("species " << s);
            REQUIRE_THAT(emis.out[s].eta_E, WithinRel(fact[s] * off.out[s].eta_E, 1e-12));
            REQUIRE_THAT(emis.out[s].eta_N, WithinRel(fact[s] * off.out[s].eta_N, 1e-12));
            REQUIRE(emis.out[s].kappa_a == on.out[s].kappa_a);
            REQUIRE(emis.out[s].kappa_n == on.out[s].kappa_n);
            REQUIRE(emis.out[s].kappa_s == on.out[s].kappa_s);
        }
        for (int s : {int(NUMU), int(NUMUBAR), int(NUX)}) {
            INFO("species " << s);
            REQUIRE(emis.out[s].eta_E   == on.out[s].eta_E);
            REQUIRE(emis.out[s].eta_N   == on.out[s].eta_N);
            REQUIRE(emis.out[s].kappa_a == on.out[s].kappa_a);
            REQUIRE(emis.out[s].kappa_s == on.out[s].kappa_s);
        }
    }
}

namespace {
// Constant-valued Weakhub table with nspec slots on a 2^4 grid bracketing (lrho, lT): slot s of table t holds
// (1 + t) x (1 + s) x 1e-6 cm^-1 everywhere inside, so only the assembly of the rates is tested.
weakhub::device_handle constant_weakhub_handle(double lrho, double ltemp, int nspec = 5)
{
    weakhub::device_handle h;
    h.n_species_table = nspec;
    h.nrho = 2; h.ntemp = 2; h.nye = 2; h.nymu = 2;
    const double ax[4][2] = {{lrho - 1.0, lrho + 1.0}, {ltemp - 1.0, ltemp + 1.0},
                             {0.01, 0.5}, {-12.0, -1.0}};
    h.logrho_min = ax[0][0]; h.logrho_max = ax[0][1];
    h.logtemp_min = ax[1][0]; h.logtemp_max = ax[1][1];
    h.ye_min = ax[2][0]; h.ye_max = ax[2][1];
    h.logymu_min = ax[3][0]; h.logymu_max = ax[3][1];
    Kokkos::View<double*>* axes[4] = {&h.logrho_axis, &h.logtemp_axis, &h.ye_axis, &h.logymu_axis};
    for (int a = 0; a < 4; ++a) {
        *axes[a] = Kokkos::View<double*>("wh_axis", 2);
        auto m = Kokkos::create_mirror_view(*axes[a]);
        m(0) = ax[a][0]; m(1) = ax[a][1];
        Kokkos::deep_copy(*axes[a], m);
    }
    const int ntot = nspec * 16;
    Kokkos::View<double*>* tabs[3] = {&h.kappa_a_en_table, &h.kappa_a_num_table, &h.kappa_s_table};
    for (int t = 0; t < 3; ++t) {
        *tabs[t] = Kokkos::View<double*>("wh_tab", ntot);
        auto m = Kokkos::create_mirror_view(*tabs[t]);
        for (int i = 0; i < ntot; ++i) m(i) = 1.0e-6 * (1 + t) * (1 + i % nspec);   // cm^-1
        Kokkos::deep_copy(*tabs[t], m);
    }
    h.valid = true;
    return h;
}

// The production call site is device code: evaluate compute_all_species_weakhub in a kernel (analytic extras off)
// and copy the five outputs of every species back, index s * 5 + q with q = eta_E, eta_N, kappa_a, kappa_n, kappa_s.
std::array<double, 5 * NUMSPECIES> weakhub_rates_on_device(
    weakhub::device_handle const& h, fugacity_state const& F, tau_policy_analytic_density const& tau,
    bool temp_corr = false, const double* eps = nullptr, bool temp_corr_emission = false)
{
    Kokkos::View<double*> out("wh_rates_out", 5 * NUMSPECIES);
    Kokkos::View<double*> eps_d("wh_rates_eps", NUMSPECIES);
    {
        auto m = Kokkos::create_mirror_view(eps_d);
        for (int s = 0; s < NUMSPECIES; ++s) m(s) = eps ? eps[s] : 0.0;
        Kokkos::deep_copy(eps_d, m);
    }
    Kokkos::parallel_for("wh_rates", 1, KOKKOS_LAMBDA(int) {
        double e[NUMSPECIES];
        for (int s = 0; s < NUMSPECIES; ++s) e[s] = eps_d(s);
        const double xyz[3] = {0.0, 0.0, 0.0};
        const nu_rates_all_out r = compute_all_species_weakhub(
            h, F, false, false, false, xyz, tau, temp_corr, e, temp_corr_emission);
        for (int s = 0; s < NUMSPECIES; ++s) {
            out(5 * s)     = r.out[s].eta_E;   out(5 * s + 1) = r.out[s].eta_N;
            out(5 * s + 2) = r.out[s].kappa_a; out(5 * s + 3) = r.out[s].kappa_n;
            out(5 * s + 4) = r.out[s].kappa_s;
        }
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(m, out);
    std::array<double, 5 * NUMSPECIES> a{};
    for (int i = 0; i < 5 * NUMSPECIES; ++i) a[i] = m(i);
    return a;
}
}  // namespace

#if GRACE_M1_NU_SPECIES >= 5
namespace {
// Tiny, constant-potential leptonic EOS: exercises the production residual's
// leptonic dispatch without external EOS files or the runtime singleton.
leptonic_eos_4d_t equilibrium_test_eos(double eta) {
    using E = leptonic_eos_4d_t;
    using namespace nu_constants;
    auto axis = [](char const* name, double lo, double hi) {
        Kokkos::View<double*> a(name, 2);
        auto h = Kokkos::create_mirror_view(a);
        h(0) = lo; h(1) = hi; Kokkos::deep_copy(a, h);
        return a;
    };
    auto lr = axis("eq_rho", std::log(1e14 * RHOGF), std::log(1e16 * RHOGF));
    auto lt = axis("eq_T", std::log(1.0), std::log(4.0));
    auto ye = axis("eq_Ye", 0.01, 0.5);
    auto lm = axis("eq_Ymu", std::log(1e-4), std::log(0.1));
    Kokkos::View<double****> bar("eq_bar", 2, 2, 2, E::N_TAB_VARS_BARYON);
    Kokkos::View<double****> ele("eq_ele", 2, 2, 2, E::N_TAB_VARS_ELE);
    Kokkos::View<double****> mu("eq_mu", 2, 2, 2, E::N_TAB_VARS_MUON);
    Kokkos::View<double**> cold("eq_cold", 2, E::N_CTAB_VARS);
    auto b = Kokkos::create_mirror_view(bar);
    auto e = Kokkos::create_mirror_view(ele);
    auto m = Kokkos::create_mirror_view(mu);
    Kokkos::deep_copy(b, 0.0); Kokkos::deep_copy(e, 0.0); Kokkos::deep_copy(m, 0.0);
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) for (int k=0; k<2; ++k) {
        b(i,j,k,E::TABEPS) = std::log(0.02);
        b(i,j,k,E::TABMUN) = 40.0 - Qnp - 2.0 * eta;
        b(i,j,k,E::TABXN) = 0.7; b(i,j,k,E::TABXP) = 0.3;
        b(i,j,k,E::TABABAR) = 1.0; b(i,j,k,E::TABZBAR) = 1.0;
        e(i,j,k,E::TABMUELE) = 40.0;
        m(i,j,k,E::TABMUMU) = 40.0;
    }
    Kokkos::deep_copy(bar, b); Kokkos::deep_copy(ele, e); Kokkos::deep_copy(mu, m);
    return E(bar, lr, lt, ye, ele, ye, mu, lm, cold, lr,
             1e16 * RHOGF, 1e14 * RHOGF, 4.0, 1.0, 0.5, 0.01, 0.1, 1e-4,
             1.0, 0.0, 0.0, 10.0, 1.0, 10.0, 1.0, 0.1, 1e-4, false, true);
}
} // namespace

TEST_CASE("Beta-equilibrium muon target cancels WeakHub charged-current sources",
          "[m1rates][betaeq][dilute][cc-equilibrium]") {
    using namespace nu_constants;
    using Op = neutrinos_eas_op<leptonic_eos_4d_t>;
    // Both signs beyond the old clamp, its endpoints, and the nondegenerate limit.
    for (double eta : {-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0}) {
        for (bool enabled : {false, true}) {
            INFO("eta " << eta << " dilute target " << enabled);
            auto const eos = equilibrium_test_eos(eta);
            constexpr double rho = 1e15 * RHOGF, T = 2.0, Ye = 0.1, Ymu = 0.01;
            auto const h = constant_weakhub_handle(std::log(rho), std::log(T), 6);
            Kokkos::View<double*> result("cc_equilibrium", 12);
            Kokkos::parallel_for("cc_equilibrium", 1, KOKKOS_LAMBDA(int) {
                double const xyz[3] = {0,0,0};
                tau_policy_fixed tau; tau.tau = {{1000,1000,1000,1000,1000}};
                double const mu_eta = enabled ? eta : Kokkos::fmax(-5.0, Kokkos::fmin(5.0, eta));
                double const etas[5] = {eta, -eta, mu_eta, -mu_eta, 0.0};
                double E[5], N[5];
                // The closure intentionally retains a low-density trapping cutoff.
                // Conserved targets include it; local CC equilibrium is the FULL
                // Fermi gas, so do not mistake this cutoff for the eta-clamp bug.
                double const trapped = Kokkos::exp(-1e11 / 1e15);
                double u = rho * 1.02;
                for (int s=0; s<5; ++s) {
                    N[s] = black_body_number(1, T, etas[s]) / clight * mnuc_cgs * RHOGF;
                    E[s] = black_body_energy(1, T, etas[s]) / clight * mev_to_erg * EPSGF * RHOGF;
                    u += trapped * E[s] * (s == NUX ? 2.0 : 1.0);
                }
                double const Yle = Ye + trapped * (N[NUE] - N[NUEBAR]) / rho;
                double const Ylmu = Ymu + trapped * (N[NUMU] - N[NUMUBAR]) / rho;
                // Solve the production muon-lepton residual at fixed T and Ye.
                // Constant EOS potentials make the independently specified root Ymu.
                double lo=0.001, hi=0.03, f[3] = {};
                bool ok = true;
                for (int it=0; it<60; ++it) {
                    double const mid = 0.5 * (lo + hi);
                    ok = Op::betaeq_residuals(eos, 1.0, enabled, rho, xyz, tau,
                                              Ye, mid, T, Yle, Ylmu, u, f) && ok;
                    if (f[1] > 0) hi=mid; else lo=mid;
                }
                double const solved = 0.5 * (lo + hi);
                ok = Op::betaeq_residuals(eos, 1.0, enabled, rho, xyz, tau,
                                          Ye, solved, T, Yle, Ylmu, u, f) && ok;
                result(0) = ok ? solved : -1.0;
                result(1) = f[0]; result(2) = f[1]; result(3) = f[2] / u;
                auto F = make_fugacity_state(eos, rho, T, Ye, solved, 1.0, xyz, tau);
                if (enabled) set_dilute_muon_target(F,
                    eos.raw_mumu__temp_rho_ye_ymu(T, rho, Ye, solved), true, false);
                auto const rates = compute_all_species_weakhub(h, F, false, false, false,
                                                               xyz, tau, false, nullptr);
                for (int s=0; s<4; ++s) {
                    auto const& r = rates.out[s];
                    // Normalize by emission: catches nonzero sources even for
                    // the exponentially small anti-species at |eta| = 20.
                    result(4+s) = (r.eta_E - r.kappa_a * E[s]) / r.eta_E;
                    result(8+s) = (r.eta_N - r.kappa_n * N[s]) / r.eta_N;
                }
            });
            auto const r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
            REQUIRE_THAT(r(0), WithinAbs(Ymu, 1e-12));
            for (int n=1; n<4; ++n) REQUIRE_THAT(r(n), WithinAbs(0.0, 1e-12));
            for (int n=4; n<12; ++n) REQUIRE_THAT(r(n), WithinAbs(0.0, 1e-12));
        }
    }
}

// Ng et al. 2026, Eq. 20 on the fugacity state: at the threshold the muon neutrino takes the electron
// neutrino's equilibrium degeneracy; above it the table potential returns; eta_nu is clamped unless raw.
TEST_CASE("Dilute muon target sets the muon-flavour degeneracy", "[m1rates][dilute]") {
    using namespace nu_constants;
    fugacity_state F; F.temp_mev=2.0; F.mu_e=20.0; F.mu_p=-30.0; F.mu_n=-12.0; F.eta_nu[NUMU]=-5; F.eta_nu[NUMUBAR]=5;
    double const eta_e=(F.mu_e+F.mu_p-F.mu_n-Qnp)/F.temp_mev;
    for(double ymu:{5.0e-4,6.0e-4}) {                       // floor and threshold: mu_mu = mu_e
        F.ymu=ymu; set_dilute_muon_target(F,100.0,true,false);
        REQUIRE(F.dilute_cc_target); REQUIRE(F.mu_mu_reaction==F.mu_e);
        REQUIRE_THAT(F.eta_cc_mu,Catch::Matchers::WithinRel(eta_e,1e-14));
        REQUIRE_THAT(F.eta_nu[NUMU],Catch::Matchers::WithinRel(eta_e,1e-14)); REQUIRE(F.eta_nu[NUMUBAR]==-F.eta_nu[NUMU]);
    }
    F.ymu=0.01; set_dilute_muon_target(F,100.0,true,false);  // resolved muons: table value, eta on the clamp
    REQUIRE_THAT(F.mu_mu_reaction,Catch::Matchers::WithinRel(100.0,1e-12));
    REQUIRE_THAT(F.eta_cc_mu,Catch::Matchers::WithinRel((100.0-18.0-Qnp)/2.0,1e-12));
    REQUIRE(F.eta_nu[NUMU]==5.0); REQUIRE(F.eta_nu[NUMUBAR]==-5.0);
    set_dilute_muon_target(F,100.0,true,true);               // raw: no clamp
    REQUIRE(F.eta_nu[NUMU]==F.eta_cc_mu); REQUIRE(F.eta_nu[NUMUBAR]==-F.eta_cc_mu);
    F.ymu=5.0e-4; set_dilute_muon_target(F,3.0,true,false);  // hot branch, table value below mu_e: unchanged
    REQUIRE(F.mu_mu_reaction==3.0);
    F.eta_nu[NUMU]=1.25; F.eta_nu[NUMUBAR]=-1.25; set_dilute_muon_target(F,100.0,false,false);
    REQUIRE(F.eta_nu[NUMU]==1.25); REQUIRE(F.eta_nu[NUMUBAR]==-1.25);   // set_eta false leaves eta_nu alone
}

TEST_CASE("Dilute WeakHub CC target changes emission without removing absorption", "[m1rates][weakhub][dilute]") {
    using namespace nu_constants;
    mock_eos_t eos; tau_policy_analytic_density tau;
    auto F=make_fugacity_state(eos,1e13*RHOGF,10.0,0.1,5e-4,1.0,kXyzOrigin,tau);
    F.eta_nu[NUMU]=5; F.eta_nu[NUMUBAR]=-5;
    auto const h=constant_weakhub_handle(std::log(F.rho_code),std::log(F.temp_mev),6);
    auto const before=weakhub_rates_on_device(h,F,tau);
    F.dilute_cc_target=true; F.eta_cc_mu=0;
    auto const after=weakhub_rates_on_device(h,F,tau);
    for(int s=0;s<NUMSPECIES;++s) {
        for(int v=2;v<5;++v) REQUIRE(after[5*s+v]==before[5*s+v]);
        if(s!=NUMU && s!=NUMUBAR) {
            REQUIRE(after[5*s]==before[5*s]); REQUIRE(after[5*s+1]==before[5*s+1]);
        } else {
            double const equilibrium=R_to_code(black_body_number(1,F.temp_mev,0),1)/kappa_to_code(1,1);
            REQUIRE_THAT(after[5*s+1]/after[5*s+3],WithinRel(equilibrium,1e-12));
            // Incident radiation above the new target must still be absorbed.
            REQUIRE(after[5*s+1]-after[5*s+3]*(2*equilibrium)<0);
        }
    }
    REQUIRE(after[5*NUMU+1]<before[5*NUMU+1]);
    REQUIRE(after[5*NUMUBAR+1]>before[5*NUMUBAR+1]);
}
#endif

TEST_CASE("compute_all_species_weakhub: temperature_correction_emission scales Q and R "
          "of nue/nuebar only", "[m1rates][integration][tnu][weakhub]") {
    // FIL's Weakhub routine multiplies Q, R and all kappas of nue/nuebar by the factor;
    // GRACE's default scales the kappas only.
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
    const double T_mev = 5.0;
    const fugacity_state F = make_fugacity_state(
        eos, 1.0e11 * RHOGF, T_mev, 0.1, 1.0e-3, 1.0, kXyzOrigin, tau);
    const weakhub::device_handle h = constant_weakhub_handle(std::log(F.rho_code), std::log(T_mev));

    double eps[NUMSPECIES] = {0.0};
    eps[NUE]    =  3.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[NUE]);
    eps[NUEBAR] = 10.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[NUEBAR]);
    // Hot radiation in the other species too: nux may scale kappa_s only, the muon flavours nothing (T_nu = 3 T).
    for (int s : {int(NUMU), int(NUMUBAR), int(NUX)}) eps[s] = 3.0 * T_mev / fermi::FDR<2,3>::get(F.eta_nu[s]);
    const double fact[2] = {9.0, 100.0};

    // Device-side assembly (production call site), three variants copied back.
    Kokkos::View<double*> out("wh_tnu_out", 3 * NUMSPECIES * 5);
    Kokkos::View<double*> eps_d("wh_tnu_eps", NUMSPECIES);
    {
        auto m = Kokkos::create_mirror_view(eps_d);
        for (int s = 0; s < NUMSPECIES; ++s) m(s) = eps[s];
        Kokkos::deep_copy(eps_d, m);
    }
    Kokkos::parallel_for("wh_tnu", 1, KOKKOS_LAMBDA(int) {
        double e[NUMSPECIES];
        for (int s = 0; s < NUMSPECIES; ++s) e[s] = eps_d(s);
        const double xyz[3] = {0.0, 0.0, 0.0};
        for (int v = 0; v < 3; ++v) {
            const nu_rates_all_out r = compute_all_species_weakhub(
                h, F, false, false, false, xyz, tau, v > 0, e, v == 2);
            for (int s = 0; s < NUMSPECIES; ++s) {
                const int o = (v * NUMSPECIES + s) * 5;
                out(o) = r.out[s].eta_E; out(o + 1) = r.out[s].eta_N;
                out(o + 2) = r.out[s].kappa_a; out(o + 3) = r.out[s].kappa_n;
                out(o + 4) = r.out[s].kappa_s;
            }
        }
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(m, out);
    auto val = [&](int v, int s, int q) { return m((v * NUMSPECIES + s) * 5 + q); };

    for (int s : {int(NUE), int(NUEBAR)}) {
        INFO("species " << s);
        REQUIRE(val(0, s, 0) > 1.0e-50);                         // a real rate, not the floor
        REQUIRE(val(1, s, 0) == val(0, s, 0));                   // default: Q, R untouched
        REQUIRE(val(1, s, 1) == val(0, s, 1));
        for (int q = 2; q < 5; ++q)                             // kappas scaled in both
            REQUIRE_THAT(val(1, s, q), WithinRel(fact[s] * val(0, s, q), 1e-12));
        REQUIRE_THAT(val(2, s, 0), WithinRel(fact[s] * val(0, s, 0), 1e-12));
        REQUIRE_THAT(val(2, s, 1), WithinRel(fact[s] * val(0, s, 1), 1e-12));
        for (int q = 2; q < 5; ++q) REQUIRE(val(2, s, q) == val(1, s, q));
    }
    // nux: the factor (3 T -> 9) touches kappa_s alone; numu, anumu are never corrected (the table carries their
    // spectral information); the emission flag changes nothing for the three.
    REQUIRE(val(0, NUX, 4) > 1.0e-50);
    REQUIRE_THAT(val(1, NUX, 4), WithinRel(9.0 * val(0, NUX, 4), 1e-12));
    for (int q = 0; q < 4; ++q) REQUIRE(val(1, NUX, q) == val(0, NUX, q));
    for (int s : {int(NUMU), int(NUMUBAR)})
        for (int q = 0; q < 5; ++q) {
            INFO("species " << s << " quantity " << q);
            REQUIRE(val(1, s, q) == val(0, s, q));
        }
    for (int s : {int(NUMU), int(NUMUBAR), int(NUX)})
        for (int q = 0; q < 5; ++q) {
            INFO("species " << s << " quantity " << q);
            REQUIRE(val(2, s, q) == val(1, s, q));
        }
}

#if GRACE_M1_NU_SPECIES >= 3
TEST_CASE("compute_all_species_weakhub: a 6-species table maps its slots per build, the heavy-lepton multiplicity "
          "enters the emission once, and the muon gate scales emission and absorption together",
          "[m1rates][integration][weakhub][species][gate]") {
    // Table slots 0..5 = nue, anue, numu, anumu, nutau, anutau, tags (1 + slot) x 1e-6 cm^-1 (x2 kappa_n, x3 kappa_s).
    // kappa is intensive: a clumped species holds the MEAN of its slots and the species count enters only through
    // g_nu in the Kirchhoff emission, so E_eq = eta_E / kappa_a is g_nu single-species black bodies.
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
#if GRACE_M1_NU_SPECIES >= 5
    const double tag[NUMSPECIES]  = {1.0, 2.0, 3.0, 4.0, 5.5};
    const double g_nu[NUMSPECIES] = {1.0, 1.0, 1.0, 1.0, 2.0};
#else
    const double tag[NUMSPECIES]  = {1.0, 2.0, 0.0, 0.0, 4.5};
    const double g_nu[NUMSPECIES] = {1.0, 1.0, 0.0, 0.0, 4.0};
#endif
    for (double T_mev : {10.0, 1.0}) {          // gate open (S ~ 1) and nearly closed (S ~ 4e-3) at rho = 1e13 g/cc
        const fugacity_state F = make_fugacity_state(eos, 1.0e13 * RHOGF, T_mev, 0.1, 1.0e-3, 1.0, kXyzOrigin, tau);
        const weakhub::device_handle h = constant_weakhub_handle(std::log(F.rho_code), std::log(T_mev), 6);
        const auto r = weakhub_rates_on_device(h, F, tau);
        const double S = muon_rate_gate(F.rho_cgs, F.temp_mev);
        INFO("T = " << T_mev << " MeV, gate S = " << S);
        for (int s = 0; s < NUMSPECIES; ++s) {
            INFO("species " << s);
#if GRACE_M1_NU_SPECIES >= 5
            const double gate = (s == NUMU || s == NUMUBAR) ? S : 1.0;
#else
            const double gate = 1.0;
#endif
            if (tag[s] == 0.0) {                 // slots this build does not evolve: every output at the floor
                for (int q = 0; q < 5; ++q) REQUIRE(r[5 * s + q] == weakhub::kappa_floor_code);
                continue;
            }
            const double ka = 1.0e-6 * tag[s], kn = 2.0e-6 * tag[s], ks = 3.0e-6 * tag[s];
            REQUIRE_THAT(r[5 * s + 2], WithinRel(kappa_to_code(gate * ka, 1.0), 1e-12));
            REQUIRE_THAT(r[5 * s + 3], WithinRel(kappa_to_code(gate * kn, 1.0), 1e-12));
            REQUIRE_THAT(r[5 * s + 4], WithinRel(kappa_to_code(ks, 1.0), 1e-12));       // scattering is never gated
            REQUIRE_THAT(r[5 * s], WithinRel(
                Q_mev_to_code(gate * ka * black_body_energy(g_nu[s], F.temp_mev, F.eta_nu[s]), 1.0), 1e-12));
            REQUIRE_THAT(r[5 * s + 1], WithinRel(
                R_to_code(gate * kn * black_body_number(g_nu[s], F.temp_mev, F.eta_nu[s]), 1.0), 1e-12));
            // the equilibrium the collision term relaxes to: g_nu single-species black bodies, whatever the gate
            REQUIRE_THAT(r[5 * s] / r[5 * s + 2], WithinRel(
                g_nu[s] * Q_mev_to_code(black_body_energy(1.0, F.temp_mev, F.eta_nu[s]), 1.0) / kappa_to_code(1.0, 1.0), 1e-10));
        }
#if GRACE_M1_NU_SPECIES >= 5
        if (T_mev < 2.0) { REQUIRE(S > 0.0); REQUIRE(S < 0.01); }
#endif
    }
}
#endif

TEST_CASE("compute_species wrapper agrees with compute_all_species",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;
    const double rho_code = 1.0e13 * RHOGF;

    const nu_rates_all_out all = compute_all_species(
                make_fugacity_state(eos, rho_code, 10.0, 0.1, 0.0, 1.0, kXyzOrigin, tau),
                true, true, true, true,
                kXyzOrigin, tau, true);

    for (int s = 0; s < NUMSPECIES; ++s) {
        const nu_rates_out one = compute_species(
            eos, rho_code, 10.0, 0.1, 0.0, 1.0,
            true, true, true, true, s, kXyzOrigin, tau, true);
        REQUIRE(one.eta_E   == all.out[s].eta_E);
        REQUIRE(one.eta_N   == all.out[s].eta_N);
        REQUIRE(one.kappa_a == all.out[s].kappa_a);
        REQUIRE(one.kappa_n == all.out[s].kappa_n);
        REQUIRE(one.kappa_s == all.out[s].kappa_s);
    }
}

TEST_CASE("betaeq_tau_min: only lepton-carrying species open the timescale gate",
          "[m1][rates][betaeq]")
{
    // One species trapped (kappa_a = kappa_s = 1e3), every other rate zero.
    auto const trapped = [](int s) {
        nu_rates_all_out all{};
        all.out[s].kappa_a = 1.0e3;
        all.out[s].kappa_s = 1.0e3;
        return betaeq_tau_min(all);
    };
    const double tau_trapped = 1.0 / std::sqrt(1.0e3 * 2.0e3);

    REQUIRE(trapped(NUX) > 1.0e20);   // nux alone: the gate stays closed (FIL)
    REQUIRE_THAT(trapped(NUE),    WithinRel(tau_trapped, kExactTol));
    REQUIRE_THAT(trapped(NUEBAR), WithinRel(tau_trapped, kExactTol));
#if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(trapped(NUMU),    WithinRel(tau_trapped, kExactTol));
    REQUIRE_THAT(trapped(NUMUBAR), WithinRel(tau_trapped, kExactTol));
#else
    REQUIRE(trapped(NUMU) > 1.0e20);  // not evolved at 3 species
#endif
}

#if GRACE_M1_NU_SPECIES >= 5
// ===========================================================================
// Vacuum muon decay in the standard path (emission only).
// ===========================================================================
namespace {
// Michel-averaged unblocking by a 4000-point midpoint rule, the reference for the Gauss nodes.
double unblocking_reference(double eta, double T) {
    double sum = 0.0; int const n = 4000;
    for (int i = 0; i < n; ++i) {
        double const x = (i + 0.5) / n, E = 0.5 * nu_constants::mmu_mev * x;
        sum += 2.0 * x * x * (3.0 - 2.0 * x) / (1.0 + std::exp(eta - E / T)) / n;
    }
    return sum;
}
// Dilute hot matter with an open decay: outside the muon gate, non-degenerate electrons, no neutrino sea.
fugacity_state make_decay_state() {
    fugacity_state F = make_hot_state();          // T 10
    F.rho_cgs = 1.0e8; F.nb = F.rho_cgs * nu_constants::avogadro;   // gate factor 1 - 1e-15
    F.ymu = 0.05; F.ymu_dilute = 6.0e-4; F.mu_mu = 100.0;
    F.eta_e = -60.0;
    F.eta_nu = {{-50.0, -50.0, -50.0, -50.0, 0.0}};
    return F;
}
}  // namespace

TEST_CASE("Vacuum muon decay: free rate, Michel energies, dilute-band floor",
          "[m1rates][decay]") {
    using namespace nu_constants;
    fugacity_state F = make_decay_state();
    rates_accum out{};
    add_muon_decay_emission(F, out, 0.0);
    double const R_expected = (F.ymu - F.ymu_dilute) * (F.rho_cgs / mnuc_cgs) / tau_mu_s;
    REQUIRE_THAT(out.R[NUMU],   WithinRel(R_expected, 1e-10));
    REQUIRE_THAT(out.R[NUEBAR], WithinRel(R_expected, 1e-10));
    REQUIRE_THAT(out.Q[NUMU] / out.R[NUMU],     WithinRel(0.35 * mmu_mev, 1e-12));
    REQUIRE_THAT(out.Q[NUEBAR] / out.R[NUEBAR], WithinRel(0.30 * mmu_mev, 1e-12));
    // mu+: the Boltzmann density at mu_mu = 100 MeV, T = 10 is ~1e-9 of the mu- excess.
    REQUIRE(out.R[NUMUBAR] > 0.0);
    REQUIRE(out.R[NUMUBAR] < 1e-6 * R_expected);
    REQUIRE(out.R[NUE] == out.R[NUMUBAR]);
    for (int s = 0; s < NUMSPECIES; ++s) { REQUIRE(out.kappa_a[s] == 0.0); REQUIRE(out.kappa_n[s] == 0.0); }
    REQUIRE(out.R[NUX] == 0.0);
    SECTION("nothing below the dilute band") {
        F.ymu = 5.0e-4; rates_accum z{}; add_muon_decay_emission(F, z, 0.0);
        REQUIRE(z.R[NUMU] == 0.0); REQUIRE(z.Q[NUEBAR] == 0.0);
    }
    SECTION("inside the muon gate (dense and hot) the sink carries the factor 1 - S") {
        F.rho_cgs = 1.0e13; F.nb = F.rho_cgs * avogadro;
        rates_accum z{}; add_muon_decay_emission(F, z, 0.0);
        double const open = 1.0 - muon_rate_gate(F.rho_cgs, F.temp_mev);
        REQUIRE(open < 1e-3);
        REQUIRE_THAT(z.R[NUMU], WithinRel(open * (F.ymu - F.ymu_dilute) * (F.rho_cgs / mnuc_cgs) / tau_mu_s, 1e-10));
    }
}

TEST_CASE("Vacuum muon decay: electron blocking and the Michel quadrature", "[m1rates][decay]") {
    // Degenerate electrons at low T: the Fermi sea covers the 52.8 MeV endpoint.
    REQUIRE(muon_decay_lepton_unblocking(100.0, 1.0) < 1e-10);
    REQUIRE_THAT(muon_decay_lepton_unblocking(-60.0, 10.0), WithinRel(1.0, 1e-12));
    for (double eta : {-2.0, 1.0, 3.0, 6.0}) {
        for (double T : {10.0, 30.0})
            REQUIRE_THAT(muon_decay_lepton_unblocking(eta, T), WithinRel(unblocking_reference(eta, T), 1e-6));
        REQUIRE_THAT(muon_decay_lepton_unblocking(eta, 3.0), WithinRel(unblocking_reference(eta, 3.0), 2e-3));
    }
}

TEST_CASE("Vacuum muon decay: joint neutrino blocking and the stage-safe rate", "[m1rates][decay]") {
    using namespace nu_constants;
    fugacity_state F = make_decay_state();
    rates_accum ref{}; add_muon_decay_emission(F, ref, 0.0);
    SECTION("one event makes one numu and one anti-nue: a common blocking factor") {
        F.eta_nu[NUMU] = 5.0; F.eta_nu[NUEBAR] = 2.0;
        rates_accum out{}; add_muon_decay_emission(F, out, 0.0);
        double const block = (1.0 + std::exp(5.0 - 0.35 * mmu_mev / F.temp_mev))
                           * (1.0 + std::exp(2.0 - 0.30 * mmu_mev / F.temp_mev));
        REQUIRE(out.R[NUMU] == out.R[NUEBAR]);
        REQUIRE_THAT(out.R[NUMU], WithinRel(ref.R[NUMU] / block, 1e-12));
    }
    SECTION("the emitted number over the full step never exceeds the excess") {
        double const gamma = 1.0 / tau_mu_s;   // open decay
        for (double dt_s : {1e-8, 1e-6, 1e-5}) {
            rates_accum out{}; add_muon_decay_emission(F, out, dt_s * TIMEGF);
            double const expected = ref.R[NUMU] * (-std::expm1(-gamma * dt_s)) / (gamma * dt_s);
            REQUIRE_THAT(out.R[NUMU], WithinRel(expected, 1e-12));
            REQUIRE(out.R[NUMU] * dt_s <= (F.ymu - F.ymu_dilute) * (F.rho_cgs / mnuc_cgs));
        }
    }
}

TEST_CASE("compute_all_species: the decay sink enters after the muon gate and the spectral correction",
          "[m1rates][decay][integration]") {
    using namespace nu_constants;
    mock_eos_t eos; eos.mu_mu = 100.0; eos.mu_e = 5.0;   // mock EOS: no dilute band, whole Y_mu decays
    tau_policy_analytic_density tau;
    auto const F = make_fugacity_state(eos, 1.0e13 * RHOGF, 1.0, 0.1, 0.05, 1.0, kXyzOrigin, tau);   // T = 1
    REQUIRE(F.ymu_dilute == 0.0);
    REQUIRE(muon_rate_gate(F.rho_cgs, F.temp_mev) < 0.01);
    rates_accum raw{}; add_muon_decay_emission(F, raw, 0.0);
    REQUIRE(raw.R[NUMU] > 0.0);
    double eps_rad[NUMSPECIES] = {30.0, 30.0, 30.0, 30.0, 30.0};   // hot radiation: the correction is active
    for (bool corr : {false, true}) {
        auto const off = compute_all_species(F, true, true, true, true, kXyzOrigin, tau, corr, eps_rad, corr, true, false, 0.0);
        auto const on  = compute_all_species(F, true, true, true, true, kXyzOrigin, tau, corr, eps_rad, corr, true, true,  0.0);
        for (int s : {static_cast<int>(NUMU), static_cast<int>(NUEBAR)}) {
            REQUIRE_THAT(on.out[s].eta_N - off.out[s].eta_N, WithinRel(R_to_code(raw.R[s], 1), 1e-9));
            REQUIRE_THAT(on.out[s].eta_E - off.out[s].eta_E, WithinRel(Q_mev_to_code(raw.Q[s], 1), 1e-9));
        }
        for (int s = 0; s < NUMSPECIES; ++s) {
            REQUIRE(on.out[s].kappa_a == off.out[s].kappa_a);
            REQUIRE(on.out[s].kappa_n == off.out[s].kappa_n);
            REQUIRE(on.out[s].eta_N >= off.out[s].eta_N);
        }
    }
}
#endif  // GRACE_M1_NU_SPECIES >= 5

#if GRACE_M1_NU_SPECIES >= 5 && defined(GRACE_ENABLE_MUONS)
// ===========================================================================
// Startup consistency: Weakhub table content and the vacuum muon decay sink.
// ===========================================================================
#include <grace/physics/eas_kinds.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/null_sink.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace {
struct setup_config_guard {
    YAML::Node eas = YAML::Clone(config_parser::get()["m1"]["eas"]);
    YAML::Node eos = YAML::Clone(config_parser::get()["eos"]);
    ~setup_config_guard() {
        config_parser::get()["m1"]["eas"] = eas;
        config_parser::get()["eos"] = eos;
    }
    // A Weakhub run with the vacuum decay switched on; the lightweight Kokkos main has no GRACE
    // loggers, and the ERROR/INFO paths need them (no files are written by these sinks).
    void enable(std::string const& table) {
        if (!spdlog::get("output_console")) spdlog::stdout_logger_mt("output_console");
        if (!spdlog::get("error_console")) spdlog::stderr_logger_mt("error_console");
        for (auto const* prefix : {"file_logger_", "error_file_logger_", "backtrace_logger_"}) {
            std::string const name = std::string(prefix) + std::to_string(parallel::mpi_comm_rank());
            if (!spdlog::get(name)) spdlog::register_logger(std::make_shared<spdlog::logger>(
                name, std::make_shared<spdlog::sinks::null_sink_mt>()));
        }
        auto& c = config_parser::get();
        c["m1"]["eas"]["kind"] = "neutrino_weakhub";
        c["m1"]["eas"]["kinds"] = std::vector<std::string>{};
        c["m1"]["eas"]["weakhub_table"] = table;
        c["m1"]["eas"]["weakhub_table_content"] = "auto";
        c["m1"]["eas"]["pair_treatment"] = "standard";
        c["m1"]["eas"]["muon_decay"] = true;
        c["m1"]["eas"]["pair_annihilation"] = true;
        c["m1"]["eas"]["bremsstrahlung"] = true;
        c["eos"]["eos_type"] = "leptonic";
        c["eos"]["leptonic"]["dilute_muon_suppression"] = true;
    }
};
// A Weakhub tableinfo log next to a (non-existent) table in a scratch directory.
std::string write_tableinfo(char const* stem, bool pair, bool brems, bool decay) {
    std::string const dir = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp");
    std::string const table = dir + "/" + stem + "_Weakhub_grey.h5";
    std::ofstream log(dir + "/" + stem + "_Weakhub_grey_tableinfo.log");
    log << " Interactions included in 4D table:----------------------------\n"
        << " Nucleon beta processes     : T\n"
        << " Electron-positron pair (Emiss) :" << (pair ? 'T' : 'F') << "\n"
        << " Nucleon Nucleon Brems  (Emiss) :" << (brems ? 'T' : 'F') << "\n"
        << " Nucleon Nucleon Brems (Kernel) :F\n"
        << " Electron-positron pair (Kernel) :F\n";
    if (decay) log << " Inverse lepton decay               : T\n";
    return table;
}
}  // namespace

TEST_CASE("Weakhub table content is read from the companion log and switches the extras off",
          "[m1rates][setup]") {
    setup_config_guard guard;
    guard.enable(write_tableinfo("grace_setup_nopp", false, false, false));
    REQUIRE(get_weakhub_table_content(true) == weakhub_content_t::none);
    check_m1_setup();
    REQUIRE(get_param<bool>("m1", "eas", "pair_annihilation"));
    REQUIRE(get_param<bool>("m1", "eas", "bremsstrahlung"));

    config_parser::get()["m1"]["eas"]["weakhub_table"] = write_tableinfo("grace_setup_pp", true, true, false);
    REQUIRE(get_weakhub_table_content(true) == weakhub_content_t::thermal);
    check_m1_setup();
    REQUIRE_FALSE(get_param<bool>("m1", "eas", "pair_annihilation"));
    REQUIRE_FALSE(get_param<bool>("m1", "eas", "bremsstrahlung"));

    config_parser::get()["m1"]["eas"]["weakhub_table"] = write_tableinfo("grace_setup_ild", true, true, true);
    REQUIRE(get_weakhub_table_content(true) == weakhub_content_t::thermal_decay);

    config_parser::get()["m1"]["eas"]["weakhub_table_content"] = "thermal";
    REQUIRE(get_weakhub_table_content(true) == weakhub_content_t::thermal);   // an explicit value wins
}

// Run in a subprocess by check_m1_setup.cmake: each setup terminates via ERROR.
TEST_CASE("M1 setup rejects incompatible table content and decay settings", "[.][m1-setup-invalid]") {
    setup_config_guard guard;
    guard.enable(write_tableinfo("grace_setup_invalid", false, false, false));
    auto& c = config_parser::get();
    char const* env = std::getenv("GRACE_M1_SETUP_INVALID"); REQUIRE(env != nullptr);
    std::string const mode = env;
    if (mode == "log-missing") c["m1"]["eas"]["weakhub_table"] = "/nonexistent/table_Weakhub_grey.h5";
    else if (mode == "bad-keyword") c["m1"]["eas"]["weakhub_table_content"] = "maybe";
    else if (mode == "decay-table") c["m1"]["eas"]["weakhub_table_content"] = "thermal+decay";
    else if (mode == "sink-eos") c["eos"]["eos_type"] = "ideal";
    else if (mode == "sink-suppression") c["eos"]["leptonic"]["dilute_muon_suppression"] = false;
    else FAIL("Unknown invalid setup case");
    (void)get_weakhub_table_content(true);
    check_m1_setup();
    FAIL("Incompatible M1 setup was accepted");
}
#endif  // GRACE_M1_NU_SPECIES >= 5 && GRACE_ENABLE_MUONS
