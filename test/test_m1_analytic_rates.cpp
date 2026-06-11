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

#ifdef M1_NU_FIVESPECIES
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

TEST_CASE("Plasmon decay: electron channels equal, heavy-lepton split",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    F.eta_nu = {{0, 0, 0, 0, 0}};
    rates_accum out{};
    add_plasmon_decay_emission(F, out);

    REQUIRE(out.R[NUE] > 0.0);
    REQUIRE_THAT(out.R[NUEBAR], WithinRel(out.R[NUE], kExactTol));
    REQUIRE_THAT(out.Q[NUEBAR], WithinRel(out.Q[NUE], kExactTol));
    REQUIRE(out.R[NUX] > 0.0);

#ifdef M1_NU_FIVESPECIES
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

#ifdef M1_NU_FIVESPECIES
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
#ifdef M1_NU_FIVESPECIES
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

TEST_CASE("Kirchhoff inversion round-trips kappa -> (Q,R) -> kappa",
          "[m1rates][kernels]") {
    fugacity_state F = make_hot_state();
    std::array<double, NUMSPECIES> g_nu{{1, 1, 0, 0, 4}};
#ifdef M1_NU_FIVESPECIES
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
                eos, rho_cgs * RHOGF, T_mev, 0.1, 0.0, 1.0,
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
            eos, rho_code, T_mev, 0.1, 0.0, 1.0,
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
            eos, rho_code, T_mev, 0.1, 0.0, 1.0,
            true, false, false, false,
            kXyzOrigin, tau, false);
        REQUIRE(all.out[NUE].eta_E      > 0.0);
        REQUIRE(all.out[NUEBAR].eta_E   > 0.0);
        REQUIRE(all.out[NUE].kappa_a    > 0.0);
        REQUIRE(all.out[NUX].eta_E     == 0.0);
        REQUIRE(all.out[NUX].kappa_a   == 0.0);
    }
}

#ifdef M1_NU_FIVESPECIES
TEST_CASE("compute_all_species: muon species suppressed below threshold",
          "[m1rates][integration]") {
    using namespace nu_constants;
    mock_eos_t eos;
    tau_policy_analytic_density tau;

    // Below either threshold (rho < 1e10 g/cc OR T < 2.5 MeV) the muon
    // channels are zeroed; kappa_s intentionally survives.
    SECTION("cold matter suppresses muon emission") {
        const nu_rates_all_out all = compute_all_species(
            eos, 1.0e13 * RHOGF, /*T=*/1.0, 0.1, 0.05, 1.0,
            true, true, true, true,
            kXyzOrigin, tau, false);
        for (int s : {static_cast<int>(NUMU), static_cast<int>(NUMUBAR)}) {
            REQUIRE(all.out[s].eta_E   == 0.0);
            REQUIRE(all.out[s].eta_N   == 0.0);
            REQUIRE(all.out[s].kappa_a == 0.0);
            REQUIRE(all.out[s].kappa_n == 0.0);
        }
    }

    SECTION("hot dense matter keeps muon channels alive") {
        const nu_rates_all_out all = compute_all_species(
            eos, 1.0e13 * RHOGF, /*T=*/10.0, 0.1, 0.05, 1.0,
            true, true, true, true,
            kXyzOrigin, tau, false);
        REQUIRE(all.out[NUMU].eta_E    > 0.0);
        REQUIRE(all.out[NUMUBAR].eta_E > 0.0);
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
        eos, rho_code, 10.0, 0.1, 0.0, 1.0,
        true, true, true, true, kXyzOrigin, tau, true);

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
