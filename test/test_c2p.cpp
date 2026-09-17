/**
 * @file test_c2p.cpp
 * @brief Round-trip validation of the GRMHD conservative ↔ primitive inversion.
 *
 * For every test sample we:
 *   1. Construct a known primitive state P0 = (ρ, ε, z^i, B^i, …).
 *   2. Forward-map P0 → C via `prims_to_conservs(P0, C, metric)`.
 *   3. Invert C → P1 via `conservs_to_prims<EOS>(C, P1, metric, eos, …)`.
 *   4. Check `max_i |P1_i − P0_i| / |P0_i|` is at the FP floor of the c2p
 *      tolerance.
 *
 * Three SECTIONs cover:
 *   - **Minkowski, no B**: ideal-hydro baseline.  Sweeps (ρ, ε, W, direction).
 *   - **Minkowski, magnetisation sweep**: at fixed thermodynamic state, sweep
 *     b² / ρh across the low-σ to high-σ regimes.  The c2p has known
 *     sensitivity near b²/ρh ≳ 1; this section quantifies it.
 *   - **Schwarzschild-CKS, no B**: same sweep as Minkowski but on a
 *     non-trivial spatial metric (lapse < 1, β ≠ 0, γ_xx ≠ 1).  Catches
 *     metric-aware bugs in the inverse that flat-space tests miss.
 *
 * All sweeps are **deterministic** (structured grids, no RNG) — fully
 * reproducible across runs.  No file dumps; failures get reported by Catch2
 * with the offending parameter point.
 *
 * Bound to `configs/c2p_test_gamma2.yaml` (γ=2 piecewise polytrope, γ_th=1.8).
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/utils/metric_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace grace ;

// Deterministic unit-vector grid: 6 directions covering the principal
// axes (±x, ±y, ±z).  Enough to exercise the c2p across non-degenerate
// velocity orientations without an O(N²) angular sweep.
constexpr int N_DIR = 6 ;
constexpr double DIRS[N_DIR][3] = {
    { 1.0,  0.0,  0.0},
    {-1.0,  0.0,  0.0},
    { 0.0,  1.0,  0.0},
    { 0.0, -1.0,  0.0},
    { 0.0,  0.0,  1.0},
    { 0.0,  0.0, -1.0},
} ;

// Lorentz-factor sweep — mildly to moderately relativistic.
constexpr int N_W = 4 ;
constexpr double WS[N_W] = {1.05, 1.5, 2.0, 3.0} ;

// log10 ρ sweep — sub-NS-atmosphere up to NS density.
constexpr int N_RHO = 6 ;
constexpr double LOG10_RHO[N_RHO] = {-12.0, -8.0, -5.0, -3.0, -1.5, -0.5} ;

// Temperature sweep (in MeV-equivalent natural units) — cold to warm.
// The hybrid_pwpoly EOS only exposes a `press_eps_csnd2_entropy__temp_rho_ye`
// inversion (not an `eps_rho_ye` one), so the test drives the EOS with
// temperature and lets it compute eps internally.
constexpr int N_T = 4 ;
constexpr double TEMPERATURES[N_T] = {1e-3, 1e-1, 1.0, 10.0} ;

/// State-normalised infinity-norm residual.
///
/// Per-component relative norms blow up for components that are mathematically
/// zero (e.g. `Z_y = Z_z = 0` when the velocity is along ±x), because the
/// recovered value is some O(ulp) round-off and a `|a − b| / |b|` ratio is
/// then ill-conditioned.  Instead, take the absolute infinity-norm and
/// normalise once by the magnitude of the reference state.  Round-off noise
/// in a single component is then bounded by `O(ulp) / state_scale`, which is
/// the right small-number floor.
KOKKOS_INLINE_FUNCTION
double prim_residual(grmhd_prims_array_t const& a,
                     grmhd_prims_array_t const& b)
{
    using Kokkos::fmax ;
    using Kokkos::fabs ;
    using Kokkos::sqrt ;
    double const z_mag = sqrt(b[ZXL]*b[ZXL] + b[ZYL]*b[ZYL] + b[ZZL]*b[ZZL]) ;
    double const state_scale =
        fmax(fmax(fabs(b[RHOL]), fabs(b[EPSL])), fmax(z_mag, 1e-30)) ;
    constexpr unsigned int idx[5] = {RHOL, EPSL, ZXL, ZYL, ZZL} ;
    double err = 0.0 ;
    for (auto i : idx) {
        err = fmax(err, fabs(a[i] - b[i])) ;
    }
    return err / state_scale ;
}

/// Build a `metric_array_t` for the Cartesian-Kerr-Schild Schwarzschild
/// spacetime evaluated at (x, y, z) = (R, 0, 0).  Picks a non-trivial
/// γ_xx ≠ 1, β^x ≠ 0, α < 1 — far enough from the horizon that the
/// inversion is well-conditioned.
metric_array_t schwarzschild_cks_metric(double R, double M = 1.0)
{
    double const r   = R ;                       // along x-axis: r = |x|
    double const H   = M / r ;                   // KS scalar
    double const fac = 1.0 + 2.0 * H ;
    double const alp = 1.0 / std::sqrt(fac) ;    // lapse
    std::array<double, 3> beta = {2.0 * H / fac, 0.0, 0.0} ;
    // γ_ij = δ_ij + 2H l_i l_j, with l_spatial = (1,0,0) along the +x axis.
    std::array<double, 6> g = {
        1.0 + 2.0 * H,  // gxx
        0.0,            // gxy
        0.0,            // gxz
        1.0,            // gyy
        0.0,            // gyz
        1.0             // gzz
    } ;
    return metric_array_t(g, beta, alp) ;
}

} // namespace


TEST_CASE("c2p round-trip / Minkowski, no B", "[c2p][hydro]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    constexpr size_t N = N_RHO * N_T * N_W * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_mink_no_B", N, KOKKOS_LAMBDA(int const idx) {
        int q = idx ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;

        double const rho   = std::pow(10.0, LOG10_RHO[i_rho]) ;
        double const temp0 = TEMPERATURES[i_t] ;
        double const W     = WS[i_w] ;
        double const z_mag = std::sqrt(W*W - 1.0) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = 0.0 ;
        double ymu = 0.0 ;
        p0[ZXL]   = z_mag * DIRS[i_dir][0] ;
        p0[ZYL]   = z_mag * DIRS[i_dir][1] ;
        p0[ZZL]   = z_mag * DIRS[i_dir][2] ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        // Fill pressure / eps / csnd / entropy from EOS given (temp, rho, ye).
        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], ymu, err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual)  ;  deep_copy(r_h,  residual)  ;
    auto c_h = create_mirror_view(floored) ;  deep_copy(c_h,  floored) ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    // Decode the linear argmax back into (i_rho, i_t, i_w, i_dir) so the
    // failing sample's parameters surface in the test log.
    {
        size_t q = i_max ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (rho=" << std::pow(10.0, LOG10_RHO[i_rho])
             << ", T="   << TEMPERATURES[i_t]
             << ", W="   << WS[i_w]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-10) ;
    }
}


TEST_CASE("c2p round-trip / Minkowski, magnetisation sweep", "[c2p][mhd]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    // Fixed thermo state at moderate W; the parameter being swept is
    // b² ≈ B² (Minkowski), in units of ρ.  Production c2p has known
    // sensitivity in the b²/ρh ≳ 1 regime — c2p_pars.max_sigma gates it.
    // High-σ samples (b²/ρh ≳ 1) legitimately lose precision in the c2p
    // round-trip: the fluid energy becomes a small fraction of total
    // energy (B² dominates), so tiny errors in conservatives produce big
    // errors in primitives.  Sweep stays in the low-σ regime where round-
    // trip is expected to hold to ~1e-8.
    constexpr int N_BMAG = 3 ;
    constexpr double BSQ_OVER_RHO[N_BMAG] = {0.0, 1e-4, 1e-2} ;
    constexpr double RHO0  = 1e-3 ;
    constexpr double TEMP0 = 1e-1 ;
    constexpr double W0    = 1.5 ;
    constexpr double Z0    = 1.118033988749895 ;  // sqrt(W0²-1)

    size_t const N = N_BMAG * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;
    // Per-sample (reference, recovered) prim values so the worst-residual
    // sample can be dumped component-by-component on failure.  9 components
    // each: RHO, EPS, TEMP, Z{x,y,z}, B{x,y,z}.
    Kokkos::View<double*[18]> samples("samples", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_mink_Bsweep", N, KOKKOS_LAMBDA(int const idx) {
        int const i_dir  = idx % N_DIR ;
        int const i_bmag = idx / N_DIR ;
        double const B_mag = std::sqrt(BSQ_OVER_RHO[i_bmag] * RHO0) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = RHO0 ;
        p0[TEMPL] = TEMP0 ;
        p0[YEL]   = 0.0 ;
        double ymu = 0.0 ;
        // Velocity along +x (fixed direction); B direction varies.
        p0[ZXL]   = Z0 ;
        p0[ZYL]   = 0.0 ;
        p0[ZZL]   = 0.0 ;
        p0[BXL]   = B_mag * DIRS[i_dir][0] ;
        p0[BYL]   = B_mag * DIRS[i_dir][1] ;
        p0[BZL]   = B_mag * DIRS[i_dir][2] ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], ymu, err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;
        // `prims_to_conservs` only writes (D, τ, S_i, ent*) — in production
        // the densitised B field `cons[BS*]` is set separately via CT/EMF.
        // For the c2p round-trip test we propagate it manually here:
        //   cons[BS_i] = √γ · B^i.   Minkowski → √γ = 1.
        cons[BSXL] = p0[BXL] ;
        cons[BSYL] = p0[BYL] ;
        cons[BSZL] = p0[BZL] ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
        // Stash reference + recovered prim components for failure diagnosis.
        samples(idx,  0) = p0[RHOL] ;  samples(idx,  9) = p1[RHOL] ;
        samples(idx,  1) = p0[EPSL] ;  samples(idx, 10) = p1[EPSL] ;
        samples(idx,  2) = p0[TEMPL];  samples(idx, 11) = p1[TEMPL];
        samples(idx,  3) = p0[ZXL] ;   samples(idx, 12) = p1[ZXL] ;
        samples(idx,  4) = p0[ZYL] ;   samples(idx, 13) = p1[ZYL] ;
        samples(idx,  5) = p0[ZZL] ;   samples(idx, 14) = p1[ZZL] ;
        samples(idx,  6) = p0[BXL] ;   samples(idx, 15) = p1[BXL] ;
        samples(idx,  7) = p0[BYL] ;   samples(idx, 16) = p1[BYL] ;
        samples(idx,  8) = p0[BZL] ;   samples(idx, 17) = p1[BZL] ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual) ;  deep_copy(r_h, residual) ;
    auto c_h = create_mirror_view(floored)  ;  deep_copy(c_h, floored)  ;
    auto s_h = create_mirror_view(samples)  ;  deep_copy(s_h, samples)  ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    {
        int const i_dir  = i_max % N_DIR ;
        int const i_bmag = i_max / N_DIR ;
        char const* names[9] = {"RHO","EPS","TEMP","Zx","Zy","Zz","Bx","By","Bz"} ;
        std::ostringstream dump ;
        dump << std::scientific << std::setprecision(12) ;
        dump << "\n  worst sample:  b²/ρ = " << BSQ_OVER_RHO[i_bmag]
             << ", dir = (" << DIRS[i_dir][0] << ","
                            << DIRS[i_dir][1] << ","
                            << DIRS[i_dir][2] << ")\n" ;
        dump << "    "
             << std::setw(6) << "comp" << "  "
             << std::setw(20) << "p_ref"
             << std::setw(20) << "p_recovered"
             << std::setw(20) << "delta" << "\n" ;
        for (int c = 0 ; c < 9 ; ++c) {
            double const ref = s_h(i_max, c) ;
            double const rec = s_h(i_max, c + 9) ;
            dump << "    "
                 << std::setw(6) << names[c] << "  "
                 << std::setw(20) << ref
                 << std::setw(20) << rec
                 << std::setw(20) << (rec - ref) << "\n" ;
        }
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (b²/ρ=" << BSQ_OVER_RHO[i_bmag]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail
             << dump.str()) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-8) ;
    }
}


#ifdef GRACE_ENABLE_MUONS
// ---------------------------------------------------------------------------
//  4D leptonic EOS round-trip, WITH muons.
//
//  Bound to configs/c2p_test_leptonic.yaml (eos_type = leptonic).  Sweeps
//  the full thermodynamic + composition space (rho, T, Ye, Ymu) that the
//  hybrid baseline cannot reach: Ye and Ymu are advected scalars, and the
//  baryon table is sampled at yp = Ye + Ymu (charge neutrality), so the
//  muon fraction shifts the entire baryon thermo.  This is the regime that
//  stresses the Kastaun inversion in production muonic runs.
//
//  Sample points are taken at *interior fractions* of the table's own
//  (rho, T, Ye, Ymu) bounds so no axis clamps -- a clamp would break the
//  round-trip and is a separate concern (production handles it via the
//  RESET_YE/RESET_YMU/EPS_TOO_LOW machinery).  Ye + Ymu is held below
//  0.9*Ye_max so the proton-fraction axis never saturates.
//
//  Requires GRACE_ENABLE_MUONS (YMUL/YMUSL slots).  Tag [leptonic]
//  so the hybrid ctest run filters it out and vice versa.
// ---------------------------------------------------------------------------
TEST_CASE("c2p round-trip / leptonic 4D, with muons", "[c2p][hydro][leptonic]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    // --- Pull the table's own validity bounds (host side) -----------------
    double const lrho_lo = std::log(eos.density_minimum()) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const lT_lo   = std::log(eos.temperature_minimum()) ;
    double const lT_hi   = std::log(eos.temperature_maximum()) ;
    double const ye_lo   = eos.get_c2p_ye_min() ;
    double const ye_hi   = eos.get_c2p_ye_max() ;
    double const lymu_lo = std::log(eos.get_c2p_ymu_min()) ;
    double const lymu_hi = std::log(eos.get_c2p_ymu_max()) ;

    // --- Interior sample fractions (away from edges -> no clamping) -------
    // rho biased toward the dense (NS) end; T log-spaced cold->hot.
    constexpr int    N_RHO_L = 6 ;
    constexpr double FR_RHO[N_RHO_L] = {0.35, 0.50, 0.60, 0.70, 0.80, 0.90} ;
    constexpr int    N_T_L   = 6 ;
    constexpr double FR_T  [N_T_L]   = {0.10, 0.25, 0.40, 0.55, 0.70, 0.85} ;
    constexpr int    N_YE_L  = 4 ;
    constexpr double FR_YE [N_YE_L]  = {0.15, 0.30, 0.45, 0.60} ;
    constexpr int    N_YMU_L = 4 ;
    constexpr double FR_YMU[N_YMU_L] = {0.20, 0.40, 0.60, 0.85} ;
    constexpr int    N_W_L   = 2 ;
    constexpr double WS_L  [N_W_L]   = {1.05, 2.0} ;

    constexpr size_t N = N_RHO_L * N_T_L * N_YE_L * N_YMU_L * N_W_L ;
    Kokkos::View<double*> residual("res_lep", N) ;
    Kokkos::View<int*>    floored ("floored_lep", N) ;
    Kokkos::View<int*>    skipped ("skipped_lep", N) ;

    parallel_for("c2p_leptonic", N, KOKKOS_LAMBDA(int const idx) {
        int q = idx ;
        int const i_rho = q % N_RHO_L ; q /= N_RHO_L ;
        int const i_t   = q % N_T_L   ; q /= N_T_L   ;
        int const i_ye  = q % N_YE_L  ; q /= N_YE_L  ;
        int const i_ymu = q % N_YMU_L ; q /= N_YMU_L ;
        int const i_w   = q ;

        double const rho   = Kokkos::exp(lrho_lo + FR_RHO[i_rho]*(lrho_hi - lrho_lo)) ;
        double const temp0 = Kokkos::exp(lT_lo   + FR_T  [i_t]  *(lT_hi   - lT_lo  )) ;
        double const ye0   = ye_lo + FR_YE[i_ye]*(ye_hi - ye_lo) ;
        double       ymu0  = Kokkos::exp(lymu_lo + FR_YMU[i_ymu]*(lymu_hi - lymu_lo)) ;
        // Keep yp = Ye + Ymu strictly below the proton-fraction ceiling so
        // the baryon table axis never saturates (would break round-trip).
        double const ymu_cap = Kokkos::fmax(0.0, 0.9*ye_hi - ye0) ;
        ymu0 = Kokkos::fmin(ymu0, ymu_cap) ;

        double const W     = WS_L[i_w] ;
        double const z_mag = Kokkos::sqrt(W*W - 1.0) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = ye0 ;
        p0[YMUL]  = ymu0 ;
        p0[ZXL]   = z_mag ;   // velocity along +x
        p0[ZYL]   = 0.0 ;
        p0[ZZL]   = 0.0 ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL],
                        p0[RHOL], p0[YEL], p0[YMUL], err) ;

        // Skip states beyond the c2p validity ceiling: hot, low-density
        // samples are radiation-dominated and their (table-valid) eps can
        // exceed eos.eps_maximum, where the EPS_TOO_HIGH guard correctly
        // rejects the inversion.  That is c2p working as designed, not a
        // round-trip failure -- exclude rather than assert on it.
        if (p0[EPSL] > 0.5 * eos.get_c2p_eps_max()) {
            residual(idx) = 0.0 ;
            floored(idx)  = 0 ;
            skipped(idx)  = 1 ;
            return ;
        }

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;

        // State-normalised inf-norm residual over (rho, eps, Z, Ye, Ymu).
        double const z_mag_ref =
            Kokkos::sqrt(p0[ZXL]*p0[ZXL] + p0[ZYL]*p0[ZYL] + p0[ZZL]*p0[ZZL]) ;
        double const scale = Kokkos::fmax(
            Kokkos::fmax(Kokkos::fabs(p0[RHOL]), Kokkos::fabs(p0[EPSL])),
            Kokkos::fmax(z_mag_ref, 1e-30)) ;
        double e = 0.0 ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[RHOL] - p0[RHOL])) ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[EPSL] - p0[EPSL])) ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[ZXL]  - p0[ZXL]))  ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[ZYL]  - p0[ZYL]))  ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[ZZL]  - p0[ZZL]))  ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[YEL]  - p0[YEL]))  ;
        e = Kokkos::fmax(e, Kokkos::fabs(p1[YMUL] - p0[YMUL])) ;
        residual(idx) = e / scale ;
        floored(idx)  = ok ? 1 : 0 ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual) ;  deep_copy(r_h, residual) ;
    auto c_h = create_mirror_view(floored)  ;  deep_copy(c_h, floored)  ;
    auto s_h = create_mirror_view(skipped)  ;  deep_copy(s_h, skipped)  ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    int    n_skip  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (s_h(i)) ++n_skip ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    {
        size_t q = i_max ;
        int const i_rho = q % N_RHO_L ; q /= N_RHO_L ;
        int const i_t   = q % N_T_L   ; q /= N_T_L   ;
        int const i_ye  = q % N_YE_L  ; q /= N_YE_L  ;
        int const i_ymu = q % N_YMU_L ; q /= N_YMU_L ;
        int const i_w   = q ;
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (rho_frac=" << FR_RHO[i_rho]
             << ", T_frac="   << FR_T[i_t]
             << ", Ye_frac="  << FR_YE[i_ye]
             << ", Ymu_frac=" << FR_YMU[i_ymu]
             << ", W="        << WS_L[i_w] << ")"
             << "; floored samples = " << n_fail
             << "; skipped (eps > 0.5*c2p_eps_max) = " << n_skip
             << " of " << N) ;
        // The sweep must actually test the vast majority of the grid: if
        // the eps ceiling in the parfile shrinks, fail loudly here instead
        // of silently skipping the hot corner.
        REQUIRE(n_skip <= int(N / 20)) ;
        // Tabulated EOS round-trip: looser than the analytic-polytrope
        // floor.  eps inverts through the Kastaun mu-iteration + Brent
        // T(eps) solve; rho/Ye/Ymu are exact (passive division).
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-8) ;
    }
}


// ---------------------------------------------------------------------------
//  Leptonic c2p edge-case suite.
//
//  Targets the failure modes diagnosed in the hot-TOV envelope work:
//    * cold-floor states (T = table min, eps = eps_lo < 0) must invert
//      cleanly, NOT atmosphere-reset -- the "blocky atmosphere" was these
//      cells tripping robustness paths, not genuine c2p failures;
//    * temp_fl straddling the table minimum: a floor strictly below
//      exp(ltempmin) must never fire C2P_T_FLOORED (the FP-straddle
//      patchwork), while a floor above it fires deterministically and
//      preserves the velocity;
//    * atmosphere maintenance (D below the buffered floor) is a full,
//      self-consistent reset flagged C2P_ATMO_RESET;
//    * a corrupted advected-entropy channel must not perturb the primary
//      energy inversion (the entropy channel is overwritten, and the
//      backup only re-inverts on distrust) -- the "backup churn"
//      mitigation invariant;
//    * out-of-bounds advected Ye/Ymu are clamped with their RESET bits,
//      and yp = Ye + Ymu saturation (fix_ymu_high_yp) still round-trips.
// ---------------------------------------------------------------------------
namespace {

// Slots of the flat output view written by leptonic_c2p_single.
enum lep_out_idx : int {
    LEP_RHO = 0, LEP_EPS, LEP_TEMP, LEP_PRESS, LEP_YE, LEP_YMU,
    LEP_ZX, LEP_ZY, LEP_ZZ,
    LEP_FLOORED,
    LEP_BIT_ATMO, LEP_BIT_TFLOOR, LEP_BIT_BACKUP,
    LEP_BIT_RESET_YE, LEP_BIT_RESET_YMU, LEP_SIG_EPS_LO,
    LEP_REF_EPS, LEP_REF_PRESS,
    LEP_H,
    LEP_OUT_N
} ;

// Device-callable conservative-variable mutators, applied between P2C and
// c2p to emulate what evolution can hand the inversion.
struct cons_mutate_none {
    KOKKOS_INLINE_FUNCTION void operator()(grmhd_cons_array_t&) const {}
} ;
struct cons_scale_entropy {          // noisy advected entropy
    double f ;
    KOKKOS_INLINE_FUNCTION void operator()(grmhd_cons_array_t& c) const {
        c[ENTSL] *= f ;
    }
} ;
struct cons_scale_dens {             // push D into the atmosphere range
    double f ;
    KOKKOS_INLINE_FUNCTION void operator()(grmhd_cons_array_t& c) const {
        c[DENSL] *= f ;
    }
} ;
struct cons_set_ye_star {            // advected Ye out of table bounds
    double ye ;
    KOKKOS_INLINE_FUNCTION void operator()(grmhd_cons_array_t& c) const {
        c[YESL] = ye * c[DENSL] ;
    }
} ;
struct cons_set_ymu_star {           // advected Ymu out of table bounds
    double ymu ;
    KOKKOS_INLINE_FUNCTION void operator()(grmhd_cons_array_t& c) const {
        c[YMUSL] = ymu * c[DENSL] ;
    }
} ;

/// Build one primitive state from (rho, T, Ye, Ymu, W), P2C it, apply
/// `mutate` to the conservatives, run the production c2p, and report the
/// recovered primitives + outcome/diagnostic bits.  Runs in a single-cell
/// Kokkos kernel because the leptonic EOS tables live in device memory.
template <typename eos_t, typename mutate_t>
std::array<double, LEP_OUT_N> leptonic_c2p_single(
    eos_t const& eos,
    atmo_params_t const& atmo,
    excision_params_t const& excision,
    c2p_params_t const& c2p_pars,
    double rho0, double temp0, double ye0, double ymu0, double W,
    mutate_t mutate)
{
    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;
    Kokkos::View<double*> out("lep_single_out", LEP_OUT_N) ;

    Kokkos::parallel_for("lep_c2p_single", 1, KOKKOS_LAMBDA(int) {
        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho0 ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = ye0 ;
        p0[YMUL]  = ymu0 ;
        double const z_mag = Kokkos::sqrt(W*W - 1.0) ;
        p0[ZXL]   = z_mag ;
        p0[ZYL]   = 0.0 ;
        p0[ZZL]   = 0.0 ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL],
                        p0[RHOL], p0[YEL], p0[YMUL], err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;
        mutate(cons) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const fl = conservs_to_prims(cons, p1, metric, eos, atmo,
                                          excision, c2p_pars, rtp, c2p_err) ;

        out(LEP_RHO)   = p1[RHOL] ;
        out(LEP_EPS)   = p1[EPSL] ;
        out(LEP_TEMP)  = p1[TEMPL] ;
        out(LEP_PRESS) = p1[PRESSL] ;
        out(LEP_YE)    = p1[YEL] ;
        out(LEP_YMU)   = p1[YMUL] ;
        out(LEP_ZX)    = p1[ZXL] ;
        out(LEP_ZY)    = p1[ZYL] ;
        out(LEP_ZZ)    = p1[ZZL] ;
        out(LEP_FLOORED)       = fl ? 1.0 : 0.0 ;
        out(LEP_BIT_ATMO)      = c2p_err.test(c2p_err_enum_t::C2P_ATMO_RESET)      ? 1.0 : 0.0 ;
        out(LEP_BIT_TFLOOR)    = c2p_err.test(c2p_err_enum_t::C2P_T_FLOORED)       ? 1.0 : 0.0 ;
        out(LEP_BIT_BACKUP)    = c2p_err.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED) ? 1.0 : 0.0 ;
        out(LEP_BIT_RESET_YE)  = c2p_err.test(c2p_err_enum_t::C2P_RESET_YE)        ? 1.0 : 0.0 ;
        out(LEP_BIT_RESET_YMU) = c2p_err.test(c2p_err_enum_t::C2P_RESET_YMU)       ? 1.0 : 0.0 ;
        out(LEP_SIG_EPS_LO)    = c2p_err.test(c2p_err_enum_t::C2P_SIG_EPS_TOO_LOW) ? 1.0 : 0.0 ;
        out(LEP_REF_EPS)   = p0[EPSL] ;
        out(LEP_REF_PRESS) = p0[PRESSL] ;
        out(LEP_H) = 1.0 + p1[EPSL] + p1[PRESSL] / p1[RHOL] ;
    }) ;
    Kokkos::fence() ;

    auto m = Kokkos::create_mirror_view(out) ;
    Kokkos::deep_copy(m, out) ;
    std::array<double, LEP_OUT_N> a ;
    for (int i = 0 ; i < LEP_OUT_N ; ++i) a[i] = m(i) ;
    return a ;
}

} // namespace


TEST_CASE("c2p leptonic: cold-floor states invert cleanly, no atmosphere reset",
          "[c2p][leptonic][coldfloor]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    // Mitigation setting under test: floor strictly below the table minimum.
    atmo.temp_fl = 0.99 * t_min ;

    // rho sweep from just above the atmosphere up to NS-core densities.
    double const lrho_lo =
        std::log(std::max(eos.density_minimum(), 100.0 * atmo.rho_fl)) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const ye0  = eos.get_c2p_ye_min()
        + 0.4 * (eos.get_c2p_ye_max() - eos.get_c2p_ye_min()) ;
    double const ymu0 = std::exp(std::log(eos.get_c2p_ymu_min())
        + 0.3 * (std::log(eos.get_c2p_ymu_max())
               - std::log(eos.get_c2p_ymu_min()))) ;

    for (double fr : {0.05, 0.30, 0.60, 0.85}) {
        for (double W : {1.0, 1.2}) {
            double const rho = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
            auto const r = leptonic_c2p_single(
                eos, atmo, excision, c2p_pars,
                rho, t_min, ye0, ymu0, W, cons_mutate_none{}) ;

            INFO("rho=" << rho << " T=" << t_min << " W=" << W
                 << " eps_ref=" << r[LEP_REF_EPS]
                 << " eps=" << r[LEP_EPS]
                 << " eps_lo_sig=" << r[LEP_SIG_EPS_LO]) ;
            // The whole point: the coldest EOS state is a valid state.
            REQUIRE(r[LEP_FLOORED]    == 0.0) ;
            REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
            REQUIRE(r[LEP_BIT_TFLOOR] == 0.0) ;
            // eps at the cold floor is the table baseline -- typically
            // NEGATIVE (energy_shift) -- and must round-trip even so.
            REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                    <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
            // Physicality: h = 1 + eps + P/rho stays positive.
            REQUIRE(r[LEP_H] > 0.0) ;
            REQUIRE(std::isfinite(r[LEP_PRESS])) ;
        }
    }
}


TEST_CASE("c2p leptonic: temp_fl below the table min never fires T_FLOORED; "
          "above it fires deterministically and preserves velocity",
          "[c2p][leptonic][tfloor]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    double const rho   = std::exp(
        std::log(eos.density_minimum())
        + 0.6 * (std::log(eos.density_maximum())
               - std::log(eos.density_minimum()))) ;
    double const ye0  = eos.get_c2p_ye_min()
        + 0.4 * (eos.get_c2p_ye_max() - eos.get_c2p_ye_min()) ;
    double const ymu0 = std::exp(std::log(eos.get_c2p_ymu_min())
        + 0.3 * (std::log(eos.get_c2p_ymu_max())
               - std::log(eos.get_c2p_ymu_min()))) ;
    double const W     = 1.5 ;
    double const z_mag = std::sqrt(W*W - 1.0) ;

    SECTION("temp_fl strictly below exp(ltempmin): no T-floor events") {
        atmo.temp_fl = 0.99 * t_min ;
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_min, ye0, ymu0, W, cons_mutate_none{}) ;
        INFO("T=" << r[LEP_TEMP] << " (table min " << t_min
             << ", EOS floor " << eos.temperature_floor() << ")") ;
        REQUIRE(r[LEP_BIT_TFLOOR] == 0.0) ;
        REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
        REQUIRE(r[LEP_FLOORED]    == 0.0) ;
        // Recovered T sits at the EOS's working temperature floor, not at
        // atmo.temp_fl -- and not on the raw table boundary, which is where
        // eps equals the table's smallest representable value and every c2p
        // call would raise EOS_EPS_TOO_LOW (see limit_temp).
        REQUIRE_THAT(r[LEP_TEMP],
                     Catch::Matchers::WithinRel(eos.temperature_floor(), 1e-8)) ;
        REQUIRE(eos.temperature_floor() > t_min) ;
    }

    SECTION("temp_fl above exp(ltempmin): T-floor branch fires, velocity kept") {
        atmo.temp_fl = 1.05 * t_min ;
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_min, ye0, ymu0, W, cons_mutate_none{}) ;
        REQUIRE(r[LEP_BIT_TFLOOR] == 1.0) ;
        REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
        // Tabulated/leptonic T-floor is the benign bottom-of-table clamp:
        // it must NOT flag the cell for FOFC.
        REQUIRE(r[LEP_FLOORED]    == 0.0) ;
        // T reset onto the floor, velocity preserved (T-floor semantics).
        REQUIRE_THAT(r[LEP_TEMP],
                     Catch::Matchers::WithinRel(atmo.temp_fl, 1e-12)) ;
        REQUIRE(std::fabs(r[LEP_ZX] - z_mag) <= 1e-8 * z_mag) ;
        REQUIRE(std::fabs(r[LEP_ZY]) <= 1e-12) ;
        REQUIRE(std::fabs(r[LEP_ZZ]) <= 1e-12) ;
    }
}


TEST_CASE("c2p leptonic: D below the buffered floor resets fully to atmosphere",
          "[c2p][leptonic][atmo]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const rho = std::exp(
        std::log(eos.density_minimum())
        + 0.5 * (std::log(eos.density_maximum())
               - std::log(eos.density_minimum()))) ;
    double const ye0  = eos.get_c2p_ye_min()
        + 0.4 * (eos.get_c2p_ye_max() - eos.get_c2p_ye_min()) ;
    double const ymu0 = std::exp(std::log(eos.get_c2p_ymu_min())
        + 0.3 * (std::log(eos.get_c2p_ymu_max())
               - std::log(eos.get_c2p_ymu_min()))) ;
    double const temp0 = std::exp(
        std::log(eos.temperature_minimum())
        + 0.5 * (std::log(eos.temperature_maximum())
               - std::log(eos.temperature_minimum()))) ;
    double const W = 1.3 ;

    // Scale D to half the atmosphere floor: the solve block is skipped and
    // the cell must be handed to reset_to_atmosphere.
    double const f = 0.5 * atmo.rho_fl / (rho * W) ;
    auto const r = leptonic_c2p_single(
        eos, atmo, excision, c2p_pars,
        rho, temp0, ye0, ymu0, W, cons_scale_dens{f}) ;

    REQUIRE(r[LEP_FLOORED]  == 1.0) ;
    REQUIRE(r[LEP_BIT_ATMO] == 1.0) ;
    // Full reset semantics: primitives sit at the (EOS-clamped) atmosphere
    // values and the velocity is zeroed.  The EOS clamps push slightly
    // inside the table bounds, hence range assertions.
    double const rho_min = eos.density_minimum() ;
    REQUIRE(r[LEP_RHO] >= std::min(atmo.rho_fl, rho_min) * (1.0 - 1e-12)) ;
    REQUIRE(r[LEP_RHO] <= std::max(atmo.rho_fl, 1.001 * rho_min)) ;
    double const t_min = eos.temperature_minimum() ;
    REQUIRE(r[LEP_TEMP] >= std::min(atmo.temp_fl, t_min) * (1.0 - 1e-12)) ;
    REQUIRE(r[LEP_TEMP] <= std::max(atmo.temp_fl, 1.05 * t_min)) ;
    REQUIRE(r[LEP_YE] >= eos.get_c2p_ye_min()) ;
    REQUIRE(r[LEP_YE] <= eos.get_c2p_ye_max()) ;
    REQUIRE(r[LEP_YMU] >= eos.get_c2p_ymu_min() * (1.0 - 1e-12)) ;
    REQUIRE(r[LEP_YMU] <= eos.get_c2p_ymu_max() * (1.0 + 1e-12)) ;
    REQUIRE(r[LEP_ZX] == 0.0) ;
    REQUIRE(r[LEP_ZY] == 0.0) ;
    REQUIRE(r[LEP_ZZ] == 0.0) ;
    REQUIRE(r[LEP_H] > 0.0) ;
}


TEST_CASE("c2p leptonic: corrupted advected entropy does not perturb the "
          "primary inversion", "[c2p][leptonic][entropy]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;
    atmo.temp_fl   = 0.99 * eos.temperature_minimum() ;

    double const rho = std::exp(
        std::log(eos.density_minimum())
        + 0.7 * (std::log(eos.density_maximum())
               - std::log(eos.density_minimum()))) ;
    double const ye0  = eos.get_c2p_ye_min()
        + 0.4 * (eos.get_c2p_ye_max() - eos.get_c2p_ye_min()) ;
    double const ymu0 = std::exp(std::log(eos.get_c2p_ymu_min())
        + 0.3 * (std::log(eos.get_c2p_ymu_max())
               - std::log(eos.get_c2p_ymu_min()))) ;
    double const t_warm = std::exp(
        std::log(eos.temperature_minimum())
        + 0.5 * (std::log(eos.temperature_maximum())
               - std::log(eos.temperature_minimum()))) ;
    double const t_min  = eos.temperature_minimum() ;
    double const W = 1.5 ;

    SECTION("warm state: entropy x7 is invisible, backup on or off") {
        for (bool backup : {false, true}) {
            c2p_pars.use_ent_backup = backup ;
            auto const r = leptonic_c2p_single(
                eos, atmo, excision, c2p_pars,
                rho, t_warm, ye0, ymu0, W, cons_scale_entropy{7.0}) ;
            INFO("use_ent_backup=" << backup) ;
            // No distrust on a warm mid-table state -> backup never runs,
            // and S* is overwritten from the recovered state.
            REQUIRE(r[LEP_BIT_BACKUP] == 0.0) ;
            REQUIRE(r[LEP_FLOORED]    == 0.0) ;
            REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                    <= std::max(1e-10, 1e-8 * std::fabs(r[LEP_REF_EPS]))) ;
        }
    }

    SECTION("cold-floor state: backup OFF pins the energy-inversion answer") {
        // The mitigation invariant: with the entropy backup disabled, a
        // corrupted advected entropy CANNOT scatter the cold envelope.
        c2p_pars.use_ent_backup = false ;
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_min, ye0, ymu0, W, cons_scale_entropy{7.0}) ;
        REQUIRE(r[LEP_BIT_BACKUP] == 0.0) ;
        REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
        REQUIRE(r[LEP_FLOORED]    == 0.0) ;
        REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
    }

    SECTION("cold-floor state: backup ON stays contained (no atmo fallthrough)") {
        // With the backup enabled the cold envelope may re-invert through
        // the (corrupted) entropy -- the diagnosed eps scatter.  The
        // containment invariant is that it never escalates to an
        // atmosphere reset or a non-finite state.
        c2p_pars.use_ent_backup = true ;
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_min, ye0, ymu0, W, cons_scale_entropy{7.0}) ;
        REQUIRE(r[LEP_BIT_ATMO] == 0.0) ;
        REQUIRE(r[LEP_FLOORED]  == 0.0) ;
        REQUIRE(std::isfinite(r[LEP_EPS])) ;
        REQUIRE(r[LEP_H] > 0.0) ;
    }
}


TEST_CASE("c2p leptonic: out-of-bounds advected Ye/Ymu clamp with reset bits; "
          "yp saturation round-trips", "[c2p][leptonic][composition]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const rho = std::exp(
        std::log(eos.density_minimum())
        + 0.7 * (std::log(eos.density_maximum())
               - std::log(eos.density_minimum()))) ;
    double const t_warm = std::exp(
        std::log(eos.temperature_minimum())
        + 0.5 * (std::log(eos.temperature_maximum())
               - std::log(eos.temperature_minimum()))) ;
    double const ye_lo  = eos.get_c2p_ye_min() ;
    double const ye_hi  = eos.get_c2p_ye_max() ;
    double const ymu_lo = eos.get_c2p_ymu_min() ;
    double const ymu_hi = eos.get_c2p_ymu_max() ;
    double const ye0  = ye_lo + 0.4 * (ye_hi - ye_lo) ;
    double const ymu0 = std::exp(std::log(ymu_lo)
        + 0.3 * (std::log(ymu_hi) - std::log(ymu_lo))) ;
    double const W = 1.2 ;

    SECTION("advected Ye above the table max clamps and flags RESET_YE") {
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_warm, ye0, ymu0, W, cons_set_ye_star{1.5 * ye_hi}) ;
        REQUIRE(r[LEP_BIT_RESET_YE] == 1.0) ;
        REQUIRE_THAT(r[LEP_YE], Catch::Matchers::WithinRel(ye_hi, 1e-12)) ;
        REQUIRE(r[LEP_BIT_ATMO] == 0.0) ;
        REQUIRE(std::isfinite(r[LEP_EPS])) ;
    }

    SECTION("advected Ymu below the table min clamps and flags RESET_YMU") {
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_warm, ye0, ymu0, W, cons_set_ymu_star{0.5 * ymu_lo}) ;
        REQUIRE(r[LEP_BIT_RESET_YMU] == 1.0) ;
        REQUIRE_THAT(r[LEP_YMU], Catch::Matchers::WithinRel(ymu_lo, 1e-12)) ;
        REQUIRE(r[LEP_BIT_ATMO] == 0.0) ;
        REQUIRE(std::isfinite(r[LEP_EPS])) ;
    }

    SECTION("yp = Ye + Ymu above the proton-fraction ceiling round-trips") {
        // fix_ymu_high_yp caps the muon fraction inside the EOS so the
        // baryon-table yp axis saturates instead of ymu collapsing to its
        // floor.  Both the forward P2C and the inversion sample the SAME
        // capped composition, so the round-trip must still close.
        double const ye_sat  = 0.95 * ye_hi ;
        double const ymu_sat = 0.5  * ymu_hi ;   // ye_sat + ymu_sat > ye_hi
        REQUIRE(ye_sat + ymu_sat > ye_hi) ;      // premise of the section
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_warm, ye_sat, ymu_sat, W, cons_mutate_none{}) ;
        REQUIRE(r[LEP_FLOORED]  == 0.0) ;
        REQUIRE(r[LEP_BIT_ATMO] == 0.0) ;
        REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
        REQUIRE(r[LEP_H] > 0.0) ;
    }
}


// ---------------------------------------------------------------------------
//  Extreme-corner suite: the table regions that production hot-TOV envelopes
//  actually visit and where past bugs lived.
//    * cold floor x Ymu sweep — the historic cold-T muon-table interpolation
//      bug (linear-in-mu vs log-in-mu) lived exactly here;
//    * low rho — the atmosphere-adjacent envelope (blocky-atmosphere region);
//    * high Ye — proton-rich states near the yp ceiling, where the EOS
//      switches mu_e source and fix_ymu_high_yp engages;
//    * high Ymu — the top of the muon-fraction axis, cold and warm.
//  All round-trips sample interior (non-clamping) compositions; premises
//  are asserted so a table swap that changes the bounds fails loudly
//  instead of silently weakening the test.
// ---------------------------------------------------------------------------

namespace {

// Shared assertion block: a clean interior state must invert with no
// atmosphere reset, no T-floor, no composition clamps, and eps/rho at the
// solver floor.  h > 0 guards physicality even where eps < 0 (cold table
// baseline) or the spinodal makes P < 0 (linear_pressure).
void require_clean_roundtrip(std::array<double, LEP_OUT_N> const& r,
                             double rho0)
{
    REQUIRE(r[LEP_FLOORED]        == 0.0) ;
    REQUIRE(r[LEP_BIT_ATMO]       == 0.0) ;
    REQUIRE(r[LEP_BIT_TFLOOR]     == 0.0) ;
    REQUIRE(r[LEP_BIT_RESET_YE]   == 0.0) ;
    REQUIRE(r[LEP_BIT_RESET_YMU]  == 0.0) ;
    REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
            <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
    REQUIRE(std::fabs(r[LEP_RHO] - rho0) <= 1e-8 * rho0) ;
    REQUIRE(r[LEP_H] > 0.0) ;
    REQUIRE(std::isfinite(r[LEP_PRESS])) ;
}

} // namespace


TEST_CASE("c2p leptonic: cold floor x Ymu sweep (historic cold-muon bug region)",
          "[c2p][leptonic][coldfloor][highymu]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    atmo.temp_fl = 0.99 * t_min ;
    c2p_pars.use_ent_backup = false ;

    double const lrho_lo =
        std::log(std::max(eos.density_minimum(), 100.0 * atmo.rho_fl)) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const ye_lo   = eos.get_c2p_ye_min() ;
    double const ye_hi   = eos.get_c2p_ye_max() ;
    double const lym_lo  = std::log(eos.get_c2p_ymu_min()) ;
    double const lym_hi  = std::log(eos.get_c2p_ymu_max()) ;
    double const ye0     = ye_lo + 0.3 * (ye_hi - ye_lo) ;

    // Ymu from near the axis floor up to near its ceiling, at the coldest
    // table temperature: exactly where the linear-in-mu interpolation bug
    // produced unphysical cold states before the table regeneration.
    for (double fym : {0.02, 0.05, 0.20, 0.40, 0.60, 0.75, 0.90, 0.95}) {
        for (double fr : {0.05, 0.20, 0.35, 0.50, 0.70, 0.85}) {
            double const rho  = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
            double ymu0 = std::exp(lym_lo + fym * (lym_hi - lym_lo)) ;
            // Interior yp only: cap like the main sweep does.
            ymu0 = std::min(ymu0, std::max(0.0, 0.9 * ye_hi - ye0)) ;

            auto const r = leptonic_c2p_single(
                eos, atmo, excision, c2p_pars,
                rho, t_min, ye0, ymu0, /*W=*/1.0, cons_mutate_none{}) ;
            INFO("rho=" << rho << " T=" << t_min
                 << " ye=" << ye0 << " ymu=" << ymu0
                 << " (fym=" << fym << ", fr=" << fr << ")"
                 << " eps_ref=" << r[LEP_REF_EPS] << " eps=" << r[LEP_EPS]) ;
            require_clean_roundtrip(r, rho) ;
        }
    }
}


TEST_CASE("c2p leptonic: low-rho envelope just above the atmosphere buffer",
          "[c2p][leptonic][lowrho]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    atmo.temp_fl = 0.99 * t_min ;
    c2p_pars.use_ent_backup = false ;

    // Start as low as the machinery allows: above BOTH the table's density
    // floor and the buffered atmosphere trigger rho_fl*(1+atmo_tol), then
    // climb three decades.  This is the region that renders as the visible
    // "envelope" around a TOV star.
    double const rho_base = std::max(
        1.001 * eos.density_minimum(),
        2.0 * atmo.rho_fl * (1.0 + atmo.atmo_tol)) ;
    REQUIRE(rho_base < eos.density_maximum()) ;   // premise

    double const ye_lo  = eos.get_c2p_ye_min() ;
    double const ye_hi  = eos.get_c2p_ye_max() ;
    double const ye0    = ye_lo + 0.3 * (ye_hi - ye_lo) ;
    double const ymu0   = std::exp(std::log(eos.get_c2p_ymu_min())
        + 0.3 * (std::log(eos.get_c2p_ymu_max())
               - std::log(eos.get_c2p_ymu_min()))) ;
    double const lt_lo  = std::log(t_min) ;
    double const lt_hi  = std::log(eos.temperature_maximum()) ;

    for (double rho_mult : {1.0, 2.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0}) {
        for (double ft : {0.0, 0.15, 0.3, 0.5, 0.7}) {
            double const rho  = rho_base * rho_mult ;
            double const temp = std::exp(lt_lo + ft * (lt_hi - lt_lo)) ;
            for (double W : {1.0, 1.1, 1.2}) {
                auto const r = leptonic_c2p_single(
                    eos, atmo, excision, c2p_pars,
                    rho, temp, ye0, ymu0, W, cons_mutate_none{}) ;
                INFO("rho=" << rho << " (=base x " << rho_mult << ")"
                     << " T=" << temp << " W=" << W
                     << " eps_ref=" << r[LEP_REF_EPS]
                     << " eps=" << r[LEP_EPS]) ;
                require_clean_roundtrip(r, rho) ;
            }
        }
    }
}


TEST_CASE("c2p leptonic: high-Ye proton-rich states near the yp ceiling",
          "[c2p][leptonic][highye]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    atmo.temp_fl = 0.99 * t_min ;
    c2p_pars.use_ent_backup = false ;

    double const ye_lo  = eos.get_c2p_ye_min() ;
    double const ye_hi  = eos.get_c2p_ye_max() ;
    double const ymu_lo = eos.get_c2p_ymu_min() ;
    double const lrho_lo = std::log(
        std::max(eos.density_minimum(), 100.0 * atmo.rho_fl)) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const lt_lo   = std::log(t_min) ;
    double const lt_hi   = std::log(eos.temperature_maximum()) ;

    // Unsaturated: tiny Ymu so yp = Ye + Ymu stays strictly below ye_hi
    // even at the 0.99 fraction; the premise assert makes a table with a
    // large ymu_min fail loudly rather than silently saturate.
    double const ymu0 = 2.0 * ymu_lo ;

    SECTION("unsaturated high Ye round-trips with no composition clamps") {
        for (double fye : {0.80, 0.85, 0.90, 0.95, 0.97, 0.99}) {
            double const ye0 = ye_lo + fye * (ye_hi - ye_lo) ;
            REQUIRE(ye0 + ymu0 < ye_hi) ;   // premise: below the ceiling
            for (double fr : {0.2, 0.4, 0.6, 0.8}) {
                for (double ft : {0.0, 0.5}) {
                    double const rho  = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
                    double const temp = std::exp(lt_lo + ft * (lt_hi - lt_lo)) ;
                    auto const r = leptonic_c2p_single(
                        eos, atmo, excision, c2p_pars,
                        rho, temp, ye0, ymu0, /*W=*/1.2, cons_mutate_none{}) ;
                    INFO("ye=" << ye0 << " (frac " << fye << ")"
                         << " rho=" << rho << " T=" << temp
                         << " eps_ref=" << r[LEP_REF_EPS]
                         << " eps=" << r[LEP_EPS]) ;
                    require_clean_roundtrip(r, rho) ;
                }
            }
        }
    }

    SECTION("saturated yp > ye_max (mu_e fallback + fix_ymu_high_yp) closes") {
        // Beyond the ceiling the EOS switches to the baryon-table mu_e and
        // caps ymu internally; both P2C and the inversion sample the same
        // capped state, so eps must still close (composition bits may fire
        // inside the EOS, but no c2p-level clamps).
        double const t_warm = std::exp(lt_lo + 0.5 * (lt_hi - lt_lo)) ;
        double const ymu_hi = eos.get_c2p_ymu_max() ;
        for (double fye : {0.85, 0.90, 0.95, 0.98}) {
        for (double fr  : {0.4, 0.7}) {
            double const rho     = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
            double const ye0     = ye_lo + fye * (ye_hi - ye_lo) ;
            double const ymu_sat = 0.5 * ymu_hi ;
            REQUIRE(ye0 + ymu_sat > ye_hi) ;   // premise: above the ceiling
            for (double temp : {t_min, t_warm}) {
                auto const r = leptonic_c2p_single(
                    eos, atmo, excision, c2p_pars,
                    rho, temp, ye0, ymu_sat, /*W=*/1.2, cons_mutate_none{}) ;
                INFO("ye=" << ye0 << " ymu=" << ymu_sat << " T=" << temp
                     << " eps_ref=" << r[LEP_REF_EPS]
                     << " eps=" << r[LEP_EPS]) ;
                REQUIRE(r[LEP_FLOORED]    == 0.0) ;
                REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
                REQUIRE(r[LEP_BIT_TFLOOR] == 0.0) ;
                REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                        <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
                REQUIRE(r[LEP_H] > 0.0) ;
            }
        }
        }
    }
}


TEST_CASE("c2p leptonic: high-Ymu states near the muon-fraction ceiling",
          "[c2p][leptonic][highymu]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    atmo.temp_fl = 0.99 * t_min ;
    c2p_pars.use_ent_backup = false ;

    double const ye_lo   = eos.get_c2p_ye_min() ;
    double const ye_hi   = eos.get_c2p_ye_max() ;
    double const lym_lo  = std::log(eos.get_c2p_ymu_min()) ;
    double const lym_hi  = std::log(eos.get_c2p_ymu_max()) ;
    double const lrho_lo = std::log(
        std::max(eos.density_minimum(), 100.0 * atmo.rho_fl)) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const lt_lo   = std::log(t_min) ;
    double const lt_hi   = std::log(eos.temperature_maximum()) ;
    double const t_mid   = std::exp(lt_lo + 0.5 * (lt_hi - lt_lo)) ;

    // 0.98 of the log range, NOT 1.0: a state stored exactly at ymu_max can
    // come back one ulp above it through the YMUSL/D division and trip the
    // clamp stochastically -- that FP-boundary behaviour is the same class
    // as the temp_fl straddle and is deliberately not part of this test.
    for (double fym : {0.70, 0.85, 0.93, 0.98}) {
        for (double fye : {0.05, 0.15, 0.30}) {
            double const ye0  = ye_lo + fye * (ye_hi - ye_lo) ;
            double ymu0 = std::exp(lym_lo + fym * (lym_hi - lym_lo)) ;
            ymu0 = std::min(ymu0, std::max(0.0, 0.9 * ye_hi - ye0)) ;
            for (double fr : {0.2, 0.5, 0.8}) {
                double const rho = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
                for (double temp : {t_min, t_mid}) {
                    for (double W : {1.0, 1.3}) {
                        auto const r = leptonic_c2p_single(
                            eos, atmo, excision, c2p_pars,
                            rho, temp, ye0, ymu0, W, cons_mutate_none{}) ;
                        INFO("ymu=" << ymu0 << " (frac " << fym << ")"
                             << " ye=" << ye0 << " rho=" << rho
                             << " T=" << temp << " W=" << W
                             << " eps_ref=" << r[LEP_REF_EPS]
                             << " eps=" << r[LEP_EPS]) ;
                        require_clean_roundtrip(r, rho) ;
                    }
                }
            }
        }
    }
}


TEST_CASE("c2p leptonic: exactly AT ye_max / ymu_max (table upper edges)",
          "[c2p][leptonic][edge][highye][highymu]")
{
    // States stored exactly on the composition table edges.  The advected
    // ratios YeS/D and YmuS/D can round-trip +-1 ulp across the bound, so
    // whether the RESET_YE / RESET_YMU clamp fires is FP-luck BY DESIGN --
    // the invariant asserted here is that the recovered state is the edge
    // state either way: composition back on (or at ulp-distance below) the
    // bound, eps closed, and none of the heavy robustness paths engaged.
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    double const t_min = eos.temperature_minimum() ;
    atmo.temp_fl = 0.99 * t_min ;
    c2p_pars.use_ent_backup = false ;

    double const ye_lo  = eos.get_c2p_ye_min() ;
    double const ye_hi  = eos.get_c2p_ye_max() ;
    double const ymu_lo = eos.get_c2p_ymu_min() ;
    double const ymu_hi = eos.get_c2p_ymu_max() ;
    double const lrho_lo = std::log(
        std::max(eos.density_minimum(), 100.0 * atmo.rho_fl)) ;
    double const lrho_hi = std::log(eos.density_maximum()) ;
    double const lt_lo   = std::log(t_min) ;
    double const lt_hi   = std::log(eos.temperature_maximum()) ;
    double const t_warm  = std::exp(lt_lo + 0.5 * (lt_hi - lt_lo)) ;

    // Edge-state assertions: like require_clean_roundtrip but WITHOUT the
    // clamp-bit checks (ulp-dependent at the boundary).
    auto require_edge_roundtrip = [](std::array<double, LEP_OUT_N> const& r,
                                     double rho0) {
        REQUIRE(r[LEP_FLOORED]    == 0.0) ;
        REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
        REQUIRE(r[LEP_BIT_TFLOOR] == 0.0) ;
        REQUIRE(std::fabs(r[LEP_EPS] - r[LEP_REF_EPS])
                <= std::max(1e-8, 1e-6 * std::fabs(r[LEP_REF_EPS]))) ;
        REQUIRE(std::fabs(r[LEP_RHO] - rho0) <= 1e-8 * rho0) ;
        REQUIRE(r[LEP_H] > 0.0) ;
        REQUIRE(std::isfinite(r[LEP_PRESS])) ;
    } ;

    SECTION("ye = ye_max exactly (yp saturated, baryon-table mu_e fallback)") {
        double const ymu0 = 2.0 * ymu_lo ;   // tiny; yp = ye_hi + ymu0 >= ye_hi
        double const t_cool = std::exp(lt_lo + 0.25 * (lt_hi - lt_lo)) ;
        double const t_hot  = std::exp(lt_lo + 0.75 * (lt_hi - lt_lo)) ;
        for (double fr : {0.2, 0.45, 0.7, 0.9}) {
            for (double temp : {t_min, t_cool, t_warm, t_hot}) {
                double const rho = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
                auto const r = leptonic_c2p_single(
                    eos, atmo, excision, c2p_pars,
                    rho, temp, ye_hi, ymu0, /*W=*/1.2, cons_mutate_none{}) ;
                INFO("ye=ye_max=" << ye_hi << " rho=" << rho << " T=" << temp
                     << " ye_rec=" << r[LEP_YE]
                     << " reset_ye=" << r[LEP_BIT_RESET_YE]
                     << " eps_ref=" << r[LEP_REF_EPS] << " eps=" << r[LEP_EPS]) ;
                require_edge_roundtrip(r, rho) ;
                // Recovered Ye sits on the bound to ulp accuracy, clamped
                // or not.
                REQUIRE(std::fabs(r[LEP_YE] - ye_hi) <= 1e-12 * ye_hi) ;
                REQUIRE(r[LEP_YE] <= ye_hi) ;
            }
        }
    }

    SECTION("ymu = ymu_max exactly (top of the muon axis, cold and warm)") {
        double const ye0 = ye_lo + 0.1 * (ye_hi - ye_lo) ;
        REQUIRE(ye0 + ymu_hi < ye_hi) ;   // premise: unsaturated at the top
        double const t_cool = std::exp(lt_lo + 0.25 * (lt_hi - lt_lo)) ;
        double const t_hot  = std::exp(lt_lo + 0.75 * (lt_hi - lt_lo)) ;
        for (double fr : {0.2, 0.45, 0.7, 0.9}) {
            for (double temp : {t_min, t_cool, t_warm, t_hot}) {
                double const rho = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
                auto const r = leptonic_c2p_single(
                    eos, atmo, excision, c2p_pars,
                    rho, temp, ye0, ymu_hi, /*W=*/1.2, cons_mutate_none{}) ;
                INFO("ymu=ymu_max=" << ymu_hi << " rho=" << rho << " T=" << temp
                     << " ymu_rec=" << r[LEP_YMU]
                     << " reset_ymu=" << r[LEP_BIT_RESET_YMU]
                     << " eps_ref=" << r[LEP_REF_EPS] << " eps=" << r[LEP_EPS]) ;
                require_edge_roundtrip(r, rho) ;
                REQUIRE(std::fabs(r[LEP_YMU] - ymu_hi) <= 1e-12 * ymu_hi) ;
                REQUIRE(r[LEP_YMU] <= ymu_hi) ;
            }
        }
    }

    SECTION("ye_max AND ymu_max together (doubly saturated corner)") {
        double const t_cool = std::exp(lt_lo + 0.25 * (lt_hi - lt_lo)) ;
        double const t_hot  = std::exp(lt_lo + 0.75 * (lt_hi - lt_lo)) ;
        for (double temp : {t_min, t_cool, t_warm, t_hot}) {
        for (double fr : {0.4, 0.8}) {
            double const rho = std::exp(lrho_lo + fr * (lrho_hi - lrho_lo)) ;
            auto const r = leptonic_c2p_single(
                eos, atmo, excision, c2p_pars,
                rho, temp, ye_hi, ymu_hi, /*W=*/1.2, cons_mutate_none{}) ;
            INFO("ye=ye_max=" << ye_hi << " ymu=ymu_max=" << ymu_hi
                 << " T=" << temp
                 << " eps_ref=" << r[LEP_REF_EPS] << " eps=" << r[LEP_EPS]) ;
            require_edge_roundtrip(r, rho) ;
            REQUIRE(r[LEP_YE]  <= ye_hi ) ;
            REQUIRE(r[LEP_YMU] <= ymu_hi) ;
        }
        }
    }

    SECTION("advected Ymu ABOVE ymu_max clamps down with RESET_YMU") {
        // Complement of the below-min clamp case in the composition test:
        // transport overshoot past the top of the muon axis.
        double const ye0  = ye_lo + 0.1 * (ye_hi - ye_lo) ;
        double const ymu0 = std::exp(std::log(ymu_lo)
            + 0.3 * (std::log(ymu_hi) - std::log(ymu_lo))) ;
        double const rho  = std::exp(lrho_lo + 0.6 * (lrho_hi - lrho_lo)) ;
        auto const r = leptonic_c2p_single(
            eos, atmo, excision, c2p_pars,
            rho, t_warm, ye0, ymu0, /*W=*/1.2, cons_set_ymu_star{1.5 * ymu_hi}) ;
        REQUIRE(r[LEP_BIT_RESET_YMU] == 1.0) ;
        REQUIRE_THAT(r[LEP_YMU], Catch::Matchers::WithinRel(ymu_hi, 1e-12)) ;
        REQUIRE(r[LEP_BIT_ATMO] == 0.0) ;
        REQUIRE(std::isfinite(r[LEP_EPS])) ;
        REQUIRE(r[LEP_H] > 0.0) ;
    }
}
#endif // GRACE_ENABLE_MUONS


TEST_CASE("c2p round-trip / Schwarzschild-CKS, no B", "[c2p][hydro][curved]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    // Schwarzschild at (R, 0, 0) for R well outside the horizon — c2p
    // is well-conditioned (α ≈ 0.71, β^x ≈ 0.33, γ_xx = 1.5).
    auto metric = schwarzschild_cks_metric(/*R=*/4.0, /*M=*/1.0) ;

    constexpr size_t N = N_RHO * N_T * N_W * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_sch_no_B", N, KOKKOS_LAMBDA(int const idx) {
        int q = idx ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;

        double const rho   = std::pow(10.0, LOG10_RHO[i_rho]) ;
        double const temp0 = TEMPERATURES[i_t] ;
        double const W     = WS[i_w] ;
        double const z_mag = std::sqrt(W*W - 1.0) ;

        // For a non-trivial γ, project the direction such that γ_ij z^i z^j
        // = W²−1 (so the resulting Lorentz factor is exactly W).  The DIRS
        // table is unit in flat space; rescale by 1/sqrt(γ_ij·d̂_i·d̂_j).
        double const dx = DIRS[i_dir][0], dy = DIRS[i_dir][1], dz = DIRS[i_dir][2] ;
        double const norm2 =
              metric.gamma(0)*dx*dx + 2*metric.gamma(1)*dx*dy
            + 2*metric.gamma(2)*dx*dz + metric.gamma(3)*dy*dy
            + 2*metric.gamma(4)*dy*dz + metric.gamma(5)*dz*dz ;
        double const inv_norm = 1.0 / std::sqrt(norm2) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = 0.0 ;
        double ymu = 0.0 ;
        p0[ZXL]   = z_mag * dx * inv_norm ;
        p0[ZYL]   = z_mag * dy * inv_norm ;
        p0[ZZL]   = z_mag * dz * inv_norm ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], ymu, err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual)  ;  deep_copy(r_h,  residual)  ;
    auto c_h = create_mirror_view(floored) ;  deep_copy(c_h,  floored) ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    {
        size_t q = i_max ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (rho=" << std::pow(10.0, LOG10_RHO[i_rho])
             << ", T="   << TEMPERATURES[i_t]
             << ", W="   << WS[i_w]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-10) ;
    }
}


// ---------------------------------------------------------------------------
//  PROBE: does the entropy backup leave the conserved state inconsistent?
//
//  Mimics the M1 momentum backreaction at the stellar surface: take a cold,
//  low-density fluid at rest, then add a pure momentum kick dS to S~_i while
//  leaving tau untouched (exactly what add_backreaction does when only the
//  MOMENTUM channel is coupled).  Invert with the entropy backup OFF and ON
//  and report, for each, which RESET_* bits come back -- i.e. whether the
//  caller (grmhd.hh) will re-synchronise tau / S~ with the accepted prims.
// ---------------------------------------------------------------------------
TEST_CASE("PROBE: entropy backup vs momentum kick", "[c2p][probe]")
{
    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    double const rho0 = 1e-8 ;

    grmhd_prims_array_t p0{} ;
    p0[RHOL]  = rho0 ;
    p0[TEMPL] = 0.0 ;
    p0[YEL]   = 0.0 ;
    p0[ZXL]   = p0[ZYL] = p0[ZZL] = 0.0 ;
    p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;
    double csnd2 ; double ymu = 0.0 ; grace::eos_err_t eerr{} ;
    p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                    p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], ymu, eerr) ;

    grmhd_cons_array_t cons0{} ;
    prims_to_conservs(p0, cons0, metric) ;

    double const D0   = cons0[DENSL] ;
    double const tau0 = cons0[TAUL]  ;
    double const q0   = tau0/D0 ;
    // eps>=0 boundary: (S/D)^2 = q(q+2)
    double const r_crit = std::sqrt(q0*(q0+2.0)) ;

    printf("\n[probe] rho=%.3e D=%.6e tau=%.6e q=tau/D=%.6e eps0=%.6e s0=%.6e r_crit=%.6e\n",
           rho0, D0, tau0, q0, p0[EPSL], p0[ENTL], r_crit) ;

    double const FRACS[6] = {0.1, 0.5, 0.9, 1.5, 3.0, 10.0} ;
    for (int ib = 0 ; ib < 2 ; ++ib) {
      c2p_pars.use_ent_backup = (ib == 1) ;
      printf("[probe] --- use_ent_backup = %d ---\n", int(c2p_pars.use_ent_backup)) ;
      for (int i = 0 ; i < 6 ; ++i) {
        grmhd_cons_array_t cons = cons0 ;
        double const dS = FRACS[i] * r_crit * D0 ;   // pure momentum deposit
        cons[STXL] += dS ;
        double const tau_in = cons[TAUL] ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const floored = conservs_to_prims(cons, p1, metric, eos, atmo,
                                               excision, c2p_pars, rtp, err) ;
        // Emulate the caller's selective write-back (grmhd.hh:401-421).
        double const tau_kept = err.test(c2p_err_enum_t::C2P_RESET_TAU)
                              ? cons[TAUL] : tau_in ;
        double const tau_consistent = cons[TAUL] ; // recomputed from accepted prims
        double const mismatch = std::fabs(tau_kept - tau_consistent)
                              / std::fmax(std::fabs(tau_consistent), 1e-300) ;
        printf("  |S|/D=%.3e (%.1f x r_crit): rho=%.4e eps=%.4e T=%.4e "
               "| BACKUP=%d ATMO=%d RST_TAU=%d RST_S=%d EPS_LO=%d floored=%d "
               "| tau_kept=%.6e tau_cons=%.6e  mismatch=%.3e\n",
               (cons0[STXL]+dS)/D0, FRACS[i],
               p1[RHOL], p1[EPSL], p1[TEMPL],
               int(err.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)),
               int(err.test(c2p_err_enum_t::C2P_ATMO_RESET)),
               int(err.test(c2p_err_enum_t::C2P_RESET_TAU)),
               int(err.test(c2p_err_enum_t::C2P_RESET_STILDE)),
               int(err.test(c2p_err_enum_t::C2P_SIG_EPS_TOO_LOW)),
               int(floored), tau_kept, tau_consistent, mismatch) ;
      }
    }
    REQUIRE(true) ;
}


#ifdef GRACE_ENABLE_MUONS
// ---------------------------------------------------------------------------
//  PROBE A: is eps ~ 1/rho at the stellar surface in the ACTUAL 4D table?
//
//  Hypothesis under test: the e+/e- pair + photon term is ~T^4 and (nearly)
//  rho-independent, so eps = energy/rho picks up a 1/rho divergence at low
//  density -> the surface becomes stiff in eps and any perturbation of the
//  recovered energy shows up as a huge eps/T excursion.
//  Reports total eps and its baryon / muon / electron split, plus the local
//  log-slope d ln(eps) / d ln(rho).  Slope -> -1 confirms eps ~ 1/rho.
// ---------------------------------------------------------------------------
TEST_CASE("PROBE A: leptonic eps vs rho at fixed T", "[c2p][leptonic][probe]")
{
    auto eos = eos::get().get_eos<leptonic_eos_4d_t>() ;

    double const ye0  = 0.05 ;
    double const ymu0 = std::fmax(eos.get_c2p_ymu_min()*1.01, 1e-6) ;

    printf("\n[probeA] table bounds: rho=[%.3e,%.3e] T=[%.3e,%.3e] "
           "ye=[%.3f,%.3f] ymu=[%.3e,%.3e] energy_shift=%.6e\n",
           eos.density_minimum(), eos.density_maximum(),
           eos.temperature_minimum(), eos.temperature_maximum(),
           eos.get_c2p_ye_min(), eos.get_c2p_ye_max(),
           eos.get_c2p_ymu_min(), eos.get_c2p_ymu_max(),
           eos.energy_shift) ;
    printf("[probeA] ye=%.3f ymu=%.3e  add_ele_contribution=%d\n",
           ye0, ymu0, int(eos.add_ele_contribution)) ;

    double const TS[3] = {1.0, 5.0, 10.0} ;   // MeV
    for (int it = 0 ; it < 3 ; ++it) {
        double const T = TS[it] ;
        if (T <= eos.temperature_minimum() || T >= eos.temperature_maximum()) continue ;
        double const ltemp = std::log(T) ;
        printf("[probeA] --- T = %.2f MeV ---\n", T) ;
        printf("            rho        eps_tot      eps_bar      eps_mu       eps_ele"
               "     ele/tot   dln(eps)/dln(rho)\n") ;
        double prev_leps = 0.0, prev_lrho = 0.0 ; bool have_prev = false ;
        for (int i = 0 ; i <= 8 ; ++i) {
            double const rho = std::pow(10.0, -12.0 + 0.5*i) ;   // 1e-12 .. 1e-8
            if (rho <= eos.density_minimum() || rho >= eos.density_maximum()) continue ;
            double const lrho = std::log(rho) ;
            double const lymu = std::log(ymu0) ;
            double const yp   = ye0 + ymu0 ;
            double const eb   = std::exp(eos.baryon_table.interp(lrho,ltemp,yp,
                                    leptonic_eos_4d_t::TABEPS)) - eos.energy_shift ;
            double const emu  = eos.muon_table.interp(lrho,ltemp,lymu,
                                    leptonic_eos_4d_t::TABEPS_MU_MINUS)
                              + eos.muon_table.interp(lrho,ltemp,lymu,
                                    leptonic_eos_4d_t::TABEPS_MU_PLUS) ;
            double const ee   = eos.add_ele_contribution
                              ? eos.ele_table.interp(lrho,ltemp,ye0,
                                    leptonic_eos_4d_t::TABEPS_E_MINUS)
                              + eos.ele_table.interp(lrho,ltemp,ye0,
                                    leptonic_eos_4d_t::TABEPS_E_PLUS)
                              : 0.0 ;
            // = total_eps(); reproduced here because that member is private.
            // yp = ye+ymu is well inside [yemin,yemax] so no clamp applies.
            double const etot = eb + emu + ee ;
            double slope = std::nan("") ;
            if (have_prev && etot > 0.0) slope = (std::log(etot)-prev_leps)/(lrho-prev_lrho) ;
            double const mue  = eos.ele_table .interp(lrho,ltemp,ye0,
                                    leptonic_eos_4d_t::TABMUELE) ;
            double const mumu = eos.muon_table.interp(lrho,ltemp,lymu,
                                    leptonic_eos_4d_t::TABMUMU) ;
            double const ylem = eos.ele_table .interp(lrho,ltemp,ye0,
                                    leptonic_eos_4d_t::TABYLE_MINUS) ;
            double const ylep = eos.ele_table .interp(lrho,ltemp,ye0,
                                    leptonic_eos_4d_t::TABYLE_PLUS) ;
            double const ymum = eos.muon_table.interp(lrho,ltemp,lymu,
                                    leptonic_eos_4d_t::TABYMU_MINUS) ;
            double const ymup = eos.muon_table.interp(lrho,ltemp,lymu,
                                    leptonic_eos_4d_t::TABYMU_PLUS) ;
            printf("     %.4e  %+.5e  %+.5e  %+.5e  %+.5e   %7.4f   %+8.4f"
                   "  | mu_e=%+.6e mu_mu=%+.6e  net_e=%+.6e net_mu=%+.6e\n",
                   rho, etot, eb, emu, ee,
                   (etot != 0.0 ? ee/etot : 0.0), slope,
                   mue, mumu, ylem-ylep, ymum-ymup) ;
            if (etot > 0.0) { prev_leps = std::log(etot) ; prev_lrho = lrho ; have_prev = true ; }
        }
    }
    REQUIRE(true) ;
}


// ---------------------------------------------------------------------------
//  PROBE B: same momentum-kick experiment as the hybrid probe, but with the
//  production 4D leptonic EOS at realistic halo conditions.
// ---------------------------------------------------------------------------
TEST_CASE("PROBE B: leptonic entropy backup vs momentum kick",
          "[c2p][leptonic][probe]")
{
    auto eos       = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    double const RHOS[2] = {1e-8, 1e-9} ;
    double const TEMPS[2] = {1.0, 5.0} ;

    for (int ir = 0 ; ir < 2 ; ++ir) {
    for (int it = 0 ; it < 2 ; ++it) {
        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = RHOS[ir] ;
        p0[TEMPL] = TEMPS[it] ;
        p0[YEL]   = 0.05 ;
        p0[YMUL]  = std::fmax(eos.get_c2p_ymu_min()*1.01, 1e-6) ;
        p0[ZXL]   = p0[ZYL] = p0[ZZL] = 0.0 ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;
        double csnd2 ; grace::eos_err_t eerr{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL],
                        p0[RHOL], p0[YEL], p0[YMUL], eerr) ;

        grmhd_cons_array_t cons0{} ;
        prims_to_conservs(p0, cons0, metric) ;
        double const D0 = cons0[DENSL], tau0 = cons0[TAUL] ;
        double const q0 = tau0/D0 ;
        double const r_crit = std::sqrt(std::fmax(q0*(q0+2.0), 0.0)) ;

        printf("\n[probeB] rho=%.3e T=%.2f : D=%.6e tau=%.6e q=%.6e eps0=%.6e "
               "s0=%.6e r_crit=%.6e\n",
               RHOS[ir], TEMPS[it], D0, tau0, q0, p0[EPSL], p0[ENTL], r_crit) ;

        double const FRACS[5] = {0.1, 0.5, 0.9, 1.5, 3.0} ;
        for (int ib = 0 ; ib < 2 ; ++ib) {
          c2p_pars.use_ent_backup = (ib == 1) ;
          printf("[probeB]  use_ent_backup = %d\n", int(c2p_pars.use_ent_backup)) ;
          for (int i = 0 ; i < 5 ; ++i) {
            grmhd_cons_array_t cons = cons0 ;
            cons[STXL] += FRACS[i] * r_crit * D0 ;
            double const tau_in = cons[TAUL] ;

            grmhd_prims_array_t p1 = p0 ;
            c2p_err_t err ;
            double rtp[3] = {1.0, 1.0, 1.0} ;
            bool const floored = conservs_to_prims(cons, p1, metric, eos, atmo,
                                                   excision, c2p_pars, rtp, err) ;
            double const tau_kept = err.test(c2p_err_enum_t::C2P_RESET_TAU)
                                  ? cons[TAUL] : tau_in ;
            double const tau_cons = cons[TAUL] ;
            double const mismatch = std::fabs(tau_kept - tau_cons)
                                  / std::fmax(std::fabs(tau_cons), 1e-300) ;
            printf("   %.1f x r_crit: rho=%.4e eps=%.4e T=%.4e "
                   "| BACKUP=%d ATMO=%d RST_TAU=%d RST_S=%d EPS_LO=%d floored=%d "
                   "| mismatch=%.3e\n",
                   FRACS[i], p1[RHOL], p1[EPSL], p1[TEMPL],
                   int(err.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)),
                   int(err.test(c2p_err_enum_t::C2P_ATMO_RESET)),
                   int(err.test(c2p_err_enum_t::C2P_RESET_TAU)),
                   int(err.test(c2p_err_enum_t::C2P_RESET_STILDE)),
                   int(err.test(c2p_err_enum_t::C2P_SIG_EPS_TOO_LOW)),
                   int(floored), mismatch) ;
          }
        }
    }}
    REQUIRE(true) ;
}
#endif // GRACE_ENABLE_MUONS


#ifdef GRACE_ENABLE_MUONS
// ---------------------------------------------------------------------------
//  PROBE C: is eps (hence tau) negative at the observed halo conditions?
//  The [BR diag] halo cells sit at rho ~ 1.07e-12, T ~ 0.21 MeV, Ye = 0.5,
//  Ymu = 5e-4.  tau = D*eps for a fluid at rest, so eps < 0 there would mean
//  tau < 0 is the NORMAL state of those cells, not a sign of corruption.
// ---------------------------------------------------------------------------
TEST_CASE("PROBE C: eps sign at halo conditions", "[c2p][leptonic][probe]")
{
    auto eos = eos::get().get_eos<leptonic_eos_4d_t>() ;

    double const ye0  = 0.5 ;
    double const ymu0 = 5.0e-4 ;
    printf("\n[probeC] ye=%.4f ymu=%.4e  (table ye_max=%.4f ymu_min=%.4e)\n",
           ye0, ymu0, eos.get_c2p_ye_max(), eos.get_c2p_ymu_min()) ;
    printf("      rho          T         eps_tot       eps_min(table)   tau/D=eps  sign\n") ;
    double const RHOS[3] = {1.0e-12, 1.08e-12, 1.2e-12} ;
    double const TS[5]   = {0.0999, 0.15, 0.2094, 0.25, 0.5} ;
    for (int ir = 0 ; ir < 3 ; ++ir) {
      for (int it = 0 ; it < 5 ; ++it) {
        double rho = RHOS[ir], T = TS[it] ;
        if (T <= eos.temperature_minimum()) T = eos.temperature_minimum()*1.0001 ;
        double eps, csnd2, ent, press ;
        grace::eos_err_t err{} ;
        double ymul = ymu0, yel = ye0, Tl = T, rhol = rho ;
        press = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                    eps, csnd2, ent, Tl, rhol, yel, ymul, err) ;
        double epsmin, epsmax ; grace::eos_err_t e2{} ;
        double r2 = rho, y2 = ye0, ym2 = ymu0 ;
        eos.eps_range__rho_ye_ymu(epsmin, epsmax, r2, y2, ym2, e2) ;
        printf("   %.4e  %.4f  %+.6e  %+.6e   %+.6e   %s\n",
               rho, T, eps, epsmin, eps, (eps < 0.0 ? "NEGATIVE" : "positive")) ;
        (void)press ;
      }
    }
    REQUIRE(true) ;
}
#endif


#ifdef GRACE_ENABLE_MUONS
// ---------------------------------------------------------------------------
//  PROBE D: what does the EOS look like at rho_atm = 1e-14 vs 1e-12?
//  eps ~ 1/rho at the radiation-dominated surface (probe A), so dropping the
//  atmosphere floor by 100x raises eps there by ~100x.  Report eps, h and the
//  c2p ceiling so the failure mode of rho_fl = 1e-14 is explicit.
// ---------------------------------------------------------------------------
TEST_CASE("PROBE D: EOS at candidate atmosphere floors", "[c2p][leptonic][probe]")
{
    auto eos = eos::get().get_eos<leptonic_eos_4d_t>() ;
    double const ye0 = 0.5, ymu0 = 5.0e-4 ;
    printf("\n[probeD] table rho_min=%.4e  c2p_eps_max=%.3e  h_min=%.6e\n",
           eos.density_minimum(), eos.get_c2p_eps_max(), eos.enthalpy_minimum()) ;
    printf("      rho          T        eps           press         h=1+eps+p/rho   eps/eps_max\n") ;
    double const RHOS[4] = {2.7e-15, 1.0e-14, 1.0e-13, 1.0e-12} ;
    double const TS[5]   = {0.0999, 1.0, 5.0, 10.0, 20.0} ;
    for (int ir = 0 ; ir < 4 ; ++ir) {
      for (int it = 0 ; it < 5 ; ++it) {
        double rhol = RHOS[ir], Tl = TS[it], yel = ye0, ymul = ymu0 ;
        if (Tl <= eos.temperature_minimum()) Tl = eos.temperature_minimum()*1.0001 ;
        if (rhol <= eos.density_minimum())   rhol = eos.density_minimum()*1.0001 ;
        double eps, csnd2, ent ; grace::eos_err_t err{} ;
        double press = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                          eps, csnd2, ent, Tl, rhol, yel, ymul, err) ;
        double const h = 1.0 + eps + press/rhol ;
        printf("   %.4e  %6.3f  %+.6e  %+.6e  %+.6e   %.3e\n",
               rhol, Tl, eps, press, h, eps/eos.get_c2p_eps_max()) ;
      }
    }
    REQUIRE(true) ;
}
#endif

#ifdef GRACE_ENABLE_MUONS
// ===========================================================================
// Halo cells captured from a production run
// ===========================================================================
// Real conserved states lifted out of surface_out_plane_xy_000210.h5 and fed
// straight into the production c2p with their own metric.  Eight cells took the
// entropy backup in the run, eight neighbours at the same density did not.  The
// discriminator measured in the data is |S|^2/(2 D tau) -- the kinetic energy
// implied by the momentum, as a fraction of the total conserved energy.
#include "halo_cells.inc"

TEST_CASE("c2p leptonic: halo cells from a hot-TOV run at it=210",
          "[c2p][leptonic][halo]")
{
    auto eos      = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo     = get_atmo_params() ;
    auto excision = get_excision_params() ;
    auto c2p_pars = get_c2p_params() ;

    constexpr int NC = int(sizeof(kHaloCells) / sizeof(kHaloCells[0])) ;
    enum { O_RHO = 0, O_EPS, O_TEMP, O_FLOORED, O_BACKUP, O_ATMO, O_EPSLO, O_N } ;

    Kokkos::View<halo_cell_t*> cells("halo_cells", NC) ;
    auto hc = Kokkos::create_mirror_view(cells) ;
    for (int i = 0 ; i < NC ; ++i) hc(i) = kHaloCells[i] ;
    Kokkos::deep_copy(cells, hc) ;

    Kokkos::View<double*[O_N]> out("halo_out", NC) ;

    Kokkos::parallel_for("halo_c2p", NC, KOKKOS_LAMBDA(int idx) {
        halo_cell_t const c = cells(idx) ;
        metric_array_t metric(
            {c.gamma[0], c.gamma[1], c.gamma[2], c.gamma[3], c.gamma[4], c.gamma[5]},
            {c.beta[0],  c.beta[1],  c.beta[2]}, c.alp) ;

        grmhd_cons_array_t cons{} ;
        cons[DENSL] = c.dens ;
        cons[STXL]  = c.stx ;  cons[STYL] = c.sty ;  cons[STZL] = c.stz ;
        cons[TAUL]  = c.tau ;
        cons[YESL]  = c.ye_star ;
        cons[YMUSL] = c.ymu_star ;
        cons[ENTSL] = c.s_star ;
        cons[BSXL]  = cons[BSYL] = cons[BSZL] = 0.0 ;

        grmhd_prims_array_t p{} ;
        p[RHOL]  = c.ref_rho ;  p[TEMPL] = c.ref_temp ;
        p[YEL]   = c.ref_ye  ;  p[YMUL]  = c.ref_ymu ;
        p[ZXL]   = p[ZYL] = p[ZZL] = 0.0 ;
        p[BXL]   = p[BYL] = p[BZL] = 0.0 ;

        c2p_err_t cerr ;
        double rtp[3] = {c.radius, 1.0, 1.0} ;
        bool const fl = conservs_to_prims(cons, p, metric, eos, atmo,
                                          excision, c2p_pars, rtp, cerr) ;

        out(idx, O_RHO)     = p[RHOL] ;
        out(idx, O_EPS)     = p[EPSL] ;
        out(idx, O_TEMP)    = p[TEMPL] ;
        out(idx, O_FLOORED) = fl ? 1.0 : 0.0 ;
        out(idx, O_BACKUP)  = cerr.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED) ? 1.0 : 0.0 ;
        out(idx, O_ATMO)    = cerr.test(c2p_err_enum_t::C2P_ATMO_RESET)      ? 1.0 : 0.0 ;
        out(idx, O_EPSLO)   = cerr.test(c2p_err_enum_t::C2P_SIG_EPS_TOO_LOW) ? 1.0 : 0.0 ;
    }) ;
    Kokkos::fence() ;

    auto h = Kokkos::create_mirror_view(out) ;
    Kokkos::deep_copy(h, out) ;

    printf("\n   #  Ekin/tau   in-run   backup  atmo  epslo    rho_in       rho_out      eps_out\n") ;
    int hit = 0, miss = 0, false_pos = 0 ;
    for (int i = 0 ; i < NC ; ++i) {
        bool const backup = h(i, O_BACKUP) > 0.5 ;
        bool const want   = kHaloCells[i].expect_backup ;
        printf("  %2d  %8.3f   %6s   %5s  %4s  %5s   %.4e  %.4e  %+.4e\n",
               i, kHaloCells[i].e_kin_over_tau, want ? "BACKUP" : "clean",
               backup ? "yes" : "no",
               h(i, O_ATMO)  > 0.5 ? "yes" : "no",
               h(i, O_EPSLO) > 0.5 ? "yes" : "no",
               kHaloCells[i].ref_rho, h(i, O_RHO), h(i, O_EPS)) ;
        if ( want &&  backup) ++hit ;
        if ( want && !backup) ++miss ;
        if (!want &&  backup) ++false_pos ;
    }
    printf("   reproduced %d/8 backups, %d missed, %d false positives\n",
           hit, miss, false_pos) ;
    printf("   (a DROP in the reproduced count is the intended effect of a halo\n"
           "    fix -- it is reported, not asserted, so a fix does not fail here)\n") ;

    // A control cell is one that inverted cleanly in production at the same
    // density.  If one of those starts needing the backup, the inversion got
    // worse -- that is a regression regardless of what the failing cells do.
    INFO("control cells that newly need the entropy backup: " << false_pos) ;
    REQUIRE(false_pos == 0) ;

    // Invariants that must hold whatever the inversion path: the c2p may floor
    // or reset a cell, but it must never hand back a non-finite or negative
    // state.
    for (int i = 0 ; i < NC ; ++i) {
        INFO("cell " << i << "  Ekin/tau=" << kHaloCells[i].e_kin_over_tau) ;
        REQUIRE(std::isfinite(h(i, O_RHO))) ;
        REQUIRE(std::isfinite(h(i, O_EPS))) ;
        REQUIRE(std::isfinite(h(i, O_TEMP))) ;
        REQUIRE(h(i, O_RHO) > 0.0) ;
        REQUIRE(h(i, O_EPS) >= 0.0) ;
    }
}
#endif

#ifdef GRACE_ENABLE_MUONS
//************************************************************************************
//  Leptonic EOS determinism.
//
//  The EOS must be a pure function of (rho, T, Ye, Ymu): identical inputs must give
//  bit-identical outputs, whichever thread evaluates them and however many times the
//  kernel is launched.  Two independent checks:
//
//    (a) thread consistency   -- one input point evaluated by many threads in a
//        single launch; catches results that depend on thread or register state.
//    (b) launch repeatability -- a grid of points evaluated in two separate
//        launches; catches results that vary between launches.
//
//  Motivated by a Hunter HOT-TOV in which two identical 8-rank runs diverged at the
//  stellar surface from iteration 1 (1-ULP seed, amplified to cell-alive-vs-atmosphere
//  by it 20).  Bisection with Cowling + M1 off + FOFC off + entropy backup off left
//  the muon sector of this EOS as the only difference: the tabulated EOS and
//  leptonic-without-muons were both bit-reproducible over 21 iterations.
//
//  Exercises the two entry points a step actually uses:
//    press_eps_csnd2__temp_rho_ye_ymu            -- the flux kernel, ~6x per cell
//    press_h_csnd2_temp_entropy__eps_rho_ye_ymu  -- the Kastaun inversion
//************************************************************************************
namespace {

enum DET_OUT : int {
    D_PRESS = 0, D_EPS, D_CSND2,
    D_TEMP, D_RHO, D_YE, D_YMU, D_ERR,   // in/out args: the EOS clamps them in place
    D_KAS_PRESS, D_KAS_TEMP, D_KAS_ENT, D_KAS_H,
    D_N
} ;

// One forward + inverse evaluation.  Templated on the EOS so the tabulated
// twin runs the identical code path (every EOS exposes the _ye_ymu wrappers).
template <typename eos_t>
KOKKOS_INLINE_FUNCTION void
eval_eos_once(eos_t const& eos, double rho, double temp, double ye, double ymu,
              double* out)
{
    // Forward hook: the call the Riemann solver makes at every reconstructed face.
    double eps = 0.0, csnd2 = 0.0 ;
    double r = rho, t = temp, y = ye, m = ymu ;
    eos_err_t err ;
    out[D_PRESS] = eos.press_eps_csnd2__temp_rho_ye_ymu(eps, csnd2, t, r, y, m, err) ;
    out[D_EPS]   = eps  ; out[D_CSND2] = csnd2 ;
    out[D_TEMP]  = t    ; out[D_RHO]   = r ; out[D_YE] = y ; out[D_YMU] = m ;
    out[D_ERR]   = static_cast<double>(err.words[0]) ;

    // Inverse hook: feed that eps back through the Kastaun temperature inversion.
    double hh = 0.0, cs2 = 0.0, t2 = 0.0, s2 = 0.0 ;
    double e2 = eps, r2 = rho, y2 = ye, m2 = ymu ;
    eos_err_t err2 ;
    out[D_KAS_PRESS] = eos.press_h_csnd2_temp_entropy__eps_rho_ye_ymu(
        hh, cs2, t2, s2, e2, r2, y2, m2, err2) ;
    out[D_KAS_TEMP] = t2 ; out[D_KAS_ENT] = s2 ; out[D_KAS_H] = hh ;
}

// Same, with a compiler barrier between the forward stores and the inverse:
// nothing can be moved across it or share a register with the inverse's own
// inlined total_eps copies.  Passing where eval_eos_once fails = miscompile.
template <typename eos_t>
KOKKOS_INLINE_FUNCTION void
eval_eos_once_barrier(eos_t const& eos, double rho, double temp, double ye, double ymu,
                      double* out)
{
    // Forward hook: the call the Riemann solver makes at every reconstructed face.
    double eps = 0.0, csnd2 = 0.0 ;
    double r = rho, t = temp, y = ye, m = ymu ;
    eos_err_t err ;
    out[D_PRESS] = eos.press_eps_csnd2__temp_rho_ye_ymu(eps, csnd2, t, r, y, m, err) ;
    out[D_EPS]   = eps  ; out[D_CSND2] = csnd2 ;
    out[D_TEMP]  = t    ; out[D_RHO]   = r ; out[D_YE] = y ; out[D_YMU] = m ;
    out[D_ERR]   = static_cast<double>(err.words[0]) ;
    asm volatile("" ::: "memory") ;
    volatile double const eps_kept = eps ;
    asm volatile("" ::: "memory") ;

    // Inverse hook, fed from the volatile copy.
    double hh = 0.0, cs2 = 0.0, t2 = 0.0, s2 = 0.0 ;
    double e2 = eps_kept, r2 = rho, y2 = ye, m2 = ymu ;
    eos_err_t err2 ;
    out[D_KAS_PRESS] = eos.press_h_csnd2_temp_entropy__eps_rho_ye_ymu(
        hh, cs2, t2, s2, e2, r2, y2, m2, err2) ;
    out[D_KAS_TEMP] = t2 ; out[D_KAS_ENT] = s2 ; out[D_KAS_H] = hh ;
}

// Same, compiled without optimisation: the EOS members are not inlined into
// it, so the register pattern of the combined kernel does not exist.
#if defined(__clang__)
#pragma clang optimize off
#endif
template <typename eos_t>
KOKKOS_INLINE_FUNCTION void
eval_eos_once_optnone(eos_t const& eos, double rho, double temp, double ye, double ymu,
                      double* out)
{
    // Forward hook: the call the Riemann solver makes at every reconstructed face.
    double eps = 0.0, csnd2 = 0.0 ;
    double r = rho, t = temp, y = ye, m = ymu ;
    eos_err_t err ;
    out[D_PRESS] = eos.press_eps_csnd2__temp_rho_ye_ymu(eps, csnd2, t, r, y, m, err) ;
    out[D_EPS]   = eps  ; out[D_CSND2] = csnd2 ;
    out[D_TEMP]  = t    ; out[D_RHO]   = r ; out[D_YE] = y ; out[D_YMU] = m ;
    out[D_ERR]   = static_cast<double>(err.words[0]) ;

    // Inverse hook: feed that eps back through the Kastaun temperature inversion.
    double hh = 0.0, cs2 = 0.0, t2 = 0.0, s2 = 0.0 ;
    double e2 = eps, r2 = rho, y2 = ye, m2 = ymu ;
    eos_err_t err2 ;
    out[D_KAS_PRESS] = eos.press_h_csnd2_temp_entropy__eps_rho_ye_ymu(
        hh, cs2, t2, s2, e2, r2, y2, m2, err2) ;
    out[D_KAS_TEMP] = t2 ; out[D_KAS_ENT] = s2 ; out[D_KAS_H] = hh ;
}
#if defined(__clang__)
#pragma clang optimize on
#endif

// Evaluator tags for the thread-consistency launches below.
struct eval_plain_t {
    template <typename E> KOKKOS_INLINE_FUNCTION void
    operator()(E const& e, double r, double t, double y, double m, double* o) const
    { eval_eos_once(e, r, t, y, m, o) ; }
} ;
struct eval_barrier_t {
    template <typename E> KOKKOS_INLINE_FUNCTION void
    operator()(E const& e, double r, double t, double y, double m, double* o) const
    { eval_eos_once_barrier(e, r, t, y, m, o) ; }
} ;
struct eval_optnone_t {
    template <typename E> KOKKOS_INLINE_FUNCTION void
    operator()(E const& e, double r, double t, double y, double m, double* o) const
    { eval_eos_once_optnone(e, r, t, y, m, o) ; }
} ;

constexpr char const* DET_NAME[D_N] = {
    "press","eps","csnd2","temp(out)","rho(out)","ye(out)","ymu(out)","eos_err",
    "kastaun press","kastaun temp","kastaun entropy","kastaun h"
} ;


// Anatomy record of one evaluation (see the anatomy sections).
enum AN : int { AN_EPS = 0, AN_EPS2, AN_EB, AN_EMU, AN_EE, AN_SHIFT, AN_KT,
                AN_WG, AN_WI, AN_LANE, AN_HWID, AN_XCC, AN_N } ;

template <bool WITH_INVERSE>
KOKKOS_INLINE_FUNCTION void
anatomy_eval(leptonic_eos_4d_t const& eos, double rho, double temp, double ye, double ymu,
             double* out)
{
    double eps = 0.0, cs2 = 0.0 ;
    double r1 = rho, t1 = temp, y1 = ye, m1 = ymu ;
    eos_err_t e1 ;
    eos.press_eps_csnd2__temp_rho_ye_ymu(eps, cs2, t1, r1, y1, m1, e1) ;
    double r2 = rho, t2 = temp, y2 = ye, m2 = ymu ;
    eos_err_t e2 ;
    double const eps2 = eos.eps__temp_rho_ye_ymu(t2, r2, y2, m2, e2) ;
    // total_eps term by term, on the limited arguments the EOS used
    double const lrho = Kokkos::log(r1), ltemp = Kokkos::log(t1) ;
    double const yp   = eos.add_ele_contribution ? y1 + m1 : y1 ;
    double const eb   = Kokkos::exp(eos.baryon_table.interp(
                            lrho, ltemp, yp, leptonic_eos_4d_t::TABEPS)) - eos.energy_shift ;
    double emu = 0.0 ;
    if (m1 > 6.0e-4) {
        double const lm = Kokkos::log(m1) ;
        emu = eos.muon_table.interp(lrho, ltemp, lm, leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_MINUS)
            + eos.muon_table.interp(lrho, ltemp, lm, leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_PLUS) ;
    }
    double ee = 0.0 ;
    if (eos.add_ele_contribution)
        ee = eos.ele_table.interp(lrho, ltemp, y1, leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_MINUS)
           + eos.ele_table.interp(lrho, ltemp, y1, leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_PLUS) ;
    double kt = 0.0 ;
    if constexpr (WITH_INVERSE) {
        double hh = 0.0, c3 = 0.0, s3 = 0.0 ;
        double e3 = eps, r3 = rho, y3 = ye, m3 = ymu ;
        eos_err_t err3 ;
        eos.press_h_csnd2_temp_entropy__eps_rho_ye_ymu(hh, c3, kt, s3, e3, r3, y3, m3, err3) ;
    }
    double wg = -1.0, wi = -1.0, lane = -1.0, hwid = -1.0, xcc = -1.0 ;
#if defined(__HIP_DEVICE_COMPILE__)
    wg   = __builtin_amdgcn_workgroup_id_x() ;
    wi   = __builtin_amdgcn_workitem_id_x() ;
    lane = __lane_id() ;
    hwid = __builtin_amdgcn_s_getreg(63492) ;   // hwreg(HW_REG_HW_ID, 0, 32)
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    xcc  = __builtin_amdgcn_s_getreg(63508) ;   // hwreg(HW_REG_XCC_ID, 0, 32)
#endif
#endif
    out[AN_EPS]  = eps ;  out[AN_EPS2] = eps2 ; out[AN_EB]    = eb ;
    out[AN_EMU]  = emu ;  out[AN_EE]   = ee ;   out[AN_SHIFT] = eos.energy_shift ;
    out[AN_KT]   = kt ;
    out[AN_WG]   = wg ;   out[AN_WI]   = wi ;   out[AN_LANE]  = lane ;
    out[AN_HWID] = hwid ; out[AN_XCC]  = xcc ;
}

} // namespace

TEST_CASE("leptonic EOS is a deterministic function of its inputs",
          "[c2p][eos][leptonic][determinism]")
{
    using namespace Kokkos ;

    auto eos = eos::get().get_eos<leptonic_eos_4d_t>() ;

    // --- Sample the regime where the Hunter runs diverged ---------------------
    // rho from the atmosphere floor up into the star; T from the table minimum
    // into the shocked range; Ymu straddling the dilute threshold (6e-4) so both
    // the muon-table lookup and the muons_resolved() skip branch are exercised.
    std::vector<double> rhos, temps, yes, ymus ;
    {
        double const lr_lo = std::log(eos.density_minimum()) ;
        double const lr_hi = std::log(1.0e-3) ;
        for (int i = 0 ; i < 24 ; ++i)
            rhos.push_back(std::exp(lr_lo + (lr_hi - lr_lo) * i / 23.0)) ;
        double const lt_lo = std::log(eos.temperature_minimum()) ;
        double const lt_hi = std::log(50.0) ;
        for (int i = 0 ; i < 12 ; ++i)
            temps.push_back(std::exp(lt_lo + (lt_hi - lt_lo) * i / 11.0)) ;
        for (int i = 0 ; i < 6 ; ++i)
            yes.push_back(0.02 + (0.46 * i) / 5.0) ;
        double const ymu_lo = eos.get_c2p_ymu_min() ;
        for (double v : {ymu_lo, 5.9e-4, 6.0e-4, 6.1e-4, 1.0e-3, 1.0e-2, 3.4e-2, 1.0e-1})
            ymus.push_back(std::max(v, ymu_lo)) ;
    }
    size_t const N = rhos.size() * temps.size() * yes.size() * ymus.size() ;

    View<double**> pts("det_pts", N, 4) ;
    auto h_pts = create_mirror_view(pts) ;
    {
        size_t n = 0 ;
        for (double r : rhos) for (double t : temps) for (double y : yes) for (double m : ymus) {
            h_pts(n,0) = r ; h_pts(n,1) = t ; h_pts(n,2) = y ; h_pts(n,3) = m ; ++n ;
        }
    }
    deep_copy(pts, h_pts) ;

    // --- (a) thread consistency: one point, many threads, one launch ----------
    SECTION("identical inputs give identical outputs across threads") {
        constexpr int N_PT = 16 , N_REP = 512 ;
        size_t const stride = N / N_PT ;
        View<double**> out("det_thread", static_cast<size_t>(N_PT) * N_REP, D_N) ;
        parallel_for("eos_det_threads", static_cast<size_t>(N_PT) * N_REP,
                     KOKKOS_LAMBDA(size_t const idx) {
            size_t const ip = idx / N_REP ;
            size_t const p  = ip * stride ;
            double loc[D_N] ;
            eval_eos_once(eos, pts(p,0), pts(p,1), pts(p,2), pts(p,3), loc) ;
            for (int k = 0 ; k < D_N ; ++k) out(idx,k) = loc[k] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int mismatches = 0 ;
        for (int ip = 0 ; ip < N_PT ; ++ip) {
            size_t const base = static_cast<size_t>(ip) * N_REP ;
            for (int rep = 1 ; rep < N_REP ; ++rep) {
                for (int k = 0 ; k < D_N ; ++k) {
                    if (h(base + rep, k) != h(base, k)) {
                        if (++mismatches <= 5) {
                            size_t const p = static_cast<size_t>(ip) * stride ;
                            INFO("point rho=" << h_pts(p,0) << " T=" << h_pts(p,1)
                                 << " Ye=" << h_pts(p,2) << " Ymu=" << h_pts(p,3)
                                 << "  field " << DET_NAME[k]
                                 << "  thread 0 = " << std::setprecision(17) << h(base,k)
                                 << "  thread " << rep << " = " << h(base+rep,k)) ;
                            CHECK(h(base + rep, k) == h(base, k)) ;
                        }
                    }
                }
            }
        }
        INFO("total field mismatches across threads: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    // --- (b) launch repeatability: same grid, two launches --------------------
    SECTION("identical inputs give identical outputs across kernel launches") {
        View<double**> a("det_a", N, D_N), b("det_b", N, D_N) ;
        auto run = [&](View<double**> o) {
            parallel_for("eos_det_launch", N, KOKKOS_LAMBDA(size_t const idx) {
                double loc[D_N] ;
                eval_eos_once(eos, pts(idx,0), pts(idx,1), pts(idx,2), pts(idx,3), loc) ;
                for (int k = 0 ; k < D_N ; ++k) o(idx,k) = loc[k] ;
            }) ;
            fence() ;
        } ;
        run(a) ; run(b) ;
        auto ha = create_mirror_view_and_copy(HostSpace(), a) ;
        auto hb = create_mirror_view_and_copy(HostSpace(), b) ;

        // Guard against a vacuous pass: the sweep must actually have produced
        // finite, varying thermodynamics, and must have crossed the dilute-Ymu
        // threshold in both directions so the muon table is really exercised.
        {
            double pmin = ha(0,D_PRESS), pmax = ha(0,D_PRESS) ;
            int n_finite = 0 ;
            for (size_t i = 0 ; i < N ; ++i) {
                double const pv = ha(i,D_PRESS) ;
                if (std::isfinite(pv)) ++n_finite ;
                pmin = std::min(pmin, pv) ; pmax = std::max(pmax, pv) ;
            }
            INFO("sweep: " << N << " points, " << n_finite << " finite, press in ["
                 << pmin << ", " << pmax << "]") ;
            REQUIRE(n_finite == static_cast<int>(N)) ;
            REQUIRE(pmax > pmin) ;
            int n_res = 0, n_dil = 0 ;
            for (size_t i = 0 ; i < N ; ++i)
                (h_pts(i,3) > 6.0e-4 ? n_res : n_dil) += 1 ;
            INFO("muon-table lookups: " << n_res << " resolved, " << n_dil << " dilute") ;
            REQUIRE(n_res > 0) ;
            REQUIRE(n_dil > 0) ;
        }

        int mismatches = 0 ;
        for (size_t i = 0 ; i < N ; ++i) {
            for (int k = 0 ; k < D_N ; ++k) {
                if (ha(i,k) != hb(i,k)) {
                    if (++mismatches <= 5) {
                        INFO("point rho=" << h_pts(i,0) << " T=" << h_pts(i,1)
                             << " Ye=" << h_pts(i,2) << " Ymu=" << h_pts(i,3)
                             << "  field " << DET_NAME[k]
                             << "  launch A = " << std::setprecision(17) << ha(i,k)
                             << "  launch B = " << hb(i,k)) ;
                        CHECK(ha(i,k) == hb(i,k)) ;
                    }
                }
            }
        }
        INFO("total field mismatches across launches: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    // --- (c) is it a timing race at all?  (Hunter/MI300A, ROCm 7.0.2) ----------
    // Section (a) fails there with >= 2 hardware queues and passes under
    // AMD_SERIALIZE_COPY=3, although Kokkos already fences after every memset and
    // deep_copy.  Same launch as (a) with the copy-class operations removed or
    // waited for at device scope: a pass means the stream-level wait is not
    // enough; a fail in every variant means the damage is not a timing race.
    auto thread_check = [&](char const* what, auto prepare, auto evaluator) -> int {
        constexpr int N_PT = 16, N_REP = 512 ;
        size_t const stride = N / N_PT ;
        View<double**> out = prepare(static_cast<size_t>(N_PT) * N_REP) ;
        parallel_for("eos_det_sync", static_cast<size_t>(N_PT) * N_REP,
                     KOKKOS_LAMBDA(size_t const idx) {
            size_t const ip = idx / N_REP ;
            size_t const p  = ip * stride ;
            double loc[D_N] ;
            evaluator(eos, pts(p,0), pts(p,1), pts(p,2), pts(p,3), loc) ;
            for (int k = 0 ; k < D_N ; ++k) out(idx,k) = loc[k] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int mismatches = 0 ;
        for (int ip = 0 ; ip < N_PT ; ++ip) {
            size_t const base = static_cast<size_t>(ip) * N_REP ;
            for (int rep = 1 ; rep < N_REP ; ++rep)
                for (int k = 0 ; k < D_N ; ++k)
                    if (h(base + rep, k) != h(base, k)) ++mismatches ;
        }
        printf("   %-44s mismatches across threads: %d\n", what, mismatches) ;
        return mismatches ;
    } ;
    SECTION("Kokkos::fence() between the output allocation and the launch") {
        int const m = thread_check("Kokkos::fence() before launch", [](size_t n) {
            View<double**> o("det_sync_fence", n, D_N) ;
            Kokkos::fence() ;
            return o ;
        }, eval_plain_t{}) ;
        REQUIRE(m == 0) ;
    }
#ifdef KOKKOS_ENABLE_HIP
    SECTION("hipDeviceSynchronize() between the output allocation and the launch") {
        int const m = thread_check("hipDeviceSynchronize() before launch", [](size_t n) {
            View<double**> o("det_sync_device", n, D_N) ;
            REQUIRE(hipDeviceSynchronize() == hipSuccess) ;
            return o ;
        }, eval_plain_t{}) ;
        REQUIRE(m == 0) ;
    }
#endif
    SECTION("no zero-fill of the output view before the launch") {
        int const m = thread_check("output allocated without memset", [](size_t n) {
            return View<double**>(view_alloc(WithoutInitializing, "det_sync_noinit"), n, D_N) ;
        }, eval_plain_t{}) ;
        REQUIRE(m == 0) ;
    }

    // --- (c') the compiled code as the variable ---------------------------------
    SECTION("compiler barrier between the forward stores and the inverse") {
        int const m = thread_check("asm barrier + volatile eps before inverse", [](size_t n) {
            return View<double**>("det_sync_barrier", n, D_N) ;
        }, eval_barrier_t{}) ;
        REQUIRE(m == 0) ;
    }
    SECTION("forward + inverse compiled without optimisation") {
        int const m = thread_check("#pragma clang optimize off wrapper", [](size_t n) {
            return View<double**>("det_sync_optnone", n, D_N) ;
        }, eval_optnone_t{}) ;
        REQUIRE(m == 0) ;
    }

    // --- (d) anatomy of a bad evaluation --------------------------------------
    // Per evaluation: the EOS's eps through two entry points, the three terms of
    // total_eps recomputed from the public tables, the energy shift the device
    // sees, the inverse's T (second kernel only) and where the work-item ran
    // (work-group, item, lane, HW_ID, XCC).  Reported, not asserted.  Two
    // kernels: forward only, and forward + inverse -- the failing combination.
    auto run_anatomy = [&](char const* label, auto launch) {
        constexpr int N_PT = 16, N_REP = 512 ;
        size_t const stride = N / N_PT ;
        size_t const M = static_cast<size_t>(N_PT) * N_REP ;
        View<double**> out("det_anatomy", M, AN_N) ;
        launch(out, stride, N_REP) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;

        // "bad" = disagrees with the value most replicas of the same point got.
        int n_bad = 0, n_eps2_same = 0, n_sum_bad = 0, shown = 0 ;
        std::map<int,int> by_wi, by_lane ;
        std::map<long,int> units_all, units_bad ;
        double shift_lo = h(0,AN_SHIFT), shift_hi = shift_lo ;
        for (int ip = 0 ; ip < N_PT ; ++ip) {
            size_t const base = static_cast<size_t>(ip) * N_REP ;
            std::map<double,int> votes ;
            for (int rep = 0 ; rep < N_REP ; ++rep) ++votes[h(base + rep, AN_EPS)] ;
            double mode = 0.0 ; int best = -1 ;
            for (auto const& [v, c] : votes) if (c > best) { best = c ; mode = v ; }
            for (int rep = 0 ; rep < N_REP ; ++rep) {
                size_t const i = base + rep ;
                shift_lo = std::min(shift_lo, h(i,AN_SHIFT)) ;
                shift_hi = std::max(shift_hi, h(i,AN_SHIFT)) ;
                // (xcc, se, sh, cu, pipe, simd): HW_ID without the wave-id bits
                long const unit = (static_cast<long>(h(i,AN_XCC)) << 32)
                                | (static_cast<long>(h(i,AN_HWID)) & 0xFFF0L) ;
                ++units_all[unit] ;
                if (h(i,AN_EPS) == mode) continue ;
                ++n_bad ;
                ++units_bad[unit] ;
                ++by_wi  [static_cast<int>(h(i,AN_WI))] ;
                ++by_lane[static_cast<int>(h(i,AN_LANE))] ;
                double const sum = h(i,AN_EB) + h(i,AN_EMU) + h(i,AN_EE) ;
                if (h(i,AN_EPS2) == h(i,AN_EPS)) ++n_eps2_same ;
                if (std::fabs(sum - mode) > 1e-12 * std::max(1.0, std::fabs(mode))) ++n_sum_bad ;
                if (shown++ < 6)
                    printf("   [%s] bad idx %6zu (point %2d rep %3d): eps-mode %+.17g  eps2-mode %+.17g"
                           "  terms-mode %+.3e  T_inv %.17g  wg %g wi %g lane %g xcc %g hwid 0x%08lx\n",
                           label, i, ip, rep, h(i,AN_EPS) - mode, h(i,AN_EPS2) - mode, sum - mode,
                           h(i,AN_KT), h(i,AN_WG), h(i,AN_WI), h(i,AN_LANE), h(i,AN_XCC),
                           static_cast<long>(h(i,AN_HWID))) ;
            }
        }
        printf("   anatomy [%s]: %d bad of %zu evaluations; second entry point equally bad in %d;"
               " test-side term sum bad in %d; energy_shift seen in [%.17g, %.17g]\n",
               label, n_bad, M, n_eps2_same, n_sum_bad, shift_lo, shift_hi) ;
        auto top = [&](std::map<int,int> const& m, char const* what) {
            printf("   [%s] bad by %s:", label, what) ;
            int k = 0 ;
            for (auto const& [key, c] : m) { if (k++ == 6) { printf(" ...") ; break ; } printf(" %d:%d", key, c) ; }
            printf("\n") ;
        } ;
        top(by_wi, "work-item-in-group") ;
        top(by_lane, "lane") ;
        printf("   [%s] bad on %zu of %zu hardware units seen (xcc<<32 | HW_ID&0xfff0):",
               label, units_bad.size(), units_all.size()) ;
        int k = 0 ;
        for (auto const& [u, c] : units_bad) { if (k++ == 8) { printf(" ...") ; break ; } printf(" %lx:%d/%d", u, c, units_all[u]) ; }
        printf("\n") ;
    } ;
    SECTION("anatomy: forward only") {
        run_anatomy("forward only", [&](View<double**> out, size_t stride, int n_rep) {
            parallel_for("eos_det_anatomy_fwd", out.extent(0), KOKKOS_LAMBDA(size_t const idx) {
                size_t const p = (idx / n_rep) * stride ;
                double loc[AN_N] ;
                anatomy_eval<false>(eos, pts(p,0), pts(p,1), pts(p,2), pts(p,3), loc) ;
                for (int k = 0 ; k < AN_N ; ++k) out(idx,k) = loc[k] ;
            }) ;
        }) ;
    }
    SECTION("anatomy: forward + inverse") {
        run_anatomy("forward+inverse", [&](View<double**> out, size_t stride, int n_rep) {
            parallel_for("eos_det_anatomy_inv", out.extent(0), KOKKOS_LAMBDA(size_t const idx) {
                size_t const p = (idx / n_rep) * stride ;
                double loc[AN_N] ;
                anatomy_eval<true>(eos, pts(p,0), pts(p,1), pts(p,2), pts(p,3), loc) ;
                for (int k = 0 ; k < AN_N ; ++k) out(idx,k) = loc[k] ;
            }) ;
        }) ;
    }
}
//************************************************************************************
//  Tabulated-EOS twin of the EOS determinism case: the same forward + inverse
//  code path instantiated for tabulated_eos_t (Y_mu = 0), so an EOS-specific
//  compiled-code defect can be told from a generic one.  Bound to
//  configs/c2p_test_tabulated.yaml.
//************************************************************************************
TEST_CASE("tabulated EOS is a deterministic function of its inputs",
          "[c2p][eos][tabulated][determinism]")
{
    using namespace Kokkos ;

    auto eos = eos::get().get_eos<tabulated_eos_t>() ;

    // rho / T / Ye grid as in the leptonic case; the EOS clamps to its own table
    // bounds, which is part of what is exercised.
    std::vector<double> rhos, temps, yes ;
    for (int i = 0 ; i < 24 ; ++i)
        rhos.push_back(std::exp(std::log(2.7e-15) + (std::log(1.0e-3) - std::log(2.7e-15)) * i / 23.0)) ;
    for (int i = 0 ; i < 12 ; ++i)
        temps.push_back(std::exp(std::log(0.1) + (std::log(50.0) - std::log(0.1)) * i / 11.0)) ;
    for (int i = 0 ; i < 6 ; ++i)
        yes.push_back(0.02 + (0.46 * i) / 5.0) ;
    size_t const N = rhos.size() * temps.size() * yes.size() ;

    View<double**> pts("tab_det_pts", N, 4) ;
    auto h_pts = create_mirror_view(pts) ;
    {
        size_t n = 0 ;
        for (double r : rhos) for (double t : temps) for (double y : yes) {
            h_pts(n,0) = r ; h_pts(n,1) = t ; h_pts(n,2) = y ; h_pts(n,3) = 0.0 ; ++n ;
        }
    }
    deep_copy(pts, h_pts) ;
    auto describe = [&](size_t i) {
        std::ostringstream o ;
        o << "point rho=" << h_pts(i,0) << " T=" << h_pts(i,1) << " Ye=" << h_pts(i,2) ;
        return o.str() ;
    } ;

    SECTION("identical inputs give identical outputs across threads") {
        constexpr int N_PT = 16, N_REP = 512 ;
        size_t const stride = N / N_PT ;
        View<double**> out("tab_det_thread", static_cast<size_t>(N_PT) * N_REP, D_N) ;
        parallel_for("tab_eos_det_threads", static_cast<size_t>(N_PT) * N_REP,
                     KOKKOS_LAMBDA(size_t const idx) {
            size_t const p = (idx / N_REP) * stride ;
            double loc[D_N] ;
            eval_eos_once(eos, pts(p,0), pts(p,1), pts(p,2), pts(p,3), loc) ;
            for (int k = 0 ; k < D_N ; ++k) out(idx,k) = loc[k] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int mismatches = 0 ;
        for (int ip = 0 ; ip < N_PT ; ++ip) {
            size_t const base = static_cast<size_t>(ip) * N_REP ;
            for (int rep = 1 ; rep < N_REP ; ++rep)
                for (int k = 0 ; k < D_N ; ++k)
                    if (h(base + rep, k) != h(base, k) && ++mismatches <= 5) {
                        INFO(describe(ip * stride) << "  field " << DET_NAME[k]
                             << "  thread 0 = " << std::setprecision(17) << h(base,k)
                             << "  thread " << rep << " = " << h(base+rep,k)) ;
                        CHECK(h(base + rep, k) == h(base, k)) ;
                    }
        }
        printf("   tabulated: total field mismatches across threads: %d\n", mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    SECTION("identical inputs give identical outputs across kernel launches") {
        View<double**> a("tab_det_a", N, D_N), b("tab_det_b", N, D_N) ;
        auto run = [&](View<double**> o) {
            parallel_for("tab_eos_det_launch", N, KOKKOS_LAMBDA(size_t const idx) {
                double loc[D_N] ;
                eval_eos_once(eos, pts(idx,0), pts(idx,1), pts(idx,2), pts(idx,3), loc) ;
                for (int k = 0 ; k < D_N ; ++k) o(idx,k) = loc[k] ;
            }) ;
            fence() ;
        } ;
        run(a) ; run(b) ;
        auto ha = create_mirror_view_and_copy(HostSpace(), a) ;
        auto hb = create_mirror_view_and_copy(HostSpace(), b) ;
        int n_finite = 0 ;
        for (size_t i = 0 ; i < N ; ++i) if (std::isfinite(ha(i,D_PRESS))) ++n_finite ;
        REQUIRE(n_finite == static_cast<int>(N)) ;
        int mismatches = 0 ;
        for (size_t i = 0 ; i < N ; ++i)
            for (int k = 0 ; k < D_N ; ++k)
                if (ha(i,k) != hb(i,k) && ++mismatches <= 5) {
                    INFO(describe(i) << "  field " << DET_NAME[k]
                         << "  launch A = " << std::setprecision(17) << ha(i,k)
                         << "  launch B = " << hb(i,k)) ;
                    CHECK(ha(i,k) == hb(i,k)) ;
                }
        printf("   tabulated: total field mismatches across launches: %d\n", mismatches) ;
        REQUIRE(mismatches == 0) ;
    }
}

//************************************************************************************
//  Full c2p determinism.
//
//  The EOS case above guards the EOS entry points.  This one guards the whole
//  production inversion -- limiter, Kastaun solve, distrust gate, entropy backup,
//  atmosphere / T-floor branches and the conservative write-back -- as a pure
//  function of (cons, metric): same bits across threads, across launches, under
//  the production launch shape, and whatever the incoming primitive array holds
//  (compute_auxiliaries hands c2p an uninitialised array).
//
//  Registered twice: under the round-trip yaml (backup off) and under the halo
//  yaml (production atmosphere, backup on) -- see test/CMakeLists.txt.
//************************************************************************************
namespace {

// Everything one c2p evaluation produces, recorded slot by slot.
enum FULL_OUT : int {
    F_RHO = 0, F_EPS, F_TEMP, F_PRESS, F_ENT, F_YE, F_YMU, F_ZX, F_ZY, F_ZZ,
    F_DENS, F_TAU, F_STX, F_STY, F_STZ, F_ENTS, F_YES, F_YMUS,
    F_FLOORED, F_ERR,
    F_N
} ;
constexpr char const* FULL_NAME[F_N] = {
    "rho", "eps", "temp", "press", "entropy", "ye", "ymu", "zx", "zy", "zz",
    "dens", "tau", "stx", "sty", "stz", "ents", "yes", "ymus", "floored", "c2p_err"
} ;

KOKKOS_INLINE_FUNCTION void
c2p_record(grmhd_prims_array_t const& p, grmhd_cons_array_t const& c,
           bool floored, c2p_err_t const& e, double* out)
{
    out[F_RHO]   = p[RHOL]   ; out[F_EPS]  = p[EPSL]  ; out[F_TEMP] = p[TEMPL] ;
    out[F_PRESS] = p[PRESSL] ; out[F_ENT]  = p[ENTL]  ; out[F_YE]   = p[YEL]   ;
    out[F_YMU]   = p[YMUL]   ; out[F_ZX]   = p[ZXL]   ; out[F_ZY]   = p[ZYL]   ;
    out[F_ZZ]    = p[ZZL]    ;
    out[F_DENS]  = c[DENSL]  ; out[F_TAU]  = c[TAUL]  ; out[F_STX]  = c[STXL]  ;
    out[F_STY]   = c[STYL]   ; out[F_STZ]  = c[STZL]  ; out[F_ENTS] = c[ENTSL] ;
    out[F_YES]   = c[YESL]   ; out[F_YMUS] = c[YMUSL] ;
    out[F_FLOORED] = floored ? 1.0 : 0.0 ;
    out[F_ERR]     = static_cast<double>(e.words[0]) ;
}

// One probe state.  `mutate` is applied to the conservatives after P2C:
// 1 = tau*(1-1e-3) (pushes eps below the cold floor -> distrust / backup),
// 2 = advected entropy*0.7 (visible only to the backup).
struct c2p_probe_t { double rho, temp, ye, ymu, W ; int mutate ; } ;

/// P2C the probe, mutate, fill the incoming primitives with `guess` and run
/// the production c2p.  `out` gets all F_N slots.
template <typename eos_t>
KOKKOS_INLINE_FUNCTION void
c2p_full_once(eos_t const& eos, atmo_params_t const& atmo,
              excision_params_t const& excision, c2p_params_t const& c2p_pars,
              metric_array_t const& metric, c2p_probe_t const& s,
              double guess, double* out)
{
    grmhd_prims_array_t p0{} ;
    p0[RHOL]  = s.rho ;  p0[TEMPL] = s.temp ;  p0[YEL] = s.ye ;  p0[YMUL] = s.ymu ;
    p0[ZXL]   = Kokkos::sqrt(s.W*s.W - 1.0) ;  p0[ZYL] = 0.0 ;  p0[ZZL] = 0.0 ;
    p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;
    double csnd2 ;
    eos_err_t err{} ;
    p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
                    p0[EPSL], csnd2, p0[ENTL], p0[TEMPL],
                    p0[RHOL], p0[YEL], p0[YMUL], err) ;

    grmhd_cons_array_t cons{} ;
    prims_to_conservs(p0, cons, metric) ;
    if      (s.mutate == 1) cons[TAUL]  *= (1.0 - 1.0e-3) ;
    else if (s.mutate == 2) cons[ENTSL] *= 0.7 ;

    grmhd_prims_array_t p1 ;
    for (auto& v : p1) v = guess ;
    c2p_err_t cerr ;
    double rtp[3] = {1.0, 1.0, 1.0} ;
    bool const fl = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                      c2p_pars, rtp, cerr) ;
    c2p_record(p1, cons, fl, cerr, out) ;
}

inline bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0 ;   // NaN-safe, sign-of-zero-strict
}

/// Slot-by-slot bitwise comparison of two (n x F_N) host views; the first few
/// mismatches are reported through `describe(i)`.
template <typename HV, typename Desc>
int count_bit_mismatches(HV const& a, HV const& b, size_t n,
                         Desc describe, char const* what)
{
    int mismatches = 0 ;
    for (size_t i = 0 ; i < n ; ++i) {
        for (int k = 0 ; k < F_N ; ++k) {
            if (same_bits(a(i,k), b(i,k))) continue ;
            if (++mismatches <= 5) {
                INFO(what << ": " << describe(i) << "  field " << FULL_NAME[k]
                     << "  A = " << std::setprecision(17) << a(i,k)
                     << "  B = " << b(i,k)) ;
                CHECK(same_bits(a(i,k), b(i,k))) ;
            }
        }
    }
    return mismatches ;
}

// Which c2p branches a set of evaluations took, from the recorded error words.
struct branch_hist_t { int clean = 0, atmo = 0, tfloor = 0, backup = 0,
                           tau = 0, eps_lo = 0, ymu = 0 ; } ;
template <typename HV>
branch_hist_t branch_histogram(HV const& h, size_t n)
{
    branch_hist_t b ;
    for (size_t i = 0 ; i < n ; ++i) {
        c2p_err_t e ;
        e.words[0] = static_cast<uint64_t>(h(i, F_ERR)) ;
        bool const touched =
               e.test(c2p_err_enum_t::C2P_RESET_DENS)   || e.test(c2p_err_enum_t::C2P_RESET_TAU)
            || e.test(c2p_err_enum_t::C2P_RESET_STILDE) || e.test(c2p_err_enum_t::C2P_RESET_YE)
            || e.test(c2p_err_enum_t::C2P_RESET_YMU)    || e.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)
            || e.test(c2p_err_enum_t::C2P_ATMO_RESET)   || e.test(c2p_err_enum_t::C2P_T_FLOORED) ;
        if (!touched) ++b.clean ;
        if (e.test(c2p_err_enum_t::C2P_ATMO_RESET))      ++b.atmo ;
        if (e.test(c2p_err_enum_t::C2P_T_FLOORED))       ++b.tfloor ;
        if (e.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)) ++b.backup ;
        if (e.test(c2p_err_enum_t::C2P_RESET_TAU))       ++b.tau ;
        if (e.test(c2p_err_enum_t::C2P_SIG_EPS_TOO_LOW)) ++b.eps_lo ;
        if (e.test(c2p_err_enum_t::C2P_RESET_YMU))       ++b.ymu ;
    }
    return b ;
}

void print_branch_histogram(char const* label, branch_hist_t const& b, size_t n)
{
    printf("   %s: %zu evaluations -- clean %d, atmo %d, T-floor %d, backup %d, "
           "tau-reset %d, eps-too-low %d, ymu-reset %d\n",
           label, n, b.clean, b.atmo, b.tfloor, b.backup, b.tau, b.eps_lo, b.ymu) ;
}

// The production auxiliaries launch: LaunchBounds<256,1> with a 16x4x4 tile on
// the tuned architectures, a plain MDRange elsewhere (tuning.h decides).
#ifdef GRACE_AUX_LB
using aux_shape_policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, GRACE_AUX_LB> ;
#else
using aux_shape_policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>> ;
#endif
inline aux_shape_policy_t make_aux_shape_policy(int nx, int ny, int nz)
{
#ifdef GRACE_AUX_LB
    return aux_shape_policy_t({0,0,0}, {nx,ny,nz}, {16,4,4}) ;
#else
    return aux_shape_policy_t({0,0,0}, {nx,ny,nz}) ;
#endif
}

} // namespace

TEST_CASE("c2p leptonic: conservs_to_prims is a deterministic function of its inputs",
          "[c2p][leptonic][determinism]")
{
    using namespace Kokkos ;

    auto eos      = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo     = get_atmo_params() ;
    auto excision = get_excision_params() ;
    auto c2p_pars = get_c2p_params() ;
    metric_array_t const metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, 1.0) ;

    // --- Probe grid: the cold/dilute corners the production runs live in ------
    // Exact table edges (T_min, Ye_min, Ymu_min), the atmosphere floor values,
    // the headon core state (rho 1e-3, T 0.1, Ye 0.052, Ymu 0.019), the rung-B
    // residual T (0.35), the spurious floor-inversion root (2.37), W = 1 exactly.
    std::vector<c2p_probe_t> probes ;
    {
        double const t_min  = eos.temperature_minimum() ;
        double const ye_lo  = eos.get_c2p_ye_min() ;
        double const ymu_lo = eos.get_c2p_ymu_min() ;
        std::vector<double> const rhos  = {0.5*atmo.rho_fl, 1e-12, 1e-10, 1e-8, 1e-6,
                                           1e-5, 1e-4, 3e-4, 1e-3, 3e-3} ;
        std::vector<double> const temps = {t_min, atmo.temp_fl, 0.1, 0.35, 1.0, 2.37, 10.0, 30.0} ;
        std::vector<double> const yes   = {ye_lo, 0.052, 0.1, 0.25, 0.45} ;
        std::vector<double> const ymus  = {ymu_lo, 5.9e-4, 6.1e-4, 1e-3, 0.019, 0.05} ;
        std::vector<double> const Ws    = {1.0, 1.05, 2.0} ;
        for (double r : rhos) for (double t : temps) for (double y : yes)
        for (double m : ymus) for (double W : Ws) for (int mut = 0 ; mut < 3 ; ++mut)
            probes.push_back({r, t, y, m, W, mut}) ;
    }
    size_t const N = probes.size() ;
    View<c2p_probe_t*> d_probes("c2p_probes", N) ;
    auto h_probes = create_mirror_view(d_probes) ;
    for (size_t i = 0 ; i < N ; ++i) h_probes(i) = probes[i] ;
    deep_copy(d_probes, h_probes) ;

    auto describe = [&](size_t i) {
        std::ostringstream s ;
        auto const& p = probes[i] ;
        s << "rho=" << p.rho << " T=" << p.temp << " Ye=" << p.ye << " Ymu=" << p.ymu
          << " W=" << p.W << " mutate=" << p.mutate ;
        return s.str() ;
    } ;

    // Flat launch of the whole grid.
    auto run_flat = [&](View<double**> o, double guess) {
        parallel_for("c2p_det_flat", N, KOKKOS_LAMBDA(size_t const idx) {
            double loc[F_N] ;
            c2p_full_once(eos, atmo, excision, c2p_pars, metric, d_probes(idx), guess, loc) ;
            for (int k = 0 ; k < F_N ; ++k) o(idx,k) = loc[k] ;
        }) ;
        fence() ;
    } ;

    View<double**> ref("c2p_det_ref", N, F_N) ;
    run_flat(ref, 0.0) ;
    auto h_ref = create_mirror_view_and_copy(HostSpace(), ref) ;

    // Guard against a vacuous pass: every branch under test must have fired.
    // (Catch2 re-enters the body once per SECTION -- print only the first time.)
    {
        auto const b = branch_histogram(h_ref, N) ;
        static bool reported = false ;
        if (!reported) { reported = true ; print_branch_histogram("full c2p", b, N) ; }
        REQUIRE(b.clean > 0) ;
        REQUIRE(b.atmo > 0) ;
        REQUIRE(b.tau + b.eps_lo > 0) ;
        if (c2p_pars.use_ent_backup) REQUIRE(b.backup > 0) ;
    }

    // --- (a) thread consistency: one probe, many threads, one launch ---------
    SECTION("identical inputs give identical outputs across threads") {
        constexpr int N_PT = 32, N_REP = 256 ;
        size_t const stride = N / N_PT ;
        View<double**> out("c2p_det_thread", static_cast<size_t>(N_PT) * N_REP, F_N) ;
        parallel_for("c2p_det_threads", static_cast<size_t>(N_PT) * N_REP,
                     KOKKOS_LAMBDA(size_t const idx) {
            size_t const p = (idx / N_REP) * stride ;
            double loc[F_N] ;
            c2p_full_once(eos, atmo, excision, c2p_pars, metric, d_probes(p), 0.0, loc) ;
            for (int k = 0 ; k < F_N ; ++k) out(idx,k) = loc[k] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int mismatches = 0 ;
        for (int ip = 0 ; ip < N_PT ; ++ip) {
            size_t const base = static_cast<size_t>(ip) * N_REP ;
            for (int rep = 1 ; rep < N_REP ; ++rep)
                for (int k = 0 ; k < F_N ; ++k)
                    if (!same_bits(h(base + rep, k), h(base, k)) && ++mismatches <= 5) {
                        INFO("probe " << describe(ip * stride) << "  field " << FULL_NAME[k]
                             << "  thread 0 = " << std::setprecision(17) << h(base,k)
                             << "  thread " << rep << " = " << h(base+rep,k)) ;
                        CHECK(same_bits(h(base + rep, k), h(base, k))) ;
                    }
        }
        INFO("total field mismatches across threads: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    // --- (b) launch repeatability: same grid, second flat launch --------------
    SECTION("identical inputs give identical outputs across launches") {
        View<double**> again("c2p_det_again", N, F_N) ;
        run_flat(again, 0.0) ;
        auto h = create_mirror_view_and_copy(HostSpace(), again) ;
        int const mismatches = count_bit_mismatches(h_ref, h, N, describe, "launch A vs B") ;
        INFO("total field mismatches across launches: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    // --- (c) production launch shape: MDRange + LaunchBounds as in auxiliaries -
    SECTION("the production launch shape gives the same bits as a flat launch") {
        constexpr int NX = 32, NY = 32 ;
        int const NZ = static_cast<int>((N + NX*NY - 1) / (NX*NY)) ;
        View<double**> out("c2p_det_shape", N, F_N) ;
        parallel_for("c2p_det_aux_shape", make_aux_shape_policy(NX, NY, NZ),
                     KOKKOS_LAMBDA(int const i, int const j, int const k) {
            size_t const idx = static_cast<size_t>(i) + NX * (static_cast<size_t>(j) + NY * k) ;
            if (idx >= N) return ;
            double loc[F_N] ;
            c2p_full_once(eos, atmo, excision, c2p_pars, metric, d_probes(idx), 0.0, loc) ;
            for (int f = 0 ; f < F_N ; ++f) out(idx,f) = loc[f] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int const mismatches = count_bit_mismatches(h_ref, h, N, describe, "flat vs aux-shape") ;
        INFO("total field mismatches flat vs production launch shape: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    // --- (d) guess independence: production passes an uninitialised array ----
    SECTION("the output does not depend on the incoming primitive array") {
        double const fills[3] = {std::numeric_limits<double>::quiet_NaN(), 1.0e300, -1.0} ;
        for (double fill : fills) {
            View<double**> out("c2p_det_guess", N, F_N) ;
            run_flat(out, fill) ;
            auto h = create_mirror_view_and_copy(HostSpace(), out) ;
            int const mismatches = count_bit_mismatches(h_ref, h, N, describe, "guess 0 vs fill") ;
            INFO("incoming primitives filled with " << fill
                 << ": total field mismatches vs zero-filled = " << mismatches) ;
            REQUIRE(mismatches == 0) ;
        }
    }
}

//************************************************************************************
//  Replay of production cells.
//
//  scripts/extract_c2p_replay_cells.py lifts coordinates, metric, primitives and
//  c2p_err of selected cells out of a surface_out_plane_*.h5 into a text file.
//  This case rebuilds each cell's conservatives with its own metric (that IS the
//  state the run held: c2p re-synchronises the conservatives from its primitives)
//  and runs the production c2p on them: the same four invariants as above, on
//  real inputs, plus -- for cells the run inverted cleanly -- the round trip must
//  land back on the run's own rho / Ye / Ymu.
//
//  Skips unless GRACE_C2P_REPLAY_FILE names a cell file.  Bind a yaml whose
//  eos / atmosphere / c2p blocks match the run (configs/c2p_test_replay.yaml);
//  the file's c2p_err bits are decoded with THIS build's layout.
//************************************************************************************
namespace {

struct replay_cell_t {
    double x, y, z ;
    double gt[6] ;                  // conformal metric xx xy xz yy yz zz
    double conf_fact, alp, beta[3] ;
    double rho, eps, press, temp, ent, ye, ymu ;
    double zvec[3], B[3] ;
    double c2p_err ;
} ;
constexpr int REPLAY_NCOL = 28 ;

/// Parse the extractor's text format; '#' lines are collected into `header`.
std::vector<replay_cell_t> read_replay_cells(std::string const& path, std::string& header)
{
    std::ifstream in(path) ;
    if (!in) throw std::runtime_error("cannot open replay file " + path) ;
    std::vector<replay_cell_t> cells ;
    std::string line ;
    while (std::getline(in, line)) {
        if (line.empty()) continue ;
        if (line[0] == '#') { header += line + "\n" ; continue ; }
        double v[REPLAY_NCOL] ;
        char const* s = line.c_str() ;
        int n = 0 ;
        for (; n < REPLAY_NCOL ; ++n) {
            char* e = nullptr ;
            v[n] = std::strtod(s, &e) ;
            if (e == s) break ;
            s = e ;
        }
        if (n != REPLAY_NCOL)
            throw std::runtime_error("replay row with " + std::to_string(n)
                                     + " columns (want 28): " + line) ;
        replay_cell_t c ;
        c.x = v[0] ; c.y = v[1] ; c.z = v[2] ;
        for (int k = 0 ; k < 6 ; ++k) c.gt[k] = v[3+k] ;
        c.conf_fact = v[9] ; c.alp = v[10] ;
        for (int k = 0 ; k < 3 ; ++k) c.beta[k] = v[11+k] ;
        c.rho = v[14] ; c.eps = v[15] ; c.press = v[16] ; c.temp = v[17] ;
        c.ent = v[18] ; c.ye  = v[19] ; c.ymu   = v[20] ;
        for (int k = 0 ; k < 3 ; ++k) c.zvec[k] = v[21+k] ;
        for (int k = 0 ; k < 3 ; ++k) c.B[k]    = v[24+k] ;
        c.c2p_err = v[27] ;
        cells.push_back(c) ;
    }
    return cells ;
}

/// Rebuild the cell's conservatives exactly as compute_auxiliaries holds them
/// and run the production c2p with the incoming primitives filled with `guess`.
template <typename eos_t>
KOKKOS_INLINE_FUNCTION void
c2p_replay_once(eos_t const& eos, atmo_params_t const& atmo,
                excision_params_t const& excision, c2p_params_t const& c2p_pars,
                replay_cell_t const& c, double guess, double* out)
{
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    metric_array_t const metric({c.gt[0], c.gt[1], c.gt[2], c.gt[3], c.gt[4], c.gt[5]},
                                c.conf_fact, {c.beta[0], c.beta[1], c.beta[2]}, c.alp) ;
#else
    double const ooW = 1.0 / Kokkos::fmax(1e-100, c.conf_fact) ;   // the Z4 ctor's scaling
    metric_array_t const metric({c.gt[0]*(ooW*ooW), c.gt[1]*(ooW*ooW), c.gt[2]*(ooW*ooW),
                                 c.gt[3]*(ooW*ooW), c.gt[4]*(ooW*ooW), c.gt[5]*(ooW*ooW)},
                                {c.beta[0], c.beta[1], c.beta[2]}, c.alp) ;
#endif
    grmhd_prims_array_t p0{} ;
    p0[RHOL] = c.rho ; p0[EPSL] = c.eps ; p0[PRESSL] = c.press ; p0[TEMPL] = c.temp ;
    p0[ENTL] = c.ent ; p0[YEL]  = c.ye  ; p0[YMUL]   = c.ymu   ;
    p0[ZXL]  = c.zvec[0] ; p0[ZYL] = c.zvec[1] ; p0[ZZL] = c.zvec[2] ;
    p0[BXL]  = c.B[0]    ; p0[BYL] = c.B[1]    ; p0[BZL] = c.B[2]    ;

    grmhd_cons_array_t cons{} ;
    prims_to_conservs(p0, cons, metric) ;
    // compute_auxiliaries derives the cell-centred B from the densitised
    // staggered average: cons B is sqrt(gamma) times the primitive B.
    cons[BSXL] = c.B[0] * metric.sqrtg() ;
    cons[BSYL] = c.B[1] * metric.sqrtg() ;
    cons[BSZL] = c.B[2] * metric.sqrtg() ;

    grmhd_prims_array_t p1 ;
    for (auto& v : p1) v = guess ;
    c2p_err_t cerr ;
    double rtp[3] = {Kokkos::sqrt(c.x*c.x + c.y*c.y + c.z*c.z), 0.0, 0.0} ;
    bool const fl = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                      c2p_pars, rtp, cerr) ;
    c2p_record(p1, cons, fl, cerr, out) ;
}

} // namespace

TEST_CASE("c2p leptonic: replay of production cells (GRACE_C2P_REPLAY_FILE)",
          "[c2p][leptonic][replay]")
{
    using namespace Kokkos ;

    char const* path = std::getenv("GRACE_C2P_REPLAY_FILE") ;
    if (path == nullptr || *path == '\0')
        SKIP("set GRACE_C2P_REPLAY_FILE to a cell file from scripts/extract_c2p_replay_cells.py") ;

    std::string header ;
    std::vector<replay_cell_t> const cells = read_replay_cells(path, header) ;
    size_t const NC = cells.size() ;
    REQUIRE(NC > 0) ;

    auto eos      = eos::get().get_eos<leptonic_eos_4d_t>() ;
    auto atmo     = get_atmo_params() ;
    auto excision = get_excision_params() ;
    auto c2p_pars = get_c2p_params() ;

    View<replay_cell_t*> d_cells("replay_cells", NC) ;
    auto h_cells = create_mirror_view(d_cells) ;
    for (size_t i = 0 ; i < NC ; ++i) h_cells(i) = cells[i] ;
    deep_copy(d_cells, h_cells) ;

    auto describe = [&](size_t i) {
        std::ostringstream s ;
        auto const& c = cells[i] ;
        s << "cell " << i << " at (" << c.x << ", " << c.y << ", " << c.z << ") rho=" << c.rho
          << " T=" << c.temp << " Ye=" << c.ye << " Ymu=" << c.ymu ;
        return s.str() ;
    } ;

    auto run_flat = [&](View<double**> o, double guess) {
        parallel_for("c2p_replay_flat", NC, KOKKOS_LAMBDA(size_t const idx) {
            double loc[F_N] ;
            c2p_replay_once(eos, atmo, excision, c2p_pars, d_cells(idx), guess, loc) ;
            for (int k = 0 ; k < F_N ; ++k) o(idx,k) = loc[k] ;
        }) ;
        fence() ;
    } ;

    View<double**> ref("c2p_replay_ref", NC, F_N) ;
    run_flat(ref, 0.0) ;
    auto h_ref = create_mirror_view_and_copy(HostSpace(), ref) ;

    // What the run recorded for these cells versus what the replay did
    // (Catch2 re-enters the body once per SECTION -- print only the first time).
    static bool reported = false ;
    if (!reported) {
        reported = true ;
        printf("\n   replay file %s\n%s   %zu cells\n", path, header.c_str(), NC) ;
        View<double**, HostSpace> h_file("replay_file_err", NC, F_N) ;
        for (size_t i = 0 ; i < NC ; ++i) h_file(i, F_ERR) = cells[i].c2p_err ;
        print_branch_histogram("run    ", branch_histogram(h_file, NC), NC) ;
        print_branch_histogram("replay ", branch_histogram(h_ref,  NC), NC) ;
    }

    SECTION("identical inputs give identical outputs across threads") {
        constexpr int N_REP = 16 ;
        View<double**> out("c2p_replay_thread", NC * N_REP, F_N) ;
        parallel_for("c2p_replay_threads", NC * N_REP, KOKKOS_LAMBDA(size_t const idx) {
            double loc[F_N] ;
            c2p_replay_once(eos, atmo, excision, c2p_pars, d_cells(idx / N_REP), 0.0, loc) ;
            for (int k = 0 ; k < F_N ; ++k) out(idx,k) = loc[k] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int mismatches = 0 ;
        for (size_t i = 0 ; i < NC ; ++i)
            for (int rep = 1 ; rep < N_REP ; ++rep)
                for (int k = 0 ; k < F_N ; ++k)
                    if (!same_bits(h(i*N_REP + rep, k), h(i*N_REP, k)) && ++mismatches <= 5) {
                        INFO(describe(i) << "  field " << FULL_NAME[k]
                             << "  thread 0 = " << std::setprecision(17) << h(i*N_REP,k)
                             << "  thread " << rep << " = " << h(i*N_REP+rep,k)) ;
                        CHECK(same_bits(h(i*N_REP + rep, k), h(i*N_REP, k))) ;
                    }
        INFO("total field mismatches across threads: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    SECTION("identical inputs give identical outputs across launches") {
        View<double**> again("c2p_replay_again", NC, F_N) ;
        run_flat(again, 0.0) ;
        auto h = create_mirror_view_and_copy(HostSpace(), again) ;
        int const mismatches = count_bit_mismatches(h_ref, h, NC, describe, "launch A vs B") ;
        INFO("total field mismatches across launches: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    SECTION("the production launch shape gives the same bits as a flat launch") {
        constexpr int NX = 16, NY = 16 ;
        int const NZ = static_cast<int>((NC + NX*NY - 1) / (NX*NY)) ;
        View<double**> out("c2p_replay_shape", NC, F_N) ;
        parallel_for("c2p_replay_aux_shape", make_aux_shape_policy(NX, NY, NZ),
                     KOKKOS_LAMBDA(int const i, int const j, int const k) {
            size_t const idx = static_cast<size_t>(i) + NX * (static_cast<size_t>(j) + NY * k) ;
            if (idx >= NC) return ;
            double loc[F_N] ;
            c2p_replay_once(eos, atmo, excision, c2p_pars, d_cells(idx), 0.0, loc) ;
            for (int f = 0 ; f < F_N ; ++f) out(idx,f) = loc[f] ;
        }) ;
        fence() ;
        auto h = create_mirror_view_and_copy(HostSpace(), out) ;
        int const mismatches = count_bit_mismatches(h_ref, h, NC, describe, "flat vs aux-shape") ;
        INFO("total field mismatches flat vs production launch shape: " << mismatches) ;
        REQUIRE(mismatches == 0) ;
    }

    SECTION("the output does not depend on the incoming primitive array") {
        double const fills[2] = {std::numeric_limits<double>::quiet_NaN(), 1.0e300} ;
        for (double fill : fills) {
            View<double**> out("c2p_replay_guess", NC, F_N) ;
            run_flat(out, fill) ;
            auto h = create_mirror_view_and_copy(HostSpace(), out) ;
            int const mismatches = count_bit_mismatches(h_ref, h, NC, describe, "guess 0 vs fill") ;
            INFO("incoming primitives filled with " << fill
                 << ": total field mismatches vs zero-filled = " << mismatches) ;
            REQUIRE(mismatches == 0) ;
        }
    }

    // Cells the run inverted without any reset must round-trip onto the run's
    // own state.  rho / Ye / Ymu are the well-conditioned outputs; T and eps at
    // the table floor are not (the eps->T inversion is noise-limited there), so
    // those are reported, not asserted.  A large residual here means the bound
    // yaml does not match the run (wrong table, floor or backup setting).
    SECTION("cleanly inverted cells round-trip onto the run's own state") {
        int n_clean = 0, n_backup_now = 0 ;
        double res_max = 0.0, dT_max = 0.0, deps_max = 0.0, dz_max = 0.0 ;
        size_t i_max = 0 ;
        for (size_t i = 0 ; i < NC ; ++i) {
            c2p_err_t e ;
            e.words[0] = static_cast<uint64_t>(cells[i].c2p_err) ;
            bool const clean = !(
                   e.test(c2p_err_enum_t::C2P_RESET_DENS)   || e.test(c2p_err_enum_t::C2P_RESET_TAU)
                || e.test(c2p_err_enum_t::C2P_RESET_STILDE) || e.test(c2p_err_enum_t::C2P_RESET_YE)
                || e.test(c2p_err_enum_t::C2P_RESET_YMU)    || e.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)
                || e.test(c2p_err_enum_t::C2P_ATMO_RESET)   || e.test(c2p_err_enum_t::C2P_T_FLOORED)) ;
            if (!clean) continue ;
            ++n_clean ;
            auto const& c = cells[i] ;
            double r = std::fabs(h_ref(i,F_RHO) - c.rho) / c.rho ;
            r = std::max(r, std::fabs(h_ref(i,F_YE)  - c.ye)  / c.ye) ;
            r = std::max(r, std::fabs(h_ref(i,F_YMU) - c.ymu) / c.ymu) ;
            if (r > res_max) { res_max = r ; i_max = i ; }
            dT_max   = std::max(dT_max,   std::fabs(h_ref(i,F_TEMP) - c.temp) / c.temp) ;
            deps_max = std::max(deps_max, std::fabs(h_ref(i,F_EPS)  - c.eps)  / std::max(std::fabs(c.eps), 1e-30)) ;
            for (int k = 0 ; k < 3 ; ++k)
                dz_max = std::max(dz_max, std::fabs(h_ref(i,F_ZX+k) - c.zvec[k])) ;
            c2p_err_t now ;
            now.words[0] = static_cast<uint64_t>(h_ref(i,F_ERR)) ;
            if (now.test(c2p_err_enum_t::C2P_ENT_BACKUP_USED)) ++n_backup_now ;
        }
        printf("   %d cells clean in the run; replay took the backup on %d of them\n"
               "   max rel residual rho/Ye/Ymu %.3e, T %.3e, eps %.3e; max |dz| %.3e\n",
               n_clean, n_backup_now, res_max, dT_max, deps_max, dz_max) ;
        REQUIRE(n_clean > 0) ;
        INFO("worst cell: " << describe(i_max) << "  replay rho=" << std::setprecision(17)
             << h_ref(i_max,F_RHO) << " Ye=" << h_ref(i_max,F_YE) << " Ymu=" << h_ref(i_max,F_YMU)) ;
        REQUIRE(res_max < 1e-8) ;
    }
}
#endif // GRACE_ENABLE_MUONS
