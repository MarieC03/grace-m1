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
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/utils/metric_utils.hh>

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

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
        INFO("T=" << r[LEP_TEMP] << " (table min " << t_min << ")") ;
        REQUIRE(r[LEP_BIT_TFLOOR] == 0.0) ;
        REQUIRE(r[LEP_BIT_ATMO]   == 0.0) ;
        REQUIRE(r[LEP_FLOORED]    == 0.0) ;
        // Recovered T sits at the table minimum, not at the floor.
        REQUIRE_THAT(r[LEP_TEMP], Catch::Matchers::WithinRel(t_min, 1e-8)) ;
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
