/**
 * @file test_lbm_geodesic.cpp
 * @brief Backward null-geodesic integration of the LBM geometry pass, on the
 *        analytic Kerr-Schild metric and on a hand-built grid sampled from it.
 *
 * Checks: the Killing energy nu (alpha - beta_i v^i) is conserved along each
 * ray on a static background, the null norm gamma_ij v^i v^j = 1 is kept, an
 * ingoing radial ray in ingoing Kerr-Schild coordinates has dr/dt = -1 exactly,
 * the flat-space departure point is x - v dt, and the grid sampler (trilinear
 * metric + finite-difference derivatives) agrees with the analytic field.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm_geodesic.hh>
#include <grace/physics/id/kerr_schild_subexpressions.hh>

#include <cmath>

using Catch::Matchers::WithinAbs;
using namespace grace ;

namespace {

// Kerr-Schild (a = 0) metric and its derivatives (centred differences of the
// analytic functions, h = 1e-5) at a physical point.
void ks_adm(double const x[3], double g[6], double& alp, double beta[3], double K[6])
{
    double gi[6] ;
    kerr_schild_adm_metric(x, 0.0, 0.0, &g[0],&g[1],&g[2],&g[3],&g[4],&g[5],
                           &gi[0],&gi[1],&gi[2],&gi[3],&gi[4],&gi[5], &alp, &beta[0],&beta[1],&beta[2],
                           &K[0],&K[1],&K[2],&K[3],&K[4],&K[5]) ;
}

// Write an ADM metric into the state slots of this build's metric evolution:
// Cowling stores gamma_ij, K_ij; Z4 stores W = det^{-1/6}, gt = W^2 gamma,
// Khat = tr K (Theta = 0), At = W^2 (K - gamma tr K / 3).
template < typename view_t >
void set_metric_slots(view_t& hs, int const i, int const j, int const k,
                      double const g[6], double const alp, double const beta[3], double const K[6])
{
    hs(i,j,k,ALP_,0) = alp ; for ( int a = 0; a < 3; ++a ) hs(i,j,k,BETAX_+a,0) = beta[a] ;
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
    for ( int c = 0; c < 6; ++c ) { hs(i,j,k,GXX_+c,0) = g[c] ; hs(i,j,k,KXX_+c,0) = K[c] ; }
    #else
    metric_array_t const m{ {g[0],g[1],g[2],g[3],g[4],g[5]}, {beta[0],beta[1],beta[2]}, alp } ;
    double const det = g[0]*(g[3]*g[5] - g[4]*g[4]) - g[1]*(g[1]*g[5] - g[4]*g[2]) + g[2]*(g[1]*g[4] - g[3]*g[2]) ;
    double const W = std::pow(det, -1.0/6.0) ;
    double trK = 0. ;
    for ( int c = 0; c < 6; ++c ) trK += ((c==0||c==3||c==5) ? 1.0 : 2.0) * m.invgamma(c) * K[c] ;
    hs(i,j,k,CHI_,0) = W ; hs(i,j,k,THETA_,0) = 0. ; hs(i,j,k,KHAT_,0) = trK ;
    for ( int a = 0; a < 3; ++a ) { hs(i,j,k,GAMMATX_+a,0) = 0. ; hs(i,j,k,BDRIVERX_+a,0) = 0. ; }
    for ( int c = 0; c < 6; ++c ) { hs(i,j,k,GTXX_+c,0) = W*W*g[c] ; hs(i,j,k,ATXX_+c,0) = W*W*(K[c] - g[c]*trK/3.0) ; }
    #endif
}

struct analytic_field_t {
    double xc[3] ;
    bool sample(int64_t, int, int, int, double const disp[3], lbm::metric_sample_t& s) const
    {
        double const x[3] = { xc[0]+disp[0], xc[1]+disp[1], xc[2]+disp[2] } ;
        ks_adm(x, s.g, s.alp, s.beta, s.K) ;
        metric_array_t const m{ {s.g[0],s.g[1],s.g[2],s.g[3],s.g[4],s.g[5]}, {s.beta[0],s.beta[1],s.beta[2]}, s.alp } ;
        for ( int c = 0; c < 6; ++c ) s.ginv[c] = m.invgamma(c) ;
        double const h = 1e-5 ;
        for ( int a = 0; a < 3; ++a ) {
            double xp[3] = {x[0],x[1],x[2]}, xm[3] = {x[0],x[1],x[2]} ; xp[a] += h ; xm[a] -= h ;
            double gp[6], gm[6], ap, am, bp[3], bm[3], Kp[6], Km[6] ;
            ks_adm(xp, gp, ap, bp, Kp) ; ks_adm(xm, gm, am, bm, Km) ;
            s.dalp[a] = (ap - am)/(2*h) ;
            for ( int i = 0; i < 3; ++i ) s.dbeta[3*a+i]  = (bp[i] - bm[i])/(2*h) ;
            for ( int c = 0; c < 6; ++c ) s.dgamma[6*a+c] = (gp[c] - gm[c])/(2*h) ;
        }
        return true ;
    }
} ;

struct flat_field_t {
    bool sample(int64_t, int, int, int, double const*, lbm::metric_sample_t& s) const
    {
        for ( int c = 0; c < 6; ++c ) { s.g[c] = s.ginv[c] = (c==0||c==3||c==5) ? 1.0 : 0.0 ; s.K[c] = 0. ; }
        s.alp = 1. ; for ( int a = 0; a < 3; ++a ) { s.beta[a] = 0. ; s.dalp[a] = 0. ; }
        for ( int m = 0; m < 9; ++m ) s.dbeta[m] = 0. ;
        for ( int m = 0; m < 18; ++m ) s.dgamma[m] = 0. ;
        return true ;
    }
} ;

double killing_energy(lbm::metric_sample_t const& s, double const v[3], double const nu)
{
    // beta_i v^i = gamma_ij beta^j v^i
    double bv = 0. ;
    for ( int i = 0; i < 3; ++i ) for ( int j = 0; j < 3; ++j ) bv += s.g[lbm::sym6(i,j)]*s.beta[j]*v[i] ;
    return nu*(s.alp - bv) ;
}
double gamma_norm(lbm::metric_sample_t const& s, double const v[3])
{
    double n = 0. ;
    for ( int i = 0; i < 3; ++i ) for ( int j = 0; j < 3; ++j ) n += s.g[lbm::sym6(i,j)]*v[i]*v[j] ;
    return n ;
}
// unit (gamma-norm) Eulerian velocity along a coordinate direction
void unit_velocity(lbm::metric_sample_t const& s, double const dir[3], double v[3])
{
    double const n = std::sqrt(gamma_norm(s, dir)) ;
    for ( int a = 0; a < 3; ++a ) v[a] = dir[a]/n ;
}
}

TEST_CASE("LBM geodesic: flat space departure point is x - v dt", "[lbm][geodesic]")
{
    flat_field_t const fld ;
    double const xc[3] = {0.,0.,0.}, v0[3] = {0.36, 0.48, 0.80}, dt = 0.05625 ;
    auto const r = lbm::integrate_backward(fld, 0, 0,0,0, xc, v0, dt, 1e-10, lbm::excision_t{}) ;
    REQUIRE(r.flag == lbm::GEOM_OK) ;
    for ( int a = 0; a < 3; ++a ) {
        REQUIRE_THAT(r.disp[a], WithinAbs(-v0[a]*dt, 1e-15)) ;
        REQUIRE(r.v[a] == v0[a]) ;
    }
    REQUIRE(r.S == 1.0) ;
}

TEST_CASE("LBM geodesic: Killing energy, null norm and radial speeds on Kerr-Schild (M = 1)", "[lbm][geodesic]")
{
    analytic_field_t fld{ {1.0, -0.5, 6.0} } ;
    double const dt = 0.5 ;
    lbm::metric_sample_t s0 ; double const zero[3] = {0.,0.,0.} ;
    fld.sample(0,0,0,0, zero, s0) ;
    double const dirs[5][3] = { {1,0,0}, {0,1,0}, {0,0,-1}, {0.6,0.0,0.8}, {-0.3,0.5,-0.4} } ;
    for ( auto const& d : dirs ) {
        double v0[3] ; unit_velocity(s0, d, v0) ;
        auto const r = lbm::integrate_backward(fld, 0, 0,0,0, fld.xc, v0, dt, 1e-10, lbm::excision_t{}) ;
        REQUIRE(r.flag == lbm::GEOM_OK) ;
        lbm::metric_sample_t s1 ; fld.sample(0,0,0,0, r.disp, s1) ;
        double const nu1 = 1.0/r.S ;
        REQUIRE_THAT(killing_energy(s1, r.v, nu1), WithinAbs(killing_energy(s0, v0, 1.0), 1e-7)) ;
        REQUIRE_THAT(gamma_norm(s1, r.v), WithinAbs(1.0, 1e-8)) ;
        REQUIRE(r.S > 0.) ;
    }

    // radial rays on the z axis: ingoing dr/dt = -1 exactly in ingoing Kerr-Schild,
    // so going back in time by dt the ingoing photon was dt further out
    analytic_field_t axis{ {0.0, 0.0, 6.0} } ;
    lbm::metric_sample_t sa ; axis.sample(0,0,0,0, zero, sa) ;
    double vin[3] ; { double const dn[3] = {0,0,-1} ; unit_velocity(sa, dn, vin) ; }
    auto const rin = lbm::integrate_backward(axis, 0, 0,0,0, axis.xc, vin, dt, 1e-10, lbm::excision_t{}) ;
    REQUIRE_THAT(rin.disp[2], WithinAbs(+dt, 1e-8)) ;
    REQUIRE_THAT(rin.disp[0], WithinAbs(0.0, 1e-12)) ;
    REQUIRE_THAT(rin.disp[1], WithinAbs(0.0, 1e-12)) ;
    // outgoing: dr/dt = (1 - 2M/r)/(1 + 2M/r); integrate that 1-D equation backward with fine RK4
    double vout[3] ; { double const up[3] = {0,0,1} ; unit_velocity(sa, up, vout) ; }
    auto const rout = lbm::integrate_backward(axis, 0, 0,0,0, axis.xc, vout, dt, 1e-10, lbm::excision_t{}) ;
    double r = 6.0 ; int const nsub = 20000 ; double const h = -dt/nsub ;
    auto f = [](double rr) { return (1.0 - 2.0/rr)/(1.0 + 2.0/rr) ; } ;
    for ( int n = 0; n < nsub; ++n ) {
        double const k1 = f(r), k2 = f(r + 0.5*h*k1), k3 = f(r + 0.5*h*k2), k4 = f(r + h*k3) ;
        r += h*(k1 + 2*k2 + 2*k3 + k4)/6.0 ;
    }
    REQUIRE_THAT(6.0 + rout.disp[2], WithinAbs(r, 1e-7)) ;
}

TEST_CASE("LBM geodesic: the grid sampler agrees with the analytic field", "[lbm][geodesic]")
{
    // padded quadrant of 16^3 cells + 4 ghosts, dx = 0.25, centred at (1,-0.5,6): r >= 3.5 > 2M
    constexpr int N = 16, NGZ = 4, NP = N + 2*NGZ ;
    double const h = 0.25, x0[3] = {1.0 - 0.5*N*h, -0.5 - 0.5*N*h, 6.0 - 0.5*N*h} ;
    var_array_t state("state", NP,NP,NP, N_EVOL_VARS, 1), aux("aux", NP,NP,NP, N_AUX_VARS, 1) ;
    scalar_array_t<GRACE_NSPACEDIM> dx("dx", GRACE_NSPACEDIM, 1), idx("idx", GRACE_NSPACEDIM, 1) ;
    {
        auto hs = Kokkos::create_mirror_view(state) ;
        for ( int k = 0; k < NP; ++k ) for ( int j = 0; j < NP; ++j ) for ( int i = 0; i < NP; ++i ) {
            double const x[3] = { x0[0] + (i-NGZ+0.5)*h, x0[1] + (j-NGZ+0.5)*h, x0[2] + (k-NGZ+0.5)*h } ;
            double g[6], alp, beta[3], K[6] ; ks_adm(x, g, alp, beta, K) ;
            set_metric_slots(hs, i,j,k, g, alp, beta, K) ;
        }
        Kokkos::deep_copy(state, hs) ;
        auto hd = Kokkos::create_mirror_view(dx) ; auto hi = Kokkos::create_mirror_view(idx) ;
        for ( int a = 0; a < 3; ++a ) { hd(a,0) = h ; hi(a,0) = 1.0/h ; }
        Kokkos::deep_copy(dx, hd) ; Kokkos::deep_copy(idx, hi) ;
    }
    lbm::compute_metric_derivatives(state, aux, idx, NP, NP, NP, 1) ;
    lbm::metric_field_t grid{ state, aux, dx, {NP,NP,NP} } ;

    int const ic = NGZ + N/2, jc = NGZ + N/2, kc = NGZ + N/2 ;   // cell centre at x0 + (N/2+0.5) h
    double const xc[3] = { x0[0] + (N/2+0.5)*h, x0[1] + (N/2+0.5)*h, x0[2] + (N/2+0.5)*h } ;
    analytic_field_t ana{ {xc[0], xc[1], xc[2]} } ;
    lbm::metric_sample_t s0 ; double const zero[3] = {0.,0.,0.} ; ana.sample(0,0,0,0, zero, s0) ;
    double const dt = 0.9*h ;
    double const dirs[4][3] = { {1,0,0}, {0,0,-1}, {0.6,0.0,0.8}, {-0.3,0.5,-0.4} } ;
    double worst = 0. ;
    for ( auto const& d : dirs ) {
        double v0[3] ; unit_velocity(s0, d, v0) ;
        auto const ra = lbm::integrate_backward(ana,  0, ic,jc,kc, xc, v0, dt, 1e-10, lbm::excision_t{}) ;
        auto const rg = lbm::integrate_backward(grid, 0, ic,jc,kc, xc, v0, dt, 1e-10, lbm::excision_t{}) ;
        REQUIRE(rg.flag == lbm::GEOM_OK) ;
        for ( int a = 0; a < 3; ++a ) {
            worst = std::fmax(worst, std::fabs(rg.disp[a] - ra.disp[a])/dt) ;
            worst = std::fmax(worst, std::fabs(rg.v[a] - ra.v[a])) ;
        }
        worst = std::fmax(worst, std::fabs(rg.S - ra.S)) ;
    }
    // trilinear metric between cells of dx = 0.25 at r ~ 6M: second-order interpolation error
    REQUIRE(worst < 1e-3) ;
}
#endif // GRACE_ENABLE_LBM
