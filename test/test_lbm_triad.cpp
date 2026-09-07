/**
 * @file test_lbm_triad.cpp
 * @brief The Eulerian triad, null coordinate speed and Killing energy density
 *        of the LBM geometry helpers (lbm_geometry.hh).
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm_geometry.hh>

#include <array>
#include <cmath>

using Catch::Matchers::WithinAbs;
using grace::lbm::triad_t;

namespace {
// A generic SPD 3-metric: gamma = A A^T + I with a non-trivial A.
grace::metric_array_t curved_metric()
{
    double const A[3][3] = { {0.9, 0.3, -0.2}, {0.1, 1.2, 0.4}, {-0.5, 0.2, 0.7} } ;
    double g[3][3] ;
    for ( int i = 0; i < 3; ++i ) for ( int j = 0; j < 3; ++j ) {
        g[i][j] = (i==j) ? 1.0 : 0.0 ;
        for ( int k = 0; k < 3; ++k ) g[i][j] += A[i][k]*A[j][k] ;
    }
    return grace::metric_array_t{ {g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2]},
                                  {0.1, -0.2, 0.05}, 0.8 } ;
}
}

TEST_CASE("LBM triad: gamma-orthonormal legs, exact inverse, gamma^ij from the identity tensor", "[lbm][triad]")
{
    auto const m = curved_metric() ;
    triad_t const tr(m) ;
    double const gam[3][3] = { {m.gamma(0), m.gamma(1), m.gamma(2)},
                               {m.gamma(1), m.gamma(3), m.gamma(4)},
                               {m.gamma(2), m.gamma(4), m.gamma(5)} } ;
    for ( int a = 0; a < 3; ++a )
    for ( int b = 0; b < 3; ++b ) {
        double s = 0., id = 0. ;
        for ( int i = 0; i < 3; ++i ) {
            id += tr.E[a][i]*tr.e[i][b] ;
            for ( int j = 0; j < 3; ++j ) s += tr.e[i][a]*tr.e[j][b]*gam[i][j] ;
        }
        REQUIRE_THAT(s,  WithinAbs(a==b ? 1.0 : 0.0, 1e-13)) ;
        REQUIRE_THAT(id, WithinAbs(a==b ? 1.0 : 0.0, 1e-13)) ;
    }
    // P^ab = delta^ab rotates to gamma^ij
    double const P[6] = {1,0,0,1,0,1} ; double Pu[6] ;
    tr.to_coord_sym(P, Pu) ;
    for ( int c = 0; c < 6; ++c ) REQUIRE_THAT(Pu[c], WithinAbs(m.invgamma(c), 1e-12)) ;
    // a triad-frame unit vector has unit gamma-norm in coordinates
    double const cvec[3] = {0.6, 0.0, 0.8} ; double v[3], back[3] ;
    tr.to_coord(cvec, v) ; tr.to_triad(v, back) ;
    double n2 = 0. ;
    for ( int i = 0; i < 3; ++i ) for ( int j = 0; j < 3; ++j ) n2 += gam[i][j]*v[i]*v[j] ;
    REQUIRE_THAT(n2, WithinAbs(1.0, 1e-13)) ;
    for ( int a = 0; a < 3; ++a ) REQUIRE_THAT(back[a], WithinAbs(cvec[a], 1e-13)) ;
}

TEST_CASE("LBM triad: the flat triad is the identity bit-exactly", "[lbm][triad]")
{
    grace::metric_array_t const m{ {1,0,0,1,0,1}, {0,0,0}, 1.0 } ;
    triad_t const tr(m) ;
    for ( int i = 0; i < 3; ++i ) for ( int a = 0; a < 3; ++a ) {
        REQUIRE(tr.e[i][a] == (i==a ? 1.0 : 0.0)) ;
        REQUIRE(tr.E[a][i] == (i==a ? 1.0 : 0.0)) ;
    }
    double const c[3] = {0.123456789, -0.98765, 0.31415} ; double v[3], P[6] = {1.1,0.2,-0.3,0.7,0.05,2.2}, Pu[6] ;
    tr.to_coord(c, v) ; tr.to_coord_sym(P, Pu) ;
    for ( int i = 0; i < 3; ++i ) REQUIRE(v[i] == c[i]) ;
    for ( int k = 0; k < 6; ++k ) REQUIRE(Pu[k] == P[k]) ;
}

TEST_CASE("LBM geometry: null coordinate speed and Killing energy density", "[lbm][triad]")
{
    grace::metric_array_t const flat{ {1,0,0,1,0,1}, {0,0,0}, 1.0 } ;
    REQUIRE(grace::lbm::null_coordinate_speed(flat) == 1.0) ;
    double const Fu[3] = {0.1, 0.2, 0.3} ;
    REQUIRE(grace::lbm::killing_energy_density(flat, 2.0, Fu) == 2.0) ;

    // diagonal metric: alpha sqrt(gamma^ii) + |beta^i| = (0.55, 0.6, 1.0)
    grace::metric_array_t const diag{ {4,0,0,1,0,0.25}, {0.3,-0.1,0}, 0.5 } ;
    REQUIRE_THAT(grace::lbm::null_coordinate_speed(diag), WithinAbs(1.0, 1e-14)) ;
    // sqrt(gamma) = 1, alpha E - beta_i F^i with beta_i = gamma_ij beta^j = (1.2, -0.1, 0)
    double const F2[3] = {1.0, 2.0, 3.0} ;
    REQUIRE_THAT(grace::lbm::killing_energy_density(diag, 2.0, F2), WithinAbs(0.5*2.0 - (1.2*1.0 - 0.1*2.0), 1e-14)) ;
}
#endif // GRACE_ENABLE_LBM
