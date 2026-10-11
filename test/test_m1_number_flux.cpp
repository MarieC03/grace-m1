/**
 * @file test_m1_number_flux.cpp
 * @brief The M1 number flux against its definition and against the energy flux, with sqrt(gamma) != 1.
 *        Regression for the extra sqrt(gamma) that moved neutrino number sqrt(gamma) times faster
 *        than energy in curved space (flat-space tests cannot see it).
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/physics/m1_number_flux.hh>

#include <array>

using Catch::Matchers::WithinRel ;

TEST_CASE("M1 number flux equals sqrt(gamma) alpha n (W (v - beta/alpha) + H/J)", "[m1][number_flux]")
{
    // sqrtg, alp, beta, Gamma, W, v, H, J, n: generic states, incl. a moving fluid
    std::array<std::array<double,9>,4> const states {{
        {1.0, 1.0,  0.0, 1.0, 1.0,   0.0,  0.7, 1.0, 2.0},
        {1.6, 0.87, 0.21, 1.13, 1.05, 0.3, -0.4, 0.8, 0.5},
        {2.4, 0.71, -0.05, 0.92, 1.4, -0.69, 0.2, 1.7, 3.0e-9},
        {1.2, 0.95, 0.33, 1.5, 2.3, 0.89, 0.9, 1.1, 7.0},
    }} ;
    for ( auto const& s : states ) {
        double const sqrtg = s[0], alp = s[1], beta = s[2], Gamma = s[3], W = s[4], v = s[5], H = s[6], J = s[7], n = s[8] ;
        double const N = sqrtg * Gamma * n ;                    // densitised (evolved) number
        double const expected = sqrtg * alp * n * ( W * (v - beta/alp) + H/J ) ;
        CHECK_THAT( grace::m1_number_flux(N, alp, beta, Gamma, W, v, H, J), WithinRel(expected, 1e-14) ) ;
    }
}

TEST_CASE("M1 number flux carries N/E times the energy flux for radiation at rest w.r.t. the normal observer", "[m1][number_flux]")
{
    // v = 0: W = Gamma = 1, J = E, H^i = F^i.  Energy flux sqrt(gamma) (alpha F^i - beta^i E).
    double const E = 0.3, F = 0.29, ratio = 0.04 ;                 // primitive E, F^i; N/E
    for ( double const sqrtg : {1.0, 1.25, 1.6, 2.4} ) {
        for ( double const beta : {0.0, 0.2, -0.33} ) {
            double const alp = 0.8 ;
            double const f_E = sqrtg * ( alp * F - beta * E ) ;
            double const N   = ratio * sqrtg * E ;
            CHECK_THAT( grace::m1_number_flux(N, alp, beta, 1.0, 1.0, 0.0, F, E), WithinRel(ratio * f_E, 1e-14) ) ;
        }
    }
}
