/**
 * @file test_lbm_moments.cpp
 * @brief Moments <-> intensities round trip for the LBM initial-data ansatz.
 *
 * The von Mises-Fisher distribution I(n) = E exp(sigma n.f^ - ln(sinh sigma/
 * sigma)) integrates to E over the sphere and has flux ratio
 * |F|/E = coth sigma - 1/sigma.  sigma_of_relative_flux inverts that, so
 * (E, f) -> I_d -> quadrature moments must give back (E, f) up to the
 * quadrature error of a smooth integrand.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm.hh>

#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("LBM: sigma_of_relative_flux inverts flux_of_sigma", "[lbm][moments]")
{
    for ( double f : {1e-6, 0.05, 0.3, 0.5, 0.7, 0.9, 0.95} ) {
        double const s = grace::lbm::sigma_of_relative_flux(f) ;
        REQUIRE_THAT(grace::lbm::flux_of_sigma(s), WithinAbs(f, 1e-10)) ;
    }
    REQUIRE(grace::lbm::sigma_of_relative_flux(0.0) == 0.0) ;
    // beyond the resolvable range the inversion saturates rather than blowing up
    REQUIRE(std::isfinite(grace::lbm::sigma_of_relative_flux(0.999))) ;
    REQUIRE(std::isfinite(grace::lbm::sigma_of_relative_flux(1.0))) ;
}

TEST_CASE("LBM: (E,F) -> populations -> (E,F) round trip on the stencil", "[lbm][moments]")
{
    auto st = grace::lbm::load_stencil(GRACE_LBM_DATA_DIR) ;
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.w) ;
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;

    double const E0 = 2.5 ;
    // flux along an off-axis direction so no stencil symmetry helps
    double const fh[3] = { 0.36, 0.48, 0.80 } ;   // unit
    for ( double f : {0.0, 0.2, 0.5, 0.8} ) {
        double const sigma = grace::lbm::sigma_of_relative_flux(f) ;
        double E = 0., F[3] = {0,0,0} ;
        for ( int d = 0; d < st.ndir; ++d ) {
            double const ndotf = c(d,0)*fh[0] + c(d,1)*fh[1] + c(d,2)*fh[2] ;
            double const I = grace::lbm::vmf_intensity(sigma, E0, ndotf) ;
            E += w(d)*I ;
            for ( int i = 0; i < 3; ++i ) F[i] += w(d)*I*c(d,i) ;
        }
        REQUIRE_THAT(E, WithinRel(E0, 1e-6)) ;
        double const Fn = std::sqrt(F[0]*F[0] + F[1]*F[1] + F[2]*F[2]) ;
        REQUIRE_THAT(Fn/E, WithinAbs(f, 1e-6)) ;
        if ( f > 0 ) for ( int i = 0; i < 3; ++i ) REQUIRE_THAT(F[i]/Fn, WithinAbs(fh[i], 1e-6)) ;
    }
}
#endif // GRACE_ENABLE_LBM
