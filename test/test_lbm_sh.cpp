/**
 * @file test_lbm_sh.cpp
 * @brief The Lebedev-5 streaming quadrature and the l <= 2 spherical-harmonic
 *        fit that carries the geodesic map: orthonormality of the basis under
 *        the 14-point quadrature, and exact reproduction of the flat
 *        straight-line map at all directions of the compiled-in stencil.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm_stencil.hh>
#include <grace/physics/lbm_geometry.hh>

#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using grace::lbm::sh_ncoef;

TEST_CASE("LBM SH: Lebedev-5 quadrature makes the l <= 2 real harmonics orthonormal", "[lbm][sh]")
{
    auto const qd = grace::lbm::load_quadrature(std::string(GRACE_LBM_DATA_DIR) + "/Lebedev5") ;
    REQUIRE(qd.n == 14) ;
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, qd.w) ;
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, qd.c) ;
    double G[sh_ncoef][sh_ncoef] = {} ;
    for ( int d = 0; d < qd.n; ++d ) {
        double Y[sh_ncoef] ;
        grace::lbm::sh_basis(c(d,0), c(d,1), c(d,2), Y) ;
        for ( int i = 0; i < sh_ncoef; ++i ) for ( int j = 0; j < sh_ncoef; ++j )
            G[i][j] += 4.0*M_PI*w(d)*Y[i]*Y[j] ;
    }
    for ( int i = 0; i < sh_ncoef; ++i ) for ( int j = 0; j < sh_ncoef; ++j )
        REQUIRE_THAT(G[i][j], WithinAbs(i==j ? 1.0 : 0.0, 1e-12)) ;
}

TEST_CASE("LBM SH: the fit of the flat map x - n dt is exact at every stencil direction", "[lbm][sh]")
{
    auto const qd = grace::lbm::load_quadrature(std::string(GRACE_LBM_DATA_DIR) + "/Lebedev5") ;
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, qd.w) ;
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, qd.c) ;
    auto const st = grace::lbm::load_stencil(GRACE_LBM_DATA_DIR) ;
    auto sc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;

    double const x0[3] = {0.37, -1.25, 2.0}, dt = 0.05625 ;
    // the 7 fitted quantities: 1/s (=1 in flat space), departure point, departure direction
    std::vector<double> f(qd.n) ;
    double coef[7][sh_ncoef] ;
    for ( int d = 0; d < qd.n; ++d ) f[d] = 1.0 ;
    grace::lbm::sh_fit(qd.n, w, c, f.data(), coef[0]) ;
    for ( int k = 0; k < 3; ++k ) {
        for ( int d = 0; d < qd.n; ++d ) f[d] = x0[k] - c(d,k)*dt ;
        grace::lbm::sh_fit(qd.n, w, c, f.data(), coef[1+k]) ;
        for ( int d = 0; d < qd.n; ++d ) f[d] = c(d,k) ;
        grace::lbm::sh_fit(qd.n, w, c, f.data(), coef[4+k]) ;
    }
    double max_err = 0. ;
    for ( int d = 0; d < st.ndir; ++d ) {
        double Y[sh_ncoef] ;
        grace::lbm::sh_basis(sc(d,0), sc(d,1), sc(d,2), Y) ;
        max_err = std::fmax(max_err, std::fabs(grace::lbm::sh_eval(coef[0], Y) - 1.0)) ;
        for ( int k = 0; k < 3; ++k ) {
            max_err = std::fmax(max_err, std::fabs(grace::lbm::sh_eval(coef[1+k], Y) - (x0[k] - sc(d,k)*dt))) ;
            max_err = std::fmax(max_err, std::fabs(grace::lbm::sh_eval(coef[4+k], Y) - sc(d,k))) ;
        }
    }
    REQUIRE(max_err < 1e-13) ;
}
#endif // GRACE_ENABLE_LBM
