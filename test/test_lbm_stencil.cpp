/**
 * @file test_lbm_stencil.cpp
 * @brief Quadrature identities of the compiled-in LBM direction stencil.
 *
 * The Lebedev tables are exact for real spherical harmonics up to their
 * order, so on the unit sphere with weights normalised to 1:
 *   sum_d w_d                 = 1
 *   sum_d w_d c_i             = 0
 *   sum_d w_d c_i c_j         = delta_ij / 3
 *   sum_d w_d c_i c_j c_k c_l = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 15
 * These are what the moments E, F^i, P^ij and the closed-form collision rely
 * on.  Also checked: every direction is a unit vector, and the set is closed
 * under each coordinate reflection -- the property the (not yet implemented)
 * reflection parity of the populations will need.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm_stencil.hh>

#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {
struct host_stencil {
    int n ;
    Kokkos::View<double*,  Kokkos::HostSpace> w ;
    Kokkos::View<double**, Kokkos::HostSpace> c ;
    host_stencil() {
        auto st = grace::lbm::load_stencil(GRACE_LBM_DATA_DIR) ;
        n = st.ndir ;
        w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.w) ;
        c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
    }
} ;
}

TEST_CASE("LBM stencil: table loads with the compiled-in direction count", "[lbm][stencil]")
{
    host_stencil hs ;
    REQUIRE(hs.n == GRACE_LBM_NDIR) ;
    REQUIRE(static_cast<int>(hs.w.extent(0)) == GRACE_LBM_NDIR) ;
    for ( int d = 0; d < hs.n; ++d ) {
        double const n2 = hs.c(d,0)*hs.c(d,0) + hs.c(d,1)*hs.c(d,1) + hs.c(d,2)*hs.c(d,2) ;
        REQUIRE_THAT(n2, WithinAbs(1.0, 1e-12)) ;
        REQUIRE(hs.w(d) > 0.0) ;
    }
}

TEST_CASE("LBM stencil: quadrature is exact for moments up to fourth order", "[lbm][stencil]")
{
    host_stencil hs ;
    double m0 = 0., m1[3] = {0,0,0}, m2[3][3] = {}, m4[3][3][3][3] = {} ;
    for ( int d = 0; d < hs.n; ++d ) {
        double const w = hs.w(d) ;
        double const c[3] = { hs.c(d,0), hs.c(d,1), hs.c(d,2) } ;
        m0 += w ;
        for ( int i = 0; i < 3; ++i ) {
            m1[i] += w*c[i] ;
            for ( int j = 0; j < 3; ++j ) {
                m2[i][j] += w*c[i]*c[j] ;
                for ( int k = 0; k < 3; ++k )
                for ( int l = 0; l < 3; ++l )
                    m4[i][j][k][l] += w*c[i]*c[j]*c[k]*c[l] ;
            }
        }
    }
    REQUIRE_THAT(m0, WithinAbs(1.0, 1e-13)) ;
    for ( int i = 0; i < 3; ++i ) {
        REQUIRE_THAT(m1[i], WithinAbs(0.0, 1e-14)) ;
        for ( int j = 0; j < 3; ++j ) {
            REQUIRE_THAT(m2[i][j], WithinAbs(i==j ? 1.0/3.0 : 0.0, 1e-12)) ;
            for ( int k = 0; k < 3; ++k )
            for ( int l = 0; l < 3; ++l ) {
                double const dij = (i==j), dkl = (k==l), dik = (i==k), djl = (j==l), dil = (i==l), djk = (j==k) ;
                REQUIRE_THAT(m4[i][j][k][l], WithinAbs((dij*dkl + dik*djl + dil*djk)/15.0, 1e-12)) ;
            }
        }
    }
}

TEST_CASE("LBM stencil: closed under every coordinate reflection", "[lbm][stencil]")
{
    host_stencil hs ;
    for ( int ax = 0; ax < 3; ++ax ) {
        for ( int d = 0; d < hs.n; ++d ) {
            double m[3] = { hs.c(d,0), hs.c(d,1), hs.c(d,2) } ;
            m[ax] = -m[ax] ;
            bool found = false ;
            for ( int e = 0; e < hs.n && !found; ++e ) {
                double const dist = std::abs(hs.c(e,0)-m[0]) + std::abs(hs.c(e,1)-m[1]) + std::abs(hs.c(e,2)-m[2]) ;
                if ( dist < 1e-12 ) {
                    found = true ;
                    REQUIRE_THAT(hs.w(e), WithinAbs(hs.w(d), 1e-15)) ;   // mirrored directions share the weight
                }
            }
            REQUIRE(found) ;
        }
    }
}
#endif // GRACE_ENABLE_LBM
