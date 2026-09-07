/**
 * @file test_lbm_velocity_mesh.cpp
 * @brief The spherical Delaunay triangulation of the stencil directions and
 *        the barycentric locator: Euler count, manifold adjacency, outward
 *        orientation, exactness at the nodes, partition of unity and the
 *        accuracy on a linear function of the direction.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm_stencil.hh>
#include <grace/physics/lbm_velocity_mesh.hh>

#include <cmath>
#include <map>

using Catch::Matchers::WithinAbs;

TEST_CASE("LBM velocity mesh: 2N-4 outward triangles with manifold adjacency", "[lbm][vmesh]")
{
    auto const st = grace::lbm::load_stencil(GRACE_LBM_DATA_DIR) ;
    auto const vi = grace::lbm::build_velocity_interp(st, 64, 128) ;
    REQUIRE(vi.ntri == 2*st.ndir - 4) ;
    auto tri = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vi.tri) ;
    auto adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vi.adj) ;
    auto c   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
    std::map<std::pair<int,int>,int> edges ;
    std::vector<int> valence(st.ndir, 0) ;
    for ( int f = 0; f < vi.ntri; ++f ) {
        double p[3][3] ;
        for ( int k = 0; k < 3; ++k ) { for ( int a = 0; a < 3; ++a ) p[k][a] = c(tri(f,k),a) ; ++valence[tri(f,k)] ; }
        REQUIRE(grace::lbm::velocity_interp_t::det3(p[0], p[1], p[2]) > 0.0) ;
        for ( int k = 0; k < 3; ++k ) {
            ++edges[{tri(f,(k+1)%3), tri(f,(k+2)%3)}] ;
            int const g = adj(f,k) ;
            REQUIRE(g >= 0) ; REQUIRE(g != f) ;
            // the neighbour shares the edge opposite vertex k
            int shared = 0 ;
            for ( int m = 0; m < 3; ++m ) for ( int n = 0; n < 3; ++n ) if ( tri(g,m) == tri(f,n) && n != k ) ++shared ;
            REQUIRE(shared == 2) ;
        }
    }
    for ( auto const& e : edges ) {
        REQUIRE(e.second == 1) ;                                   // each directed edge once
        REQUIRE(edges.count({e.first.second, e.first.first}) == 1) ; // and its twin exists
    }
    for ( int d = 0; d < st.ndir; ++d ) REQUIRE(valence[d] >= 3) ;
}

TEST_CASE("LBM velocity mesh: locate is exact at the nodes and interpolates linear functions", "[lbm][vmesh]")
{
    auto const st = grace::lbm::load_stencil(GRACE_LBM_DATA_DIR) ;
    auto const vi = grace::lbm::build_velocity_interp(st, 64, 128) ;
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
    auto tri = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vi.tri) ;

    // largest edge angle of the mesh bounds the barycentric error on a linear function
    double hmax = 0. ;
    for ( int f = 0; f < vi.ntri; ++f ) for ( int k = 0; k < 3; ++k ) {
        int const a = tri(f,k), b = tri(f,(k+1)%3) ;
        double dot = 0. ; for ( int m = 0; m < 3; ++m ) dot += c(a,m)*c(b,m) ;
        hmax = std::fmax(hmax, std::acos(std::fmin(1.0, dot))) ;
    }
    // nodes: weight 1 on the node itself
    double err_node = 0., err_sum = 0. ;
    Kokkos::parallel_reduce("nodes", st.ndir, KOKKOS_LAMBDA (int const d, double& en, double& es) {
        double u[3] = { vi.c(d,0), vi.c(d,1), vi.c(d,2) } ; int idx[3] ; double lam[3] ;
        vi.locate(u, idx, lam) ;
        double wd = 0., s = 0. ;
        for ( int k = 0; k < 3; ++k ) { s += lam[k] ; if ( idx[k] == d ) wd += lam[k] ; }
        en = Kokkos::fmax(en, Kokkos::fabs(wd - 1.0)) ;
        es = Kokkos::fmax(es, Kokkos::fabs(s - 1.0)) ;
    }, Kokkos::Max<double>(err_node), Kokkos::Max<double>(err_sum)) ;
    REQUIRE(err_node < 1e-12) ;
    REQUIRE(err_sum  < 1e-14) ;

    // a spread of directions: positivity, partition of unity, linear reproduction, containment
    int const nsamp = 20000 ;
    double err_lin = 0., min_lam = 1., err_cont = 0. ;
    double const kvec[3] = {0.3, -0.5, 0.8} ;
    Kokkos::parallel_reduce("samples", nsamp, KOKKOS_LAMBDA (int const n, double& el, double& ml, double& ec) {
        // Fibonacci sphere
        double const z = 1.0 - 2.0*(n + 0.5)/nsamp, r = Kokkos::sqrt(1.0 - z*z), ph = 2.399963229728653*n ;
        double u[3] = { r*Kokkos::cos(ph), r*Kokkos::sin(ph), z } ; int idx[3] ; double lam[3] ;
        vi.locate(u, idx, lam) ;
        double f = 0., p[3] = {0.,0.,0.} ;
        for ( int k = 0; k < 3; ++k ) {
            ml = Kokkos::fmin(ml, lam[k]) ;
            for ( int a = 0; a < 3; ++a ) { f += lam[k]*kvec[a]*vi.c(idx[k],a) ; p[a] += lam[k]*vi.c(idx[k],a) ; }
        }
        double const exact = kvec[0]*u[0] + kvec[1]*u[1] + kvec[2]*u[2] ;
        el = Kokkos::fmax(el, Kokkos::fabs(f - exact)) ;
        // the interpolation point p is the gnomonic projection of u: parallel to u
        double const pn = Kokkos::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) ;
        double const cosang = (p[0]*u[0]+p[1]*u[1]+p[2]*u[2])/pn ;
        ec = Kokkos::fmax(ec, 1.0 - cosang) ;
    }, Kokkos::Max<double>(err_lin), Kokkos::Min<double>(min_lam), Kokkos::Max<double>(err_cont)) ;
    REQUIRE(min_lam >= 0.0) ;
    REQUIRE(err_cont < 1e-12) ;               // the query always lies inside the returned triangle
    REQUIRE(err_lin <= 1.0 - std::cos(hmax)) ; // planar chord error bound
    REQUIRE(err_lin < 2e-2) ;
}
#endif // GRACE_ENABLE_LBM
