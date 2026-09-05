/**
 * @file test_lbm_streaming.cpp
 * @brief The semi-Lagrangian streaming kernel on a hand-built quadrant.
 *
 *   [translate]  dt = dx along an axis-aligned stencil direction is an exact
 *                one-cell shift (the trilinear weights collapse to 0/1).
 *   [halfstep]   dt = dx/2 splits a delta into 1/2, 1/2 on the two cells the
 *                departure point straddles.
 *   [floor]      empty populations come out at the isotropic floor.
 * No grid, no parfile: views are built by hand exactly like
 * test_m1_backreaction.cpp does.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm.hh>

using Catch::Matchers::WithinAbs;
using namespace grace ;

namespace {
constexpr int NX = 8, NGZ = 2, N = NX + 2*NGZ ;

struct fixture {
    var_array_t I_old, I_new, aux ;
    scalar_array_t<GRACE_NSPACEDIM> dx ;
    lbm::stencil_t st ;
    lbm::params_t p ;
    int d_px = -1 ;   // the +x stencil direction (Lebedev tables contain the axes exactly)

    fixture(double I_fl)
        : I_old("Iold", N,N,N, N_EVOL_VARS, 1)
        , I_new("Inew", N,N,N, N_EVOL_VARS, 1)
        , aux  ("aux",  N,N,N, N_AUX_VARS,  1)
        , dx   ("dx",   GRACE_NSPACEDIM, 1)
        , st(lbm::load_stencil(GRACE_LBM_DATA_DIR))
    {
        auto hdx = Kokkos::create_mirror_view(dx) ;
        for ( int i = 0; i < GRACE_NSPACEDIM; ++i ) hdx(i,0) = 1.0 ;
        Kokkos::deep_copy(dx, hdx) ;
        p = lbm::params_t{ 1e-4, 100, I_fl, 1.0 } ;
        auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
        for ( int d = 0; d < st.ndir; ++d ) if ( c(d,0) > 0.999999 ) d_px = d ;
        REQUIRE(d_px >= 0) ;
    }
    lbm::system_t sys() const { return lbm::system_t{ I_old, I_new, aux, st, dx, p } ; }

    void stream(double dt) {
        auto s = sys() ;
        Kokkos::parallel_for("stream",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({NGZ,NGZ,NGZ,0},{NX+NGZ,NX+NGZ,NX+NGZ,1}),
            KOKKOS_LAMBDA (int i, int j, int k, int q) { s.stream(i,j,k,q,dt) ; }) ;
        Kokkos::fence() ;
    }
} ;
}

TEST_CASE("LBM streaming: dt = dx along +x is an exact one-cell shift", "[lbm][streaming]")
{
    fixture fx(0.0) ;
    int const iv = lbm::idx(0, fx.d_px) ;
    int const i0 = NGZ+3, j0 = NGZ+4, k0 = NGZ+5 ;
    auto h = Kokkos::create_mirror_view(fx.I_old) ;
    h(i0,j0,k0,iv,0) = 1.0 ;
    Kokkos::deep_copy(fx.I_old, h) ;

    fx.stream(1.0) ;   // dt = dx

    auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fx.I_new) ;
    for ( int i = NGZ; i < NX+NGZ; ++i )
    for ( int j = NGZ; j < NX+NGZ; ++j )
    for ( int k = NGZ; k < NX+NGZ; ++k ) {
        double const expect = ( i == i0+1 && j == j0 && k == k0 ) ? 1.0 : 0.0 ;
        REQUIRE(hn(i,j,k,iv,0) == expect) ;   // exact, no tolerance
    }
    // the other populations of the cell stayed empty
    for ( int d = 0; d < fx.st.ndir; ++d ) if ( d != fx.d_px ) REQUIRE(hn(i0+1,j0,k0,lbm::idx(0,d),0) == 0.0) ;
}

TEST_CASE("LBM streaming: dt = dx/2 splits a delta 1/2 : 1/2", "[lbm][streaming]")
{
    fixture fx(0.0) ;
    int const iv = lbm::idx(0, fx.d_px) ;
    int const i0 = NGZ+3, j0 = NGZ+2, k0 = NGZ+6 ;
    auto h = Kokkos::create_mirror_view(fx.I_old) ;
    h(i0,j0,k0,iv,0) = 1.0 ;
    Kokkos::deep_copy(fx.I_old, h) ;

    fx.stream(0.5) ;

    auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fx.I_new) ;
    REQUIRE_THAT(hn(i0  ,j0,k0,iv,0), WithinAbs(0.5, 1e-15)) ;
    REQUIRE_THAT(hn(i0+1,j0,k0,iv,0), WithinAbs(0.5, 1e-15)) ;
    REQUIRE(hn(i0-1,j0,k0,iv,0) == 0.0) ;
    REQUIRE(hn(i0+2,j0,k0,iv,0) == 0.0) ;
    // total is conserved by the interpolation
    double tot = 0. ;
    for ( int i = NGZ; i < NX+NGZ; ++i ) for ( int j = NGZ; j < NX+NGZ; ++j ) for ( int k = NGZ; k < NX+NGZ; ++k ) tot += hn(i,j,k,iv,0) ;
    REQUIRE_THAT(tot, WithinAbs(1.0, 1e-14)) ;
}

TEST_CASE("LBM streaming: empty populations come out at the isotropic floor", "[lbm][streaming]")
{
    double const fl = 3e-7 ;
    fixture fx(fl) ;
    fx.stream(0.3) ;
    auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fx.I_new) ;
    for ( int d = 0; d < fx.st.ndir; d += 37 )
        REQUIRE(hn(NGZ+1,NGZ+1,NGZ+1,lbm::idx(0,d),0) == fl) ;
}
#endif // GRACE_ENABLE_LBM
