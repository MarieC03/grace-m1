/**
 * @file test_m1_test_id.cpp
 * @brief Initial data of the M1 test problems, evaluated on a small flat grid.
 *   [beam]    the default straight beam is the historical box, bit for bit; an oblique
 *             beam lights exactly the cylinder behind `start`, with F = ff E n.
 *   [sphere]  sphere wave: unit E inside the radius, floor outside, no flux, N = E.
 *   [plane]   plane wave: E = E_bg (1 + A sin(2 pi x / lambda)), F_x = f E, N = E.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/id/m1_initial_data.hh>

#include <cmath>

using Catch::Matchers::WithinAbs;
using namespace grace ;

#if GRACE_M1_NU_SPECIES >= 1 && GRACE_NSPACEDIM == 3

namespace {

constexpr int N = 32 ;   // dx = 1/16: the default beam (half-width 1/16) spans 2 cells
constexpr double LO = -1.0, DX = 2.0 / N, E_FL = 1.0e-14 ;

m1_atmo_params_t atmo() {
    m1_atmo_params_t a{} ;
    a.E_fl = E_FL ; a.eps_fl = 1.0 ; a.N_fl = E_FL ; a.r_damping = 1.0e30 ;
    return a ;
}
m1_excision_params_t no_excision() {
    m1_excision_params_t e{} ; e.excise_by_radius = false ; e.r_ex = 0. ; e.alp_ex = 0. ;
    return e ;
}

coord_array_t<3> cell_centres() {
    coord_array_t<3> c("coords", N, N, N, 3, 1) ;
    auto h = Kokkos::create_mirror_view(c) ;
    for ( int i=0; i<N; ++i ) for ( int j=0; j<N; ++j ) for ( int k=0; k<N; ++k ) {
        h(i,j,k,0,0) = LO + (i+0.5)*DX ; h(i,j,k,1,0) = LO + (j+0.5)*DX ; h(i,j,k,2,0) = LO + (k+0.5)*DX ;
    }
    Kokkos::deep_copy(c, h) ;
    return c ;
}

// E, Fx, Fy, Fz, N of species 1 at every cell, on the host.
template < typename id_t >
Kokkos::View<double****, Kokkos::HostSpace> evaluate(id_t const& id) {
    Kokkos::View<double****> out("out", N, N, N, 5) ;
    Kokkos::parallel_for("eval", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{N,N,N}),
        KOKKOS_LAMBDA (int i, int j, int k) {
            m1_id_t const r = id(i, j, k, 0) ;
            out(i,j,k,0) = r.erad1 ; out(i,j,k,1) = r.fradx1 ; out(i,j,k,2) = r.frady1 ;
            out(i,j,k,3) = r.fradz1 ; out(i,j,k,4) = r.nrad1 ;
        }) ;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out) ;
    return h ;
}

double ctr(int i) { return LO + (i+0.5)*DX ; }

}  // namespace

TEST_CASE("M1 test ID: the default straight beam is the historical box", "[m1][id][beam]")
{
    auto const v = evaluate(straight_beam_m1_id_t(atmo(), no_excision(), cell_centres())) ;
    int lit = 0 ;
    for ( int i=0; i<N; ++i ) for ( int j=0; j<N; ++j ) for ( int k=0; k<N; ++k ) {
        double const x = ctr(i), y = ctr(j), z = ctr(k) ;
        bool const old = x <= -0.25 && y < 0.0625 && y > -0.0625 && z < 0.0625 && z > -0.0625 ;
        lit += old ;
        REQUIRE( v(i,j,k,0) == (old ? 1.0 : E_FL) ) ;
        REQUIRE( v(i,j,k,1) == (old ? M1_ID_CAUSAL_FRAC : 0.0) ) ;
        REQUIRE( v(i,j,k,2) == 0.0 ) ; REQUIRE( v(i,j,k,3) == 0.0 ) ;
        REQUIRE( v(i,j,k,4) == v(i,j,k,0) ) ;
    }
    REQUIRE( lit > 0 ) ;
}

TEST_CASE("M1 test ID: an oblique beam lights the cylinder behind start along n", "[m1][id][beam]")
{
    double const nx = 1.0, ny = 1.0, nz = 0.5, hw = 0.3, start = -0.2, ff = 0.99999 ;
    double const nn = std::sqrt(nx*nx + ny*ny + nz*nz) ;
    auto const v = evaluate(straight_beam_m1_id_t(atmo(), no_excision(), cell_centres(), nx, ny, nz, hw, start, ff)) ;
    int lit = 0, dark = 0 ;
    for ( int i=0; i<N; ++i ) for ( int j=0; j<N; ++j ) for ( int k=0; k<N; ++k ) {
        double const x = ctr(i), y = ctr(j), z = ctr(k) ;
        double const s = (x*nx + y*ny + z*nz) / nn, p2 = x*x + y*y + z*z - s*s ;
        bool const in = s <= start && p2 < hw*hw ;
        if ( in ) {
            ++lit ;
            REQUIRE( v(i,j,k,0) == 1.0 ) ;
            REQUIRE_THAT( v(i,j,k,1), WithinAbs(ff * nx/nn, 1e-14) ) ;
            REQUIRE_THAT( v(i,j,k,2), WithinAbs(ff * ny/nn, 1e-14) ) ;
            REQUIRE_THAT( v(i,j,k,3), WithinAbs(ff * nz/nn, 1e-14) ) ;
        } else {
            ++dark ;
            REQUIRE( v(i,j,k,0) == E_FL ) ;
            REQUIRE( v(i,j,k,1) == 0.0 ) ; REQUIRE( v(i,j,k,2) == 0.0 ) ; REQUIRE( v(i,j,k,3) == 0.0 ) ;
        }
        REQUIRE( v(i,j,k,4) == v(i,j,k,0) ) ;
    }
    REQUIRE( lit > 20 ) ; REQUIRE( dark > lit ) ;
}

TEST_CASE("M1 test ID: sphere wave", "[m1][id][sphere]")
{
    double const R = 0.4, x0 = 0.1, y0 = -0.2, z0 = 0.05 ;
    auto const v = evaluate(sphere_wave_m1_id_t(atmo(), no_excision(), cell_centres(), R, x0, y0, z0)) ;
    int inside = 0 ;
    for ( int i=0; i<N; ++i ) for ( int j=0; j<N; ++j ) for ( int k=0; k<N; ++k ) {
        double const r = std::sqrt( std::pow(ctr(i)-x0,2) + std::pow(ctr(j)-y0,2) + std::pow(ctr(k)-z0,2) ) ;
        inside += r < R ;
        REQUIRE( v(i,j,k,0) == (r < R ? 1.0 : E_FL) ) ;
        REQUIRE( v(i,j,k,1) == 0.0 ) ; REQUIRE( v(i,j,k,2) == 0.0 ) ; REQUIRE( v(i,j,k,3) == 0.0 ) ;
        REQUIRE( v(i,j,k,4) == v(i,j,k,0) ) ;
    }
    REQUIRE( inside > 8 ) ;
}

TEST_CASE("M1 test ID: plane wave", "[m1][id][plane]")
{
    double const bg = 2.0, A = 0.5, lambda = 1.0 ;
    for ( double const f : {0.5, 1.0} ) {
        auto const v = evaluate(plane_wave_m1_id_t(atmo(), no_excision(), cell_centres(), bg, A, lambda, f)) ;
        double const fc = std::fmin(f, M1_ID_CAUSAL_FRAC) ;
        for ( int i=0; i<N; ++i ) for ( int j=0; j<N; ++j ) for ( int k=0; k<N; ++k ) {
            double const E = bg * (1.0 + A * std::sin(2.0 * M_PI * ctr(i) / lambda)) ;
            REQUIRE_THAT( v(i,j,k,0), WithinAbs(E, 1e-14) ) ;
            REQUIRE_THAT( v(i,j,k,1), WithinAbs(fc * E, 1e-14) ) ;
            REQUIRE( v(i,j,k,2) == 0.0 ) ; REQUIRE( v(i,j,k,3) == 0.0 ) ;
            REQUIRE( v(i,j,k,4) == v(i,j,k,0) ) ;
            REQUIRE( v(i,j,k,0) == v(i,0,0,0) ) ;          // uniform in y and z
        }
    }
}

#endif
