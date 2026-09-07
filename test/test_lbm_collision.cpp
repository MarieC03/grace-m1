/**
 * @file test_lbm_collision.cpp
 * @brief The static-fluid, isotropic-scattering collision step on one cell.
 *
 *   [absorb]    kappa_a only: every population and E scale by 1/(1+dt kappa_a).
 *   [scatter]   kappa_s only: E is conserved to round-off, the flux relaxes by
 *               1/(1+dt kappa_s) -- scattering isotropises, never heats.
 *   [emit]      eta only with kappa_a: E -> eta/kappa_a as dt -> infinity.
 *   [identity]  all rates at the floor: nothing changes, zero iterations.
 * Rates go straight into the aux slots M1's EAS providers write.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#ifdef GRACE_ENABLE_LBM
#include <grace/physics/lbm.hh>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace grace ;

namespace {
constexpr int N = 3, C = 1 ;   // one interior cell (1,1,1) with a 1-cell pad

struct cell {
    var_array_t I, aux ;
    scalar_array_t<GRACE_NSPACEDIM> dx ;
    lbm::stencil_t st ;
    lbm::params_t p{1e-4, 100, 0.0, 1.0} ;

    cell() : I("I", N,N,N, N_EVOL_VARS, 1), aux("aux", N,N,N, N_AUX_VARS, 1)
           , dx("dx", GRACE_NSPACEDIM, 1), st(lbm::load_stencil(GRACE_LBM_DATA_DIR))
    {
        auto hdx = Kokkos::create_mirror_view(dx) ;
        for ( int i = 0; i < GRACE_NSPACEDIM; ++i ) hdx(i,0) = 1.0 ;
        Kokkos::deep_copy(dx, hdx) ;
    }
    void set_rates(double ka, double ks, double eta) {
        auto h = Kokkos::create_mirror_view(aux) ;
        h(C,C,C, lbm::kappaa_idx(0), 0) = ka ;
        h(C,C,C, lbm::kappas_idx(0), 0) = ks ;
        h(C,C,C, lbm::eta_idx(0),    0) = eta ;
        Kokkos::deep_copy(aux, h) ;
    }
    template <typename F> void set_populations(F f) {   // f(cx,cy,cz) -> I_d
        auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
        auto h = Kokkos::create_mirror_view(I) ;
        for ( int d = 0; d < st.ndir; ++d ) h(C,C,C, lbm::idx(0,d), 0) = f(c(d,0),c(d,1),c(d,2)) ;
        // flat metric: the collision runs at the lapse rate (alpha dt)
        h(C,C,C, ALP_, 0) = 1.0 ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        h(C,C,C, GXX_, 0) = h(C,C,C, GYY_, 0) = h(C,C,C, GZZ_, 0) = 1.0 ;
        #else
        h(C,C,C, CHI_, 0) = h(C,C,C, GTXX_, 0) = h(C,C,C, GTYY_, 0) = h(C,C,C, GTZZ_, 0) = 1.0 ;
        #endif
        Kokkos::deep_copy(I, h) ;
    }
    int collide(double dt) {
        lbm::system_t s{ I, I, aux, st, dx, p } ;
        Kokkos::View<int> n("n") ;
        Kokkos::parallel_for("collide", 1, KOKKOS_LAMBDA (int) { n() = s.collide(C,C,C,0,dt) ; }) ;
        Kokkos::fence() ;
        auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, n) ;
        return hn() ;
    }
    void moments(double& E, double (&F)[3]) {
        auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.w) ;
        auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, I) ;
        E = 0 ; F[0] = F[1] = F[2] = 0 ;
        for ( int d = 0; d < st.ndir; ++d ) {
            double const wI = w(d)*h(C,C,C, lbm::idx(0,d), 0) ;
            E += wI ; for ( int i = 0; i < 3; ++i ) F[i] += wI*c(d,i) ;
        }
    }
} ;
}

TEST_CASE("LBM collision: pure absorption scales everything by 1/(1+dt kappa_a)", "[lbm][collision]")
{
    cell cl ;
    cl.set_populations([](double,double,double){ return 1.0 ; }) ;
    cl.set_rates(2.0, 0.0, 0.0) ;
    double const dt = 0.1 ;
    REQUIRE(cl.collide(dt) == 1) ;
    double E, F[3] ; cl.moments(E, F) ;
    REQUIRE_THAT(E, WithinRel(1.0/(1.0 + dt*2.0), 1e-14)) ;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cl.I) ;
    REQUIRE_THAT(h(C,C,C, lbm::idx(0,7), 0), WithinRel(1.0/1.2, 1e-14)) ;
}

TEST_CASE("LBM collision: isotropic scattering conserves E and damps the flux", "[lbm][collision]")
{
    cell cl ;
    cl.set_populations([](double,double,double cz){ return 1.0 + 0.6*cz ; }) ;   // E = 1, F_z = 0.2
    double E0, F0[3] ; cl.moments(E0, F0) ;
    REQUIRE_THAT(E0, WithinAbs(1.0, 1e-13)) ;
    REQUIRE_THAT(F0[2], WithinAbs(0.2, 1e-13)) ;

    double const ks = 5.0, dt = 0.3 ;
    cl.set_rates(0.0, ks, 0.0) ;
    REQUIRE(cl.collide(dt) == 1) ;
    double E1, F1[3] ; cl.moments(E1, F1) ;
    REQUIRE_THAT(E1, WithinAbs(1.0, 1e-13)) ;
    REQUIRE_THAT(F1[2], WithinRel(0.2/(1.0 + dt*ks), 1e-12)) ;
    REQUIRE_THAT(F1[0], WithinAbs(0.0, 1e-13)) ;
}

TEST_CASE("LBM collision: emission saturates at eta/kappa_a", "[lbm][collision]")
{
    cell cl ;
    cl.set_populations([](double,double,double){ return 0.0 ; }) ;
    double const ka = 1.0, eta = 3.0, dt = 1e6 ;
    cl.set_rates(ka, 0.0, eta) ;
    cl.collide(dt) ;
    double E, F[3] ; cl.moments(E, F) ;
    REQUIRE_THAT(E, WithinRel(eta/ka, 1e-5)) ;   // (dt eta)/(1 + dt ka) -> eta/ka
    REQUIRE_THAT(F[0], WithinAbs(0.0, 1e-12)) ;
}

TEST_CASE("LBM collision: rates at the floor leave the cell untouched", "[lbm][collision]")
{
    cell cl ;
    cl.set_populations([](double cx,double,double){ return 2.0 + cx ; }) ;
    cl.set_rates(0.0, 0.0, 0.0) ;
    REQUIRE(cl.collide(0.5) == 0) ;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cl.I) ;
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cl.st.c) ;
    for ( int d = 0; d < cl.st.ndir; d += 11 ) REQUIRE(h(C,C,C, lbm::idx(0,d), 0) == 2.0 + c(d,0)) ;
}
#endif // GRACE_ENABLE_LBM
