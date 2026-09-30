/**
 * @file test_m1_optical_depth.cpp
 * @brief Unit tests of the eikonal optical-depth relaxation kernel (relax_cell).
 *
 * A 9^3 block in flat space, outer layer held fixed as the boundary condition, swept
 * with the production kernel.  The regression: tau must RISE when opacity appears in
 * cells seeded near zero (moving stars), not only fall from a large seed.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/eas_optical_depth.hh>

#include <cmath>

using namespace grace ;
using Catch::Matchers::WithinRel ;

#if defined(GRACE_M1_OPTICAL_DEPTH) && GRACE_M1_NU_SPECIES >= 1

namespace {

constexpr int    NB   = 9 ;          // cells per side, including the fixed outer layer
constexpr double DX   = 0.1 ;
constexpr double SEED = 2.553e-8 ;   // cold-fit seed at the atmosphere density

struct block_t {
    var_array_t a   = var_array_t("optd_a",   NB, NB, NB, N_EVOL_VARS, 1) ;
    var_array_t b   = var_array_t("optd_b",   NB, NB, NB, N_EVOL_VARS, 1) ;
    var_array_t aux = var_array_t("optd_aux", NB, NB, NB, N_AUX_VARS,  1) ;

    block_t() {
        auto A = a ; auto B = b ;
        Kokkos::parallel_for("optd_flat", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NB,NB,NB}),
            KOKKOS_LAMBDA (int i, int j, int k) {
            for ( auto s : {A, B} ) {
                #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
                s(i,j,k,GTXX_,0) = 1.0 ; s(i,j,k,GTYY_,0) = 1.0 ; s(i,j,k,GTZZ_,0) = 1.0 ; s(i,j,k,CHI_,0) = 1.0 ;
                #else
                s(i,j,k,GXX_,0) = 1.0 ; s(i,j,k,GYY_,0) = 1.0 ; s(i,j,k,GZZ_,0) = 1.0 ;
                #endif
            }
        }) ;
        Kokkos::fence() ;
    }

    /// tau of block 0 everywhere, in both buffers (the outer layer then stays as set).
    void set_tau(double tau_bg, double tau_core) {
        auto A = a ; auto B = b ;
        Kokkos::parallel_for("optd_seed", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NB,NB,NB}),
            KOKKOS_LAMBDA (int i, int j, int k) {
            bool const core = (i>=3 && i<=5 && j>=3 && j<=5 && k>=3 && k<=5) ;
            A(i,j,k,m1_optd_idx<0>(),0) = core ? tau_core : tau_bg ;
            B(i,j,k,m1_optd_idx<0>(),0) = core ? tau_core : tau_bg ;
        }) ;
        Kokkos::fence() ;
    }

    /// kappa_a of species 0 in the central 3^3 cells, zero elsewhere.
    void set_kappa_core(double kappa) {
        auto X = aux ;
        Kokkos::parallel_for("optd_kappa", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{NB,NB,NB}),
            KOKKOS_LAMBDA (int i, int j, int k) {
            bool const core = (i>=3 && i<=5 && j>=3 && j<=5 && k>=3 && k<=5) ;
            X(i,j,k,m1_kappaa_idx<0>(),0) = core ? kappa : 0.0 ;
            X(i,j,k,m1_kappas_idx<0>(),0) = 0.0 ;
        }) ;
        Kokkos::fence() ;
    }

    /// n Jacobi sweeps over the interior with the production kernel.
    void sweep(int n) {
        for ( int s = 0 ; s < n ; ++s ) {
            auto R = a ; auto W = b ; auto X = aux ;
            Kokkos::parallel_for("optd_sweep", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1},{NB-1,NB-1,NB-1}),
                KOKKOS_LAMBDA (int i, int j, int k) {
                double tau_out[5] = {0,0,0,0,0} ;
                optd_detail::relax_cell(R, X, VEC(i,j,k), 0, DX, DX, DX, tau_out) ;
                W(i,j,k,m1_optd_idx<0>(),0) = tau_out[0] ;
            }) ;
            Kokkos::fence() ;
            std::swap(a, b) ;
        }
    }

    double tau(int i, int j, int k) const {
        auto m = Kokkos::create_mirror_view(a) ; Kokkos::deep_copy(m, a) ;
        return m(i,j,k,m1_optd_idx<0>(),0) ;
    }
} ;

}  // namespace

TEST_CASE("eikonal tau rises when opacity appears in cells seeded near zero",
          "[m1][optd]")
{
    // A star that has moved onto atmosphere cells: tau holds the atmosphere seed, kappa is large.
    // Centre -> face cell of the core: kappa dx; face cell -> first transparent cell: kappa dx / 2.
    double const kappa = 100.0 ;
    block_t blk ;
    blk.set_tau(SEED, SEED) ;
    blk.set_kappa_core(kappa) ;
    blk.sweep(10) ;
    REQUIRE_THAT(blk.tau(4,4,4), WithinRel(1.5 * kappa * DX + SEED, 1e-12)) ;
    REQUIRE_THAT(blk.tau(5,4,4), WithinRel(0.5 * kappa * DX + SEED, 1e-12)) ;
    REQUIRE_THAT(blk.tau(1,1,1), WithinRel(SEED, 1e-12)) ;       // transparent cells keep the seed
}

TEST_CASE("eikonal tau falls when the matter has left", "[m1][optd]")
{
    block_t blk ;
    blk.set_tau(SEED, 1.0e4) ;          // stale cold-fit seed where the star used to be
    blk.set_kappa_core(0.0) ;
    blk.sweep(10) ;
    REQUIRE_THAT(blk.tau(4,4,4), WithinRel(SEED, 1e-12)) ;
}

TEST_CASE("eikonal tau is a fixed point: more sweeps do not inflate it", "[m1][optd]")
{
    // Without the self term two opaque neighbours could push each other up forever; the
    // min path to the transparent exterior caps them.
    block_t blk ;
    blk.set_tau(SEED, SEED) ;
    blk.set_kappa_core(100.0) ;
    blk.sweep(10) ;
    double const t10 = blk.tau(4,4,4) ;
    blk.sweep(400) ;
    REQUIRE(blk.tau(4,4,4) == t10) ;
}

#endif  // GRACE_M1_OPTICAL_DEPTH && GRACE_M1_NU_SPECIES >= 1
