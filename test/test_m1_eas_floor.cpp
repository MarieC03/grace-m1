/**
 * @file test_m1_eas_floor.cpp
 * @brief floor_neutrino_eas (atmosphere cells and the idle-M1 pass): every slot it owns is
 *        overwritten, so no stale value survives, and nothing else is touched.
 */
#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/eas_policies.hh>

#include <vector>

using namespace grace ;

TEST_CASE("EAS floor: every owned slot is overwritten, nothing else", "[m1][eas][floor]")
{
    var_array_t aux("floor_aux", 1, 1, 1, N_AUX_VARS, 1) ;
    constexpr double junk = 7.25 ;
    Kokkos::deep_copy(aux, junk) ;
    Kokkos::parallel_for("floor", 1, KOKKOS_LAMBDA(int) { floor_neutrino_eas(aux, VEC(0,0,0), 0) ; }) ;
    Kokkos::fence() ;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), aux) ;
    auto at = [&] (int v) { return h(VEC(0,0,0), v, 0) ; } ;

    std::vector<int> const rates = {
        ETA1_, KAPPAA1_, KAPPAS1_, ETAN1_, KAPPAAN1_
        #if GRACE_M1_NU_SPECIES >= 3
        , ETA2_, KAPPAA2_, KAPPAS2_, ETAN2_, KAPPAAN2_, ETA3_, KAPPAA3_, KAPPAS3_, ETAN3_, KAPPAAN3_
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
        , ETA4_, KAPPAA4_, KAPPAS4_, ETAN4_, KAPPAAN4_, ETA5_, KAPPAA5_, KAPPAS5_, ETAN5_, KAPPAAN5_
        #endif
    } ;
    for ( int v : rates ) REQUIRE( at(v) == weakhub::kappa_floor_code ) ;
    REQUIRE( at(BETAEQ_ERR_) == 0.0 ) ;

    #ifdef GRACE_M1_PHYSICS_OUTPUT
    std::vector<int> const fugacity = {
        ETANU1_, MUE_, MUP_, MUN_
        #if GRACE_M1_NU_SPECIES >= 3
        , ETANU2_
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
        , ETANU3_, ETANU4_
        #endif
        #ifdef GRACE_ENABLE_MUONS
        , MUMU_
        #endif
    } ;
    for ( int v : fugacity ) REQUIRE( at(v) == 1.0e-30 ) ;
    // The backreaction output is reset once per step (evolve.cpp), never by the floor.
    REQUIRE( at(M1_HEATCOOL_) == junk ) ;
    REQUIRE( at(M1_BR_REJECT_) == junk ) ;
    #endif

    #ifdef GRACE_M1_DIAGNOSTICS
    REQUIRE( at(XN_) == 1.0e-30 ) ;
    REQUIRE( at(MUDELTA_NPE_) == 1.0e-30 ) ;
    REQUIRE( at(BETAEQ_TSCALE_) == 1.0e30 ) ;
    REQUIRE( at(BETAEQ_T_EQ_) == -1.0 ) ;
    REQUIRE( at(M1_IMPLICIT_ERR_) == junk ) ;
    #endif

    // Matter fields are not the floor's business.
    REQUIRE( at(RHO_) == junk ) ;
    REQUIRE( at(TEMP_) == junk ) ;
}
