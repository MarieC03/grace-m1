/**
 * @file test_betaeq_fallback.cpp
 * @brief Decision logic of the timescale beta-eq closure with the electron-only
 *        fallback (betaeq_solve_with_e_fallback, eas_policies.hh).
 *
 * The solve is a mock with the contract of m1_get_beta_equilibrium: outputs hold
 * the old state on entry, are written only on success, and a failure sets bits.
 *   [success]  a joint solve that succeeds is returned untouched, fallback on or off,
 *              and the electron-only solve is never called.
 *   [failure]  fallback off: the joint failure is reported exactly as before.
 *   [rescue]   fallback on: the electron-only result replaces the failure, Ymu is
 *              held, and berr carries only the rescue bit.
 *   [both]     both fail: berr and the outputs are the joint solve's.
 *   [floor]    betaeq_held_ymu picks the table floor only when the variant is on and
 *              the muon residual at the floor is positive; the floor bit survives a
 *              successful retry and nothing else.
 * 5-species builds only (the fallback does not exist otherwise).
 */
#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/eas_policies.hh>

using namespace grace ;

namespace {

constexpr double kT = 7.0, kYe = 0.026, kYmu = 6.1e-4 ;   // class-A cell (head-on tip)

struct mock_solve_t {
    bool joint_ok, e_ok ;
    int* n_joint ; int* n_e ;
    bool e_floor = false ;   // the retry held Ymu at the floor
    bool operator()(bool const electron_only, betaeq_err_t& be,
                    double& T, double& Ye, double& Ymu) const {
        T = kT ; Ye = kYe ; Ymu = kYmu ;
        if ( !electron_only ) {
            ++*n_joint ;
            if ( joint_ok ) { T = 9.0 ; Ye = 0.05 ; Ymu = 2.0e-3 ; return true ; }
            be.set(BETAEQ_NOT_CONVERGED) ; be.set(BETAEQ_AT_BOUND) ; be.set(BETAEQ_MUON_SECTOR) ;
            return false ;
        }
        ++*n_e ;
        if ( e_floor ) { Ymu = 5.0e-4 ; be.set(BETAEQ_YMU_FLOOR_HOLD) ; }
        if ( e_ok ) { T = 8.0 ; Ye = 0.11 ; return true ; }   // Ymu held
        be.set(BETAEQ_NOT_CONVERGED) ; be.set(BETAEQ_RESIDUAL_LARGE) ;
        return false ;
    }
} ;

uint64_t joint_failure_bits() {
    betaeq_err_t b{} ;
    b.set(BETAEQ_NOT_CONVERGED) ; b.set(BETAEQ_AT_BOUND) ; b.set(BETAEQ_MUON_SECTOR) ;
    return b.words[0] ;
}

}  // namespace

#if defined(GRACE_ENABLE_MUONS) && GRACE_M1_NU_SPECIES >= 5

TEST_CASE("betaeq fallback: a successful joint solve is returned untouched",
          "[betaeq][fallback][success]")
{
    for ( bool const fb : {false, true} ) {
        INFO("fallback " << fb) ;
        int nj = 0, ne = 0 ;
        mock_solve_t const s{true, true, &nj, &ne} ;
        betaeq_err_t berr{} ; berr.set(BETAEQ_NO_BRACKET) ;       // earlier bits survive
        double T = kT, Ye = kYe, Ymu = kYmu ;
        REQUIRE( betaeq_solve_with_e_fallback(s, fb, berr, T, Ye, Ymu) ) ;
        REQUIRE( nj == 1 ) ; REQUIRE( ne == 0 ) ;
        REQUIRE( T == 9.0 ) ; REQUIRE( Ye == 0.05 ) ; REQUIRE( Ymu == 2.0e-3 ) ;
        REQUIRE( berr.words[0] == (uint64_t(1) << BETAEQ_NO_BRACKET) ) ;
    }
}

TEST_CASE("betaeq fallback: off, a joint failure is reported as before",
          "[betaeq][fallback][failure]")
{
    int nj = 0, ne = 0 ;
    mock_solve_t const s{false, true, &nj, &ne} ;
    betaeq_err_t berr{} ;
    double T = kT, Ye = kYe, Ymu = kYmu ;
    REQUIRE_FALSE( betaeq_solve_with_e_fallback(s, false, berr, T, Ye, Ymu) ) ;
    REQUIRE( nj == 1 ) ; REQUIRE( ne == 0 ) ;
    REQUIRE( T == kT ) ; REQUIRE( Ye == kYe ) ; REQUIRE( Ymu == kYmu ) ;
    REQUIRE( berr.words[0] == joint_failure_bits() ) ;
    REQUIRE_FALSE( berr.test(BETAEQ_PARTIAL_E_FALLBACK) ) ;
}

TEST_CASE("betaeq fallback: on, the electron-only solve rescues a joint failure",
          "[betaeq][fallback][rescue]")
{
    int nj = 0, ne = 0 ;
    mock_solve_t const s{false, true, &nj, &ne} ;
    betaeq_err_t berr{} ;
    double T = kT, Ye = kYe, Ymu = kYmu ;
    REQUIRE( betaeq_solve_with_e_fallback(s, true, berr, T, Ye, Ymu) ) ;
    REQUIRE( nj == 1 ) ; REQUIRE( ne == 1 ) ;
    REQUIRE( T == 8.0 ) ; REQUIRE( Ye == 0.11 ) ; REQUIRE( Ymu == kYmu ) ;
    REQUIRE( berr.words[0] == (uint64_t(1) << BETAEQ_PARTIAL_E_FALLBACK) ) ;
}

TEST_CASE("betaeq fallback: on, both solves fail -> the joint failure only",
          "[betaeq][fallback][both]")
{
    int nj = 0, ne = 0 ;
    mock_solve_t const s{false, false, &nj, &ne} ;
    betaeq_err_t berr{} ;
    double T = kT, Ye = kYe, Ymu = kYmu ;
    REQUIRE_FALSE( betaeq_solve_with_e_fallback(s, true, berr, T, Ye, Ymu) ) ;
    REQUIRE( nj == 1 ) ; REQUIRE( ne == 1 ) ;
    REQUIRE( T == kT ) ; REQUIRE( Ye == kYe ) ; REQUIRE( Ymu == kYmu ) ;
    REQUIRE( berr.words[0] == joint_failure_bits() ) ;
}

TEST_CASE("betaeq fallback: the held Ymu is the floor only below-floor equilibria",
          "[betaeq][fallback][floor]")
{
    constexpr double floor = 5.0e-4 ;
    REQUIRE( betaeq_held_ymu(false, kYmu, floor, true,  1.0e-3) == kYmu ) ;   // variant off
    REQUIRE( betaeq_held_ymu(true,  kYmu, floor, true,  1.0e-3) == floor ) ;  // f1(floor) > 0
    REQUIRE( betaeq_held_ymu(true,  kYmu, floor, true,  0.0)    == kYmu ) ;   // root at the floor
    REQUIRE( betaeq_held_ymu(true,  kYmu, floor, true, -1.0e-3) == kYmu ) ;   // root above
    REQUIRE( betaeq_held_ymu(true,  kYmu, floor, false, 1.0e-3) == kYmu ) ;   // residual failed
}

TEST_CASE("betaeq fallback: the floor bit survives a successful retry only",
          "[betaeq][fallback][floor]")
{
    SECTION("rescued at the floor") {
        int nj = 0, ne = 0 ;
        mock_solve_t s{false, true, &nj, &ne} ; s.e_floor = true ;
        betaeq_err_t berr{} ;
        double T = kT, Ye = kYe, Ymu = kYmu ;
        REQUIRE( betaeq_solve_with_e_fallback(s, true, berr, T, Ye, Ymu) ) ;
        REQUIRE( Ymu == 5.0e-4 ) ;
        REQUIRE( berr.words[0] == ( (uint64_t(1) << BETAEQ_PARTIAL_E_FALLBACK)
                                  | (uint64_t(1) << BETAEQ_YMU_FLOOR_HOLD) ) ) ;
    }
    SECTION("retry at the floor fails -> joint failure bits only") {
        int nj = 0, ne = 0 ;
        mock_solve_t s{false, false, &nj, &ne} ; s.e_floor = true ;
        betaeq_err_t berr{} ;
        double T = kT, Ye = kYe, Ymu = kYmu ;
        REQUIRE_FALSE( betaeq_solve_with_e_fallback(s, true, berr, T, Ye, Ymu) ) ;
        REQUIRE( Ymu == kYmu ) ;
        REQUIRE( berr.words[0] == joint_failure_bits() ) ;
    }
    SECTION("joint solve succeeds -> the retry and its bit never happen") {
        int nj = 0, ne = 0 ;
        mock_solve_t s{true, true, &nj, &ne} ; s.e_floor = true ;
        betaeq_err_t berr{} ;
        double T = kT, Ye = kYe, Ymu = kYmu ;
        REQUIRE( betaeq_solve_with_e_fallback(s, true, berr, T, Ye, Ymu) ) ;
        REQUIRE( ne == 0 ) ;
        REQUIRE( berr.words[0] == 0 ) ;
    }
}

#endif
