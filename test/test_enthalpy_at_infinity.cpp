/**
 * @file test_enthalpy_at_infinity.cpp
 * @brief Bernoulli h_inf(Y_p) built from synthetic in-memory tables (enthalpy_at_infinity.hh).
 *
 * Coverage:
 *   [leptonic]  Signed-linear and log baryon pressure, with and without the electron table:
 *               h_inf equals the brute-force minimum over density of 1 + eps + P/rho at T_min.
 *   [ele_axis]  An electron table on its own Y_e axis is blended linearly in Y_e.
 *   [tabulated] The 3D table (ln P, ln(eps + shift)) gives the same minimum.
 *   [interp]    Linear in Y_p between the nodes, constant beyond the axis, h_const without a table.
 *   [edge]      Minima on the first or last density node are counted.
 *   [hybrid]    Cold backbone of the hybrid EOS: the minimum over its rows, not the first row.
 *
 * No HDF5 file or parfile is needed.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/eos/enthalpy_at_infinity.hh>

#include <cmath>
#include <vector>

namespace {

using namespace grace ;
using L = leptonic_eos_4d_t ;
using Catch::Matchers::WithinRel ;

// Uniform axes as the interpolators require; T_min is the first temperature node.
constexpr int NR = 7, NT = 3, NY = 4, NM = 2 ;
double lr_(int i) { return std::log(1e-8) + 1.0*i ; }
double lt_(int j) { return std::log(0.1) + 0.7*j ; }
double yq_(int k) { return 0.1 + 0.1*k ; }
constexpr double SHIFT = 0.01 ;

// Baryons: interior minimum over density for k < 3, minimum on the first node for k = 3.
double eps_b(int i, int j, int k) {
    return k < 3 ? -0.004 + 0.0005*(i - 3)*(i - 3) + 0.003*j + 0.001*k
                 :  0.002 + 0.0004*i + 0.003*j ;
}
double pb(int i, int j, int k) { return std::exp(lr_(i)) * (1e-4 + 1e-4*k + 2e-4*j) ; }
// Electrons + positrons, linear in Y_e (so a linear blend in Y_e is exact).
double eps_em(int i, double y) { return 2e-4*(1. + y)*(1. + 0.1*i) ; }
double eps_ep()                { return 5e-5 ; }
double pem(int i, double y)    { return std::exp(lr_(i)) * 4e-4 * y ; }
double pep(int i)              { return std::exp(lr_(i)) * 2e-5 ; }

// Brute force at T_min: min over density, and whether it sits on the first/last density node.
struct expect_t { std::vector<double> h ; int n_edge = 0 ; } ;
expect_t brute_force(bool add_ele) {
    expect_t e ;
    for (int k=0; k<NY; ++k) {
        int imin = 0 ; double hmin = 1e300 ;
        for (int i=0; i<NR; ++i) {
            double const rho = std::exp(lr_(i)), y = yq_(k) ;
            double eps = eps_b(i,0,k), p = pb(i,0,k) ;
            if ( add_ele ) { eps += eps_em(i,y) + eps_ep() ; p += pem(i,y) + pep(i) ; }
            double const h = 1. + eps + p/rho ;
            if ( h < hmin ) { hmin = h ; imin = i ; }
        }
        e.h.push_back(hmin) ;
        if ( imin == 0 || imin == NR-1 ) ++e.n_edge ;
    }
    return e ;
}

Kokkos::View<double*> axis(char const* label, int n, double (*f)(int)) {
    Kokkos::View<double*> v(label, n) ;
    auto h = Kokkos::create_mirror_view(v) ;
    for (int m=0; m<n; ++m) h(m) = f(m) ;
    Kokkos::deep_copy(v, h) ;
    return v ;
}

// Leptonic EOS; the electron table lives on yle(m) = yle0 + dyle*m, nyle nodes.
L make_leptonic(bool add_ele, bool log_press, int nyle, double yle0, double dyle) {
    Kokkos::View<double****> bar("bar", NR, NT, NY, L::N_TAB_VARS_BARYON) ;
    Kokkos::View<double****> ele("ele", NR, NT, nyle, L::N_TAB_VARS_ELE) ;
    Kokkos::View<double****> mu ("mu",  NR, NT, NM, L::N_TAB_VARS_MUON) ;
    auto hb = Kokkos::create_mirror_view(bar) ;
    auto he = Kokkos::create_mirror_view(ele) ;
    Kokkos::deep_copy(hb, 0.) ; Kokkos::deep_copy(he, 0.) ;
    for (int i=0; i<NR; ++i) for (int j=0; j<NT; ++j) {
        for (int k=0; k<NY; ++k) {
            double const y = yq_(k) ;
            // log mode: TABPRESS = ln(P_b + P_e(Y_e=Y_q)) with electrons, ln P_b without
            double const p = pb(i,j,k) + (add_ele ? pem(i,y) + pep(i) : 0.) ;
            hb(i,j,k,L::TABPRESS) = log_press ? std::log(p) : pb(i,j,k) ;
            hb(i,j,k,L::TABEPS)   = std::log(eps_b(i,j,k) + SHIFT) ;
        }
        for (int m=0; m<nyle; ++m) {
            double const y = yle0 + dyle*m ;
            he(i,j,m,L::TABPRESS_E_MINUS) = pem(i,y) ; he(i,j,m,L::TABPRESS_E_PLUS) = pep(i) ;
            he(i,j,m,L::TABEPS_E_MINUS)   = eps_em(i,y) ; he(i,j,m,L::TABEPS_E_PLUS) = eps_ep() ;
        }
    }
    Kokkos::deep_copy(bar, hb) ; Kokkos::deep_copy(ele, he) ;
    Kokkos::View<double*> yle("yle", nyle) ;
    auto hy = Kokkos::create_mirror_view(yle) ;
    for (int m=0; m<nyle; ++m) hy(m) = yle0 + dyle*m ;
    Kokkos::deep_copy(yle, hy) ;
    Kokkos::View<double**, grace::default_space> cold("cold", 2, L::N_CTAB_VARS) ;   // unused here
    Kokkos::View<double*,  grace::default_space> cold_lr("cold_lr", 2) ;
    return L( bar, axis("lr", NR, lr_), axis("lt", NT, lt_), axis("yq", NY, yq_),
              ele, yle,
              mu,  axis("lm", NM, [](int m) { return std::log(1e-6) + m ; }),
              cold, cold_lr,
              std::exp(lr_(NR-1)), std::exp(lr_(0)), std::exp(lt_(NT-1)), std::exp(lt_(0)),
              yq_(NY-1), yq_(0), std::exp(std::log(1e-6) + 1.), 1e-6,
              /*baryon_mass*/ 1.0, SHIFT, /*epsmin,max*/ -SHIFT, 10.0, /*hmin,max*/ 0.9, 10.0,
              /*temp_atm*/ std::exp(lt_(0)), /*ye_atm*/ 0.25, /*ymu_atm*/ 1e-6,
              /*atmo_is_beta_eq*/ false, add_ele, log_press ) ;
}

void require_matches(enthalpy_at_infinity_t const& got, expect_t const& want) {
    REQUIRE( got.h.size() == want.h.size() ) ;
    for (std::size_t k=0; k<want.h.size(); ++k) REQUIRE_THAT( got.h[k], WithinRel(want.h[k], 1e-13) ) ;
    REQUIRE( got.n_edge == want.n_edge ) ;
    REQUIRE_THAT( got.yp0, WithinRel(yq_(0), 1e-15) ) ;
    REQUIRE_THAT( got.inv_dyp, WithinRel(10.0, 1e-12) ) ;
    REQUIRE( got.axis_dev < 1e-9 ) ;
}

} // namespace

TEST_CASE("h_inf of the leptonic EOS is the cold isotherm's minimum over density", "[leptonic][edge]")
{
    for ( bool const add_ele : {true, false} ) for ( bool const log_press : {false, true} ) {
        CAPTURE(add_ele, log_press) ;
        auto const eos = make_leptonic(add_ele, log_press, NY, yq_(0), 0.1) ;
        require_matches(enthalpy_at_infinity_from(eos), brute_force(add_ele)) ;
    }
    REQUIRE( brute_force(true).n_edge == 1 ) ;   // the k = 3 column: minimum on the first node
}

TEST_CASE("an electron table on its own Y_e axis is blended linearly", "[ele_axis]")
{
    // yle = 0.075, 0.125, ...: every Y_p node lies halfway between two electron nodes
    auto const eos = make_leptonic(true, false, 9, 0.075, 0.05) ;
    require_matches(enthalpy_at_infinity_from(eos), brute_force(true)) ;
}

TEST_CASE("h_inf of the 3D tabulated EOS", "[tabulated]")
{
    using T = tabulated_eos_t ;
    Kokkos::View<double****, grace::default_space> tab("tab", NR, NT, NY, T::N_TAB_VARS) ;
    auto ht = Kokkos::create_mirror_view(tab) ;
    Kokkos::deep_copy(ht, 0.) ;
    for (int i=0; i<NR; ++i) for (int j=0; j<NT; ++j) for (int k=0; k<NY; ++k) {
        double const y = yq_(k) ;   // a with-electron table: electrons folded into P and eps
        ht(i,j,k,T::TABPRESS) = std::log(pb(i,j,k) + pem(i,y) + pep(i)) ;
        ht(i,j,k,T::TABEPS)   = std::log(eps_b(i,j,k) + eps_em(i,y) + eps_ep() + SHIFT) ;
    }
    Kokkos::deep_copy(tab, ht) ;
    Kokkos::View<double**, grace::default_space> cold("cold", 2, T::N_CTAB_VARS) ;
    Kokkos::View<double*,  grace::default_space> cold_lr("cold_lr", 2) ;
    T const eos( tab, axis("lr", NR, lr_), axis("lt", NT, lt_), axis("yq", NY, yq_), cold, cold_lr,
                 std::exp(lr_(NR-1)), std::exp(lr_(0)), std::exp(lt_(NT-1)), std::exp(lt_(0)),
                 yq_(NY-1), yq_(0), /*baryon_mass*/ 1.0, SHIFT, -SHIFT, 10.0, 0.9, 10.0,
                 std::exp(lt_(0)), 0.25, false ) ;
    require_matches(enthalpy_at_infinity_from(eos), brute_force(true)) ;
}

TEST_CASE("h_inf is linear in Y_p between the nodes and constant beyond", "[interp]")
{
    auto const hinf = enthalpy_at_infinity_from(make_leptonic(true, false, NY, yq_(0), 0.1)) ;
    for (int k=0; k<NY; ++k) REQUIRE_THAT( hinf(yq_(k)), WithinRel(hinf.h[k], 1e-13) ) ;
    REQUIRE_THAT( hinf(0.15),  WithinRel(0.5*(hinf.h[0] + hinf.h[1]), 1e-13) ) ;
    REQUIRE_THAT( hinf(0.325), WithinRel(0.75*hinf.h[2] + 0.25*hinf.h[3], 1e-13) ) ;
    REQUIRE( hinf(0.0)  == hinf.h.front() ) ;
    REQUIRE( hinf(0.6)  == hinf.h.back() ) ;
    enthalpy_at_infinity_t none ;
    none.h_const = 1.0 ;
    REQUIRE( none(0.3) == 1.0 ) ;
}

TEST_CASE("h_inf of the hybrid EOS's cold backbone is the minimum over its rows", "[hybrid][edge]")
{
    using T = tabulated_eos_t ;
    constexpr int NC = 9 ;
    constexpr double CSHIFT = 0.003 ;
    // eps_of(i) shapes h over the rows: radiation-like rise at the first rows, or monotone.
    auto const make_cold = [&] (auto const& eps_of) {
        Kokkos::View<double**, grace::default_space> tab("cold", NC, T::N_CTAB_VARS) ;
        Kokkos::View<double*,  grace::default_space> lr("cold_lr", NC) ;
        auto ht = Kokkos::create_mirror_view(tab) ;
        auto hl = Kokkos::create_mirror_view(lr) ;
        Kokkos::deep_copy(ht, 0.) ;
        for (int i=0; i<NC; ++i) {
            hl(i) = std::log(1e-14) + 1.0*i ;
            ht(i,T::CTABPRESS) = std::log(1e-4 * std::exp(hl(i))) ;   // P/rho = 1e-4 on every row
            ht(i,T::CTABEPS)   = std::log(eps_of(i) + CSHIFT) ;
        }
        Kokkos::deep_copy(tab, ht) ; Kokkos::deep_copy(lr, hl) ;
        return tabulated_cold_eos_t(tab, lr, std::exp(hl(0)), std::exp(hl(NC-1)), 1.0, CSHIFT) ;
    } ;
    auto const brute_min = [&] (auto const& eps_of) {
        double m = 1e300 ;
        for (int i=0; i<NC; ++i) m = std::min(m, 1. + eps_of(i) + 1e-4) ;
        return m ;
    } ;

    auto const bowl = [] (int i) { return -0.002 + 0.0004*(i - 4)*(i - 4) ; } ;
    auto const cold = make_cold(bowl) ;
    auto const hinf = enthalpy_at_infinity_from(cold) ;
    REQUIRE( hinf.h.empty() ) ;
    REQUIRE_THAT( hinf.h_const, WithinRel(brute_min(bowl), 1e-13) ) ;
    REQUIRE( hinf(0.12) == hinf.h_const ) ;
    REQUIRE( hinf.n_edge == 0 ) ;
    REQUIRE( cold.h_minimum > hinf.h_const + 6e-3 ) ;   // the c2p's first-row value lies above it

    auto const ramp = [] (int i) { return 0.001*i ; } ;   // minimum on the first row
    REQUIRE( enthalpy_at_infinity_from(make_cold(ramp)).n_edge == 1 ) ;
}
