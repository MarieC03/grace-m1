/**
 * @file test_leptonic_logpress.cpp
 * @brief Log interpolation of the leptonic EOS's baryon pressure on synthetic
 *        in-memory tables (leptonic_eos_4d_t, convert_baryon_press_to_log).
 *
 * With eos.tabulated_eos.linear_pressure=false the loader rewrites the baryon
 * TABPRESS to ln(P_b + P_e(Y_e=Y_q)) (ln P_b without add_ele_contribution) and
 * total_press adds the electrons back as P_e(Y_e) - P_e(Y_p).
 *
 * Coverage:
 *   [nodes]  On every baryon node the log and the signed-linear EOS give the
 *            same total pressure, including negative P_b and P_e >> |P_b|.
 *   [powerlaw] Between nodes the log mode is exact for a power law, the
 *            linear mode is not.
 *   [noele]  add_ele_contribution = false: TABPRESS = ln P_b, no shift.
 *   [guard]  Non-positive or non-finite log arguments are counted, the first
 *            is reported, and the table is left untouched.
 *   [invert] P -> T inversion round-trips in log mode.
 *   [csnd2]  The sound speed stays finite and inside (0,1) in log mode.
 *
 * No HDF5 file or parfile is needed.
 */
#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/physics/eos/leptonic_eos_4d.hh>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace {

using namespace grace ;
using E = leptonic_eos_4d_t ;

// Axes: uniform in log rho, log T, Y_q and log Y_mu, as the interpolators require.
constexpr int NR = 8, NT = 4, NY = 5, NM = 4 ;
double lr_(int i) { return std::log(1e-6) + 0.5*i ; }
double lt_(int j) { return std::log(0.5)  + 0.5*j ; }
double yq_(int k) { return 0.05 + 0.1*k ; }
double lm_(int m) { return std::log(1e-6) + m*(std::log(0.1) - std::log(1e-6))/(NM-1) ; }

// Electron (e- + e+) pressure: degenerate part plus radiation-like part.
constexpr double A_E = 1e-3, B_E = 1e-12 ;
double pe_of(double rho, double T, double y, bool ye_dependent) {
    return A_E*std::pow(rho*(ye_dependent ? y : 1.0), 4./3.) + B_E*std::pow(T, 4) ;
}

struct tables_t {
    Kokkos::View<double****> bar, ele, mu ;
    Kokkos::View<double*> lr, lt, yq, lm ;
    Kokkos::View<double**, grace::default_space> cold ;
    Kokkos::View<double*,  grace::default_space> cold_lr ;
    std::vector<double> pb ;   // host copy of the signed P_b, index (i*NT + j)*NY + k
    std::vector<double> pe ;   // host copy of P_e at Y_e = Y_q, same index
} ;

inline int flat(int i, int j, int k) { return (i*NT + j)*NY + k ; }

// P_b as a function of (rho, T, Y_q, P_e(Y_e=Y_q)); electrons with or without Y_e dependence.
tables_t make_tables(std::function<double(double,double,double,double)> const& pb_of,
                     bool const ye_dependent)
{
    tables_t t ;
    t.bar = Kokkos::View<double****>("bar", NR, NT, NY, E::N_TAB_VARS_BARYON) ;
    t.ele = Kokkos::View<double****>("ele", NR, NT, NY, E::N_TAB_VARS_ELE) ;
    t.mu  = Kokkos::View<double****>("mu",  NR, NT, NM, E::N_TAB_VARS_MUON) ;
    t.lr  = Kokkos::View<double*>("lr", NR) ;
    t.lt  = Kokkos::View<double*>("lt", NT) ;
    t.yq  = Kokkos::View<double*>("yq", NY) ;
    t.lm  = Kokkos::View<double*>("lm", NM) ;
    t.cold    = Kokkos::View<double**, grace::default_space>("cold", 2, E::N_CTAB_VARS) ;
    t.cold_lr = Kokkos::View<double*,  grace::default_space>("cold_lr", 2) ;
    t.pb.assign(NR*NT*NY, 0.) ;
    t.pe.assign(NR*NT*NY, 0.) ;

    auto hb = Kokkos::create_mirror_view(t.bar) ;
    auto he = Kokkos::create_mirror_view(t.ele) ;
    auto hm = Kokkos::create_mirror_view(t.mu) ;
    auto hlr = Kokkos::create_mirror_view(t.lr) ;
    auto hlt = Kokkos::create_mirror_view(t.lt) ;
    auto hyq = Kokkos::create_mirror_view(t.yq) ;
    auto hlm = Kokkos::create_mirror_view(t.lm) ;
    auto hc  = Kokkos::create_mirror_view(t.cold) ;
    auto hcl = Kokkos::create_mirror_view(t.cold_lr) ;
    Kokkos::deep_copy(hb, 0.) ; Kokkos::deep_copy(he, 0.) ; Kokkos::deep_copy(hm, 0.) ;

    for (int i=0; i<NR; ++i) hlr(i) = lr_(i) ;
    for (int j=0; j<NT; ++j) hlt(j) = lt_(j) ;
    for (int k=0; k<NY; ++k) hyq(k) = yq_(k) ;
    for (int m=0; m<NM; ++m) hlm(m) = lm_(m) ;

    for (int i=0; i<NR; ++i) for (int j=0; j<NT; ++j) {
        double const rho = std::exp(lr_(i)), T = std::exp(lt_(j)) ;
        for (int k=0; k<NY; ++k) {
            double const y  = yq_(k) ;
            double const pe = pe_of(rho, T, y, ye_dependent) ;
            double const pb = pb_of(rho, T, y, pe) ;
            t.pb[flat(i,j,k)] = pb ; t.pe[flat(i,j,k)] = pe ;
            hb(i,j,k,E::TABPRESS)   = pb ;
            hb(i,j,k,E::TABEPS)     = std::log(0.01 + 0.002*T*T) ;   // energy_shift = 0
            hb(i,j,k,E::TABCSND2)   = 0.1 ;
            hb(i,j,k,E::TABENTROPY) = 0.1*T ;
            hb(i,j,k,E::TABXA) = hb(i,j,k,E::TABXH) = hb(i,j,k,E::TABXN) = hb(i,j,k,E::TABXP) = 0.25 ;
            hb(i,j,k,E::TABABAR) = 1. ; hb(i,j,k,E::TABZBAR) = 0.5 ;
            he(i,j,k,E::TABMUELE) = 1. ;
            he(i,j,k,E::TABYLE_MINUS) = y ;
            he(i,j,k,E::TABPRESS_E_MINUS) = 0.9*(pe - B_E*std::pow(T,4)) + 0.5*B_E*std::pow(T,4) ;
            he(i,j,k,E::TABPRESS_E_PLUS)  = 0.1*(pe - B_E*std::pow(T,4)) + 0.5*B_E*std::pow(T,4) ;
            he(i,j,k,E::TABEPS_E_MINUS) = 1e-4*T*(1.+y) ;
            he(i,j,k,E::TABS_E_MINUS)   = 1e-3*T ;
        }
        for (int m=0; m<NM; ++m) {
            double const ymu = std::exp(lm_(m)) ;
            hm(i,j,m,E::TABMUMU) = 105.66 ;
            hm(i,j,m,E::TABYMU_MINUS) = ymu ;
            hm(i,j,m,E::TABPRESS_MU_MINUS) = 0.5e-3*rho*T*ymu ;
            hm(i,j,m,E::TABPRESS_MU_PLUS)  = 0.5e-3*rho*T*ymu ;
            hm(i,j,m,E::TABEPS_MU_MINUS)   = 1e-5*T ;
            hm(i,j,m,E::TABS_MU_MINUS)     = 1e-4*T ;
        }
    }
    for (int r=0; r<2; ++r) {   // the cold slice is not used here, but must have two rows
        hcl(r) = lr_(r*(NR-1)) ;
        hc(r,E::CTABTEMP) = lt_(0) ; hc(r,E::CTABYE) = 0.1 ; hc(r,E::CTABYMU) = 1e-6 ;
        hc(r,E::CTABPRESS) = std::log(1e-9) ; hc(r,E::CTABEPS) = std::log(0.01) ;
        hc(r,E::CTABCSND2) = 0.1 ; hc(r,E::CTABENTROPY) = 0.01 ;
    }
    Kokkos::deep_copy(t.bar, hb) ; Kokkos::deep_copy(t.ele, he) ; Kokkos::deep_copy(t.mu, hm) ;
    Kokkos::deep_copy(t.lr, hlr) ; Kokkos::deep_copy(t.lt, hlt) ;
    Kokkos::deep_copy(t.yq, hyq) ; Kokkos::deep_copy(t.lm, hlm) ;
    Kokkos::deep_copy(t.cold, hc) ; Kokkos::deep_copy(t.cold_lr, hcl) ;
    return t ;
}

Kokkos::View<double****> copy_of(Kokkos::View<double****> const& v) {
    Kokkos::View<double****> c("bar_copy", v.extent(0), v.extent(1), v.extent(2), v.extent(3)) ;
    Kokkos::deep_copy(c, v) ;
    return c ;
}

E make_eos(tables_t const& t, Kokkos::View<double****> const& bar, bool add_ele, bool log_press) {
    return E( bar, t.lr, t.lt, t.yq,
              t.ele, t.yq,
              t.mu,  t.lm,
              t.cold, t.cold_lr,
              std::exp(lr_(NR-1)), std::exp(lr_(0)),
              std::exp(lt_(NT-1)), std::exp(lt_(0)),
              yq_(NY-1), yq_(0),
              std::exp(lm_(NM-1)), std::exp(lm_(0)),
              /*baryon_mass*/ 1.0, /*energy_shift*/ 0.0,
              /*epsmin,max*/ 0.0, 10.0, /*hmin,max*/ 1.0, 10.0,
              /*temp_atm*/ std::exp(lt_(0)), /*ye_atm*/ 0.25, /*ymu_atm*/ std::exp(lm_(0)),
              /*atmo_is_beta_eq*/ false, add_ele, log_press ) ;
}

struct query_t { double rho, temp, ye, ymu ; } ;

// Evaluate f(eos, query) on the device for every query.
template <typename F>
std::vector<double> eval(E const& eos, std::vector<query_t> const& q, F const f)
{
    Kokkos::View<query_t*> dq("q", q.size()) ;
    auto hq = Kokkos::create_mirror_view(dq) ;
    for (std::size_t n=0; n<q.size(); ++n) hq(n) = q[n] ;
    Kokkos::deep_copy(dq, hq) ;
    Kokkos::View<double*> dr("r", q.size()) ;
    Kokkos::parallel_for("eval", Kokkos::RangePolicy<>(0, q.size()),
        KOKKOS_LAMBDA (int const n) { dr(n) = f(eos, dq(n)) ; }) ;
    auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dr) ;
    return std::vector<double>(hr.data(), hr.data() + q.size()) ;
}

struct press_f {
    KOKKOS_INLINE_FUNCTION double operator()(E const& eos, query_t q) const {
        eos_err_t err ;
        return eos.press__temp_rho_ye_ymu(q.temp, q.rho, q.ye, q.ymu, err) ;
    }
} ;
struct csnd2_f {
    KOKKOS_INLINE_FUNCTION double operator()(E const& eos, query_t q) const {
        eos_err_t err ; double eps, cs2 ;
        eos.press_eps_csnd2__temp_rho_ye_ymu(eps, cs2, q.temp, q.rho, q.ye, q.ymu, err) ;
        return cs2 ;
    }
} ;
// T -> P -> T: returns the recovered temperature.
struct invert_f {
    KOKKOS_INLINE_FUNCTION double operator()(E const& eos, query_t q) const {
        eos_err_t err ; double h, cs2, T1, s ;
        double t = q.temp, r = q.rho, y = q.ye, m = q.ymu ;
        double p = eos.press__temp_rho_ye_ymu(t, r, y, m, err) ;
        eos.eps_h_csnd2_temp_entropy__press_rho_ye_ymu(h, cs2, T1, s, p, r, y, m, err) ;
        return T1 ;
    }
} ;

// Off-node queries: rho mid-cell, T and Y_e between nodes, muons dilute and resolved.
std::vector<query_t> off_node_queries() {
    std::vector<query_t> q ;
    for (int i=0; i<NR-1; ++i) for (int j=0; j<NT-1; ++j)
    for (double ymu : {1e-6, 0.02}) {
        q.push_back({ std::exp(lr_(i) + 0.25), std::exp(lt_(j) + 0.185), yq_(1) + 0.037, ymu }) ;
        q.push_back({ std::exp(lr_(i) + 0.25), std::exp(lt_(j) + 0.41),  yq_(3) + 0.061, ymu }) ;
    }
    return q ;
}

// Power law with ln S linear in (ln rho, ln T, Y_q): trilinear log interpolation is exact.
constexpr double K_S = 1.0, C_S = 2.0 ;
double s_power(double rho, double T, double y) { return K_S*rho*rho*std::sqrt(T)*std::exp(C_S*y) ; }

} // namespace

TEST_CASE("leptonic log pressure: log and linear agree on every table node",
          "[leptonic][logpress][nodes]")
{
    // P_b negative at low rho, plus hand-set nodes with |P_b| << P_e and S << P_e.
    auto pb_of = [](double rho, double, double, double pe) { return rho*rho - 0.3*pe ; } ;
    tables_t t = make_tables(pb_of, /*ye_dependent=*/true) ;
    {
        auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
        auto set = [&](int i, int j, int k, double pb) { hb(i,j,k,E::TABPRESS) = pb ; t.pb[flat(i,j,k)] = pb ; } ;
        set(3,1,2,  1e-9*t.pe[flat(3,1,2)]) ;
        set(4,2,3, -1e-9*t.pe[flat(4,2,3)]) ;
        set(5,1,1, (1e-6 - 1.)*t.pe[flat(5,1,1)]) ;   // S = 1e-6 P_e
        Kokkos::deep_copy(t.bar, hb) ;
    }
    int n_negative = 0 ;
    for (double pb : t.pb) n_negative += (pb < 0.) ;
    REQUIRE(n_negative > 10) ;

    E const lin = make_eos(t, t.bar, true, false) ;
    auto const bar_log = copy_of(t.bar) ;
    E const lg  = make_eos(t, bar_log, true, true) ;
    auto const rep = convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, true) ;
    REQUIRE(rep.n_bad == 0) ;

    std::vector<query_t> q ; std::vector<int> node ;
    for (int i=0; i<NR; ++i) for (int j=0; j<NT; ++j) for (int k=1; k<NY; ++k)
    for (double ymu : {1e-6, 0.02}) {    // Y_e chosen so that Y_p = Y_e + Y_mu is the node
        q.push_back({ std::exp(lr_(i)), std::exp(lt_(j)), yq_(k) - ymu, ymu }) ;
        node.push_back(flat(i,j,k)) ;
    }
    auto const p_lin = eval(lin, q, press_f{}) ;
    auto const p_log = eval(lg,  q, press_f{}) ;
    for (std::size_t n=0; n<q.size(); ++n) {
        double const scale = std::fabs(t.pb[node[n]]) + 2.*t.pe[node[n]] + std::fabs(p_lin[n]) ;
        INFO("node " << node[n] << " P_b " << t.pb[node[n]] << " P_e " << t.pe[node[n]]
             << " linear " << p_lin[n] << " log " << p_log[n]) ;
        REQUIRE(std::isfinite(p_log[n])) ;
        REQUIRE(std::fabs(p_log[n] - p_lin[n]) <= 1e-13*scale) ;
    }
}

TEST_CASE("leptonic log pressure: exact for a power law between nodes, linear is not",
          "[leptonic][logpress][powerlaw]")
{
    // Electrons independent of Y_e, so the Y_p -> Y_e correction vanishes and P = S.
    auto pb_of = [](double rho, double T, double y, double pe) { return s_power(rho, T, y) - pe ; } ;
    tables_t t = make_tables(pb_of, /*ye_dependent=*/false) ;
    int n_negative = 0 ;
    for (double pb : t.pb) n_negative += (pb < 0.) ;
    REQUIRE(n_negative > 0) ;

    E const lin = make_eos(t, t.bar, true, false) ;
    auto const bar_log = copy_of(t.bar) ;
    E const lg  = make_eos(t, bar_log, true, true) ;
    REQUIRE(convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, true).n_bad == 0) ;

    std::vector<query_t> q ;
    for (auto const& x : off_node_queries()) if (x.ymu < 1e-3) q.push_back(x) ;   // muons dilute
    auto const p_lin = eval(lin, q, press_f{}) ;
    auto const p_log = eval(lg,  q, press_f{}) ;
    for (std::size_t n=0; n<q.size(); ++n) {
        double const exact = s_power(q[n].rho, q[n].temp, q[n].ye + q[n].ymu) ;
        INFO("rho " << q[n].rho << " T " << q[n].temp << " ye " << q[n].ye
             << " exact " << exact << " log " << p_log[n] << " linear " << p_lin[n]) ;
        REQUIRE(std::fabs(p_log[n]/exact - 1.) < 1e-12) ;
        REQUIRE(std::fabs(p_lin[n]/exact - 1.) > 1e-3) ;
    }
}

TEST_CASE("leptonic log pressure: add_ele_contribution = false stores ln P_b",
          "[leptonic][logpress][noele]")
{
    auto pb_of = [](double rho, double T, double y, double) { return s_power(rho, T, y) ; } ;
    tables_t t = make_tables(pb_of, /*ye_dependent=*/true) ;
    auto const bar_log = copy_of(t.bar) ;
    E const lg = make_eos(t, bar_log, /*add_ele=*/false, true) ;
    REQUIRE(convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, false).n_bad == 0) ;

    auto const hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bar_log) ;
    for (int i=0; i<NR; ++i) for (int j=0; j<NT; ++j) for (int k=0; k<NY; ++k) {
        double const expect = std::log(t.pb[flat(i,j,k)]) ;
        REQUIRE(std::fabs(hb(i,j,k,E::TABPRESS) - expect) <= 4.*std::numeric_limits<double>::epsilon()*std::fabs(expect)) ;
    }
    std::vector<query_t> q ;
    for (auto const& x : off_node_queries()) if (x.ymu < 1e-3) q.push_back(x) ;
    auto const p = eval(lg, q, press_f{}) ;
    for (std::size_t n=0; n<q.size(); ++n) {
        double const exact = s_power(q[n].rho, q[n].temp, q[n].ye + q[n].ymu) ;
        INFO("exact " << exact << " log " << p[n]) ;
        REQUIRE(std::fabs(p[n]/exact - 1.) < 1e-12) ;
    }
}

TEST_CASE("leptonic log pressure: non-positive arguments are reported, table untouched",
          "[leptonic][logpress][guard]")
{
    auto pb_of = [](double rho, double, double, double pe) { return rho*rho - 0.3*pe ; } ;

    SECTION("with electrons") {
        tables_t t = make_tables(pb_of, true) ;
        {
            auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
            hb(2,2,2,E::TABPRESS) = -(t.pe[flat(2,2,2)] + 1.) ;
            hb(1,1,1,E::TABPRESS) = -(t.pe[flat(1,1,1)] + 1.) ;
            hb(6,3,4,E::TABPRESS) = std::numeric_limits<double>::quiet_NaN() ;
            Kokkos::deep_copy(t.bar, hb) ;
        }
        auto const before = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
        E const lg = make_eos(t, t.bar, true, true) ;
        auto const rep = convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, true) ;
        REQUIRE(rep.n_bad == 3) ;
        REQUIRE(rep.i == 1) ; REQUIRE(rep.j == 1) ; REQUIRE(rep.k == 1) ;
        REQUIRE(rep.lrho  == lr_(1)) ;
        REQUIRE(rep.ltemp == lt_(1)) ;
        REQUIRE(rep.yq    == yq_(1)) ;
        REQUIRE(rep.p_baryon == before(1,1,1,E::TABPRESS)) ;
        REQUIRE(std::fabs(rep.p_ele/t.pe[flat(1,1,1)] - 1.) < 1e-12) ;
        auto const after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
        REQUIRE(std::memcmp(before.data(), after.data(), before.span()*sizeof(double)) == 0) ;
    }
    SECTION("without electrons") {
        tables_t t = make_tables(pb_of, true) ;
        auto const before = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
        int n_nonpositive = 0 ;
        for (double pb : t.pb) n_nonpositive += !(pb > 0.) ;
        REQUIRE(n_nonpositive > 0) ;
        E const lg = make_eos(t, t.bar, false, true) ;
        auto const rep = convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, false) ;
        REQUIRE(rep.n_bad == n_nonpositive) ;
        REQUIRE(rep.p_ele == 0.) ;
        REQUIRE(!(rep.p_baryon > 0.)) ;
        auto const after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.bar) ;
        REQUIRE(std::memcmp(before.data(), after.data(), before.span()*sizeof(double)) == 0) ;
    }
}

TEST_CASE("leptonic log pressure: P -> T inversion and sound speed",
          "[leptonic][logpress][invert][csnd2]")
{
    auto pb_of = [](double rho, double T, double y, double pe) { return s_power(rho, T, y) - pe ; } ;
    tables_t t = make_tables(pb_of, /*ye_dependent=*/true) ;
    auto const bar_log = copy_of(t.bar) ;
    E const lg = make_eos(t, bar_log, true, true) ;
    REQUIRE(convert_baryon_press_to_log(lg.baryon_table, lg.ele_table, true).n_bad == 0) ;

    auto const q  = off_node_queries() ;
    auto const T1 = eval(lg, q, invert_f{}) ;
    auto const cs = eval(lg, q, csnd2_f{}) ;
    for (std::size_t n=0; n<q.size(); ++n) {
        INFO("rho " << q[n].rho << " T " << q[n].temp << " ye " << q[n].ye << " ymu " << q[n].ymu
             << " T back " << T1[n] << " cs2 " << cs[n]) ;
        REQUIRE(std::fabs(T1[n]/q[n].temp - 1.) < 1e-9) ;
        REQUIRE(std::isfinite(cs[n])) ;
        REQUIRE(cs[n] > 1e-10) ;
        REQUIRE(cs[n] < 1.) ;
    }
}
