#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/neutrino_pair_update.hh>
#include <grace/physics/m1.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace grace;
using namespace grace::pairs;
using Catch::Matchers::WithinRel;

namespace {
void fd(grid const& g,double T,double eta,double* f,double& n,double& J) {
    n=J=0;
    for(int i=0;i<g.n;++i) {
        f[i]=occupation(g.e[i]/T-eta);
        n+=g.w[i]*f[i]; J+=g.w[i]*g.e[i]*f[i];
    }
}
material const matter{10,20,6e37,0.8,0.2};
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
TEST_CASE("Exact thermal cache checks every input and isolates tau mutation", "[pairs][thermal_cache]") {
    channels const ch{true,true,true};
    kernel original,got;
    REQUIRE(original.init(matter,ch,8,19));
    thermal_kernel_cache<4,4> cache;
    REQUIRE_FALSE(cache.find(matter,ch,original.g,false,got));
    cache.insert(matter,ch,false,original);
    REQUIRE(cache.find(matter,ch,original.g,false,got));
    got.symmetrize(); got.a[0][1]=-1; got.boltz[0]=-1;
    REQUIRE(cache.find(matter,ch,original.g,false,got));
    auto equal=[&](kernel const& a,kernel const& b) {
        REQUIRE(a.g.n==b.g.n); REQUIRE(a.g.scale==b.g.scale);
        for(int i=0;i<a.g.n;++i) {
            REQUIRE(a.g.e[i]==b.g.e[i]); REQUIRE(a.g.w[i]==b.g.w[i]);
            REQUIRE(a.boltz[i]==b.boltz[i]);
            for(int j=0;j<a.g.n;++j) REQUIRE(a.a[i][j]==b.a[i][j]);
        }
    };
    equal(got,original);
    for(int kind=0;kind<5;++kind) {
        auto changed=matter;
        double* fields[]={&changed.T,&changed.mu_e,&changed.nb,&changed.yn,&changed.yp};
        *fields[kind]=std::nextafter(*fields[kind],2*(*fields[kind]));
        REQUIRE_FALSE(cache.find(changed,ch,original.g,false,got));
    }
    for(int kind=0;kind<4;++kind) {
        auto changed=original.g;
        if(kind==0) changed.scale=std::nextafter(changed.scale,2*changed.scale);
        if(kind==1) changed.e[0]=std::nextafter(changed.e[0],2*changed.e[0]);
        if(kind==2) changed.w[0]=std::nextafter(changed.w[0],2*changed.w[0]);
        if(kind==3) --changed.n;
        REQUIRE_FALSE(cache.find(matter,ch,changed,false,got));
    }
    REQUIRE_FALSE(cache.find(matter,{false,true,true},original.g,false,got));
    REQUIRE_FALSE(cache.find(matter,{true,false,true},original.g,false,got));
    REQUIRE_FALSE(cache.find(matter,{true,true,false},original.g,false,got));
    REQUIRE_FALSE(cache.find(matter,ch,original.g,true,got));
    kernel symmetric; REQUIRE(symmetric.init(matter,ch,8,19,true));
    cache.insert(matter,ch,true,symmetric);
    REQUIRE(cache.find(matter,ch,original.g,true,got)); equal(got,symmetric);
    for(int i=0;i<4;++i) {
        auto m=matter; m.T+=i+1;
        kernel k; REQUIRE(k.init(m,ch,8,19)); cache.insert(m,ch,false,k);
    }
    REQUIRE_FALSE(cache.find(matter,ch,original.g,false,got));
    auto last=matter; last.T+=4;
    REQUIRE(cache.find(last,ch,original.g,false,got));
    kernel fresh; REQUIRE(fresh.init(last,ch,8,19)); equal(got,fresh);

    // A supplied quadrature rule, not just its nominal order, determines the key.
    grid_rule rule; rule.init(8); rule.derivatives[0]*=1.001;
    REQUIRE(fresh.init(matter,ch,8,19,false,nullptr,&rule));
    diagnostic d;
    REQUIRE(init_thermal_kernel(got,matter,ch,8,19,false,&d,&rule,true)); equal(got,fresh);
    got.symmetrize();
    REQUIRE(init_thermal_kernel(got,matter,ch,8,19,false,&d,&rule,true)); equal(got,fresh);
    auto bad=matter; bad.T=-1;
    REQUIRE_FALSE(init_thermal_kernel(got,bad,ch,8,19,false,&d,nullptr,true));
    REQUIRE(d.code==failure_code::kernel_input);
    REQUIRE_FALSE(init_thermal_kernel(got,matter,ch,0,19,false,&d,nullptr,true));
    REQUIRE(d.code==failure_code::kernel_input);
    REQUIRE(init_thermal_kernel(got,matter,ch,8,19,false,&d,&rule,true));
    REQUIRE(d.code==failure_code::none);
}

TEST_CASE("Shared thermal cache preserves cold misses and warm hits across threads", "[pairs][thermal_cache]") {
    material const m{13,-17,7e36,0.7,0.3}; channels const ch{true,true,true};
    kernel reference; REQUIRE(reference.init(m,ch,8,21));
    for(int repeat=0;repeat<2;++repeat) {
        Kokkos::View<int*> errors("thermal cache errors",64);
        Kokkos::parallel_for("concurrent thermal cache",64,KOKKOS_LAMBDA(int a) {
            kernel k;
            int bad=!init_thermal_kernel(k,m,ch,8,21,false,nullptr,nullptr,true);
            if(!bad) for(int i=0;i<k.g.n;++i) {
                bad+=k.boltz[i]!=reference.boltz[i];
                for(int j=0;j<k.g.n;++j) bad+=k.a[i][j]!=reference.a[i][j];
            }
            k.symmetrize(); // Mutating one return value must not affect other threads.
            errors(a)=bad;
        });
        auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},errors);
        for(int i=0;i<64;++i) REQUIRE(host(i)==0);
    }
}
#endif

TEST_CASE("Pair failures retain reconstruction context and bounded device samples", "[pairs][diagnostics]") {
    kernel k; diagnostic d;
    auto bad=matter; bad.T=-1;
    REQUIRE_FALSE(k.init(bad,{true,false,false},16,10,false,&d));
    REQUIRE(d.code==failure_code::kernel_input);
    REQUIRE(k.init(matter,{true,false,false},16,10,false,&d));
    REQUIRE(d.code==failure_code::none);
    double f[max_order];
    REQUIRE_FALSE(reconstruct(k.g,-1,1,f,&d));
    REQUIRE(d.code==failure_code::moment_input);
    REQUIRE(d.n==-1);
    REQUIRE(d.J==1);
    double n[4]={0,0,-1,0},J[4]={0,0,1,0},opacity[4];
    REQUIRE_FALSE(leptonic_transport_opacities(k,nullptr,true,n,J,opacity,&d));
    REQUIRE(d.species==2);
    REQUIRE(d.code==failure_code::moment_input);
    // Positive moments whose mean lies below every quadrature node cannot
    // be represented. This is a reconstruction failure, not a kernel error.
    REQUIRE_FALSE(reconstruct(k.g,1e20,1e20*k.g.e[0]*0.5,f,&d));
    REQUIRE(d.code==failure_code::moment_grid_range);
    REQUIRE(d.iteration==-1);
    REQUIRE(d.emin==k.g.e[0]);
    REQUIRE(d.emax==k.g.e[k.g.n-1]);
    REQUIRE(reconstruct(k.g,0,0,f,&d));
    REQUIRE(d.code==failure_code::none);

    failure_buffer buffer; buffer.allocate();
    Kokkos::parallel_for("pair_failure_capture_test",19,KOKKOS_LAMBDA(int cell) {
        failure_record r;
        r.i=cell; r.detail.code=failure_code::moment_input; r.detail.n=-cell-1;
        buffer.save(r);
    });
    auto count=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),buffer.count);
    auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),buffer.records);
    REQUIRE(count()==19);
    for(int slot=0;slot<failure_buffer::capacity;++slot) {
        REQUIRE(records(slot).i>=0);
        REQUIRE(records(slot).i<19);
        REQUIRE(records(slot).detail.n==-records(slot).i-1);
        REQUIRE(records(slot).detail.code==failure_code::moment_input);
        for(int other=0;other<slot;++other) REQUIRE(records(other).i!=records(slot).i);
    }
}

TEST_CASE("Pair units match existing EAS", "[pairs]") {
    REQUIRE_THAT(energy_unit/time_unit,WithinRel(Q_mev_to_code(1,1),1e-14));
    REQUIRE_THAT(number_unit/time_unit,WithinRel(R_to_code(1,1),1e-14));
}

TEST_CASE("Hunter moments exceed the pair quadrature support", "[pairs][diagnostics]") {
    // rank 2, iteration 0, q=80, (28,23,15), from the 2026-10-01 crash.
    double const n=3.71514358939047201e27,J=4.07411727181158377e34;
    for(int order:{16,24,32}) {
        grid g; g.init(order,2.79366770784541222e3);
        REQUIRE(J/n>g.e[order-1]);
        diagnostic d; double f[max_order];
        REQUIRE_FALSE(reconstruct(g,n,J,f,&d));
        REQUIRE(d.code==failure_code::moment_grid_range);
        REQUIRE(d.iteration==-1);
        // The logged antineutrino moments are representable on the same grid.
        REQUIRE(reconstruct(g,2.12570665646473441e31,2.87870240689971931e33,f,&d));
    }
}

TEST_CASE("Pair quadrature and moment reconstruction", "[pairs]") {
    grid g; g.init(32,10);
    double f[max_order],r[max_order],n,J;
    fd(g,10,0,f,n,J);
    REQUIRE_THAT(n,WithinRel(phase*1000*1.8030853547393914,2e-5));
    REQUIRE_THAT(J,WithinRel(phase*10000*5.682196976983475,3e-5));
    for(double eta: {-8.,0.,4.,12.}) {
        fd(g,10,eta,f,n,J);
        REQUIRE(reconstruct(g,n,J,r));
        double nr=0,jr=0;
        for(int i=0;i<g.n;++i) { nr+=g.w[i]*r[i]; jr+=g.w[i]*g.e[i]*r[i]; }
        REQUIRE_THAT(nr,WithinRel(n,3e-10)); REQUIRE_THAT(jr,WithinRel(J,3e-10));
    }
    REQUIRE(reconstruct(g,0,0,r));
    REQUIRE_FALSE(reconstruct(g,-1,1,r));
}

TEST_CASE("Blocked kernels obey detailed balance and kernel orientation", "[pairs]") {
    for(channels c: {channels{true,false,false},channels{false,true,false},channels{false,false,true}}) {
        kernel k; REQUIRE(k.init(matter,c,24,10));
        double f[max_order],b[max_order],n,J,nb,Jb;
        fd(k.g,10,2,f,n,J); fd(k.g,10,-2,b,nb,Jb);
        auto const s=evaluate(k,f,b);
        REQUIRE(s.emission[0]>0);
        REQUIRE(std::abs(s.energy[0])<3e-13*s.emission[0]);
        REQUIRE(std::abs(s.energy[1])<3e-13*s.emission[1]);
        REQUIRE(std::abs(s.number)<3e-13*s.emission_n[0]);
        REQUIRE_THAT(s.emission[0],WithinRel(s.loss[0],3e-13));
        REQUIRE_THAT(s.emission_n[1],WithinRel(s.loss_n[1],3e-13));
        for(int i=0;i<k.g.n;++i) f[i]=b[i]=0;
        auto const empty=evaluate(k,f,b);
        REQUIRE(empty.number>0); REQUIRE(empty.loss[0]==0);
        // One occupied partner is not enough for inverse annihilation.
        fd(k.g,16,1,f,n,J);
        auto const single=evaluate(k,f,b);
        REQUIRE(single.number>=0);
        REQUIRE(single.number<empty.number);
    }
    MyEOSParams eos{}; eos.temp=matter.T; eos.mu_e=matter.mu_e;
    PairKernelParams p{}; p.omega=12; p.omega_prime=27;
    auto const ref=PairKernels(&eos,&p);
    REQUIRE_THAT(absorption_kernel(matter,{true,false,false},12,27),WithinRel(ref.abs[id_nux]*1e-21,1e-12));
    REQUIRE_THAT(absorption_kernel(matter,{true,false,false},27,12),WithinRel(ref.abs[id_anux]*1e-21,1e-12));
}

// Every element of a built kernel is the per-element reference bit for bit: the e+e- Fermi-Dirac
// integrals are cached across the matrix and the bremsstrahlung factors hoisted, in the library's order.
TEST_CASE("Kernel build equals the per-element reference bitwise", "[pairs][kernel]") {
    unsigned long long s=0x9E3779B97F4A7C15ull;
    auto const rnd=[&]{ s^=s<<13; s^=s>>7; s^=s<<17; return double(s>>11)/9007199254740992.0; };
    auto const logu=[&](double lo,double hi){ return lo*std::pow(hi/lo,rnd()); };
    long checked=0, built=0;
    for(int trial=0;trial<40;++trial) {
        material m{logu(0.1,50),logu(0.01,300),logu(1e27,3e38),0,0};
        m.yn=0.1+0.8*rnd(); m.yp=(1-m.yn)*rnd();
        int const order=8+8*(trial%4);
        double const scale=m.T*logu(1,400);
        for(int ch=1;ch<8;++ch) {
            channels const c{bool(ch&1),bool(ch&2),bool(ch&4)};
            kernel k,ks;
            bool const ok=k.init(m,c,order,scale);
            REQUIRE(ks.init(m,c,order,scale,true)==ok);
            if(!ok) continue;
            ++built;
            kernel sym=k; sym.symmetrize();
            for(int i=0;i<k.g.n;++i) for(int j=0;j<k.g.n;++j) {
                double const r=absorption_kernel(m,c,k.g.e[i],k.g.e[j]);
                REQUIRE(k.a[i][j]==r);
                if(i<=j) REQUIRE(ks.a[i][j]==0.5*(r+absorption_kernel(m,c,k.g.e[j],k.g.e[i])));
                else REQUIRE(ks.a[i][j]==ks.a[j][i]);
                REQUIRE(sym.a[i][j]==ks.a[i][j]);
                ++checked;
            }
        }
    }
    INFO("kernels built " << built << ", elements checked " << checked);
    REQUIRE(built>200); REQUIRE(checked>100000);
}

TEST_CASE("Evolved pairs conserve number difference in an implicit M1 stage", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,false},16,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double f[max_order],n,J,old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        fd(k.g,12-s*3,0.3-s,f,n,J);
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
    }
    for(double dt: {1e-8,1e-6,1e-4,1e-2}) {
        INFO("dt [s] = " << dt);
        REQUIRE(implicit_update(k,metric,base,eas,old,out,dt*time_unit,res));
        REQUIRE(res<2e-8);
        REQUIRE_THAT(out[4]-out[9],WithinRel(old[4]-old[9],5e-10));
        REQUIRE(out[0]>0); REQUIRE(out[5]>0); REQUIRE(out[4]>0); REQUIRE(out[9]>0);
    }
}

// Cold dilute surface cell of the Mac HOT-TOV (T = 0.114 MeV, rho = 9e3 g/cc, mu_mu = 103.6 MeV):
// eta_numu = +782, so the anti-nu_mu LTE weights exp(-782 - e/T) underflow to 0.
TEST_CASE("LTE pair rates stay finite when one orientation's occupation underflows", "[pairs][lte]") {
    material const cold{0.1143, 0.0304, 5.6e27, 0.64, 0.36};
    kernel k; REQUIRE(k.init(cold,{true,true,true},16,17.9));
    auto const eval=[&](double const* f,double const* fb){ return evaluate(k,f,fb); };
    // by hand, the direct FD sums: valid as long as nothing underflows (|eta| <= ~700 here)
    auto const direct=[&](double eta,lte_rates& r) {
        double f[2][max_order],n[2]={0,0},J[2]={0,0};
        for(int a=0;a<2;++a) fd(k.g,cold.T,a==0?eta:-eta,f[a],n[a],J[a]);
        auto const ps=evaluate(k,f[0],f[1]);
        for(int a=0;a<2;++a) { r.emission[a]=ps.emission[a]; r.emission_n[a]=ps.emission_n[a];
                               r.kappa_E[a]=ps.loss[a]/J[a]; r.kappa_N[a]=ps.loss_n[a]/n[a]; }
    };
    lte_rates r,d;
    for(double eta: {-700.,-600.,-30.,-5.,0.3,5.,30.,600.,700.}) {   // both forms valid: must agree
        INFO("eta = " << eta);
        REQUIRE(lte_pair_rates(k.g,cold.T,eta,-eta,eval,r)); direct(eta,d);
        for(int a=0;a<2;++a) {
            REQUIRE_THAT(r.kappa_E[a],WithinRel(d.kappa_E[a],1e-10));
            REQUIRE_THAT(r.kappa_N[a],WithinRel(d.kappa_N[a],1e-10));
            REQUIRE(r.emission[a]==d.emission[a]); REQUIRE(r.emission_n[a]==d.emission_n[a]);
        }
    }
    for(double eta: {-900.,-782.,782.,900.}) {   // the direct sums underflow: finite, non-negative
        INFO("eta = " << eta);
        REQUIRE(lte_pair_rates(k.g,cold.T,eta,-eta,eval,r));
        for(int a=0;a<2;++a) {
            REQUIRE(std::isfinite(r.kappa_E[a])); REQUIRE(r.kappa_E[a]>=0);
            REQUIRE(std::isfinite(r.kappa_N[a])); REQUIRE(r.kappa_N[a]>=0);
            REQUIRE(std::isfinite(r.emission[a])); REQUIRE(r.emission[a]>=0);
        }
    }
    // the suppressed orientation's opacity is the annihilation rate on the partner's filled sea: it changes
    // only through the partner's Fermi surface, smoothly, across the underflow
    lte_rates a,b; REQUIRE(lte_pair_rates(k.g,cold.T,740,-740,eval,a)); REQUIRE(lte_pair_rates(k.g,cold.T,760,-760,eval,b));
    REQUIRE_THAT(b.kappa_E[1],WithinRel(a.kappa_E[1],0.2)); REQUIRE_THAT(b.kappa_N[1],WithinRel(a.kappa_N[1],0.2));
}

// Hunter/Mac HOT-TOV core (T = 20 MeV, rho = 1.04e-3) from floor radiation: h*kappa_a = 720, the scaled
// unknown reaches 1e4, so the residual floor is roundoff (~1e-9) and an absolute 2e-10 target can never be met.
TEST_CASE("Evolved pairs converge from floor radiation in a stiff hot core", "[pairs][stiff]") {
    material const core{19.99999957, 167.6738138, 3.872150214e38, 0.9060033822, 0.09399661775};
    kernel k; REQUIRE(k.init(core,{true,true,true},16,20));
    metric_array_t metric({1.914363,0,0,1.914363,0,1.914363},{0,0,0},0.724636);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double old[10]={},out[10],res;
    for(int s=0;s<2;++s) { old[5*s]=base[s][ERADL]=1e-16; old[5*s+4]=base[s][NRADL]=1e-14; }
    eas[0][KAL]=2.614182e4; eas[0][KSL]=648.1693; eas[0][KANL]=1.491492e4; eas[0][ETAL]=4.522421e-4; eas[0][ETANL]=3.841020e-3;
    eas[1][KAL]=1.961922;   eas[1][KSL]=317.0434; eas[1][KANL]=0.6802474;  eas[1][ETAL]=3.394129e-8; eas[1][ETANL]=1.751877e-7;
    double const h=2.745874e-2;
    REQUIRE(implicit_update(k,metric,base,eas,old,out,h,res));
    // residual is scaled by max(old, 1e-4 LTE): the state sits at ~1e4 there and the 2e-11 moment
    // reconstruction tolerance leaves noise of ~1e4 * 2e-11 = 2e-7, i.e. 2e-11 relative to the state
    REQUIRE(res<newton_accept*1e4);
    // one stiff step takes nu_mu from the floor to the order of its thermal LTE (eta/kappa); the pair
    // production on the still-filling anti-nu_mu partner shifts the exact fixed point, so order of magnitude only
    double const Jeq=eas[0][ETAL]/eas[0][KAL], neq=eas[0][ETANL]/eas[0][KANL];
    REQUIRE(out[0]>0.5*Jeq); REQUIRE(out[0]<2*Jeq);
    REQUIRE(out[4]>0.5*neq); REQUIRE(out[4]<2*neq);
    REQUIRE(out[5]>0.1*Jeq); REQUIRE(out[5]<out[0]);   // the partner is fed by pairs only
    REQUIRE(out[9]>0.1*neq); REQUIRE(out[9]<out[4]);
    for(double v: out) REQUIRE(std::isfinite(v));
}

namespace {
// Minimal EOS stand-in for make_fugacity_state (as in test_m1_analytic_rates.cpp).
struct mock_eos_t {
    double mu_e{10.0}, mu_mu{0.0}, mu_p{5.0}, mu_n{20.0};
    double Xa{0.05}, Xh{0.05}, Xn{0.6}, Xp{0.3};
    double Abar{20.0}, Zbar{8.0};
    double mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(
        double& mumu, double& mup, double& mun, double& Xa_o, double& Xh_o, double& Xn_o, double& Xp_o,
        double& Abar_o, double& Zbar_o, double&, double&, double&, double&, eos_err_t&) const {
        mumu=mu_mu; mup=mu_p; mun=mu_n; Xa_o=Xa; Xh_o=Xh; Xn_o=Xn; Xp_o=Xp; Abar_o=Abar; Zbar_o=Zbar;
        return mu_e;
    }
};
constexpr double kXyzOrigin[3]={0,0,0};
// Electron fugacity for n_e = rho Ye / m_u at temperature T: invert n_e = 8 pi T^3/(hc)^3 [F2(eta) - F2(-eta)].
double electron_eta(double rho_cgs, double ye, double T) {
    double const ne=rho_cgs*ye/nu_constants::mnuc_cgs, c=8*pairs::pi/std::pow(nu_constants::hc_mevcm,3)*T*T*T;
    double lo=-50, hi=3000;
    for(int it=0;it<200;++it) {
        double const mid=0.5*(lo+hi), n=c*(fermi::FD<2>::get(mid)-fermi::FD<2>::get(-mid));
        (n<ne?lo:hi)=mid;
    }
    return 0.5*(lo+hi);
}
}

// PHYSICS CHECK 1: her kernel-integrated thermal emissivities at eta_nu = 0 against the analytic fits GRACE's
// standard path adds for the heavy flavours (Ruffert-type pair, Burrows-type brems, Ruffert plasmon), per
// helicity species.  These fits are themselves approximate (tens of per cent), so only the order of magnitude
// is asserted; the table is printed for inspection.
TEST_CASE("Kernel-integrated thermal emissivities against GRACE's analytic fits", "[pairs][physics]") {
    mock_eos_t eos; tau_policy_analytic_density tau;
    double const mass_scale=1.0;
    std::printf("\n  per-species emissivity Q [MeV cm^-3 s^-1] at eta_nu=0: kernel / GRACE fit (ratio)\n");
    std::printf("  %7s %8s %6s %6s | %22s | %22s | %22s\n","rho","T","Ye","eta_e","e+e- pairs","NN brems","plasmon");
    for(double T: {5.0,10.0,20.0}) for(double rho: {1e12,1e13,1e14}) {
        double const ye=0.1, eta_e=electron_eta(rho,ye,T), mu_e=eta_e*T;
        eos.mu_e=mu_e; eos.mu_mu=0; eos.mu_p=0; eos.mu_n=0; eos.Xn=1-ye; eos.Xp=ye; eos.Xa=0; eos.Xh=0;
        auto F=make_fugacity_state(eos,rho*nu_constants::RHOGF,T,ye,0.0,mass_scale,kXyzOrigin,tau);
        for(int s=0;s<NUMSPECIES;++s) F.eta_nu[s]=0;   // thermal pairs: compare at zero fugacity
        rates_accum fit_ee{}, fit_nn{}, fit_pl{};
        add_pair_process_emission(F,fit_ee); add_brems_emission(F,fit_nn); add_plasmon_decay_emission(F,fit_pl,false);
        #if GRACE_M1_NU_SPECIES >= 5
        double const per_species=2.0;      // NUX = tau + anti-tau
        #else
        double const per_species=4.0;      // NUX = all four heavy species
        #endif
        material const m{T,mu_e,rho/nu_constants::mnuc_cgs,1-ye,ye};
        double q[3], ratio[3];
        channels const ch[3]={{true,false,false},{false,true,false},{false,false,true}};
        double const fitq[3]={fit_ee.Q[NUX]/per_species,fit_nn.Q[NUX]/per_species,fit_pl.Q[NUX]/per_species};
        for(int c=0;c<3;++c) {
            kernel k; REQUIRE(k.init(m,ch[c],32,T));
            lte_rates r; REQUIRE(lte_pair_rates(k.g,T,0,0,[&](double const* f,double const* fb){ return evaluate(k,f,fb); },r));
            q[c]=0.5*(r.emission[0]+r.emission[1]); ratio[c]=q[c]/fitq[c];
        }
        std::printf("  %7.0e %8.1f %6.2f %6.2f | %9.2e %9.2e %5.2f | %9.2e %9.2e %5.2f | %9.2e %9.2e %5.2f\n",
                    rho,T,ye,eta_e,q[0],fitq[0],ratio[0],q[1],fitq[1],ratio[1],q[2],fitq[2],ratio[2]);
        REQUIRE(ratio[0]>0.2); REQUIRE(ratio[0]<5);
        REQUIRE(ratio[1]>0.2); REQUIRE(ratio[1]<5);
    }
    // cold dilute surface cell (T 0.114 MeV, mu_mu 103.6 -> eta_numu +782): the LTE target is a filled 89 MeV
    // nu_mu sea.  The massless-electron kernel still emits there (T < m_e, where it overestimates pairs), so the
    // physical criterion is the time to fill that target: it must be astronomically long.
    material const cold{0.1143,0.0304,5.6e27,0.64,0.36};
    kernel k; REQUIRE(k.init(cold,{true,true,true},16,17.9));
    lte_rates r; REQUIRE(lte_pair_rates(k.g,cold.T,782,-782,[&](double const* f,double const* fb){ return evaluate(k,f,fb); },r));
    double f[max_order],n_sea,J_sea; fd(k.g,cold.T,782,f,n_sea,J_sea);
    double const fill_time=J_sea/r.emission[0];   // seconds
    std::printf("  cold dilute cell, eta=+782: nu_mu emission %.2e MeV cm^-3 s^-1 vs LTE J %.2e -> filling time %.1e s;"
                " kappa_E nu_mu %.2e s^-1, anu_mu %.2e s^-1\n",r.emission[0],J_sea,fill_time,r.kappa_E[0],r.kappa_E[1]);
    REQUIRE(fill_time>1e12);
}

namespace {
// B1: one head-on state (rho, T, Ye) with eta_e from charge neutrality; kernel / fit of the per-species emissivity
// (q) and of the LTE opacity (k: Q/B for the fit, loss/J over c for the kernel) for the three thermal channels.
struct fit_record { double rho, T, ye, eta_e, q[3], k[3]; };
std::vector<fit_record> fits_vs_kernel_grid() {
    mock_eos_t eos; tau_policy_analytic_density tau;
    channels const ch[3]={{true,false,false},{false,true,false},{false,false,true}};
    std::vector<fit_record> out;
    for(double ye: {0.1,0.3}) for(double T: {5.0,10.0,20.0,40.0}) for(double rho: {1e11,1e12,1e13,1e14}) {
        double const eta_e=electron_eta(rho,ye,T), mu_e=eta_e*T;
        eos.mu_e=mu_e; eos.mu_mu=0; eos.mu_p=0; eos.mu_n=0; eos.Xn=1-ye; eos.Xp=ye; eos.Xa=0; eos.Xh=0;
        auto F=make_fugacity_state(eos,rho*nu_constants::RHOGF,T,ye,0.0,1.0,kXyzOrigin,tau);
        for(int s=0;s<NUMSPECIES;++s) F.eta_nu[s]=0;
        material const m{T,mu_e,rho/nu_constants::mnuc_cgs,1-ye,ye};
        fit_record rec{rho,T,ye,eta_e,{0,0,0},{0,0,0}};
        for(int c=0;c<3;++c) {
            rates_accum fit{};
            if(c==0) add_pair_process_emission(F,fit);
            if(c==1) add_brems_emission(F,fit);
            if(c==2) add_plasmon_decay_emission(F,fit,false);
            #if GRACE_M1_NU_SPECIES >= 5
            double const q_fit=fit.Q[NUMU];                 // one heavy species
            #else
            double const q_fit=fit.Q[NUX]/4.0;
            #endif
            double const k_fit=q_fit/black_body_energy(1.0,T,0.0);
            kernel k; REQUIRE(k.init(m,ch[c],32,T));
            lte_rates r; REQUIRE(lte_pair_rates(k.g,T,0,0,[&](double const* f,double const* fb){ return evaluate(k,f,fb); },r));
            rec.q[c]=0.5*(r.emission[0]+r.emission[1])/q_fit;
            rec.k[c]=0.5*(r.kappa_E[0]+r.kappa_E[1])/nu_constants::clight/k_fit;
        }
        out.push_back(rec);
    }
    return out;
}
}  // namespace

// B1 (31 Oct 2026): the grey fits of the standard path against the kernel LTE rates over the head-on conditions.
// Pairs are a fit of the same Bruenn-type rate and agree; the brems fit sits above HR98 and the plasmon model below
// the transverse constant-mass kernel, by the documented factors.  Kirchhoff and the kernel agree on what kappa
// means at LTE, so the opacity ratio equals the emissivity ratio.  The 20 % band is the hidden case below.
TEST_CASE("Grey thermal fits against the kernel LTE rates over the head-on grid", "[pairs][physics][fits]") {
    auto const recs=fits_vs_kernel_grid();
    double lo[3]={1e300,1e300,1e300}, hi[3]={0,0,0};
    std::printf("\n  B1 kernel/fit at eta_nu = 0: emissivity (opacity)\n  %7s %5s %5s %6s | %-16s | %-16s | %-16s\n",
                "rho","T","Ye","eta_e","e+e- pairs","NN brems","plasmon");
    for(auto const& r: recs) {
        std::printf("  %7.0e %5.1f %5.2f %6.2f | %6.3f (%6.3f)  | %6.3f (%6.3f)  | %6.3f (%6.3f)\n",
                    r.rho,r.T,r.ye,r.eta_e,r.q[0],r.k[0],r.q[1],r.k[1],r.q[2],r.k[2]);
        for(int c=0;c<3;++c) {
            INFO("rho " << r.rho << " T " << r.T << " Ye " << r.ye << " channel " << c);
            REQUIRE(r.q[c]>0);
            REQUIRE_THAT(r.k[c],WithinRel(r.q[c],0.02));
            lo[c]=std::min(lo[c],r.q[c]); hi[c]=std::max(hi[c],r.q[c]);
        }
    }
    std::printf("  kernel/fit over the grid: pairs %.3f..%.3f, brems %.3f..%.3f, plasmon %.3f..%.3f\n",
                lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
    // Measured 31 Oct 2026: pairs 0.93..1.08 (0.958 wherever the electrons are non-degenerate), brems 0.38..0.96
    // (the fit is 1.05..2.7x above HR98, worst at T = 5 MeV and 1e14 g/cc, within 5 % at 40 MeV), plasmon 0.71..1.13.
    // Bands with margin; tighten when a fit is replaced (the strict 20 % band is the hidden case).
    REQUIRE(lo[0]>0.85);  REQUIRE(hi[0]<1.15);
    REQUIRE(lo[1]>0.30);  REQUIRE(hi[1]<1.05);
    REQUIRE(lo[2]>0.60);  REQUIRE(hi[2]<1.25);
}

TEST_CASE("Grey thermal fits within 20 % of the kernel LTE rates", "[.][fits-strict]") {
    for(auto const& r: fits_vs_kernel_grid()) for(int c=0;c<3;++c) {
        INFO("rho " << r.rho << " T " << r.T << " Ye " << r.ye << " channel " << c << " (0 pairs, 1 brems, 2 plasmon)");
        REQUIRE_THAT(r.q[c],WithinRel(1.0,0.2));
    }
}

// B2 (31 Oct 2026): at eta_numu = eta, eta_anumu = -eta the kernel's emissivity of a species carries only the
// PARTNER's blocking (its own 1 - f multiplies the transport term) and its effective opacity loss/J the rest.  The
// Ruffert fit puts both blockings into Q and Kirchhoff makes kappa = Q/B: the equilibrium is right, the relaxation
// rate of the degenerate side is low by the own-blocking factor 1 + exp(eta - <E>/T).  Measured against the kernel,
// with continuity of the fit across the +-5 clamp the standard path applies to the muon flavours.
TEST_CASE("Pair fit at the clamped muon fugacity: own-species blocking sits in Q, not in kappa",
          "[pairs][physics][clamp]") {
    mock_eos_t eos; tau_policy_analytic_density tau;
    std::printf("\n  B2 e+e- pairs at (eta, -eta): kernel/fit emissivity and opacity, nu_mu side | anti-nu_mu side; Ruffert own-blocking b0\n");
    for(auto const st: {std::pair<double,double>{1e13,10.0},std::pair<double,double>{1e12,5.0}}) {
        double const rho=st.first, T=st.second, ye=0.1, eta_e=electron_eta(rho,ye,T), mu_e=eta_e*T;
        eos.mu_e=mu_e; eos.mu_mu=0; eos.mu_p=0; eos.mu_n=0; eos.Xn=1-ye; eos.Xp=ye; eos.Xa=0; eos.Xh=0;
        material const m{T,mu_e,rho/nu_constants::mnuc_cgs,1-ye,ye};
        kernel k; REQUIRE(k.init(m,{true,false,false},32,T));
        auto const eval=[&](double const* f,double const* fb){ return evaluate(k,f,fb); };
        double const mean_over_T=0.5*(fermi::FDR<4,3>::get(eta_e)+fermi::FDR<4,3>::get(-eta_e));   // the fit's <E>/T
        auto const fit_at=[&](double eta,rates_accum& fit,fugacity_state& F) {
            F=make_fugacity_state(eos,rho*nu_constants::RHOGF,T,ye,0.0,1.0,kXyzOrigin,tau);
            for(int s=0;s<NUMSPECIES;++s) F.eta_nu[s]=0;
            F.eta_nu[NUMU]=eta; F.eta_nu[NUMUBAR]=-eta;
            fit=rates_accum{}; add_pair_process_emission(F,fit);
        };
        double r0_ref=0, r1_ref=0, b0_ref=0;
        for(double eta: {0.0,1.0,2.0,3.0,5.0}) {
            rates_accum fit{}; fugacity_state F; fit_at(eta,fit,F);
            REQUIRE(fit.Q[NUMU]==fit.Q[NUMUBAR]);                               // the fit blocks both sides in one Q
            double const kf0=fit.Q[NUMU]/black_body_energy(1.0,T,eta), kf1=fit.Q[NUMUBAR]/black_body_energy(1.0,T,-eta);
            lte_rates r; REQUIRE(lte_pair_rates(k.g,T,eta,-eta,eval,r));
            double const rq0=r.emission[0]/fit.Q[NUMU], rq1=r.emission[1]/fit.Q[NUMUBAR];
            double const rk0=r.kappa_E[0]/nu_constants::clight/kf0, rk1=r.kappa_E[1]/nu_constants::clight/kf1;
            double const b0=1+std::exp(eta-mean_over_T), b1=1+std::exp(-eta-mean_over_T);
            std::printf("  rho %7.0e T %4.1f eta %3.1f | %6.3f (%6.3f) | %6.3f (%6.3f) | b0 %5.2f b1 %5.3f\n",rho,T,eta,rq0,rk0,rq1,rk1,b0,b1);
            REQUIRE_THAT(rk0,WithinRel(rq0,0.02)); REQUIRE_THAT(rk1,WithinRel(rq1,0.02));
            if(eta==0.0) { r0_ref=rq0; r1_ref=rq1; b0_ref=b0; continue; }
            // the degenerate side: kernel/fit grows exactly like the fit's own-blocking factor
            REQUIRE_THAT(rq0/r0_ref,WithinRel(b0/b0_ref,0.25));
            // the dilute side keeps its eta = 0 agreement
            REQUIRE_THAT(rq1/r1_ref,WithinRel(1.0,0.25));
        }
        // continuity of the fit across the clamp value
        rates_accum a{}, b{}; fugacity_state Fa, Fb; fit_at(4.99,a,Fa); fit_at(5.0,b,Fb);
        REQUIRE_THAT(a.Q[NUMU],WithinRel(b.Q[NUMU],0.02));
        REQUIRE_THAT(a.Q[NUMU]/black_body_energy(1.0,T,4.99),WithinRel(b.Q[NUMU]/black_body_energy(1.0,T,5.0),0.02));
    }
}

// Her e+e- kernel integrated to grey production rates (unblocked, f_nu = 0) at the states listed in the file
// named by GRACE_PAIR_STATES ("T eta_e" per line), with the heavy-flavour couplings the kernel uses and, for
// the electron flavour, Pons' alpha_0 couplings on the same Psi functions.  For the comparison with Harry's
// spectral Weakhub pair table (harry_pp_grid.py on iota).  Without the file the test does nothing.
TEST_CASE("Kernel pair production at listed states for the spectral-table comparison", "[pairs][harry]") {
    char const* path=std::getenv("GRACE_PAIR_STATES");
    if(!path) { SUCCEED("GRACE_PAIR_STATES not set"); return; }
    std::ifstream in(path); REQUIRE(in.good());
    std::printf("# T eta_e | heavy R Q | electron R Q   (per helicity species; R cm^-3 s^-1, Q MeV cm^-3 s^-1)\n");
    double T,eta; double zero[max_order]={};
    while(in>>T>>eta) {
        material const m{T,eta*T,1e30,0.5,0.5};
        kernel k; REQUIRE(k.init(m,{true,false,false},32,T));
        auto const sh=evaluate(k,zero,zero);
        kernel ke=k;   // electron-flavour couplings on the same grid
        for(int i=0;i<k.g.n;++i) for(int j=0;j<k.g.n;++j) {
            double psi[2]; PairPsi(0,k.g.e[i]/T,k.g.e[j]/T,eta,psi);
            double const a=kBS_Pair_Alpha1_0, b=kBS_Pair_Alpha2_0;
            double const r=-0.5*kBS_Pair_Phi*T*T*(a*a*psi[0]+b*b*psi[1])/(-std::expm1(-(k.g.e[i]+k.g.e[j])/T));
            ke.a[i][j]=std::max(0.0,r)*1e-21;
        }
        auto const se=evaluate(ke,zero,zero);
        kernel kp; REQUIRE(kp.init(m,{false,false,true},32,T));   // her transverse plasmon model, heavy coupling
        auto const sp=evaluate(kp,zero,zero);
        std::printf("%.6f %.6f %.6e %.6e %.6e %.6e %.6e %.6e\n",T,eta,sh.emission_n[0],sh.emission[0],
                    se.emission_n[0],se.emission[0],sp.emission_n[0],sp.emission[0]);
    }
}

// Her muon-decay kernel integrated to unblocked grey production rates at the states listed in the file named
// by GRACE_DECAY_STATES ("T eta_e eta_mu" per line; chemical potentials with rest masses): mu- decay
// (charge 0: nu_mu, anu_e) and mu+ decay (charge 1: anu_mu, nu_e).  For the comparison with Harry's ILD
// kernels (harry_ild_grid.py on iota).  Without the file the test does nothing.
TEST_CASE("Kernel muon-decay production at listed states for the spectral-table comparison", "[pairs][harry_ild]") {
    char const* path=std::getenv("GRACE_DECAY_STATES");
    if(!path) { SUCCEED("GRACE_DECAY_STATES not set"); return; }
    std::ifstream in(path); REQUIRE(in.good());
    std::printf("# T eta_e eta_mu | mu- decay: R_numu Q_numu R_anue Q_anue | mu+ decay: R_anumu Q_anumu R_nue Q_nue\n");
    double T,eta_e,eta_mu; double zero[max_order]={};
    while(in>>T>>eta_e>>eta_mu) {
        grid g; g.init(32,T);
        decay_kernel k; REQUIRE(k.init(g,T,eta_e*T,eta_mu*T,24));
        auto const s0=evaluate_decay(k,0,zero,zero), s1=evaluate_decay(k,1,zero,zero);
        std::printf("%.6f %.6f %.6f %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",T,eta_e,eta_mu,
                    s0.emission_n[0],s0.emission[0],s0.emission_n[1],s0.emission[1],
                    s1.emission_n[0],s1.emission[0],s1.emission_n[1],s1.emission[1]);
    }
}

// PHYSICS CHECK 2: the two-moment (max-entropy FD) closure against a spectral reference built from the SAME
// kernel: df_i/dt = sum_j w_j [R_p (1-f_i)(1-fb_j) - R_a f_i fb_j] on the 32-node grid, integrated explicitly
// with small steps.  Pair-only relaxation of a hot box from vacuum to LTE, and from a hot non-thermal spectrum.
TEST_CASE("Two-moment pair relaxation against the spectral reference", "[pairs][physics][closure]") {
    material const m{10,20,6e37,0.8,0.2};
    kernel k; REQUIRE(k.init(m,{true,true,false},32,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    int const n=k.g.n;
    auto const moments=[&](double const* f,double& nn,double& J) { nn=J=0; for(int i=0;i<n;++i) { nn+=k.g.w[i]*f[i]; J+=k.g.w[i]*k.g.e[i]*f[i]; } };
    double f_lte[max_order],n_lte,J_lte; fd(k.g,10,0,f_lte,n_lte,J_lte);
    for(int scenario=0;scenario<2;++scenario) {
        double f[max_order],fb[max_order];
        for(int i=0;i<n;++i) f[i]=fb[i]=scenario==0?0.0:0.3*std::exp(-k.g.e[i]/25.0);   // vacuum | hot, dilute, non-thermal
        // spectral reference time scale from the vacuum production rate
        double rate=0; for(int i=0;i<n;++i) { double s=0; for(int j=0;j<n;++j) s+=k.g.w[j]*k.a[i][j]*k.boltz[i]*k.boltz[j]; rate=std::max(rate,s/std::max(f_lte[i],1e-300)); }
        double const dt=0.02/rate;                      // seconds; ~50 steps per LTE e-folding of the slowest node
        // grey state: one representative species pair, undensitised code units
        m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{}; double old[10]={},out[10],res;
        { double n0,J0; moments(f,n0,J0); for(int s=0;s<2;++s) { old[5*s]=base[s][ERADL]=J0*energy_unit; old[5*s+4]=base[s][NRADL]=n0*number_unit; } }
        std::printf("\n  %s: step  t[ms]  | spectral n/n_lte  J/J_lte  <e>[MeV] | grey n/n_lte  J/J_lte  <e>[MeV] | rel err n  J\n",
                    scenario==0?"from vacuum":"from a hot non-thermal spectrum");
        double worst_n=0, worst_J=0;
        for(int step=1;step<=400;++step) {
            // spectral: 10 explicit substeps per grey step
            for(int sub=0;sub<10;++sub) {
                double df[max_order],dfb[max_order];
                for(int i=0;i<n;++i) { df[i]=dfb[i]=0; }
                for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
                    double const net=k.a[i][j]*(k.boltz[i]*k.boltz[j]*(1-f[i])*(1-fb[j])-f[i]*fb[j]);
                    df[i]+=k.g.w[j]*net; dfb[j]+=k.g.w[i]*net;
                }
                for(int i=0;i<n;++i) { f[i]+=0.1*dt*df[i]; fb[i]+=0.1*dt*dfb[i]; }
            }
            // grey: one implicit two-moment step of the same length
            REQUIRE(implicit_update(k,metric,base,eas,old,out,dt*time_unit,res));
            for(int a=0;a<10;++a) old[a]=out[a];
            for(int s=0;s<2;++s) { base[s][ERADL]=out[5*s]; base[s][NRADL]=out[5*s+4]; }
            double ns,Js; moments(f,ns,Js);
            double const ng=out[4]/number_unit, Jg=out[0]/energy_unit;
            double const en=std::abs(ng/ns-1), eJ=std::abs(Jg/Js-1);
            if(step>5) { worst_n=std::max(worst_n,en); worst_J=std::max(worst_J,eJ); }
            if(step%40==0 || step<=5)
                std::printf("  %4d %7.3f | %8.4f %8.4f %7.2f | %8.4f %8.4f %7.2f | %.1e %.1e\n",
                            step,1e3*step*dt,ns/n_lte,Js/J_lte,Js/ns,ng/n_lte,Jg/J_lte,Jg/ng,en,eJ);
        }
        std::printf("  worst relative error after the first 5 steps: n %.2e, J %.2e\n",worst_n,worst_J);
        REQUIRE(worst_n<0.05); REQUIRE(worst_J<0.05);
    }
}

TEST_CASE("Transverse plasmon event rate agrees with thermal photon decay", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{false,false,true},32,10));
    double f[max_order]={}; auto const s=evaluate(k,f,f);
    constexpr double alpha=1/137.035999084,gf=1.1663787e-11,hbar=6.582119569e-22;
    constexpr double cv=-0.5+2*0.2325;
    double const m2=4*alpha/(3*pairs::pi)*(400+pairs::pi*pairs::pi*100/3);
    double count=0,energy=0;
    // Independent one-dimensional integral over thermal photon momentum.
    for(int i=0;i<k.g.n;++i) {
        double const omega=std::sqrt(k.g.e[i]*k.g.e[i]+m2);
        double const gamma=gf*gf*cv*cv*m2*m2*m2/(48*pairs::pi*pairs::pi*alpha*hbar*omega);
        double const r=2*k.g.w[i]*gamma/std::expm1(omega/10);
        count+=r; energy+=r*omega/2;
    }
    REQUIRE_THAT(s.number,WithinRel(count,0.03));
    REQUIRE_THAT(s.energy[0],WithinRel(energy,0.03));
}

TEST_CASE("Pair fixed point and nonzero ordinary EAS share one stage", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,true},16,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double f[max_order],n,J,old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        fd(k.g,10,s==0?1:-1,f,n,J);
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
        eas[s][KAL]=2; eas[s][KANL]=3;
        eas[s][ETAL]=2*old[5*s]; eas[s][ETANL]=3*old[5*s+4];
    }
    REQUIRE(implicit_update(k,metric,base,eas,old,out,100,res));
    for(int s=0;s<2;++s) {
        REQUIRE_THAT(out[5*s],WithinRel(old[5*s],2e-9));
        REQUIRE_THAT(out[5*s+4],WithinRel(old[5*s+4],2e-9));
    }
}

TEST_CASE("Pairs emit into vacuum and preserve moving-fluid LTE", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,false},24,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double old[10]={},out[10],res;
    REQUIRE(implicit_update(k,metric,base,eas,old,out,0.1,res));
    REQUIRE(out[0]>0); REQUIRE(out[5]>0);
    REQUIRE(out[4]==out[9]);
    for(double v: {0.1,0.5}) {
        double const W=1/std::sqrt(1-v*v);
        for(int s=0;s<2;++s) {
            double f[max_order],n,J;
            fd(k.g,10,s==0?0.4:-0.4,f,n,J);
            old[5*s]=base[s][ERADL]=J*energy_unit*(4*W*W-1)/3;
            old[5*s+1]=base[s][FXL]=J*energy_unit*4*W*W*v/3;
            old[5*s+4]=base[s][NRADL]=n*number_unit*W;
            base[s][grace::ZXL]=W*v;
        }
        REQUIRE(implicit_update(k,metric,base,eas,old,out,10,res));
        for(int s=0;s<2;++s) {
            REQUIRE_THAT(out[5*s],WithinRel(old[5*s],3e-8));
            REQUIRE_THAT(out[5*s+1],WithinRel(old[5*s+1],3e-8));
            REQUIRE_THAT(out[5*s+4],WithinRel(old[5*s+4],3e-8));
        }
    }
}

TEST_CASE("Pair kernels execute on the selected Kokkos backend", "[pairs]") {
    Kokkos::View<double*> result("pair_backend_result",3);
    Kokkos::parallel_for("pair_backend_check",1,KOKKOS_LAMBDA(int) {
        kernel k;
        material m{8,12,1e37,1,0}; // also exercise pure-neutron composition
        result(0)=k.init(m,{true,true,true},16,8);
        if(result(0)!=1) return;
        double f[max_order]={},b[max_order]={};
        auto const s=evaluate(k,f,b);
        result(1)=s.number; result(2)=s.energy[0];
    });
    auto const r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
    REQUIRE(r(0)==1); REQUIRE(r(1)>0); REQUIRE(std::isfinite(r(2)));
}

TEST_CASE("GRACE pair source wiring preserves densitization and aggregate weights", "[pairs]") {
    var_array_t state("pair_state",1,1,1,N_EVOL_VARS,1);
    var_array_t next("pair_next",1,1,1,N_EVOL_VARS,1);
    var_array_t aux("pair_aux",1,1,1,N_AUX_VARS,1);
    auto st=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state);
    auto ax=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),aux);
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
    st(0,0,0,GXX_,0)=4; st(0,0,0,GYY_,0)=st(0,0,0,GZZ_,0)=1;
    #else
    st(0,0,0,GTXX_,0)=4; st(0,0,0,GTYY_,0)=st(0,0,0,GTZZ_,0)=1;
    st(0,0,0,CHI_,0)=1;
    #endif
    st(0,0,0,ALP_,0)=0.7;
    ax(0,0,0,TEMP_,0)=matter.T; ax(0,0,0,PAIR_MUE_,0)=matter.mu_e;
    ax(0,0,0,RHO_,0)=matter.nb/nu_constants::avogadro*nu_constants::RHOGF;   // the solve rebuilds n_b from rho
    ax(0,0,0,PAIR_YN_,0)=matter.yn;
    ax(0,0,0,PAIR_YP_,0)=matter.yp; ax(0,0,0,PAIR_ACTIVE_,0)=3;
    kernel k; REQUIRE(k.init(matter,{true,true,false},16,10,true));
    double f[max_order],n,J; fd(k.g,7,0,f,n,J);
    #if GRACE_M1_NU_SPECIES >= 5
    constexpr int x=4,g=2;
    st(0,0,0,ERAD3_,0)=st(0,0,0,ERAD4_,0)=2*J*energy_unit;
    st(0,0,0,NRAD3_,0)=st(0,0,0,NRAD4_,0)=2*n*number_unit;
    #else
    constexpr int x=2,g=4;
    #endif
    st(0,0,0,m1_erad_idx<x>(),0)=2*g*J*energy_unit;
    st(0,0,0,m1_nrad_idx<x>(),0)=2*g*n*number_unit;
    Kokkos::deep_copy(state,st); Kokkos::deep_copy(next,state); Kokkos::deep_copy(aux,ax);
    m1_equations_system_t system(state,staggered_variable_arrays_t{},aux);
    scalar_array_t<GRACE_NSPACEDIM> idx;
    Kokkos::parallel_for("pair_wiring",1,KOKKOS_LAMBDA(int) {
        #if GRACE_M1_NU_SPECIES >= 5
        system.compute_pair_implicit_update<2,3>(0,VEC(0,0,0),idx,next,0.1,1,16);
        #endif
        system.compute_pair_implicit_update<x,x,g>(0,VEC(0,0,0),idx,next,0.1,1,16);
    });
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),next);
    // the launch-shared kernel (one build per cell, symmetrized for the aggregate) must reproduce
    // the per-solve builds bit for bit when the scales coincide, as they do here
    var_array_t next2("pair_next2",1,1,1,N_EVOL_VARS,1); Kokkos::deep_copy(next2,state);
    Kokkos::parallel_for("pair_wiring_shared",1,KOKKOS_LAMBDA(int) {
        auto const mat=system.pair_material(0,VEC(0,0,0));
        pairs::kernel shared;
        shared.init(mat,{true,true,false},16,system.pair_scale(0,VEC(0,0,0),mat.T,false));
        #if GRACE_M1_NU_SPECIES >= 5
        system.compute_pair_implicit_update<2,3>(0,VEC(0,0,0),idx,next2,0.1,1,16,&shared);
        #endif
        shared.symmetrize();
        system.compute_pair_implicit_update<x,x,g>(0,VEC(0,0,0),idx,next2,0.1,1,16,&shared);
    });
    auto got2=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),next2);
    for(int v=0;v<N_EVOL_VARS;++v) REQUIRE(got2(0,0,0,v,0)==got(0,0,0,v,0));
    metric_array_t metric({4,0,0,1,0,1},{0,0,0},0.7);
    REQUIRE(metric.sqrtg()==2);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
    }
    REQUIRE(implicit_update(k,metric,base,eas,old,out,0.1,res));
    REQUIRE_THAT(got(0,0,0,m1_erad_idx<x>(),0),WithinRel(2*g*out[0],2e-10));
    REQUIRE_THAT(got(0,0,0,m1_nrad_idx<x>(),0),WithinRel(2*g*out[4],2e-10));
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(got(0,0,0,NRAD3_,0),WithinRel(got(0,0,0,NRAD4_,0),2e-10));
    #endif
}
