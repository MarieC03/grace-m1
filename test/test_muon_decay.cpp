#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/neutrino_muon_decay.hh>
#include <grace/physics/neutrino_leptonic_update.hh>
#include <grace/physics/m1.hh>
#include <grace/physics/eas_kinds.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/null_sink.h>
#include <cmath>
#include <cstdlib>
using namespace grace;
using namespace grace::pairs;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {
void fd_moments(grid const& g,double T,double eta,double& n,double& J) {
    n=J=0;
    for(int i=0;i<g.n;++i) {
        double const f=occupation(g.e[i]/T-eta);
        n+=g.w[i]*f; J+=g.w[i]*g.e[i]*f;
    }
}
void set_radiation(grid const& g,double T,double eta,int s,
    m1_prims_array_t (&base)[4],double (&old)[20],double velocity=0) {
    double n,J; fd_moments(g,T,eta,n,J);
    double const W=1/std::sqrt(1-velocity*velocity);
    old[5*s]=base[s][ERADL]=J*energy_unit*(4*W*W-1)/3;
    old[5*s+1]=base[s][FXL]=J*energy_unit*4*W*W*velocity/3;
    old[5*s+4]=base[s][NRADL]=n*number_unit*W;
    base[s][grace::ZXL]=W*velocity;
}
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
TEST_CASE("Exact decay cache invalidates every physical and quadrature input", "[decay][cache]") {
    grid g; g.init(8,35);
    decay_kernel original,got;
    REQUIRE(original.init(g,12,70,110,8));
    decay_kernel_cache<4,4> cache; // One bucket forces hash collisions/eviction.
    REQUIRE_FALSE(cache.find(g,12,70,110,8,got));
    cache.insert(original,70,110,8);
    REQUIRE(cache.find(g,12,70,110,8,got));
    REQUIRE(got.T==original.T); REQUIRE(got.delta==original.delta);
    for(int i=0;i<g.n;++i) {
        REQUIRE(got.g.e[i]==g.e[i]); REQUIRE(got.g.w[i]==g.w[i]);
        for(int j=0;j<g.n;++j) for(int c=0;c<2;++c)
            REQUIRE(got.total[c][i][j]==original.total[c][i][j]);
    }
    REQUIRE_FALSE(cache.find(g,std::nextafter(12.,13.),70,110,8,got));
    REQUIRE_FALSE(cache.find(g,12,std::nextafter(70.,71.),110,8,got));
    REQUIRE_FALSE(cache.find(g,12,70,std::nextafter(110.,111.),8,got));
    REQUIRE_FALSE(cache.find(g,12,70,110,9,got));
    for(int kind=0;kind<4;++kind) {
        grid changed=g;
        if(kind==0) changed.e[0]=std::nextafter(g.e[0],2*g.e[0]);
        if(kind==1) changed.w[0]=std::nextafter(g.w[0],2*g.w[0]);
        if(kind==2) changed.scale=std::nextafter(g.scale,2*g.scale);
        if(kind==3) --changed.n;
        REQUIRE_FALSE(cache.find(changed,12,70,110,8,got));
    }
    for(int a=0;a<4;++a) {
        decay_kernel other; REQUIRE(other.init(g,13+a,70,110,8));
        cache.insert(other,70,110,8);
    }
    REQUIRE_FALSE(cache.find(g,12,70,110,8,got));
    REQUIRE(cache.find(g,16,70,110,8,got));
    decay_kernel fresh; REQUIRE(fresh.init(g,16,70,110,8));
    for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j) for(int c=0;c<2;++c)
        REQUIRE(got.total[c][i][j]==fresh.total[c][i][j]);
}

TEST_CASE("Shared decay cache is exact under concurrent cold misses and warm hits", "[decay][cache]") {
    grid g; g.init(8,37);
    decay_kernel reference; REQUIRE(reference.init(g,13,71,111,8));
    for(int repeat=0;repeat<2;++repeat) {
        Kokkos::View<int*> errors("cache errors",64);
        Kokkos::parallel_for("concurrent decay cache",64,KOKKOS_LAMBDA(int a) {
            decay_kernel k;
            int bad=!init_decay_kernel(k,g,13,71,111,8,true);
            if(!bad) for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j) for(int c=0;c<2;++c)
                bad+=k.total[c][i][j]!=reference.total[c][i][j];
            errors(a)=bad;
        });
        auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},errors);
        for(int a=0;a<64;++a) REQUIRE(host(a)==0);
    }
    decay_kernel bad;
    REQUIRE_FALSE(init_decay_kernel(bad,g,-1,71,111,8,true));
}
#endif

TEST_CASE("Muon decay finite-mass normalization and daughter orientation", "[decay]") {
    grid g; g.init(32,30);
    decay_kernel k; REQUIRE(k.init(g,20,-400,20,48));
    double f[max_order]={}; auto const src=evaluate_decay(k,0,f,f);
    double density=0,dilated=0;
    grid p; p.init(32,50);
    for(int i=0;i<p.n;++i) {
        double const E=std::sqrt(p.e[i]*p.e[i]+muon_mass*muon_mass);
        double const dn=2*p.w[i]*occupation((E-20)/20);
        density+=dn; dilated+=dn*muon_mass/E;
    }
    double const r=std::pow(electron_mass/muon_mass,2);
    double const phase_space=1-8*r+8*r*r*r-r*r*r*r-12*r*r*std::log(r);
    double const width=decay_gf*decay_gf*std::pow(muon_mass,5)/(192*std::pow(pi,3)*hbar_mev_s)*phase_space;
    INFO("rate=" << src.number/(dilated*width) << "; numu energy="
         <<src.energy[0]/(density*width*muon_mass)<<"; anue energy="
         <<src.energy[1]/(density*width*muon_mass));
    REQUIRE_THAT(src.number,WithinRel(dilated*width,0.035));
    // Massless-electron Michel first moments; finite-me correction < 0.1%.
    REQUIRE_THAT(src.energy[0],WithinRel(0.35*density*width*muon_mass,0.035));
    REQUIRE_THAT(src.energy[1],WithinRel(0.30*density*width*muon_mass,0.035));
}

TEST_CASE("Muon decay detailed balance, charge conjugation and inverse channel", "[decay]") {
    grid g; g.init(24,20);
    decay_kernel k,cp;
    REQUIRE(k.init(g,15,70,110,24)); REQUIRE(cp.init(g,15,-70,-110,24));
    double f[max_order],b[max_order];
    for(int c=0;c<2;++c) {
        double const d=c==0?40:-40;
        for(int i=0;i<g.n;++i) {
            f[i]=occupation((g.e[i]-d/2-7)/15);
            b[i]=occupation((g.e[i]-d/2+7)/15);
            for(int j=0;j<g.n;++j)
                REQUIRE_THAT(k.total[c][i][j],WithinRel(cp.total[1-c][i][j],2e-13));
        }
        auto const src=evaluate_decay(k,c,f,b);
        REQUIRE(src.emission_n[0]>0);
        REQUIRE(std::abs(src.number)<2e-13*src.emission_n[0]);
        REQUIRE(std::abs(src.energy[0])<2e-13*src.emission[0]);
        REQUIRE(std::abs(src.energy[1])<2e-13*src.emission[1]);
    }
    // Zero net muon chemical potential must NOT suppress inverse decay.
    REQUIRE(k.init(g,15,100,0,24));
    for(int i=0;i<g.n;++i) f[i]=b[i]=0.9;
    REQUIRE(evaluate_decay(k,0,f,b).number<0);
}

TEST_CASE("Muon decay inner integration converges and executes on backend", "[decay]") {
    decay_rule a,b; a.init(32); b.init(64);
    for(double e: {8.,25.,60.}) for(double ep: {10.,35.,70.}) {
        double x[2],y[2];
        decay_total(20,50,90,e,ep,a,x); decay_total(20,50,90,e,ep,b,y);
        for(int c=0;c<2;++c) REQUIRE_THAT(x[c],WithinRel(y[c],0.015));
    }
    Kokkos::View<double*> r("decay_result",2);
    Kokkos::parallel_for("decay_backend",1,KOKKOS_LAMBDA(int) {
        decay_rule q; q.init(16); double x[2];
        decay_total(20,50,90,25,35,q,x); r(0)=x[0]; r(1)=x[1];
    });
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),r);
    REQUIRE(h(0)>0); REQUIRE(h(1)>0);
}

TEST_CASE("Muon decay inner quadrature resolves representative degenerate matter", "[decay]") {
    decay_rule a,b; a.init(32); b.init(64);
    double const matter[3][3]={{3,150,180},{10,150,180},{30,50,90}};
    for(auto const& m: matter) for(double e: {8.,25.,60.}) for(double ep: {10.,35.,70.}) {
        double x[2],y[2];
        decay_total(m[0],m[1],m[2],e,ep,a,x); decay_total(m[0],m[1],m[2],e,ep,b,y);
        for(int c=0;c<2;++c) {
            INFO("T="<<m[0]<<", e="<<e<<", ep="<<ep<<", charge="<<c<<", ratio="<<x[c]/y[c]);
            REQUIRE_THAT(x[c],WithinRel(y[c],0.02));
        }
    }
}

TEST_CASE("Shared decay quadrature preserves both oriented matrix entries", "[decay]") {
    // Compare the triangular build with independently integrated ordered
    // entries, including asymmetric energies and degenerate charged leptons.
    double const states[][4]={{20,50,90,35},{0.1143,0.0304,103.6,17.9},{8,120,150,40}};
    decay_rule rule; rule.init(24);
    for(auto const& m: states) {
        CAPTURE(m[0],m[1],m[2]);
        grid g; g.init(12,m[3]);
        decay_kernel k;
        REQUIRE(k.init(g,m[0],m[1],m[2],24));
        bool asymmetric=false;
        for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j) {
            CAPTURE(i,j);
            double direct[2];
            decay_total(m[0],m[1],m[2],g.e[i],g.e[j],rule,direct);
            for(int c=0;c<2;++c) {
                REQUIRE_THAT(k.total[c][i][j],WithinRel(direct[c],2e-13));
                asymmetric|=k.total[c][i][j]!=k.total[c][j][i];
            }
        }
        REQUIRE(asymmetric);
    }

    // The shared occupation calculation must retain tiny holes as well as
    // tiny populations. In particular, 1-occupation(-700) would be zero.
    for(double x: {-700.,-100.,-1.,0.,1.,100.,700.}) {
        double f,hole;
        decay_occupation_and_hole(x,f,hole);
        REQUIRE_THAT(f,WithinRel(occupation(x),2e-15));
        REQUIRE_THAT(hole,WithinRel(occupation(-x),2e-15));
    }
}

TEST_CASE("Four-flavour network preserves LTE including ordinary EAS and moving matter", "[decay]") {
    kernel thermal; REQUIRE(thermal.init({15,70,6e37,0.8,0.2},{true,true,false},16,35));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    for(double ymu: {5e-4,6.5e-4,0.01}) for(double v: {0.,0.4}) {
        double const mu=dilute_muon_cc_potential(110,70,ymu);
        decay_kernel decay; REQUIRE(decay.init(thermal.g,15,70,mu,24));
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},out[20],res;
        double const eta[4]={-0.3,0.3,(mu-70)/15-0.3,-(mu-70)/15+0.3};
        for(int s=0;s<4;++s) {
            set_radiation(thermal.g,15,eta[s],s,base,old,v);
            double n,J; fd_moments(thermal.g,15,eta[s],n,J);
            eas[s][KAL]=2; eas[s][KANL]=3; eas[s][KSL]=4;
            eas[s][ETAL]=2*J*energy_unit; eas[s][ETANL]=3*n*number_unit;
        }
        REQUIRE(leptonic_implicit_update(thermal,decay,true,metric,base,eas,old,out,100,res));
        REQUIRE(res<2e-8);
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
        int iterations=0; bool fallback=true;
        REQUIRE(leptonic_lagged_update(thermal,decay,true,metric,base,eas,old,out,100,res,nullptr,&iterations,&fallback));
        REQUIRE_FALSE(fallback);
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
        REQUIRE(leptonic_singlepass_update(thermal,decay,true,metric,base,eas,old,out,100,res));
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
    }
}

TEST_CASE("Lagged decay converges to the coupled source away from equilibrium", "[decay][lagged]") {
    kernel thermal; REQUIRE(thermal.init({15,70,6e37,0.8,0.2},{true,true,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,15,70,110,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    for(double h: {0.001,0.1,10.}) {
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},reference[20],out[20],res;
        for(int s=0;s<4;++s) set_radiation(thermal.g,12+s,-0.2*s,s,base,old);
        REQUIRE(leptonic_implicit_update(thermal,decay,true,metric,base,eas,old,reference,h,res));
        int iterations=0; bool fallback=false;
        REQUIRE(leptonic_lagged_update(thermal,decay,true,metric,base,eas,old,out,h,res,nullptr,&iterations,&fallback));
        for(int s=0;s<4;++s) for(int v: {0,4})
            REQUIRE_THAT(out[5*s+v],WithinRel(reference[5*s+v],2e-6));
        std::printf("lagged h=%g iterations=%d fallback=%d residual=%g\n",h,iterations,int(fallback),res);
    }
}

TEST_CASE("Single-pass decay exposes its charge defect and converges with timestep", "[decay][singlepass]") {
    kernel thermal; REQUIRE(thermal.init({15,70,6e37,0.8,0.2},{true,true,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,15,70,110,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double old[20]{},ref[20],out[20],res,last=1e100;
    for(int s=0;s<4;++s) set_radiation(thermal.g,12+s,-0.2*s,s,base,old);
    for(double h:{1e-3,5e-4,2.5e-4}) {
        REQUIRE(leptonic_implicit_update(thermal,decay,true,metric,base,eas,old,ref,h,res));
        REQUIRE(leptonic_singlepass_update(thermal,decay,true,metric,base,eas,old,out,h,res));
        double error=0,q=0,norm=0;
        for(int s=0;s<4;++s) {
            error=std::max(error,std::abs(out[5*s+4]-ref[5*s+4])/ref[5*s+4]);
            q+=(s==0||s==2?1:-1)*(out[5*s+4]-old[5*s+4]);
            norm+=old[5*s+4];
        }
        REQUIRE(error<0.6*last); last=error;
        REQUIRE(std::abs(q)/norm>1e-12);
        std::printf("SINGLEPASS_TEST h=%g relative_N_error=%g charge_defect_over_N=%g\n",h,error,q/norm);
    }
}

TEST_CASE("Fermi admissibility rejects the measured deficit and respects multiplicity", "[decay][transport]") {
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t p{};
    double const n=9.475e32;
    p[NRADL]=n*number_unit;
    p[ERADL]=n*4.065*energy_unit;
    REQUIRE_FALSE(fermi_moments_admissible(p,metric,1));
    p[ERADL]=1.01*minimum_fermi_energy_density(n)*energy_unit;
    REQUIRE(fermi_moments_admissible(p,metric,1));
    p[ERADL]*=2; p[NRADL]*=2;
    REQUIRE(fermi_moments_admissible(p,metric,2));
    REQUIRE_FALSE(fermi_moments_admissible(p,metric,1));
    p[FXL]=1.1*p[ERADL];
    REQUIRE_FALSE(fermi_moments_admissible(p,metric,2));
}

TEST_CASE("Dilute CC target reaches electron potential only on demuonizing branch", "[decay][rates]") {
    REQUIRE(dilute_muon_cc_potential(100,20,5e-4)==20);
    REQUIRE(dilute_muon_cc_potential(100,20,6e-4)==20);
    REQUIRE(dilute_muon_cc_potential(10,20,5e-4)==10);
    REQUIRE_THAT(dilute_muon_cc_potential(100,20,0.01),WithinRel(100,1e-14));
    REQUIRE(dilute_muon_cc_potential(100,20,6e-4+1e-12)-20<2e-6);
}

TEST_CASE("A previous-stage guess solves an admissible implicit state from a negative RHS", "[decay]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double expected[20]{},rhs[20],guess[20],out[20],residual;
    // At this state each decay edge obeys chemical detailed balance.
    double const eta[4]={-1,1,2,-2};
    for(int s=0;s<4;++s) set_radiation(thermal.g,20,eta[s],s,base,expected);
    for(int a=0;a<20;++a) { rhs[a]=expected[a]; guess[a]=0.98*expected[a]; }
    // Manufacture an inadmissible explicit RHS whose full implicit solution
    // is the LTE state above. The constant ordinary source supplies the deficit.
    double const h=0.01;
    rhs[0]=base[0][ERADL]=-expected[0];
    rhs[4]=base[0][NRADL]=-expected[4];
    eas[0][ETAL]=2*expected[0]/h;
    eas[0][ETANL]=2*expected[4]/h;
    REQUIRE_FALSE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,rhs,out,h,residual));
    REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,rhs,out,h,residual,guess));
    for(int s=0;s<4;++s) {
        REQUIRE_THAT(out[5*s],WithinRel(expected[5*s],1e-7));
        REQUIRE_THAT(out[5*s+4],WithinRel(expected[5*s+4],1e-7));
    }
}

TEST_CASE("Equilibrium decay EAS uses j plus absorption and has the proper fixed point", "[decay]") {
    grid g; g.init(24,35);
    decay_kernel k; REQUIRE(k.init(g,20,50,90,24));
    for(int charge=0;charge<2;++charge) {
        double const eta[2]={charge==0?1.7:-1.7,charge==0?0.3:-0.3};
        decay_lte_rates rate; REQUIRE(equilibrium_decay_rates(k,charge,eta[0],eta[1],rate));
        double n[2],J[2];
        for(int s=0;s<2;++s) {
            fd_moments(g,20,eta[s],n[s],J[s]);
            REQUIRE(rate.kappa_E[s]>0); REQUIRE(rate.kappa_N[s]>0);
            REQUIRE_THAT(rate.eta_E[s],WithinRel(rate.kappa_E[s]*J[s],3e-13));
            REQUIRE_THAT(rate.eta_N[s],WithinRel(rate.kappa_N[s]*n[s],3e-13));
        }
        // This is deliberately a regression of the approximation's limitation:
        // a depleted first daughter and equilibrium second daughter give
        // independent unequal sources. The evolved collision term does not.
        double const source0=rate.eta_N[0]-rate.kappa_N[0]*0.5*n[0];
        double const source1=rate.eta_N[1]-rate.kappa_N[1]*n[1];
        REQUIRE(source0>0);
        REQUIRE(std::abs(source1)<1e-12*source0);
    }
}

// Cold dilute HOT-TOV surface cell: T = 0.114 MeV, mu_mu = 103.6 MeV, eta_numu = +782 -> the anti-nu_mu
// orientation of the charge-1 decay channel underflows.
TEST_CASE("Equilibrium decay rates stay finite when a partner occupation underflows", "[decay][lte]") {
    grid g; g.init(16,35.2);
    decay_kernel k; REQUIRE(k.init(g,0.1143,0.0304,103.6,24));
    auto const direct=[&](int charge,double eta0,double eta1,decay_lte_rates& r) {
        double f[2][max_order],n[2]={0,0},J[2]={0,0}; double const eta[2]={eta0,eta1};
        for(int s=0;s<2;++s) for(int i=0;i<g.n;++i) {
            f[s][i]=occupation(g.e[i]/k.T-eta[s]); n[s]+=g.w[i]*f[s][i]; J[s]+=g.w[i]*g.e[i]*f[s][i];
        }
        auto const ps=evaluate_decay(k,charge,f[0],f[1]);
        for(int s=0;s<2;++s) { r.eta_E[s]=ps.emission[s]; r.eta_N[s]=ps.emission_n[s];
                               r.kappa_E[s]=ps.loss[s]/J[s]; r.kappa_N[s]=ps.loss_n[s]/n[s]; }
    };
    decay_lte_rates r,d;
    for(double eta: {600.,-600.}) {   // both forms valid
        INFO("eta_numu = " << eta);
        REQUIRE(equilibrium_decay_rates(k,0,eta,-50,r)); direct(0,eta,-50,d);
        for(int s=0;s<2;++s) { REQUIRE_THAT(r.kappa_E[s],WithinRel(d.kappa_E[s],1e-10)); REQUIRE_THAT(r.kappa_N[s],WithinRel(d.kappa_N[s],1e-10)); }
        REQUIRE(equilibrium_decay_rates(k,1,-eta,50,r)); direct(1,-eta,50,d);
        for(int s=0;s<2;++s) { REQUIRE_THAT(r.kappa_E[s],WithinRel(d.kappa_E[s],1e-10)); REQUIRE_THAT(r.kappa_N[s],WithinRel(d.kappa_N[s],1e-10)); }
    }
    for(double eta: {782.,900.}) {    // the direct sums underflow
        INFO("eta_numu = " << eta);
        REQUIRE(equilibrium_decay_rates(k,0,eta,-50,r));
        REQUIRE(equilibrium_decay_rates(k,1,-eta,50,r));
        for(int s=0;s<2;++s) {
            REQUIRE(std::isfinite(r.kappa_E[s])); REQUIRE(r.kappa_E[s]>=0);
            REQUIRE(std::isfinite(r.kappa_N[s])); REQUIRE(r.kappa_N[s]>=0);
            REQUIRE(std::isfinite(r.eta_E[s])); REQUIRE(r.eta_E[s]>=0);
        }
    }
}

TEST_CASE("Four-flavour source emits into vacuum with equal daughter counts", "[decay]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double old[20]={},out[20],res;
    REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,old,out,0.01,res));
    REQUIRE(res<2e-8);
    for(int s=0;s<4;++s) { REQUIRE(out[5*s]>0); REQUIRE(out[5*s+4]>0); }
    REQUIRE(out[4]==out[19]); REQUIRE(out[9]==out[14]);
    REQUIRE(out[10]/out[14]>out[5]/out[9]); // oriented hard/soft daughters
}

TEST_CASE("Evolved decay transport damping matches LTE coefficients without duplicating EAS", "[decay]") {
    kernel thermal; REQUIRE(thermal.init({20,50,6e37,0.8,0.2},{true,true,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,50,90,24));
    double const eta[4]={-0.3,0.3,1.7,-1.7};
    double n[4],J[4],opacity[4],ordinary_pair[4];
    for(int s=0;s<4;++s) fd_moments(thermal.g,20,eta[s],n[s],J[s]);
    REQUIRE(leptonic_transport_opacities(thermal,nullptr,true,n,J,ordinary_pair));
    REQUIRE(leptonic_transport_opacities(thermal,&decay,true,n,J,opacity));
    REQUIRE(ordinary_pair[0]==0); REQUIRE(ordinary_pair[1]==0);
    for(int charge=0;charge<2;++charge) {
        int const a=charge==0?2:3,b=charge==0?1:0;
        decay_lte_rates rate; REQUIRE(equilibrium_decay_rates(decay,charge,eta[a],eta[b],rate));
        REQUIRE_THAT(opacity[a]-ordinary_pair[a],WithinRel(rate.kappa_E[0]/time_unit,5e-9));
        REQUIRE_THAT(opacity[b],WithinRel(rate.kappa_E[1]/time_unit,5e-9));
    }
    // Actual transport reader adds the extra damping; ordinary source slots
    // themselves remain untouched, so they cannot apply decay a second time.
    var_array_t state("transport_state",1,1,1,N_EVOL_VARS,1);
    var_array_t aux("transport_aux",1,1,1,N_AUX_VARS,1);
    auto ax=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),aux);
    ax(0,0,0,KAPPAA1_,0)=2; ax(0,0,0,KAPPAS1_,0)=3;
    ax(0,0,0,PAIR_KAPPA1_,0)=opacity[0];
    Kokkos::deep_copy(aux,ax);
    m1_equations_system_t sys(state,staggered_variable_arrays_t{},aux);
    Kokkos::View<double*> result("transport_opacity",2);
    Kokkos::parallel_for("transport_decay",1,KOKKOS_LAMBDA(int) {
        result(0)=sys.transport_opacity<0>(0,VEC(0,0,0));
        double xyz[3]={};
        auto const tau=make_lagged_kappa_tau(aux,VEC(0,0,0),0,2,xyz);
        result(1)=tau.tau[NUE];
    });
    auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
    REQUIRE_THAT(r(0),WithinRel(5+opacity[0],1e-14));
    REQUIRE_THAT(r(1),WithinRel(2*(5+opacity[0]),1e-14));
    auto kept=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),aux);
    REQUIRE(kept(0,0,0,KAPPAA1_,0)==2);
}

TEST_CASE("Stiff four-flavour network conserves charge after removing ordinary sources", "[decay]") {
    for(bool pairs: {false,true}) {
        kernel thermal; REQUIRE(thermal.init({20,50,6e37,0.8,0.2},{pairs,pairs,false},16,35));
        decay_kernel decay; REQUIRE(decay.init(thermal.g,20,50,90,24));
        metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},out[20],res;
        for(int s=0;s<4;++s) {
            set_radiation(thermal.g,10+2*s,-0.4+0.2*s,s,base,old);
            eas[s][KANL]=0.2*(s+1); eas[s][KAL]=0.3*(s+1);
            eas[s][ETANL]=eas[s][KANL]*old[5*s+4];
            eas[s][ETAL]=eas[s][KAL]*old[5*s];
        }
        for(double seconds: {1e-8,1e-6,1e-4,1e-2}) {
            INFO("thermal="<<pairs<<", dt [s]="<<seconds);
            double const dt=seconds*time_unit;
            REQUIRE(leptonic_implicit_update(thermal,decay,pairs,metric,base,eas,old,out,dt,res));
            REQUIRE(res<2e-8);
            double q[4],norm=0;
            for(int s=0;s<4;++s) {
                REQUIRE(out[5*s]>0); REQUIRE(out[5*s+4]>0);
                q[s]=out[5*s+4]-old[5*s+4]-dt*metric.alp()*(eas[s][ETANL]-eas[s][KANL]*out[5*s+4]);
                norm+=out[5*s+4]+old[5*s+4];
            }
            REQUIRE(std::abs(q[0]-q[1]+q[2]-q[3])<1e-10*norm);
            if(!pairs) { REQUIRE(std::abs(q[0]-q[3])<1e-10*norm); REQUIRE(std::abs(q[1]-q[2])<1e-10*norm); }
        }
    }
}

#if GRACE_M1_NU_SPECIES >= 5
namespace {
struct decay_config_guard {
    YAML::Node eas=YAML::Clone(config_parser::get()["m1"]["eas"]);
    YAML::Node eos=YAML::Clone(config_parser::get()["eos"]);
    ~decay_config_guard() {
        config_parser::get()["m1"]["eas"]=eas;
        config_parser::get()["eos"]=eos;
    }
    void enable() {
        // The lightweight Kokkos main has no GRACE loggers; setup warnings
        // and ERROR paths require them. No files are created by these sinks.
        if(!spdlog::get("output_console")) spdlog::stdout_logger_mt("output_console");
        if(!spdlog::get("error_console")) spdlog::stderr_logger_mt("error_console");
        for(auto const* prefix: {"file_logger_","error_file_logger_","backtrace_logger_"}) {
            std::string const name=std::string(prefix)+std::to_string(parallel::mpi_comm_rank());
            if(!spdlog::get(name)) spdlog::register_logger(std::make_shared<spdlog::logger>(
                name,std::make_shared<spdlog::sinks::null_sink_mt>()));
        }
        auto& c=config_parser::get();
        c["m1"]["eas"]["kind"]="neutrino_analytic";
        c["m1"]["eas"]["kinds"]=std::vector<std::string>{};
        c["m1"]["eas"]["muon_decay"]=true;
        c["m1"]["eas"]["muon_decay_treatment"]="auto";
        c["m1"]["eas"]["pair_treatment"]="evolved";
        c["m1"]["eas"]["betaeq_policy"]="off";
        c["m1"]["eas"]["plasmon_decay"]=false;
        c["eos"]["eos_type"]="leptonic";
        c["eos"]["leptonic"]["dilute_muon_suppression"]=false;
        c["eos"]["leptonic"]["add_ele_contribution"]=true;
    }
};
}

TEST_CASE("Muon decay setup is opt-in and accepts both explicit partner modes", "[decay]") {
    REQUIRE_FALSE(get_muon_decay());
    REQUIRE(get_muon_decay_kernel_order()==24);
    decay_config_guard save; save.enable();
    REQUIRE(get_muon_decay());
    REQUIRE(get_spectral_muon_decay());
    REQUIRE_FALSE(get_vacuum_muon_decay());
    REQUIRE(get_pair_treatment()==pair_treatment_t::evolved);
    config_parser::get()["m1"]["eas"]["muon_decay_treatment"]="lagged";
    REQUIRE(get_muon_decay_treatment()==muon_decay_treatment_t::lagged);
    REQUIRE(get_spectral_muon_decay());
    config_parser::get()["m1"]["eas"]["muon_decay_treatment"]="singlepass";
    REQUIRE(get_muon_decay_treatment()==muon_decay_treatment_t::singlepass);
    REQUIRE(get_spectral_muon_decay());
    config_parser::get()["m1"]["eas"]["muon_decay_treatment"]="auto";
    config_parser::get()["m1"]["eas"]["pair_treatment"]="equilibrium";
    REQUIRE(get_pair_treatment()==pair_treatment_t::equilibrium);
    config_parser::get()["m1"]["eas"]["kind"]="neutrino_weakhub";
    config_parser::get()["m1"]["eas"]["weakhub_table_content"]="none";
    (void)get_weakhub_table_content(true);
    REQUIRE(get_pair_treatment()==pair_treatment_t::equilibrium);
}

TEST_CASE("Vacuum decay can accompany either thermal pair kernel without selecting spectral decay",
          "[decay][setup]") {
    decay_config_guard save; save.enable();
    auto& c=config_parser::get();
    c["m1"]["eas"]["muon_decay_treatment"]="vacuum";
    c["eos"]["leptonic"]["dilute_muon_suppression"]=true;
    for(auto const* mode: {"equilibrium","evolved"}) {
        c["m1"]["eas"]["pair_treatment"]=mode;
        REQUIRE(get_pair_treatment()==(std::string(mode)=="evolved"
            ? pair_treatment_t::evolved : pair_treatment_t::equilibrium));
        check_m1_setup();
        REQUIRE(get_vacuum_muon_decay());
        REQUIRE_FALSE(get_spectral_muon_decay());
    }
    // The original standard configuration still selects vacuum automatically.
    c["m1"]["eas"]["pair_treatment"]="standard";
    c["m1"]["eas"]["muon_decay_treatment"]="auto";
    check_m1_setup();
    REQUIRE(get_vacuum_muon_decay());
    // The master switch overrides an explicit prescription.
    c["m1"]["eas"]["muon_decay"]=false;
    for(auto const* mode: {"auto","vacuum","spectral"}) {
        c["m1"]["eas"]["muon_decay_treatment"]=mode;
        REQUIRE_FALSE(get_vacuum_muon_decay());
        REQUIRE_FALSE(get_spectral_muon_decay());
    }
}

// The standard path's vacuum decay sink against the spectral decay kernel at a dilute,
// non-degenerate state: muons at rest with Michel energies vs the finite-mass thermal kernel
// (time dilation ~3T/2m and the thermal boost of the energies are the expected differences).
TEST_CASE("Vacuum decay sink agrees with the spectral decay kernel in dilute matter", "[decay][vacuum]") {
    double const T=5.0, mu_e=1.0, mu_mu=90.0;
    grid g; g.init(32,40.0);
    decay_kernel k; REQUIRE(k.init(g,T,mu_e,mu_mu,32));
    decay_lte_rates kr; REQUIRE(equilibrium_decay_rates(k,0,-50.0,-50.0,kr));   // no partner sea
    // The kernel's mu- density: FD integral with the finite mass, g = 2.
    double n_mu=0; { int const N=40000; double const emax=muon_mass+60*T;
        double const pmax=std::sqrt(emax*emax-muon_mass*muon_mass), dp=pmax/N;
        for(int i=0;i<N;++i) { double const p=(i+0.5)*dp, E=std::sqrt(p*p+muon_mass*muon_mass);
            n_mu+=p*p/(std::exp((E-mu_mu)/T)+1)*dp; }
        n_mu/=M_PI*M_PI*hbarc_mev_cm*hbarc_mev_cm*hbarc_mev_cm; }
    fugacity_state F;
    F.rho_cgs=n_mu*nu_constants::mnuc_cgs; F.ymu=1.0; F.ymu_dilute=0.0;   // rho/m_p = n_mu
    F.temp_mev=T; F.mu_e=mu_e; F.mu_mu=mu_mu; F.eta_e=mu_e/T;
    F.eta_nu={{-50.0,-50.0,-50.0,-50.0,0.0}};
    rates_accum out{}; add_muon_decay_emission(F,out,0.0);
    REQUIRE_THAT(out.R[NUMU],WithinRel(kr.eta_N[0],0.15));
    REQUIRE_THAT(out.R[NUEBAR],WithinRel(kr.eta_N[1],0.15));
    REQUIRE_THAT(out.Q[NUMU]/out.R[NUMU],WithinRel(kr.eta_E[0]/kr.eta_N[0],0.10));
    REQUIRE_THAT(out.Q[NUEBAR]/out.R[NUEBAR],WithinRel(kr.eta_E[1]/kr.eta_N[1],0.10));
}

// Run in a subprocess: invalid setup intentionally terminates via ERROR.
TEST_CASE("Muon decay rejects incompatible setup", "[.][decay-invalid]") {
    decay_config_guard save; save.enable();
    auto& c=config_parser::get();
    char const* env=std::getenv("GRACE_DECAY_INVALID"); REQUIRE(env!=nullptr);
    std::string const mode=env;
    if(mode=="eos") c["eos"]["eos_type"]="ideal";
    else if(mode=="dilute") c["eos"]["leptonic"]["dilute_muon_suppression"]=true;
    else if(mode=="electrons") c["eos"]["leptonic"]["add_ele_contribution"]=false;
    else if(mode=="table") c["m1"]["eas"]["kind"]="neutrino_weakhub";
    else if(mode=="betaeq") c["m1"]["eas"]["betaeq_policy"]="gieg";
    else if(mode=="order") c["m1"]["eas"]["muon_decay_kernel_order"]=65;
    else if(mode=="vacuum_dilute") c["m1"]["eas"]["muon_decay_treatment"]="vacuum";
    else if(mode=="spectral_standard") {
        c["m1"]["eas"]["muon_decay_treatment"]="spectral";
        c["m1"]["eas"]["pair_treatment"]="standard";
    }
    else if(mode=="grey_equilibrium") {
        c["m1"]["eas"]["muon_decay_treatment"]="grey_decay";
        c["m1"]["eas"]["pair_treatment"]="equilibrium";
    }
    else if(mode=="treatment") c["m1"]["eas"]["muon_decay_treatment"]="invalid";
    else FAIL("Unknown invalid setup case");
    (void)get_pair_treatment();
    check_m1_setup();
    FAIL("Incompatible decay setup was accepted");
}

TEST_CASE("GRACE cross-flavour source wiring uses the correct species and densitization", "[decay]") {
    for(bool grey: {false,true}) {
    var_array_t state("decay_state",1,1,1,N_EVOL_VARS,1),next("decay_next",1,1,1,N_EVOL_VARS,1);
    var_array_t aux("decay_aux",1,1,1,N_AUX_VARS,1);
    auto st=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state);
    auto ax=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),aux);
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
    st(0,0,0,GXX_,0)=4; st(0,0,0,GYY_,0)=st(0,0,0,GZZ_,0)=1;
    #else
    st(0,0,0,GTXX_,0)=4; st(0,0,0,GTYY_,0)=st(0,0,0,GTZZ_,0)=1; st(0,0,0,CHI_,0)=1;
    #endif
    st(0,0,0,ALP_,0)=0.7;
    ax(0,0,0,TEMP_,0)=20; ax(0,0,0,PAIR_MUE_,0)=30; ax(0,0,0,PAIR_MUMU_,0)=90;
    ax(0,0,0,RHO_,0)=6e37/nu_constants::avogadro*nu_constants::RHOGF;   // n_b = 6e37 cm^-3
    ax(0,0,0,PAIR_YN_,0)=0.8; ax(0,0,0,PAIR_YP_,0)=0.2;
    ax(0,0,0,PAIR_ACTIVE_,0)=8;   // decay only, no thermal channels
    Kokkos::deep_copy(state,st); Kokkos::deep_copy(next,state); Kokkos::deep_copy(aux,ax);
    m1_equations_system_t system(state,staggered_variable_arrays_t{},aux);
    scalar_array_t<GRACE_NSPACEDIM> idx;
    Kokkos::parallel_for("decay_wiring",1,KOKKOS_LAMBDA(int) {
        system.compute_leptonic_implicit_update(0,VEC(0,0,0),idx,next,0.01,1,16,24,nullptr,{},false,false,true,true,true,grey);
    });
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),next);
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,muon_mass/3));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24,grey));
    metric_array_t metric({4,0,0,1,0,1},{0,0,0},0.7);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double old[20]={},out[20],res;
    REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,old,out,0.01,res));
    int const e[4]={ERAD1_,ERAD2_,ERAD3_,ERAD4_},n[4]={NRAD1_,NRAD2_,NRAD3_,NRAD4_};
    for(int s=0;s<4;++s) {
        REQUIRE_THAT(got(0,0,0,e[s],0),WithinRel(2*out[5*s],2e-10));
        REQUIRE_THAT(got(0,0,0,n[s],0),WithinRel(2*out[5*s+4],2e-10));
    }
    REQUIRE(got(0,0,0,NRAD5_,0)==0);
    }
}
#endif

TEST_CASE("Analytic collision blocks reproduce every single-species probe", "[decay][block_jacobian]") {
    kernel k; REQUIRE(k.init({20,50,6e37,0.8,0.2},{true,true,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(k.g,20,50,90,16));
    for(int reaction=0;reaction<3;++reaction) for(double eta: {-12.,0.,4.}) {
        double f[max_order],fb[max_order];
        for(int i=0;i<k.g.n;++i) { f[i]=occupation(k.g.e[i]/17-eta); fb[i]=occupation(k.g.e[i]/23+eta); }
        auto eval=[&](double const* a,double const* b) {
            return reaction==0?evaluate(k,a,b):evaluate_decay(decay,reaction-1,a,b);
        };
        collision_block block;
        block.init(k.g,f,fb,eval(f,fb),[&](int i,int j,double& p,double& a) {
            if(reaction) decay.rates(reaction-1,i,j,p,a);
            else { a=k.a[i][j]; p=a*k.boltz[i]*k.boltz[j]; }
        });
        for(int side=0;side<2;++side) for(double factor: {0.3,1.00001,1.4}) {
            double a[max_order],b[max_order];
            for(int i=0;i<k.g.n;++i) {
                a[i]=side==0?f[i]*factor/(1+(factor-1)*f[i]):f[i];
                b[i]=side==1?fb[i]*factor/(1+(factor-1)*fb[i]):fb[i];
            }
            auto exact=eval(a,b),got=block.evaluate(k.g,a,b);
            REQUIRE_THAT(got.number,WithinAbs(exact.number,2e-13*(exact.loss_n[0]+exact.emission_n[0])));
            for(int s=0;s<2;++s) {
                REQUIRE_THAT(got.energy[s],WithinAbs(exact.energy[s],2e-13*(exact.loss[s]+exact.emission[s])));
                REQUIRE_THAT(got.loss[s],WithinRel(exact.loss[s],2e-13));
            }
        }
    }
}

TEST_CASE("Block Jacobian matches conservative spectral solves in moving and stiff matter", "[decay][block_jacobian]") {
    for(bool pairs: {false,true}) for(double velocity: {0.,0.25}) for(double eta: {-10.,0.}) {
        kernel thermal; REQUIRE(thermal.init({20,50,6e37,0.8,0.2},{pairs,pairs,false},16,35));
        decay_kernel decay; REQUIRE(decay.init(thermal.g,20,50,90,16));
        metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},ref[20],out[20],res;
        for(int s=0;s<4;++s) {
            set_radiation(thermal.g,10+2*s,eta-0.4+0.2*s,s,base,old,velocity);
            eas[s][KANL]=0.2*(s+1); eas[s][KAL]=0.3*(s+1);
            eas[s][ETANL]=eas[s][KANL]*old[5*s+4]; eas[s][ETAL]=eas[s][KAL]*old[5*s];
        }
        for(double seconds: {1e-8,1e-6,1e-4,1e-2}) {
            INFO("pairs="<<pairs<<" velocity="<<velocity<<" eta="<<eta<<" seconds="<<seconds);
            double const dt=seconds*time_unit;
            REQUIRE(leptonic_implicit_update(thermal,decay,pairs,metric,base,eas,old,ref,dt,res));
            REQUIRE(leptonic_implicit_update(thermal,decay,pairs,metric,base,eas,old,out,dt,res,nullptr,false,true));
            double q[4],norm=0;
            for(int s=0;s<4;++s) {
                for(int v=0;v<5;++v) REQUIRE_THAT(out[5*s+v],WithinAbs(ref[5*s+v],2e-8*ref[5*s+(v==4?4:0)]));
                m1_closure_t cl{base[s],metric};
                cl.update_closure(out[5*s],{out[5*s+1],out[5*s+2],out[5*s+3]},0,true);
                q[s]=(1+dt*metric.alp()*eas[s][KANL]/cl.Gamma)*out[5*s+4]-old[5*s+4]-dt*metric.alp()*eas[s][ETANL];
                norm+=out[5*s+4]+old[5*s+4];
            }
            REQUIRE(std::abs(q[0]-q[1]+q[2]-q[3])<1e-10*norm);
        }
    }
}

TEST_CASE("Block Jacobian: Four-flavour network preserves LTE including ordinary EAS and moving matter", "[decay][block_jacobian]") {
    kernel thermal; REQUIRE(thermal.init({15,70,6e37,0.8,0.2},{true,true,false},16,35));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    for(double ymu: {5e-4,6.5e-4,0.01}) for(double v: {0.,0.4}) {
        double const mu=dilute_muon_cc_potential(110,70,ymu);
        decay_kernel decay; REQUIRE(decay.init(thermal.g,15,70,mu,24));
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},out[20],res;
        double const eta[4]={-0.3,0.3,(mu-70)/15-0.3,-(mu-70)/15+0.3};
        for(int s=0;s<4;++s) {
            set_radiation(thermal.g,15,eta[s],s,base,old,v);
            double n,J; fd_moments(thermal.g,15,eta[s],n,J);
            eas[s][KAL]=2; eas[s][KANL]=3; eas[s][KSL]=4;
            eas[s][ETAL]=2*J*energy_unit; eas[s][ETANL]=3*n*number_unit;
        }
        REQUIRE(leptonic_implicit_update(thermal,decay,true,metric,base,eas,old,out,100,res,nullptr,false,true));
        REQUIRE(res<2e-8);
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
        int iterations=0; bool fallback=true;
        REQUIRE(leptonic_lagged_update(thermal,decay,true,metric,base,eas,old,out,100,res,nullptr,&iterations,&fallback));
        REQUIRE_FALSE(fallback);
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
        REQUIRE(leptonic_singlepass_update(thermal,decay,true,metric,base,eas,old,out,100,res));
        for(int s=0;s<4;++s) for(int a: {0,1,4})
            REQUIRE_THAT(out[5*s+a],WithinAbs(old[5*s+a],3e-8*old[5*s]));
    }
}

TEST_CASE("Block Jacobian: A previous-stage guess solves an admissible implicit state from a negative RHS", "[decay][block_jacobian]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double expected[20]{},rhs[20],guess[20],out[20],residual;
    // At this state each decay edge obeys chemical detailed balance.
    double const eta[4]={-1,1,2,-2};
    for(int s=0;s<4;++s) set_radiation(thermal.g,20,eta[s],s,base,expected);
    for(int a=0;a<20;++a) { rhs[a]=expected[a]; guess[a]=0.98*expected[a]; }
    // Manufacture an inadmissible explicit RHS whose full implicit solution
    // is the LTE state above. The constant ordinary source supplies the deficit.
    double const h=0.01;
    rhs[0]=base[0][ERADL]=-expected[0];
    rhs[4]=base[0][NRADL]=-expected[4];
    eas[0][ETAL]=2*expected[0]/h;
    eas[0][ETANL]=2*expected[4]/h;
    REQUIRE_FALSE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,rhs,out,h,residual,nullptr,false,true));
    REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,rhs,out,h,residual,guess,false,true));
    for(int s=0;s<4;++s) {
        REQUIRE_THAT(out[5*s],WithinRel(expected[5*s],1e-7));
        REQUIRE_THAT(out[5*s+4],WithinRel(expected[5*s+4],1e-7));
    }
}

TEST_CASE("Block Jacobian: Four-flavour source emits into vacuum with equal daughter counts", "[decay][block_jacobian]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double old[20]={},out[20],res;
    REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,old,out,0.01,res,nullptr,false,true));
    REQUIRE(res<2e-8);
    for(int s=0;s<4;++s) { REQUIRE(out[5*s]>0); REQUIRE(out[5*s+4]>0); }
    REQUIRE(out[4]==out[19]); REQUIRE(out[9]==out[14]);
    REQUIRE(out[10]/out[14]>out[5]/out[9]); // oriented hard/soft daughters
}

TEST_CASE("Grey decay is opt-in and requires conservative evolved partners", "[decay][grey_decay]") {
    decay_config_guard save; save.enable();
    config_parser::get()["m1"]["eas"]["muon_decay_treatment"]="grey_decay";
    config_parser::get()["m1"]["eas"]["grey_decay_kernel_nodes"]=8;
    REQUIRE(get_muon_decay_treatment()==muon_decay_treatment_t::grey_decay);
    REQUIRE(get_spectral_muon_decay()); REQUIRE_FALSE(get_vacuum_muon_decay());
    REQUIRE(get_grey_muon_decay());
    REQUIRE(get_pair_treatment()==pair_treatment_t::evolved);
}


TEST_CASE("Cheap grey decay obeys detailed balance and charge conjugation without kernels", "[decay][grey_decay]") {
    #ifdef GRACE_LEPTON_PROFILE
    auto const kernels=source_profile::calls[source_profile::decay_kernel].load();
    auto const hits=source_profile::decay_cache_hits.load(),misses=source_profile::decay_cache_misses.load();
    #endif
    grid g; g.init(16,35);
    for(auto const& matter: {std::array<double,3>{3,30,90}, {20,50,90}, {50,100,150}, {.1,0,0}}) {
        double const T=matter[0],mue=matter[1],mumu=matter[2];
        decay_kernel k; REQUIRE(init_decay_kernel(k,g,T,mue,mumu,16,true,8)); REQUIRE(k.grey);
        grey_decay_rates conjugate; REQUIRE(conjugate.init(T,-mue,-mumu));
        for(int c=0;c<2;++c) {
            auto const& r=k.grey_rates;
            REQUIRE(r.production[c]>=0); REQUIRE(r.absorption[c]>=0);
            REQUIRE(r.production[c]==conjugate.production[1-c]);
            REQUIRE(r.absorption[c]==conjugate.absorption[1-c]);
            double const delta=(c==0?1:-1)*(mumu-mue);
            double sh0[]={-.3,1/T},sh1[]={-(delta/T-.3),1/T};
            auto eq=r.evaluate(c,sh0,sh1);
            double const scale=eq.emission_n[0]+eq.loss_n[0];
            REQUIRE(std::abs(eq.number)<=1e-11*scale+1e-300);
            double empty[]={1e300,0},filled[]={-100,0};
            auto forward=r.evaluate(c,empty,empty),inverse=r.evaluate(c,filled,filled);
            REQUIRE(forward.number==r.production[c]);
            REQUIRE(inverse.number<=0);
            REQUIRE(r.evaluate(c,empty,filled).number>=0); // missing incoming partner forbids inverse
            REQUIRE(forward.energy[0]==r.energy[c][0]*forward.number);
            REQUIRE(forward.energy[1]==r.energy[c][1]*forward.number);
            decay_lte_rates lte; REQUIRE(equilibrium_decay_rates(k,c,-1000,1000,lte));
            REQUIRE(std::isfinite(lte.kappa_E[0]+lte.kappa_E[1]));
        }
    }
    #ifdef GRACE_LEPTON_PROFILE
    REQUIRE(source_profile::calls[source_profile::decay_kernel].load()==kernels);
    REQUIRE(source_profile::decay_cache_hits.load()==hits);
    REQUIRE(source_profile::decay_cache_misses.load()==misses);
    #endif
}
TEST_CASE("Grey rest limit recovers vacuum lifetime and Michel means", "[decay][grey_decay]") {
    grey_decay_rates r; REQUIRE(r.init(.01,0,105.5));
    double const M=105.6583755;
    REQUIRE(std::abs(r.mean[0]/M-1)<2e-4);
    REQUIRE(std::abs(r.production[0]/r.density[0]*2.197e-6-1)<2e-4);
    REQUIRE(std::abs(r.energy[0][0]/(.35*M)-1)<2e-4);
    REQUIRE(std::abs(r.energy[0][1]/(.30*M)-1)<2e-4);
}
TEST_CASE("FD fit parameters reproduce moments and point occupations", "[decay][grey_decay]") {
    grid g; g.init(16,35);
    for(double eta: {-20.,0.,10.}) {
        double n=0,J=0,f[max_order],shape[2];
        for(int i=0;i<g.n;++i) { double v=occupation(g.e[i]/20-eta);n+=g.w[i]*v;J+=g.w[i]*g.e[i]*v; }
        REQUIRE(reconstruct(g,n,J,f,nullptr,shape));
        REQUIRE(std::abs(shape[0]+eta)<1e-8);
        REQUIRE(std::abs(shape[1]-.05)<1e-10);
    }
}

TEST_CASE("Reversible grey source conserves daughter events during emission inverse decay and stiff evolution", "[decay][grey_decay]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,16,8));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    for(double eta: {-1000.,0.,5.}) for(double velocity: {0.,0.25}) {
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{}; double old[20]={},out[20],res;
        if(eta>-100) for(int s=0;s<4;++s) set_radiation(thermal.g,20,eta,s,base,old,velocity);
        for(double seconds: {1e-8,1e-6,1e-4}) {
            REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,old,out,seconds*time_unit,res,nullptr,false,true));
            double const q0=out[4]-old[4],q1=out[9]-old[9],q2=out[14]-old[14],q3=out[19]-old[19];
            double const scale=out[4]+out[9]+out[14]+out[19]+old[4]+old[9]+old[14]+old[19];
            REQUIRE(std::abs(q0-q3)<1e-12*scale); REQUIRE(std::abs(q1-q2)<1e-12*scale);
            if(eta<-100) { REQUIRE(q1>0); REQUIRE(q0>0); }
            if(eta==5) { REQUIRE(q1<0); REQUIRE(q0<0); } // incoming neutrinos actually create muons
        }
    }
}

TEST_CASE("Grey transport damping and chemical LTE use the same reaction", "[decay][grey_decay]") {
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,.8,.2},{false,false,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,16,1));
    double const eta[4]={.1,2.8,.2,-3.1};
    double n[4]{},J[4]{},opacity[4];
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
    double old[20]{},out[20],res;
    for(int s=0;s<4;++s) {
        set_radiation(thermal.g,20,eta[s],s,base,old,0);
        n[s]=old[5*s+4]/number_unit; J[s]=old[5*s]/energy_unit;
    }
    REQUIRE(leptonic_transport_opacities(thermal,&decay,false,n,J,opacity));
    for(int c=0;c<2;++c) {
        int const a=c==0?2:3,b=c==0?1:0;
        double shape0[]={-eta[a],.05},shape1[]={-eta[b],.05};
        auto src=decay.grey_rates.evaluate(c,shape0,shape1);
        REQUIRE_THAT(opacity[a],WithinRel(src.loss[0]/J[a]/time_unit,1e-8));
        REQUIRE_THAT(opacity[b],WithinRel(src.loss[1]/J[b]/time_unit,1e-8));
    }
    for(bool block: {false,true}) {
        REQUIRE(leptonic_implicit_update(thermal,decay,false,metric,base,eas,old,out,1e-4*time_unit,res,nullptr,false,block));
        for(int s=0;s<4;++s) {
            REQUIRE_THAT(out[5*s],WithinRel(old[5*s],1e-8));
            REQUIRE_THAT(out[5*s+4],WithinRel(old[5*s+4],1e-8));
        }
    }
}
