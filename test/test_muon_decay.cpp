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

TEST_CASE("Four-flavour network preserves LTE including ordinary EAS and moving matter", "[decay]") {
    kernel thermal; REQUIRE(thermal.init({15,70,6e37,0.8,0.2},{true,true,false},16,35));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,15,70,110,24));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    for(double v: {0.,0.4}) {
        m1_prims_array_t base[4]{}; m1_eas_array_t eas[4]{};
        double old[20]={},out[20],res;
        double const eta[4]={-0.3,0.3,40./15-0.3,-40./15+0.3};
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
    REQUIRE(get_pair_treatment()==pair_treatment_t::evolved);
    config_parser::get()["m1"]["eas"]["pair_treatment"]="equilibrium";
    REQUIRE(get_pair_treatment()==pair_treatment_t::equilibrium);
    config_parser::get()["m1"]["eas"]["kind"]="neutrino_weakhub";
    config_parser::get()["m1"]["eas"]["weakhub_pair_content"]="none";
    config_parser::get()["m1"]["eas"]["weakhub_muon_decay_content"]="none";
    REQUIRE(get_pair_treatment()==pair_treatment_t::equilibrium);
}

// Run in a subprocess: invalid setup intentionally terminates via ERROR.
TEST_CASE("Muon decay rejects incompatible setup", "[.][decay-invalid]") {
    decay_config_guard save; save.enable();
    auto& c=config_parser::get();
    char const* env=std::getenv("GRACE_DECAY_INVALID"); REQUIRE(env!=nullptr);
    std::string const mode=env;
    if(mode=="legacy") c["m1"]["eas"]["pair_treatment"]="legacy";
    else if(mode=="eos") c["eos"]["eos_type"]="ideal";
    else if(mode=="dilute") c["eos"]["leptonic"]["dilute_muon_suppression"]=true;
    else if(mode=="electrons") c["eos"]["leptonic"]["add_ele_contribution"]=false;
    else if(mode=="table") c["m1"]["eas"]["kind"]="neutrino_weakhub";
    else if(mode=="betaeq") c["m1"]["eas"]["betaeq_policy"]="gieg";
    else if(mode=="order") c["m1"]["eas"]["muon_decay_kernel_order"]=65;
    else FAIL("Unknown invalid setup case");
    (void)get_pair_treatment();
    FAIL("Incompatible decay setup was accepted");
}

TEST_CASE("GRACE cross-flavour source wiring uses the correct species and densitization", "[decay]") {
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
    ax(0,0,0,PAIR_T_,0)=20; ax(0,0,0,PAIR_MUE_,0)=30; ax(0,0,0,PAIR_MUMU_,0)=90;
    ax(0,0,0,PAIR_NB_,0)=6e37; ax(0,0,0,PAIR_YN_,0)=0.8; ax(0,0,0,PAIR_YP_,0)=0.2;
    ax(0,0,0,PAIR_DECAY_,0)=1; ax(0,0,0,PAIR_DORDER_,0)=24;
    Kokkos::deep_copy(state,st); Kokkos::deep_copy(next,state); Kokkos::deep_copy(aux,ax);
    m1_equations_system_t system(state,staggered_variable_arrays_t{},aux);
    scalar_array_t<GRACE_NSPACEDIM> idx;
    Kokkos::parallel_for("decay_wiring",1,KOKKOS_LAMBDA(int) {
        system.compute_leptonic_implicit_update(0,VEC(0,0,0),idx,next,0.01,1,16);
    });
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),next);
    kernel thermal; REQUIRE(thermal.init({20,30,6e37,0.8,0.2},{false,false,false},16,muon_mass/3));
    decay_kernel decay; REQUIRE(decay.init(thermal.g,20,30,90,24));
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
#endif
