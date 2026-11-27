// Explicit validation experiments, using the production EOS and source solver.
// All scientific failures are recorded; no rates or populations are clipped.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include "leptonic_source_replay.hh"
#include <catch2/catch_session.hpp>
#include <grace/system/mpi_runtime.hh>
#include <grace/system/kokkos_runtime.hh>
#include <grace/physics/eas_harry_rates.hh>
#include <grace/physics/eas_kinds.hh>
#include <grace/physics/harry_pair_kernels.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/neutrino_leptonic_update.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace grace;
using namespace grace::pairs;
namespace grace { void initialize_loggers(); }
// Source-only experiments do not initialize a full star: doing so would test
// unrelated out-of-domain cells before the explicitly selected box states.
int main(int argc,char** argv) {
    std::string parfile;std::vector<char*> args{argv[0]};
    for(int i=1;i<argc;++i) {
        if(std::string(argv[i])=="--grace-parfile" && i+1<argc) parfile=argv[++i];
        else args.push_back(argv[i]);
    }
    if(parfile.empty()) throw std::runtime_error("harry_validation requires --grace-parfile");
    grace::config_parser::initialize(parfile);
    grace::mpi_runtime::initialize(argc,argv);
    grace::kokkos_runtime::initialize(&argc,argv);
    grace::initialize_loggers();
    grace::eos::initialize();
    grace::check_m1_setup();
    grace::weakhub::initialize_weakhub_from_params();
    int const catch_argc=int(args.size());args.push_back(nullptr);
    int result=Catch::Session().run(catch_argc,args.data());
    grace::weakhub::finalize_weakhub();
    return result;
}
namespace {
struct state { double rho,T,ye,ymu; };
TEST_CASE("Loaded provider survives an unavailable rate file until teardown", "[.][provider_lifetime]") {
    auto& config=config_parser::get();
    REQUIRE(get_param<std::string>("m1","eas","weakhub_format")=="auto");
    bool const harry_expected=!weakhub::get_device_handle().valid;
    REQUIRE(get_harry_table_mode()==harry_expected);
    auto const original=get_param<std::string>("m1","eas","weakhub_table");
    config["m1"]["eas"]["weakhub_table"]=original+".deliberately-absent";
    REQUIRE(get_harry_table_mode()==harry_expected);
    weakhub::finalize_weakhub();
    REQUIRE_THROWS_AS(get_harry_table_mode(),std::runtime_error);
    config["m1"]["eas"]["weakhub_table"]=original;
    weakhub::initialize_weakhub_from_params();
    REQUIRE(get_harry_table_mode()==harry_expected);
}
bool potentials(leptonic_eos_4d_t const& e,state s,double& mue,double& mumu) {
    auto initial=s; double mup,mun,Xa,Xh,Xn,Xp,A,Z; eos_err_t err{};
    mue=e.mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(
        mumu,mup,mun,Xa,Xh,Xn,Xp,A,Z,s.T,s.rho,s.ye,s.ymu,err);
    return s.rho==initial.rho && s.T==initial.T && s.ye==initial.ye && s.ymu==initial.ymu
        && std::isfinite(mue+mumu);
}
double internal_energy(leptonic_eos_4d_t const& e,state s) {
    eos_err_t err{}; return e.eps__temp_rho_ye_ymu(s.T,s.rho,s.ye,s.ymu,err);
}
std::string output(char const* stem) {
    char const* path=std::getenv("GRACE_CC_VALIDATION_OUTPUT");
    return std::string(path?path:".")+"/"+stem+"_rank"+std::to_string(parallel::mpi_comm_rank())+".csv";
}
std::ostringstream stream() { std::ostringstream s;s<<std::setprecision(17);return s; }
void spectrum(grid const& g,double T,double eta,double* f,double* shape) {
    shape[0]=-eta;shape[1]=1/T;
    for(int i=0;i<g.n;++i) f[i]=occupation(g.e[i]/T-eta);
}
std::vector<state> states(double ymu_floor) {
    std::vector<state> v;
    for(double rho:{1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15})
    for(double T:{.5,1.,3.,5.,10.,20.,40.,80.})
    for(double ym:{ymu_floor,.00055,.0006,.000625,.00065,.0008,.001,.01,.05,.1})
    for(double ye:{.05,.15,.3}) v.push_back({rho*nu_constants::RHOGF,T,ye,ym});
    return v;
}
fugacity_state rate_state(leptonic_eos_4d_t const& e,state s,bool regularize) {
    double xyz[3]{};
    auto F=make_fugacity_state(e,s.rho,s.T,s.ye,s.ymu,1.,xyz,tau_policy_none{});
    F.mu_mu_reaction=F.mu_mu;
    if(regularize) {
        F.dilute_cc_target=true;
        F.mu_mu_reaction=dilute_muon_cc_potential(e.raw_mumu__temp_rho_ye_ymu(s.T,s.rho,s.ye,s.ymu),F.mu_e,s.ymu);
    }
    F.eta_cc_mu=(F.mu_mu_reaction+F.mu_p-F.mu_n-nu_constants::Qnp)/s.T;
    F.eta_nu[NUMU]=F.eta_cc_mu;F.eta_nu[NUMUBAR]=-F.eta_cc_mu;
    F.eta_nu[NUE]=(F.mu_e+F.mu_p-F.mu_n-nu_constants::Qnp)/s.T;
    F.eta_nu[NUEBAR]=-F.eta_nu[NUE];
    return F;
}
nu_rates_all_out ordinary(fugacity_state const& F,double const* eps) {
    double xyz[3]{};
    static bool const correct=get_param<bool>("m1","eas","temperature_correction");
    if(get_harry_table_mode()) return compute_all_species_harry(harry::get_harry_device_handle(),F,eps);
    return compute_all_species_weakhub(weakhub::get_device_handle(),F,false,false,false,
        xyz,tau_policy_none{},correct,eps,false,false,false,0);
}
m1_eas_array_t rate_array(nu_rates_out const& r,double multiplicity=1) {
    m1_eas_array_t a{};a[KAL]=r.kappa_a;a[KSL]=r.kappa_s;a[KANL]=r.kappa_n;
    a[ETAL]=r.eta_E/multiplicity;a[ETANL]=r.eta_N/multiplicity;return a;
}
bool table_inside(state s) {
    if(get_harry_table_mode()) {
        auto const& h=harry::get_harry_device_handle();
        auto const& e=eos::get().get_eos<leptonic_eos_4d_t>();
        double x[4]={std::log(s.rho*harry_density_ratio(e.get_baryon_mass())),std::log(s.T),s.ye,std::log(s.ymu)};
        return h.cc[0].contains(x)&&h.cc[2].contains(x);
    }
    auto const& h=weakhub::get_device_handle();
    return std::log(s.rho)>=h.logrho_min && std::log(s.rho)<=h.logrho_max
        && std::log(s.T)>=h.logtemp_min && std::log(s.T)<=h.logtemp_max
        && s.ye>=h.ye_min && s.ye<=h.ye_max
        && std::log(s.ymu)>=h.logymu_min && std::log(s.ymu)<=h.logymu_max;
}

}
namespace { std::string ablation_bounds(leptonic_eos_4d_t const&,state,bool,harry::device_handle const&); }
namespace {
// Identify an outgoing trajectory only when its accepted state is already
// at the face to source-solver precision. A distant rejected Newton probe
// cannot classify the trajectory. The caller verifies the sign twice.
std::string outgoing_composition_face(leptonic_eos_4d_t const& eos, state accepted,
                                      state trial, bool harry_mode) {
    auto face=[](double value,double next,double lower,double upper,std::string name) {
        if(next<lower && std::abs(value-lower)<=1e-9*lower) return name+"_low";
        if(next>upper && std::abs(value-upper)<=1e-9*upper) return name+"_high";
        return std::string{};
    };
    double lower=eos.get_c2p_ymu_min(), upper=eos.get_c2p_ymu_max();
    if(!harry_mode) {
        auto const& table=weakhub::get_device_handle();
        lower=std::max(lower,std::exp(table.logymu_min));
        upper=std::min(upper,std::exp(table.logymu_max));
    }
    auto label=face(accepted.ymu,trial.ymu,lower,upper,"Ymu");
    if(!label.empty()) return label;
    label=face(accepted.ye+accepted.ymu,trial.ye+trial.ymu,
               eos.get_c2p_yp_min(),eos.get_c2p_yp_max(),"Yp");
    if(!label.empty()) return label;
    return face(accepted.ye,trial.ye,eos.get_c2p_ye_min(),eos.get_c2p_ye_max(),"Ye");
}
}
TEST_CASE("Lepton chemical potentials use their native EOS grid", "[.][native_lepton_grid]") {
    auto eos4=eos::get().get_eos<leptonic_eos_4d_t>();
    // File-based regression using a temperature-decimated SFHo lepton grid.
    auto baryon=eos4.baryon_table;
    REQUIRE(2*eos4.ele_table._logT.extent(0)-1==baryon._logT.extent(0));
    for(auto const& table:{eos4.ele_table,eos4.muon_table}) {
        int const ir=table._logrho.extent(0)/2, it=table._logT.extent(0)/2, iy=table._ye.extent(0)/2;
        Kokkos::View<double*> result("lepton axis regression",2);
        Kokkos::parallel_for("native lepton midpoint",1,KOKKOS_LAMBDA(int) {
            // Query the omitted baryon temperature row, halfway between
            // two stored lepton rows in log T.
            double r=baryon.lrho(ir),t=baryon.ltemp(2*it+1),y=table.ye(iy);
            result(0)=table.interp(r,t,y,0);
            result(1)=.5*(table(ir,it,iy,0)+table(ir,it+1,iy,0));
        });
        auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
        REQUIRE(std::isfinite(host(0)));
        REQUIRE(std::abs(host(0)-host(1))<1e-11*std::max(1.,std::abs(host(1))));
    }
}
TEST_CASE("Cold released pair kernels and thermal photon EOS", "[.][cold_harry_photons]") {
    auto eos4=eos::get().get_eos<leptonic_eos_4d_t>();
    auto bare=eos4;
    REQUIRE(eos4.photon_pressure_at_1MeV>0);
    REQUIRE(eos4.get_c2p_eps_max()==get_param<double>("eos","eps_maximum"));
    bare.photon_pressure_at_1MeV=0;
    auto const& h=harry::get_harry_device_handle();
    REQUIRE_FALSE(table_inside({1.e14*nu_constants::RHOGF,149.,.1,.01}));
    REQUIRE_FALSE(table_inside({1.e14*nu_constants::RHOGF,20.,.005,.01}));
    REQUIRE_FALSE(table_inside({1.e14*nu_constants::RHOGF,20.,.1,.13}));
    for(double density:{1.e14,6.e14}) {
        state s{density*nu_constants::RHOGF,.2,.1,.01};
        auto F=rate_state(eos4,s,true);
        REQUIRE(std::abs(F.harry_density_factor-939.56542052/931.49410242)<1e-12);
        REQUIRE(F.mu_e/s.T>100);
        material m{s.T,F.mu_e,F.nb,F.Xn,F.Xp};
        for(int species:{0,2,4}) {
            kernel k; bool clamped=false;
            REQUIRE(init_harry_thermal(k,h,guo_nnbrem::get_device_handle(),m,{true,true,false},
                                       species,species==4,&clamped));
            REQUIRE(clamped);
            REQUIRE(std::isfinite(k.a[0][0]));
        }
        decay_kernel decay;
        REQUIRE(init_harry_decay_kernel(decay,h,s.T,F.mu_e,F.mu_mu_reaction,.5));
    }
    for(double T:{5.,20.}) {
        double rho=1e11*nu_constants::RHOGF,ye=.1,ymu=.0005,cs,cs0,eps,eps0,s,s0;
        eos_err_t err{};
        double p=eos4.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(eps,cs,s,T,rho,ye,ymu,err);
        double p0=bare.press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(eps0,cs0,s0,T,rho,ye,ymu,err);
        double pg=7.56573325e-15*std::pow(T*1.160451812e10,4)/3*(CGS_units/GEOM_units).pressure;
        REQUIRE(std::abs((p-p0)/pg-1)<1e-6);
        REQUIRE(std::abs((eps-eps0)*rho/(3*pg)-1)<1e-6);
        REQUIRE(std::abs((s-s0)*rho*T/(4*pg*eos4.photon_mb_MeV)-1)<1e-6);
        REQUIRE((cs>0 && cs<1));
        double recovered=0;
        eos4.press_temp__eps_rho_ye_ymu_impl(recovered,eps,rho,ye,ymu,err);
        CAPTURE(T, recovered, eps, rho, ye, ymu);
        REQUIRE(std::abs(recovered-T)<1e-7);
        std::cout<<"PHOTON_TEST T="<<T<<" P_fraction="<<pg/p<<" cs2="<<cs<<" T_recovered="<<recovered<<"\n";
    }
}
TEST_CASE("Audit released ILD emission at matched matter states", "[.][released_ild_emission]") {
    auto eos4=eos::get().get_eos<leptonic_eos_4d_t>();
    auto const& hh=harry::get_harry_device_handle();
    std::ofstream file(output("ild_emission"));
    file<<"rho,T,Ye,Ymu,mue,mumu,charge,table_N,spectral_N,table_E0,table_E1\n"<<std::setprecision(17);
    for(state initial: {state{1e10,10.,.1,.05},state{1e12,3.,.1,.05},state{1e14,30.,.1,.05}}) {
        auto matter=initial; matter.rho*=nu_constants::RHOGF;
        auto F=rate_state(eos4,matter,true);
        decay_kernel table,analytic;
        REQUIRE(init_harry_decay_kernel(table,hh,F.temp_mev,F.mu_e,F.mu_mu_reaction,.5));
        REQUIRE(init_decay_kernel(analytic,table.g,F.temp_mev,F.mu_e,F.mu_mu_reaction,16,false));
        double empty[max_order]{};
        for(int c=0;c<2;++c) {
            auto rates=evaluate_decay(table,c,empty,empty);
            auto reference=evaluate_decay(analytic,c,empty,empty);
            REQUIRE(std::isfinite(rates.number));
            REQUIRE(std::isfinite(reference.number));
            file<<initial.rho<<','<<initial.T<<','<<initial.ye<<','<<initial.ymu<<','<<F.mu_e<<','
                <<F.mu_mu_reaction<<','<<c<<','<<rates.number<<','<<reference.number<<','
                <<rates.energy[0]<<','<<rates.energy[1]<<'\n';
        }
    }
}
// Homogeneous five-species source experiment: production WeakHub CC, evolved
// thermal pairs, optimized spectral decay, and conservative EOS feedback.
// No spatial transport; adaptive backward-Euler matter refresh, not full IMEX.
TEST_CASE("Harry versus historical rates closed-box evolution", "[.][harry_boxes]") {
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    bool const is_harry=get_harry_table_mode();
    auto const domain_harry=is_harry?harry::get_harry_device_handle():harry::device_handle{};
    std::vector<state> all;
    for(double rho:{1e10,1e12,1e14}) for(double T:{3.,10.,30.})
    for(double ym:{.0005,.0006,.001,.05}) all.push_back({rho*nu_constants::RHOGF,T,.1,ym});
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    int const total=int(all.size())*2*2;std::vector<std::string> lines(total);
    bool const floor_only=std::getenv("GRACE_CC_FLOOR_ONLY")!=nullptr;
    char const* selected_id=std::getenv("GRACE_BOX_CASE_ID");
    std::ofstream out_file(output("boxes"));
    out_file<<"id,rho,T0,Ymu0,radiation,recipe,target_seconds,time_seconds,status,steps,rejections,T,Ye,Ymu,energy_error,Le_error,Lmu_error,residual,wall_seconds,last_rejection_reason,low_eta_clamped_kernel_builds_all_attempts,last_rejection_flags,last_eos_error_mask,last_source_phase,last_source_fit,last_trial_T,last_trial_Ye,last_trial_Ymu,face_species,relative_FD_slack,relative_FD_slack_rate\n";
    out_file.flush();
    #pragma omp parallel for schedule(dynamic)
    for(int task=rank;task<total;task+=ranks) {
        int const recipe=task%2,rad=(task/2)%2,id=task/4;
        if(selected_id && id!=std::stoi(selected_id)) continue;
        if(floor_only && id%5!=0) continue;
        state s=all[id],initial=s;auto row=stream();double mue,mumu;
        int status=potentials(e,s,mue,mumu)?(table_inside(s)?0:7):1,steps=0,rejected=0,last_failure=0;
        unsigned long long low_eta_clamped_kernel_builds=0,last_eos_mask=0;
        int last_source_phase=0,last_source_fit=0;state last_trial=s;
        std::string last_rejection_flags="none";
        std::string previous_outgoing_face;
        int face_species=-1;
        int face_side=0,previous_closure_face=0;
        double face_slack=1.,face_rate=0.;
        bool const regularize=true;
        int const order=16,inner=16;
        double const maxdt=recipe==0?2e-5:5e-6;
        double const change_limit=recipe==0?.02:.01;
        double u[20]{},out[20],ut[10]{},ot[10]{},t=0,h=change_limit*5e-8,eps=internal_energy(e,s),maxres=0;
        // Identical initial radiation moments across solver quadrature orders.
        grid g;g.init(32,std::max(s.T,muon_mass/3));
        if(rad) for(int a=0;a<4;++a) for(int j=0;j<g.n;++j) {
            double const f=occupation(g.e[j]/s.T);
            u[5*a]+=g.w[j]*g.e[j]*f*energy_unit;u[5*a+4]+=g.w[j]*f*number_unit;
        }
        if(rad) for(int a=0;a<2;++a) for(int j=0;j<g.n;++j) {
            double const f=occupation(g.e[j]/s.T);
            ut[5*a]+=g.w[j]*g.e[j]*f*energy_unit;ut[5*a+4]+=g.w[j]*f*number_unit;
        }
        auto energy=[&](double const* a,double ep) { double v=s.rho*ep;for(int j=0;j<4;++j)v+=a[5*j];v+=ut[0]+ut[5];return v; };
        // GRACE evolves baryon-mass-weighted neutrino N: composition exchange
        // is delta(N)/rho, exactly as add_leptonic_backreaction (no Avogadro conversion).
        double const E0=energy(u,eps),Le0=s.ye+(u[4]-u[9])/s.rho,Lm0=s.ymu+(u[14]-u[19])/s.rho;
        auto begin=std::chrono::steady_clock::now();
        auto record=[&](double target) {
            row<<id<<','<<initial.rho/nu_constants::RHOGF<<','<<initial.T<<','<<initial.ymu<<','<<rad<<','<<recipe
               <<','<<target<<','<<t<<','<<status<<','<<steps<<','<<rejected<<','<<s.T<<','<<s.ye<<','<<s.ymu
               <<','<<(energy(u,internal_energy(e,s))-E0)/std::max(std::abs(E0),1e-100)
               <<','<<s.ye+(u[4]-u[9])/s.rho-Le0<<','<<s.ymu+(u[14]-u[19])/s.rho-Lm0
               <<','<<maxres<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<','<<last_failure<<','<<low_eta_clamped_kernel_builds<<','<<last_rejection_flags<<','<<last_eos_mask<<','<<last_source_phase<<','<<last_source_fit
               <<','<<last_trial.T<<','<<last_trial.ye<<','<<last_trial.ymu
               <<','<<face_species<<','<<face_slack<<','<<face_rate<<'\n';
        };
        metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
        m1_prims_array_t base[4]{},pt[2]{};m1_eas_array_t eas[4]{},et[2]{};
        for(double target:{1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}) {
            while(!status && t<target*(1-1e-13)) {
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()>240) {status=6;break;}
                if(steps+rejected>50000) {status=2;break;}
                double eps_rad[NUMSPECIES]{};
                for(int a=0;a<4;++a) if(u[5*a+4]>0)
                    eps_rad[a]=u[5*a]/u[5*a+4]*number_unit/energy_unit;
                if(ut[4]+ut[9]>0) eps_rad[4]=(ut[0]+ut[5])/(ut[4]+ut[9])*number_unit/energy_unit;
                h=std::min({h,maxdt,target-t});
                double res=0,tres=0,enew=eps;
                state next=s,requested=s;
                leptonic_diagnostic source_trace;
                bool four_ok=false;
                bool diagnose_face=false;
                bool const coupled_spectrum=is_harry;
                std::string rejection_flags;
                unsigned long long eos_mask=0;
                // The radiation stage returns a conservative matter target. Iterate
                // the three matter variables at the end of the step for stiff feedback.
                auto evaluate=[&](double const* x,double* residual) {
                    if(diagnose_face) { face_species=-1;face_slack=1.;face_rate=0.;face_side=0;four_ok=false; }
                    state matter{s.rho,x[0]*s.T,x[1]*s.ye,x[2]*s.ymu};
                    if(!std::isfinite(matter.T+matter.ye+matter.ymu) || matter.T<=0 ||
                       !e.composition_in_domain(matter.ye,matter.ymu) || !table_inside(matter) ||
                       !potentials(e,matter,mue,mumu)) return false;
                    auto F=rate_state(e,matter,regularize);
                    double means[NUMSPECIES]{};
                    for(int a=0;a<5;++a) {
                        means[a]=eps_rad[a]*(coupled_spectrum?x[3+a]:1.);
                        if(!std::isfinite(means[a]) || means[a]<0) return false;
                    }
                    auto rates=ordinary(F,means);
                    for(int a=0;a<4;++a) eas[a]=rate_array(rates.out[a]);
                    for(int a=0;a<2;++a) et[a]=rate_array(rates.out[4],2);
                    material mat{F.temp_mev,F.mu_e,F.nb,F.Xn,F.Xp};
                    kernel thermal,kt,electron; kernel const* electron_ptr=nullptr;double scale=std::max(matter.T,muon_mass/3),tscale=matter.T;
                    for(int a=0;a<4;++a) if(u[5*a+4]>0)
                        scale=std::max(scale,std::sqrt(matter.T*u[5*a]/u[5*a+4]*number_unit/energy_unit/3));
                    if(ut[4]>0) tscale=std::max(tscale,std::sqrt(matter.T*ut[0]/ut[4]*number_unit/energy_unit/3));
                    bool valid=false; decay_kernel decay;
                    if(get_harry_table_mode()) {
                        auto const& hh=harry::get_harry_device_handle();
                        auto const& gg=guo_nnbrem::get_device_handle();
                        auto build_thermal=[&](kernel& k,int species,bool symmetric=false) {
                            bool clamped=false;
                            bool const ok=init_harry_thermal(k,hh,gg,mat,{true,true,false},species,symmetric,&clamped);
                            if (clamped && mat.mu_e < mat.T*std::exp(hh.pair0.axes[1](0))) ++low_eta_clamped_kernel_builds;
                            return ok;
                        };
                        // Count clamped kernel constructions including subsequently rejected timesteps.
                        valid=build_thermal(thermal,2) && build_thermal(kt,4,true) && build_thermal(electron,0);
                        electron_ptr=&electron;
                        valid=valid && (get_harry_spectral_decay()
                           ?init_decay_kernel(decay,thermal.g,matter.T,F.mu_e,F.mu_mu_reaction,inner,true)
                           :init_harry_decay_kernel(decay,hh,matter.T,F.mu_e,F.mu_mu_reaction,.5));
                    } else {
                        valid=init_thermal_kernel(thermal,mat,{true,true,false},order,scale,false,nullptr,nullptr,true)
                           && init_thermal_kernel(kt,mat,{true,true,false},order,tscale,true,nullptr,nullptr,true)
                           && init_decay_kernel(decay,thermal.g,matter.T,F.mu_e,F.mu_mu_reaction,inner,true);
                    }
                    if(!valid) return false;
                    four_ok=leptonic_implicit_update(thermal,decay,true,metric,base,eas,u,out,h*time_unit,res,nullptr,false,true,electron_ptr,&source_trace);
                    if(diagnose_face && !four_ok) {
                        // At fixed number the coldest possible spectrum fills
                        // the lowest bins first. Its boundary slope dJ/dn is
                        // the energy of the partially filled bin.
                        double f[4][max_order]{},shape[4][2]{},dn[4],dJ[4];
                        auto const& g=thermal.g;
                        bool fitted=true;
                        for(int a=0;a<4;++a) {
                            dn[a]=(eas[a][ETANL]-eas[a][KANL]*u[5*a+4])*time_unit/number_unit;
                            dJ[a]=(eas[a][ETAL]-eas[a][KAL]*u[5*a])*time_unit/energy_unit;
                            fitted &= reconstruct(g,u[5*a+4]/number_unit,u[5*a]/energy_unit,f[a],nullptr,shape[a]);
                        }
                        if(fitted) {
                            auto add=[&](sources p,int a,int b){
                                dn[a]+=p.number;dn[b]+=p.number;
                                dJ[a]+=p.energy[0];dJ[b]+=p.energy[1];
                            };
                            add(grace::pairs::evaluate(thermal,f[2],f[3]),2,3);
                            if(electron_ptr) add(grace::pairs::evaluate(electron,f[0],f[1]),0,1);
                            add(evaluate_decay(decay,0,f[2],f[1],shape[2],shape[1]),2,1);
                            add(evaluate_decay(decay,1,f[3],f[0],shape[3],shape[0]),3,0);
                            face_slack=1.;face_species=-1;
                            for(int a=0;a<4;++a) {
                                double n=u[5*a+4]/number_unit,J=u[5*a]/energy_unit;
                                if(!(J>0 && n>0)) continue;
                                double Jmin=0,pivot=g.e[0],wsum=0,esum=0;
                                for(int i=0;i<g.n;++i) {
                                    wsum+=g.w[i]; esum+=g.w[i]*g.e[i];
                                    if(n<=0) continue;
                                    double fill=std::min(n,g.w[i]); Jmin+=fill*g.e[i];n-=fill;pivot=g.e[i];
                                }
                                double low=(J-Jmin)/J;
                                double high=(esum/wsum*u[5*a+4]/number_unit-J)/J;
                                if(std::min(low,high)<face_slack) {
                                    face_species=a;face_slack=std::min(low,high);
                                    face_side=low<high?-1:1;
                                    face_rate=low<high?(dJ[a]-pivot*dn[a])/J:(esum/wsum*dn[a]-dJ[a])/J;
                                }
                            }
                        }
                    }
                    bool inner_ok=four_ok;
                    inner_ok=inner_ok && implicit_update(kt,metric,pt,et,ut,ot,h*time_unit,tres);
                    res=std::max(res,tres);
                    if(!inner_ok) return false;
                    next=s;enew=eps;
                    next.ye-=((out[4]-u[4])-(out[9]-u[9]))/s.rho;
                    next.ymu-=((out[14]-u[14])-(out[19]-u[19]))/s.rho;
                    for(int a=0;a<4;++a) enew-=(out[5*a]-u[5*a])/s.rho;
                    enew-=((ot[0]-ut[0])+(ot[5]-ut[5]))/s.rho;
                    double const escale=std::max(std::abs(eps),1e-10);
                    residual[0]=(internal_energy(e,matter)-enew)/escale;
                    residual[1]=(matter.ye-next.ye)/s.ye;
                    residual[2]=(matter.ymu-next.ymu)/s.ymu;
                    if(coupled_spectrum) {
                        for(int a=0;a<5;++a) {
                            double energy=a<4?out[5*a]:ot[0]+ot[5];
                            double number=a<4?out[5*a+4]:ot[4]+ot[9];
                            // An initially empty species uses the existing
                            // thermal fallback for this first populated step.
                            residual[3+a]=eps_rad[a]>0 && number>0
                                ? x[3+a]-energy/number*(number_unit/energy_unit)/eps_rad[a]
                                : x[3+a]-1.;
                            if(!std::isfinite(residual[3+a])) return false;
                        }
                    }
                    return std::isfinite(residual[0]+residual[1]+residual[2]);
                };
                double r[8]{},outer_residual=0;
                bool ok;
                if(coupled_spectrum) {
                    double x[8]={1,1,1,1,1,1,1,1};
                    ok=newton<8>(evaluate,x,outer_residual,nullptr,12) && evaluate(x,r);
                } else {
                    double x[3]={1,1,1};
                    ok=newton<3>(evaluate,x,outer_residual,nullptr,12) && evaluate(x,r);
                }
                last_failure=ok?0:1;
                if(!ok) {
                    // Diagnose the direction of exchange at the accepted state,
                    // without classifying an out-of-domain Newton probe as a table exit.
                    double const at_start[8]={1,1,1,1,1,1,1,1};
                    diagnose_face=true;
                    if(evaluate(at_start,r)) {
                        requested=next;
                        rejection_flags=ablation_bounds(e,requested,is_harry,domain_harry);
                        if(rejection_flags.empty()) rejection_flags="matter_iteration|";
                    } else {
                        rejection_flags=four_ok?"tau_or_material_evaluation|":"four_or_material_evaluation|";
                    }
                }
                if(ok) {
                    auto check=next;requested=next;double checkeps=enew;eos_err_t err{};
                    e.press_temp__eps_rho_ye_ymu(next.T,checkeps,next.rho,next.ye,next.ymu,err);
                    bool const eos_ok=table_inside(next) && std::isfinite(next.T+next.ye+next.ymu+checkeps)
                       && next.rho==check.rho && next.ye==check.ye && next.ymu==check.ymu
                       && std::abs(internal_energy(e,next)-enew)<1e-8*std::max(std::abs(enew),1e-10);
                    bool const change_ok=std::abs(next.T-s.T)<change_limit*s.T
                       && std::abs(next.ye-s.ye)<change_limit*std::max(s.ye,1e-6)
                       && std::abs(next.ymu-s.ymu)<change_limit*std::max(s.ymu,1e-6);
                    ok=eos_ok && change_ok;last_failure=!eos_ok?2:!change_ok?3:0;
                    for(int bit=0;bit<EOS_NUM_ERRORS;++bit)if(err.test(bit))eos_mask|=1ULL<<bit;
                    rejection_flags=ablation_bounds(e,requested,is_harry,domain_harry)+ablation_bounds(e,next,is_harry,domain_harry);
                    if(next.rho!=check.rho)rejection_flags+="EOS_changed_rho|";
                    if(next.ye!=check.ye)rejection_flags+="EOS_changed_Ye|";
                    if(next.ymu!=check.ymu)rejection_flags+="EOS_changed_Ymu|";
                    if(!(std::abs(internal_energy(e,next)-enew)<1e-8*std::max(std::abs(enew),1e-10)))rejection_flags+="eps_inverse_mismatch|";
                    if(!(std::abs(next.T-s.T)<change_limit*s.T))rejection_flags+="change_T|";
                    if(!(std::abs(next.ye-s.ye)<change_limit*std::max(s.ye,1e-6)))rejection_flags+="change_Ye|";
                    if(!(std::abs(next.ymu-s.ymu)<change_limit*std::max(s.ymu,1e-6)))rejection_flags+="change_Ymu|";
                }
                if(!ok) {
                    last_rejection_flags=rejection_flags;last_eos_mask=eos_mask;last_source_phase=source_trace.phase;last_source_fit=int(source_trace.reconstruction.code);
                    last_trial=next;last_trial.ye=requested.ye;last_trial.ymu=requested.ymu;
                    ++rejected;
                    auto const face=outgoing_composition_face(e,s,requested,is_harry);
                    if(!face.empty() && previous_outgoing_face==face) {
                        last_rejection_flags="table_exit_"+face;
                        status=8;
                        break;
                    }
                    // A physical-time increment cannot follow an outward
                    // derivative past the finite-grid FD domain. Stop only
                    // after two failed, successively halved source steps at
                    // the unchanged matter/radiation state. The remaining
                    // margin must be below the nonlinear acceptance scale
                    // and would be exhausted within the smaller trial step.
                    int const closure_face=face_side*(face_species+1);
                    if(diagnose_face && !four_ok && closure_face!=0 && face_rate<0 &&
                       face_slack<newton_accept && face_slack+h*face_rate<0) {
                        if(previous_closure_face==closure_face) {
                            last_rejection_flags=face_side<0?"FD_min_energy|":"FD_max_positiveT_energy|";
                            status=9;
                            break;
                        }
                        previous_closure_face=closure_face;
                    } else previous_closure_face=0;
                    previous_outgoing_face=face;
                    h*=.5;if(h<1e-16)status=5;continue;
                }
                previous_outgoing_face.clear();
                previous_closure_face=0;
                state const previous=s;maxres=std::max(maxres,res);s=next;eps=enew;
                std::copy(out,out+20,u);std::copy(ot,ot+10,ut);t+=h;++steps;
                double const change=std::max({std::abs(next.T/previous.T-1),std::abs(next.ye/previous.ye-1),std::abs(next.ymu/previous.ymu-1)});
                h*=std::clamp(.8*change_limit/std::max(change,1e-12),.5,1.6);
            }
            record(target);
        }
        lines[task]=row.str();
        #pragma omp critical(grey_box_output)
        { out_file<<lines[task];out_file.flush(); }
    }
    REQUIRE(out_file.good());
}


namespace {
// Diagnostic-only: evaluate the existing analytic backend on an explicitly
// supplied grid. No table electron-pair surrogate is invented here.
bool old_thermal_on_grid(kernel& out,material const& m,grid const& g,bool symmetric=false) {
    out.g=g;
    for(int i=0;i<g.n;++i) {
        out.boltz[i]=std::exp(-g.e[i]/m.T);
        for(int j=0;j<g.n;++j) {
            double r=absorption_kernel(m,{true,true,false},g.e[i],g.e[j]);
            if(symmetric) r=.5*(r+absorption_kernel(m,{true,true,false},g.e[j],g.e[i]));
            if(!(r>=0) || !std::isfinite(r)) return false;
            out.a[i][j]=r;
        }
    }
    return true;
}
std::string ablation_bounds(leptonic_eos_4d_t const& eos,state s,
        bool new_rates,harry::device_handle const& h) {
    std::string flags;
    auto bound=[&](double x,double lo,double hi,std::string const& name) {
        if(!std::isfinite(x)) flags+=name+"_nonfinite|";
        else if(x<lo) flags+=name+"_low|";
        else if(x>hi) flags+=name+"_high|";
    };
    bound(s.rho,eos.density_minimum(),eos.density_maximum(),"EOS_rho");
    bound(s.T,eos.temperature_floor(),eos.temperature_maximum(),"EOS_T");
    bound(s.ye,eos.get_c2p_ye_min(),eos.get_c2p_ye_max(),"EOS_Ye");
    bound(s.ymu,eos.get_c2p_ymu_min(),eos.get_c2p_ymu_max(),"EOS_Ymu");
    bound(s.ye+s.ymu,eos.get_c2p_yp_min(),eos.get_c2p_yp_max(),"EOS_Yp");
    if(new_rates) {
        for(int species:{0,2}) {
            auto const& t=h.cc[species]; std::string const p=species==0?"CCe_":"CCmu_";
            bound(std::log(s.rho*harry_density_ratio(eos.get_baryon_mass())),t.axes[0](0),t.axes[0](t.size[0]-1),p+"lnrho");
            bound(std::log(s.T),t.axes[1](0),t.axes[1](t.size[1]-1),p+"lnT");
            bound(s.ye,t.axes[2](0),t.axes[2](t.size[2]-1),p+"Ye");
            bound(std::log(s.ymu),t.axes[3](0),t.axes[3](t.size[3]-1),p+"lnYmu");
        }
    } else {
        auto const& t=weakhub::get_device_handle();
        bound(std::log(s.rho),t.logrho_min,t.logrho_max,"oldCC_lnrho");
        bound(std::log(s.T),t.logtemp_min,t.logtemp_max,"oldCC_lnT");
        bound(s.ye,t.ye_min,t.ye_max,"oldCC_Ye");
        bound(std::log(s.ymu),t.logymu_min,t.logymu_max,"oldCC_lnYmu");
    }
    return flags;
}
}

TEST_CASE("Diagnostic analytic kernel agrees on an identical grid", "[harry_fixed_grid_formula]") {
    for(material m: {material{3,20,6e37,.8,.2},material{30,1,6e33,.9,.1}}) {
        for(bool symmetric: {false,true}) {
            kernel reference,got;
            REQUIRE(reference.init(m,{true,true,false},16,35,symmetric));
            REQUIRE(old_thermal_on_grid(got,m,reference.g,symmetric));
            for(int i=0;i<16;++i) {
                REQUIRE(got.boltz[i]==reference.boltz[i]);
                for(int j=0;j<16;++j)
                    REQUIRE(std::abs(got.a[i][j]-reference.a[i][j])<=1e-12*std::max(reference.a[i][j],1e-300));
            }
        }
    }
}

namespace {
void capture_leptonic_cell(std::string const& path,material const& m,double mumu,double h,
        kernel const& thermal,decay_kernel const& decay,double const (&old)[20],
        m1_eas_array_t const (&eas)[4],leptonic_diagnostic const& trace) {
    std::ofstream f(path);f<<std::setprecision(17);
    f<<1<<'\n'<<m.T<<' '<<m.mu_e<<' '<<m.nb<<' '<<m.yn<<' '<<m.yp<<' '<<mumu<<' '<<h<<'\n';
    f<<thermal.g.n<<' '<<thermal.g.scale<<'\n';
    for(int i=0;i<thermal.g.n;++i)f<<thermal.g.e[i]<<' '<<thermal.g.w[i]<<' '<<thermal.boltz[i]<<'\n';
    for(double v:old)f<<v<<' ';f<<'\n';
    for(int a=0;a<4;++a){for(int v:{KAL,KSL,KANL,ETAL,ETANL})f<<eas[a][v]<<' ';f<<'\n';}
    for(int i=0;i<thermal.g.n;++i){for(int j=0;j<thermal.g.n;++j)f<<thermal.a[i][j]<<' ';f<<'\n';}
    for(int c=0;c<2;++c)for(int i=0;i<thermal.g.n;++i){for(int j=0;j<thermal.g.n;++j)f<<decay.total[c][i][j]<<' ';f<<'\n';}
    f<<trace.phase<<' '<<trace.species<<' '<<trace.continuation<<' '<<trace.nonlinear.stage<<' '
     <<trace.nonlinear.iteration<<' '<<trace.nonlinear.column<<' '<<int(trace.reconstruction.code)<<' '
     <<trace.reconstruction.n<<' '<<trace.reconstruction.J<<' '<<trace.reconstruction.residual<<'\n';
    if(!f.good())throw std::runtime_error("Cannot write source reproducer: "+path);
}
}

TEST_CASE("Replay captured cold leptonic source failure", "[.][leptonic_replay]") {
    grace::test::replay_captured_leptonic_source();
}
// Ablations: A old adaptive; B old fixed-grid; C Harry CC + old fixed-grid;
// D Harry CC + Harry heavy pairs/Guo; E full mixed provider with electron pairs.
// A32 repeats A with outer order32; every mode retains inner decay order16.
// B/C intentionally use direct analytic matrix assembly for isolation; their
// walltimes are diagnostic harness costs, not optimized production timings.
// The historical YAML must remain selected so the old opacity handle coexists
// with the explicit extra tables loaded below. All modes retain the same EOS,
// dilute-muon target, conservative spectral decay and inner order 16.
TEST_CASE("Focused Harry rate ablations", "[.][harry_ablations]") {
    REQUIRE_FALSE(get_harry_table_mode());
    auto hp=std::getenv("GRACE_HARRY_ABLATION_TABLE");
    auto gp=std::getenv("GRACE_HARRY_ABLATION_GUO");
    REQUIRE(hp!=nullptr);REQUIRE(gp!=nullptr);
    harry::initialize_harry_table(hp,false);
    auto const hh=harry::get_harry_device_handle();
    auto const gg=guo_nnbrem::load(gp);
    auto eos=grace::eos::get().get_eos<leptonic_eos_4d_t>();
    grid fixed;REQUIRE(harry_grid(hh,fixed));
    std::array<state,3> cases{{{1e10*nu_constants::RHOGF,30,.1,.05},
                             {1e10*nu_constants::RHOGF,30,.1,.0005},
                             {1e14*nu_constants::RHOGF,3,.1,.05}}};
    std::ofstream file(output("ablations"));
    file<<"case,mode,recipe,rho,T0,Ymu0,radiation,target_seconds,time_seconds,status,steps,rejections,T,Ye,Ymu,energy_error,Le_error,Lmu_error,max_residual,wall_seconds,low_eta_clamped_kernel_builds_all_attempts,last_failed_h,last_failed_time,last_trial_T,last_trial_Ye,last_trial_Ymu,last_eps_target,last_eps_inverted,last_eps_reconstructed,last_source_residual,last_eos_error_mask,last_rejection_flags\n";
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    #pragma omp parallel for schedule(dynamic)
    for(int task=rank;task<36;task+=ranks) {
        int const recipe=task%2,mode=(task/2)%6,id=task/12,rad=id==1?1:0;
        bool const new_rates=mode>=2 && mode<=4;
        char const* names[]={"A","B","C","D","E","A32"};
        if(auto c=std::getenv("GRACE_HARRY_ABLATION_CASE"))if(id!=std::atoi(c))continue;
        if(auto c=std::getenv("GRACE_HARRY_ABLATION_MODE"))if(std::string(c)!=names[mode])continue;
        bool captured=false;
        state s=cases[id],initial=s;auto row=stream();
        double mue,mumu;
        int status=potentials(eos,s,mue,mumu)?0:1,steps=0,rejected=0;
        if(!status && !ablation_bounds(eos,s,new_rates,hh).empty())status=7;
        double const maxdt=recipe==0?1e-6:5e-7,change=recipe==0?.02:.01;
        double u[20]{},out[20]{},ut[10]{},ot[10]{},t=0,h=change*5e-8;
        double eps=internal_energy(eos,s),maxres=0;
        unsigned long long clamped_builds=0;
        grid initial_grid;initial_grid.init(32,std::max(s.T,muon_mass/3));
        if(rad) {
            for(int a=0;a<4;++a) for(int j=0;j<initial_grid.n;++j) {
                double const f=occupation(initial_grid.e[j]/s.T);
                u[5*a]+=initial_grid.w[j]*initial_grid.e[j]*f*energy_unit;
                u[5*a+4]+=initial_grid.w[j]*f*number_unit;
            }
            for(int a=0;a<2;++a) for(int j=0;j<initial_grid.n;++j) {
                double const f=occupation(initial_grid.e[j]/s.T);
                ut[5*a]+=initial_grid.w[j]*initial_grid.e[j]*f*energy_unit;
                ut[5*a+4]+=initial_grid.w[j]*f*number_unit;
            }
        }
        auto energy=[&](double ep) {double v=s.rho*ep+ut[0]+ut[5];for(int a=0;a<4;++a)v+=u[5*a];return v;};
        double const E0=energy(eps),Le0=s.ye+(u[4]-u[9])/s.rho,Lmu0=s.ymu+(u[14]-u[19])/s.rho;
        auto start=std::chrono::steady_clock::now();
        metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
        m1_prims_array_t base[4]{},pt[2]{};m1_eas_array_t eas[4]{},et[2]{};
        double last_h=0,last_t=0,last_eps=0,last_inv=0,last_reconstructed=0,last_res=0;
        state last_trial=s;unsigned long long last_mask=0;std::string last_reason="none";
        for(double target:{1e-8,1e-7,1e-6}) {
            while(!status && t<target*(1-1e-13)) {
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>90){status=6;break;}
                if(steps+rejected>50000){status=2;break;}
                if(!potentials(eos,s,mue,mumu)){status=3;break;}
                auto F=rate_state(eos,s,true);double means[NUMSPECIES]{};
                for(int a=0;a<4;++a) if(u[5*a+4]>0)means[a]=u[5*a]/u[5*a+4]*number_unit/energy_unit;
                if(ut[4]+ut[9]>0)means[4]=(ut[0]+ut[5])/(ut[4]+ut[9])*number_unit/energy_unit;
                auto rates=new_rates?compute_all_species_harry(hh,F,means):ordinary(F,means);
                for(int a=0;a<4;++a)eas[a]=rate_array(rates.out[a]);
                for(int a=0;a<2;++a)et[a]=rate_array(rates.out[4],2);
                material const mat{F.temp_mev,F.mu_e,F.nb,F.Xn,F.Xp};
                kernel thermal,tau,electron;kernel const* ep=nullptr;decay_kernel decay;
                bool valid=false;
                if(mode==0 || mode==5) {
                    double scale=std::max(s.T,muon_mass/3),tscale=s.T;
                    for(int a=0;a<4;++a)if(u[5*a+4]>0)scale=std::max(scale,std::sqrt(s.T*means[a]/3));
                    if(ut[4]>0)tscale=std::max(tscale,std::sqrt(s.T*ut[0]/ut[4]*number_unit/energy_unit/3));
                    valid=init_thermal_kernel(thermal,mat,{true,true,false},mode==5?32:16,scale,false,nullptr,nullptr,true)
                       && init_thermal_kernel(tau,mat,{true,true,false},mode==5?32:16,tscale,true,nullptr,nullptr,true);
                } else if(mode<=2) {
                    valid=old_thermal_on_grid(thermal,mat,fixed)&&old_thermal_on_grid(tau,mat,fixed,true);
                } else {
                    auto build=[&](kernel& k,int spec,bool symmetric=false) {
                        bool clamped=false;bool const ok=init_harry_thermal(k,hh,gg,mat,{true,true,false},spec,symmetric,&clamped);
                        if (clamped && mat.mu_e < mat.T*std::exp(hh.pair0.axes[1](0))) ++clamped_builds;
                        return ok;
                    };
                    valid=build(thermal,2)&&build(tau,4,true);
                    if(mode==4){valid=valid&&build(electron,0);ep=&electron;}
                }
                valid=valid&&init_decay_kernel(decay,thermal.g,s.T,F.mu_e,F.mu_mu_reaction,16,true);
                if(!valid){status=4;last_reason="kernel_init";break;}
                h=std::min({h,maxdt,target-t});double res=0,tres=0;
                leptonic_diagnostic source_trace;
                bool const four_ok=leptonic_implicit_update(thermal,decay,true,metric,base,eas,u,out,h*time_unit,res,nullptr,false,true,ep,&source_trace);
                if(!four_ok && !captured && h<1e-14)if(auto folder=std::getenv("GRACE_HARRY_CAPTURE_FAILURE")) {
                    capture_leptonic_cell(std::string(folder)+"/source_case"+std::to_string(id)+"_"+names[mode]+"_recipe"+std::to_string(recipe)+".txt",mat,F.mu_mu_reaction,h*time_unit,thermal,decay,u,eas,source_trace);
                    captured=true;
                }
                bool const tau_ok=four_ok&&implicit_update(tau,metric,pt,et,ut,ot,h*time_unit,tres);
                bool ok=four_ok&&tau_ok;res=std::max(res,tres);
                state next=s,requested=s;double enew=eps,eps_inv=eps,eps_check=eps;unsigned long long mask=0;
                std::string reason=ok?"":four_ok?"source_tau|":"source_four|";
                if(!four_ok) reason+="phase="+std::to_string(source_trace.phase)+"|species="+std::to_string(source_trace.species)+"|newton="+std::to_string(source_trace.nonlinear.stage)+"|fit="+std::to_string(int(source_trace.reconstruction.code))+"|";
                if(ok) {
                    next.ye-=((out[4]-u[4])-(out[9]-u[9]))/s.rho;
                    next.ymu-=((out[14]-u[14])-(out[19]-u[19]))/s.rho;
                    for(int a=0;a<4;++a)enew-=(out[5*a]-u[5*a])/s.rho;
                    enew-=((ot[0]-ut[0])+(ot[5]-ut[5]))/s.rho;
                    auto before=next;requested=next;eps_inv=enew;eos_err_t err{};
                    eos.press_temp__eps_rho_ye_ymu(next.T,eps_inv,next.rho,next.ye,next.ymu,err);
                    for(int bit=0;bit<EOS_NUM_ERRORS;++bit)if(err.test(bit))mask|=1ULL<<bit;
                    reason=ablation_bounds(eos,before,new_rates,hh);
                    reason+=ablation_bounds(eos,next,new_rates,hh);
                    if(next.rho!=before.rho)reason+="EOS_changed_rho|";
                    if(next.ye!=before.ye)reason+="EOS_changed_Ye|";
                    if(next.ymu!=before.ymu)reason+="EOS_changed_Ymu|";
                    eps_check=internal_energy(eos,next);
                    if(!std::isfinite(next.T+next.ye+next.ymu+eps_inv))reason+="inverse_nonfinite|";
                    if(!(std::abs(eps_check-enew)<1e-8*std::max(std::abs(enew),1e-10)))reason+="eps_inverse_mismatch|";
                    if(!(std::abs(next.T-s.T)<change*s.T))reason+="change_T|";
                    if(!(std::abs(next.ye-s.ye)<change*std::max(s.ye,1e-6)))reason+="change_Ye|";
                    if(!(std::abs(next.ymu-s.ymu)<change*std::max(s.ymu,1e-6)))reason+="change_Ymu|";
                    ok=reason.empty();
                }
                if(!ok) {
                    ++rejected;last_h=h;last_t=t;last_trial=next;
                    // Requested composition survives EOS clamping; T is the inverse result.
                    last_trial.ye=requested.ye;last_trial.ymu=requested.ymu;last_eps=enew;last_inv=eps_inv;last_reconstructed=eps_check;
                    last_res=res;last_mask=mask;last_reason=reason;
                    h*=.5;if(h<1e-16)status=5;continue;
                }
                maxres=std::max(maxres,res);s=next;eps=enew;
                std::copy(out,out+20,u);std::copy(ot,ot+10,ut);t+=h;++steps;h*=1.4;
            }
            row<<id<<','<<names[mode]<<','<<recipe<<','<<initial.rho/nu_constants::RHOGF<<','<<initial.T<<','<<initial.ymu<<','<<rad
               <<','<<target<<','<<t<<','<<status<<','<<steps<<','<<rejected<<','<<s.T<<','<<s.ye<<','<<s.ymu
               <<','<<(energy(internal_energy(eos,s))-E0)/std::max(std::abs(E0),1e-100)
               <<','<<s.ye+(u[4]-u[9])/s.rho-Le0<<','<<s.ymu+(u[14]-u[19])/s.rho-Lmu0
               <<','<<maxres<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<','<<clamped_builds
               <<','<<last_h<<','<<last_t<<','<<last_trial.T<<','<<last_trial.ye<<','<<last_trial.ymu<<','<<last_eps<<','<<last_inv
               <<','<<last_reconstructed<<','<<last_res<<','<<last_mask<<','<<last_reason<<'\n';
        }
        #pragma omp critical(harry_ablation_output)
        {file<<row.str();file.flush();}
    }
    REQUIRE(file.good());
}
