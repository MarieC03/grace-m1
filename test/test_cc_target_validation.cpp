// Explicit validation experiments, using the production EOS and source solver.
// All scientific failures are recorded; no rates or populations are clipped.
#include <catch2/catch_test_macros.hpp>
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
namespace {
struct state { double rho,T,ye,ymu; };
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
std::vector<state> states() {
    std::vector<state> v;
    for(double rho:{1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15})
    for(double T:{.5,1.,3.,5.,10.,20.,40.,80.})
    for(double ym:{.0005,.00055,.0006,.000625,.00065,.0008,.001,.01,.05,.1})
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
nu_rates_all_out ordinary(fugacity_state const& F) {
    double xyz[3]{},eps[NUMSPECIES]{};
    return compute_all_species_weakhub(weakhub::get_device_handle(),F,false,false,false,
        xyz,tau_policy_none{},false,eps,false,false,false,0);
}
m1_eas_array_t rate_array(nu_rates_out const& r,double multiplicity=1) {
    m1_eas_array_t a{};a[KAL]=r.kappa_a;a[KSL]=r.kappa_s;a[KANL]=r.kappa_n;
    a[ETAL]=r.eta_E/multiplicity;a[ETANL]=r.eta_N/multiplicity;return a;
}
bool table_inside(state s) {
    auto const& h=weakhub::get_device_handle();
    return std::log(s.rho)>=h.logrho_min && std::log(s.rho)<=h.logrho_max
        && std::log(s.T)>=h.logtemp_min && std::log(s.T)<=h.logtemp_max
        && s.ye>=h.ye_min && s.ye<=h.ye_max
        && std::log(s.ymu)>=h.logymu_min && std::log(s.ymu)<=h.logymu_max;
}

}
// Homogeneous five-species source experiment: production WeakHub CC, evolved
// thermal pairs, optimized spectral decay, and conservative EOS feedback.
// No spatial transport; adaptive backward-Euler matter refresh, not full IMEX.
TEST_CASE("Dilute CC target coupled closed-box evolution", "[.][cc_boxes]") {
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    std::vector<state> all;
    for(double rho:{1e10,1e12,1e14,1e15}) for(double T:{1.,10.,50.})
    for(double ym:{.0005,.0006,.00065,.001,.05}) all.push_back({rho*nu_constants::RHOGF,T,.1,ym});
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    int const total=int(all.size())*2*6;std::vector<std::string> lines(total);
    std::ofstream out_file(output("boxes"));
    out_file<<"id,rho,T0,Ymu0,radiation,recipe,target_seconds,time_seconds,status,steps,rejections,T,Ye,Ymu,energy_error,Le_error,Lmu_error,residual,wall_seconds,last_rejection_reason\n";
    out_file.flush();
    #pragma omp parallel for schedule(dynamic)
    for(int task=rank;task<total;task+=ranks) {
        int const recipe=task%6,rad=(task/6)%2,id=task/12;
        state s=all[id],initial=s;auto row=stream();double mue,mumu;
        int status=potentials(e,s,mue,mumu)?(table_inside(s)?0:7):1,steps=0,rejected=0,last_failure=0;
        bool const regularize=recipe%2;
        int const order=recipe>=4?32:16,inner=recipe>=4?32:16;
        double const maxdt=(recipe<2)?1e-6:5e-7;
        double const change_limit=(recipe<2)?.02:.01;
        double const nb=s.rho/nu_constants::RHOGF*nu_constants::avogadro;
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
               <<','<<maxres<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<','<<last_failure<<'\n';
        };
        metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
        m1_prims_array_t base[4]{},pt[2]{};m1_eas_array_t eas[4]{},et[2]{};
        for(double target:{1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}) {
            while(!status && t<target*(1-1e-13)) {
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()>90) {status=6;break;}
                if(steps+rejected>50000) {status=2;break;}
                if(!potentials(e,s,mue,mumu)) {status=3;break;}
                auto F=rate_state(e,s,regularize);
                auto rates=ordinary(F);
                for(int a=0;a<4;++a) eas[a]=rate_array(rates.out[a]);
                for(int a=0;a<2;++a) et[a]=rate_array(rates.out[4],2);
                material mat{F.temp_mev,F.mu_e,F.nb,F.Xn,F.Xp};
                kernel thermal,kt;double scale=std::max(s.T,muon_mass/3),tscale=s.T;
                for(int a=0;a<4;++a) if(u[5*a+4]>0)
                    scale=std::max(scale,std::sqrt(s.T*u[5*a]/u[5*a+4]*number_unit/energy_unit/3));
                if(ut[4]>0) tscale=std::max(tscale,std::sqrt(s.T*ut[0]/ut[4]*number_unit/energy_unit/3));
                bool const valid=init_thermal_kernel(thermal,mat,{true,true,false},order,scale,false,nullptr,nullptr,true)
                    && init_thermal_kernel(kt,mat,{true,true,false},order,tscale,true,nullptr,nullptr,true);
                decay_kernel decay;
                if(!valid || !init_decay_kernel(decay,thermal.g,s.T,F.mu_e,F.mu_mu_reaction,inner,true)) {status=4;break;}
                h=std::min({h,maxdt,target-t});double res=0,tres=0;
                bool ok=leptonic_implicit_update(thermal,decay,true,metric,base,eas,u,out,h*time_unit,res,nullptr,false,true);
                ok=ok && implicit_update(kt,metric,pt,et,ut,ot,h*time_unit,tres);
                res=std::max(res,tres);
                last_failure=ok?0:1;
                state next=s;double enew=eps;
                if(ok) {
                    next.ye-=((out[4]-u[4])-(out[9]-u[9]))/s.rho;
                    next.ymu-=((out[14]-u[14])-(out[19]-u[19]))/s.rho;
                    for(int a=0;a<4;++a) enew-=(out[5*a]-u[5*a])/s.rho;
                    enew-=((ot[0]-ut[0])+(ot[5]-ut[5]))/s.rho;
                    auto check=next;double checkeps=enew;eos_err_t err{};
                    e.press_temp__eps_rho_ye_ymu(next.T,checkeps,next.rho,next.ye,next.ymu,err);
                    bool const eos_ok=table_inside(next) && std::isfinite(next.T+next.ye+next.ymu+checkeps)
                       && next.rho==check.rho && next.ye==check.ye && next.ymu==check.ymu
                       && std::abs(internal_energy(e,next)-enew)<1e-8*std::max(std::abs(enew),1e-10);
                    bool const change_ok=std::abs(next.T-s.T)<change_limit*s.T
                       && std::abs(next.ye-s.ye)<change_limit*std::max(s.ye,1e-6)
                       && std::abs(next.ymu-s.ymu)<change_limit*std::max(s.ymu,1e-6);
                    ok=eos_ok && change_ok;last_failure=!eos_ok?2:!change_ok?3:0;
                }
                if(!ok) {++rejected;h*=.5;if(h<1e-16)status=5;continue;}
                maxres=std::max(maxres,res);s=next;eps=enew;
                std::copy(out,out+20,u);std::copy(ot,ot+10,ut);t+=h;++steps;h*=1.4;
            }
            record(target);
        }
        lines[task]=row.str();
        #pragma omp critical(grey_box_output)
        { out_file<<lines[task];out_file.flush(); }
    }
    REQUIRE(out_file.good());
}

TEST_CASE("Dilute CC target EOS-domain rate audit", "[.][cc_rates]") {
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    REQUIRE(weakhub::get_device_handle().valid);
    auto all=states();std::vector<std::string> lines(all.size());
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    #pragma omp parallel for schedule(dynamic)
    for(int id=rank;id<int(all.size());id+=ranks) {
        auto s=all[id];double mue=0,mumu=0;bool valid=potentials(e,s,mue,mumu),inside=table_inside(s);
        auto row=stream();
        for(int mode=0;mode<2;++mode) {
            auto F=rate_state(e,s,mode);auto rates=ordinary(F);
            for(int sp=0;sp<5;++sp) {
                auto r=rates.out[sp];
                row<<id<<','<<s.rho/nu_constants::RHOGF<<','<<s.T<<','<<s.ye<<','<<s.ymu<<','<<valid<<','<<inside
                   <<','<<mode<<','<<sp<<','<<F.mu_e<<','<<F.mu_mu<<','<<F.mu_mu_reaction<<','<<F.eta_cc_mu
                   <<','<<r.eta_E<<','<<r.eta_N<<','<<r.kappa_a<<','<<r.kappa_n<<','<<r.kappa_s<<'\n';
            }
        }
        lines[id]=row.str();
    }
    std::ofstream out(output("rates"));
    out<<"id,rho,T,Ye,Ymu,eos_valid,table_inside,mode,species,mue,mumu,reaction_mu,eta_cc,eta_E,eta_N,kappa_a,kappa_n,kappa_s\n";
    for(auto const& s:lines) out<<s;
    REQUIRE(out.good());
}
