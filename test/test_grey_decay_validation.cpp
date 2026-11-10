// Explicit validation experiments, using the production EOS and source solver.
// All scientific failures are recorded; no rates or populations are clipped.
#include <catch2/catch_test_macros.hpp>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/neutrino_leptonic_update.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>
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
    char const* path=std::getenv("GRACE_GREY_VALIDATION_OUTPUT");
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
    for(double ym:{1e-6,.0006,.001,.01,.05,.1})
    for(double ye:{.05,.15,.3}) v.push_back({rho*nu_constants::RHOGF,T,ye,ym});
    return v;
}
}
TEST_CASE("Grey EOS-domain rates and spectral convergence", "[.][grey_domain]") {
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    auto all=states();std::vector<std::string> lines(all.size());
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    #pragma omp parallel for schedule(dynamic)
    for(int id=rank;id<int(all.size());id+=ranks) {
        auto s=all[id];auto row=stream();double mue=0,mumu=0;
        bool const valid=potentials(e,s,mue,mumu);
        auto prefix=[&](int recipe,int radiation,int charge) {
            row<<id<<','<<s.rho/nu_constants::RHOGF<<','<<s.T<<','<<s.ye<<','<<s.ymu
                <<','<<mue<<','<<mumu<<','<<valid<<','<<recipe<<','<<radiation<<','<<charge;
        };
        if(!valid) { prefix(-1,-1,-1);row<<",0,0,0,0,0,0,0,0,0\n";lines[id]=row.str();continue; }
        // Same FD radiation distributions for all methods; no fit/grid ambiguity.
        for(int recipe=0;recipe<5;++recipe) {
            int const order=recipe<2?16:32,inner=recipe<2?16:recipe==2?32:64;
            grid g;g.init(recipe==4?16:order,std::max(s.T,muon_mass/3));
            decay_kernel k;auto begin=std::chrono::steady_clock::now();
            bool const ok=k.init(g,s.T,mue,mumu,inner,recipe==0);
            double const seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            for(int radiation=0;radiation<5;++radiation) for(int charge=0;charge<2;++charge) {
                double f[max_order]{},fb[max_order]{},sh[2]={1e300,0},shb[2]={1e300,0};
                double const delta=(charge==0?1:-1)*(mumu-mue);
                if(radiation) {
                    double eta= radiation==1?-5: radiation==2?.3: radiation==3?5:0;
                    double etab=radiation==2?delta/s.T-eta:eta;
                    spectrum(g,s.T*(radiation==4?.5:1),eta,f,sh);
                    spectrum(g,s.T*(radiation==4?2:1),etab,fb,shb);
                }
                sources src;if(ok) src=evaluate_decay(k,charge,f,fb,sh,shb);
                prefix(recipe,radiation,charge);
                row<<','<<ok<<','<<seconds<<','<<src.number<<','<<src.energy[0]<<','<<src.energy[1]
                   <<','<<src.emission_n[0]<<','<<src.loss_n[0]<<','<<src.emission[0]<<','<<src.loss[0]<<'\n';
            }
        }
        lines[id]=row.str();
    }
    std::ofstream out(output("rates"));
    out<<"id,rho,T,Ye,Ymu,mue,mumu,eos_valid,recipe,radiation,charge,kernel_ok,build_seconds,R,Q0,Q1,emissionN,lossN,emissionE,lossE\n";
    for(auto const& s:lines) out<<s;
    REQUIRE(out.good());
}

// A homogeneous, closed matter + four-neutrino box at fixed baryon density.
// Only muon decay is active. Evolve matter energy, Ye and Ymu using the same
// signed exchanges as GRACE; recompute production EOS potentials each step.
// This isolates decay accuracy from CC, hydrodynamics and transport.
TEST_CASE("Grey long closed-box evolution with matter feedback", "[.][grey_boxes]") {
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    std::vector<state> all;
    for(double rho:{1e10,1e12,1e14,1e15}) for(double T:{1.,10.,50.})
    for(double ym:{.0006,.01,.05}) all.push_back({rho*nu_constants::RHOGF,T,.1,ym});
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    int const total=int(all.size())*2*5;std::vector<std::string> lines(total);
    std::ofstream out_file(output("boxes"));
    out_file<<"id,rho,T0,Ymu0,radiation,recipe,target_seconds,time_seconds,status,steps,rejections,T,Ye,Ymu,energy_error,Le_error,Lmu_error,residual,wall_seconds,last_rejection_reason\n";
    out_file.flush();
    #pragma omp parallel for schedule(dynamic)
    for(int task=rank;task<total;task+=ranks) {
        int const recipe=task%5,rad=(task/5)%2,id=task/10;
        state s=all[id],initial=s;auto row=stream();double mue,mumu;
        int status=potentials(e,s,mue,mumu)?0:1,steps=0,rejected=0,last_failure=0;
        bool const grey=recipe<2;
        int const order=recipe==4?32:16,inner=recipe==4?32:16;
        double const maxdt=(recipe==0 || recipe==2)?1e-5:5e-6;
        double const change_limit=(recipe==0 || recipe==2)?.02:.01;
        double const nb=s.rho/nu_constants::RHOGF*nu_constants::avogadro;
        double u[20]{},out[20],t=0,h=change_limit*5e-8,eps=internal_energy(e,s),maxres=0;
        // Identical initial radiation moments across solver quadrature orders.
        grid g;g.init(32,std::max(s.T,muon_mass/3));
        if(rad) for(int a=0;a<4;++a) for(int j=0;j<g.n;++j) {
            double const f=occupation(g.e[j]/s.T-3);
            u[5*a]+=g.w[j]*g.e[j]*f*energy_unit;u[5*a+4]+=g.w[j]*f*number_unit;
        }
        auto energy=[&](double const* a,double ep) { double v=s.rho*ep;for(int j=0;j<4;++j)v+=a[5*j];return v; };
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
        m1_prims_array_t base[4]{};m1_eas_array_t eas[4]{};
        for(double target:{1e-6,1e-5,1e-4,1e-3,1e-2}) {
            while(!status && t<target*(1-1e-13)) {
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()>120) {status=6;break;}
                if(steps+rejected>50000) {status=2;break;}
                if(!potentials(e,s,mue,mumu)) {status=3;break;}
                kernel thermal;double scale=std::max(s.T,muon_mass/3);
                for(int a=0;a<4;++a) if(u[5*a+4]>0)
                    scale=std::max(scale,std::sqrt(s.T*u[5*a]/u[5*a+4]*number_unit/energy_unit/3));
                bool const valid=thermal.init({s.T,mue,nb,.8,.2},{false,false,false},order,scale);
                decay_kernel decay;if(!valid || !decay.init(thermal.g,s.T,mue,mumu,inner,grey)) {status=4;break;}
                h=std::min({h,maxdt,target-t});double res=0;
                bool ok=leptonic_implicit_update(thermal,decay,false,metric,base,eas,u,out,h*time_unit,res,nullptr,false,true);
                last_failure=ok?0:1;
                state next=s;double enew=eps;
                if(ok) {
                    next.ye-=((out[4]-u[4])-(out[9]-u[9]))/s.rho;
                    next.ymu-=((out[14]-u[14])-(out[19]-u[19]))/s.rho;
                    for(int a=0;a<4;++a) enew-=(out[5*a]-u[5*a])/s.rho;
                    auto check=next;double checkeps=enew;eos_err_t err{};
                    e.press_temp__eps_rho_ye_ymu(next.T,checkeps,next.rho,next.ye,next.ymu,err);
                    bool const eos_ok=std::isfinite(next.T+next.ye+next.ymu+checkeps)
                       && next.rho==check.rho && next.ye==check.ye && next.ymu==check.ymu
                       && std::abs(internal_energy(e,next)-enew)<1e-8*std::max(std::abs(enew),1e-10);
                    bool const change_ok=std::abs(next.T-s.T)<change_limit*s.T
                       && std::abs(next.ye-s.ye)<change_limit*std::max(s.ye,1e-6)
                       && std::abs(next.ymu-s.ymu)<change_limit*std::max(s.ymu,1e-6);
                    ok=eos_ok && change_ok;last_failure=!eos_ok?2:!change_ok?3:0;
                }
                if(!ok) {++rejected;h*=.5;if(h<1e-16)status=5;continue;}
                maxres=std::max(maxres,res);s=next;eps=enew;
                std::copy(out,out+20,u);t+=h;++steps;h*=1.4;
            }
            record(target);
        }
        lines[task]=row.str();
        #pragma omp critical(grey_box_output)
        { out_file<<lines[task];out_file.flush(); }
    }
    REQUIRE(out_file.good());
}
