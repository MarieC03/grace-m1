// Diagnostic exporter only. The independent reference lives in Python and
// reads the raw HDF5 with its own interpolation and quadrature implementation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <grace/system/mpi_runtime.hh>
#include <grace/system/kokkos_runtime.hh>
#include <grace/physics/eas_harry_rates.hh>
#include <grace/physics/eas_kinds.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
using namespace grace;
namespace grace { void initialize_loggers(); }
int main(int argc,char** argv) {
    std::string parfile;std::vector<char*> args{argv[0]};
    for(int i=1;i<argc;++i) {
        if(std::string(argv[i])=="--grace-parfile" && i+1<argc)parfile=argv[++i];
        else args.push_back(argv[i]);
    }
    if(parfile.empty())throw std::runtime_error("CC reference exporter requires --grace-parfile");
    config_parser::initialize(parfile);mpi_runtime::initialize(argc,argv);
    kokkos_runtime::initialize(&argc,argv);initialize_loggers();eos::initialize();
    int const n=int(args.size());args.push_back(nullptr);
    int const result=Catch::Session().run(n,args.data());
    harry::finalize_harry_table();return result;
}
TEST_CASE("Export Harry CC states for an independent raw-table reference", "[.][harry_cc_reference]") {
    char const* path=std::getenv("GRACE_HARRY_CC_TABLE");
    char const* output=std::getenv("GRACE_CC_REFERENCE_OUTPUT");
    REQUIRE(path);REQUIRE(output);
    REQUIRE(get_param<bool>("m1","eas","dilute_muon_cc_target"));
    harry::initialize_harry_table(path,false);auto const& h=harry::get_harry_device_handle();
    auto e=eos::get().get_eos<leptonic_eos_4d_t>();
    auto suffix="_rank"+std::to_string(parallel::mpi_comm_rank())+".csv";
    std::ofstream rates(std::string(output)+"/cc_rates"+suffix),spectra(std::string(output)+"/cc_spectra"+suffix),skips(std::string(output)+"/cc_skips"+suffix);
    rates<<std::setprecision(17);spectra<<std::setprecision(17);skips<<std::setprecision(17);
    std::string const common="id,rho_cgs,T,Ye,Ymu,mu_e,mu_mu,raw_mu_mu,mu_mu_reaction,mu_p,mu_n,Xn,Xp,eta_nue,eta_numu";
    rates<<common<<",species,radiation_factor,eps_rad,eta_N_code,eta_E_code,kappa_N_code,kappa_E_code\n";
    spectra<<common<<",species,bin,energy_MeV,width_MeV,ln_kappa_raw\n";
    skips<<"id,rho_cgs,T,Ye,Ymu,reason\n";
    int id=-1,exported=0;
    int const rank=parallel::mpi_comm_rank(),ranks=parallel::mpi_comm_size();
    for(double rc:{1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15})
    for(double temp:{.1,.3,1.,3.,10.,30.,80.})
    for(double ye:{.05,.1,.3})
    for(double ymu:{.0005,.0006,.001,.01,.05,.1}) {
        ++id;if(id%ranks!=rank)continue;
        double rho=rc*nu_constants::RHOGF,T=temp,Y=ye,M=ymu,mumu,mup,mun,Xa,Xh,Xn,Xp,A,Z;
        eos_err_t err{};
        double mue=e.mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(mumu,mup,mun,Xa,Xh,Xn,Xp,A,Z,T,rho,Y,M,err);
        auto skip=[&](char const* why){skips<<id<<','<<rc<<','<<temp<<','<<ye<<','<<ymu<<','<<why<<'\n';};
        auto changed=[](double a,double b){return std::abs(a-b)>32*2.2204460492503131e-16*std::max(std::abs(a),std::abs(b));};
        if(changed(rho,rc*nu_constants::RHOGF)||changed(T,temp)||changed(Y,ye)||changed(M,ymu)||!std::isfinite(mue+mumu+mup+mun)) {skip("EOS_adjusted_or_nonfinite");continue;}
        bool domain=true;
        for(int s:{0,2}) {
            auto const& a=h.cc[s];double x[4]={std::log(rho),std::log(T),Y,std::log(M)};
            if(a.above(0,x[0])||a.above(1,x[1]))domain=false;
            if(!a.below(0,x[0])&&!a.below(1,x[1])&&!a.contains(x))domain=false;
        }
        if(!domain){skip("CC_domain");continue;}
        double xyz[3]{};auto F=make_fugacity_state(e,rho,T,Y,M,1.,xyz,tau_policy_none{});
        F.dilute_cc_target=true;
        double const raw_mu_mu=e.raw_mumu__temp_rho_ye_ymu(T,rho,Y,M);
        F.mu_mu_reaction=dilute_muon_cc_potential(raw_mu_mu,F.mu_e,M);
        F.eta_cc_mu=(F.mu_mu_reaction+F.mu_p-F.mu_n-nu_constants::Qnp)/T;
        double const eta_e=(F.mu_e+F.mu_p-F.mu_n-nu_constants::Qnp)/T;
        auto prefix=[&](std::ostream& stream) {stream<<id<<','<<rc<<','<<T<<','<<Y<<','<<M<<','<<F.mu_e<<','<<F.mu_mu<<','<<raw_mu_mu<<','<<F.mu_mu_reaction<<','<<F.mu_p<<','<<F.mu_n<<','<<F.Xn<<','<<F.Xp<<','<<eta_e<<','<<F.eta_cc_mu;};
        bool safe=true;
        for(int s=0;s<4;++s)for(int i=0;i<h.n_groups;++i) {
            auto const& a=h.cc[s];bool const active=!a.below(0,std::log(rho))&&!a.below(1,std::log(T));
            double const eta=(s<2?eta_e:F.eta_cc_mu)*(s%2?-1:1);
            double l=active?h.cc_log(s,i,rho,T,Y,M):-INFINITY;
            prefix(spectra);spectra<<','<<s<<','<<i<<','<<h.energies(i)<<','<<h.widths(i)<<','<<l<<'\n';
            // Leave headroom for phase-space weights and the physical-unit
            // prefactor before production converts the result to code units.
            if(l+eta-h.energies(i)/T>550)safe=false;
        }
        if(!safe){skip("spectral_emissivity_overflow_risk");continue;}
        for(double factor:{.5,1.,2.}) {
            double means[NUMSPECIES]{};
            for(int s=0;s<4;++s) {
                double const eta=(s<2?eta_e:F.eta_cc_mu)*(s%2?-1:1);
                means[s]=factor*T/fermi::FDR<2,3>::get(eta);
            }
            means[4]=factor*T/fermi::FDR<2,3>::get(0.);
            auto out=compute_all_species_harry(h,F,means);
            for(int s=0;s<4;++s) {
                prefix(rates);auto const& r=out.out[s];
                rates<<','<<s<<','<<factor<<','<<means[s]<<','<<r.eta_N<<','<<r.eta_E<<','<<r.kappa_n<<','<<r.kappa_a<<'\n';
            }
        }
        ++exported;
    }
    REQUIRE(rates.good());REQUIRE(spectra.good());REQUIRE(skips.good());
    REQUIRE(exported>0);
}
