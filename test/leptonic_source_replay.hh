#pragma once
#include <catch2/catch_test_macros.hpp>
#include <grace/physics/neutrino_leptonic_update.hh>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace grace::test {
inline void replay_captured_leptonic_source() {
    using namespace grace;
    using namespace grace::pairs;
    auto path=std::getenv("GRACE_LEPTON_REPLAY");REQUIRE(path!=nullptr);
    std::ifstream f(path);int version;f>>version;REQUIRE(version==1);
    material m;double mumu,h;f>>m.T>>m.mu_e>>m.nb>>m.yn>>m.yp>>mumu>>h;
    kernel thermal;decay_kernel decay;f>>thermal.g.n>>thermal.g.scale;
    REQUIRE(thermal.g.n>0);REQUIRE(thermal.g.n<=max_order);
    for(int i=0;i<thermal.g.n;++i)f>>thermal.g.e[i]>>thermal.g.w[i]>>thermal.boltz[i];
    double old[20],out[20];for(double& v:old)f>>v;
    m1_eas_array_t eas[4]{};m1_prims_array_t base[4]{};
    for(int a=0;a<4;++a)for(int v:{KAL,KSL,KANL,ETAL,ETANL})f>>eas[a][v];
    for(int i=0;i<thermal.g.n;++i)for(int j=0;j<thermal.g.n;++j)f>>thermal.a[i][j];
    decay.g=thermal.g;decay.T=m.T;decay.delta=mumu-m.mu_e;decay.grey=false;
    for(int c=0;c<2;++c)for(int i=0;i<thermal.g.n;++i)for(int j=0;j<thermal.g.n;++j)f>>decay.total[c][i][j];
    REQUIRE(f.good());metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    for(int s=0;s<4;++s) {
        species_trial_cache c;diagnostic d;
        bool const valid=c.update(thermal.g,metric,base[s],eas[s],old+5*s,&d);
        double remaining=old[5*s+4]/number_unit,discrete_min=0;
        for(int i=0;i<thermal.g.n && remaining>0;++i) {
            double const dn=std::min(remaining,thermal.g.w[i]);discrete_min+=dn*thermal.g.e[i];remaining-=dn;
        }
        std::cout<<"REPLAY_DISCRETE_BOUND species="<<s<<" J="<<old[5*s]/energy_unit<<" Jmin="<<discrete_min
                 <<" excessN="<<remaining<<" ratio="<<(old[5*s]/energy_unit/discrete_min)<<'\n';
        std::cout<<"REPLAY_INITIAL species="<<s<<" valid="<<valid<<" code="<<int(d.code)<<" n="<<d.n<<" J="<<d.J
                 <<" residual="<<d.residual<<" iteration="<<d.iteration<<'\n';
    }
    for(int order:{thermal.g.n,32}) {
        kernel k=thermal;decay_kernel dk=decay;
        if(order!=thermal.g.n){REQUIRE(k.init(m,{true,true,false},order,thermal.g.scale));REQUIRE(dk.init(k.g,m.T,m.mu_e,mumu,16));}
        for(double factor:{1.,.1,.01})for(bool block:{false,true}){
            double res=0;leptonic_diagnostic d;
            bool const ok=leptonic_implicit_update(k,dk,true,metric,base,eas,old,out,h*factor,res,nullptr,false,block,nullptr,&d);
            std::cout<<"REPLAY order="<<order<<" factor="<<factor<<" block="<<block<<" ok="<<ok<<" residual="<<res
                     <<" phase="<<d.phase<<" species="<<d.species<<" continuation="<<d.continuation<<" newton="<<d.nonlinear.stage
                     <<" iteration="<<d.nonlinear.iteration<<" column="<<d.nonlinear.column<<" fit="<<int(d.reconstruction.code)
                     <<" fitres="<<d.reconstruction.residual<<'\n';
            if(!ok && d.reconstruction.n>0) {
                double remaining=d.reconstruction.n,jmin=0;
                for(int i=0;i<k.g.n && remaining>0;++i) {
                    double const dn=std::min(remaining,k.g.w[i]);jmin+=dn*k.g.e[i];remaining-=dn;
                }
                std::cout<<std::setprecision(17)<<"REPLAY_FAILED_MOMENTS order="<<order<<" factor="<<factor<<" block="<<block
                         <<" species="<<d.species<<" n="<<d.reconstruction.n<<" J="<<d.reconstruction.J
                         <<" Jmin="<<jmin<<" JminusJmin="<<(d.reconstruction.J-jmin)
                         <<" relative_slack="<<((d.reconstruction.J-jmin)/d.reconstruction.J)<<" excessN="<<remaining<<'\n';
            }
            if(std::getenv("GRACE_REPLAY_REQUIRE_SUCCESS")) CHECK(ok);
            if(ok) {
                double charge=0,scale=0,energy_change=0;
                for(int species=0;species<4;++species) {
                    double const lhs=(1+h*factor*eas[species][KANL])*out[5*species+4];
                    double const rhs=old[5*species+4]+h*factor*eas[species][ETANL];
                    charge+=(species==0||species==2?1:-1)*(lhs-rhs);
                    scale+=std::abs(lhs)+std::abs(rhs);
                    energy_change+=out[5*species]-old[5*species];
                }
                std::cout<<"REPLAY_CONSERVATION order="<<order<<" factor="<<factor<<" block="<<block
                         <<" charge_scaled="<<charge/std::max(scale,1e-100)<<" energy_increment="<<energy_change<<'\n';
                REQUIRE(std::abs(charge)<1e-12*std::max(scale,1e-100));
            }
        }
    }
}
} // namespace grace::test
