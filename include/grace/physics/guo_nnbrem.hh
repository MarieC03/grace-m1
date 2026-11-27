#ifndef GRACE_PHYSICS_GUO_NNBREM_HH
#define GRACE_PHYSICS_GUO_NNBREM_HH

#include <Kokkos_Core.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace grace::guo_nnbrem {
constexpr int nt=25, nd=37, ny=26, nw=40;
enum class status { ok, thermo_outside, energy_outside, uninitialized };
struct local_response { double log_s[nw]{}; status state=status::uninitialized; };
struct kernel_result {
    double production0=0, annihilation0=0, production1=0, annihilation1=0;
    status state=status::uninitialized;
};

// S_sigma [MeV^-1], log-interpolated in T, log(nB), Yp, log(omega).
// Domain failures return zero and an explicit status, exactly as Harry's
// get_NNbrem_kernel_form_Guo; this is not an extrapolation or analytic fallback.
struct device_handle {
    Kokkos::View<double*> log_s;
    double log_density[nd]{};
    KOKKOS_INLINE_FUNCTION static int index(int t,int d,int y,int w) {
        return (((t*nd+d)*ny+y)*nw+w);
    }
    KOKKOS_INLINE_FUNCTION local_response local(double t,double density,double yp) const {
        local_response out;
        if(log_s.extent(0)==0) return out;
        if(!(t>=2 && t<=50 && density>=1e-4 && density<=1 && yp>=0 && yp<=.5)) {
            out.state=status::thermo_outside; return out;
        }
        int it=int((t-2)/2); if(it>=nt-1)it=nt-2;
        int iy=int(yp/.02); if(iy>=ny-1)iy=ny-2;
        double ld=Kokkos::log(density);
        int id=0; while(id<nd-2 && ld>log_density[id+1])++id;
        double ft=(t-(2+2*it))/2, fy=(yp-.02*iy)/.02;
        double fd=(ld-log_density[id])/(log_density[id+1]-log_density[id]);
        for(int w=0;w<nw;++w) {
            double s=0;
            for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c)
                s+=(a?ft:1-ft)*(b?fd:1-fd)*(c?fy:1-fy)*log_s(index(it+a,id+b,iy+c,w));
            out.log_s[w]=s;
        }
        out.state=status::ok;return out;
    }
    KOKKOS_INLINE_FUNCTION static double response(local_response const& l,double omega,status& state) {
        state=l.state;if(state!=status::ok)return 0;
        constexpr double lo=.05011872336272722,hi=398.1071705534973;
        if(!(omega>=lo && omega<=hi)){state=status::energy_outside;return 0;}
        double z=(Kokkos::log(omega)/2.302585092994045684+1.3)*10;
        int i=int(z);if(i<0)i=0;if(i>=nw-1)i=nw-2;
        double f=z-i;return Kokkos::exp((1-f)*l.log_s[i]+f*l.log_s[i+1]);
    }
    KOKKOS_INLINE_FUNCTION static kernel_result kernel(local_response const& l,double omega,double t,double nfree) {
        kernel_result out;
        if(!(t>0 && nfree>0)){out.state=status::thermo_outside;return out;}
        double s=response(l,omega,out.state);if(out.state!=status::ok)return out;
        // Harry's normalized Legendre coefficients in cm^3/s. Phase-space
        // (2*pi)^-3 and angular factors belong to the collision integrator.
        constexpr double ga=1.27641,gf=1.1663787e-11,hc=1.973269804e-11,hbar=6.582119569e-22;
        constexpr double hcfm=hc*1e13;
        double a=ga*ga*gf*gf*nfree*hcfm*hcfm*hcfm*s*.25*hc*hc*hc/hbar;
        out.annihilation0=6*a;out.annihilation1=-2*a/3;
        double db=Kokkos::exp(-omega/t);
        out.production0=db*out.annihilation0;out.production1=db*out.annihilation1;
        return out;
    }
    KOKKOS_INLINE_FUNCTION bool evaluate_phi0(double t,double nfree,double yp,double omega,double& phi) const {
        auto k=kernel(local(t,nfree,yp),omega,t,nfree);phi=k.annihilation0;return k.state==status::ok;
    }
};

inline device_handle load(std::string const& filename) {
    std::ifstream input(filename);if(!input)throw std::runtime_error("Cannot open Guo response table: "+filename);
    std::vector<double> states,values;std::string line;bool pending=false;
    while(std::getline(input,line)) {
        std::istringstream is(line);std::vector<double> row;double v;
        while(is>>v)row.push_back(v);
        if(row.empty())continue;
        if(row.size()==3 && !pending){states.insert(states.end(),row.begin(),row.end());pending=true;}
        else if(row.size()==nw && pending){values.insert(values.end(),row.begin(),row.end());pending=false;}
        else throw std::runtime_error("Malformed Guo response table record");
    }
    if(pending || states.size()!=nt*nd*ny*3 || values.size()!=nt*nd*ny*nw)
        throw std::runtime_error("Wrong number of Guo response table records");
    device_handle h;double density[nd];int k=0;
    for(int p=-4;p<=-1;++p)for(int j=1;j<=9;++j)density[k++]=j*std::pow(10.,p);
    density[nd-1]=1;
    for(int d=0;d<nd;++d)h.log_density[d]=std::log(density[d]);
    auto close=[](double a,double b){return std::isfinite(a) && std::abs(a-b)<=1e-8*std::max(std::abs(b),1e-12);};
    for(int t=0;t<nt;++t)for(int d=0;d<nd;++d)for(int y=0;y<ny;++y){
        int i=((t*nd+d)*ny+y)*3;
        if(!close(states[i],2*(t+1)) || !close(states[i+1],density[d]) || !close(states[i+2],.02*y))
            throw std::runtime_error("Guo response table axes/order mismatch");
    }
    h.log_s=Kokkos::View<double*>("Guo log S_sigma",values.size());auto host=Kokkos::create_mirror_view(h.log_s);
    for(std::size_t i=0;i<values.size();++i){
        if(!(values[i]>0 && std::isfinite(values[i])))throw std::runtime_error("Nonpositive/nonfinite Guo response");
        host(i)=std::log(values[i]);
    }
    Kokkos::deep_copy(h.log_s,host);return h;
}
inline device_handle& storage(){static device_handle h;return h;}
inline void initialize(std::string const& filename){storage()=load(filename);}
inline device_handle get_device_handle(){return storage();}
// Call before Kokkos::finalize (as for the other table providers).
inline void finalize(){storage()=device_handle{};}
}
#endif
