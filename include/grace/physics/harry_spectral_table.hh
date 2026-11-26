#ifndef GRACE_HARRY_SPECTRAL_TABLE_HH
#define GRACE_HARRY_SPECTRAL_TABLE_HH
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <string>
#include <cstddef>

namespace grace::harry {
// Fortran axis 0 is contiguous; channel is the outermost HDF5 dimension.
// Coordinates are in the file's representation (ln rho_code, ln T_MeV,
// Ye, ln Ymu or ln degeneracy). Values remain ln(rate) or Phi1/Phi0.
struct tensor {
    int rank=0, channels=0;
    int size[4]={0,0,0,0};
    Kokkos::View<double*> axes[4];
    Kokkos::View<double*> values;
    // Endpoints reconstructed through log/exp or unit conversion can differ
    // by a few ulps. Interpolation already holds the endpoint in that case;
    // this allowance is roundoff only, not a physical table extension.
    KOKKOS_INLINE_FUNCTION double axis_roundoff(int d) const {
        return 32*2.2204460492503131e-16*Kokkos::fmax(1.0,
            Kokkos::fmax(Kokkos::abs(axes[d](0)),Kokkos::abs(axes[d](size[d]-1))));
    }
    KOKKOS_INLINE_FUNCTION bool below(int d,double x) const {
        return x<axes[d](0)-axis_roundoff(d);
    }
    KOKKOS_INLINE_FUNCTION bool above(int d,double x) const {
        return x>axes[d](size[d]-1)+axis_roundoff(d);
    }
    KOKKOS_INLINE_FUNCTION bool contains(double const* x) const {
        for(int d=0;d<rank;++d)
            if(!Kokkos::isfinite(x[d])||below(d,x[d])||above(d,x[d])) return false;
        return true;
    }
    struct stencil {
        std::size_t offset[16], channel_stride;
        double weight[16];
        int count;
    };
    KOKKOS_INLINE_FUNCTION stencil locate(double const* x) const {
        if(rank<1 || rank>4) Kokkos::abort("Invalid Harry tensor rank");
        stencil s{};
        int lo[4]={0,0,0,0}; double w[4]={0,0,0,0};
        std::size_t stride[4]={1,1,1,1}, total=1;
        for(int d=0;d<rank;++d) {
            stride[d]=total; total*=size[d];
            int a=0,b=size[d]-1;
            while(b-a>1) { int m=(a+b)/2; if(x[d]<axes[d](m)) b=m; else a=m; }
            lo[d]=a;
            w[d]=(x[d]-axes[d](a))/(axes[d](a+1)-axes[d](a));
            w[d]=w[d]<0?0:(w[d]>1?1:w[d]);
        }
        s.count=1<<rank; s.channel_stride=total;
        for(int c=0;c<s.count;++c) {
            double weight=1; std::size_t offset=0;
            for(int d=0;d<rank;++d) {int hi=(c>>d)&1;weight*=hi?w[d]:1-w[d];offset+=(lo[d]+hi)*stride[d];}
            s.weight[c]=weight; s.offset[c]=offset;
        }
        return s;
    }
    KOKKOS_INLINE_FUNCTION double interpolate(int channel,stencil const& s) const {
        if(channel<0 || channel>=channels) Kokkos::abort("Invalid Harry tensor channel");
        auto const base=std::size_t(channel)*s.channel_stride;
        double result=0;
        for(int c=0;c<s.count;++c) result+=s.weight[c]*values(base+s.offset[c]);
        return result;
    }
    KOKKOS_INLINE_FUNCTION double interpolate(int channel,double const* x) const {
        return interpolate(channel,locate(x));
    }
};
// Prefix sums of E^2 dE and E^3 dE for the fixed collision quadrature.
// Input moments are per helicity, divided by the common phase-space factor.
struct moment_envelope {
    Kokkos::View<double*[3]> bins;
    double flat_mean=0;
    KOKKOS_INLINE_FUNCTION bool contains(double n,double J) const {
        int const count=bins.extent_int(0);
        if(count==0) return true;
        constexpr double tol=1e-12;
        if(n>bins(count-1,1)*(1+tol) || J>n*flat_mean*(1+tol)) return false;
        int lo=0,hi=count-1;
        while(lo<hi) {
            int mid=(lo+hi)/2;
            if(n>bins(mid,1)) lo=mid+1;
            else hi=mid;
        }
        // Minimum energy at fixed number fills the lowest bins first.
        double const below_n=lo?bins(lo-1,1):0,below_J=lo?bins(lo-1,2):0;
        return J>=(below_J+(n-below_n)*bins(lo,0))*(1-tol);
    }
};
struct device_handle {
    int n_groups=0;
    bool ild_available=false;
    Kokkos::View<double*> energies, widths;
    moment_envelope envelope;
    tensor cc[4], scattering, pair0,ild0;
    KOKKOS_INLINE_FUNCTION static double log_positive(double x) {return Kokkos::log(x>1e-300?x:1e-300);}
    KOKKOS_INLINE_FUNCTION double cc_log(int species,int energy,double rho_code,double T,double Ye,double Ymu) const {
        double x[4]={log_positive(rho_code),log_positive(T),Ye,log_positive(Ymu)};
        return cc[species].interpolate(energy,x);
    }
    KOKKOS_INLINE_FUNCTION double scatter_log(int species,int energy,double rho_code,double T,double Yp) const {
        double x[3]={log_positive(rho_code),log_positive(T),Yp};
        return scattering.interpolate(species*n_groups+energy,x);
    }
    KOKKOS_INLINE_FUNCTION double pair_log_ann(double T,double eta_e,int species,int i,int j) const {
        double x[2]={log_positive(T),log_positive(eta_e)};
        return pair0.interpolate((species*n_groups+i)*n_groups+j,x);
    }
    KOKKOS_INLINE_FUNCTION double ild_log_out(double T,double eta_e,double eta_mu,int charge,int i,int j) const {
        if(!ild_available) Kokkos::abort("Harry ILD data unavailable or malformed");
        double x[3]={log_positive(T),log_positive(eta_e),log_positive(eta_mu)};
        int hi=i>j?i:j,lo=i>j?j:i;
        return ild0.interpolate(charge*n_groups*(n_groups+1)/2+hi*(hi+1)/2+lo,x);
    }
};
bool is_harry_table(std::string const& path);
void initialize_harry_table(std::string const& path,bool require_ild=false);
device_handle const& get_harry_device_handle();
void finalize_harry_table();
}
#endif
