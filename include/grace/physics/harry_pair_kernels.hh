#ifndef GRACE_HARRY_PAIR_KERNELS_HH
#define GRACE_HARRY_PAIR_KERNELS_HH
#include <grace/physics/harry_spectral_table.hh>
#include <grace/physics/neutrino_muon_decay.hh>
#include <grace/physics/guo_nnbrem.hh>

namespace grace::pairs {
// Phi0 is an angular integral, whereas a is an angular average. The caller
// supplies its verified normalization explicitly; no hidden unit conversion.
// Temperature/upper-degeneracy excursions are rejected. The explicit lower
// electron-degeneracy boundary policy below follows Harry's pair reader.
KOKKOS_INLINE_FUNCTION bool harry_grid(harry::device_handle const& h,grid& g) {
    if(h.n_groups<2 || h.n_groups>max_order) return false;
    g.n=h.n_groups; g.scale=0;
    for(int i=0;i<g.n;++i) {
        g.e[i]=h.energies(i);
        double const de=h.widths(i);
        if(!(g.e[i]>0 && de>0) || !Kokkos::isfinite(g.e[i]+de)) return false;
        if(i && !(g.e[i]>g.e[i-1])) return false;
        g.w[i]=phase*g.e[i]*g.e[i]*de;
    }
    return true;
}

// Species is 0=nue, 2=numu, 4=nutau. The opposite orientation is transposed
// by the common event contraction. The tau aggregate is explicitly symmetric.
enum class harry_pair_boundary_policy { lower_eta_clamp, strict };
KOKKOS_INLINE_FUNCTION bool init_harry_pair_kernel(kernel& k,
    harry::device_handle const& h,material const& m,int species,
    double phi0_to_average,bool symmetric=false,
    harry_pair_boundary_policy boundary=harry_pair_boundary_policy::lower_eta_clamp,
    bool* eta_clamped=nullptr) {
    if(eta_clamped) *eta_clamped=false;
    if(!(m.T>0 && m.mu_e>=0 && phi0_to_average>0) || !Kokkos::isfinite(m.T)
        || !Kokkos::isfinite(m.mu_e) || h.pair0.rank!=2
        || species<0 || species>4 || species%2) return false;
    double const eta=m.mu_e/m.T;
    double const eta_min=Kokkos::exp(h.pair0.axes[1](0));
    double const logeta=eta>0?Kokkos::log(eta):h.pair0.axes[1](0)-1;
    bool const below=eta==0 || logeta<h.pair0.axes[1](0);
    if(below && boundary==harry_pair_boundary_policy::strict) return false;
    double const eta_lookup=below?eta_min:eta;
    double x[2]={Kokkos::log(m.T),below?h.pair0.axes[1](0):logeta};
    if(!h.pair0.contains(x) || !harry_grid(h,k.g)) return false;
    if(eta_clamped) *eta_clamped=below;
    for(int i=0;i<k.g.n;++i) {
        // Only the kernel amplitude uses the table-edge eta. Pair DB depends
        // on actual T alone (mu_e + mu_positron = 0), preserving LTE exactly.
        k.boltz[i]=Kokkos::exp(-k.g.e[i]/m.T);
        for(int j=0;j<k.g.n;++j) {
            double const r=phi0_to_average*Kokkos::exp(h.pair_log_ann(m.T,eta_lookup,species,i,j));
            if(!(r>=0) || !Kokkos::isfinite(r)) return false;
            k.a[i][j]=r;
        }
    }
    if(symmetric) k.symmetrize();
    return true;
}

KOKKOS_INLINE_FUNCTION bool init_harry_decay_kernel(decay_kernel& k,
    harry::device_handle const& h,double T,double mue,double mumu,double phi0_to_average) {
    if(!h.ild_available || !(T>0 && phi0_to_average>0) || !Kokkos::isfinite(T)
        || !Kokkos::isfinite(mue) || !Kokkos::isfinite(mumu)
        || !Kokkos::isfinite(phi0_to_average) || !harry_grid(h,k.g)) return false;
    k.grey=false;
    // Harry's non-debug ILD lookup has no contribution below any lower face.
    // A zero matrix must also reset a previously populated kernel.
    double state[3]={T,mue/T,mumu/T};
    double x[3]{};
    bool clamped=false;
    for(int d=0;d<3;++d) {
        if(state[d]>0) x[d]=Kokkos::log(state[d]);
        if(state[d]<=0 || h.ild0.below(d,x[d])) {
            k.T=T; k.delta=0;
            for(int c=0;c<2;++c) for(int i=0;i<k.g.n;++i)
                for(int j=0;j<k.g.n;++j) k.total[c][i][j]=0;
            return true;
        }
        if(h.ild0.above(d,x[d])) {
            x[d]=h.ild0.axes[d](h.ild0.size[d]-1);
            state[d]=Kokkos::exp(x[d]);
            clamped=true;
        }
    }
    // Apply the same T and chemical degeneracies to interpolation and DB.
    k.T=state[0]; k.delta=clamped?k.T*(state[2]-state[1]):mumu-mue;
    double const log_scale=Kokkos::log(phi0_to_average);
    for(int c=0;c<2;++c) for(int i=0;i<k.g.n;++i) for(int j=0;j<=i;++j) {
        int const channel=c*k.g.n*(k.g.n+1)/2+i*(i+1)/2+j;
        double const loga=log_scale+h.ild0.interpolate(channel,x);
        double const exponent=((c==0?k.delta:-k.delta)-k.g.e[i]-k.g.e[j])/k.T;
        // Store p+a: existing bounded rates() recovers both orientations,
        // including strongly degenerate cases without exp(+large)*tiny.
        double const logtotal=loga+Kokkos::fmax(0.0,exponent)+Kokkos::log(1+Kokkos::exp(-Kokkos::fabs(exponent)));
        k.total[c][i][j]=Kokkos::exp(logtotal);
        if(!Kokkos::isfinite(k.total[c][i][j])) return false;
        k.total[c][j][i]=k.total[c][i][j];
    }
    return true;
}

KOKKOS_INLINE_FUNCTION bool init_harry_thermal(kernel& k,harry::device_handle const& h,
    guo_nnbrem::device_handle const& guo,material const& m,channels c,int species,bool symmetric=false,
    bool* eta_clamped=nullptr,
    harry_pair_boundary_policy boundary=harry_pair_boundary_policy::lower_eta_clamp) {
    if(eta_clamped) *eta_clamped=false;
    if(c.ee) { if(!init_harry_pair_kernel(k,h,m,species,0.5,symmetric,boundary,eta_clamped)) return false; }
    else {
        if(!harry_grid(h,k.g)) return false;
        for(int i=0;i<k.g.n;++i) {
            k.boltz[i]=Kokkos::exp(-k.g.e[i]/m.T);
            for(int j=0;j<k.g.n;++j) k.a[i][j]=0;
        }
    }
    double const free=m.yn+m.yp,nfree=m.nb*free*1e-39;
    auto const response=guo.local(m.T,nfree,free>0?m.yp/free:0);
    if(c.nn && response.state==guo_nnbrem::status::uninitialized) return false;
    for(int i=0;i<k.g.n;++i) for(int j=0;j<k.g.n;++j) {
        if(c.nn) k.a[i][j]+=0.5*guo_nnbrem::device_handle::kernel(response,k.g.e[i]+k.g.e[j],m.T,nfree).annihilation0;
        if(c.plasmon) k.a[i][j]+=plasmon_abs(m,k.g.e[i],k.g.e[j]);
        if(!(k.a[i][j]>=0) || !Kokkos::isfinite(k.a[i][j])) return false;
    }
    return true;
}
}
#endif
