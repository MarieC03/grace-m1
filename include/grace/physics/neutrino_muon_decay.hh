/** @file
 * Finite-mass, tree-level muon decay / inverse decay, angular R0 closure.
 * Oriented energies: (nu_mu, anti_nu_e) for mu-, (anti_nu_mu, nu_e) for mu+.
 * See doc/developer_guide/muon_decay.rst for normalization and limitations.
 */
#ifndef GRACE_NEUTRINO_MUON_DECAY_HH
#define GRACE_NEUTRINO_MUON_DECAY_HH
#include <grace/physics/neutrino_pair_collision.hh>
#include <grace/physics/neutrino_grey_decay.hh>
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#endif

namespace grace::pairs {
constexpr double muon_mass = 105.6583755;
constexpr double electron_mass = 0.51099895000;
constexpr double decay_gf = 1.1663787e-11; // MeV^-2
constexpr double hbar_mev_s = 6.582119569e-22;
constexpr double hbarc_mev_cm = 1.973269804e-11;

// Positive Gauss-Legendre rule on (0,1), constructed once per kernel matrix.
struct decay_rule {
    int n;
    double x[64], w[64];
    KOKKOS_INLINE_FUNCTION void init(int order) {
        n=order;
        for(int i=0;i<n;++i) {
            double z=Kokkos::cos(pi*(i+0.75)/(n+0.5)),dp=0;
            for(int it=0;it<24;++it) {
                double p=1,pm=0;
                for(int j=1;j<=n;++j) {
                    double const pn=((2*j-1)*z*p-(j-1)*pm)/j;
                    pm=p; p=pn;
                }
                dp=n*(z*p-pm)/(z*z-1);
                double const dz=p/dp; z-=dz;
                if(Kokkos::fabs(dz)<2e-15) break;
            }
            x[i]=(1-z)/2;
            w[i]=1/((1-z*z)*dp*dp);
        }
    }
};

// Compute the occupation and hole from the same bounded exponential. The
// subtraction 1-f would erase the hole when f rounds to one.
KOKKOS_INLINE_FUNCTION void decay_occupation_and_hole(double x,double& f,double& hole) {
    double const z=Kokkos::exp(-Kokkos::fabs(x));
    double const denom=1+z;
    double const low=z/denom, high=1/denom;
    f=x>=0?low:high;
    hole=x>=0?high:low;
}

// Rp+Ra for both charge conjugates. Swapping the neutrino energies keeps
// the positive phase-space domain and occupations and reverses the linear
// angular term. Keep separate sums for the two oriented matrix elements.
template <bool BothOrientations>
KOKKOS_INLINE_FUNCTION void decay_total_impl(double T,double mue,double mumu,
    double e,double ep,decay_rule const& rule,double (&rate)[2],double* reverse) {
    rate[0]=rate[1]=0;
    if constexpr(BothOrientations) reverse[0]=reverse[1]=0;
    double const M2=muon_mass*muon_mass, m2=electron_mass*electron_mass;
    double const omega=e+ep;
    double const smax=Kokkos::fmin(4*e*ep,(muon_mass-electron_mass)*(muon_mass-electron_mass));
    // The discarded charged-lepton tail is exponentially suppressed by >=48T.
    double const cut=Kokkos::fmax(electron_mass,
        Kokkos::fmax(Kokkos::fabs(mue),Kokkos::fabs(mumu)-omega))+48*T;
    constexpr double units=hbarc_mev_cm*hbarc_mev_cm*hbarc_mev_cm/hbar_mev_s;
    for(int a=0;a<rule.n;++a) {
        double const s=smax*rule.x[a], Q=Kokkos::sqrt(omega*omega-s);
        double const A=(M2-m2-s)/2, B=Kokkos::sqrt(Kokkos::fmax(0.0,A*A-s*m2));
        if(!(B>0 && Q>0)) continue;
        double const denom=omega*A+Q*B;
        // Rationalized lower endpoint avoids catastrophic cancellation as s->0.
        double const lo=(A*A+Q*Q*m2)/denom;
        double const hi=Kokkos::fmin(denom/s,cut);
        if(!(hi>lo)) continue;
        double const ca=(e-ep)/Q;
        double const weighted_interval=units*smax*rule.w[a]*(hi-lo);
        double const pref=weighted_interval/(64*pi*e*e*ep*ep*Q);
        double reverse_pref=0;
        if constexpr(BothOrientations) reverse_pref=weighted_interval/(64*pi*ep*ep*e*e*Q);
        for(int b=0;b<rule.n;++b) {
            double const Ee=lo+(hi-lo)*rule.x[b], Em=Ee+omega;
            double const u=Kokkos::fmax(-1.0,Kokkos::fmin(1.0,(s*Ee-omega*A)/(Q*B)));
            double const t=u*ca;
            double const t2=t*t+0.5*(1-u*u)*Kokkos::fmax(0.0,1-ca*ca);
            // <|M|^2>_phi = 64 GF^2 <(p_e.q_numu)(p_mu.q_anue)>_phi.
            double const matrix=16*decay_gf*decay_gf*
                Kokkos::fmax(0.0,A*(A+s)-s*B*t-B*B*t2);
            double const weight=pref*rule.w[b]*matrix;
            double reverse_weight=0;
            if constexpr(BothOrientations) {
                double const reverse_matrix=16*decay_gf*decay_gf*
                    Kokkos::fmax(0.0,A*(A+s)-s*B*(-t)-B*B*t2);
                reverse_weight=reverse_pref*rule.w[b]*reverse_matrix;
            }
            for(int charge=0;charge<2;++charge) {
                double const sign=charge==0?1:-1;
                double const xe=(Ee-sign*mue)/T, xm=(Em-sign*mumu)/T;
                double fe,he,fm,hm;
                decay_occupation_and_hole(xe,fe,he);
                decay_occupation_and_hole(xm,fm,hm);
                double const occupations=fm*he+fe*hm;
                rate[charge]+=weight*occupations;
                if constexpr(BothOrientations) reverse[charge]+=reverse_weight*occupations;
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION void decay_total(double T,double mue,double mumu,
    double e,double ep,decay_rule const& rule,double (&rate)[2]) {
    decay_total_impl<false>(T,mue,mumu,e,ep,rule,rate,nullptr);
}

struct decay_kernel {
    grid g;
    double T, delta;
    bool grey=false;
    grey_decay_rates grey_rates;
    double total[2][max_order][max_order];
    KOKKOS_INLINE_FUNCTION bool init(grid const& energy_grid,double temp,
        double mue,double mumu,int inner_order,int grey_nodes=0) {
        g=energy_grid; T=temp; delta=mumu-mue; grey=grey_nodes!=0;
        if(grey) return grey_rates.init(temp,mue,mumu);
        GRACE_SOURCE_PROFILE(decay_kernel);
        if(!(temp>0) || !Kokkos::isfinite(mue) || !Kokkos::isfinite(mumu)
           || inner_order<8 || inner_order>64) return false;
        decay_rule rule; rule.init(inner_order);
        for(int i=0;i<g.n;++i) for(int j=i;j<g.n;++j) {
            double r[2],reverse[2];
            if(i==j) decay_total(T,mue,mumu,g.e[i],g.e[j],rule,r);
            else decay_total_impl<true>(T,mue,mumu,g.e[i],g.e[j],rule,r,reverse);
            for(int c=0;c<2;++c) {
                if(!(r[c]>=0) || !Kokkos::isfinite(r[c])) return false;
                total[c][i][j]=r[c];
                if(i!=j) {
                    if(!(reverse[c]>=0) || !Kokkos::isfinite(reverse[c])) return false;
                    total[c][j][i]=reverse[c];
                }
            }
        }
        return true;
    }
    KOKKOS_INLINE_FUNCTION void rates(int charge,int i,int j,double& p,double& a) const {
        if(grey) Kokkos::abort("Grey decay has no spectral matrix");
        double const x=(g.e[i]+g.e[j]-(charge==0?delta:-delta))/T;
        // One bounded exponential, retaining both tiny rates without 1-f
        // cancellation. These rates are evaluated repeatedly in the source solve.
        double const r=Kokkos::exp(-Kokkos::fabs(x));
        double const big=total[charge][i][j]/(1+r),small=big*r;
        p=x>=0?small:big; a=x>=0?big:small;
    }
};

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
// Exact CPU cache, shared across OpenMP threads in one MPI process. No cell
// identity is used: AMR/repartition/restart cannot make an entry stale. All
// inputs, including active grid nodes/weights, must match bit for bit.
template<int Capacity=4096,int Ways=4> class decay_kernel_cache {
    static_assert(Capacity>0 && Ways>0 && Capacity%Ways==0);
    struct key {
        int n, order;
        std::uint64_t value[4+2*max_order]{};
        explicit key(grid const& g,double T,double mue,double mumu,int inner):n(g.n),order(inner) {
            value[0]=std::bit_cast<std::uint64_t>(T);
            value[1]=std::bit_cast<std::uint64_t>(mue);
            value[2]=std::bit_cast<std::uint64_t>(mumu);
            value[3]=std::bit_cast<std::uint64_t>(g.scale);
            for(int i=0;i<n;++i) {
                value[4+2*i]=std::bit_cast<std::uint64_t>(g.e[i]);
                value[5+2*i]=std::bit_cast<std::uint64_t>(g.w[i]);
            }
        }
        key()=default;
        bool operator==(key const& b) const {
            if(n!=b.n || order!=b.order) return false;
            for(int i=0;i<4+2*n;++i) if(value[i]!=b.value[i]) return false;
            return true;
        }
        std::uint64_t hash() const {
            std::uint64_t h=14695981039346656037ULL;
            auto mix=[&](std::uint64_t x) { h^=x; h*=1099511628211ULL; h^=h>>32; };
            mix(n); mix(order);
            for(int i=0;i<4+2*n;++i) mix(value[i]);
            return h;
        }
    };
    struct entry { bool valid=false; key input{}; decay_kernel kernel{}; };
    struct bucket { std::mutex mutex; unsigned next=0; entry entries[Ways]; };
    std::unique_ptr<bucket[]> buckets=std::make_unique<bucket[]>(Capacity/Ways);
    static void copy(decay_kernel const& from,decay_kernel& to) {
        to.g.n=from.g.n; to.g.scale=from.g.scale; to.T=from.T; to.delta=from.delta; to.grey=from.grey;
        for(int i=0;i<from.g.n;++i) {
            to.g.e[i]=from.g.e[i]; to.g.w[i]=from.g.w[i];
            for(int j=0;j<from.g.n;++j) for(int c=0;c<2;++c)
                to.total[c][i][j]=from.total[c][i][j];
        }
    }
public:
    bool find(grid const& g,double T,double mue,double mumu,int order,decay_kernel& out) {
        if(g.n<1 || g.n>max_order) return false;
        key const k(g,T,mue,mumu,order);
        auto& b=buckets[k.hash()%(Capacity/Ways)];
        std::lock_guard lock(b.mutex);
        for(auto const& e:b.entries) if(e.valid && e.input==k) { copy(e.kernel,out); return true; }
        return false;
    }
    void insert(decay_kernel const& kernel,double mue,double mumu,int order) {
        key const k(kernel.g,kernel.T,mue,mumu,order);
        auto& b=buckets[k.hash()%(Capacity/Ways)];
        std::lock_guard lock(b.mutex);
        // Two threads may compute the same miss outside the lock. Store once.
        for(auto const& e:b.entries) if(e.valid && e.input==k) return;
        auto& e=b.entries[b.next]; b.next=(b.next+1)%Ways;
        e.input=k; copy(kernel,e.kernel); e.valid=true;
    }
};
inline decay_kernel_cache<>& shared_decay_kernel_cache() {
    static decay_kernel_cache<> cache;
    return cache;
}
#endif

KOKKOS_INLINE_FUNCTION bool init_decay_kernel(decay_kernel& out,grid const& g,
    double T,double mue,double mumu,int order,bool reuse=false,int grey_nodes=0) {
    if(grey_nodes) return out.init(g,T,mue,mumu,order,grey_nodes); // no spectral cache or fallback
    if(order<8 || order>64) return false;
    // Disjoint cache recipes: exact reuse must not mix spectral and surrogate kernels.
    int const cache_order=order+128*grey_nodes;
    #if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
    if(reuse) {
        auto& cache=shared_decay_kernel_cache();
        if(cache.find(g,T,mue,mumu,cache_order,out)) {
            #ifdef GRACE_LEPTON_PROFILE
            source_profile::decay_cache_hits.fetch_add(1,std::memory_order_relaxed);
            #endif
            return true;
        }
        #ifdef GRACE_LEPTON_PROFILE
        source_profile::decay_cache_misses.fetch_add(1,std::memory_order_relaxed);
        #endif
        if(!out.init(g,T,mue,mumu,order,grey_nodes)) return false;
        cache.insert(out,mue,mumu,cache_order);
        return true;
    }
    #else
    (void)reuse; // The host parameter validation rejects caching on GPU builds.
    #endif
    return out.init(g,T,mue,mumu,order,grey_nodes);
}

KOKKOS_INLINE_FUNCTION sources evaluate_decay(decay_kernel const& k,int charge,
    double const* f,double const* fb,double const* shape0=nullptr,double const* shape1=nullptr) {
    if(k.grey) {
        if(!shape0 || !shape1) Kokkos::abort("Grey decay requires FD fit parameters");
        return k.grey_rates.evaluate(charge,shape0,shape1);
    }
    sources s;
    for(int i=0;i<k.g.n;++i) for(int j=0;j<k.g.n;++j) {
        double p,a; k.rates(charge,i,j,p,a);
        double const weight=k.g.w[i]*k.g.w[j]; p*=weight; a*=weight;
        double const net=p*(1-f[i])*(1-fb[j])-a*f[i]*fb[j];
        s.number+=net; s.energy[0]+=k.g.e[i]*net; s.energy[1]+=k.g.e[j]*net;
        double const j0=p*(1-fb[j]),j1=p*(1-f[i]);
        s.emission_n[0]+=j0; s.emission_n[1]+=j1;
        s.emission[0]+=k.g.e[i]*j0; s.emission[1]+=k.g.e[j]*j1;
        double const l0=(j0+a*fb[j])*f[i],l1=(j1+a*f[i])*fb[j];
        s.loss_n[0]+=l0; s.loss_n[1]+=l1;
        s.loss[0]+=k.g.e[i]*l0; s.loss[1]+=k.g.e[j]*l1;
    }
    return s;
}

// Literal equilibrium-partner grey approximation. Coefficients are in CGS:
// eta_E [MeV cm^-3 s^-1], eta_N [cm^-3 s^-1], kappa_{E,N} [s^-1].
// It has the correct LTE fixed point, but independent out-of-LTE relaxation
// does not preserve equal daughter event counts (see documentation).
struct decay_lte_rates {
    double eta_E[2],eta_N[2],kappa_E[2],kappa_N[2];
};
KOKKOS_INLINE_FUNCTION bool equilibrium_decay_rates(decay_kernel const& k,
    int charge,double eta0,double eta1,decay_lte_rates& out) {
    if(k.grey) {
        double const eta[2]={eta0,eta1},shape[2][2]={{-eta0,1/k.T},{-eta1,1/k.T}};
        auto const src=k.grey_rates.evaluate(charge,shape[0],shape[1]);
        for(int a=0;a<2;++a) {
            double n=0,J=0;
            bool const shifted=k.g.e[0]/k.T-eta[a]>500;
            for(int i=0;i<k.g.n;++i) {
                double const f=shifted?Kokkos::exp(-(k.g.e[i]-k.g.e[0])/k.T):occupation(k.g.e[i]/k.T-eta[a]);
                n+=k.g.w[i]*f; J+=k.g.w[i]*k.g.e[i]*f;
            }
            if(!(n>0 && J>0)) return false;
            out.eta_E[a]=src.emission[a]; out.eta_N[a]=src.emission_n[a];
            if(shifted) {
                double const partner=occupation(k.grey_rates.energy[charge][1-a]/k.T-eta[1-a]);
                double const loss=(src.emission_n[a]+k.grey_rates.absorption[charge]*partner)
                    *Kokkos::exp((k.g.e[0]-k.grey_rates.energy[charge][a])/k.T);
                out.kappa_N[a]=loss/n; out.kappa_E[a]=loss*k.grey_rates.energy[charge][a]/J;
            } else {
                out.kappa_N[a]=src.loss_n[a]/n; out.kappa_E[a]=src.loss[a]/J;
            }
            if(!Kokkos::isfinite(out.eta_E[a]+out.eta_N[a]+out.kappa_N[a]+out.kappa_E[a])) return false;
        }
        return true;
    }
    // Shared with the thermal pairs: an underflowing orientation (|eta| > ~500) keeps a finite kappa.
    lte_rates r;
    if(!lte_pair_rates(k.g,k.T,eta0,eta1,
        [&](double const* f,double const* fb){ return evaluate_decay(k,charge,f,fb); },r)) return false;
    for(int s=0;s<2;++s) {
        out.eta_E[s]=r.emission[s]; out.eta_N[s]=r.emission_n[s];
        out.kappa_E[s]=r.kappa_E[s]; out.kappa_N[s]=r.kappa_N[s];
    }
    return true;
}

// Current-state damping for the optically thick transport-flux correction.
// Keep it OUT of ordinary EAS: the coupled collision residual already adds
// these reactions. Inputs are individual-species comoving CGS/MeV moments.
KOKKOS_INLINE_FUNCTION bool leptonic_transport_opacities(kernel const& thermal,
    decay_kernel const* decay,bool thermal_active,double const (&n)[4],
    double const (&J)[4],double (&opacity)[4],diagnostic* diag=nullptr) {
    double f[4][max_order],shape[4][2];
    for(int s=0;s<4;++s) {
        opacity[s]=0;
        if(!decay && s<2) continue;
        if(!reconstruct(thermal.g,n[s],J[s],f[s],diag,shape[s])) {
            if(diag) diag->species=s;
            return false;
        }
    }
    auto const add=[&](sources const& src,int a,int b) {
        if(J[a]>0) opacity[a]+=src.loss[0]/J[a]/time_unit;
        if(J[b]>0) opacity[b]+=src.loss[1]/J[b]/time_unit;
    };
    if(thermal_active) add(evaluate(thermal,f[2],f[3]),2,3);
    if(decay) {
        add(evaluate_decay(*decay,0,f[2],f[1],shape[2],shape[1]),2,1);
        add(evaluate_decay(*decay,1,f[3],f[0],shape[3],shape[0]),3,0);
    }
    return true;
}
} // namespace grace::pairs
#endif
