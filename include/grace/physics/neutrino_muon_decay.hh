/** @file
 * Finite-mass, tree-level muon decay / inverse decay, angular R0 closure.
 * Oriented energies: (nu_mu, anti_nu_e) for mu-, (anti_nu_mu, nu_e) for mu+.
 * See doc/developer_guide/muon_decay.rst for normalization and limitations.
 */
#ifndef GRACE_NEUTRINO_MUON_DECAY_HH
#define GRACE_NEUTRINO_MUON_DECAY_HH
#include <grace/physics/neutrino_pair_collision.hh>

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

// Rp+Ra for both charge conjugates. Direct positive phase-space integration:
// no exp(delta_mu/T) times an underflowed absorption kernel.
KOKKOS_INLINE_FUNCTION void decay_total(double T,double mue,double mumu,
    double e,double ep,decay_rule const& rule,double (&rate)[2]) {
    rate[0]=rate[1]=0;
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
        double const pref=units*smax*rule.w[a]*(hi-lo)/(64*pi*e*e*ep*ep*Q);
        for(int b=0;b<rule.n;++b) {
            double const Ee=lo+(hi-lo)*rule.x[b], Em=Ee+omega;
            double const u=Kokkos::fmax(-1.0,Kokkos::fmin(1.0,(s*Ee-omega*A)/(Q*B)));
            double const t=u*ca;
            double const t2=t*t+0.5*(1-u*u)*Kokkos::fmax(0.0,1-ca*ca);
            // <|M|^2>_phi = 64 GF^2 <(p_e.q_numu)(p_mu.q_anue)>_phi.
            double const matrix=16*decay_gf*decay_gf*
                Kokkos::fmax(0.0,A*(A+s)-s*B*t-B*B*t2);
            for(int charge=0;charge<2;++charge) {
                double const sign=charge==0?1:-1;
                double const xe=(Ee-sign*mue)/T, xm=(Em-sign*mumu)/T;
                double const prod=occupation(xm)*occupation(-xe);
                double const abs=occupation(xe)*occupation(-xm);
                rate[charge]+=pref*rule.w[b]*matrix*(prod+abs);
            }
        }
    }
}

struct decay_kernel {
    grid g;
    double T, delta;
    double total[2][max_order][max_order];
    KOKKOS_INLINE_FUNCTION bool init(grid const& energy_grid,double temp,
        double mue,double mumu,int inner_order) {
        if(!(temp>0) || !Kokkos::isfinite(mue) || !Kokkos::isfinite(mumu)
           || inner_order<8 || inner_order>64) return false;
        g=energy_grid; T=temp; delta=mumu-mue;
        decay_rule rule; rule.init(inner_order);
        for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j) {
            double r[2]; decay_total(T,mue,mumu,g.e[i],g.e[j],rule,r);
            for(int c=0;c<2;++c) {
                if(!(r[c]>=0) || !Kokkos::isfinite(r[c])) return false;
                total[c][i][j]=r[c];
            }
        }
        return true;
    }
    KOKKOS_INLINE_FUNCTION void rates(int charge,int i,int j,double& p,double& a) const {
        double const x=(g.e[i]+g.e[j]-(charge==0?delta:-delta))/T;
        // One bounded exponential, retaining both tiny rates without 1-f
        // cancellation. These rates are evaluated repeatedly in the source solve.
        double const r=Kokkos::exp(-Kokkos::fabs(x));
        double const big=total[charge][i][j]/(1+r),small=big*r;
        p=x>=0?small:big; a=x>=0?big:small;
    }
};

KOKKOS_INLINE_FUNCTION sources evaluate_decay(decay_kernel const& k,int charge,
    double const* f,double const* fb) {
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
    double f[2][max_order],n[2]={},J[2]={};
    double const eta[2]={eta0,eta1};
    for(int s=0;s<2;++s) for(int i=0;i<k.g.n;++i) {
        f[s][i]=occupation(k.g.e[i]/k.T-eta[s]);
        n[s]+=k.g.w[i]*f[s][i]; J[s]+=k.g.w[i]*k.g.e[i]*f[s][i];
    }
    auto const ps=evaluate_decay(k,charge,f[0],f[1]);
    for(int s=0;s<2;++s) {
        if(!(n[s]>0 && J[s]>0)) return false;
        out.eta_E[s]=ps.emission[s]; out.eta_N[s]=ps.emission_n[s];
        out.kappa_E[s]=ps.loss[s]/J[s]; out.kappa_N[s]=ps.loss_n[s]/n[s];
    }
    return true;
}

// Current-state damping for the optically thick transport-flux correction.
// Keep it OUT of ordinary EAS: the coupled collision residual already adds
// these reactions. Inputs are individual-species comoving CGS/MeV moments.
KOKKOS_INLINE_FUNCTION bool leptonic_transport_opacities(kernel const& thermal,
    decay_kernel const* decay,bool thermal_active,double const (&n)[4],
    double const (&J)[4],double (&opacity)[4],diagnostic* diag=nullptr) {
    double f[4][max_order];
    for(int s=0;s<4;++s) {
        opacity[s]=0;
        if(!decay && s<2) continue;
        if(!reconstruct(thermal.g,n[s],J[s],f[s],diag)) {
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
        add(evaluate_decay(*decay,0,f[2],f[1]),2,1);
        add(evaluate_decay(*decay,1,f[3],f[0]),3,0);
    }
    return true;
}
} // namespace grace::pairs
#endif
