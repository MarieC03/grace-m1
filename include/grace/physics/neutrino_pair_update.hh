/** Coupled local M1 pair update. Fixed matter during an IMEX diagonal stage. */
#ifndef GRACE_NEUTRINO_PAIR_UPDATE_HH
#define GRACE_NEUTRINO_PAIR_UPDATE_HH
#include <grace/physics/neutrino_pair_collision.hh>
#include <grace/physics/m1_helpers.hh>
#include <cstdint>

namespace grace::pairs {

// Necessary continuum bounds in both frames. Discrete quadrature support
// remains a separate reconstruction check.
KOKKOS_INLINE_FUNCTION bool fermi_moments_admissible(
    m1_prims_array_t const& p,metric_array_t const& metric,double multiplicity) {
    double const E=p[ERADL], N=p[NRADL];
    if(!(E>=0 && N>=0) || !Kokkos::isfinite(E) || !Kokkos::isfinite(N)) return false;
    if(metric.square_covec({p[FXL],p[FYL],p[FZL]})>E*E*(1+1e-12)) return false;
    if(N==0) return E==0;
    double const nlab=N/(number_unit*multiplicity);
    if(E/(energy_unit*multiplicity)<minimum_fermi_energy_density(nlab)*(1-1e-12)) return false;
    m1_closure_t cl{p,metric}; cl.update_closure(0);
    double const n=nlab/cl.Gamma;
    return Kokkos::isfinite(cl.J) && cl.J/(energy_unit*multiplicity)>=minimum_fermi_energy_density(n)*(1-1e-12);
}

// Residual tolerances are relative to max(1,|x|): floor radiation relaxing onto a hot LTE puts the
// scaled unknowns at ~1e4, where roundoff in the stiff source (h*kappa ~ 1e3) floors the residual
// near 1e-9 and an absolute 2e-10 target can never be met.
constexpr double newton_target=2e-10, newton_accept=2e-8;
template<int N>
KOKKOS_INLINE_FUNCTION double residual_scale(double const (&x)[N]) {
    double m=1; for(int i=0;i<N;++i) m=Kokkos::fmax(m,Kokkos::fabs(x[i])); return m;
}

// Small, scaled, damped Newton solve; no allocation, exceptions, or vendor API.
template<int N, class Function, class Prepare, class Probe>
KOKKOS_INLINE_FUNCTION bool newton_blocks(Function const& fun, Prepare const& prepare,
    Probe const& probe, double (&x)[N], double& norm) {
    for(int it=0;it<60;++it) {
        double r[N], a[N][N], d[N];
        if(!fun(x,r)) return false;
        norm=0;
        for(int i=0;i<N;++i) norm=Kokkos::fmax(norm,Kokkos::fabs(r[i]));
        if(norm<newton_target*residual_scale(x)) return true;
        prepare();
        for(int j=0;j<N;++j) {
            double h=1e-5*Kokkos::fmax(Kokkos::fabs(x[j]),1e-4), rp[N];
            bool ok=false;
            double const save=x[j];
            for(int k=0;k<12 && !ok;++k) {
                x[j]=save+h; ok=probe(x,rp);
                if(!ok) { h=-h; x[j]=save+h; ok=probe(x,rp); }
                if(!ok) h=Kokkos::fabs(h)*0.1;
            }
            x[j]=save;
            if(!ok) return false;
            for(int i=0;i<N;++i) a[i][j]=(rp[i]-r[i])/h;
        }
        for(int i=0;i<N;++i) d[i]=-r[i];
        for(int j=0;j<N;++j) {
            int p=j;
            for(int i=j+1;i<N;++i) if(Kokkos::fabs(a[i][j])>Kokkos::fabs(a[p][j])) p=i;
            if(!(Kokkos::fabs(a[p][j])>1e-25)) return false;
            if(p!=j) {
                for(int k=j;k<N;++k) { double t=a[j][k]; a[j][k]=a[p][k]; a[p][k]=t; }
                double t=d[j]; d[j]=d[p]; d[p]=t;
            }
            for(int i=j+1;i<N;++i) {
                double const v=a[i][j]/a[j][j];
                for(int k=j+1;k<N;++k) a[i][k]-=v*a[j][k];
                d[i]-=v*d[j];
            }
        }
        for(int i=N-1;i>=0;--i) {
            for(int j=i+1;j<N;++j) d[i]-=a[i][j]*d[j];
            d[i]/=a[i][i];
        }
        bool ok=false;
        for(double t=1;t>1e-10;t*=0.5) {
            double xp[N], rp[N], np=0;
            for(int i=0;i<N;++i) xp[i]=x[i]+t*d[i];
            if(!fun(xp,rp)) continue;
            for(int i=0;i<N;++i) np=Kokkos::fmax(np,Kokkos::fabs(rp[i]));
            if(np<norm) { for(int i=0;i<N;++i) x[i]=xp[i]; ok=true; break; }
        }
        // A stalled line search below the acceptance tolerance is convergence to the roundoff floor.
        if(!ok) return norm<newton_accept*residual_scale(x);
    }
    return false;
}

template<int N, class Function>
KOKKOS_INLINE_FUNCTION bool newton(Function const& fun,double (&x)[N],double& norm) {
    return newton_blocks<N>(fun,[](){},fun,x,norm);
}

// Per-species trial data for one local solve. Grid, metric, base and EAS
// remain fixed for the lifetime of the cache, including during continuation.
// A finite-difference perturbation of the other species can reuse this data.
struct species_trial_cache {
    bool valid=false;
    std::uint64_t moments[5];
    double f[max_order], shape[2], ordinary[5];
    double J, W, E, vdotF;
    m1_closure_t::vec_t vD, HD;

    KOKKOS_INLINE_FUNCTION bool update(
        grid const& g, metric_array_t const& metric, m1_prims_array_t const& base,
        m1_eas_array_t const& eas, double const* u) {
        bool same=valid;
        for(int v=0;v<5 && same;++v)
            same=moments[v]==Kokkos::bit_cast<std::uint64_t>(u[v]);
        if(same) return true;

        // A failed reconstruction can partially overwrite f. Invalidate
        // before attempting it, so no subsequent probe sees stale data.
        valid=false;
        if(!(u[0]>=0 && u[4]>=0)
           || metric.square_covec({u[1],u[2],u[3]})>u[0]*u[0]*(1+1e-12)) return false;
        m1_closure_t closure{base,metric};
        closure.update_closure(u[0],{u[1],u[2],u[3]},0,true);
        J=closure.J;
        double const n=u[4]/closure.Gamma;
        if(!reconstruct(g,n/number_unit,J/energy_unit,f,nullptr,shape)) return false;
        double energy_momentum[4];
        closure.get_implicit_sources(eas,energy_momentum);
        for(int v=0;v<4;++v) ordinary[v]=energy_momentum[v];
        ordinary[4]=metric.alp()*(eas[ETANL]-eas[KANL]*n);
        W=closure.W; E=closure.E; vdotF=closure.vdotF;
        vD=closure.vD; HD=closure.HD;
        for(int v=0;v<5;++v) moments[v]=Kokkos::bit_cast<std::uint64_t>(u[v]);
        valid=true;
        return true;
    }
};

// old/out: undensitized lab-frame E,F_i,N for each individual species.
// Same kernel event enters both N equations; ordinary EAS remains in this
// SAME backward-Euler/IMEX stage, rather than an operator-split pair kick.
KOKKOS_INLINE_FUNCTION bool implicit_update(
    kernel const& k, metric_array_t const& metric,
    m1_prims_array_t const (&base)[2], m1_eas_array_t const (&eas)[2],
    double const (&old)[10], double (&out)[10], double h, double& residual,
    double const* alternate_seed=nullptr)
{
    GRACE_SOURCE_PROFILE(thermal_solve);
    if(h==0) { for(int i=0;i<10;++i) out[i]=old[i]; residual=0; return true; }
    double scale[10], x[10];
    // A nonzero thermal scale makes the vacuum well conditioned.
    double neq=0,jeq=0;
    for(int i=0;i<k.g.n;++i) {
        double const f=k.boltz[i]/(1+k.boltz[i]);
        neq+=k.g.w[i]*f; jeq+=k.g.w[i]*k.g.e[i]*f;
    }
    for(int s=0;s<2;++s) {
        double const es=Kokkos::fmax(old[5*s],jeq*energy_unit*1e-4);
        double const ns=Kokkos::fmax(old[5*s+4],neq*number_unit*1e-4);
        for(int v=0;v<5;++v) {
            scale[5*s+v]=Kokkos::fmax(v==4?ns:es,1e-100);
            x[5*s+v]=old[5*s+v]/scale[5*s+v];
        }
    }
    double stage_h=h;
    species_trial_cache trial[2];
    auto const rhs=[&](double const* u, double* source) {
        for(int s=0;s<2;++s) {
            int const b=5*s;
            if(!trial[s].update(k.g,metric,base[s],eas[s],u+b)) return false;
            for(int v=0;v<5;++v) source[b+v]=trial[s].ordinary[v];
        }
        auto const ps=evaluate(k,trial[0].f,trial[1].f);
        for(int s=0;s<2;++s) {
            int const b=5*s;
            auto const& cl=trial[s];
            double const q=ps.energy[s]*energy_unit/time_unit;
            double const lambda=cl.J>0 ? ps.loss[s]*energy_unit/(time_unit*cl.J) : 0;
            // S^alpha = q u^alpha - lambda H^alpha (isotropic R0 model).
            source[b]+=metric.alp()*cl.W*(q-lambda*(cl.E-cl.vdotF-cl.J));
            for(int v=0;v<3;++v)
                source[b+1+v]+=metric.alp()*(cl.W*cl.vD[v]*q-lambda*cl.HD[v]);
            source[b+4]+=metric.alp()*ps.number*number_unit/time_unit;
        }
        return true;
    };
    auto const fun=[&](double const* xx, double* r) {
        double u[10], s[10];
        for(int i=0;i<10;++i) u[i]=xx[i]*scale[i];
        if(!rhs(u,s)) return false;
        for(int i=0;i<10;++i) {
            r[i]=(u[i]-old[i]-stage_h*s[i])/scale[i];
            if(!Kokkos::isfinite(r[i])) return false;
        }
        return true;
    };
    // Seed exact vacuum with a small FD field (an initial guess only).
    for(int s=0;s<2;++s) if(old[5*s]==0 && old[5*s+4]==0) {
        x[5*s]=jeq*energy_unit*1e-8/scale[5*s];
        x[5*s+4]=neq*number_unit*1e-8/scale[5*s+4];
    }
    bool solved=newton<10>(fun,x,residual);
    if(!solved && alternate_seed) {
        for(int a=0;a<10;++a) x[a]=alternate_seed[a]/scale[a];
        solved=newton<10>(fun,x,residual);
    }
    if(!solved) {
        // Continuation changes the nonlinear problem, NOT the time integrator:
        // every residual still references the original old state.
        for(int i=0;i<10;++i) x[i]=old[i]/scale[i];
        for(int s=0;s<2;++s) if(old[5*s]==0 && old[5*s+4]==0) {
            x[5*s]=jeq*energy_unit*1e-8/scale[5*s];
            x[5*s+4]=neq*number_unit*1e-8/scale[5*s+4];
        }
        for(int step=0;step<=8;++step) {
            stage_h=h*Kokkos::pow(2.0,step-8);
            if(!newton<10>(fun,x,residual)) return false;
        }
    }
    for(int i=0;i<10;++i) out[i]=x[i]*scale[i];
    // Project the two solved number equations onto their common pair increment.
    // Do not reapply h*S_N explicitly: that would amplify the final residual
    // by the stiffness. Recheck the FULL nonlinear residual after projection.
    double d[2], b[2], common=0;
    for(int s=0;s<2;++s) {
        m1_closure_t cl{base[s],metric};
        cl.update_closure(out[5*s],{out[5*s+1],out[5*s+2],out[5*s+3]},0,true);
        d[s]=1+h*metric.alp()*eas[s][KANL]/cl.Gamma;
        b[s]=old[5*s+4]+h*metric.alp()*eas[s][ETANL];
        common+=0.5*(d[s]*out[5*s+4]-b[s]);
    }
    for(int s=0;s<2;++s) out[5*s+4]=(b[s]+common)/d[s];
    for(int i=0;i<10;++i) x[i]=out[i]/scale[i];
    double r[10]; if(!fun(x,r)) return false;
    residual=0;
    for(int i=0;i<10;++i) residual=Kokkos::fmax(residual,Kokkos::fabs(r[i]));
    return residual<newton_accept*residual_scale(x);
}
} // namespace grace::pairs
#endif
