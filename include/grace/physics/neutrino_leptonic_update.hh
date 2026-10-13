/** @file Coupled nue, antinue, numu, antinumu sources at fixed matter. */
#ifndef GRACE_NEUTRINO_LEPTONIC_UPDATE_HH
#define GRACE_NEUTRINO_LEPTONIC_UPDATE_HH
#include <grace/physics/neutrino_pair_update.hh>
#include <grace/physics/neutrino_muon_decay.hh>

namespace grace::pairs {
KOKKOS_INLINE_FUNCTION bool leptonic_implicit_update(
    kernel const& thermal,decay_kernel const& decay,bool thermal_active,
    metric_array_t const& metric,m1_prims_array_t const (&base)[4],
    m1_eas_array_t const (&eas)[4],double const (&old)[20],double (&out)[20],
    double h,double& residual) {
    if(h==0) { for(int i=0;i<20;++i) out[i]=old[i]; residual=0; return true; }
    double neq=0,jeq=0,scale[20],x[20];
    for(int i=0;i<thermal.g.n;++i) {
        double const f=occupation(thermal.g.e[i]/decay.T);
        neq+=thermal.g.w[i]*f; jeq+=thermal.g.w[i]*thermal.g.e[i]*f;
    }
    for(int s=0;s<4;++s) for(int v=0;v<5;++v) {
        int const a=5*s+v;
        scale[a]=Kokkos::fmax(1e-100,Kokkos::fmax(old[5*s+(v==4?4:0)],
            1e-4*(v==4?neq*number_unit:jeq*energy_unit)));
        x[a]=old[a]/scale[a];
    }
    double stage_h=h;
    auto const rhs=[&](double const* u,double* source) {
        m1_closure_t cl[4]={{base[0],metric},{base[1],metric},{base[2],metric},{base[3],metric}};
        double f[4][max_order],J[4];
        for(int s=0;s<4;++s) {
            int const b=5*s;
            if(!(u[b]>=0 && u[b+4]>=0) ||
                metric.square_covec({u[b+1],u[b+2],u[b+3]})>u[b]*u[b]*(1+1e-12)) return false;
            cl[s].update_closure(u[b],{u[b+1],u[b+2],u[b+3]},0,true);
            J[s]=cl[s].J;
            double const n=u[b+4]/cl[s].Gamma;
            if(!reconstruct(thermal.g,n/number_unit,J[s]/energy_unit,f[s])) return false;
            double ordinary[4]; cl[s].get_implicit_sources(eas[s],ordinary);
            for(int v=0;v<4;++v) source[b+v]=ordinary[v];
            source[b+4]=metric.alp()*(eas[s][ETANL]-eas[s][KANL]*n);
        }
        auto const add=[&](sources const& ps,int a,int b) {
            for(int side=0;side<2;++side) {
                int const s=side==0?a:b;
                double const q=ps.energy[side]*energy_unit/time_unit;
                double const lambda=J[s]>0?ps.loss[side]*energy_unit/(time_unit*J[s]):0;
                source[5*s]+=metric.alp()*cl[s].W*(q-lambda*(cl[s].E-cl[s].vdotF-cl[s].J));
                for(int v=0;v<3;++v)
                    source[5*s+1+v]+=metric.alp()*(cl[s].W*cl[s].vD[v]*q-lambda*cl[s].HD[v]);
                source[5*s+4]+=metric.alp()*ps.number*number_unit/time_unit;
            }
        };
        if(thermal_active) add(evaluate(thermal,f[2],f[3]),2,3);
        add(evaluate_decay(decay,0,f[2],f[1]),2,1);
        add(evaluate_decay(decay,1,f[3],f[0]),3,0);
        return true;
    };
    auto const fun=[&](double const* xx,double* r) {
        double u[20],source[20];
        for(int a=0;a<20;++a) u[a]=xx[a]*scale[a];
        if(!rhs(u,source)) return false;
        for(int a=0;a<20;++a) {
            r[a]=(u[a]-old[a]-stage_h*source[a])/scale[a];
            if(!Kokkos::isfinite(r[a])) return false;
        }
        return true;
    };
    auto const seed=[&]() {
        for(int i=0;i<20;++i) x[i]=old[i]/scale[i];
        for(int s=0;s<4;++s) if(old[5*s]==0 && old[5*s+4]==0) {
            x[5*s]=jeq*energy_unit*1e-8/scale[5*s];
            x[5*s+4]=neq*number_unit*1e-8/scale[5*s+4];
        }
    };
    seed();
    if(!newton<20>(fun,x,residual)) {
        seed();
        for(int step=0;step<=10;++step) {
            stage_h=h*Kokkos::pow(2.0,step-10);
            if(!newton<20>(fun,x,residual)) return false;
        }
    }
    for(int a=0;a<20;++a) out[a]=x[a]*scale[a];
    // Project only roundoff/Newton error onto the reaction stoichiometry.
    // Do not compute h*net_rate explicitly again in a stiff problem.
    double d[4],b[4],q[4];
    for(int s=0;s<4;++s) {
        m1_closure_t cl{base[s],metric};
        cl.update_closure(out[5*s],{out[5*s+1],out[5*s+2],out[5*s+3]},0,true);
        d[s]=1+h*metric.alp()*eas[s][KANL]/cl.Gamma;
        b[s]=old[5*s+4]+h*metric.alp()*eas[s][ETANL];
        q[s]=d[s]*out[5*s+4]-b[s];
    }
    if(thermal_active) {
        double const defect=(q[0]-q[1]+q[2]-q[3])/4;
        q[0]-=defect; q[1]+=defect; q[2]-=defect; q[3]+=defect;
    } else {
        q[0]=q[3]=0.5*(q[0]+q[3]);
        q[1]=q[2]=0.5*(q[1]+q[2]);
    }
    for(int s=0;s<4;++s) out[5*s+4]=(b[s]+q[s])/d[s];
    for(int a=0;a<20;++a) x[a]=out[a]/scale[a];
    double r[20]; if(!fun(x,r)) return false;
    residual=0;
    for(int a=0;a<20;++a) residual=Kokkos::fmax(residual,Kokkos::fabs(r[a]));
    return residual<2e-8;
}
} // namespace grace::pairs
#endif
