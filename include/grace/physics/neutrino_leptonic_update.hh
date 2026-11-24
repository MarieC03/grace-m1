/** @file Coupled nue, antinue, numu, antinumu sources at fixed matter. */
#ifndef GRACE_NEUTRINO_LEPTONIC_UPDATE_HH
#define GRACE_NEUTRINO_LEPTONIC_UPDATE_HH
#include <grace/physics/neutrino_pair_update.hh>
#include <grace/physics/neutrino_muon_decay.hh>

namespace grace::pairs {
struct leptonic_diagnostic {
    // phase 0 success, 1 grid, 2 main Newton, 3 continuation Newton,
    // 4 projected residual evaluation, 5 projected residual tolerance.
    int phase=0,species=-1,continuation=-1;
    diagnostic reconstruction{};
    newton_diagnostic nonlinear{};
};
KOKKOS_INLINE_FUNCTION bool same_collision_grid(grid const& a,grid const& b) {
    if(a.n!=b.n) return false;
    for(int i=0;i<a.n;++i) if(a.e[i]!=b.e[i] || a.w[i]!=b.w[i]) return false;
    return true;
}
// Project the signed number equations in their Newton scaling. Equal
// absolute corrections would transfer roundoff from a populated species
// into a nearly empty one. Since q_s=d_s*N_s-b_s, weight d_s*scale(N_s)
// makes every induced correction delta(N_s)/scale(N_s) equally small.
KOKKOS_INLINE_FUNCTION bool project_leptonic_number_increments(
    double (&q)[4],double const (&weight)[4],bool connected_pairs,
    double const* lower=nullptr,double const* upper=nullptr) {
    if(connected_pairs) {
        double largest=0;for(int s=0;s<4;++s)largest=Kokkos::fmax(largest,weight[s]);
        double w[4];bool active[4]={true,true,true,true};
        for(int s=0;s<4;++s)w[s]=weight[s]/largest;
        // At most four active-set eliminations. A species at a Fermi bound
        // cannot absorb a correction toward the forbidden side; redistribute
        // that part to the remaining species without changing the invariant.
        for(int pass=0;pass<5;++pass) {
            double const defect=q[0]-q[1]+q[2]-q[3];
            if(defect==0) return true;
            double sum=0;for(int s=0;s<4;++s)if(active[s])sum+=w[s];
            if(!(sum>0)) return false;
            double candidate[4];bool clipped=false;
            for(int s=0;s<4;++s) {
                candidate[s]=q[s];if(!active[s])continue;
                candidate[s]-=(s==0||s==2?1:-1)*defect*(w[s]/sum);
                if(lower && candidate[s]<lower[s]) {
                    q[s]=lower[s];active[s]=false;clipped=true;
                } else if(upper && candidate[s]>upper[s]) {
                    q[s]=upper[s];active[s]=false;clipped=true;
                }
            }
            if(!clipped) {for(int s=0;s<4;++s)q[s]=candidate[s];return true;}
        }
        return false;
    } else {
        // Independent decay channels require q_nue=q_antinumu and
        // q_antinue=q_numu, with the same scaling within each channel.
        for(int a=0;a<2;++a) {
            int const b=3-a;
            double const largest=Kokkos::fmax(weight[a],weight[b]);
            double const wa=weight[a]/largest,wb=weight[b]/largest;
            double common=(wb*q[a]+wa*q[b])/(wa+wb);
            if(lower && upper) {
                double const lo=Kokkos::fmax(lower[a],lower[b]),hi=Kokkos::fmin(upper[a],upper[b]);
                if(lo>hi)return false;
                common=Kokkos::fmax(lo,Kokkos::fmin(hi,common));
            }
            q[a]=q[b]=common;
        }
    }
    return true;
}

// Positive-temperature discrete FD spectra at fixed J range from a constant
// occupation (T -> infinity) to filled low-energy bins (T -> zero).
KOKKOS_INLINE_FUNCTION void discrete_number_bounds(grid const& g,double J,double& nmin,double& nmax) {
    double total_n=0,total_J=0,remaining=J;nmax=0;
    for(int i=0;i<g.n;++i) {
        total_n+=g.w[i];total_J+=g.w[i]*g.e[i];
        double const energy=Kokkos::fmax(0.0,Kokkos::fmin(remaining,g.w[i]*g.e[i]));
        nmax+=energy/g.e[i];remaining-=energy;
    }
    nmin=J*(total_n/total_J);
}
KOKKOS_INLINE_FUNCTION bool leptonic_implicit_update(
    kernel const& thermal,decay_kernel const& decay,bool thermal_active,
    metric_array_t const& metric,m1_prims_array_t const (&base)[4],
    m1_eas_array_t const (&eas)[4],double const (&old)[20],double (&out)[20],
    double h,double& residual,double const* alternate_seed=nullptr,bool prefer_seed=false,bool block_jacobian=false,kernel const* electron=nullptr,
    leptonic_diagnostic* trace=nullptr) {
    GRACE_SOURCE_PROFILE(coupled);
    if(trace) { *trace={};trace->phase=1; }
    if(electron && !same_collision_grid(thermal.g,electron->g)) return false;
    if(h==0) { for(int i=0;i<20;++i) out[i]=old[i]; residual=0; if(trace)trace->phase=0; return true; }
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
    species_trial_cache trial[4];
    collision_block blocks[4];
    bool probing=false;
    auto const rhs=[&](double const* u,double* source) {
        for(int s=0;s<4;++s) {
            int const b=5*s;
            if(!trial[s].update(thermal.g,metric,base[s],eas[s],u+b,trace?&trace->reconstruction:nullptr)) {
                if(trace)trace->species=s;
                return false;
            }
            for(int v=0;v<5;++v) source[b+v]=trial[s].ordinary[v];
        }
        auto const add=[&](sources const& ps,int a,int b) {
            for(int side=0;side<2;++side) {
                int const s=side==0?a:b;
                auto const& cl=trial[s];
                double const q=ps.energy[side]*energy_unit/time_unit;
                double const lambda=cl.J>0?ps.loss[side]*energy_unit/(time_unit*cl.J):0;
                source[5*s]+=metric.alp()*cl.W*(q-lambda*(cl.E-cl.vdotF-cl.J));
                for(int v=0;v<3;++v)
                    source[5*s+1+v]+=metric.alp()*(cl.W*cl.vD[v]*q-lambda*cl.HD[v]);
                source[5*s+4]+=metric.alp()*ps.number*number_unit/time_unit;
            }
        };
        if(thermal_active) add(probing?blocks[0].evaluate(thermal.g,trial[2].f,trial[3].f)
                                      :evaluate(thermal,trial[2].f,trial[3].f),2,3);
        if(electron) add(probing?blocks[3].evaluate(thermal.g,trial[0].f,trial[1].f)
                                      :evaluate(*electron,trial[0].f,trial[1].f),0,1);
        add(probing && !decay.grey?blocks[1].evaluate(thermal.g,trial[2].f,trial[1].f)
                   :evaluate_decay(decay,0,trial[2].f,trial[1].f,trial[2].shape,trial[1].shape),2,1);
        add(probing && !decay.grey?blocks[2].evaluate(thermal.g,trial[3].f,trial[0].f)
                   :evaluate_decay(decay,1,trial[3].f,trial[0].f,trial[3].shape,trial[0].shape),3,0);
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
    auto const prepare=[&]() {
        #ifdef GRACE_LEPTON_PROFILE
        source_profile::block_builds.fetch_add(1,std::memory_order_relaxed);
        #endif
        if(thermal_active) blocks[0].init(thermal.g,trial[2].f,trial[3].f,
            evaluate(thermal,trial[2].f,trial[3].f),[&](int i,int j,double& p,double& a) {
                a=thermal.a[i][j]; p=a*thermal.boltz[i]*thermal.boltz[j];
            });
        if(electron) blocks[3].init(thermal.g,trial[0].f,trial[1].f,
            evaluate(*electron,trial[0].f,trial[1].f),[&](int i,int j,double& p,double& a) {
                a=electron->a[i][j]; p=a*electron->boltz[i]*electron->boltz[j];
            });
        for(int c=0;!decay.grey && c<2;++c) {
            int const a=c==0?2:3,b=c==0?1:0;
            blocks[c+1].init(thermal.g,trial[a].f,trial[b].f,
                evaluate_decay(decay,c,trial[a].f,trial[b].f,trial[a].shape,trial[b].shape),
                [&](int i,int j,double& p,double& a) { decay.rates(c,i,j,p,a); });
        }
    };
    auto const probe=[&](double const* xx,double* r) {
        probing=true;
        bool const ok=fun(xx,r);
        probing=false;
        return ok;
    };
    auto const solve=[&]() {
        if(!block_jacobian) return newton<20>(fun,x,residual,trace?&trace->nonlinear:nullptr);
        // Failed accelerated solves retain the same full-residual numerical
        // fallback, including the caller's seed and continuation strategies.
        #ifdef GRACE_LEPTON_PROFILE
        source_profile::block_solves.fetch_add(1,std::memory_order_relaxed);
        #endif
        if(newton_blocks<20>(fun,prepare,probe,x,residual,trace?&trace->nonlinear:nullptr)) return true;
        #ifdef GRACE_LEPTON_PROFILE
        source_profile::block_fallbacks.fetch_add(1,std::memory_order_relaxed);
        #endif
        return newton<20>(fun,x,residual,trace?&trace->nonlinear:nullptr);
    };
    auto const seed=[&](double const* guess) {
        for(int i=0;i<20;++i) x[i]=guess[i]/scale[i];
        for(int s=0;s<4;++s) if(guess[5*s]==0 && guess[5*s+4]==0) {
            x[5*s]=jeq*energy_unit*1e-8/scale[5*s];
            x[5*s+4]=neq*number_unit*1e-8/scale[5*s+4];
        }
    };
    seed(prefer_seed && alternate_seed ? alternate_seed : old);
    if(trace)trace->phase=2;
    bool solved=solve();
    if(!solved && alternate_seed) {
        // The ARS232 stage RHS can lie outside the realizable set even
        // when its full implicit solution is admissible. Use the previous
        // stage only to start Newton; fun still subtracts the original RHS.
        seed(alternate_seed);
        solved=solve();
    }
    if(!solved) {
        seed(old);
        for(int step=0;step<=10;++step) {
            if(trace) {trace->phase=3;trace->continuation=step;}
            stage_h=h*Kokkos::pow(2.0,step-10);
            if(!solve()) return false;
        }
    }
    for(int a=0;a<20;++a) out[a]=x[a]*scale[a];
    // Project only roundoff/Newton error onto the reaction stoichiometry.
    // Do not compute h*net_rate explicitly again in a stiff problem.
    double d[4],b[4],q[4],projection_weight[4],lower[4],upper[4];
    for(int s=0;s<4;++s) {
        m1_closure_t cl{base[s],metric};
        cl.update_closure(out[5*s],{out[5*s+1],out[5*s+2],out[5*s+3]},0,true);
        d[s]=1+h*metric.alp()*eas[s][KANL]/cl.Gamma;
        b[s]=old[5*s+4]+h*metric.alp()*eas[s][ETANL];
        q[s]=d[s]*out[5*s+4]-b[s];
        projection_weight[s]=d[s]*scale[5*s+4];
        double nmin,nmax;
        discrete_number_bounds(thermal.g,cl.J/energy_unit,nmin,nmax);
        double const Nmin=nmin*number_unit*cl.Gamma,Nmax=nmax*number_unit*cl.Gamma;
        // Keep the already reconstructed input admissible to its existing
        // roundoff tolerance, but never push it farther beyond a moment bound.
        lower[s]=q[s]+d[s]*Kokkos::fmin(0.0,Nmin-out[5*s+4]);
        upper[s]=q[s]+d[s]*Kokkos::fmax(0.0,Nmax-out[5*s+4]);
    }
    if(!project_leptonic_number_increments(q,projection_weight,thermal_active || electron,lower,upper)) {
        if(trace)trace->phase=6;
        return false;
    }
    for(int s=0;s<4;++s) out[5*s+4]=(b[s]+q[s])/d[s];
    for(int a=0;a<20;++a) x[a]=out[a]/scale[a];
    if(trace)trace->phase=4;
    double r[20]; if(!fun(x,r)) return false;
    residual=0;
    for(int a=0;a<20;++a) residual=Kokkos::fmax(residual,Kokkos::fabs(r[a]));
    bool const accepted=residual<newton_accept*residual_scale(x);
    if(trace)trace->phase=accepted?0:5;
    return accepted;
}
// Lagged spectra give independent five-moment solves. The coupled residual
// and stoichiometric projection remain the acceptance test. Stiff cells that
// do not converge use the reference solve, so lagging cannot hide charge error.
KOKKOS_INLINE_FUNCTION bool leptonic_lagged_update(
    kernel const& thermal,decay_kernel const& decay,bool thermal_active,
    metric_array_t const& metric,m1_prims_array_t const (&base)[4],
    m1_eas_array_t const (&eas)[4],double const (&old)[20],double (&out)[20],
    double h,double& residual,double const* alternate_seed=nullptr,
    int* iterations=nullptr,bool* fallback=nullptr,bool block_jacobian=false,kernel const* electron=nullptr) {
    GRACE_SOURCE_PROFILE(lagged);
    if(electron && !same_collision_grid(thermal.g,electron->g)) return false;
    if(fallback) *fallback=false;
    double current[20];
    for(int a=0;a<20;++a) current[a]=alternate_seed ? alternate_seed[a] : old[a];
    for(int iteration=0;iteration<12;++iteration) {
        #ifdef GRACE_LEPTON_PROFILE
        source_profile::lagged_iterations.fetch_add(1,std::memory_order_relaxed);
        #endif
        if(iterations) *iterations=iteration+1;
        species_trial_cache trial[4];
        double n[4],J[4];
        bool valid=true;
        for(int s=0;s<4;++s) {
            valid=valid && trial[s].update(thermal.g,metric,base[s],eas[s],current+5*s);
            if(!valid) break;
            m1_closure_t cl{base[s],metric};
            cl.update_closure(current[5*s],{current[5*s+1],current[5*s+2],current[5*s+3]},0,true);
            n[s]=current[5*s+4]/cl.Gamma/number_unit;
            J[s]=cl.J/energy_unit;
            if(!(n[s]>0 && J[s]>0)) valid=false;
        }
        if(!valid) break;
        m1_eas_array_t lagged[4];
        for(int s=0;s<4;++s) lagged[s]=eas[s];
        auto const add=[&](sources const& src,int a,int b) {
            for(int side=0;side<2;++side) {
                int const s=side==0?a:b;
                lagged[s][ETAL]+=src.emission[side]*energy_unit/time_unit;
                lagged[s][ETANL]+=src.emission_n[side]*number_unit/time_unit;
                lagged[s][KAL]+=src.loss[side]/J[s]/time_unit;
                lagged[s][KANL]+=src.loss_n[side]/n[s]/time_unit;
            }
        };
        if(thermal_active) add(evaluate(thermal,trial[2].f,trial[3].f),2,3);
        if(electron) add(evaluate(*electron,trial[0].f,trial[1].f),0,1);
        add(evaluate_decay(decay,0,trial[2].f,trial[1].f,trial[2].shape,trial[1].shape),2,1);
        add(evaluate_decay(decay,1,trial[3].f,trial[0].f,trial[3].shape,trial[0].shape),3,0);
        double change=0;
        for(int s=0;s<4 && valid;++s) {
            double scale[5],x[5];
            for(int v=0;v<5;++v) {
                scale[v]=Kokkos::fmax(1e-100,current[5*s+(v==4?4:0)]);
                x[v]=current[5*s+v]/scale[v];
            }
            auto const fun=[&](double const* xx,double* r) {
                double u[5]; for(int v=0;v<5;++v) u[v]=xx[v]*scale[v];
                if(!(u[0]>0 && u[4]>0) || metric.square_covec({u[1],u[2],u[3]})>u[0]*u[0]) return false;
                m1_closure_t cl{base[s],metric};
                cl.update_closure(u[0],{u[1],u[2],u[3]},0,true);
                double src[4]; cl.get_implicit_sources(lagged[s],src);
                for(int v=0;v<4;++v) r[v]=(u[v]-old[5*s+v]-h*src[v])/scale[v];
                double const number_source=metric.alp()*(lagged[s][ETANL]-lagged[s][KANL]*u[4]/cl.Gamma);
                r[4]=(u[4]-old[5*s+4]-h*number_source)/scale[4];
                return true;
            };
            valid=newton<5>(fun,x,residual);
            if(valid) for(int v=0;v<5;++v) {
                double const next=x[v]*scale[v];
                change=Kokkos::fmax(change,Kokkos::fabs(next-current[5*s+v])/scale[v]);
                current[5*s+v]=next;
            }
        }
        if(!valid) break;
        if(change<1e-9) {
            if(leptonic_implicit_update(thermal,decay,thermal_active,metric,base,eas,old,out,h,residual,current,true,block_jacobian,electron))
                return true;
            // A stationary frozen-coefficient iterate need not be a usable
            // coupled Newton seed. Retry the original reference path too.
            break;
        }
    }
    if(fallback) *fallback=true;
    #ifdef GRACE_LEPTON_PROFILE
    source_profile::lagged_fallbacks.fetch_add(1,std::memory_order_relaxed);
    #endif
    return leptonic_implicit_update(thermal,decay,thermal_active,metric,base,eas,old,out,h,residual,alternate_seed,false,block_jacobian,electron);
}
// One frozen-partner decay update, with no outer iteration, 20-variable
// acceptance solve, or stoichiometric projection. Thermal pairs keep their
// existing conservative two-species solver. Grey opacity weighting uses the
// previous-stage target spectrum as well as the previous-stage partner.
KOKKOS_INLINE_FUNCTION bool leptonic_singlepass_update(
    kernel const& thermal,decay_kernel const& decay,bool thermal_active,
    metric_array_t const& metric,m1_prims_array_t const (&base)[4],
    m1_eas_array_t const (&eas)[4],double const (&old)[20],double (&out)[20],
    double h,double& residual,double const* previous=nullptr,kernel const* electron=nullptr,int* failure_stage=nullptr) {
    GRACE_SOURCE_PROFILE(singlepass);
    if(failure_stage) *failure_stage=1;
    if(electron && !same_collision_grid(thermal.g,electron->g)) return false;
    if(h==0) { for(int a=0;a<20;++a) out[a]=old[a]; residual=0; return true; }
    double const* lag=previous?previous:old;
    double f[4][max_order],shape[4][max_order],sn[4]{},sj[4]{};
    for(int s=0;s<4;++s) {
        m1_closure_t cl{base[s],metric};
        cl.update_closure(lag[5*s],{lag[5*s+1],lag[5*s+2],lag[5*s+3]},0,true);
        if(!reconstruct(thermal.g,lag[5*s+4]/cl.Gamma/number_unit,cl.J/energy_unit,f[s])) return false;
        for(int i=0;i<thermal.g.n;++i) {
            shape[s][i]=lag[5*s+4]>0?f[s][i]:occupation(thermal.g.e[i]/decay.T);
            sn[s]+=thermal.g.w[i]*shape[s][i]; sj[s]+=thermal.g.w[i]*thermal.g.e[i]*shape[s][i];
        }
        if(!(sn[s]>0 && sj[s]>0)) return false;
    }
    m1_eas_array_t frozen[4];
    for(int s=0;s<4;++s) frozen[s]=eas[s];
    for(int charge=0;charge<2;++charge) {
        int const a=charge==0?2:3,b=charge==0?1:0;
        double jn[2]{},je[2]{},kn[2]{},ke[2]{};
        for(int i=0;i<thermal.g.n;++i) for(int j=0;j<thermal.g.n;++j) {
            double p,k; decay.rates(charge,i,j,p,k);
            double const w=thermal.g.w[i]*thermal.g.w[j];
            double const emit[2]={p*(1-f[b][j]),p*(1-f[a][i])};
            double const loss[2]={(emit[0]+k*f[b][j])*shape[a][i],(emit[1]+k*f[a][i])*shape[b][j]};
            double const energy[2]={thermal.g.e[i],thermal.g.e[j]};
            for(int side=0;side<2;++side) {
                jn[side]+=w*emit[side]; je[side]+=w*energy[side]*emit[side];
                kn[side]+=w*loss[side]; ke[side]+=w*energy[side]*loss[side];
            }
        }
        for(int side=0;side<2;++side) {
            int const s=side==0?a:b;
            frozen[s][ETAL]+=je[side]*energy_unit/time_unit;
            frozen[s][ETANL]+=jn[side]*number_unit/time_unit;
            frozen[s][KAL]+=ke[side]/sj[s]/time_unit;
            frozen[s][KANL]+=kn[side]/sn[s]/time_unit;
        }
    }
    residual=0;
    for(int s=electron?2:0;s<(thermal_active?2:4);++s) {
        double const scale=Kokkos::fmax(1e-100,Kokkos::fmax(lag[5*s],h*metric.alp()*frozen[s][ETAL]));
        double x[4]; for(int v=0;v<4;++v) x[v]=lag[5*s+v]/scale;
        if(x[0]==0) x[0]=1;
        auto const fun=[&](double const* xx,double* r) {
            if(!(xx[0]>0) || metric.square_covec({xx[1],xx[2],xx[3]})>xx[0]*xx[0]) return false;
            m1_closure_t cl{base[s],metric};
            cl.update_closure(xx[0]*scale,{xx[1]*scale,xx[2]*scale,xx[3]*scale},0,true);
            double src[4]; cl.get_implicit_sources(frozen[s],src);
            for(int v=0;v<4;++v) r[v]=(xx[v]*scale-old[5*s+v]-h*src[v])/scale;
            return true;
        };
        double res=0; if(!newton<4>(fun,x,res)) return false;
        residual=Kokkos::fmax(residual,res);
        for(int v=0;v<4;++v) out[5*s+v]=x[v]*scale;
        m1_closure_t cl{base[s],metric};
        cl.update_closure(out[5*s],{out[5*s+1],out[5*s+2],out[5*s+3]},0,true);
        out[5*s+4]=(old[5*s+4]+h*metric.alp()*frozen[s][ETANL])/(1+h*metric.alp()*frozen[s][KANL]/cl.Gamma);
    }
    if(electron) {
        if(failure_stage) *failure_stage=10;
        m1_prims_array_t p[2]={base[0],base[1]};
        m1_eas_array_t r[2]={frozen[0],frozen[1]};
        double rhs[10],seed[10],result[10],res=0;
        for(int a=0;a<10;++a) { rhs[a]=old[a]; seed[a]=lag[a]; }
        if(!implicit_update(*electron,metric,p,r,rhs,result,h,res,seed)) {residual=res;return false;}
        for(int a=0;a<10;++a) out[a]=result[a];
        residual=Kokkos::fmax(residual,res);
    }
    if(thermal_active) {
        if(failure_stage) *failure_stage=20;
        m1_prims_array_t p[2]={base[2],base[3]};
        m1_eas_array_t r[2]={frozen[2],frozen[3]};
        double rhs[10],seed[10],result[10],res=0;
        for(int a=0;a<10;++a) { rhs[a]=old[a+10]; seed[a]=lag[a+10]; }
        if(!implicit_update(thermal,metric,p,r,rhs,result,h,res,seed)) {residual=res;return false;}
        for(int a=0;a<10;++a) out[a+10]=result[a];
        residual=Kokkos::fmax(residual,res);
    }
    if(failure_stage) *failure_stage=30;
    for(int s=0;s<4;++s) {
        species_trial_cache check;
        if(!check.update(thermal.g,metric,base[s],eas[s],out+5*s)) return false;
    }
    return true;
}
} // namespace grace::pairs
#endif
