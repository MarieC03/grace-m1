/** @file
 * Isotropic, blocked pair collision operator for ONE physical heavy flavour.
 * Energies: MeV; densities: cm^-3; kernels: cm^3/s. See pair_processes.md.
 * No opacity is inferred from Kirchhoff away from equilibrium.
 */
#ifndef GRACE_NEUTRINO_PAIR_COLLISION_HH
#define GRACE_NEUTRINO_PAIR_COLLISION_HH

#include <grace_config.h>
#ifndef GRACE_M1_PAIR_KERNELS
#error "neutrino_pair_collision.hh needs a build with -DGRACE_M1_PAIR_KERNELS=ON"
#endif
#include <Kokkos_Core.hpp>
#include <grace/physics/neutrino_pair_diagnostics.hh>
#ifdef GRACE_HAVE_BNS_NURATES
#include <kernel_pair.hpp>
#include <kernel_brem_HR98.hpp>
#endif

namespace grace::pairs {
constexpr int max_order = 32;
constexpr double pi = 3.14159265358979323846;
constexpr double hc = 1.23984172e-10; // MeV cm, same convention as EAS
constexpr double phase = 4*pi/(hc*hc*hc);
// GRACE's existing energy and baryon-mass-weighted number normalisations.
constexpr double energy_unit = 1.60217733e-6 * 1.61887093132742e-18 * 1.11265005605362e-21;
constexpr double number_unit = (938.27208943*1.602176634e-6/(2.99792458e10*2.99792458e10)) * 1.61887093132742e-18;
constexpr double time_unit = 2.03040204956746e5;

struct material {
    double T, mu_e, nb, yn, yp; // nb in cm^-3, chemical potential includes rest mass
};
struct channels { bool ee, nn, plasmon; };

KOKKOS_INLINE_FUNCTION failure_record make_failure_record(
    material const& m, diagnostic const& diag, long long q, int i, int j=0, int k=0) {
    failure_record r;
    r.detail=diag; r.q=q; r.i=i; r.j=j; r.k=k;
    r.T=m.T; r.mu_e=m.mu_e; r.nb=m.nb; r.yn=m.yn; r.yp=m.yp;
    return r;
}

KOKKOS_INLINE_FUNCTION double occupation(double x) {
    double const z = Kokkos::exp(-Kokkos::fabs(x));
    return x >= 0 ? z/(1+z) : 1/(1+z);
}

// Rational Gauss-Legendre quadrature: epsilon = scale*x/(1-x), x in (0,1).
// Shared nodes are essential: a kernel element is one event for BOTH partners.
struct grid {
    int n;
    double scale;                      // energy scale of the rational map, for diagnostics
    double e[max_order], w[max_order]; // w includes the one-helicity phase space
    KOKKOS_INLINE_FUNCTION void init(int order, double scale) {
        n = order; this->scale = scale;
        for (int i=0; i<n; ++i) {
            double z = Kokkos::cos(pi*(i+0.75)/(n+0.5)), dp=0;
            for (int it=0; it<20; ++it) {
                double p=1, pm=0;
                for (int j=1; j<=n; ++j) {
                    double const pn=((2*j-1)*z*p-(j-1)*pm)/j; pm=p; p=pn;
                }
                dp=n*(z*p-pm)/(z*z-1);
                double const dz=p/dp; z-=dz;
                if (Kokkos::fabs(dz)<2e-15) break;
            }
            double const x=(1-z)/2;
            e[i]=scale*x/(1-x);
            w[i]=phase*e[i]*e[i]*scale/((1-x)*(1-x)*(1-z*z)*dp*dp);
        }
    }
};

// Transverse, constant-mass electron plasmon model. Two polarisations, Z=1,
// omega^2=k^2+omega_p^2, ultrarelativistic electron omega_p. This is NOT the
// full Braaten-Segel/WeakHub plasma kernel (no L, axial or mixed channel).
KOKKOS_INLINE_FUNCTION double plasmon_abs(material const& m, double e, double ep) {
    constexpr double alpha=1/137.035999084, gf=1.1663787e-11;
    constexpr double cv=-0.5+2*0.2325, hbar=6.582119569e-22;
    double const mass2=4*alpha/(3*pi)*(m.mu_e*m.mu_e+pi*pi*m.T*m.T/3);
    double const om=e+ep, k2=om*om-mass2;
    if (!(k2>0) || (e-ep)*(e-ep)>k2) return 0;
    // dP/de = 3(1+cos(theta*)^2)/(4k); k/v_g=omega.
    // R_abs includes stimulated photon production 1+f_BE, NOT 1-f_BE.
    double const bose=1/(-Kokkos::expm1(-om/m.T));
    double const gamma_omega=gf*gf*cv*cv*mass2*mass2*mass2/(48*pi*pi*alpha*hbar);
    double const joint=2*phase*gamma_omega*bose*0.75*(1+(e-ep)*(e-ep)/k2);
    // k/v_g cancels the k in dP/de for the constant-mass dispersion.
    return joint/(phase*phase*e*e*ep*ep);
}

#ifdef GRACE_HAVE_BNS_NURATES
// e+e- absorption kernel from the two Psi_0 orientations: algebraically PairKernels.abs[nux],
// evaluated without exp(+s/T) times an underflowed production kernel.  The oriented Psi terms
// must NOT be symmetrised for separately evolved nu and antinu.
KOKKOS_INLINE_FUNCTION double ee_absorption(material const& m, double e, double ep, double const (&psi)[2]) {
    double const a=kBS_Pair_Alpha1_1, b=kBS_Pair_Alpha2_1;
    return -0.5*kBS_Pair_Phi*m.T*m.T*(a*a*psi[0]+b*b*psi[1])/(-Kokkos::expm1(-(e+ep)/m.T));
}

// PairPsi's 28 Fermi-Dirac integrals at eta +- y_i, eta and (eta +- y_i) +- y_j: the one-index ones are
// shared by the whole matrix, so a build needs ~1700 FDI calls instead of 7168.  Same arguments and
// polynomial in the library's floating-point order, so psi() is bitwise PairPsi(0,y_i,y_j,eta).
struct ee_cache {
    double eta, y[max_order], p3e, p4e;
    double p1m[max_order], p1p[max_order], p2m[max_order], p2p[max_order], p3m[max_order],
           p3p[max_order], p4m[max_order], p4p[max_order], p5m[max_order], p5p[max_order];
    KOKKOS_INLINE_FUNCTION void init(material const& m, grid const& g) {
        eta=m.mu_e/m.T;
        for(int i=0;i<g.n;++i) {
            y[i]=g.e[i]/m.T;
            double const em=eta-y[i], ep=eta+y[i];
            p1m[i]=FDI_p1(em); p1p[i]=FDI_p1(ep); p2m[i]=FDI_p2(em); p2p[i]=FDI_p2(ep);
            p3m[i]=FDI_p3(em); p3p[i]=FDI_p3(ep); p4m[i]=FDI_p4(em); p4p[i]=FDI_p4(ep);
            p5m[i]=FDI_p5(em); p5p[i]=FDI_p5(ep);
        }
        p3e=FDI_p3(eta); p4e=FDI_p4(eta);
    }
    KOKKOS_INLINE_FUNCTION void psi(int i, int j, double (&out)[2]) const {
        double const yy=y[i], z=y[j];
        if(eta-yy-z>200.0) { out[0]=out[1]=0; return; }   // full electron phase space (library guard)
        double const emyz=eta-yy-z, epyz=eta+yy+z;
        double const p3_emyz=FDI_p3(emyz), p3_epyz=FDI_p3(epyz), p4_emyz=FDI_p4(emyz),
                     p4_epyz=FDI_p4(epyz), p5_emyz=FDI_p5(emyz), p5_epyz=FDI_p5(epyz);
        double const x0=20.0*p4m[i], x1=20.0*p4p[j], x2=120.0*p2m[i]-120.0*p2p[j];
        double const x3=-40.0*p3m[i]+40.0*p3p[j], x4=40.0*p3e, x5=40.0*p3_emyz-x4, x6=-20.0*p4e;
        double const x7=20.0*p4_emyz+x6, x8=-40.0*p3_epyz+x4, x9=20.0*p4_epyz+x6;
        double const x10=-4.0*p5m[i]+4.0*p5_emyz-4.0*p5m[j]+4.0*p5p[i]-4.0*p5_epyz+4.0*p5p[j];
        double const x11=20.0*p4p[i], x12=20.0*p4m[j], x13=-40.0*p3m[j]+40.0*p3p[i];
        double const x14=120.0*p2m[j]-120.0*p2p[i];
        double const aux=15.0*((yy*z)*(yy*z));
        out[0]=x10+yy*(-x0+x1+x7+yy*(x3+x5+z*(x2+z*(-120.0*p1m[i]+120.0*p1p[j])))
                      +z*(80.0*p3m[i]-80.0*p3p[j]-x2*z))+z*(x0-x1+x9+z*(x3+x8));
        out[1]=x10+yy*(-x11+x12+x9+yy*(x13+x8))
              +z*(x11-x12+x7+yy*(80.0*p3m[j]-80.0*p3p[i]-x14*yy)
                  +z*(x13+x5+yy*(x14+yy*(-120.0*p1m[j]+120.0*p1p[i]))));
        out[0]/=aux; out[1]/=aux;
    }
};

// Hannestad-Raffelt 1998 NN bremsstrahlung, all channels, in the library's nm^3/s.  Callers
// multiply by 1e-21 in the accumulation itself (result += r*1e-21): the compiler fuses that into one
// multiply-add, and keeping it there keeps the kernel bitwise what it was before the hoisting.
KOKKOS_INLINE_FUNCTION double nn_absorption(material const& m, double e, double ep) {
    MyEOSParams eos{};
    eos.temp=m.T; eos.nb=m.nb*1e-21; eos.yn=m.yn; eos.yp=m.yp;
    BremKernelParams p{e,ep,0,false};
    return BremKernelsLegCoeff(&p,&eos).abs[id_nux];
}

// nn_absorption with everything that does not depend on the element hoisted: the y-only factors once
// per build, the eta*-only ones once per channel (kernel_brem_HR98.hpp, l = 0, no medium correction).
// Every remaining expression keeps the library's association order, so element() is bitwise the library.
struct brem_cell {
    struct channel { double n, eta, gamma, hgb2, pre_sd, f_u, one_eta2, sqrt_eta, eta_02, h; } ch[3];
    double T, yc, exp_y12, y_den, p, mp, inv_p, one_y4, w_np;
    KOKKOS_INLINE_FUNCTION void init(material const& m) {
        T=m.T;
        double const nb=m.nb*1e-21, xn=m.yn, xp=m.yp, x_mean=Kokkos::sqrt(xn*xp), T_10=T*0.1;
        double const y=1.94/T_10;
        yc=(y>kBS_Brem_Ymin)?y:kBS_Brem_Ymin;
        exp_y12=SafeExp(-yc/12.0);
        y_den=0.8+0.06*Kokkos::pow(yc,1.05);
        p=0.67+0.18*Kokkos::pow(yc,0.4); mp=-p; inv_p=-1.0/p;
        one_y4=1.0+POW4(yc);
        double const n_nuc[3]={nb*xn,nb*xp,nb*x_mean}, mass[3]={kBS_MnGrams,kBS_MpGrams,kBS_MAvgGrams};
        w_np=(28.0/3.0)*n_nuc[2];
        for(int c=0;c<3;++c) {
            channel& k=ch[c]; k.n=n_nuc[c];
            double const eta_star=Kokkos::pow(3.0*kBS_PiSquared*n_nuc[c],2.0/3.0)*kBS_Brem_Aux1/(2.0*mass[c]*T);
            k.gamma=1.63*Kokkos::pow(eta_star,1.5)*T_10;
            k.hgb2=POW2(0.5*k.gamma*BremKernelG(y,eta_star));
            k.eta=(eta_star>kBS_Brem_Etamin)?eta_star:kBS_Brem_Etamin;
            double const u=Kokkos::sqrt(yc/(2.0*k.eta))+1e-10, u2=POW2(u), u_arg=u2/(2.0*Kokkos::sqrt(2.0*u2+4.0));
            double const f_u=(1.0-5.0*u*Kokkos::atan(2.0/u)/6.0+u2/(3.0*(u2+4.0))+Kokkos::atan(1.0/u_arg)*u_arg/3.0);
            k.f_u=(Kokkos::fabs(f_u)<1e-14)?1e-14:f_u;
            k.pre_sd=3.0*kBS_PiHalfToFiveHalves*Kokkos::pow(k.eta,-2.5);
            k.one_eta2=1.0+POW2(k.eta);
            k.sqrt_eta=Kokkos::sqrt(k.eta); k.eta_02=k.eta+0.2;
            k.h=0.1*k.eta/(2.39+0.1*Kokkos::pow(k.eta,1.1));
        }
    }
    KOKKOS_INLINE_FUNCTION double element(double e, double ep) const {
        double const x=(e+ep)/T, xc=(x>kBS_Brem_Xmin)?x:kBS_Brem_Xmin;
        // BremKernelS: the x,y-only pieces, shared by the three channels
        double const s_nd=2.0*kBS_SqrtPi*Kokkos::pow(xc+2.0-exp_y12,1.5)*(POW2(xc)+2.0*xc*yc+5.0*POW2(yc)/3.0+1.0)
                          /(kBS_SqrtPi+POW4(kBS_Pi2OneEighth+xc+yc));
        double const sd_x=POW2(xc)+kBS_FourPiSquared, sd_den=kBS_FourPiSquared*(1.0-SafeExp(-xc));
        double const px11=Kokkos::pow(xc,1.1), px093=Kokkos::pow(xc,0.93), px12=Kokkos::pow(xc,1.2), px28=Kokkos::pow(xc,2.8);
        double const f_x=3.0+POW2(xc-1.2)+Kokkos::pow(xc,-4.0), g_x=0.0044*px11*yc/y_den, pow_nd=Kokkos::pow(s_nd,mp);
        double single[3];
        for(int c=0;c<3;++c) {
            channel const& k=ch[c];
            double const s_d=k.pre_sd*sd_x*xc*k.f_u/sd_den;
            double const f_brem=1.0+1.0/(f_x*k.one_eta2*one_y4);
            double const g_brem=1.0-g_x*k.sqrt_eta/k.eta_02;
            double const c_brem=1.1*px11*k.h/(2.3+k.h*px093+0.0001*px12)*30.0/(30.0+0.005*px28);
            double const sb=Kokkos::pow(pow_nd+Kokkos::pow(s_d,mp),inv_p)*f_brem*(1.0+c_brem*g_brem);
            single[c]=k.gamma/(POW2(x)+k.hgb2)*sb/T;
        }
        double const tot=kBS_Brem_Const*(ch[0].n*single[0]+ch[1].n*single[1]+w_np*single[2]);
        return 3.0*tot;   // nm^3/s, like nn_absorption
    }
};
#endif

KOKKOS_INLINE_FUNCTION double absorption_kernel(material const& m, channels c,
                                                 double e, double ep) {
    double result=0;
#ifdef GRACE_HAVE_BNS_NURATES
    if (c.ee) {
        double psi[2]; PairPsi(0,e/m.T,ep/m.T,m.mu_e/m.T,psi);
        double const r=ee_absorption(m,e,ep,psi);
        if(!Kokkos::isfinite(r)) return r; // do not hide NaNs with fmax
        result += Kokkos::fmax(0.0,r)*1e-21; // nm^3/s -> cm^3/s
    }
    if (c.nn && m.nb>0 && (m.yn>0 || m.yp>0)) result += nn_absorption(m,e,ep)*1e-21; // nm^3/s -> cm^3/s
#endif
    if (c.plasmon) result += plasmon_abs(m,e,ep);
    return result;
}

struct kernel {
    grid g;
    double a[max_order][max_order], boltz[max_order];
    KOKKOS_INLINE_FUNCTION bool init(material m, channels c, int order, double scale,
                                     bool symmetric=false, diagnostic* diag=nullptr) {
        if(diag) *diag = diagnostic{};
        if (!(m.T>0 && scale>0 && m.nb>=0 && m.yn>=0 && m.yn<=1 && m.yp>=0 && m.yp<=1)
            || !Kokkos::isfinite(m.mu_e) || !Kokkos::isfinite(m.T)
            || !Kokkos::isfinite(scale) || order<8 || order>max_order) {
            if(diag) diag->code = failure_code::kernel_input;
            return false;
        }
#ifndef GRACE_HAVE_BNS_NURATES
        if(c.ee || c.nn) {
            if(diag) diag->code = failure_code::kernel_backend;
            return false;
        }
#endif
        g.init(order,scale);
        for (int i=0; i<g.n; ++i) boltz[i]=Kokkos::exp(-g.e[i]/m.T);
#ifdef GRACE_HAVE_BNS_NURATES
        ee_cache ee; if(c.ee) ee.init(m,g);
        bool const nn_active=c.nn && m.nb>0 && (m.yn>0 || m.yp>0);
        brem_cell nn; if(nn_active) nn.init(m);
#endif
        // absorption_kernel(m,c,g.e[i],g.e[j]) term by term, from the per-build caches
        auto const element=[&](int i, int j) {
            double result=0;
#ifdef GRACE_HAVE_BNS_NURATES
            if (c.ee) {
                double psi[2]; ee.psi(i,j,psi);
                double const r=ee_absorption(m,g.e[i],g.e[j],psi);
                if(!Kokkos::isfinite(r)) return r;
                result += Kokkos::fmax(0.0,r)*1e-21;
            }
            if (nn_active) result += nn.element(g.e[i],g.e[j])*1e-21;
#endif
            if (c.plasmon) result += plasmon_abs(m,g.e[i],g.e[j]);
            return result;
        };
        for (int i=0; i<g.n; ++i) for (int j=0; j<g.n; ++j) {
            if(symmetric && j<i) { a[i][j]=a[j][i]; continue; }
            double r=element(i,j);
            if (symmetric) r=0.5*(r+element(j,i));
            if (!(r>=0) || !Kokkos::isfinite(r)) {
                if(diag) {
                    diag->code = failure_code::kernel_element;
                    diag->node=i; diag->partner_node=j; diag->kernel_value=r;
                    diag->energy=g.e[i]; diag->partner_energy=g.e[j];
                }
                return false;
            }
            a[i][j]=r;
        }
        return true;
    }
    // In place: the charge-symmetric aggregate kernel 0.5(a_ij + a_ji), bitwise what init(symmetric)
    // gives on the same grid, so one build serves the nu_mu pair and the tau/nu_x aggregate.
    KOKKOS_INLINE_FUNCTION void symmetrize() {
        for(int i=0;i<g.n;++i) for(int j=i+1;j<g.n;++j) {
            double const r=0.5*(a[i][j]+a[j][i]); a[i][j]=a[j][i]=r;
        }
    }
};

// Discrete maximum-entropy FD reconstruction, matching BOTH supplied moments.
// A negative or non-Fermi-realizable state is rejected, never clipped to LTE.
// x=epsilon/mean keeps the two Newton columns well scaled.
KOKKOS_INLINE_FUNCTION bool reconstruct(grid const& g, double n, double J, double* f,
                                        diagnostic* diag=nullptr) {
    if(diag) {
        *diag = diagnostic{};
        diag->n=n; diag->J=J;
        diag->emin=g.e[0]; diag->emax=g.e[g.n-1];
    }
    if (n==0 && J==0) { for(int i=0;i<g.n;++i) f[i]=0; return true; }
    if (!(n>0 && J>0) || !Kokkos::isfinite(n) || !Kokkos::isfinite(J)) {
        if(diag) diag->code = failure_code::moment_input;
        return false;
    }
    double const mean=J/n;
    // With nonnegative occupations, J/n is a weighted average of the nodes.
    // More Newton iterations cannot fit a mean outside their convex hull.
    if(!(mean>=g.e[0] && mean<=g.e[g.n-1])) {
        if(diag) diag->code = failure_code::moment_grid_range;
        return false;
    }
    double a=Kokkos::log(2*phase*Kokkos::pow(mean/3,3)/n), b=3;
    for (int it=0; it<70; ++it) {
        double m0=0,m1=0,h0=0,h1=0,h2=0;
        for(int i=0;i<g.n;++i) {
            double const x=g.e[i]/mean;
            f[i]=occupation(a+b*x);
            double const v=g.w[i]/n*f[i], h=v*(1-f[i]);
            m0+=v; m1+=v*x; h0+=h; h1+=h*x; h2+=h*x*x;
        }
        double const err=Kokkos::fmax(Kokkos::fabs(m0-1),Kokkos::fabs(m1-1));
        if(diag) { diag->iteration=it; diag->residual=err; }
        if (err<2e-11) return true;
        double const det=h0*h2-h1*h1;
        if (!(det>0) || !Kokkos::isfinite(det)) {
            if(diag) diag->code = failure_code::reconstruction_jacobian;
            return false;
        }
        double const da=((m0-1)*h2-(m1-1)*h1)/det;
        double const db=((m1-1)*h0-(m0-1)*h1)/det;
        bool accepted=false;
        for(double t=1; t>1e-10; t*=0.5) {
            if (!(b+t*db>0)) continue;
            double p0=0,p1=0;
            for(int i=0;i<g.n;++i) {
                double const x=g.e[i]/mean, v=g.w[i]/n*occupation(a+t*da+(b+t*db)*x);
                p0+=v; p1+=v*x;
            }
            if(Kokkos::fmax(Kokkos::fabs(p0-1),Kokkos::fabs(p1-1))<err) {
                a+=t*da; b+=t*db; accepted=true; break;
            }
        }
        if(!accepted) {
            if(diag) diag->code = failure_code::reconstruction_line_search;
            return false;
        }
    }
    if(diag) diag->code = failure_code::reconstruction_iterations;
    return false;
}

struct sources {
    double number=0; // common net event rate, cm^-3 s^-1
    double energy[2]={0,0}; // separate net MeV cm^-3 s^-1
    double emission[2]={0,0}, emission_n[2]={0,0};
    double loss[2]={0,0}, loss_n[2]={0,0}; // integrated (j+a)f, not a alone
};

KOKKOS_INLINE_FUNCTION sources evaluate(kernel const& k, double const* f, double const* fb) {
    sources s;
    for(int i=0;i<k.g.n;++i) for(int j=0;j<k.g.n;++j) {
        double const wa=k.g.w[i]*k.g.w[j]*k.a[i][j];
        double const wp=wa*k.boltz[i]*k.boltz[j];
        double const prod=wp*(1-f[i])*(1-fb[j]);
        double const ann=wa*f[i]*fb[j], net=prod-ann;
        s.number+=net;
        s.energy[0]+=k.g.e[i]*net; s.energy[1]+=k.g.e[j]*net;
        double const j0=wp*(1-fb[j]), j1=wp*(1-f[i]);
        double const l0=(j0+wa*fb[j])*f[i], l1=(j1+wa*f[i])*fb[j];
        s.emission[0]+=k.g.e[i]*j0; s.emission[1]+=k.g.e[j]*j1;
        s.emission_n[0]+=j0; s.emission_n[1]+=j1;
        s.loss[0]+=k.g.e[i]*l0; s.loss[1]+=k.g.e[j]*l1;
        s.loss_n[0]+=l0; s.loss_n[1]+=l1;
    }
    return s;
}

struct lte_rates { double emission[2], emission_n[2], kappa_E[2], kappa_N[2]; };

// LTE rates of a pair at fugacities eta0/eta1 from any pair evaluator eval(f,fb) -> sources.
// Past occupation arguments of 500 an orientation's FD weights underflow, but kappa = loss/J is
// a ratio: it is taken from Boltzmann weights shifted by the smallest argument (exact there), so
// eta is never clamped and nothing aborts.  Emission keeps the true (possibly zero) weights.
template<class Eval>
KOKKOS_INLINE_FUNCTION bool lte_pair_rates(grid const& g, double T, double eta0, double eta1,
                                           Eval const& eval, lte_rates& out) {
    double f[2][max_order], n[2]={0,0}, J[2]={0,0}, amin[2]={1e300,1e300};
    double const eta[2]={eta0,eta1};
    for(int a=0;a<2;++a) for(int l=0;l<g.n;++l) {
        double const arg=g.e[l]/T-eta[a];
        f[a][l]=occupation(arg); n[a]+=g.w[l]*f[a][l]; J[a]+=g.w[l]*g.e[l]*f[a][l];
        amin[a]=Kokkos::fmin(amin[a],arg);
    }
    if(!(Kokkos::isfinite(amin[0]) && Kokkos::isfinite(amin[1]))) return false;
    sources const ps=eval(f[0],f[1]);
    for(int a=0;a<2;++a) {
        out.emission[a]=ps.emission[a]; out.emission_n[a]=ps.emission_n[a];
        if(amin[a]>500) {
            double ft[2][max_order], nt=0, Jt=0;
            for(int l=0;l<g.n;++l) {
                ft[0][l]=f[0][l]; ft[1][l]=f[1][l];
                ft[a][l]=Kokkos::exp(-(g.e[l]/T-eta[a]-amin[a]));
                nt+=g.w[l]*ft[a][l]; Jt+=g.w[l]*g.e[l]*ft[a][l];
            }
            sources const pt=eval(ft[0],ft[1]);   // only loss[a] is read: it is linear in ft[a]
            out.kappa_E[a]=pt.loss[a]/Jt; out.kappa_N[a]=pt.loss_n[a]/nt;
        } else {
            if(!(n[a]>0 && J[a]>0)) return false;
            out.kappa_E[a]=ps.loss[a]/J[a]; out.kappa_N[a]=ps.loss_n[a]/n[a];
        }
        if(!Kokkos::isfinite(out.kappa_E[a]+out.kappa_N[a]+out.emission[a]+out.emission_n[a])) return false;
    }
    return true;
}
} // namespace grace::pairs
#endif
