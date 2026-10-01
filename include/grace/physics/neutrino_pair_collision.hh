/** @file
 * Isotropic, blocked pair collision operator for ONE physical heavy flavour.
 * Energies: MeV; densities: cm^-3; kernels: cm^3/s. See pair_processes.md.
 * No opacity is inferred from Kirchhoff away from equilibrium.
 */
#ifndef GRACE_NEUTRINO_PAIR_COLLISION_HH
#define GRACE_NEUTRINO_PAIR_COLLISION_HH

#include <grace_config.h>
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
    double e[max_order], w[max_order]; // w includes the one-helicity phase space
    KOKKOS_INLINE_FUNCTION void init(int order, double scale) {
        n = order;
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

KOKKOS_INLINE_FUNCTION double absorption_kernel(material const& m, channels c,
                                                 double e, double ep) {
    double result=0;
#ifdef GRACE_HAVE_BNS_NURATES
    if (c.ee) {
        // Algebraically PairKernels.abs[nux], evaluated without exp(+s/T)
        // times an underflowed production kernel. The oriented Psi terms
        // must NOT be symmetrised for separately evolved nu and antinu.
        double psi[2]; PairPsi(0,e/m.T,ep/m.T,m.mu_e/m.T,psi);
        double const a=kBS_Pair_Alpha1_1, b=kBS_Pair_Alpha2_1;
        double const r=-0.5*kBS_Pair_Phi*m.T*m.T*(a*a*psi[0]+b*b*psi[1])
                      /(-Kokkos::expm1(-(e+ep)/m.T));
        if(!Kokkos::isfinite(r)) return r; // do not hide NaNs with fmax
        result += Kokkos::fmax(0.0,r)*1e-21; // nm^3/s -> cm^3/s
    }
    if (c.nn && m.nb>0 && (m.yn>0 || m.yp>0)) {
        MyEOSParams eos{};
        eos.temp=m.T; eos.nb=m.nb*1e-21; eos.yn=m.yn; eos.yp=m.yp;
        BremKernelParams p{e,ep,0,false};
        result += BremKernelsLegCoeff(&p,&eos).abs[id_nux]*1e-21;
    }
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
        for (int i=0; i<g.n; ++i) for (int j=0; j<g.n; ++j) {
            if(symmetric && j<i) { a[i][j]=a[j][i]; continue; }
            double r=absorption_kernel(m,c,g.e[i],g.e[j]);
            if (symmetric) r=0.5*(r+absorption_kernel(m,c,g.e[j],g.e[i]));
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
    if (!(n>0 && J>0)) {
        if(diag) diag->code = failure_code::moment_input;
        return false;
    }
    double const mean=J/n;
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
} // namespace grace::pairs
#endif
