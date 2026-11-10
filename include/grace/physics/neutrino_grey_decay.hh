#ifndef GRACE_NEUTRINO_GREY_DECAY_HH
#define GRACE_NEUTRINO_GREY_DECAY_HH
#include <grace/physics/neutrino_pair_collision.hh>

namespace grace::pairs {
// Representative-energy closure of mu -> e + nu + antinu. No collision
// kernel or spectral fallback. P,A are volume event rates, not opacities.
// The lifetime and Michel means fix the dilute, nonrelativistic limit;
// a common positive amplitude fixes detailed balance in both directions.
struct grey_decay_rates {
    double T, energy[2][2], production[2], absorption[2], density[2], mean[2];
    KOKKOS_INLINE_FUNCTION static double logf(double x) {
        return -Kokkos::fmax(x,0.)-Kokkos::log(1+Kokkos::exp(-Kokkos::fabs(x)));
    }
    KOKKOS_INLINE_FUNCTION bool init(double temp,double mue,double mumu) {
        GRACE_SOURCE_PROFILE(grey_rates);
        if(!(temp>0) || !Kokkos::isfinite(temp+mue+mumu)) return false;
        T=temp;
        constexpr double M=105.6583755,hc=1.973269804e-11,tau=2.197e-6;
        constexpr double x[]={.0950125098376374,.2816035507792589,.4580167776572274,.6178762444026438,
            .7554044083550030,.8656312023878318,.9445750230732326,.9894009349916499};
        constexpr double w[]={.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,
            .1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541};
        for(int c=0;c<2;++c) {
            double const sign=c==0?1.:-1.,mu=sign*mumu;
            double const qmax=Kokkos::fmax(0.,mu-M)+48*T;
            double const pmax=Kokkos::sqrt(qmax*(qmax+2*M));
            double const reference=logf((M-mu)/T);
            double n=0,u=0;
            for(int i=0;i<8;++i) for(int side=0;side<2;++side) {
                double const p=pmax*(1+(side==0?-x[i]:x[i]))/2;
                double const E=Kokkos::sqrt(M*M+p*p);
                double const dn=pmax*w[i]/2*p*p*Kokkos::exp(logf((E-mu)/T)-reference);
                n+=dn; u+=E*dn;
            }
            if(!(n>0)) return false;
            mean[c]=u/n;
            double const logn=Kokkos::log(n/(pi*pi*hc*hc*hc))+reference;
            density[c]=Kokkos::exp(logn);
            energy[c][0]=.35*mean[c]; energy[c][1]=.30*mean[c];
            double const xm=(mean[c]-mu)/T,xe=(.35*mean[c]-sign*mue)/T;
            double const logC=logn-logf(xm)+Kokkos::log(M/mean[c]/tau);
            production[c]=Kokkos::exp(logC+logf(xm)+logf(-xe));
            absorption[c]=Kokkos::exp(logC+logf(-xm)+logf(xe));
            if(!Kokkos::isfinite(production[c]+absorption[c])) return false;
        }
        return true;
    }
    KOKKOS_INLINE_FUNCTION sources evaluate(int c,double const* shape0,double const* shape1) const {
        sources s;
        double const arg[2]={shape0[0]+shape0[1]*energy[c][0],shape1[0]+shape1[1]*energy[c][1]};
        double f[2],h[2];
        for(int a=0;a<2;++a) { f[a]=Kokkos::exp(logf(arg[a])); h[a]=Kokkos::exp(logf(-arg[a])); }
        s.number=production[c]*h[0]*h[1]-absorption[c]*f[0]*f[1];
        for(int a=0;a<2;++a) {
            s.energy[a]=energy[c][a]*s.number;
            s.emission_n[a]=production[c]*h[1-a];
            s.emission[a]=energy[c][a]*s.emission_n[a];
            s.loss_n[a]=(s.emission_n[a]+absorption[c]*f[1-a])*f[a];
            s.loss[a]=energy[c][a]*s.loss_n[a];
        }
        return s;
    }
};
}
#endif
