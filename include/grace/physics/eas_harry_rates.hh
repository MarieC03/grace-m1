#ifndef GRACE_EAS_HARRY_RATES_HH
#define GRACE_EAS_HARRY_RATES_HH
#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <grace/physics/harry_spectral_table.hh>
#include <grace/physics/harry_fd_moments.hh>
namespace grace {
// Stable log(1+exp(x)): raw absorption must be corrected BEFORE averaging.
GRACE_HOST_DEVICE inline double harry_log1pexp(double x) {
    return Kokkos::fmax(x,0.0)+Kokkos::log1p(Kokkos::exp(-Kokkos::abs(x)));
}
GRACE_HOST_DEVICE inline double harry_fd(double x) {
    double z=Kokkos::exp(-Kokkos::abs(x)); return x>=0?z/(1+z):1/(1+z);
}
// Grey FD closure: use measured mean energy with the chosen chemical-potential
// closure. Emission uses matter thermodynamics, never the radiation temperature.
// Integrals share quadrature; continuum FD normalizers preserve the LTE target.
GRACE_HOST_DEVICE inline nu_rates_all_out compute_all_species_harry(
        harry::device_handle const& h,fugacity_state const& F,double const* eps_rad) {
    nu_rates_all_out out{};
    double const pref=4*nu_constants::pi*nu_constants::clight/
        (nu_constants::hc_mevcm*nu_constants::hc_mevcm*nu_constants::hc_mevcm);
    double const lrho=Kokkos::log(F.rho_code*F.harry_density_factor),lT=Kokkos::log(F.temp_mev);
    double const xs[]={lrho,lT,F.ye+F.ymu};
    bool const scatter=!h.scattering.below(0,lrho) && !h.scattering.below(1,lT);
    harry::tensor::stencil es_stencil;
    if(scatter) es_stencil=h.scattering.locate(xs);
    for(int s=0;s<5;++s) {
        double eta=0;
        if(s<2) eta=(F.mu_e+F.mu_p-F.mu_n-nu_constants::Qnp)/F.temp_mev;
        if(s==1) eta=-eta;
        if(s==2||s==3) {
            eta=F.dilute_cc_target?F.eta_cc_mu:
                (F.mu_mu+F.mu_p-F.mu_n-nu_constants::Qnp)/F.temp_mev;
            if(s==3) eta=-eta;
        }
        auto const moments=harry_fd_moments(eta);
        double Tnu=F.temp_mev;
        if(eps_rad && eps_rad[s]>0) Tnu=eps_rad[s]*moments.f2/moments.f3;
        if(!(Tnu>0) || !Kokkos::isfinite(Tnu)) Kokkos::abort("Invalid Harry grey spectrum");
        // Harry's spectral CC reader rejects upper rho/T excursions and
        // disables each flavor separately below its own rho/T domain.
        if(s<4 && (h.cc[s].above(0,lrho)||h.cc[s].above(1,lT)))
            Kokkos::abort("Harry CC density/temperature exceeds table bounds");
        // Unlike Harry's EOS-aware clamped DB helper, this first adapter
        // has no per-flavor EOS reevaluation. Reject composition excursions
        // instead of mixing a clamped opacity with an unclamped chemical potential.
        bool const cc=s<4 && !h.cc[s].below(0,lrho) && !h.cc[s].below(1,lT);
        harry::tensor::stencil cc_stencil;
        if(cc) {
            double x[4]={lrho,lT,F.ye,harry::device_handle::log_positive(F.ymu)};
            if(!h.cc[s].contains(x)) Kokkos::abort("Harry CC composition outside table domain; EOS-consistent boundary state required");
            cc_stencil=h.cc[s].locate(x);
        }
        double J0=0,J1=0,A0=0,A1=0,S1=0;
        for(int i=0;i<h.n_groups;++i) {
            double e=h.energies(i),w=h.widths(i)*e*e;
            // Cancel exp(eta) analytically in dilute-spectrum averages.
            // Computing f and the continuum denominator separately underflows.
            double f=eta<=0 ? Kokkos::exp(-e/Tnu)/(1+Kokkos::exp(eta-e/Tnu))
                            : harry_fd(e/Tnu-eta);
            double j=0,a=0;
            if(cc) {
                double l=h.cc[s].interpolate(i,cc_stencil);
                double x=eta-e/F.temp_mev;
                j=Kokkos::exp(l+x);
                a=Kokkos::exp(l+harry_log1pexp(x));
            }
            double ks=0;
            if(scatter) {
                ks=Kokkos::exp(h.scattering.interpolate(s*h.n_groups+i,es_stencil));
                if(s==4) ks=.5*(ks+Kokkos::exp(h.scattering.interpolate(5*h.n_groups+i,es_stencil)));
            }
            J0+=w*j; J1+=w*e*j; A0+=w*a*f; A1+=w*e*a*f; S1+=w*e*ks*f;
        }
        double const fd2=moments.f2,fd3=moments.f3;
        double B0=pref*Tnu*Tnu*Tnu*fd2,B1=pref*Tnu*Tnu*Tnu*Tnu*fd3;
        if(!(B0>0&&B1>0)) Kokkos::abort("Harry grey normalization underflow");
        auto& r=out.out[s];
        r.eta_N=R_to_code(pref*J0,F.mass_scale); r.eta_E=Q_mev_to_code(pref*J1,F.mass_scale);
        r.kappa_n=kappa_to_code(pref*A0/B0,F.mass_scale);
        r.kappa_a=kappa_to_code(pref*A1/B1,F.mass_scale);
        r.kappa_s=kappa_to_code(pref*S1/B1,F.mass_scale);
        if(!Kokkos::isfinite(r.eta_N+r.eta_E+r.kappa_n+r.kappa_a+r.kappa_s))
            Kokkos::abort("Nonfinite Harry CC/scattering rate");
    }
    return out;
}
}
#endif
