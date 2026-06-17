/**
 * @file eas_policies.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Emission/absorption/scattering policy functors (test and photon variants) plugged into the M1 source-term evaluator.
 * @date 2024-05-13
 *
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#ifndef GRACE_PHYSICS_EAS_POLICIES_HH
#define GRACE_PHYSICS_EAS_POLICIES_HH

#include <grace_config.h>

#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <grace/physics/eas_optical_depth.hh>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>

#include <grace/config/config_parser.hh>
#include <grace/physics/eas_kinds.hh>
#include <grace/system/runtime_functions.hh>

#include <string>

namespace grace {

struct test_eas_op {
    enum test_t  {
        ZERO_EAS=0,
        LARGE_KS,
        EMITTING_SPHERE,
        SHADOW_CAST,
        COUPLING_TEST
    } ;
    test_eas_op(
        grace::var_array_t _aux
    ) : aux(_aux)
    {
        auto _which_test = grace::get_param<std::string>(
            "m1", "id_type"
        ) ;
        if (_which_test == "straight_beam" or
            _which_test == "curved_beam" )
        {
            which_test = ZERO_EAS ;
        } else if (
            _which_test == "scattering"
            or _which_test == "moving_scattering"
        ) {
            which_test = LARGE_KS ;
            _ks_value = grace::get_param<double>("m1","scattering_test","k_s") ;
        } else if (
            _which_test == "shadow"
        ) {
            which_test = SHADOW_CAST;
        } else if (
            _which_test == "emitting_sphere"
        ) {
            which_test = EMITTING_SPHERE ;
            _emitting_sphere_temperature = grace::get_param<double>("m1","emitting_sphere_test","temperature") ;
            _emitting_sphere_cross_section = grace::get_param<double>("m1","emitting_sphere_test","cross_section") ;
        } else if ( _which_test == "coupling_test") {
            which_test = COUPLING_TEST ;
        } else {
            ERROR("Unknown m1 test") ;
        }
    }

    void KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(const int i, const int j, const int k), int64_t q
        , double* xyz
    ) const
    {
        auto u = Kokkos::subview(aux,VEC(i,j,k),Kokkos::ALL(),q) ;
        double r=0;
        switch (which_test) {
            case ZERO_EAS:
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ;
            break ;
            case LARGE_KS:
            u(KAPPAA1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ;
            u(KAPPAS1_) = _ks_value ;
            break ;
            case SHADOW_CAST:
            // we assume pcoords is cartesian
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ;
            if ( r<0.046875 ) {
                u(KAPPAA1_) = 1e06 ;
                u(KAPPAAN1_) = 1e06 ;
            }
            break ;
            case EMITTING_SPHERE:
            // we assume pcoords is cartesian
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ;
            if ( r < 1. ) {
                double T = _emitting_sphere_temperature ;
                double sigma = _emitting_sphere_cross_section ;
                // we set the rates according to LTE,
                // for simplicity the Stefan Boltzmann constant is 1 here
                u(KAPPAA1_) = _emitting_sphere_cross_section ;
                u(ETA1_) = SQR(T)*SQR(T) * _emitting_sphere_cross_section ;

                u(KAPPAAN1_) = _emitting_sphere_cross_section ;
                u(ETAN1_) = SQR(T)*T * _emitting_sphere_cross_section ;
            }
            break;
            case COUPLING_TEST:
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            aux(i,j,k,KAPPAS1_,q) = 0.0 ;
            if ( r < 1.0 ) {
                aux(i,j,k,KAPPAA1_,q) = 1.0 ;
                aux(i,j,k,KAPPAAN1_,q) = 1.0;
                if ( r < 0.5 ) {
                    // effectively T = 1
                    aux(i,j,k,ETA1_,q) = 0.01  ;
                    aux(i,j,k,ETAN1_,q) = 0.01 ;
                } else {
                    double T = 1. - (r-0.5)/0.5 ;
                    aux(i,j,k,ETA1_,q) = 0.01  * T * T * T * T;
                    aux(i,j,k,ETAN1_,q) = 0.01  * T * T * T ;
                }
            } else {
                aux(i,j,k,KAPPAA1_,q) = 0.0 ;
                aux(i,j,k,KAPPAAN1_,q) = 0.0;
                aux(i,j,k,ETA1_,q) = 0.0;
                aux(i,j,k,ETAN1_,q) = 0.0;
            }

            break ;
        }
        #ifdef M1_NU_THREESPECIES
        aux(i,j,k,KAPPAA2_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN2_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAS2_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA2_,q) = aux(i,j,k,ETA1_,q);
        aux(i,j,k,ETAN2_,q) = aux(i,j,k,ETAN1_,q);
        aux(i,j,k,KAPPAA3_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN3_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAS3_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA3_,q) = aux(i,j,k,ETA1_,q);
        aux(i,j,k,ETAN3_,q) = aux(i,j,k,ETAN1_,q);
        #endif
        #ifdef M1_NU_FIVESPECIES
        aux(i,j,k,KAPPAA4_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN4_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAS4_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA4_,q) = aux(i,j,k,ETA1_,q);
        aux(i,j,k,ETAN4_,q) = aux(i,j,k,ETAN1_,q);
        aux(i,j,k,KAPPAA5_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN5_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAS5_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA5_,q) = aux(i,j,k,ETA1_,q);
        aux(i,j,k,ETAN5_,q) = aux(i,j,k,ETAN1_,q);
        #endif
    }

    var_array_t aux ;
    test_t which_test;
    double _ks_value ;
    double _emitting_sphere_cross_section, _emitting_sphere_temperature;
} ;


struct photon_eas_op {
    photon_eas_op(
        var_array_t _aux
    )
     : mass_scale(grace::get_param<double>("coordinate_system","mass_scale"))
     , aux(_aux)
    {}

    void KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(const int i, const int j, const int k), int64_t q
        , double* /*xyz*/
    ) const
    {
        #ifdef GRACE_M1_PHOTONS
        // Grey photon rates: thermal bremsstrahlung (free-free, Gaunt
        // factor 1, fully ionised hydrogen-like plasma) + Thomson
        // scattering.  Photons are blackbody radiation with g = 2 and zero
        // chemical potential, so the equilibrium densities and unit
        // conversions are the same primitives used by the neutrino rates.
        using namespace nu_constants ;

        double const T_mev   = safe_pos(aux(VEC(i,j,k),TEMP_,q)) ;
        double const rho_cgs = safe_pos(aux(VEC(i,j,k),RHO_,q)) / RHOGF ;
        double const ye      = Kokkos::fmax(0.0, aux(VEC(i,j,k),YE_,q)) ;
        double const n_e     = ye * rho_cgs * avogadro ;   // cm^-3

        // Free-free emissivity (Rybicki & Lightman 5.15b, gaunt = 1):
        //   eta_ff = 1.4e-27 sqrt(T_K) n_e n_i Z^2   [erg cm^-3 s^-1]
        constexpr double mev_to_K = 1.16045e10 ;
        double const eta_ff_cgs = 1.4e-27 * Kokkos::sqrt(T_mev * mev_to_K)
                                * n_e * n_e ;
        double const Q_mev = eta_ff_cgs * erg_to_mev ;     // MeV cm^-3 s^-1

        // Kirchhoff: photon blackbody (g = 2, eta = 0).
        double const B_E = black_body_energy(2.0, T_mev, 0.0) ;
        double const B_n = black_body_number(2.0, T_mev, 0.0) ;
        double const kappa_a_cgs = Q_mev * safe_inv_pos_finite(B_E) ;
        // Grey approximation: same absorption opacity for the number
        // equation; number emissivity from Kirchhoff so that equilibrium
        // is exactly the blackbody.
        double const kappa_n_cgs = kappa_a_cgs ;
        double const R_cgs       = kappa_n_cgs * B_n ;     // cm^-3 s^-1

        // Thomson scattering.
        constexpr double sigma_T = 6.6524587e-25 ;          // cm^2
        double const kappa_s_cgs = n_e * sigma_T ;

        aux(VEC(i,j,k),ETAPH_,q)     = Q_mev_to_code(Q_mev, mass_scale) ;
        aux(VEC(i,j,k),ETANPH_,q)    = R_to_code(R_cgs, mass_scale) ;
        aux(VEC(i,j,k),KAPPAAPH_,q)  = kappa_to_code(kappa_a_cgs, mass_scale) ;
        aux(VEC(i,j,k),KAPPAANPH_,q) = kappa_to_code(kappa_n_cgs, mass_scale) ;
        aux(VEC(i,j,k),KAPPASPH_,q)  = kappa_to_code(kappa_s_cgs, mass_scale) ;
        #endif
        // Without GRACE_M1_PHOTONS there is no photon variable block:
        // the operator is a no-op (as the disabled stub was before).
    }

    var_array_t aux ;
    double mass_scale;
} ;

//------------------------------------------------------------------------------
// Neutrino EAS operator
//------------------------------------------------------------------------------
template <typename eos_t>
struct neutrinos_eas_op
    : public eos_base_t<eos_t>
{
    // All m1.eas string parameters are parsed and validated host-side in
    // eas_kinds.hh; the op carries only the resulting enums (plain ints
    // underneath, safe on device).
    neutrinos_eas_op(var_array_t _state, var_array_t _aux)
      : eos(eos::get().get_eos<eos_t>()),
        state(_state),
        aux(_aux),
        dt(grace::get_timestep()),
        mass_scale(grace::get_param<double>("coordinate_system", "mass_scale")),
        beta_decay(grace::get_param<bool>("m1", "eas", "beta_decay")),
        plasmon_decay(grace::get_param<bool>("m1", "eas", "plasmon_decay")),
        bremsstrahlung(grace::get_param<bool>("m1", "eas", "bremsstrahlung")),
        pair_annihilation(grace::get_param<bool>("m1", "eas", "pair_annihilation")),
        apply_temp_correction(grace::get_param<bool>("m1", "eas", "temperature_correction")),
        use_weakhub(grace::weakhub::weakhub_enabled_from_params()),
        betaeq_mode(get_betaeq_mode()),
        tau_kind(get_tau_policy_kind()),
        weakhub(grace::weakhub::get_device_handle())
    {
        spherical_tau.r_outer_code = grace::get_param<double>("m1", "eas", "tau_outer_radius_code");
        spherical_tau.seed_with_analytic = true;
        if ((tau_kind == tau_policy_kind_t::local_kappa ||
             tau_kind == tau_policy_kind_t::local_spherical)
            && !(spherical_tau.r_outer_code > 0.0)) {
            ERROR("m1.eas.tau_policy = local_kappa/local_spherical requires "
                  "m1.eas.tau_outer_radius_code > 0 (got "
                  << spherical_tau.r_outer_code << ").");
        }
    }

    // --- beta equilibrium: mu_e + mu_p - mu_n - Qnp = 0 ---
    // FIL/Margherita parity: the reference subtracts Qnp in the neutrino
    // chemical potential, so the Ye this drives to matches both
    // make_fugacity_state (mu_nue = mu_e+mu_p-mu_n-Qnp) and the leptonic
    // cold-table beta-eq generator.  Verified against FIL's Ye.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double beta_eq_residual(double rho, double T, double Ye, double Ymu) const {
        double mu_p=0.0, mu_mu=0.0, mu_n=0.0;
        double Xa=0.0, Xh=0.0, Xn=0.0, Xp=0.0, Abar=1.0, Zbar=1.0;
        eos_err_t err;
        double ye_loc = Ye, T_loc = T, rho_loc = rho;
        double ymu_loc = Ymu;
        const double mu_e = eos.mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(mu_mu, mu_p, mu_n, Xa, Xh, Xn, Xp, Abar, Zbar, T_loc, rho_loc, ye_loc, ymu_loc, err);
        //if (err != eos_err_t{} || !::isfinite(mu_e) || !::isfinite(mu_p) || !::isfinite(mu_n)) return 0.0;
        // KEN could not make this work
        if ( !::isfinite(mu_e) || !::isfinite(mu_p) || !::isfinite(mu_n)) return 0.0;
        return (mu_e + mu_p - mu_n - nu_constants::Qnp);
    }

    // ------------------------------------------------------------------
    // Radiation-matter beta equilibrium (Perego+ 2019; mirrors the
    // reference compute_T_ye_ymu_betaeq).
    //
    // Conserved during equilibration:
    //   Y_le  = Ye  + (N_nue - N_nuebar) / D            (electron lepton #)
    //   Y_lmu = Ymu + (N_numu - N_numubar) / D          (muon lepton #, 5sp)
    //   u     = sum_s E_s + rho (1 + eps(T, rho, Ye, Ymu))   (total energy)
    // GRACE's evolved N carries the baryon-mass normalisation (backreaction
    // adds dN directly to YESTAR_), so the lepton targets use dN/D with no
    // explicit m_nuc/rho factor.
    //
    // Solved for (Ye, Ymu, T) [5-species] or (Ye, T) [3-species] such that
    // the EQUILIBRIUM trapped-neutrino gas at the trial state reproduces
    // the targets.  Damped Newton with finite-difference Jacobian
    // (device-safe; the reference uses GSL hybrid, host-only).  The
    // optical depths of the CURRENT state are held frozen across trial
    // evaluations, as in the reference.
    // ------------------------------------------------------------------

    // Fluid-frame radiation energy density J of one species block via the
    // M1 closure (reference parity: compute_T_ye_ymu_betaeq receives the
    // closure's J, not the lab-frame E).
    template<int ispec>
    GRACE_HOST_DEVICE double fluid_frame_J(
        VEC(const int i, const int j, const int k), int64_t q,
        metric_array_t const& metric) const
    {
        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims, state, aux, q, ispec, VEC(i,j,k)) ;
        const double oosg = 1.0 / metric.sqrtg() ;
        prims[ERADL] *= oosg ;
        prims[NRADL] *= oosg ;
        prims[FXL]   *= oosg ;
        prims[FYL]   *= oosg ;
        prims[FZL]   *= oosg ;
        m1_closure_t cl{ prims, metric } ;
        cl.update_closure(0) ;
        return cl.J ;
    }

    // Fluid-frame MEAN neutrino energy [MeV] of one evolved species, from the
    // closure: eps = (J / N) * Gamma * m_nuc.  Used to drive the T_nu spectral
    // correction off the ACTUAL radiation field (FIL parity: temp_nue =
    // neutrino_temperature(eps_nue, eta), eps_nue = avg energy of the radiation,
    // NOT the emission spectrum Q/R).  Returns 0 when there is no radiation
    // (N -> 0, e.g. iteration 0 / atmosphere) so the correction stays inert
    // (fact = 1), exactly as in the reference where eps_nue -> 0 gives fact 1.
    template<int ispec>
    GRACE_HOST_DEVICE double fluid_frame_eps_mev(
        VEC(const int i, const int j, const int k), int64_t q,
        metric_array_t const& metric) const
    {
        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims, state, aux, q, ispec, VEC(i,j,k)) ;
        const double oosg = 1.0 / metric.sqrtg() ;
        prims[ERADL] *= oosg ;
        prims[NRADL] *= oosg ;
        prims[FXL]   *= oosg ;
        prims[FYL]   *= oosg ;
        prims[FZL]   *= oosg ;
        const double N = prims[NRADL] ;
        if ( !(N > 1.0e-30) ) return 0.0 ;
        m1_closure_t cl{ prims, metric } ;
        cl.update_closure(0) ;
        if ( !(cl.J > 0.0) ) return 0.0 ;
        // Mean energy in nucleon-mass units.  At the M1 ATMOSPHERE floor the
        // field has E ~ N (eps_code ~ 1), i.e. a "neutrino" carrying ~m_nuc --
        // unphysical; this is the it=0 / no-radiation state.  Treat eps_code
        // >= 0.5 as that atmosphere artifact and return 0 so the T_nu
        // correction stays inert (fact = 1), exactly like the reference where
        // eps_nue -> 0.  Real trapped/streaming neutrinos: eps_code << 0.1.
        const double eps_code = cl.J / N * cl.Gamma ;
        if ( !(eps_code < 0.5) ) return 0.0 ;
        return eps_code * nu_constants::mp_MeV ;
    }

    // (tau_policy_fixed lives in eas_optical_depth.hh.)

    // Equilibrium trapped-neutrino number fraction (dimensionless, baryon-
    // mass weighted like the evolved N) and energy density (code units) of
    // one species at a trial fugacity state.  No degeneracy factor: NUX
    // multiplicity enters explicitly in the residuals (2x in 5-species,
    // 4x in 3-species), as in the reference get_neutrino_density.  The
    // exp(-rho_lim/rho) factor is the reference's low-density cutoff.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double eq_nu_number_fraction(fugacity_state const& Ft, int s) const {
        using namespace nu_constants ;
        const double n_cm3 =
            4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm) * ipow<3>(Ft.temp_mev)
            * fermi::FD<2>::get(Ft.eta_nu[s])
            * Kokkos::exp(-1.0e11/Ft.rho_cgs) ;
        return mnuc_cgs * n_cm3 / Ft.rho_cgs ;
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double eq_nu_energy_code(fugacity_state const& Ft, int s) const {
        using namespace nu_constants ;
        const double z_mev_cm3 =
            4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm) * ipow<4>(Ft.temp_mev)
            * fermi::FD<3>::get(Ft.eta_nu[s])
            * Kokkos::exp(-1.0e11/Ft.rho_cgs) ;
        return z_mev_cm3 * mev_to_erg * EPSGF * RHOGF ;
    }

    // Residuals of the equilibrium conditions at trial (Ye_t, Ymu_t, T_t).
    // f[0]: electron lepton number, f[1]: muon lepton number (5sp only),
    // f[2]: energy.  Returns false on EOS failure / non-finite output.
    GRACE_HOST_DEVICE
    bool betaeq_residuals(
        double rho_code, const double* xyz, tau_policy_fixed const& tauf,
        double Ye_t, double Ymu_t, double T_t,
        double Yle, double Ylmu, double u,
        double* f) const
    {
        eos_err_t err{} ;
        double T_loc = T_t, rho_loc = rho_code ;
        double ye_loc = Ye_t, ymu_loc = Ymu_t ;
        const double eps =
            eos.eps__temp_rho_ye_ymu(T_loc, rho_loc, ye_loc, ymu_loc, err) ;
        if (!::isfinite(eps)) return false ;
        const double e = rho_code * (1.0 + eps) ;

        fugacity_state const Ft = make_fugacity_state(
            eos, rho_code, T_t, Ye_t, Ymu_t, mass_scale, xyz, tauf) ;

        f[0] = Ye_t + eq_nu_number_fraction(Ft, NUE)
                    - eq_nu_number_fraction(Ft, NUEBAR) - Yle ;
        #ifdef M1_NU_FIVESPECIES
        f[1] = Ymu_t + eq_nu_number_fraction(Ft, NUMU)
                     - eq_nu_number_fraction(Ft, NUMUBAR) - Ylmu ;
        f[2] = e + eq_nu_energy_code(Ft, NUE) + eq_nu_energy_code(Ft, NUEBAR)
                 + eq_nu_energy_code(Ft, NUMU) + eq_nu_energy_code(Ft, NUMUBAR)
                 + 2.0 * eq_nu_energy_code(Ft, NUX) - u ;
        #else
        (void)Ylmu ;
        f[1] = e + eq_nu_energy_code(Ft, NUE) + eq_nu_energy_code(Ft, NUEBAR)
                 + 4.0 * eq_nu_energy_code(Ft, NUX) - u ;
        f[2] = 0.0 ;
        #endif
        return ::isfinite(f[0]) && ::isfinite(f[1]) && ::isfinite(f[2]) ;
    }

    GRACE_HOST_DEVICE
    bool m1_get_beta_equilibrium(
        fugacity_state const& F,
        VEC(const int i, const int j, const int k), int64_t q,
        const double* xyz,
        double T_old, double Ye_old, double Ymu_old,
        double& T_eq, double& Ye_eq, double& Ymu_eq) const
    {
        T_eq = T_old ; Ye_eq = Ye_old ; Ymu_eq = Ymu_old ;
        #ifndef M1_NU_THREESPECIES
        // Requires at least nue + nuebar + nux evolved blocks.
        return false ;
        #else
        using namespace grace ;

        // ---- Conserved targets from the current state -------------------
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;
        const double oosg = 1.0 / metric.sqrtg() ;
        const double D    = state(VEC(i,j,k),DENS_,q) * oosg ;
        if (!(D > 0.0)) return false ;

        const double N_nue    = state(VEC(i,j,k), m1_nrad_idx<0>(), q) * oosg ;
        const double N_nuebar = state(VEC(i,j,k), m1_nrad_idx<1>(), q) * oosg ;
        const double Yle  = Ye_old + (N_nue - N_nuebar) / D ;
        #ifdef M1_NU_FIVESPECIES
        const double N_numu    = state(VEC(i,j,k), m1_nrad_idx<2>(), q) * oosg ;
        const double N_numubar = state(VEC(i,j,k), m1_nrad_idx<3>(), q) * oosg ;
        const double Ylmu = Ymu_old + (N_numu - N_numubar) / D ;
        constexpr int NEQ = 3 ;
        #else
        const double Ylmu = 0.0 ;
        constexpr int NEQ = 2 ;
        #endif

        // Total radiation energy in the FLUID frame, via the closure
        // (reference parity).  The lepton-number targets above deliberately
        // stay in the evolved (lab-frame, densitized) N/D bookkeeping —
        // that is the combination the backreaction conserves into YESTAR_.
        const double rho_code = F.rho_code ;
        double E_rad = 0.0 ;
        E_rad += fluid_frame_J<0>(VEC(i,j,k), q, metric) ;
        E_rad += fluid_frame_J<1>(VEC(i,j,k), q, metric) ;
        E_rad += fluid_frame_J<2>(VEC(i,j,k), q, metric) ;
        #ifdef M1_NU_FIVESPECIES
        E_rad += fluid_frame_J<3>(VEC(i,j,k), q, metric) ;
        E_rad += fluid_frame_J<4>(VEC(i,j,k), q, metric) ;
        #endif

        // Total-energy target: eps evaluated at the TOTAL lepton fractions
        // (as in the reference fluid_state construction).
        eos_err_t err0{} ;
        double T0 = T_old, rho0 = rho_code, yle0 = Yle, ylmu0 = Ylmu ;
        const double eps0 = eos.eps__temp_rho_ye_ymu(T0, rho0, yle0, ylmu0, err0) ;
        if (!::isfinite(eps0)) return false ;
        const double u = E_rad + rho_code * (1.0 + eps0) ;

        // Frozen taus of the current state.
        tau_policy_fixed tauf ;
        tauf.tau = F.tau_n ;

        // ---- Damped Newton with finite-difference Jacobian --------------
        double v[3]  = { Ye_old, Ymu_old, T_old } ;   // (Ye, Ymu, T)
        double fv[3] ;
        if (!betaeq_residuals(rho_code, xyz, tauf,
                              v[0], v[1], v[2], Yle, Ylmu, u, fv))
            return false ;

        const double ye_lo  = eos.get_c2p_ye_min(),  ye_hi  = eos.get_c2p_ye_max() ;
        const double ymu_lo = eos.get_c2p_ymu_min(), ymu_hi = eos.get_c2p_ymu_max() ;
        constexpr double T_lo = 1.0e-3, T_hi = 300.0 ;   // MeV
        constexpr int    max_iter = 50 ;
        constexpr double tol      = 1.0e-10 ;

        // Map solver components -> v indices: 5sp solves (Ye,Ymu,T) with
        // residuals (f0,f1,f2); 3sp solves (Ye,T) with residuals (f0,f1).
        #ifdef M1_NU_FIVESPECIES
        const int vidx[3] = {0, 1, 2} ;
        #else
        const int vidx[3] = {0, 2, 2} ;
        #endif

        bool converged = false ;
        for (int it = 0; it < max_iter && !converged; ++it) {
            // FD Jacobian J[r][c] = d f_r / d v_{vidx[c]}
            double J[3][3] = {} ;
            for (int c = 0; c < NEQ; ++c) {
                double vp[3] = { v[0], v[1], v[2] } ;
                const double h =
                    1.0e-6 * Kokkos::fmax(Kokkos::fabs(v[vidx[c]]), 1.0e-3) ;
                vp[vidx[c]] += h ;
                double fp[3] ;
                if (!betaeq_residuals(rho_code, xyz, tauf,
                                      vp[0], vp[1], vp[2], Yle, Ylmu, u, fp))
                    return false ;
                for (int r = 0; r < NEQ; ++r) J[r][c] = (fp[r] - fv[r]) / h ;
            }

            // Solve J dx = -f (Gaussian elimination with partial pivoting).
            double A[3][4] ;
            for (int r = 0; r < NEQ; ++r) {
                for (int c = 0; c < NEQ; ++c) A[r][c] = J[r][c] ;
                A[r][NEQ] = -fv[r] ;
            }
            for (int c = 0; c < NEQ; ++c) {
                int piv = c ;
                for (int r = c+1; r < NEQ; ++r)
                    if (Kokkos::fabs(A[r][c]) > Kokkos::fabs(A[piv][c])) piv = r ;
                if (Kokkos::fabs(A[piv][c]) < 1.0e-300) return false ;
                if (piv != c)
                    for (int cc = 0; cc <= NEQ; ++cc) {
                        const double tmp = A[c][cc] ;
                        A[c][cc] = A[piv][cc] ; A[piv][cc] = tmp ;
                    }
                for (int r = c+1; r < NEQ; ++r) {
                    const double m = A[r][c]/A[c][c] ;
                    for (int cc = c; cc <= NEQ; ++cc) A[r][cc] -= m*A[c][cc] ;
                }
            }
            double dx[3] = {} ;
            for (int r = NEQ-1; r >= 0; --r) {
                double sum = A[r][NEQ] ;
                for (int c = r+1; c < NEQ; ++c) sum -= A[r][c]*dx[c] ;
                dx[r] = sum / A[r][r] ;
            }

            // Damped, bounded update.
            double scale = 1.0 ;
            for (int c = 0; c < NEQ; ++c) {
                const bool is_T = (vidx[c] == 2) ;
                const double cap = is_T
                    ? 0.5*Kokkos::fmax(v[2], 1.0)   // |dT| <= max(T/2, 0.5)
                    : 0.05 ;                        // |dY| <= 0.05 per step
                if (Kokkos::fabs(dx[c]) > cap)
                    scale = Kokkos::fmin(scale, cap/Kokkos::fabs(dx[c])) ;
            }
            double vmax = 0.0 ;
            for (int c = 0; c < NEQ; ++c) {
                v[vidx[c]] += scale * dx[c] ;
                vmax = Kokkos::fmax(vmax,
                        Kokkos::fabs(scale*dx[c])
                      / Kokkos::fmax(Kokkos::fabs(v[vidx[c]]), 1.0e-3)) ;
            }
            v[0] = Kokkos::fmax(ye_lo,  Kokkos::fmin(ye_hi,  v[0])) ;
            v[1] = Kokkos::fmax(ymu_lo, Kokkos::fmin(ymu_hi, v[1])) ;
            v[2] = Kokkos::fmax(T_lo,   Kokkos::fmin(T_hi,   v[2])) ;

            if (!betaeq_residuals(rho_code, xyz, tauf,
                                  v[0], v[1], v[2], Yle, Ylmu, u, fv))
                return false ;

            converged = (vmax < tol) ;
        }
        if (!converged) return false ;

        Ye_eq  = v[0] ;
        Ymu_eq = v[1] ;
        T_eq   = v[2] ;
        return true ;
        #endif /* M1_NU_THREESPECIES */
    }

    // This is only beta eq for ye
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void find_ye_betaeq(double rho, double T, double& Ye0, double& Ymu0) const {
        // Device-friendly bisection. If no bracket -> return Ye0.
        double a = 1.0e-6, b = 0.60;
        double fa = beta_eq_residual(rho, T, a,Ymu0), fb = beta_eq_residual(rho, T, b,Ymu0);
        if (!::isfinite(fa) || !::isfinite(fb) || fa * fb > 0.0) return ;

        double left = a, right = b, fleft = fa, mid = Ye0;
        for (int it = 0; it < 40; ++it) {
            mid = 0.5 * (left + right);
            const double fm = beta_eq_residual(rho, T, mid,Ymu0);
            if (!::isfinite(fm) || fm == 0.0) break;
            if (fleft * fm <= 0.0) right = mid; else { left = mid; fleft = fm; }
            if ((right - left) < 1.0e-8) break;
        }
        Ye0 = mid;
        return;
    }

    // Main Kernel
    void KOKKOS_INLINE_FUNCTION operator()(VEC(const int i, const int j, const int k), int64_t q, double* xyz) const {
       const double rho = aux(VEC(i,j,k),RHO_,q);
       double T         = aux(VEC(i,j,k),TEMP_,q);
       double Ye        = aux(VEC(i,j,k),YE_,q);
       double Ymu       = 0.0;
       #ifdef M1_NU_FIVESPECIES
        Ymu = aux(VEC(i,j,k),YMU_,q);
        #endif
        if (betaeq_mode == betaeq_mode_t::chemical) find_ye_betaeq(rho, T, Ye, Ymu);

        // Fluid-frame mean neutrino energy per species [MeV], used to drive the
        // T_nu spectral correction off the actual radiation field (FIL parity).
        // Only needed when the correction is on; left at 0 otherwise so the
        // correction stays inert (matches it=0 / no-radiation -> fact = 1).
        double eps_rad[NUMSPECIES] = {0.0} ;
        if (apply_temp_correction) {
            metric_array_t metric ;
            FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;
            eps_rad[0] = fluid_frame_eps_mev<0>(VEC(i,j,k), q, metric) ;
            eps_rad[1] = fluid_frame_eps_mev<1>(VEC(i,j,k), q, metric) ;
            #ifdef M1_NU_FIVESPECIES
            eps_rad[2] = fluid_frame_eps_mev<2>(VEC(i,j,k), q, metric) ;
            eps_rad[3] = fluid_frame_eps_mev<3>(VEC(i,j,k), q, metric) ;
            eps_rad[4] = fluid_frame_eps_mev<4>(VEC(i,j,k), q, metric) ;
            #else
            // 3-species: evolved index 2 is NUX -> rates slot NUX (4).
            eps_rad[NUX] = fluid_frame_eps_mev<2>(VEC(i,j,k), q, metric) ;
            #endif
        }

        // The rate source (weakhub table vs analytic) and the tau policy are
        // orthogonal choices; dispatch the tau policy once and let the
        // launcher pick the source.  The fugacity state is built here, so
        // the optical depths F.tau_n are available to the caller (e.g. for
        // tau-dependent beta equilibrium) before the rates are evaluated.
        fugacity_state F;
        auto const launch = [&](auto const& tau_policy) {
            F = make_fugacity_state(
                eos, rho, T, Ye, Ymu, mass_scale, xyz, tau_policy);
            return (use_weakhub && weakhub.valid)
                ? compute_all_species_weakhub(weakhub, F, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_policy, apply_temp_correction, eps_rad)
                : compute_all_species(F, beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_policy, apply_temp_correction, eps_rad);
        };

        auto const evaluate_rates = [&]() {
            switch (tau_kind) {
            case tau_policy_kind_t::local_spherical:
                return launch(spherical_tau);
            case tau_policy_kind_t::analytic_density:
                return launch(tau_policy_analytic_density{});
            case tau_policy_kind_t::local_kappa:
                // Lagged-kappa per-species path estimate; see
                // make_lagged_kappa_tau in eas_optical_depth.hh.
                return launch(make_lagged_kappa_tau(
                    aux, VEC(i,j,k), q, spherical_tau.r_outer_code, xyz));
            case tau_policy_kind_t::eikonal:
                #ifdef GRACE_M1_OPTICAL_DEPTH
                // Read the OPTD_* fields relaxed by update_m1_optical_depth
                // (run in compute_auxiliary_quantities before this EAS pass).
                return launch(make_eikonal_tau(state, VEC(i,j,k), q));
                #else
                break;  // unreachable: parser rejects 'eikonal' without the flag
                #endif
            case tau_policy_kind_t::none:
                break;
            }
            return launch(tau_policy_none{});
        };

        nu_rates_all_out all = evaluate_rates();

        // ------------------------------------------------------------------
        // Timescale-gated beta equilibration (betaeq_policy = "timescale").
        //
        // tau_beta = 1 / sqrt(kappa_a (kappa_a + kappa_s)) is the local
        // equilibration timescale per species (kappa in code units is an
        // inverse length = inverse time with c = 1).  If the fastest
        // species equilibrates within the step (tau_beta_min < dt), relax
        // (T, Ye) toward the radiation-matter equilibrium — fully below
        // tau_beta_min/dt = 0.5, linearly interpolated up to 1 — and
        // re-evaluate the rates at the equilibrated state.
        // ------------------------------------------------------------------
        if (betaeq_mode == betaeq_mode_t::timescale && dt > 0.0) {
            double tau_beta_min = 1.0e300 ;
            for (int s = 0; s < NUMSPECIES; ++s) {
                const double ka = all.out[s].kappa_a ;
                const double ks = all.out[s].kappa_s ;
                const double tau_beta =
                    1.0 / Kokkos::sqrt(ka*(ka + ks) + 1.0e-45) ;
                tau_beta_min = Kokkos::fmin(tau_beta_min, tau_beta) ;
            }
            const double beta_equil_tscale = tau_beta_min / dt ;

            // Radiation number floors (undensitized), mirroring the
            // reference implementation's N > 1e-16 guards.
            metric_array_t metric ;
            FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;
            const double oosqrtg = 1.0 / metric.sqrtg() ;
            bool N_ok =
                state(VEC(i,j,k), m1_nrad_idx<0>(), q)*oosqrtg > 1.0e-16 ;
            #ifdef M1_NU_THREESPECIES
            N_ok = N_ok
                && state(VEC(i,j,k), m1_nrad_idx<1>(), q)*oosqrtg > 1.0e-16
                && state(VEC(i,j,k), m1_nrad_idx<2>(), q)*oosqrtg > 1.0e-16 ;
            #endif

            if (beta_equil_tscale < 1.0 && N_ok) {
                const double T_old = T, Ye_old = Ye, Ymu_old = Ymu ;
                double T_eq = T_old, Ye_eq = Ye_old, Ymu_eq = Ymu_old ;
                const bool eq_ok = m1_get_beta_equilibrium(
                    F, VEC(i,j,k), q, xyz,
                    T_old, Ye_old, Ymu_old, T_eq, Ye_eq, Ymu_eq) ;

                // On solver failure keep the current state (the reference
                // likewise falls through on GSL non-convergence).
                if (eq_ok) {
                    if (beta_equil_tscale < 0.5) {
                        // Fast equilibration: full equilibrium values.
                        T   = T_eq ;
                        Ye  = Ye_eq ;
                        Ymu = Ymu_eq ;
                    } else {
                        // Intermediate regime: linear interpolation,
                        // tscale in [0.5, 1.0] -> fac in [1.0, 0.0].
                        double fac = 2.0 * (1.0 - beta_equil_tscale) ;
                        fac = Kokkos::fmax(0.0, Kokkos::fmin(1.0, fac)) ;
                        T   = fac * T_eq   + (1.0 - fac) * T_old ;
                        Ye  = fac * Ye_eq  + (1.0 - fac) * Ye_old ;
                        Ymu = fac * Ymu_eq + (1.0 - fac) * Ymu_old ;
                    }

                    // Persist the equilibrated primitives and recompute the
                    // rates at the new state.  NB: re-syncing the conserved
                    // hydro variables to the modified (T, Ye, Ymu) is the
                    // implicit solver's responsibility, not done here.
                    aux(VEC(i,j,k),TEMP_,q) = T ;
                    aux(VEC(i,j,k),YE_,q)   = Ye ;
                    #ifdef M1_NU_FIVESPECIES
                    aux(VEC(i,j,k),YMU_,q)  = Ymu ;
                    #endif
                    all = evaluate_rates() ;
                }
            }
        }


        #if defined(M1_NU_FIVESPECIES)
        { const nu_rates_out r = all.out[NUE];     aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR];  aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMU];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMUBAR]; aux(i,j,k,ETA4_,q)=r.eta_E; aux(i,j,k,KAPPAA4_,q)=r.kappa_a; aux(i,j,k,KAPPAS4_,q)=r.kappa_s; aux(i,j,k,ETAN4_,q)=r.eta_N; aux(i,j,k,KAPPAAN4_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];     aux(i,j,k,ETA5_,q)=r.eta_E; aux(i,j,k,KAPPAA5_,q)=r.kappa_a; aux(i,j,k,KAPPAS5_,q)=r.kappa_s; aux(i,j,k,ETAN5_,q)=r.eta_N; aux(i,j,k,KAPPAAN5_,q)=r.kappa_n; }
        #elif defined(M1_NU_THREESPECIES)
        { const nu_rates_out r = all.out[NUE];    aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR]; aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        #endif

        #ifdef GRACE_M1_DEBUG_EAS
        // Debug: dump the per-species equilibrium fugacity eta_nu = mu_nu / T
        // (already (1-exp(-tau))-suppressed and clamped in make_fugacity_state)
        // so it can be compared cell-by-cell against the reference evolution.
        aux(i,j,k,ETANU1_,q) = F.eta_nu[NUE];
        #if defined(M1_NU_FIVESPECIES)
        aux(i,j,k,ETANU2_,q) = F.eta_nu[NUEBAR];
        aux(i,j,k,ETANU3_,q) = F.eta_nu[NUMU];
        aux(i,j,k,ETANU4_,q) = F.eta_nu[NUMUBAR];
        aux(i,j,k,ETANU5_,q) = F.eta_nu[NUX];
        #elif defined(M1_NU_THREESPECIES)
        aux(i,j,k,ETANU2_,q) = F.eta_nu[NUEBAR];
        aux(i,j,k,ETANU3_,q) = F.eta_nu[NUX];
        #endif
        // Matter chemical potentials [MeV] that built eta_nu, straight from the
        // EOS read in make_fugacity_state.  mu_nue = mu_e+mu_p-mu_n-Qnp, so a
        // collapsed mu_n-mu_p shows up directly as mu_n ~ mu_p here.
        aux(i,j,k,MUE_,q)  = F.mu_e;
        aux(i,j,k,MUMU_,q) = F.mu_mu;
        aux(i,j,k,MUP_,q)  = F.mu_p;
        aux(i,j,k,MUN_,q)  = F.mu_n;
        #endif
    }

  eos_t eos;
  var_array_t state;
  var_array_t aux;
  double dt;
  double mass_scale;
  bool beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation;
  bool apply_temp_correction;
  bool use_weakhub;
  betaeq_mode_t betaeq_mode;
  tau_policy_kind_t tau_kind;
  grace::weakhub::device_handle weakhub;
  grace::tau_policy_local_spherical spherical_tau{};
};

} /* namespace grace */

#endif /*GRACE_PHYSICS_EAS_POLICIES_HH*/
