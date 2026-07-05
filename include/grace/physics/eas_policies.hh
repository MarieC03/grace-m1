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
        #if GRACE_M1_NU_SPECIES < 1
        ERROR("If you want to use_test_eas_op you have to activate at least one Species \nGRACE_M1_NU_SPECIES={1,3,5} ") ;
        #endif
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
        #if GRACE_M1_NU_SPECIES >= 1
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
        #if GRACE_M1_NU_SPECIES >= 3
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
        #if GRACE_M1_NU_SPECIES >= 5
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
        #endif // GRACE_M1_NU_SPECIES >= 1
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
        eas_rho_min(grace::get_param<double>("grmhd", "atmosphere", "rho_fl")
                    * (1.0 + grace::get_param<double>("grmhd", "atmosphere", "atmo_tol"))),
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

    // Equilibrium-system selector for the per-flavor (Gieg) scheme.
    //   FULL        : both lepton sectors trapped -> solve (Ye,Ymu,T)
    //   PARTIAL_E   : only electrons trapped     -> solve (Ye,T) at fixed Ymu
    //   PARTIAL_MU  : only muons trapped         -> solve (Ymu,T) at fixed Ye
    //   ENERGY_ONLY : T from energy conservation at fixed (Ye,Ymu)
    // The energy residual includes only the participating flavours' neutrino
    // energy (paper Sec. B: "contributions only from flavour l and heavy-lepton
    // neutrinos"); heavy-lepton (NUX) always contributes.
    enum class beq_mode_t : int { FULL, PARTIAL_E, PARTIAL_MU, ENERGY_ONLY } ;

    // Residuals of the equilibrium conditions at trial (Ye_t, Ymu_t, T_t).
    // f[0]: electron lepton number, f[1]: muon lepton number (5sp only),
    // f[2]: energy.  Returns false on EOS failure / non-finite output.
    GRACE_HOST_DEVICE
    bool betaeq_residuals(
        double rho_code, const double* xyz, tau_policy_fixed const& tauf,
        double Ye_t, double Ymu_t, double T_t,
        double Yle, double Ylmu, double u,
        double* f, beq_mode_t mode = beq_mode_t::FULL) const
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
        #if GRACE_M1_NU_SPECIES >= 5
        f[1] = Ymu_t + eq_nu_number_fraction(Ft, NUMU)
                     - eq_nu_number_fraction(Ft, NUMUBAR) - Ylmu ;
        // energy: heavy-lepton always; electron sector unless PARTIAL_MU;
        // muon sector unless PARTIAL_E.  (FULL/ENERGY_ONLY include both.)
        double E_nu = 2.0 * eq_nu_energy_code(Ft, NUX) ;
        if (mode != beq_mode_t::PARTIAL_MU)
            E_nu += eq_nu_energy_code(Ft, NUE) + eq_nu_energy_code(Ft, NUEBAR) ;
        if (mode != beq_mode_t::PARTIAL_E)
            E_nu += eq_nu_energy_code(Ft, NUMU) + eq_nu_energy_code(Ft, NUMUBAR) ;
        f[2] = e + E_nu - u ;
        #else
        (void)Ylmu ; (void)mode ;
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
        double& T_eq, double& Ye_eq, double& Ymu_eq,
        beq_mode_t mode = beq_mode_t::FULL) const
    {
        T_eq = T_old ; Ye_eq = Ye_old ; Ymu_eq = Ymu_old ;
        #if GRACE_M1_NU_SPECIES < 3
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
        #if GRACE_M1_NU_SPECIES >= 5
        const double N_numu    = state(VEC(i,j,k), m1_nrad_idx<2>(), q) * oosg ;
        const double N_numubar = state(VEC(i,j,k), m1_nrad_idx<3>(), q) * oosg ;
        const double Ylmu = Ymu_old + (N_numu - N_numubar) / D ;
        #else
        const double Ylmu = 0.0 ;
        #endif

        const double rho_code = F.rho_code ;

        // ---- per-mode setup (Gieg two-timescale partial equilibria) -------
        //   uidx : active unknown v-indices (v = {Ye, Ymu, T})
        //   ridx : enforced residual rows of betaeq_residuals (f0=e-lepton,
        //          f1=mu-lepton, f2=energy)
        //   E_rad: radiation energy restricted to the participating flavours
        //   eps_y*: lepton fractions used for the matter-eps baseline
        //   FULL reproduces the original single-timescale solve exactly.
        // VALIDATION POINTS: (i) the eps0 baseline composition for partial
        //   modes, (ii) the ENERGY_ONLY target u (recomputed from the passed-in
        //   fixed Ye/Ymu, not threaded from the full solve).  Reasonable first
        //   cut; check against the reference before trusting quantitatively.
        int uidx[3] = {0,2,2}, ridx[3] = {0,1,2}, n_eq = 2 ;
        double E_rad = 0.0 ;
        double eps_yle = Yle, eps_ylmu = Ylmu ;
        #if GRACE_M1_NU_SPECIES >= 5
        const double Je = fluid_frame_J<0>(VEC(i,j,k),q,metric)
                        + fluid_frame_J<1>(VEC(i,j,k),q,metric) ;
        const double Jm = fluid_frame_J<2>(VEC(i,j,k),q,metric)
                        + fluid_frame_J<3>(VEC(i,j,k),q,metric) ;
        const double Jx = fluid_frame_J<4>(VEC(i,j,k),q,metric) ;
        switch (mode) {
            case beq_mode_t::PARTIAL_E:                 // electrons trapped only
                uidx[0]=0; uidx[1]=2; n_eq=2; ridx[0]=0; ridx[1]=2;
                E_rad = Je + Jx ; eps_ylmu = Ymu_old ; break ;
            case beq_mode_t::PARTIAL_MU:                // muons trapped only
                uidx[0]=1; uidx[1]=2; n_eq=2; ridx[0]=1; ridx[1]=2;
                E_rad = Jm + Jx ; eps_yle = Ye_old ; break ;
            case beq_mode_t::ENERGY_ONLY:              // T from energy at fixed Y
                uidx[0]=2; n_eq=1; ridx[0]=2;
                E_rad = Je + Jm + Jx ; break ;
            case beq_mode_t::FULL: default:            // both trapped
                uidx[0]=0; uidx[1]=1; uidx[2]=2; n_eq=3;
                ridx[0]=0; ridx[1]=1; ridx[2]=2;
                E_rad = Je + Jm + Jx ; break ;
        }
        #else
        (void)mode ;   // 3-species: FULL only -> unknowns (Ye,T), res (f0,f1)
        uidx[0]=0; uidx[1]=2; n_eq=2; ridx[0]=0; ridx[1]=1;
        E_rad = fluid_frame_J<0>(VEC(i,j,k),q,metric)
              + fluid_frame_J<1>(VEC(i,j,k),q,metric)
              + fluid_frame_J<2>(VEC(i,j,k),q,metric) ;
        #endif

        // Total-energy target: eps at the (mode-dependent) baseline composition.
        eos_err_t err0{} ;
        double T0 = T_old, rho0 = rho_code, yle0 = eps_yle, ylmu0 = eps_ylmu ;
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
                              v[0], v[1], v[2], Yle, Ylmu, u, fv, mode))
            return false ;

        const double ye_lo  = eos.get_c2p_ye_min(),  ye_hi  = eos.get_c2p_ye_max() ;
        const double ymu_lo = eos.get_c2p_ymu_min(), ymu_hi = eos.get_c2p_ymu_max() ;
        constexpr double T_lo = 1.0e-3, T_hi = 300.0 ;   // MeV
        constexpr int    max_iter = 50 ;
        constexpr double tol      = 1.0e-10 ;

        bool converged = false ;
        for (int it = 0; it < max_iter && !converged; ++it) {
            // FD Jacobian J[r][c] = d f_{ridx[r]} / d v_{uidx[c]}
            double J[3][3] = {} ;
            for (int c = 0; c < n_eq; ++c) {
                double vp[3] = { v[0], v[1], v[2] } ;
                const double h =
                    1.0e-6 * Kokkos::fmax(Kokkos::fabs(v[uidx[c]]), 1.0e-3) ;
                vp[uidx[c]] += h ;
                double fp[3] ;
                if (!betaeq_residuals(rho_code, xyz, tauf,
                                      vp[0], vp[1], vp[2], Yle, Ylmu, u, fp, mode))
                    return false ;
                for (int r = 0; r < n_eq; ++r)
                    J[r][c] = (fp[ridx[r]] - fv[ridx[r]]) / h ;
            }

            // Solve J dx = -f (Gaussian elimination with partial pivoting).
            double A[3][4] ;
            for (int r = 0; r < n_eq; ++r) {
                for (int c = 0; c < n_eq; ++c) A[r][c] = J[r][c] ;
                A[r][n_eq] = -fv[ridx[r]] ;
            }
            for (int c = 0; c < n_eq; ++c) {
                int piv = c ;
                for (int r = c+1; r < n_eq; ++r)
                    if (Kokkos::fabs(A[r][c]) > Kokkos::fabs(A[piv][c])) piv = r ;
                if (Kokkos::fabs(A[piv][c]) < 1.0e-300) return false ;
                if (piv != c)
                    for (int cc = 0; cc <= n_eq; ++cc) {
                        const double tmp = A[c][cc] ;
                        A[c][cc] = A[piv][cc] ; A[piv][cc] = tmp ;
                    }
                for (int r = c+1; r < n_eq; ++r) {
                    const double m = A[r][c]/A[c][c] ;
                    for (int cc = c; cc <= n_eq; ++cc) A[r][cc] -= m*A[c][cc] ;
                }
            }
            double dx[3] = {} ;
            for (int r = n_eq-1; r >= 0; --r) {
                double sum = A[r][n_eq] ;
                for (int c = r+1; c < n_eq; ++c) sum -= A[r][c]*dx[c] ;
                dx[r] = sum / A[r][r] ;
            }

            // Damped, bounded update.
            double scale = 1.0 ;
            for (int c = 0; c < n_eq; ++c) {
                const bool is_T = (uidx[c] == 2) ;
                const double cap = is_T
                    ? 0.5*Kokkos::fmax(v[2], 1.0)   // |dT| <= max(T/2, 0.5)
                    : 0.05 ;                        // |dY| <= 0.05 per step
                if (Kokkos::fabs(dx[c]) > cap)
                    scale = Kokkos::fmin(scale, cap/Kokkos::fabs(dx[c])) ;
            }
            double vmax = 0.0 ;
            for (int c = 0; c < n_eq; ++c) {
                v[uidx[c]] += scale * dx[c] ;
                vmax = Kokkos::fmax(vmax,
                        Kokkos::fabs(scale*dx[c])
                      / Kokkos::fmax(Kokkos::fabs(v[uidx[c]]), 1.0e-3)) ;
            }
            v[0] = Kokkos::fmax(ye_lo,  Kokkos::fmin(ye_hi,  v[0])) ;
            v[1] = Kokkos::fmax(ymu_lo, Kokkos::fmin(ymu_hi, v[1])) ;
            v[2] = Kokkos::fmax(T_lo,   Kokkos::fmin(T_hi,   v[2])) ;

            if (!betaeq_residuals(rho_code, xyz, tauf,
                                  v[0], v[1], v[2], Yle, Ylmu, u, fv, mode))
                return false ;

            converged = (vmax < tol) ;
        }
        if (!converged) return false ;

        Ye_eq  = v[0] ;
        Ymu_eq = v[1] ;
        T_eq   = v[2] ;
        return true ;
        #endif /* GRACE_M1_NU_SPECIES >= 3 */
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

    // Transparent low-density cell: zero every EAS rate output (emission +
    // opacity, all species) without touching the EOS.  Mirrors the per-species
    // write block at the end of operator().
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void floor_eas(VEC(const int i, const int j, const int k), int64_t q) const {
        #if (GRACE_M1_NU_SPECIES >= 1)
        aux(i,j,k,ETA1_,q)=1.e-30; aux(i,j,k,KAPPAA1_,q)=1.e-30; aux(i,j,k,KAPPAS1_,q)=1.e-30; aux(i,j,k,ETAN1_,q)=1.e-30; aux(i,j,k,KAPPAAN1_,q)=1.e-30;
        #endif
        #if (GRACE_M1_NU_SPECIES >= 3)
        aux(i,j,k,ETA2_,q)=1.e-30; aux(i,j,k,KAPPAA2_,q)=1.e-30; aux(i,j,k,KAPPAS2_,q)=1.e-30; aux(i,j,k,ETAN2_,q)=1.e-30; aux(i,j,k,KAPPAAN2_,q)=1.e-30;
        aux(i,j,k,ETA3_,q)=1.e-30; aux(i,j,k,KAPPAA3_,q)=1.e-30; aux(i,j,k,KAPPAS3_,q)=1.e-30; aux(i,j,k,ETAN3_,q)=1.e-30; aux(i,j,k,KAPPAAN3_,q)=1.e-30;
        #endif
        #if (GRACE_M1_NU_SPECIES >= 5)
        aux(i,j,k,ETA4_,q)=1.e-30; aux(i,j,k,KAPPAA4_,q)=1.e-30; aux(i,j,k,KAPPAS4_,q)=1.e-30; aux(i,j,k,ETAN4_,q)=1.e-30; aux(i,j,k,KAPPAAN4_,q)=1.e-30;
        aux(i,j,k,ETA5_,q)=1.e-30; aux(i,j,k,KAPPAA5_,q)=1.e-30; aux(i,j,k,KAPPAS5_,q)=1.e-30; aux(i,j,k,ETAN5_,q)=1.e-30; aux(i,j,k,KAPPAAN5_,q)=1.e-30;
        #endif
        #ifdef GRACE_M1_DEBUG_EAS
        // Keep the debug fugacity / chemical-potential fields consistent with a
        // transparent cell (no F is built here) so they don't show stale data.
        // To inspect F in the low-density region, set grmhd.atmosphere.atmo_tol = -1.
        #if (GRACE_M1_NU_SPECIES >= 1)
        aux(i,j,k,ETANU1_,q)=1.e-30;
        #endif
        #if (GRACE_M1_NU_SPECIES >= 5)
        aux(i,j,k,ETANU2_,q)=1.e-30; aux(i,j,k,ETANU3_,q)=1.e-30; aux(i,j,k,ETANU4_,q)=1.e-30; aux(i,j,k,ETANU5_,q)=1.e-30;
        #elif (GRACE_M1_NU_SPECIES >= 3)
        aux(i,j,k,ETANU2_,q)=1.e-30; aux(i,j,k,ETANU3_,q)=1.e-30;
        #endif
        aux(i,j,k,MUE_,q)=1.e-30; aux(i,j,k,MUMU_,q)=1.e-30; aux(i,j,k,MUP_,q)=1.e-30; aux(i,j,k,MUN_,q)=1.e-30;
        #endif
    }

    // Main Kernel
    void KOKKOS_INLINE_FUNCTION operator()(VEC(const int i, const int j, const int k), int64_t q, double* xyz) const {
       const double rho = aux(VEC(i,j,k),RHO_,q);
       // Transparent atmosphere: floor the rates and skip the EOS / fugacity /
       // rate evaluation entirely.  Most of the grid lives here.
       if (rho < eas_rho_min) { floor_eas(VEC(i,j,k), q); return; }
       double T         = aux(VEC(i,j,k),TEMP_,q);
       double Ye        = aux(VEC(i,j,k),YE_,q);
       double Ymu       = 0.0;
       #if GRACE_M1_NU_SPECIES >= 5
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
            #if GRACE_M1_NU_SPECIES >= 1
            eps_rad[0] = fluid_frame_eps_mev<0>(VEC(i,j,k), q, metric) ;
            eps_rad[1] = fluid_frame_eps_mev<1>(VEC(i,j,k), q, metric) ;
            #if GRACE_M1_NU_SPECIES >= 5
            eps_rad[2] = fluid_frame_eps_mev<2>(VEC(i,j,k), q, metric) ;
            eps_rad[3] = fluid_frame_eps_mev<3>(VEC(i,j,k), q, metric) ;
            eps_rad[4] = fluid_frame_eps_mev<4>(VEC(i,j,k), q, metric) ;
            #else
            // 3-species: evolved index 2 is NUX -> rates slot NUX (4).
            eps_rad[NUX] = fluid_frame_eps_mev<2>(VEC(i,j,k), q, metric) ;
            #endif
            #endif // GRACE_M1_NU_SPECIES >= 1
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
            #if GRACE_M1_NU_SPECIES >= 3
            N_ok = N_ok
                && state(VEC(i,j,k), m1_nrad_idx<1>(), q)*oosqrtg > 1.0e-16
                && state(VEC(i,j,k), m1_nrad_idx<2>(), q)*oosqrtg > 1.0e-16 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            // For 3 species idx<2> is NUX, so the guard above already covers all
            // species; for 5 species idx<2> is numu, so we must additionally
            // require ν̄_μ (idx 3, used by the muonic beta-eq below) and ν_x
            // (idx 4) to be populated -- matching the all-species intent.
            N_ok = N_ok
                && state(VEC(i,j,k), m1_nrad_idx<3>(), q)*oosqrtg > 1.0e-16
                && state(VEC(i,j,k), m1_nrad_idx<4>(), q)*oosqrtg > 1.0e-16 ;
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
                    #if GRACE_M1_NU_SPECIES >= 5
                    aux(VEC(i,j,k),YMU_,q)  = Ymu ;
                    #endif
                    all = evaluate_rates() ;
                }
            }
        }

        #if GRACE_M1_NU_SPECIES >= 5
        // ------------------------------------------------------------------
        // Per-flavor TWO-timescale equilibration (betaeq_policy = "gieg").
        // Gieg+ 2026, Table 3: classify the electron and muon lepton sectors
        // independently via tau_e, tau_mu and equilibrate each ONLY where its
        // own (anti)neutrinos are (partially) trapped — a free-streaming
        // flavour is left to the rates.  Replaces the single-timescale joint
        // equilibrium of the "timescale" mode.
        // ------------------------------------------------------------------
        if (betaeq_mode == betaeq_mode_t::gieg && dt > 0.0) {
            // 1. per-flavour equilibration timescale = min(nu, nubar)
            auto tau_of = [&](int s){
                const double ka = all.out[s].kappa_a, ks = all.out[s].kappa_s ;
                return 1.0 / Kokkos::sqrt(ka*(ka + ks) + 1.0e-45) ;
            } ;
            const double tau_e  = Kokkos::fmin(tau_of(NUE),  tau_of(NUEBAR))  ;
            const double tau_mu = Kokkos::fmin(tau_of(NUMU), tau_of(NUMUBAR)) ;
            // 2. regime per flavour: 0 = free (tau>=dt), 1 = partial, 2 = trapped
            auto regime = [&](double tau){ return tau >= dt ? 0 : (tau > 0.5*dt ? 1 : 2) ; } ;
            const int Re = regime(tau_e), Rmu = regime(tau_mu) ;
            // PT interpolation weight: 0 at tau=dt/2 (full eq) -> 1 at tau=dt (no change)
            auto wfac = [&](double tau){ double w = tau/(0.5*dt) - 1.0 ;
                                         return Kokkos::fmin(1.0, Kokkos::fmax(0.0, w)) ; } ;

            // per-flavour radiation-number floors (mirrors the timescale N_ok)
            metric_array_t metric ;
            FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k)) ;
            const double oosg = 1.0 / metric.sqrtg() ;
            const bool Ne  = (state(VEC(i,j,k), m1_nrad_idx<0>(), q)*oosg > 1.0e-16)
                          && (state(VEC(i,j,k), m1_nrad_idx<1>(), q)*oosg > 1.0e-16) ;
            const bool Nmu = (state(VEC(i,j,k), m1_nrad_idx<2>(), q)*oosg > 1.0e-16)
                          && (state(VEC(i,j,k), m1_nrad_idx<3>(), q)*oosg > 1.0e-16) ;
            const bool Nx  =  state(VEC(i,j,k), m1_nrad_idx<4>(), q)*oosg > 1.0e-16 ;

            const double T0 = T, Ye0 = Ye, Ymu0 = Ymu ;
            double T_eq=T0, Ye_eq=Ye0, Ymu_eq=Ymu0 ;
            const bool eCoupled  = (Re  >= 1) && Ne  && Nx ;
            const bool muCoupled = (Rmu >= 1) && Nmu && Nx ;
            bool changed = false ;

            if (eCoupled && muCoupled) {                       // both sectors trapped
                if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz, T0,Ye0,Ymu0,
                                            T_eq,Ye_eq,Ymu_eq, beq_mode_t::FULL)) {
                    Ye  = (Re ==2) ? Ye_eq  : Ye0  + (Ye_eq -Ye0 )*wfac(tau_e ) ;
                    Ymu = (Rmu==2) ? Ymu_eq : Ymu0 + (Ymu_eq-Ymu0)*wfac(tau_mu) ;
                    if (Re==2 && Rmu==2) {
                        T = T_eq ;                             // (T,T)
                    } else {
                        // mixed (T/PT): T from energy conservation at final Y's
                        double Te=T0, da=Ye, db=Ymu ;
                        if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz,
                                T0, Ye, Ymu, Te, da, db, beq_mode_t::ENERGY_ONLY))
                            T = Te ;
                        else
                            T = T_eq ;                         // fallback: full-eq T
                    }
                    changed = true ;
                }
            } else if (eCoupled) {                             // electrons only (Ymu fixed)
                if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz, T0,Ye0,Ymu0,
                                            T_eq,Ye_eq,Ymu_eq, beq_mode_t::PARTIAL_E)) {
                    Ye = (Re==2) ? Ye_eq : Ye0 + (Ye_eq-Ye0)*wfac(tau_e) ;
                    T  = (Re==2) ? T_eq  : T0  + (T_eq -T0 )*wfac(tau_e) ;
                    changed = true ;                           // Ymu untouched
                }
            } else if (muCoupled) {                            // muons only (Ye fixed)
                if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz, T0,Ye0,Ymu0,
                                            T_eq,Ye_eq,Ymu_eq, beq_mode_t::PARTIAL_MU)) {
                    Ymu = (Rmu==2) ? Ymu_eq : Ymu0 + (Ymu_eq-Ymu0)*wfac(tau_mu) ;
                    T   = (Rmu==2) ? T_eq   : T0   + (T_eq  -T0  )*wfac(tau_mu) ;
                    changed = true ;                           // Ye untouched
                }
            }
            // (F,F), or all solves failed -> leave (T,Ye,Ymu) as-is.

            if (changed) {
                aux(VEC(i,j,k),TEMP_,q) = T ;
                aux(VEC(i,j,k),YE_,q)   = Ye ;
                aux(VEC(i,j,k),YMU_,q)  = Ymu ;
                all = evaluate_rates() ;
            }
        }
        #endif


        #if (GRACE_M1_NU_SPECIES >= 5)
        { const nu_rates_out r = all.out[NUE];     aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR];  aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMU];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMUBAR]; aux(i,j,k,ETA4_,q)=r.eta_E; aux(i,j,k,KAPPAA4_,q)=r.kappa_a; aux(i,j,k,KAPPAS4_,q)=r.kappa_s; aux(i,j,k,ETAN4_,q)=r.eta_N; aux(i,j,k,KAPPAAN4_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];     aux(i,j,k,ETA5_,q)=r.eta_E; aux(i,j,k,KAPPAA5_,q)=r.kappa_a; aux(i,j,k,KAPPAS5_,q)=r.kappa_s; aux(i,j,k,ETAN5_,q)=r.eta_N; aux(i,j,k,KAPPAAN5_,q)=r.kappa_n; }
        #elif (GRACE_M1_NU_SPECIES >= 3)
        { const nu_rates_out r = all.out[NUE];    aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR]; aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        #elif (GRACE_M1_NU_SPECIES >= 1)
        { const nu_rates_out r = all.out[NUE];    aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        #endif

        #ifdef GRACE_M1_DEBUG_EAS
        // Debug: dump the per-species equilibrium fugacity eta_nu = mu_nu / T
        // (already (1-exp(-tau))-suppressed and clamped in make_fugacity_state)
        // so it can be compared cell-by-cell against the reference evolution.
        #if (GRACE_M1_NU_SPECIES >= 1)
        aux(i,j,k,ETANU1_,q) = F.eta_nu[NUE];
        #endif
        #if (GRACE_M1_NU_SPECIES >= 5)
        aux(i,j,k,ETANU2_,q) = F.eta_nu[NUEBAR];
        aux(i,j,k,ETANU3_,q) = F.eta_nu[NUMU];
        aux(i,j,k,ETANU4_,q) = F.eta_nu[NUMUBAR];
        aux(i,j,k,ETANU5_,q) = F.eta_nu[NUX];
        #elif (GRACE_M1_NU_SPECIES >= 3)
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
  double eas_rho_min;   // rho_fl * (1 + atmo_tol) from grmhd.atmosphere
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
