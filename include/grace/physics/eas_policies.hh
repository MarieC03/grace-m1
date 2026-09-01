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
#include <grace/physics/m1_trigger.hh>
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
            _which_test == "curved_beam" or
            _which_test == "zero" )              // schema default; rates stay floored
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
            ERROR("Unknown m1 test '" << _which_test << "' for m1.eas kind 'test'") ;
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
// Beta-equilibrium failure flags
//------------------------------------------------------------------------------
// Packed into aux(BETAEQ_ERR_) with sticky-OR semantics, exactly like c2p_err
// (see c2p.hh for the pattern and the decode caveat).  Bit INDICES are
// build-dependent -- BETAEQ_MUON_SECTOR only exists with muons -- so decode
// against the enum ordinal for your build, never a literal.
//
// Every one of these was previously a bare `return false` that the caller
// swallowed with `if (eq_ok) {...}` and no else, making a cell that tried and
// failed indistinguishable in the output from one that succeeded.
//
// At NAMESPACE scope rather than inside neutrinos_eas_op<eos_t>: the ordinals
// depend only on GRACE_ENABLE_MUONS, never on eos_t, and report_betaeq_failures
// (m1.cpp) needs to test a bit without instantiating the operator.
enum betaeq_err_enum_t : uint8_t {
    BETAEQ_NO_DENSITY = 0,     //!< D <= 0: no baryons to equilibrate
    BETAEQ_EPS0_NONFINITE,     //!< energy-target EOS lookup failed
    BETAEQ_RESIDUAL_NONFINITE, //!< a residual evaluation went non-finite
    BETAEQ_JACOBIAN_SINGULAR,  //!< pivot underflow in the FD Jacobian solve
    BETAEQ_NOT_CONVERGED,      //!< exhausted the Newton iteration budget
    BETAEQ_NO_BRACKET,         //!< find_ye_betaeq: no sign change in [ye_lo,ye_hi]
    BETAEQ_BISECT_DIVERGED,    //!< find_ye_betaeq: non-finite residual mid-bisection
    BETAEQ_EOS_ERROR,          //!< beta_eq_residual: EOS reported an error
    //! Qualifier on NOT_CONVERGED: at least one active unknown was sitting on
    //! its own bound when the iteration budget ran out.  This separates "the
    //! equilibrium lies outside the EOS table, and parking on the edge is the
    //! right answer" from "the solve is genuinely wandering" -- indistinguishable
    //! otherwise, because vmax is measured BEFORE the bound clamp, so a
    //! perfectly stationary pinned iterate keeps re-reporting its rejected step.
    BETAEQ_AT_BOUND,
    //! Qualifier on NOT_CONVERGED: the step test passed but the SCALED
    //! RESIDUAL did not.  Only reachable on the bound-contact path, where the
    //! step cap is backtracked -- without this guard a shrinking cap would
    //! manufacture "convergence" at a point that is nowhere near a root, and
    //! the caller would feed that bogus equilibrium to the opacities.
    BETAEQ_RESIDUAL_LARGE,
    #ifdef GRACE_ENABLE_MUONS
    BETAEQ_MUON_SECTOR,        //!< the failing solve included the muon sector
    #endif
    BETAEQ_N_ERR
} ;
using betaeq_err_t = bitset_t<BETAEQ_N_ERR> ;


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
        weakhub(grace::weakhub::get_device_handle()),
        // M1 trigger not yet fired: build the fugacity state for the
        // diagnostics but skip the rates entirely.
        diagnostics_only(!m1_is_active())
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
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double beta_eq_residual(double rho, double T, double Ye, double Ymu, bool& ok) const {
        double mu_p=0.0, mu_mu=0.0, mu_n=0.0;
        double Xa=0.0, Xh=0.0, Xn=0.0, Xp=0.0, Abar=1.0, Zbar=1.0;
        eos_err_t err;
        double ye_loc = Ye, T_loc = T, rho_loc = rho;
        double ymu_loc = Ymu;
        const double mu_e = eos.mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(mu_mu, mu_p, mu_n, Xa, Xh, Xn, Xp, Abar, Zbar, T_loc, rho_loc, ye_loc, ymu_loc, err);
        // Signal failure through `ok` rather than returning 0.0 -- 0.0 is
        // EXACTLY the converged-root condition, so a failed lookup used to read
        // as "this cell is perfectly in beta equilibrium" and was accepted as
        // the answer by the bisection below.  That is the bug being fixed here.
        //
        // Deliberately NOT testing `err`: find_ye_betaeq brackets the root over
        // Ye in [1e-6, 0.60], which straddles the table's Ye axis, so limit_ye
        // sets EOS_YE_TOO_LOW/HIGH on essentially every probe.  Rejecting on
        // err.any() disables the whole mode (measured: 1098 of 1098 active
        // cells flagged, across the full density range including the core).
        // The clamped lookups are perfectly usable; only a NON-FINITE potential
        // means the residual is meaningless.  This is why the original
        // `err != eos_err_t{}` line was commented out -- the compile error
        // (bitset_t has no operator!=) was the lesser of its two problems.
        if ( !::isfinite(mu_e) || !::isfinite(mu_p) || !::isfinite(mu_n) ) {
            ok = false ;
            return 0.0 ;
        }
        ok = true ;
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
        betaeq_err_t& berr,
        beq_mode_t mode = beq_mode_t::FULL) const
    {
        T_eq = T_old ; Ye_eq = Ye_old ; Ymu_eq = Ymu_old ;

        // Record a failure.  The mode qualifier is stamped HERE, not on entry:
        // set at entry it would mark every cell that merely ATTEMPTED a
        // muon-sector solve, so a successful cell would look flagged and the
        // per-step failure count would be badly inflated.
        auto fail = [&](betaeq_err_enum_t bit) {
            berr.set(bit) ;
            #ifdef GRACE_ENABLE_MUONS
            if ( mode != beq_mode_t::PARTIAL_E ) berr.set(BETAEQ_MUON_SECTOR) ;
            #endif
            return false ;
        } ;
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
        if (!(D > 0.0)) return fail(BETAEQ_NO_DENSITY) ;

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
        if (!::isfinite(eps0)) return fail(BETAEQ_EPS0_NONFINITE) ;
        const double u = E_rad + rho_code * (1.0 + eps0) ;

        // Frozen taus of the current state.
        tau_policy_fixed tauf ;
        tauf.tau = F.tau_n ;

        // ---- Damped Newton with finite-difference Jacobian --------------
        double v[3]  = { Ye_old, Ymu_old, T_old } ;   // (Ye, Ymu, T)
        double fv[3] ;
        if (!betaeq_residuals(rho_code, xyz, tauf,
                              v[0], v[1], v[2], Yle, Ylmu, u, fv, mode))
            return fail(BETAEQ_RESIDUAL_NONFINITE) ;

        const double ye_lo  = eos.get_c2p_ye_min(),  ye_hi  = eos.get_c2p_ye_max() ;
        const double ymu_lo = eos.get_c2p_ymu_min(), ymu_hi = eos.get_c2p_ymu_max() ;
        constexpr double T_lo = 1.0e-3, T_hi = 300.0 ;   // MeV
        constexpr int    max_iter = 50 ;
        constexpr double tol      = 1.0e-10 ;

        bool converged = false ;
        bool touched_bound = false ;

        // Step-cap backtracking on bound contact.  Measured (Aug 2026): every
        // muon-sector failure was a two-cycle against the Ymu axis -- the
        // iterate is clamped onto a bound, its next step is capped at 0.05,
        // and it lands exactly one cap away (exits at ymu_hi-0.05 = 0.15 or
        // ymu_lo+0.05 = 0.0505), forever.  Halving the cap every time the
        // clamp actually bites collapses the cycle onto the bound instead of
        // letting it orbit.  Cells that never touch a bound are untouched by
        // this, so their convergence decision is bit-for-bit as before.
        double cap_scale = 1.0 ;
        constexpr double cap_scale_min = 1.0e-12 ;
        // Convergence needs a small RESIDUAL too, but only on the bound path:
        // shrinking the cap shrinks the step test's own measure, so without
        // this a backtracked cell would declare victory at f1/Ylmu ~ 31.
        constexpr double res_tol = 1.0e-6 ;
        double vmax = 0.0, rmax = 0.0 ;
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
                    return fail(BETAEQ_RESIDUAL_NONFINITE) ;
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
                if (Kokkos::fabs(A[piv][c]) < 1.0e-300) return fail(BETAEQ_JACOBIAN_SINGULAR) ;
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
                const double cap = cap_scale * ( is_T
                    ? 0.5*Kokkos::fmax(v[2], 1.0)   // |dT| <= max(T/2, 0.5)
                    : 0.05 ) ;                      // |dY| <= 0.05 per step
                if (Kokkos::fabs(dx[c]) > cap)
                    scale = Kokkos::fmin(scale, cap/Kokkos::fabs(dx[c])) ;
            }
            vmax = 0.0 ;
            for (int c = 0; c < n_eq; ++c) {
                v[uidx[c]] += scale * dx[c] ;
                vmax = Kokkos::fmax(vmax,
                        Kokkos::fabs(scale*dx[c])
                      / Kokkos::fmax(Kokkos::fabs(v[uidx[c]]), 1.0e-3)) ;
            }
            // Sticky, because the failure mode here is a 2-CYCLE against a
            // bound, not a cell that comes to rest on one: the iterate is
            // clamped to the bound, the next step is capped at |dY| <= 0.05,
            // and it lands exactly one cap away.  Sampling only the final
            // iterate sees whichever phase iteration 50 happened to be in and
            // undercounts badly.
            double const v0c = Kokkos::fmax(ye_lo,  Kokkos::fmin(ye_hi,  v[0])) ;
            double const v1c = Kokkos::fmax(ymu_lo, Kokkos::fmin(ymu_hi, v[1])) ;
            double const v2c = Kokkos::fmax(T_lo,   Kokkos::fmin(T_hi,   v[2])) ;
            bool const hit_bound = (v0c != v[0]) || (v1c != v[1]) || (v2c != v[2]) ;
            touched_bound = touched_bound || hit_bound ;
            if (hit_bound)
                cap_scale = Kokkos::fmax(0.5*cap_scale, cap_scale_min) ;
            v[0] = v0c ; v[1] = v1c ; v[2] = v2c ;

            if (!betaeq_residuals(rho_code, xyz, tauf,
                                  v[0], v[1], v[2], Yle, Ylmu, u, fv, mode))
                return fail(BETAEQ_RESIDUAL_NONFINITE) ;

            // Scaled residual norm over the ACTIVE rows.  Row meanings differ
            // by build: 5sp is (e-lepton, mu-lepton, energy), 3sp is
            // (e-lepton, energy) -- see betaeq_residuals.
            rmax = 0.0 ;
            for (int r = 0; r < n_eq; ++r) {
                #if GRACE_M1_NU_SPECIES >= 5
                double const sc = (ridx[r] == 0)
                        ? Kokkos::fmax(Kokkos::fabs(Yle),  1.0e-3)
                    : (ridx[r] == 1)
                        ? Kokkos::fmax(Kokkos::fabs(Ylmu), 1.0e-3)
                        : Kokkos::fmax(Kokkos::fabs(u),    1.0e-30) ;
                #else
                double const sc = (ridx[r] == 0)
                        ? Kokkos::fmax(Kokkos::fabs(Yle),  1.0e-3)
                        : Kokkos::fmax(Kokkos::fabs(u),    1.0e-30) ;
                #endif
                rmax = Kokkos::fmax(rmax, Kokkos::fabs(fv[ridx[r]]) / sc) ;
            }

            converged = (vmax < tol) && (!touched_bound || rmax < res_tol) ;
        }
        if (!converged) {
            // Did the iteration ever run into a variable bound?  Measured
            // (Aug 2026): essentially every muon-sector failure is an
            // amplitude-0.05 two-cycle against ymu_lo or ymu_hi -- Ymu exits
            // at exactly ymu_lo+0.05 or ymu_hi-0.05 -- so this separates
            // "the equilibrium is outside the representable Ymu range" from a
            // genuinely wandering solve.  vmax is measured BEFORE the clamp,
            // which is why such a cell never reports convergence.
            if (touched_bound) berr.set(BETAEQ_AT_BOUND) ;
            // Step settled but residual did not: the cap backtracking worked
            // and the root is still not reachable inside the variable bounds.
            if (vmax < tol && rmax >= res_tol) berr.set(BETAEQ_RESIDUAL_LARGE) ;
            return fail(BETAEQ_NOT_CONVERGED) ;
        }

        Ye_eq  = v[0] ;
        Ymu_eq = v[1] ;
        T_eq   = v[2] ;
        return true ;
        #endif /* GRACE_M1_NU_SPECIES >= 3 */
    }

    // This is only beta eq for ye
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE bool find_ye_betaeq(double rho, double T, double& Ye0, double& Ymu0,
                                                              betaeq_err_t& berr) const {
        // Device-friendly bisection.  On ANY failure Ye0 is left untouched --
        // it used to fall through to `Ye0 = mid` even after breaking on a
        // non-finite residual, i.e. a diverged bisection silently returned its
        // current midpoint as if it were the root.
        double a = 1.0e-6, b = 0.60;
        bool ok_a = true, ok_b = true ;
        double fa = beta_eq_residual(rho, T, a, Ymu0, ok_a) ;
        double fb = beta_eq_residual(rho, T, b, Ymu0, ok_b) ;
        if ( !ok_a || !ok_b ) { berr.set(BETAEQ_EOS_ERROR) ; return false ; }
        if (!::isfinite(fa) || !::isfinite(fb) || fa * fb > 0.0) {
            berr.set(BETAEQ_NO_BRACKET) ; return false ;
        }

        double left = a, right = b, fleft = fa, mid = Ye0;
        for (int it = 0; it < 40; ++it) {
            mid = 0.5 * (left + right);
            bool ok_m = true ;
            const double fm = beta_eq_residual(rho, T, mid, Ymu0, ok_m);
            if ( !ok_m ) { berr.set(BETAEQ_EOS_ERROR) ; return false ; }
            if (!::isfinite(fm)) { berr.set(BETAEQ_BISECT_DIVERGED) ; return false ; }
            if (fm == 0.0) break;
            if (fleft * fm <= 0.0) right = mid; else { left = mid; fleft = fm; }
            if ((right - left) < 1.0e-8) break;
        }
        Ye0 = mid;
        return true;
    }

    // Transparent low-density cell: floor every EAS rate output without touching
    // the EOS.  1e-60 not 0 so log-scale plots of kappa work; this path returns
    // before the temperature correction, so the value cannot be amplified.
    // Rate slots only.  Split out of floor_eas so the fugacity-only mode (M1
    // trigger not yet fired) can floor the rates while still writing REAL
    // diagnostics from F, instead of the transparent-cell placeholders.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void floor_rates(VEC(const int i, const int j, const int k), int64_t q) const {
        #if (GRACE_M1_NU_SPECIES >= 1)
        aux(i,j,k,ETA1_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAA1_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAS1_,q)=weakhub::kappa_floor_code; aux(i,j,k,ETAN1_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAAN1_,q)=weakhub::kappa_floor_code;
        #endif
        #if (GRACE_M1_NU_SPECIES >= 3)
        aux(i,j,k,ETA2_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAA2_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAS2_,q)=weakhub::kappa_floor_code; aux(i,j,k,ETAN2_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAAN2_,q)=weakhub::kappa_floor_code;
        aux(i,j,k,ETA3_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAA3_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAS3_,q)=weakhub::kappa_floor_code; aux(i,j,k,ETAN3_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAAN3_,q)=weakhub::kappa_floor_code;
        #endif
        #if (GRACE_M1_NU_SPECIES >= 5)
        aux(i,j,k,ETA4_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAA4_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAS4_,q)=weakhub::kappa_floor_code; aux(i,j,k,ETAN4_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAAN4_,q)=weakhub::kappa_floor_code;
        aux(i,j,k,ETA5_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAA5_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAS5_,q)=weakhub::kappa_floor_code; aux(i,j,k,ETAN5_,q)=weakhub::kappa_floor_code; aux(i,j,k,KAPPAAN5_,q)=weakhub::kappa_floor_code;
        #endif
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void floor_eas(VEC(const int i, const int j, const int k), int64_t q) const {
        floor_rates(VEC(i,j,k), q) ;
        // Transparent cell: the beta-eq solver never ran, so "no failure".
        // Written (not OR'd) so a stale bit cannot survive here.
        aux(i,j,k,BETAEQ_ERR_,q) = 0.0 ;
        #ifdef GRACE_M1_DIAGNOSTICS
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
        aux(i,j,k,MUDELTA_NPE_,q)=1.e-30;
        #ifdef GRACE_ENABLE_MUONS
        aux(i,j,k,MUDELTA_NPMU_,q)=1.e-30;
        #endif
        aux(i,j,k,XN_,q)=1.e-30; aux(i,j,k,XP_,q)=1.e-30; aux(i,j,k,XA_,q)=1.e-30; aux(i,j,k,XH_,q)=1.e-30;
        aux(i,j,k,ABAR_,q)=1.e-30; aux(i,j,k,ZBAR_,q)=1.e-30;
        // Sentinel, not the 1e-30 floor: a transparent cell never equilibrates,
        // and beta_eq_tscale is a ratio where small means "equilibrates fast".
        aux(i,j,k,BETAEQ_TSCALE_,q)=1.e30;
        #endif
    }

    // Write the M1 diagnostics from a finished fugacity_state.  Factored out so
    // both the normal path and the fugacity-only path (M1 trigger not yet
    // fired) produce identical fields; only `betaeq_tscale` differs, since it
    // is the one diagnostic derived from the rates rather than from F.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void write_diagnostics(VEC(const int i, const int j, const int k), int64_t q,
                           fugacity_state const& F, double betaeq_tscale) const {
        #ifdef GRACE_M1_DIAGNOSTICS
        // M1 diagnostics -- everything below is read straight out of the
        // fugacity_state F that was already built above, so this costs stores
        // and (for the two mu_delta) two subtractions, nothing else.
        //
        // Per-species equilibrium fugacity eta_nu = mu_nu / T (already
        // (1-exp(-tau))-suppressed and +/-5-clamped in make_fugacity_state) so
        // it can be compared cell-by-cell against the reference evolution.
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
        // Raw beta-equilibrium offsets [MeV] (FIL's mu_delta_*).  Identically
        // -mu_nue and -mu_numu in GRACE's Qnp convention, but built from the
        // potentials directly so they carry neither the tau suppression nor the
        // muonic clamp that eta_nu above does.  Zero at beta equilibrium.
        aux(i,j,k,MUDELTA_NPE_,q)  = F.mu_n - F.mu_p - F.mu_e  + nu_constants::Qnp;
        #ifdef GRACE_ENABLE_MUONS
        aux(i,j,k,MUDELTA_NPMU_,q) = F.mu_n - F.mu_p - F.mu_mu + nu_constants::Qnp;
        #endif
        // Nuclear composition -- interpolated by the same EOS call that produced
        // mu_e/mu_p/mu_n above and otherwise thrown away.
        aux(i,j,k,XN_,q)   = F.Xn;
        aux(i,j,k,XP_,q)   = F.Xp;
        aux(i,j,k,XA_,q)   = F.Xa;
        aux(i,j,k,XH_,q)   = F.Xh;
        aux(i,j,k,ABAR_,q) = F.Abar;
        aux(i,j,k,ZBAR_,q) = F.Zbar;
        // tau_beta_min/dt from the "timescale" betaeq policy (< 1 => the cell
        // equilibrates within the step); large sentinel under other policies.
        aux(i,j,k,BETAEQ_TSCALE_,q) = betaeq_tscale;
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
       #ifdef GRACE_ENABLE_MUONS
        Ymu = aux(VEC(i,j,k),YMU_,q);
        #endif
        // Per-cell beta-equilibrium failure flags, sticky-OR'd into
        // aux(BETAEQ_ERR_) at the end of the kernel.  Always on (not gated by
        // GRACE_M1_DIAGNOSTICS): that flag is off in production, which is
        // exactly where a silently non-converging solver must not hide.
        betaeq_err_t berr{} ;

        if (betaeq_mode == betaeq_mode_t::chemical) find_ye_betaeq(rho, T, Ye, Ymu, berr);

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

        #ifdef GRACE_M1_DIAGNOSTICS
        // M1 idle: the diagnostics all read F alone, so build it through the
        // same tau dispatch and stop there.  The rate slots are floored rather
        // than left stale -- they are in the "rates" output group and would
        // otherwise plot as garbage.  beta_eq_tscale keeps its "never
        // equilibrates" sentinel, the honest value when no rates were computed.
        if ( diagnostics_only ) {
            switch (tau_kind) {
            case tau_policy_kind_t::local_spherical:
                F = make_fugacity_state(eos, rho, T, Ye, Ymu, mass_scale, xyz, spherical_tau); break;
            case tau_policy_kind_t::analytic_density:
                F = make_fugacity_state(eos, rho, T, Ye, Ymu, mass_scale, xyz, tau_policy_analytic_density{}); break;
            case tau_policy_kind_t::local_kappa:
                F = make_fugacity_state(eos, rho, T, Ye, Ymu, mass_scale, xyz,
                        make_lagged_kappa_tau(aux, VEC(i,j,k), q, spherical_tau.r_outer_code, xyz)); break;
            #ifdef GRACE_M1_OPTICAL_DEPTH
            case tau_policy_kind_t::eikonal:
                F = make_fugacity_state(eos, rho, T, Ye, Ymu, mass_scale, xyz,
                        make_eikonal_tau(state, VEC(i,j,k), q)); break;
            #endif
            default:
                F = make_fugacity_state(eos, rho, T, Ye, Ymu, mass_scale, xyz, tau_policy_none{}); break;
            }
            floor_rates(VEC(i,j,k), q) ;
            write_diagnostics(VEC(i,j,k), q, F, 1.0e30) ;
            return ;
        }
        #endif

        nu_rates_all_out all = evaluate_rates();

        #ifdef GRACE_M1_DIAGNOSTICS
        // Carrier for the beta_eq_tscale diagnostic.  Large sentinel = "never
        // equilibrates", which is the honest answer under every policy that
        // does not compute a timescale at all.
        double betaeq_tscale_diag = 1.0e30 ;
        #endif
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
            #ifdef GRACE_M1_DIAGNOSTICS
            betaeq_tscale_diag = beta_equil_tscale ;
            #endif

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
                    T_old, Ye_old, Ymu_old, T_eq, Ye_eq, Ymu_eq, berr) ;

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

                    // Recompute the rates at the equilibrated state.  The
                    // equilibrated (T, Ye, Ymu) stay LOCAL and are deliberately
                    // NOT written back into aux: beta-equilibration here is a
                    // closure on the OPACITIES, not a fluid state update, which
                    // is exactly what FIL does (driver_get_eas.cc keeps them in
                    // stack locals _T/_ye/_ymu, feeds them to Fugacities +
                    // calc_eas, and writes no grid function -- its schedule
                    // block declares only READS of the hydro variables).
                    //
                    // Writing them to aux would be worse than useless: the next
                    // c2p re-derives T/Ye/Ymu from the untouched conserved
                    // state and erases them one substage later, but in the
                    // meantime TEMP_/YE_/YMU_ are RECONSTRUCTION variables, so
                    // the leak perturbs the hydro fluxes -- and it leaves
                    // EPS_/PRESS_/ENTROPY_ disagreeing with TEMP_.
                    //
                    // The genuine fluid<->radiation exchange is add_backreaction
                    // (m1.hh), which updates the conserveds and pairs both
                    // halves of the exchange.
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
                                            T_eq,Ye_eq,Ymu_eq, berr, beq_mode_t::FULL)) {
                    Ye  = (Re ==2) ? Ye_eq  : Ye0  + (Ye_eq -Ye0 )*wfac(tau_e ) ;
                    Ymu = (Rmu==2) ? Ymu_eq : Ymu0 + (Ymu_eq-Ymu0)*wfac(tau_mu) ;
                    if (Re==2 && Rmu==2) {
                        T = T_eq ;                             // (T,T)
                    } else {
                        // mixed (T/PT): T from energy conservation at final Y's
                        double Te=T0, da=Ye, db=Ymu ;
                        if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz,
                                T0, Ye, Ymu, Te, da, db, berr, beq_mode_t::ENERGY_ONLY))
                            T = Te ;
                        else
                            T = T_eq ;                         // fallback: full-eq T
                    }
                    changed = true ;
                }
            } else if (eCoupled) {                             // electrons only (Ymu fixed)
                if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz, T0,Ye0,Ymu0,
                                            T_eq,Ye_eq,Ymu_eq, berr, beq_mode_t::PARTIAL_E)) {
                    Ye = (Re==2) ? Ye_eq : Ye0 + (Ye_eq-Ye0)*wfac(tau_e) ;
                    T  = (Re==2) ? T_eq  : T0  + (T_eq -T0 )*wfac(tau_e) ;
                    changed = true ;                           // Ymu untouched
                }
            } else if (muCoupled) {                            // muons only (Ye fixed)
                if (m1_get_beta_equilibrium(F, VEC(i,j,k), q, xyz, T0,Ye0,Ymu0,
                                            T_eq,Ye_eq,Ymu_eq, berr, beq_mode_t::PARTIAL_MU)) {
                    Ymu = (Rmu==2) ? Ymu_eq : Ymu0 + (Ymu_eq-Ymu0)*wfac(tau_mu) ;
                    T   = (Rmu==2) ? T_eq   : T0   + (T_eq  -T0  )*wfac(tau_mu) ;
                    changed = true ;                           // Ye untouched
                }
            }
            // (F,F), or all solves failed -> leave (T,Ye,Ymu) as-is.

            if (changed) {
                // Rates only -- the equilibrated state stays local.  See the
                // timescale branch above for why it is not written to aux.
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

        // Pack the beta-eq failure bits into aux(BETAEQ_ERR_) with sticky-OR
        // semantics over the timestep, exactly as grmhd.hh does for c2p_err:
        // reset once per step in evolve(), OR-accumulated on every substage so
        // the value at step end is the union of failures seen.  BETAEQ_N_ERR is
        // far below 64, so kWords==1 and words[0] holds the whole pattern -- do
        // NOT read/write only words[0] here if it ever grows past 64.  All
        // values are integers well under 2^53, so the double round-trip is exact.
        {
            uint64_t const prev = static_cast<uint64_t>(aux(VEC(i,j,k),BETAEQ_ERR_,q)) ;
            aux(VEC(i,j,k),BETAEQ_ERR_,q) = static_cast<double>(prev | berr.words[0]) ;
        }

        #ifdef GRACE_M1_DIAGNOSTICS
        write_diagnostics(VEC(i,j,k), q, F, betaeq_tscale_diag) ;
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
  //! Fugacity-only mode: write diagnostics from F, floor the rates, skip the
  //! rate evaluation and everything downstream of it.
  bool diagnostics_only;
  grace::weakhub::device_handle weakhub;
  grace::tau_policy_local_spherical spherical_tau{};
};

} /* namespace grace */

#endif /*GRACE_PHYSICS_EAS_POLICIES_HH*/
