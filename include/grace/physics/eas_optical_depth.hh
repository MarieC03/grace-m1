/**
 * @file eas_optical_depth.hh
 * @brief Neutrino optical-depth (tau) estimates and the tau policies used
 *        to suppress the equilibrium neutrino fugacities,
 *        eta_nu -> eta_nu * (1 - exp(-tau))   (Foucart/Bollig trick).
 *
 * Policy interface (consumed as a template parameter by
 * make_fugacity_state / compute_all_species* in
 * eas_neutrino_rates_analytic.hh):
 *   double tau_init(double rho_code, const double* xyz_code,
 *                   double mass_scale, int species, double rho_cgs) const;
 *   double tau_post(double kappa_tot_cgs, double rho_code,
 *                   const double* xyz_code, double mass_scale,
 *                   int species, double rho_cgs) const;
 *
 * Available policies:
 *   tau_policy_none             — thin limit, tau = 0 everywhere.
 *   tau_policy_analytic_density — Deaton+ 2013 density fit.  COLD NS only:
 *                                 rho-only, species-blind; known to fail
 *                                 for hot matter.
 *   tau_policy_local_spherical  — kappa * (r_outer - r) with kappa from the
 *                                 rate evaluation itself (init seeded by
 *                                 the cold fit).
 *   tau_policy_fixed            — frozen per-species values; used to hold
 *                                 the current state's taus across trial
 *                                 evaluations (beta-equilibrium solve) and
 *                                 as the carrier for make_lagged_kappa_tau.
 *
 * make_lagged_kappa_tau builds a tau_policy_fixed from the PREVIOUS step's
 * opacities stored in aux: tau_s = (kappa_a,s + kappa_s,s) * (r_outer - r).
 * Pointwise — aux is recomputed in every cell (ghosts included) from the
 * exchanged conserved state, so no communication is needed.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Volume methods to
 * simulate relativistic spacetimes and plasmas.
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
 */
#ifndef GRACE_PHYSICS_EAS_OPTICAL_DEPTH_HH
#define GRACE_PHYSICS_EAS_OPTICAL_DEPTH_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

// safe_pos, code_length_to_cm, the nu_species enum (NUMSPECIES).
#include <grace/physics/eas_neutrino_rates_analytic.hh>
// var_array_t and the m1_kappa*_idx index maps.
#include <grace/physics/m1_helpers.hh>

#include <Kokkos_MathematicalFunctions.hpp>

#include <array>

namespace grace {

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
double compute_analytic_tau_from_rho_cgs(double rho_cgs) {
    // Deaton+ 2013 fit (log10 tau vs log10 rho) for cold NS-like profiles.
    const double rcgs = safe_pos(rho_cgs);
    const double log10_tau = 0.96 * ((Kokkos::log(rcgs) / Kokkos::log(10.0)) - 11.7);
    const double tau = Kokkos::exp(Kokkos::log(10.0) * log10_tau);
    return (Kokkos::isfinite(tau) && tau > 0.0) ? tau : 0.0;
}

struct tau_policy_none {
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double /*rho_cgs*/) const {
        return 0.0;
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_post(double /*kappa_tot_cgs*/, double /*rho_code*/,
                                           const double* /*xyz_code*/, double /*mass_scale*/,
                                           int /*species*/, double /*rho_cgs*/) const {
        return 0.0;
        }
};

struct tau_policy_analytic_density {
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double rho_cgs) const {
        return compute_analytic_tau_from_rho_cgs(rho_cgs);
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_post(double /*kappa_tot_cgs*/, double /*rho_code*/,
                                           const double* /*xyz_code*/, double /*mass_scale*/,
                                           int /*species*/, double rho_cgs) const {
        return compute_analytic_tau_from_rho_cgs(rho_cgs);
        }
};

struct tau_policy_local_spherical {
    // Outer radius in code units (same coordinates as xyz). If <=0, tau_post returns 0.
    double r_outer_code{0.0};
    // Optional seed for tau_init.
    bool seed_with_analytic{true};

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double rho_cgs) const {
        return seed_with_analytic ? compute_analytic_tau_from_rho_cgs(rho_cgs) : 0.0;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    double tau_post(double kappa_tot_cgs, double /*rho_code*/,
                                           const double* xyz_code, double mass_scale,
                                           int /*species*/, double /*rho_cgs*/) const {
        if (!(r_outer_code > 0.0) || !(kappa_tot_cgs > 0.0)) return 0.0;
        const double r_code = Kokkos::sqrt(xyz_code[0]*xyz_code[0] + xyz_code[1]*xyz_code[1] + xyz_code[2]*xyz_code[2]);
        const double dr_code = (r_outer_code > r_code) ? (r_outer_code - r_code) : 0.0;
        const double dr_cm = dr_code * code_length_to_cm(mass_scale);
        const double tau = kappa_tot_cgs * dr_cm;
        return (::isfinite(tau) && tau > 0.0) ? tau : 0.0;
    }
};

// Frozen per-species taus.  Used (a) by the beta-equilibrium solver to hold
// the current state's taus across trial fugacity evaluations and (b) as the
// carrier for the lagged-kappa estimate below.
struct tau_policy_fixed {
    std::array<double, NUMSPECIES> tau{{0,0,0,0,0}} ;
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    tau_init(double, const double*, double, int s, double) const
    { return tau[s] ; }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    tau_post(double, double, const double*, double, int, double) const
    { return 0.0 ; }
} ;

// Per-species path estimate from the PREVIOUS step's opacities:
//   tau_s = (kappa_a,s + kappa_s,s) * (r_outer - r).
// The kappa aux fields are overwritten by the rate evaluation that follows,
// hence the one-step lag (FIL evolves tau_n the same way).  First call after
// ID: kappa aux is zero -> thin limit, builds up within one step.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
tau_policy_fixed make_lagged_kappa_tau(
    grace::var_array_t const& aux,
    VEC(const int i, const int j, const int k), int64_t q,
    double r_outer_code, const double* xyz)
{
    tau_policy_fixed tf{} ;
    const double r  = Kokkos::sqrt(
        xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]) ;
    const double dr = Kokkos::fmax(0.0, r_outer_code - r) ;
    tf.tau[NUE] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<0>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<0>(),q) ) * dr ;
    #ifdef M1_NU_THREESPECIES
    tf.tau[NUEBAR] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<1>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<1>(),q) ) * dr ;
    #endif
    #ifdef M1_NU_FIVESPECIES
    tf.tau[NUMU] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<2>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<2>(),q) ) * dr ;
    tf.tau[NUMUBAR] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<3>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<3>(),q) ) * dr ;
    tf.tau[NUX] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<4>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<4>(),q) ) * dr ;
    #elif defined(M1_NU_THREESPECIES)
    tf.tau[NUX] =
        ( aux(VEC(i,j,k),m1_kappaa_idx<2>(),q)
        + aux(VEC(i,j,k),m1_kappas_idx<2>(),q) ) * dr ;
    #endif
    return tf ;
}

#ifdef GRACE_M1_OPTICAL_DEPTH
// ---------------------------------------------------------------------------
// Eikonal optical-depth solver (Neilsen+ 2014).
//
// The per-block optical depths live in the evolved state (OPTD1_..OPTD5_), so
// the previous step's tau is available over the whole grid with valid ghosts
// (exchanged + AMR-prolongated + BC'd).  The relaxation is a SEPARATE grid
// kernel (update_m1_optical_depth, defined in eas_optical_depth.cpp), run in
// compute_auxiliary_quantities BEFORE set_m1_eas — so it uses the previous
// EAS's kappa (one-step lag, as in Cactus frankfurt_m1_update_tau).  Like the
// reference it sweeps INTERIOR cells only; the ghost OPTD (exchanged neighbor-
// quadrant tau) is the boundary condition and is never overwritten.
//
// make_eikonal_tau is then just the consumer: it reads the relaxed OPTD into
// a tau_policy_fixed for the rate evaluation.  Block->flavour mapping matches
// make_lagged_kappa_tau / the ERAD* layout.
// ---------------------------------------------------------------------------
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
tau_policy_fixed make_eikonal_tau(
    grace::var_array_t const& state,
    VEC(const int i, const int j, const int k), int64_t q)
{
    tau_policy_fixed tf{} ;
    tf.tau[NUE] = state(VEC(i,j,k), m1_optd_idx<0>(), q) ;
    #ifdef M1_NU_FIVESPECIES
    tf.tau[NUEBAR]  = state(VEC(i,j,k), m1_optd_idx<1>(), q) ;
    tf.tau[NUMU]    = state(VEC(i,j,k), m1_optd_idx<2>(), q) ;
    tf.tau[NUMUBAR] = state(VEC(i,j,k), m1_optd_idx<3>(), q) ;
    tf.tau[NUX]     = state(VEC(i,j,k), m1_optd_idx<4>(), q) ;
    #elif defined(M1_NU_THREESPECIES)
    tf.tau[NUEBAR]  = state(VEC(i,j,k), m1_optd_idx<1>(), q) ;
    tf.tau[NUX]     = state(VEC(i,j,k), m1_optd_idx<2>(), q) ;
    #endif
    return tf ;
}

/// Seed every optical-depth field (interior + ghosts) with the Deaton+ 2013
/// cold-NS density fit.  Grid kernel, defined in eas_optical_depth.cpp.  Run
/// once at initial data, before the first EAS.
void init_m1_optical_depth(grace::var_array_t& state, grace::var_array_t& aux) ;

/// One eikonal relaxation sweep (interior cells only, clean Jacobi).  Reads
/// neighbor tau and the metric from state_read (which has valid ghosts) and
/// kappa from aux (previous EAS); writes the relaxed tau into state_write's
/// interior OPTD fields.  Call at the end of advance_substep with
/// state_read = old_state, state_write = new_state, so the subsequent
/// apply_boundary_conditions exchanges the fresh tau before the rates use it
/// (tau -> ghost exchange -> kappa, as in Cactus frankfurt_m1).
void update_m1_optical_depth(
    grace::var_array_t const& state_read,
    grace::var_array_t&       state_write,
    grace::var_array_t const& aux ) ;
#endif /* GRACE_M1_OPTICAL_DEPTH */

} // namespace grace

#endif /* GRACE_PHYSICS_EAS_OPTICAL_DEPTH_HH */
