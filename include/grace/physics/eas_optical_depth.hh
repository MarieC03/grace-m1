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
    tf.tau[NUE] = m1_transport_opacity<0>(aux,q,VEC(i,j,k))*dr;
    #if GRACE_M1_NU_SPECIES >= 3
    tf.tau[NUEBAR] = m1_transport_opacity<1>(aux,q,VEC(i,j,k))*dr;
    #endif
    #if GRACE_M1_NU_SPECIES >= 5
    tf.tau[NUMU] = m1_transport_opacity<2>(aux,q,VEC(i,j,k))*dr;
    tf.tau[NUMUBAR] = m1_transport_opacity<3>(aux,q,VEC(i,j,k))*dr;
    tf.tau[NUX] = m1_transport_opacity<4>(aux,q,VEC(i,j,k))*dr;
    #elif (GRACE_M1_NU_SPECIES >= 3)
    tf.tau[NUX] = m1_transport_opacity<2>(aux,q,VEC(i,j,k))*dr;
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
    // Electron flavours only.  tau[NUMU]/[NUMUBAR]/[NUX] stay at their zero
    // default: nothing consumes them (the suppression in make_fugacity_state is
    // NUE/NUEBAR-only), so there are no OPTD fields for them to read.
    #if GRACE_M1_NU_SPECIES >= 1
    tf.tau[NUE] = state(VEC(i,j,k), m1_optd_idx<0>(), q) ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 3
    tf.tau[NUEBAR]  = state(VEC(i,j,k), m1_optd_idx<1>(), q) ;
    #endif
    return tf ;
}

namespace optd_detail {

// Lower 3-metric gamma_ij (xx,xy,xz,yy,yz,zz) WITHOUT the inverse/sqrtg that
// metric_array_t computes — the proper distance only needs gamma_ij, so this
// stays light on the GPU.  Cowling stores gamma_ij directly; Z4c stores the
// conformal gamma~_ij with gamma_ij = gamma~_ij / chi^2.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void read_lower_metric(grace::var_array_t const& s, int64_t q,
                       VEC(int const i, int const j, int const k),
                       double (&g)[6])
{
    using namespace grace ;
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    double const chi  = s(VEC(i,j,k), CHI_, q) ;
    double const ooc2 = 1.0 / Kokkos::fmax(1.0e-100, chi*chi) ;
    g[0] = s(VEC(i,j,k),GTXX_,q)*ooc2 ; g[1] = s(VEC(i,j,k),GTXY_,q)*ooc2 ;
    g[2] = s(VEC(i,j,k),GTXZ_,q)*ooc2 ; g[3] = s(VEC(i,j,k),GTYY_,q)*ooc2 ;
    g[4] = s(VEC(i,j,k),GTYZ_,q)*ooc2 ; g[5] = s(VEC(i,j,k),GTZZ_,q)*ooc2 ;
#else
    g[0] = s(VEC(i,j,k),GXX_,q) ; g[1] = s(VEC(i,j,k),GXY_,q) ;
    g[2] = s(VEC(i,j,k),GXZ_,q) ; g[3] = s(VEC(i,j,k),GYY_,q) ;
    g[4] = s(VEC(i,j,k),GYZ_,q) ; g[5] = s(VEC(i,j,k),GZZ_,q) ;
#endif
}

// One min-path relaxation at interior cell (i,j,k), all active blocks at once.
// ds is computed ONCE per neighbor (shared across blocks).  Reads neighbor tau
// and the metric from state_read (valid ghosts), kappa from aux; the result is
// written by the caller into state_write.  Clean Jacobi: read and write are
// distinct buffers (old_state / new_state).
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void relax_cell(
    grace::var_array_t const& state,   // state_read
    grace::var_array_t const& aux,
    VEC(int const i, int const j, int const k), int64_t q,
    double const dx0, double const dx1, double const dx2,
    double (&tau_out)[5])
{
    using namespace grace ;

    double gc[6] ;
    read_lower_metric(state, q, VEC(i,j,k), gc) ;

    // Neighbours only, as in FIL (driver_update_tau.cc skips ni=nj=nk=0): with the
    // cell's own old tau in the min, tau could only fall and never followed moving
    // matter.  It stays bounded by the min path to the transparent exterior.
    #if GRACE_M1_NU_SPECIES >= 1
    double const kc0 = m1_transport_opacity<0>(aux,q,VEC(i,j,k));
    double b0 = 1.0e200 ;
    #endif
    // Electron flavours only -- see m1_optd_idx / variable_indices.hh.
    #if GRACE_M1_NU_SPECIES >= 3
    double const kc1 = m1_transport_opacity<1>(aux,q,VEC(i,j,k));
    double b1 = 1.0e200 ;
    #endif

    for (int ni = -1; ni <= 1; ++ni)
    for (int nj = -1; nj <= 1; ++nj)
    for (int nk = -1; nk <= 1; ++nk) {
        if (ni == 0 && nj == 0 && nk == 0) continue ;
        int const ii = i+ni, jj = j+nj, kk = k+nk ;

        double const d0 = dx0*ni, d1 = dx1*nj, d2 = dx2*nk ;
        double gn[6] ;
        read_lower_metric(state, q, VEC(ii,jj,kk), gn) ;
        double const gxx = 0.5*(gc[0]+gn[0]), gxy = 0.5*(gc[1]+gn[1]) ;
        double const gxz = 0.5*(gc[2]+gn[2]), gyy = 0.5*(gc[3]+gn[3]) ;
        double const gyz = 0.5*(gc[4]+gn[4]), gzz = 0.5*(gc[5]+gn[5]) ;
        double const ds2 = gxx*d0*d0 + gyy*d1*d1 + gzz*d2*d2
                         + 2.0*( gxy*d0*d1 + gxz*d0*d2 + gyz*d1*d2 ) ;
        double const ds  = Kokkos::sqrt(Kokkos::fmax(0.0, ds2)) ;

        #if GRACE_M1_NU_SPECIES >= 1
        {
            double const kn = m1_transport_opacity<0>(aux,q,VEC(ii,jj,kk));
            b0 = Kokkos::fmin(b0, 0.5*(kc0+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<0>(),q)) ;
        }
        #endif
        #if GRACE_M1_NU_SPECIES >= 3
        {
            double const kn = m1_transport_opacity<1>(aux,q,VEC(ii,jj,kk));
            b1 = Kokkos::fmin(b1, 0.5*(kc1+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<1>(),q)) ;
        }
        #endif
    }

    #if GRACE_M1_NU_SPECIES >= 1
    tau_out[0] = Kokkos::fmax(0.0, b0) ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 3
    tau_out[1] = Kokkos::fmax(0.0, b1) ;
    #endif
}

} // namespace optd_detail

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
