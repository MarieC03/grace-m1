/**
 * @file m1_initial_data.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief M1 radiation initial-data kernels: zero, equilibrium, straight-beam, scattering / diffusion, moving-medium, and emitting-sphere configurations.
 * @date 2025-11-24
 *
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_PHYSICS_ID_M1_HH
#define GRACE_PHYSICS_ID_M1_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/m1_helpers.hh>   // m1_atmo_params_t / m1_excision_params_t
#include <grace/amr/amr_functions.hh>

namespace grace {

struct m1_id_t {
    // Zero-default every moment: a kernel that misses a field must ship a
    // harmless 0 (caught by the radiation floors), never uninitialized memory.
    double erad1 = 0., nrad1 = 0., fradx1 = 0., frady1 = 0., fradz1 = 0. ; //! lower indices
    #if GRACE_M1_NU_SPECIES >= 3
    double erad2 = 0., nrad2 = 0., fradx2 = 0., frady2 = 0., fradz2 = 0. ;
    double erad3 = 0., nrad3 = 0., fradx3 = 0., frady3 = 0., fradz3 = 0. ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 5
    double erad4 = 0., nrad4 = 0., fradx4 = 0., frady4 = 0., fradz4 = 0. ;
    double erad5 = 0., nrad5 = 0., fradx5 = 0., frady5 = 0., fradz5 = 0. ;
    #endif
    #ifdef GRACE_M1_PHOTONS
    double eradph = 0., nradph = 0., fradxph = 0., fradyph = 0., fradzph = 0. ;
    #endif
} ;

// Causality guard for M1 initial data: the closure is singular at |F| = E
// (pure free streaming), so IDs that sit exactly on that edge (e.g. the
// straight beam's E = F_x = 1) start the Brent/Newton machinery at its
// degenerate point.  Rescale the flux to CAUSAL_FRAC * E when it meets or
// exceeds it; E is never touched.  Kernels call this on species 1 BEFORE
// the copy blocks, so all species inherit the capped value.  Euclidean
// norm: the test IDs using this are all Minkowski.
constexpr double M1_ID_CAUSAL_FRAC = 0.999 ;

KOKKOS_INLINE_FUNCTION void
limit_m1_id_flux(double const erad, double& fx, double& fy, double& fz)
{
    double const f2   = fx*fx + fy*fy + fz*fz ;
    double const fmax = M1_ID_CAUSAL_FRAC * erad ;
    if ( f2 > fmax*fmax ) {
        double const fac = fmax / Kokkos::sqrt(f2) ;
        fx *= fac ; fy *= fac ; fz *= fac ;
    }
}

struct zero_m1_id_t {
    zero_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    // NB this assumes all optically thin!
    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        /* we assume coords are spherical here! */
        double rtp[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        auto E_atmo = atmo.E_fl * Kokkos::pow(rtp[0], atmo.E_fl_scaling) ;
        auto eps_atmo = atmo.eps_fl * Kokkos::pow(rtp[0], atmo.eps_fl_scaling) ;

        bool excise = excision.excise_by_radius ? rtp[0] <= excision.r_ex : false ; /*we don't have alp here*/

        if ( excise ) {
            id.erad1 = excision.E_ex ;
            id.nrad1 = excision.E_ex / excision.eps_ex ;
            #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = excision.E_ex ;
            id.nrad2 = excision.E_ex / excision.eps_ex ;
            id.erad3 = excision.E_ex ;
            id.nrad3 = excision.E_ex / excision.eps_ex ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = excision.E_ex ;
            id.nrad4 = excision.E_ex / excision.eps_ex ;
            id.erad5 = excision.E_ex ;
            id.nrad5 = excision.E_ex / excision.eps_ex ;
            #endif
            #ifdef GRACE_M1_PHOTONS
            id.eradph = excision.E_ex ;
            id.nradph = excision.E_ex / excision.eps_ex ;
            #endif
        } else {
            id.erad1 = E_atmo ;
            id.nrad1 = E_atmo / eps_atmo ;
            #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = E_atmo ;
            id.nrad2 = E_atmo / eps_atmo ;
            id.erad3 = E_atmo ;
            id.nrad3 = E_atmo / eps_atmo ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = E_atmo ;
            id.nrad4 = E_atmo / eps_atmo ;
            id.erad5 = E_atmo ;
            id.nrad5 = E_atmo / eps_atmo ;
            #endif
            #ifdef GRACE_M1_PHOTONS
            id.eradph = E_atmo ;
            id.nradph = E_atmo / eps_atmo ;
            #endif
        }
        id.fradx1 = id.frady1 = id.fradz1 = 0. ;
        #if GRACE_M1_NU_SPECIES >= 3
        id.fradx2 = id.frady2 = id.fradz2 = 0. ;
        id.fradx3 = id.frady3 = id.fradz3 = 0. ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
        id.fradx4 = id.frady4 = id.fradz4 = 0. ;
        id.fradx5 = id.frady5 = id.fradz5 = 0. ;
        #endif
        #ifdef GRACE_M1_PHOTONS
        id.fradxph = id.fradyph = id.fradzph = 0. ;
        #endif
        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
} ;

struct equil_m1_id_t {
    equil_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        var_array_t _aux,
        scalar_array_t<GRACE_NSPACEDIM> const _dx,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), aux(_aux), dx(_dx), pcoords(_pcoords)
    {}

    // Radiation in equilibrium with the (already set) hydro background.
    // NB: requires the EAS (kappa/eta in aux) to be evaluated BEFORE this
    // kernel runs — the dispatch in set_m1_initial_data calls set_m1_eas first.
    // Fluxes are zero: both limits (LTE equilibrium and static atmosphere)
    // are isotropic in the fluid frame.
    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        /* we assume coords are spherical here! */
        double rtp[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        double const E_atmo   = atmo.E_fl   * Kokkos::pow(rtp[0], atmo.E_fl_scaling)   ;
        double const eps_atmo = atmo.eps_fl * Kokkos::pow(rtp[0], atmo.eps_fl_scaling) ;
        double const N_atmo   = E_atmo / eps_atmo ;

        bool const excise = excision.excise_by_radius ? rtp[0] <= excision.r_ex : false ;

        // Per-species blend between LTE equilibrium (optically thick) and the
        // atmosphere floor (optically thin).  A = mfp/dx: A <= 2/3 fully
        // thick, A >= 1 fully thin, linear in between — the original
        // three-regime split, made continuous.  Equilibrium moments from
        // stationary collisional balance: E_eq = eta/kappa_a,
        // N_eq = eta_n/kappa_a_n; floored cells (kappa ~ 1e-30) blend to the
        // atmosphere automatically since their A is 1.
        auto eq_blend = [&] (double ka, double ks, double eta,
                             double etan, double kan,
                             double& E, double& N) {
            if ( excise ) {
                E = excision.E_ex ;
                N = excision.E_ex / excision.eps_ex ;
                return ;
            }
            double const A = Kokkos::fmin(1., 1./((ka+ks+1e-20)*dx(0,q))) ;
            double const w = Kokkos::fmin(1., Kokkos::fmax(0., 3.*(1.-A))) ;
            double E_eq = eta  / Kokkos::fmax(ka,  1e-40) ;
            double N_eq = etan / Kokkos::fmax(kan, 1e-40) ;
            if ( !Kokkos::isfinite(E_eq) || E_eq < E_atmo ) E_eq = E_atmo ;
            if ( !Kokkos::isfinite(N_eq) || N_eq < N_atmo ) N_eq = N_atmo ;
            E = w * E_eq + (1.-w) * E_atmo ;
            N = w * N_eq + (1.-w) * N_atmo ;
        } ;

        #if GRACE_M1_NU_SPECIES >= 1
        eq_blend(aux(VEC(i,j,k),KAPPAA1_,q), aux(VEC(i,j,k),KAPPAS1_,q),
                 aux(VEC(i,j,k),ETA1_,q),
                 aux(VEC(i,j,k),ETAN1_,q), aux(VEC(i,j,k),KAPPAAN1_,q),
                 id.erad1, id.nrad1) ;
        id.fradx1 = id.frady1 = id.fradz1 = 0. ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 3
        eq_blend(aux(VEC(i,j,k),KAPPAA2_,q), aux(VEC(i,j,k),KAPPAS2_,q),
                 aux(VEC(i,j,k),ETA2_,q),
                 aux(VEC(i,j,k),ETAN2_,q), aux(VEC(i,j,k),KAPPAAN2_,q),
                 id.erad2, id.nrad2) ;
        id.fradx2 = id.frady2 = id.fradz2 = 0. ;
        eq_blend(aux(VEC(i,j,k),KAPPAA3_,q), aux(VEC(i,j,k),KAPPAS3_,q),
                 aux(VEC(i,j,k),ETA3_,q),
                 aux(VEC(i,j,k),ETAN3_,q), aux(VEC(i,j,k),KAPPAAN3_,q),
                 id.erad3, id.nrad3) ;
        id.fradx3 = id.frady3 = id.fradz3 = 0. ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
        eq_blend(aux(VEC(i,j,k),KAPPAA4_,q), aux(VEC(i,j,k),KAPPAS4_,q),
                 aux(VEC(i,j,k),ETA4_,q),
                 aux(VEC(i,j,k),ETAN4_,q), aux(VEC(i,j,k),KAPPAAN4_,q),
                 id.erad4, id.nrad4) ;
        id.fradx4 = id.frady4 = id.fradz4 = 0. ;
        eq_blend(aux(VEC(i,j,k),KAPPAA5_,q), aux(VEC(i,j,k),KAPPAS5_,q),
                 aux(VEC(i,j,k),ETA5_,q),
                 aux(VEC(i,j,k),ETAN5_,q), aux(VEC(i,j,k),KAPPAAN5_,q),
                 id.erad5, id.nrad5) ;
        id.fradx5 = id.frady5 = id.fradz5 = 0. ;
        #endif
        #ifdef GRACE_M1_PHOTONS
        eq_blend(aux(VEC(i,j,k),KAPPAAPH_,q), aux(VEC(i,j,k),KAPPASPH_,q),
                 aux(VEC(i,j,k),ETAPH_,q),
                 aux(VEC(i,j,k),ETANPH_,q), aux(VEC(i,j,k),KAPPAANPH_,q),
                 id.eradph, id.nradph) ;
        id.fradxph = id.fradyph = id.fradzph = 0. ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    var_array_t aux;
    scalar_array_t<GRACE_NSPACEDIM> dx;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
} ;

struct straight_beam_m1_id_t {
    straight_beam_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        id.erad1 = atmo.E_fl ;
        id.fradx1 = id.frady1 = id.fradz1 = 0. ;

        if ( xyz[0] <= -0.25 and
            xyz[1] < 0.0625 and xyz[1] > - 0.0625 and
            xyz[2] < 0.0625 and xyz[2] > - 0.0625) {
            id.erad1 = id.fradx1 = 1.0 ;
        }
        // Unit mean energy: N tracks E (beam and atmosphere alike).
        id.nrad1 = id.erad1 ;
        limit_m1_id_flux(id.erad1, id.fradx1, id.frady1, id.fradz1) ;

        #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = id.erad1 ;
            id.nrad2 = id.nrad1 ;
            id.fradx2 = id.fradx1 ; id.frady2 = id.frady1 ; id.fradz2 = id.fradz1 ;
            id.erad3 = id.erad1 ;
            id.nrad3 = id.nrad1 ;
            id.fradx3 = id.fradx1 ; id.frady3 = id.frady1 ; id.fradz3 = id.fradz1 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = id.erad1 ;
            id.nrad4 = id.nrad1 ;
            id.fradx4 = id.fradx1 ; id.frady4 = id.frady1 ; id.fradz4 = id.fradz1 ;
            id.erad5 = id.erad1 ;
            id.nrad5 = id.nrad1 ;
            id.fradx5 = id.fradx1 ; id.frady5 = id.frady1 ; id.fradz5 = id.fradz1 ;
        #endif
        #ifdef GRACE_M1_PHOTONS
            id.eradph = id.erad1 ;
            id.nradph = id.nrad1 ;
            id.fradxph = id.fradx1 ; id.fradyph = id.frady1 ; id.fradzph = id.fradz1 ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
} ;

struct scattering_diffusion_m1_id_t {
    scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _ks, double _t0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), ks(_ks), t0(_t0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };
        double r2 = SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2]);
        double r = sqrt(r2) ;
        id.erad1 = Kokkos::pow(ks/t0,3./2.) * Kokkos::exp(-3*ks*r2/(4.*t0)) ;

        double Hr = r/(2.*t0) * id.erad1 ;

        id.fradx1 = xyz[0]/r * Hr ;
        id.frady1 = xyz[1]/r * Hr ;
        id.fradz1 = xyz[2]/r * Hr ;
        // Unit mean energy: N tracks E.
        id.nrad1 = id.erad1 ;
        // F/E = r/(2 t0) exceeds 1 beyond r = 2 t0 (diffusion profile tail)
        // limit_m1_id_flux(id.erad1, id.fradx1, id.frady1, id.fradz1) ;

        #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = id.erad1 ;
            id.nrad2 = id.nrad1 ;
            id.fradx2 = id.fradx1 ; id.frady2 = id.frady1 ; id.fradz2 = id.fradz1 ;
            id.erad3 = id.erad1 ;
            id.nrad3 = id.nrad1 ;
            id.fradx3 = id.fradx1 ; id.frady3 = id.frady1 ; id.fradz3 = id.fradz1 ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = id.erad1 ;
            id.nrad4 = id.nrad1 ;
            id.fradx4 = id.fradx1 ; id.frady4 = id.frady1 ; id.fradz4 = id.fradz1 ;
            id.erad5 = id.erad1 ;
            id.nrad5 = id.nrad1 ;
            id.fradx5 = id.fradx1 ; id.frady5 = id.frady1 ; id.fradz5 = id.fradz1 ;
        #endif
        #ifdef GRACE_M1_PHOTONS
            id.eradph = id.erad1 ;
            id.nradph = id.nrad1 ;
            id.fradxph = id.fradx1 ; id.fradyph = id.frady1 ; id.fradzph = id.fradz1 ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
    double ks, t0;
} ;

struct moving_scattering_diffusion_m1_id_t {
    moving_scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _v0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), v0(_v0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        id.erad1 = Kokkos::exp(-9.0*SQR(xyz[0])) ;

        double const W2 = 1./(1-SQR(v0)) ;
        double J = 3.*id.erad1  / (4.*W2-1.);

        id.fradx1 = 4./3. * J * W2 * v0 ;
        id.frady1 = id.fradz1 = 0. ;
        // Unit mean energy: N tracks E.
        id.nrad1 = id.erad1 ;
        // limit_m1_id_flux(id.erad1, id.fradx1, id.frady1, id.fradz1) ;

        #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = id.erad1 ;
            id.nrad2 = id.nrad1 ;
            id.fradx2 = id.fradx1 ; id.frady2 = id.frady1 ; id.fradz2 = id.fradz1 ;
            id.erad3 = id.erad1 ;
            id.nrad3 = id.nrad1 ;
            id.fradx3 = id.fradx1 ; id.frady3 = id.frady1 ; id.fradz3 = id.fradz1 ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = id.erad1 ;
            id.nrad4 = id.nrad1 ;
            id.fradx4 = id.fradx1 ; id.frady4 = id.frady1 ; id.fradz4 = id.fradz1 ;
            id.erad5 = id.erad1 ;
            id.nrad5 = id.nrad1 ;
            id.fradx5 = id.fradx1 ; id.frady5 = id.frady1 ; id.fradz5 = id.fradz1 ;
        #endif
        #ifdef GRACE_M1_PHOTONS
            id.eradph = id.erad1 ;
            id.nradph = id.nrad1 ;
            id.fradxph = id.fradx1 ; id.fradyph = id.frady1 ; id.fradzph = id.fradz1 ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
    double v0;
} ;


struct emitting_sphere_m1_id_t {
    emitting_sphere_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        double r2 = SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) ;
        double r = sqrt(r2) ;

        if ( r < 1. ) {
            id.erad1 = 1. ;
            id.fradx1=id.frady1=id.fradz1 = 0 ;
        } else {
            id.erad1 = 1/r2 ;
            id.fradx1 = 0.5/r2 * xyz[0]/r ;
            id.frady1 = 0.5/r2 * xyz[1]/r ;
            id.fradz1 = 0.5/r2 * xyz[2]/r ;
        }
        // Unit mean energy: N tracks E.
        id.nrad1 = id.erad1 ;
        // limit_m1_id_flux(id.erad1, id.fradx1, id.frady1, id.fradz1) ;

        #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = id.erad1 ;
            id.nrad2 = id.nrad1 ;
            id.fradx2 = id.fradx1 ; id.frady2 = id.frady1 ; id.fradz2 = id.fradz1 ;
            id.erad3 = id.erad1 ;
            id.nrad3 = id.nrad1 ;
            id.fradx3 = id.fradx1 ; id.frady3 = id.frady1 ; id.fradz3 = id.fradz1 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = id.erad1 ;
            id.nrad4 = id.nrad1 ;
            id.fradx4 = id.fradx1 ; id.frady4 = id.frady1 ; id.fradz4 = id.fradz1 ;
            id.erad5 = id.erad1 ;
            id.nrad5 = id.nrad1 ;
            id.fradx5 = id.fradx1 ; id.frady5 = id.frady1 ; id.fradz5 = id.fradz1 ;
        #endif
        #ifdef GRACE_M1_PHOTONS
            id.eradph = id.erad1 ;
            id.nradph = id.nrad1 ;
            id.fradxph = id.fradx1 ; id.fradyph = id.frady1 ; id.fradzph = id.fradz1 ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
} ;

struct curved_beam_m1_id_t {
    curved_beam_m1_id_t(
        m1_atmo_params_t _atmo,
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        var_array_t _state
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), state(_state)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(int const i, int const j, int const k),
        int const q) const
    {
        m1_id_t id ;
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        };

        id.erad1 = atmo.E_fl ; id.nrad1 = atmo.E_fl / atmo.eps_fl ;
        id.fradx1 = id.frady1 = id.fradz1 = 0. ;

        if ( xyz[0] <= 0.015625 and
            xyz[1] < 0.25 and xyz[1] > - 0.25 and
            xyz[2] <= 3.5 and xyz[2] >= 3.0 ) {
            id.erad1 = 1.0 ;
            // F_i F^i = E * E
            metric_array_t metric ;
            FILL_METRIC_ARRAY(metric,this->state,q,i,j,k) ;
            double FY = metric.beta(1) * id.erad1 / metric.alp() ;
            double FZ = metric.beta(2) * id.erad1 / metric.alp() ;

            double const a = metric.gamma(0) ;
            double const b = 2 * FY * metric.gamma(1) + 2 * FZ * metric.gamma(2) ;
            double const c = - 0.9999 * SQR(id.erad1) + SQR(FY) * metric.gamma(3) + SQR(FZ) * metric.gamma(5) + 2. * FY * FZ * metric.gamma(4) ;

            double const FX = (-b + sqrt(SQR(b)-4.*a*c))/(2.*a) ;
            auto Fd = metric.lower({FX,FY,FZ}) ;
            id.fradx1 = Fd[0] ; id.frady1 = Fd[1] ; id.fradz1 = Fd[2] ;
        }

        #if GRACE_M1_NU_SPECIES >= 3
            id.erad2 = id.erad1 ;
            id.nrad2 = id.nrad1 ;
            id.fradx2 = id.fradx1 ; id.frady2 = id.frady1 ; id.fradz2 = id.fradz1 ;
            id.erad3 = id.erad1 ;
            id.nrad3 = id.nrad1 ;
            id.fradx3 = id.fradx1 ; id.frady3 = id.frady1 ; id.fradz3 = id.fradz1 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            id.erad4 = id.erad1 ;
            id.nrad4 = id.nrad1 ;
            id.fradx4 = id.fradx1 ; id.frady4 = id.frady1 ; id.fradz4 = id.fradz1 ;
            id.erad5 = id.erad1 ;
            id.nrad5 = id.nrad1 ;
            id.fradx5 = id.fradx1 ; id.frady5 = id.frady1 ; id.fradz5 = id.fradz1 ;
        #endif
        #ifdef GRACE_M1_PHOTONS
            id.eradph = id.erad1 ;
            id.nradph = id.nrad1 ;
            id.fradxph = id.fradx1 ; id.fradyph = id.frady1 ; id.fradzph = id.fradz1 ;
        #endif

        return id ;
    }

    m1_atmo_params_t atmo ;
    m1_excision_params_t excision ;
    var_array_t state ;
    coord_array_t<GRACE_NSPACEDIM> pcoords ;
} ;

} /* namespace grace */
#endif /*GRACE_PHYSICS_ID_M1_HH*/
