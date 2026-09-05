/**
 * @file lbm.hh
 * @brief Lattice-Boltzmann radiation transport: per-direction intensities
 *        carried as cell-centred evolved variables, advanced once per step by
 *        a semi-Lagrangian streaming sweep and an implicit collision step.
 * @date 2026-09-05
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
 */

#ifndef GRACE_PHYSICS_LBM_HH
#define GRACE_PHYSICS_LBM_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_LBM

#ifndef GRACE_3D
#error "The Lattice-Boltzmann radiation module needs a 3-D build: its stencils are 3-D direction sets."
#endif
#if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_COWLING
#error "LBM (flat_fixed streaming) is implemented for GRACE_METRIC_EVOL=COWLING only: the pull-back x - n dt is a straight line, which is exact in flat space. Geodesic (curved) streaming is the next milestone."
#endif

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/metric_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/lbm_stencil.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace lbm {

/**
 * \defgroup lbm Lattice-Boltzmann radiation transport
 *
 * Port of the general-relativistic Lattice-Boltzmann method of Olsen &
 * Rezzolla (2025, arXiv:2502.17552; reference code
 * github.com/Tom-Olsen/3dRadiation).  The specific intensity `I_d` along each
 * of the GRACE_LBM_NDIR stencil directions is a cell-centred evolved
 * variable (variable_indices.hh, LBM_I0_ .. LBM_IEND_), inert to every MoL
 * stepper and advanced once per timestep by `step()`:
 *
 *   stream  : I_d(x) <- I_d(x - n_d dt), 8-point trilinear pull-back
 *             (reference StreamFlatFixed).  |n_d| dt <= CFL dx_finest, so the
 *             departure point is at most one cell away and the 4 ghost layers
 *             GRACE exchanges are ample.
 *   collide : implicit in I_d, in the fluid frame (reference Collide()).  This
 *             milestone is static-fluid and isotropic-scattering only, where
 *             the reference's lambda iteration on the moments has an exact
 *             closed form -- see system_t::collide.
 *   moments : E = sum_d w_d I_d, F^i = sum_d w_d I_d n^i_d, P^ij likewise,
 *             written into the M1 slots Erad* and Frad* (densitised, lowered) and
 *             the "lbm" aux group, so every downstream consumer of the M1
 *             moments works unchanged.
 *
 * Everything else -- species count, opacities from the m1.eas providers,
 * m1.atmosphere floors, the m1.trigger activation gate -- is shared with M1.
 *
 * Frame: in flat space the Eulerian tetrad is the identity, so the stencil
 * frame IS the coordinate frame and the reference's IF<->LF transforms drop
 * out.  The curved-space extension must rotate F^a, P^ab with the tetrad
 * before lowering.
 */

//---------------------------------------------------------------------------
// Index helpers.  Species s runs over the neutrino species and, if built, the
// photon block as the LAST index -- the same layout M1_PHOTON_SPECIES uses.
//---------------------------------------------------------------------------
GRACE_HOST_DEVICE constexpr int nspecies() { return GRACE_LBM_NSPECIES ; }
GRACE_HOST_DEVICE constexpr int ndir()     { return GRACE_LBM_NDIR ; }

//! Evolved slot of population d of species s.
GRACE_HOST_DEVICE constexpr int idx(int s, int d) { return LBM_I0_ + s*GRACE_LBM_NDIR + d ; }
//! Aux slot of pressure component c (xx,xy,xz,yy,yz,zz) of species s.
GRACE_HOST_DEVICE constexpr int pidx(int s, int c) { return LBM_P0_ + 6*s + c ; }

// The M1 moment / rate slots of LBM species s.  Same arithmetic as the
// GRACE_M1_IDX_FN templates in m1_helpers.hh, but with a RUNTIME species index
// so the kernels can loop over species.
GRACE_HOST_DEVICE constexpr int erad_idx(int s) {
    #ifdef GRACE_M1_PHOTONS
    if ( s == GRACE_M1_NU_SPECIES ) return ERADPH_ ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 1
    return ERAD1_ + s*GRACE_N_M1_VARS ;
    #else
    return -1 ;   // unreachable: photon-only builds only ever pass s == GRACE_M1_NU_SPECIES
    #endif
}
GRACE_HOST_DEVICE constexpr int nrad_idx (int s) { return erad_idx(s) + 1 ; }   // E,N,Fx,Fy,Fz are contiguous
GRACE_HOST_DEVICE constexpr int fradx_idx(int s) { return erad_idx(s) + 2 ; }

GRACE_HOST_DEVICE constexpr int kappaa_idx(int s) {
    #ifdef GRACE_M1_PHOTONS
    if ( s == GRACE_M1_NU_SPECIES ) return KAPPAAPH_ ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 1
    return KAPPAA1_ + s*GRACE_N_M1_AUX ;
    #else
    return -1 ;
    #endif
}
GRACE_HOST_DEVICE constexpr int kappas_idx(int s) {
    #ifdef GRACE_M1_PHOTONS
    if ( s == GRACE_M1_NU_SPECIES ) return KAPPASPH_ ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 1
    return KAPPAS1_ + s*GRACE_N_M1_AUX ;
    #else
    return -1 ;
    #endif
}
GRACE_HOST_DEVICE constexpr int eta_idx(int s) {
    #ifdef GRACE_M1_PHOTONS
    if ( s == GRACE_M1_NU_SPECIES ) return ETAPH_ ;
    #endif
    #if GRACE_M1_NU_SPECIES >= 1
    return ETA1_ + s*GRACE_N_M1_AUX ;
    #else
    return -1 ;
    #endif
}

//---------------------------------------------------------------------------
// Parameters (host-read once, captured by value).
//---------------------------------------------------------------------------
struct params_t {
    double lambda_tol ;       //!< lbm.lambda_tolerance
    int    max_lambda_iter ;  //!< lbm.max_lambda_iterations
    double I_fl ;             //!< isotropic intensity floor = m1.atmosphere.E_fl (sum_d w_d = 1)
    double eps_fl ;           //!< m1.atmosphere.eps_fl, N = E/eps_fl (grey: unit mean energy)
} ;

inline params_t get_params() {
    params_t p ;
    p.lambda_tol      = grace::get_param<double>("lbm","lambda_tolerance") ;
    p.max_lambda_iter = grace::get_param<int>   ("lbm","max_lambda_iterations") ;
    p.I_fl            = grace::get_param<double>("m1","atmosphere","E_fl") ;
    p.eps_fl          = grace::get_param<double>("m1","atmosphere","eps_fl") ;
    return p ;
}

//---------------------------------------------------------------------------
// Moments -> intensities: the reference's von Mises-Fisher ansatz
// (SpecialMath.cpp Intensity / FluxMagnitude): for a given E and flux ratio
// f = |F|/E, I(n) = E exp(sigma n.F^ - ln(sinh sigma / sigma)) with sigma the
// root of f = coth(sigma) - 1/sigma.  Used to seed the populations from the
// M1 initial-data moments.
//---------------------------------------------------------------------------
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
flux_of_sigma(double sigma) {
    if ( sigma < 1e-10 ) return 0.0 ;
    return (sigma*Kokkos::cosh(sigma) - Kokkos::sinh(sigma)) / (sigma*Kokkos::sinh(sigma)) ;
}

//! Invert flux_of_sigma by bisection.  f is clamped to flux_of_sigma(60)
//! ~ 0.983 -- beyond that the distribution is a near-delta the stencil cannot
//! resolve anyway.
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
sigma_of_relative_flux(double f) {
    constexpr double sigma_max = 60.0 ;
    if ( !(f > 1e-12) ) return 0.0 ;
    f = Kokkos::fmin(f, flux_of_sigma(sigma_max)) ;
    double lo = 0.0, hi = sigma_max ;
    for ( int it = 0; it < 60; ++it ) {
        double const mid = 0.5*(lo+hi) ;
        if ( flux_of_sigma(mid) < f ) lo = mid ; else hi = mid ;
    }
    return 0.5*(lo+hi) ;
}

double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
vmf_intensity(double sigma, double E, double ndotf) {
    if ( sigma < 1e-10 ) return E ;
    return E * Kokkos::exp(sigma*ndotf - Kokkos::log(Kokkos::sinh(sigma)/sigma)) ;
}

//---------------------------------------------------------------------------
// The per-cell operator.  Value type, captured into KOKKOS_LAMBDAs.
//---------------------------------------------------------------------------
struct system_t {
    var_array_t I_old ;   //!< populations at t^n (ghosts valid) -- read only
    var_array_t I_new ;   //!< populations at t^{n+1}; also carries the metric and the M1 moment slots
    var_array_t aux ;     //!< rates in, pressure tensor / iteration count out
    stencil_t   st ;
    scalar_array_t<GRACE_NSPACEDIM> dx ;
    params_t    p ;

    /**
     * @brief Streaming: semi-Lagrangian pull-back along straight lines.
     *
     * For every population the departure point is x - n_d dt, at most one cell
     * away for CFL <= 1 (also on coarser AMR levels, where dt is set by the
     * finest dx).  8-point trilinear interpolation of I_old at that point.
     * Clamped from below at the isotropic floor: second-order prolongation can
     * undershoot at coarse-fine faces and the reference does max(0,.) too.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    stream(int const i, int const j, int const k, int64_t const q, double const dt) const
    {
        double const ooh = 1.0 / dx(0,q) ;
        for ( int s = 0; s < nspecies(); ++s )
        for ( int d = 0; d < ndir(); ++d ) {
            // displacement of the departure point, in cell units
            double const ux = -st.cx(d) * dt * ooh ;
            double const uy = -st.cy(d) * dt * ooh ;
            double const uz = -st.cz(d) * dt * ooh ;
            int const ai = static_cast<int>(Kokkos::floor(ux)) ;
            int const aj = static_cast<int>(Kokkos::floor(uy)) ;
            int const ak = static_cast<int>(Kokkos::floor(uz)) ;
            double const fx = ux - ai, fy = uy - aj, fz = uz - ak ;
            int const i0 = i + ai, j0 = j + aj, k0 = k + ak ;
            int const iv = idx(s,d) ;
            double const v =
                  (1.-fx)*(1.-fy)*(1.-fz) * I_old(i0  ,j0  ,k0  ,iv,q)
                + (1.-fx)*(1.-fy)*    fz  * I_old(i0  ,j0  ,k0+1,iv,q)
                + (1.-fx)*    fy *(1.-fz) * I_old(i0  ,j0+1,k0  ,iv,q)
                + (1.-fx)*    fy *    fz  * I_old(i0  ,j0+1,k0+1,iv,q)
                +     fx *(1.-fy)*(1.-fz) * I_old(i0+1,j0  ,k0  ,iv,q)
                +     fx *(1.-fy)*    fz  * I_old(i0+1,j0  ,k0+1,iv,q)
                +     fx *    fy *(1.-fz) * I_old(i0+1,j0+1,k0  ,iv,q)
                +     fx *    fy *    fz  * I_old(i0+1,j0+1,k0+1,iv,q) ;
            I_new(i,j,k,iv,q) = Kokkos::fmax(v, p.I_fl) ;
        }
    }

    /**
     * @brief Moments of species s at a cell from populations I (stencil frame).
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    moments(var_array_t const& I, int const i, int const j, int const k, int64_t const q,
            int const s, double& E, double (&F)[3], double (&P)[6]) const
    {
        E = 0. ;
        F[0] = F[1] = F[2] = 0. ;
        for ( int c = 0; c < 6; ++c ) P[c] = 0. ;
        for ( int d = 0; d < ndir(); ++d ) {
            double const wI = st.weight(d) * I(i,j,k,idx(s,d),q) ;
            double const cx = st.cx(d), cy = st.cy(d), cz = st.cz(d) ;
            E    += wI ;
            F[0] += wI*cx ; F[1] += wI*cy ; F[2] += wI*cz ;
            P[0] += wI*cx*cx ; P[1] += wI*cx*cy ; P[2] += wI*cx*cz ;
            P[3] += wI*cy*cy ; P[4] += wI*cy*cz ; P[5] += wI*cz*cz ;
        }
    }

    /**
     * @brief Collision, implicit in I_d, in place on I_new.  Returns the
     *        number of lambda iterations used.
     *
     * Reference Collide() (Radiation.cpp:768-882) with the milestone-1
     * restrictions alpha = 1, u^i = 0 (Doppler factor A = 1) and kappa_1 = 0:
     *
     *   I*_d = ( I_d + dt (eta + kappa_0 E*) ) / ( 1 + dt (kappa_a + kappa_0) ).
     *
     * The reference iterates this because the scattering term depends on the
     * post-collision moments.  Here it depends on E* alone, and summing the
     * update with the weights (sum w_d = 1) gives the fixed point in closed
     * form, E* = (E + dt eta)/(1 + dt kappa_a): scattering conserves energy,
     * absorption and emission relax it.  One pass, exact -- the general
     * iteration is only needed once kappa_1 or a moving fluid couples the
     * directions, and that is where lbm.max_lambda_iterations comes in.
     *
     * kappa_0 is the isotropic scattering opacity (M1's kappa_s), kappa_a the
     * absorption opacity, eta the emissivity -- all code units from the EAS
     * provider, the same slots M1's implicit update reads.
     */
    int GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    collide(int const i, int const j, int const k, int64_t const q, double const dt) const
    {
        int niter = 0 ;
        for ( int s = 0; s < nspecies(); ++s ) {
            double const ka  = aux(i,j,k,kappaa_idx(s),q) ;
            double const ks  = aux(i,j,k,kappas_idx(s),q) ;
            double const eta = aux(i,j,k,eta_idx(s),q) ;
            // All rates at the floor: the update is the identity (m1.hh:437 does the same).
            if ( ka < 1e-30 && ks < 1e-30 && eta < 1e-30 ) continue ;

            double E, F[3], P[6] ;
            moments(I_new, i,j,k,q, s, E, F, P) ;
            double const Estar   = (E + dt*eta) / (1.0 + dt*ka) ;
            double const num     = dt * (eta + ks*Estar) ;
            double const oodenom = 1.0 / (1.0 + dt*(ka + ks)) ;
            for ( int d = 0; d < ndir(); ++d ) {
                int const iv = idx(s,d) ;
                I_new(i,j,k,iv,q) = (I_new(i,j,k,iv,q) + num) * oodenom ;
            }
            niter = 1 ;
        }
        return niter ;
    }

    /**
     * @brief Quadrature moments of I_new into the M1 slots and the "lbm" aux.
     *
     * Erad = sqrt(gamma) E, Frad_i = sqrt(gamma) gamma_ij F^j -- M1's
     * densitised, lowered convention -- and Nrad = Erad/eps_fl (grey
     * transport has no separate number density; unit mean energy, as the M1
     * test initial data assume).  Flat space: no tetrad rotation needed.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    write_moments(int const i, int const j, int const k, int64_t const q) const
    {
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, I_new, q, i,j,k) ;
        double const sg = metric.sqrtg() ;
        for ( int s = 0; s < nspecies(); ++s ) {
            double E, F[3], P[6] ;
            moments(I_new, i,j,k,q, s, E, F, P) ;
            auto const Fd = metric.lower({F[0],F[1],F[2]}) ;
            I_new(i,j,k,erad_idx(s)   ,q) = sg * E ;
            I_new(i,j,k,nrad_idx(s)   ,q) = sg * E / p.eps_fl ;
            I_new(i,j,k,fradx_idx(s)  ,q) = sg * Fd[0] ;
            I_new(i,j,k,fradx_idx(s)+1,q) = sg * Fd[1] ;
            I_new(i,j,k,fradx_idx(s)+2,q) = sg * Fd[2] ;
            for ( int c = 0; c < 6; ++c ) aux(i,j,k,pidx(s,c),q) = P[c] ;
        }
    }
} ;

//---------------------------------------------------------------------------
// Drivers (lbm.cpp)
//---------------------------------------------------------------------------

/**
 * @brief Validate the configuration and load the stencil.  Called once at
 *        startup, after the M1 trigger check.
 *
 * Refuses reflection symmetries (populations are registered as scalars, but a
 * reflection maps I_d to the mirrored direction -- not yet implemented) and
 * a non-flat background (straight-line streaming).
 */
void startup_check() ;

/// Print the accumulated stream / collide / copy wall-clock split (lbm.report_timings).
void report_timings() ;

/**
 * @brief Seed the populations from the moments the M1 initial data left in
 *        Erad* and Frad* (von Mises-Fisher ansatz), over the full padded extent so
 *        the outer ghosts hold the no-incoming-radiation floor, then rewrite
 *        the moment slots from the quadrature for consistency.
 */
template < typename eos_t >
void set_initial_data() ;

/**
 * @brief One LBM step: stream I_old -> I_new, collide, write moments, then
 *        copy the LBM block and the moment slots I_new -> I_old.
 *
 * Called once per timestep from evolve_impl() right after the
 * `deep_copy(state_p, state)`, with I_old = state_p and I_new = state.  The
 * closing copy keeps the invariant every stepper relies on for inert
 * variables: both y^n registers hold I^{n+1} before the first substage, so
 * the stage blends (rk3 linop, rk4 s3 = s2, imex222 copy) reproduce it
 * bit-exactly.  Gated by the M1 activation trigger.
 */
template < typename eos_t >
void step( var_array_t& I_old, var_array_t& I_new, var_array_t& aux, double dt ) ;

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                   \
extern template void set_initial_data<EOS>() ;                      \
extern template void step<EOS>( var_array_t&, var_array_t&, var_array_t&, double )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::leptonic_eos_4d_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE

}} // namespace grace::lbm

#endif // GRACE_ENABLE_LBM
#endif // GRACE_PHYSICS_LBM_HH
