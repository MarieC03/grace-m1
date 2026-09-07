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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/lbm_geometry.hh>
#include <grace/physics/lbm_geodesic.hh>
#include <grace/physics/lbm_velocity_mesh.hh>
#include <grace/coordinates/coordinate_systems.hh>
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
 *             GRACE exchanges are ample.  On a curved metric the pull-back
 *             follows the null geodesic (stream_curved) and, by default, moves
 *             the Killing energy of each bin with claim-normalised weights so
 *             that it is conserved (lbm.curved_remap).
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
 * Frame: the stencil directions live in the Eulerian observer's orthonormal
 * triad (lbm_geometry.hh triad_t; the reference's "IF"), so E is the Eulerian
 * energy density directly while F^a and P^ab are rotated to coordinate
 * components with e^i_a before lowering.  The triad is the identity in flat
 * space, bit-exactly.
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
    int    streaming ;        //!< 0 flat_fixed, 1 curved_fixed (lbm.streaming)
    bool   no_inflow ;        //!< lbm.bc_kind = outgoing: departure points outside the domain give the floor
    double dom_lo[3], dom_hi[3] ; //!< physical domain (amr.{x,y,z}{min,max})
    double geodesic_tol ;     //!< lbm.geodesic_tolerance
    int    geom_every ;       //!< lbm.geometry_update_every (Z4: rebuild the geodesic map every N steps)
    bool   conservative ;     //!< lbm.curved_remap = conservative: claim-normalised (Killing-energy conserving) sweep
    int    lut_nth, lut_nph ; //!< lbm.velocity_lut
    excision_t ex ;           //!< grmhd.excision (shared with M1)
} ;

inline params_t get_params() {
    params_t p ;
    p.lambda_tol      = grace::get_param<double>("lbm","lambda_tolerance") ;
    p.max_lambda_iter = grace::get_param<int>   ("lbm","max_lambda_iterations") ;
    p.I_fl            = grace::get_param<double>("m1","atmosphere","E_fl") ;
    p.eps_fl          = grace::get_param<double>("m1","atmosphere","eps_fl") ;
    p.streaming       = grace::get_param<std::string>("lbm","streaming") == "curved_fixed" ? 1 : 0 ;
    p.no_inflow       = grace::get_param<std::string>("lbm","bc_kind") == "outgoing" ;
    p.dom_lo[0] = grace::get_param<double>("amr","xmin") ; p.dom_hi[0] = grace::get_param<double>("amr","xmax") ;
    p.dom_lo[1] = grace::get_param<double>("amr","ymin") ; p.dom_hi[1] = grace::get_param<double>("amr","ymax") ;
    p.dom_lo[2] = grace::get_param<double>("amr","zmin") ; p.dom_hi[2] = grace::get_param<double>("amr","zmax") ;
    p.geodesic_tol    = grace::get_param<double>("lbm","geodesic_tolerance") ;
    p.geom_every      = grace::get_param<int>("lbm","geometry_update_every") ;
    p.conservative    = grace::get_param<std::string>("lbm","curved_remap") == "conservative" ;
    p.lut_nth         = grace::get_param<int>("lbm","velocity_lut","n_theta") ;
    p.lut_nph         = grace::get_param<int>("lbm","velocity_lut","n_phi") ;
    p.ex.by_radius    = grace::get_param<std::string>("grmhd","excision","excision_criterion") == "radius" ;
    p.ex.r_ex         = grace::get_param<double>("grmhd","excision","excision_radius") ;
    p.ex.alp_ex       = grace::get_param<double>("grmhd","excision","excision_lapse") ;
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

//! Invert flux_of_sigma by bisection.  f is clamped to flux_of_sigma(sigma_max),
//! the sharpest the quadrature carries (lbm_stencil.hh vmf_sigma_max, derived
//! from the measured exactness degree) -- beyond that it is a near-delta the stencil cannot
//! resolve anyway.
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
sigma_of_relative_flux(double f, double const sigma_max) {
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
    velocity_interp_t vi ;              //!< off-stencil interpolation (curved streaming)
    device_coordinate_system coords ;   //!< cell centres (departure-point tests, curved streaming)

    //! Departure point outside the physical domain with lbm.bc_kind = outgoing: nothing comes in.
    bool GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE no_incoming(double const xd[3]) const
    {
        if ( !p.no_inflow ) return false ;
        for ( int a = 0; a < 3; ++a ) if ( xd[a] < p.dom_lo[a] || xd[a] > p.dom_hi[a] ) return true ;
        return false ;
    }

    //! Departure data of one arrival population: corners and trilinear weights
    //! of the departure point, the three source directions with their
    //! barycentric weights, the frequency factor S and the interpolated lapse.
    struct departure_t { int i0[3] ; double w[8] ; int vid[3] ; double lam[3] ; double S, alp_dep ; } ;

    //! Floor of alpha - beta.n in the conservative remap (the Killing energy of a
    //! photon changes sign inside a horizon; those cells are excised anyway).
    static constexpr double kappa_min = 1e-2 ;

    /**
     * @brief Departure of population (i,j,k,d) from the spherical-harmonic map in
     *        aux.  Returns false when the arrival gets the floor: departure point
     *        outside the padded block, inside the excision, or outside the domain
     *        with lbm.bc_kind = outgoing.
     */
    bool GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    departure(int const i, int const j, int const k, int64_t const q, int const d,
              double const (&coef)[sh_nquant][sh_ncoef], double const xc[3], departure_t& dep) const
    {
        double Y[sh_ncoef] ;
        for ( int c = 0; c < sh_ncoef; ++c ) Y[c] = st.Ysh(d,c) ;
        dep.S = sh_eval(coef[SH_S], Y) ;
        double const disp[3] = { sh_eval(coef[SH_DX],Y), sh_eval(coef[SH_DY],Y), sh_eval(coef[SH_DZ],Y) } ;
        double C[3] = { sh_eval(coef[SH_CX],Y), sh_eval(coef[SH_CY],Y), sh_eval(coef[SH_CZ],Y) } ;
        double const cn = Kokkos::sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]) ;
        if ( cn > 1e-300 ) { C[0] /= cn ; C[1] /= cn ; C[2] /= cn ; }
        else { C[0] = st.cx(d) ; C[1] = st.cy(d) ; C[2] = st.cz(d) ; }

        double const ooh = 1.0 / dx(0,q) ;
        int const nmax[3] = { static_cast<int>(I_old.extent(0)), static_cast<int>(I_old.extent(1)), static_cast<int>(I_old.extent(2)) } ;
        double const u[3] = { i + disp[0]*ooh, j + disp[1]*ooh, k + disp[2]*ooh } ;
        double f[3] ;
        for ( int a = 0; a < 3; ++a ) {
            dep.i0[a] = static_cast<int>(Kokkos::floor(u[a])) ;
            f[a] = u[a] - dep.i0[a] ;
            if ( dep.i0[a] < 0 || dep.i0[a] + 1 >= nmax[a] ) return false ;   // cannot happen for CFL <= 1
        }
        dep.alp_dep = 0. ;
        for ( int cc = 0; cc < 8; ++cc ) {
            int const di = (cc>>2)&1, dj = (cc>>1)&1, dk = cc&1 ;
            dep.w[cc] = (di ? f[0] : 1.-f[0]) * (dj ? f[1] : 1.-f[1]) * (dk ? f[2] : 1.-f[2]) ;
            dep.alp_dep += dep.w[cc] * I_old(dep.i0[0]+di, dep.i0[1]+dj, dep.i0[2]+dk, ALP_, q) ;
        }
        double const xd[3] = { xc[0]+disp[0], xc[1]+disp[1], xc[2]+disp[2] } ;
        if ( p.ex.inside(xd, dep.alp_dep) || no_incoming(xd) ) return false ;
        vi.locate(C, dep.vid, dep.lam) ;
        return true ;
    }

    /**
     * @brief Claim pass of the conservative remap: every arrival population adds
     *        its interpolation weight, times the phase-space volume ratio of the
     *        bins, S^3 sqrt(gamma_i) w_d / (sqrt(gamma_c) w_d'), to the claim
     *        weight W of each source bin (aux LBM_WCLAIM_).  By Liouville the
     *        exact sum is 1; dividing the sources by W makes the discrete Killing
     *        energy conserved.  Runs over the interior plus a two-cell band of
     *        ghosts (whose map is exact, see LBM_METRIC_DER_ORDER), in 27
     *        colours so that no two concurrent arrivals touch the same source.
     *        Excised arrivals claim too: what they receive is absorbed.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    claim_curved(int const i, int const j, int const k, int64_t const q) const
    {
        double coef[sh_nquant][sh_ncoef] ;
        for ( int f = 0; f < sh_nquant; ++f ) for ( int c = 0; c < sh_ncoef; ++c )
            coef[f][c] = aux(i,j,k,LBM_SH_ + sh_ncoef*f + c, q) ;
        double xc[3] ; coords.get_physical_coordinates(i,j,k,q,xc) ;
        double const sg_i = aux(i,j,k,LBM_SQRTG_,q) ;
        if ( sg_i <= 0. ) return ;   // outside the band where the geometry is defined
        for ( int d = 0; d < ndir(); ++d ) {
            departure_t dep ;
            if ( !departure(i,j,k,q,d,coef,xc,dep) ) continue ;
            double const S3wd = dep.S*dep.S*dep.S * sg_i * st.weight(d) ;
            for ( int cc = 0; cc < 8; ++cc ) {
                int const ci = dep.i0[0] + ((cc>>2)&1), cj = dep.i0[1] + ((cc>>1)&1), ck = dep.i0[2] + (cc&1) ;
                double const sg_c = aux(ci,cj,ck,LBM_SQRTG_,q) ;
                if ( sg_c <= 0. ) continue ;
                for ( int kk = 0; kk < 3; ++kk ) {
                    int const dp = dep.vid[kk] ;
                    aux(ci,cj,ck,LBM_WCLAIM_+dp,q) += dep.w[cc] * dep.lam[kk] * S3wd / (sg_c * st.weight(dp)) ;
                }
            }
        }
    }

    /**
     * @brief Streaming along null geodesics (lbm.streaming = curved_fixed).
     *
     * Per direction the l <= 2 spherical-harmonic fit stored in aux
     * (LBM_SH_, from the geometry pass) gives the frequency factor
     * S = nu_here/nu_dep, the departure displacement and the departure
     * direction C in the local triad; the corner populations are interpolated
     * at C in velocity space (exact at stencil directions, so straight rays
     * reproduce stream()).  Two forms (lbm.curved_remap):
     *
     *  conservative: the Killing energy of a bin, sqrt(gamma) w (alpha - beta.n) I,
     *    is what moves; each source bin is shared out by its claim weight W
     *    (claim_curved), so the discrete sum over the grid is conserved:
     *      I_new(d) = S^3 / kappa_d  sum_c w_c sum_k lambda_k kappa_c(d'_k) I_c(d'_k) / W_c(d'_k),
     *    kappa = alpha - beta_i e^i_a n^a.  In flat space W = 1 to round-off.
     *  pointwise: the characteristic solution, alpha^4 I interpolated (constant
     *    along a ray on a stationary background) and scaled by S^4; not
     *    conservative (first-order loss of Killing energy on a curved metric).
     *
     * Excised cells and departure points inside the excision give the floor.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    stream_curved(int const i, int const j, int const k, int64_t const q, double const) const
    {
        if ( static_cast<int>(aux(i,j,k,LBM_GEOM_FLAG_,q)) == GEOM_EXCISED ) {
            for ( int s = 0; s < nspecies(); ++s ) for ( int d = 0; d < ndir(); ++d ) I_new(i,j,k,idx(s,d),q) = p.I_fl ;
            return ;
        }
        double coef[sh_nquant][sh_ncoef] ;
        for ( int f = 0; f < sh_nquant; ++f ) for ( int c = 0; c < sh_ncoef; ++c )
            coef[f][c] = aux(i,j,k,LBM_SH_ + sh_ncoef*f + c, q) ;
        double xc[3] ; coords.get_physical_coordinates(i,j,k,q,xc) ;
        double const alp_i = I_old(i,j,k,ALP_,q) ;
        double const bi[3] = { aux(i,j,k,LBM_BTRIAD_,q), aux(i,j,k,LBM_BTRIAD_+1,q), aux(i,j,k,LBM_BTRIAD_+2,q) } ;
        for ( int d = 0; d < ndir(); ++d ) {
            departure_t dep ;
            if ( !departure(i,j,k,q,d,coef,xc,dep) ) {
                for ( int s = 0; s < nspecies(); ++s ) I_new(i,j,k,idx(s,d),q) = p.I_fl ;
                continue ;
            }
            if ( p.conservative ) {
                double const kap_i = Kokkos::fmax(alp_i - (bi[0]*st.cx(d) + bi[1]*st.cy(d) + bi[2]*st.cz(d)), kappa_min) ;
                double const fac = dep.S*dep.S*dep.S / kap_i ;
                double v[GRACE_LBM_NSPECIES] ;
                for ( int s = 0; s < nspecies(); ++s ) v[s] = 0. ;
                for ( int cc = 0; cc < 8; ++cc ) {
                    int const ci = dep.i0[0] + ((cc>>2)&1), cj = dep.i0[1] + ((cc>>1)&1), ck = dep.i0[2] + (cc&1) ;
                    double const alp_c = I_old(ci,cj,ck,ALP_,q) ;
                    double const bc[3] = { aux(ci,cj,ck,LBM_BTRIAD_,q), aux(ci,cj,ck,LBM_BTRIAD_+1,q), aux(ci,cj,ck,LBM_BTRIAD_+2,q) } ;
                    for ( int kk = 0; kk < 3; ++kk ) {
                        int const dp = dep.vid[kk] ;
                        double const kap_c = Kokkos::fmax(alp_c - (bc[0]*st.cx(dp) + bc[1]*st.cy(dp) + bc[2]*st.cz(dp)), kappa_min) ;
                        double const Wc = Kokkos::fmax(aux(ci,cj,ck,LBM_WCLAIM_+dp,q), 1e-300) ;
                        double const g = dep.w[cc] * dep.lam[kk] * kap_c / Wc ;
                        for ( int s = 0; s < nspecies(); ++s ) v[s] += g * I_old(ci,cj,ck,idx(s,dp),q) ;
                    }
                }
                for ( int s = 0; s < nspecies(); ++s ) I_new(i,j,k,idx(s,d),q) = Kokkos::fmax(fac*v[s], p.I_fl) ;
            } else {
                double a4[8] ;
                for ( int cc = 0; cc < 8; ++cc ) {
                    double const al = I_old(dep.i0[0]+((cc>>2)&1), dep.i0[1]+((cc>>1)&1), dep.i0[2]+(cc&1), ALP_, q) ;
                    a4[cc] = al*al*al*al ;
                }
                double const fac = dep.S*dep.S*dep.S*dep.S / (dep.alp_dep*dep.alp_dep*dep.alp_dep*dep.alp_dep) ;
                for ( int s = 0; s < nspecies(); ++s ) {
                    int const iv0 = idx(s,dep.vid[0]), iv1 = idx(s,dep.vid[1]), iv2 = idx(s,dep.vid[2]) ;
                    double v = 0. ;
                    for ( int cc = 0; cc < 8; ++cc ) {
                        int const ci = dep.i0[0] + ((cc>>2)&1), cj = dep.i0[1] + ((cc>>1)&1), ck = dep.i0[2] + (cc&1) ;
                        v += dep.w[cc]*a4[cc]*( dep.lam[0]*I_old(ci,cj,ck,iv0,q)
                                              + dep.lam[1]*I_old(ci,cj,ck,iv1,q)
                                              + dep.lam[2]*I_old(ci,cj,ck,iv2,q) ) ;
                    }
                    I_new(i,j,k,idx(s,d),q) = Kokkos::fmax(fac*v, p.I_fl) ;
                }
            }
        }
    }

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
        double xc[3] = {0.,0.,0.} ;
        if ( p.no_inflow ) coords.get_physical_coordinates(i,j,k,q,xc) ;
        for ( int s = 0; s < nspecies(); ++s )
        for ( int d = 0; d < ndir(); ++d ) {
            if ( p.no_inflow ) {
                double const xd[3] = { xc[0] - st.cx(d)*dt, xc[1] - st.cy(d)*dt, xc[2] - st.cz(d)*dt } ;
                if ( no_incoming(xd) ) { I_new(i,j,k,idx(s,d),q) = p.I_fl ; continue ; }
            }
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
    collide(int const i, int const j, int const k, int64_t const q, double const dt_coord) const
    {
        int niter = 0 ;
        // Proper time of the Eulerian observer elapses at the lapse rate (reference: alpha*dt).
        double const dt = dt_coord * I_new(i,j,k,ALP_,q) ;
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
     * test initial data assume).  F^a and P^ab come out of the quadrature in
     * the triad frame and are rotated to coordinate components first; the
     * "lbm" aux gets P^ij and the Killing energy density.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    write_moments(int const i, int const j, int const k, int64_t const q) const
    {
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, I_new, q, i,j,k) ;
        triad_t const tr(metric) ;
        double const sg = metric.sqrtg() ;
        for ( int s = 0; s < nspecies(); ++s ) {
            double E, F[3], P[6], Fu[3], Pu[6] ;
            moments(I_new, i,j,k,q, s, E, F, P) ;
            tr.to_coord(F, Fu) ;
            tr.to_coord_sym(P, Pu) ;
            auto const Fd = metric.lower({Fu[0],Fu[1],Fu[2]}) ;
            I_new(i,j,k,erad_idx(s)   ,q) = sg * E ;
            I_new(i,j,k,nrad_idx(s)   ,q) = sg * E / p.eps_fl ;
            I_new(i,j,k,fradx_idx(s)  ,q) = sg * Fd[0] ;
            I_new(i,j,k,fradx_idx(s)+1,q) = sg * Fd[1] ;
            I_new(i,j,k,fradx_idx(s)+2,q) = sg * Fd[2] ;
            for ( int c = 0; c < 6; ++c ) aux(i,j,k,pidx(s,c),q) = Pu[c] ;
            aux(i,j,k,LBM_EKILL0_+s,q) = killing_energy_density(metric, E, Fu) ;
        }
    }
} ;

//---------------------------------------------------------------------------
// Drivers (lbm.cpp)
//---------------------------------------------------------------------------

/**
 * @brief Validate the configuration and load the stencil (and, for curved
 *        streaming, the streaming quadrature and the velocity mesh).  Called
 *        once at startup, after the M1 trigger check.  Reflection symmetries
 *        are supported through the per-variable partner table of the ghost
 *        exchange (each population mirrors to the population of the mirrored
 *        direction).
 */
void startup_check() ;

/**
 * @brief Classify the background as flat or curved from the metric on the grid
 *        and check it against lbm.streaming (a curved background needs
 *        curved_fixed).  Called after the initial data and, lazily, on restart;
 *        invalidates the curved-streaming geometry.
 */
void prepare_background() ;

/// Invalidate the curved-streaming geometry (aux is reallocated on regrid).
void on_regrid() ;

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
 * bit-exactly.  Gated by the M1 activation trigger.  Curved streaming reads
 * the metric of the y^n slice only (the one available here); under Z4 the
 * geodesic map is rebuilt every lbm.geometry_update_every steps.
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
