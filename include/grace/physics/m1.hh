/**
 * @file m1.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief M1 radiation-transport evolution-system class: per-species reconstruction, closure-based Riemann fluxes, source terms, and IMEX integration hooks.
 * @date 2025-11-26
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
#ifndef GRACE_PHYSICS_M1_HH
#define GRACE_PHYSICS_M1_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/neutrino_pair_update.hh>
#include <grace/physics/neutrino_leptonic_update.hh>
#include "fd_subexpressions.hh"
#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {

// Host-side fatal report, after the producing device launch has completed.
void check_pair_failures(pairs::failure_buffer const& failures, var_array_t state,
                         var_array_t aux, char const* phase, double stage_h=0);
void trace_pair_moments(var_array_t state, char const* label, int stage=0, double factor=0);
//**************************************************************************************************/
//**************************************************************************************************
/**
 * @brief M1 equations system.
 * \ingroup physics
 */
//**************************************************************************************************/
struct m1_equations_system_t
    : public hrsc_evolution_system_t<m1_equations_system_t>
{
    private:
    //! Base class type
    using base_t = hrsc_evolution_system_t<m1_equations_system_t>;
    #if GRACE_M1_NU_SPECIES >= 5
    static constexpr std::array<int,5> ye_coupling_sign {1,-1,1,-1,0} ;
    #elif (GRACE_M1_NU_SPECIES >= 3)
    static constexpr std::array<int,3> ye_coupling_sign {1,-1,0} ;
    #endif
    public:

    pairs::failure_buffer pair_failures;

    m1_equations_system_t(grace::var_array_t state_
                        , grace::staggered_variable_arrays_t stag_state_
                        , grace::var_array_t aux_ )
    : base_t(state_,stag_state_,aux_)
    {} ;

    m1_equations_system_t(grace::var_array_t state_
                        , grace::staggered_variable_arrays_t stag_state_
                        , grace::var_array_t aux_
                        , m1_atmo_params_t _atmo_pars
                        , m1_excision_params_t _excision_pars
                        , m1_backreaction_params_t _backreaction_pars)
    : base_t(state_,stag_state_,aux_)
    , atmo_params(_atmo_pars)
    , excision_params(_excision_pars)
    , backreaction_params(_backreaction_pars)
    {} ;

    /**
     * @brief Compute M1 fluxes in direction \f$x^1\f$
     *
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_x_flux( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<0,ispec,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^2\f$
     *
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_y_flux( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<1,ispec,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^3\f$
     *
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_z_flux( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<2,ispec,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }


    /**
     * @brief Compute geometric source terms for M1 equations.
     *
     * @tparam thread_team_t Thread team type.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param idx Inverse cell coordinate spacings.
     * @param state_new State where sources are added.
     * @param dt Timestep.
     * @param dtfact Timestep factor.
     */
    template< int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_source_terms( const int q
                         , VEC( const int i
                         ,      const int j
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t const state_new
                         , double const dt
                         , double const dtfact ) const
    {
        using namespace grace  ;
        using namespace Kokkos ;
        /**************************************************************************************************/
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        /**************************************************************************************************/
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        /**************************************************************************************************/
        // construct closure and get pressure
        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,ispec,VEC(i,j,k)) ;
        prims[ERADL] /= metric.sqrtg() ;
        prims[NRADL] /= metric.sqrtg() ;
        prims[FXL]   /= metric.sqrtg() ;
        prims[FYL]   /= metric.sqrtg() ;
        prims[FZL]   /= metric.sqrtg() ;

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        cl.compute_pressure() ;
        /**************************************************************************************************/
        auto const& PUU            = cl.PUU              ;
        double const E             = cl.E                ;
        double const * const Fu    = cl.FU.data()        ;
        double const * const Fd    = cl.FD.data()        ;
        double const * const betau = metric._beta.data() ;
        double const * const gdd   = metric._g.data()    ;
        double const * const guu   = metric._ginv.data() ;
        double const sqrtg         = metric.sqrtg()      ;
        double const alp           = metric.alp()        ;
        /**************************************************************************************************/
        /* Metric derivatives                                                                             */
        /**************************************************************************************************/
        double dalpha_dx[3], dgdd_dx[18], dbetau_dx[9] ;
        fill_deriv_scalar<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, ALP_, q, dalpha_dx, idx(0,q)) ;
        fill_deriv_vector<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, BETAX_, q, dbetau_dx, idx(0,q)) ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        fill_deriv_tensor<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, GXX_, q, dgdd_dx, idx(0,q)) ;
        #else
        // CHI_ holds W = gamma^{-1/6} (gdd = gtdd / W^2), same convention as grmhd.hh.
        double const ooW    = 1./Kokkos::fmax(1e-15, s(CHI_)) ;
        double const ooWsqr = SQR(ooW) ;
        double dchi_dx[3] ;
        fill_deriv_scalar<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, CHI_, q, dchi_dx, idx(0,q)) ;
        fill_deriv_tensor<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, GTXX_, q, dgdd_dx, idx(0,q)) ;
        // dgdd/dx = dgtdd/dx / W^2 - 2 gdd dW/dx / W
        for( int idir=0; idir<3; ++idir) {
            for( int a=0; a<6; ++a) {
                dgdd_dx[a + 6*idir] = ooWsqr * dgdd_dx[a + 6*idir] - 2. * ooW * dchi_dx[idir] * gdd[a] ;
            }
        }
        #endif
        /**************************************************************************************************/
        /* Extrinsic curvature                                                                            */
        /**************************************************************************************************/
        double Kdd[6] ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        Kdd[0] = s(KXX_) ; Kdd[1] = s(KXY_) ; Kdd[2] = s(KXZ_) ;
        Kdd[3] = s(KYY_) ; Kdd[4] = s(KYZ_) ; Kdd[5] = s(KZZ_) ;
        #else
        double Atdd[6] = {
              s(ATXX_), s(ATXY_), s(ATXZ_),
              s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
        double const Khat  = s(KHAT_);
        double const theta = s(THETA_);
        double const Ktr = Khat + 2. * theta ;
        #endif
        for( int a=0; a<6; ++a ) {
            Kdd[a] = ooWsqr * Atdd[a] + Ktr * gdd[a] / 3. ;
        }
        #endif
        /**************************************************************************************************/
        double dE, dF[3] ;
        m1_geom_source_terms(
            E, Fd, Fu, alp, Kdd,
            dalpha_dx, dgdd_dx, dbetau_dx, PUU,
            &dE, &dF
        ) ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),m1_erad_idx<ispec>(),q)  += sqrtg * dt * dtfact * dE    ;
        state_new(VEC(i,j,k),m1_fradx_idx<ispec>(),q) += sqrtg * dt * dtfact * dF[0] ;
        state_new(VEC(i,j,k),m1_frady_idx<ispec>(),q) += sqrtg * dt * dtfact * dF[1] ;
        state_new(VEC(i,j,k),m1_fradz_idx<ispec>(),q) += sqrtg * dt * dtfact * dF[2] ;
        /**************************************************************************************************/
    }

    /**
     * @brief Compute M1 auxiliary quantities.
     *
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    template< int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_auxiliaries(  VEC( const int i
                        ,      const int j
                        ,      const int k)
                        , int64_t q
                        , grace::device_coordinate_system coords) const
    {
        using namespace grace ;
        using namespace Kokkos ;

        double rtp[3] ;
        coords.get_physical_coordinates_sph(i,j,k,q,rtp) ;


        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,ispec,VEC(i,j,k)) ;

        metric_array_t metric;
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;

        prims[ERADL] /= metric.sqrtg() ;
        prims[NRADL] /= metric.sqrtg() ;
        prims[FXL] /= metric.sqrtg() ;
        prims[FYL] /= metric.sqrtg() ;
        prims[FZL] /= metric.sqrtg() ;

        m1_closure_t cl{
            prims, metric
        } ;
        // NB: no update_closure() here.  The only things this function reads
        // off the closure are cl.E and cl.F, and those are set by initialize()
        // from the constructor -- not by update_closure.  Solving the closure
        // here (a full-bracket Brent per species per cell, over the whole grid
        // INCLUDING ghosts, on every one of the 4 imex222 auxiliary passes)
        // produced only cl.J/cl.Gamma, which fed a mean-energy local left over
        // from the old atmosphere treatment and was never stored anywhere.
        // rescale if superluminal
        if ( cl.F >= cl.E ) {
            double fact = 0.9999 * cl.E / cl.F ;
            this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) *= fact ;
            this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) *= fact ;
            this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) *= fact ;
        }
        // Set atmosphere / excision
        double r = rtp[0] ;
        bool excise = excision_params.excise_by_radius
                ? r <= excision_params.r_ex
                : metric.alp() <= excision_params.alp_ex ;
        // ---- OLD atmosphere treatment (kept for reference) ------------------
        // The number floor was derived as E_fl/eps_fl, and N was compared
        // against the ENERGY floor.  With eps_fl = 1 (default) the two
        // coincide and it is invisible; otherwise the test is wrong and the
        // keep-N branch below rewrote a healthy N as E = N*eps_atmo.
        // double E_atmo = atmo_params.E_fl * Kokkos::pow(r,atmo_params.E_fl_scaling) ;
        // double eps_atmo = atmo_params.eps_fl * Kokkos::pow(r,atmo_params.eps_fl_scaling) ;
        // bool const E_bad = cl.E         < E_atmo * (1. + atmo_params.atmo_tol ) ;
        // bool const N_bad = prims[NRADL] < E_atmo * (1. + atmo_params.atmo_tol ) ;
        // ---- NEW: FIL parity (driver_M1_conserv_to_prims.cc) ----------------
        // Independent E and N floors, each tested against its own threshold.
        // cl.E and prims are already undensitised here, so no sqrtg is needed
        // in the test -- only in the write-back below.
        double const E_atmo = atmo_params.E_floor(r) ;
        double const N_atmo = atmo_params.N_floor(r) ;
        bool const E_bad = cl.E         < E_atmo * (1. + atmo_params.atmo_tol ) ;
        bool const N_bad = prims[NRADL] < N_atmo * (1. + atmo_params.atmo_tol ) ;
        // ---- OLD keep-N / full-reset branches (kept for reference) ----------
        // if ( E_bad && !N_bad ) {
        //     double const E_keepN = prims[NRADL] * eps_atmo ;   // = N when eps_fl=1
        //     this->_state(VEC(i,j,k),m1_erad_idx<ispec>(),q)  = metric.sqrtg() * E_keepN ;
        //     this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) = 0.0 ;
        //     this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) = 0.0 ;
        //     this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) = 0.0 ;
        //     epsilon = eps_atmo ;
        // } else if ( E_bad || N_bad ) {
        //     double atmo_state[4] = {E_atmo,0.0, 0.0, 0.0} ;
        //     ... ; cl.update_closure(atmo_state,0,true) ;
        //     this->_state(VEC(i,j,k),m1_nrad_idx<ispec>(),q) = metric.sqrtg()*cl.Gamma*cl.J/eps_atmo ;
        //     epsilon = eps_atmo ;
        // }
        // ---- NEW: FIL parity -- one plain reset, both floors, no keep-N ------
        // FIL sets E and N to their own floors, zeroes the flux and sets the
        // mean energy to 0 (not eps_fl), which keeps the T_nu correction inert.
        // Excision first: the EVOLVED (densitised) fields are what the output shows, so they
        // are set to the plain excision value, without sqrtg (20..4000 there), and the region
        // shows up at that value.  Tested before the floor, which would re-densitise it.
        if ( excise ) {
            this->_state(VEC(i,j,k),m1_erad_idx<ispec>(),q)  = excision_params.E_ex ;
            this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_nrad_idx<ispec>(),q)  = excision_params.E_ex/excision_params.eps_ex ;
        } else if ( E_bad || N_bad ) {
            double const sg = metric.sqrtg() ;
            this->_state(VEC(i,j,k),m1_erad_idx<ispec>(),q)  = sg * E_atmo ;
            this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) = 0.0 ;
            this->_state(VEC(i,j,k),m1_nrad_idx<ispec>(),q)  = sg * N_atmo ;
        }
    }

    // ----------------------------------------------------------------------
    // Collision update of transparent species (override with -DGRACE_M1_EXPLICIT_THIN=0)
    //   1 = one explicit step U = W + dt S(W) where
    //       4 alp W^3 (kappa_a+kappa_s) dt < GRACE_M1_EXPLICIT_THIN_MAX_RATE
    //       (THC_M1's kappa dt < 1, with the lab-frame rate of a counter-streaming
    //       beam, J <= W^2 (1+v)^2 E): monotone, keeps E > 0, needs no guess.
    //   0 = Newton solve everywhere, as in FIL.
    // The step is first order inside the IMEX stage.  At a limit of 1 it shifted the
    // diffusivity of the scattering test by 5% (kappa dt = 0.09); 0.1 keeps it to
    // cells that are transparent (kappa dx < 1/3 at cfl 0.25), which is where the
    // Newton path goes wrong (floor-scale E accepted at the absolute tolerance).
    // ----------------------------------------------------------------------
    #ifndef GRACE_M1_EXPLICIT_THIN
    #define GRACE_M1_EXPLICIT_THIN 1
    #endif
    #ifndef GRACE_M1_EXPLICIT_THIN_MAX_RATE
    #define GRACE_M1_EXPLICIT_THIN_MAX_RATE 0.1
    #endif

    /**
     * @brief Compute M1 implicit update.
     *
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    template< int ispec >
    void KOKKOS_INLINE_FUNCTION
    compute_implicit_update( const int q
                         , VEC( const int i
                         ,      const int j
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t const state_new
                         , double const dt
                         , double const dtfact ) const
    {
        using namespace grace  ;
        using namespace Kokkos ;
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        // read in eas
        m1_eas_array_t eas ;
        eas[KAL]   = this->_aux(VEC(i,j,k),m1_kappaa_idx<ispec>(),q) ;
        eas[KSL]   = this->_aux(VEC(i,j,k),m1_kappas_idx<ispec>(),q) ;
        eas[ETAL]  = this->_aux(VEC(i,j,k),m1_eta_idx<ispec>(),q) ;
        eas[ETANL] = this->_aux(VEC(i,j,k),m1_etan_idx<ispec>(),q) ;
        eas[KANL]  = this->_aux(VEC(i,j,k),m1_kappaan_idx<ispec>(),q) ;
        /**************************************************************************************************/
        // No collision term -> the implicit update is exactly the identity, and
        // the caller deep-copies old->new, so returning leaves U = W and N.
        constexpr double eas_negligible = M1_EAS_NEGLIGIBLE ;
        const bool no_collision = eas[KAL]   < eas_negligible
                               && eas[KSL]   < eas_negligible
                               && eas[ETAL]  < eas_negligible
                               && eas[ETANL] < eas_negligible
                               && eas[KANL]  < eas_negligible ;
        if ( no_collision ) return ;
        /**************************************************************************************************/
        // construct closure and update
        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,ispec,VEC(i,j,k)) ;
        prims[ERADL] /= metric.sqrtg() ;
        prims[NRADL] /= metric.sqrtg() ;
        prims[FXL] /= metric.sqrtg();
        prims[FYL] /= metric.sqrtg();
        prims[FZL] /= metric.sqrtg();

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        /**************************************************************************************************/
        // store explicitly updated state
        double  W[4]  ;
        W[0] = prims[ERADL] ; W[1] = prims[FXL] ; W[2] = prims[FYL] ; W[3] = prims[FZL] ;
        /**************************************************************************************************/
        double  U[4]  ;
#ifdef GRACE_M1_DIAGNOSTICS
        int  first_err = 0 ;
        bool used_linear_fallback = false, explicit_step = false ;
#endif
#if GRACE_M1_EXPLICIT_THIN
        // Transparent this stage: the closure above is already at the explicit state W.
        bool const non_stiff =
            4.0 * metric.alp() * cl.W*cl.W*cl.W * ( eas[KAL] + eas[KSL] ) * dt * dtfact
            < GRACE_M1_EXPLICIT_THIN_MAX_RATE ;
#else
        bool const non_stiff = false ;
#endif
        if ( non_stiff ) {
            double S[4] ;
            cl.get_implicit_sources(eas,S) ;
            for( int c=0; c<4; ++c ) U[c] = W[c] + dt*dtfact*S[c] ;
#ifdef GRACE_M1_DIAGNOSTICS
            explicit_step = true ;
#endif
        } else {
            // construct the initial guess
            cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);
            //U[0] = ( prims[ERADL] + dt * dtfact * eas[ETAL]) / ( 1. + dt * dtfact * eas[KAL]) ;
            // take a pointer so we can capture it
            // in the lambda
            m1_closure_t* pcl = &cl;
            /**************************************************************************************************/
            // construct the lambdas for the evaluation of the update
            auto const func = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
                pcl->implicit_update_func(eas,u,W,s,dt,dtfact) ;
            } ;
            auto const dfunc = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
                pcl->implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ;
            } ;
            /**************************************************************************************************/
            // call rootfinder
            unsigned long maxiter = 100 ;
            int err = 0;
#ifdef GRACE_M1_DOGLEG
            utils::rootfind_nd_dogleg<4>(
                func, dfunc, U, maxiter, 1e-15, err
            ) ;
#else
            utils::rootfind_nd_newton_raphson<4>(
                func, dfunc, U, maxiter, 1e-15, err
            ) ;
#endif
#ifdef GRACE_M1_COUNT_IMPLICIT
            printf("[IMP1] %d %.6e %.6e %.6e\n", err, cl.zeta,
                   eas[KAL]*dt*dtfact, eas[KSL]*dt*dtfact) ;
#endif
#ifdef GRACE_M1_DIAGNOSTICS
            first_err = err ;
#endif
            /**************************************************************************************************/
            if ( err != utils::nr_err_t::SUCCESS ) {
                // assume optically thick closure and
                // repeat
                cl.update_closure(prims,0.,false /*nb no update here*/) ;

                cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);

                auto const fixed_closure_func = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
                    pcl->implicit_update_func(eas,u,W,s,dt,dtfact,false) ;
                } ;
                auto const fixed_closure_dfunc = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
                    pcl->implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ;
                } ;
#ifdef GRACE_M1_DOGLEG
                utils::rootfind_nd_dogleg<4>(
                    fixed_closure_func, fixed_closure_dfunc, U, maxiter, 1e-15, err
                ) ;
#else
                utils::rootfind_nd_newton_raphson<4>(
                    fixed_closure_func, fixed_closure_dfunc, U, maxiter, 1e-15, err
                ) ;
#endif
#ifdef GRACE_M1_COUNT_IMPLICIT
                printf("[IMP2] %d\n", err) ;
#endif
                // if we failed again we just take a linear step and call it
                // (Radice+2022 sec 3.2: on non-convergence they likewise linearise
                // by fixing chi = 1/3).  Measured: the solver reports failure on
                // ~26% of solves here, so declining to deposit -- FIL's policy --
                // starves the collision term and lets Ymu run away.
                if ( err != utils::nr_err_t::SUCCESS ) {
                    cl.update_closure(prims,0,true) ;
                    cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);
#ifdef GRACE_M1_DIAGNOSTICS
                    used_linear_fallback = true ;
#endif
                }
            }
        } // Newton path
        // the compiler seems to sometimes think U is never
        // modified and just elides the whole function....
        volatile double U0 = U[0];
        volatile double U1 = U[1];
        volatile double U2 = U[2];
        volatile double U3 = U[3];
        /**************************************************************************************************/
        // write back to the new state
        state_new(i,j,k,m1_erad_idx<ispec>(),q)  = metric.sqrtg() * U[0] ;
        state_new(i,j,k,m1_fradx_idx<ispec>(),q) = metric.sqrtg() * U[1] ;
        state_new(i,j,k,m1_frady_idx<ispec>(),q) = metric.sqrtg() * U[2] ;
        state_new(i,j,k,m1_fradz_idx<ispec>(),q) = metric.sqrtg() * U[3] ;
        /**************************************************************************************************/
        /**************************************************************************************************/
        // Number source is linear
        // we need to update the closure on the starred state
        // to get the correct Gamma factor!
        double N, dN ;
        cl.update_closure(U0, {U1,U2,U3},0,true) ;
        // prims here are **not** the implicitly updated ones
        cl.get_N_implicit_update(
            prims, eas, dt, dtfact, &N, &dN
        ) ;
        state_new(VEC(i,j,k),m1_nrad_idx<ispec>(),q)  = metric.sqrtg() * N ;
#ifdef GRACE_M1_DIAGNOSTICS
        // Solver record, sticky over the step's stages (layout: m1_implicit_err_bits_t).
        // Residual R = W + dt S(U) - U of the full system at the accepted state, whichever
        // path produced it.  dU ~ R/(1 + alp W kappa dt), so res estimates |dU|/E.
        // An explicit step has no solve to judge: it records no residual, only its state.
        double Ufin[4] = { U0, U1, U2, U3 } ;
        double res = 0.0 ;
        if ( !explicit_step ) {
            double R[4] ;
            cl.implicit_update_func(eas,Ufin,W,R,dt,dtfact) ;
            for( int c=0; c<4; ++c ) res = Kokkos::fmax(res, Kokkos::fabs(R[c])) ;
            res /= ( 1.0 + metric.alp() * cl.W * ( eas[KAL] + eas[KSL] ) * dt * dtfact )
                 * Kokkos::fmax( Kokkos::fmax(Kokkos::fabs(Ufin[0]), Kokkos::fabs(W[0])), 1.0e-200 ) ;
            if ( !Kokkos::isfinite(res) ) res = 1.0e300 ;
        }
        double const F2fin = metric.square_covec({Ufin[1],Ufin[2],Ufin[3]}) ;
        bool const non_physical = !( Ufin[0] > 0.0 )
                               || !( F2fin <= (1.0+1.0e-10)*Ufin[0]*Ufin[0] ) ;
        // One bit per error kind: OR-ing the raw codes of two stages would alias.
        unsigned const bits = ( first_err != utils::nr_err_t::SUCCESS ? 1u << (first_err-1) : 0u )
                            | ( used_linear_fallback ? unsigned(M1_IMPLICIT_LINEAR)      : 0u )
                            | ( non_physical         ? unsigned(M1_IMPLICIT_NONPHYSICAL) : 0u ) ;
        unsigned const old_bits = static_cast<unsigned>( this->_aux(VEC(i,j,k),M1_IMPLICIT_ERR_,q) ) ;
        this->_aux(VEC(i,j,k),M1_IMPLICIT_ERR_,q) =
            static_cast<double>( old_bits | ( bits << (M1_IMPLICIT_ERR_STRIDE*ispec) ) ) ;
        this->_aux(VEC(i,j,k),M1_IMPLICIT_RES_,q) =
            Kokkos::fmax( this->_aux(VEC(i,j,k),M1_IMPLICIT_RES_,q), res ) ;
        if ( explicit_step ) {
            unsigned const old_ex = static_cast<unsigned>( this->_aux(VEC(i,j,k),M1_EXPLICIT_STEP_,q) ) ;
            this->_aux(VEC(i,j,k),M1_EXPLICIT_STEP_,q) = static_cast<double>( old_ex | ( 1u << ispec ) ) ;
        }
#endif
    }

    #if GRACE_M1_NU_SPECIES >= 3
    // sa==sb describes a charge-symmetric aggregate; multiplicity counts
    // physical helicity species (2 for tau+antitau, 4 for the 3-species NUX).
    template<int sa, int sb, int multiplicity=1>
    KOKKOS_INLINE_FUNCTION void compute_pair_implicit_update(
        int q, VEC(int i,int j,int k), scalar_array_t<GRACE_NSPACEDIM> const idx,
        var_array_t state_new, double dt, double dtfact, int order) const {
        int const bits=int(this->_aux(VEC(i,j,k),PAIR_ACTIVE_,q));
        if(bits==0) {
            compute_implicit_update<sa>(q,VEC(i,j,k),idx,state_new,dt,dtfact);
            if constexpr(sa!=sb) compute_implicit_update<sb>(q,VEC(i,j,k),idx,state_new,dt,dtfact);
            return;
        }
        metric_array_t metric;
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k));
        m1_prims_array_t p[2]; m1_eas_array_t eas[2];
        double old[10], out[10];
        auto const read=[&](auto species, int s) {
            constexpr int sp=decltype(species)::value;
            FILL_M1_PRIMS_ARRAY(p[s],this->_state,this->_aux,q,sp,VEC(i,j,k));
            constexpr int v[5]={ERADL,FXL,FYL,FZL,NRADL};
            for(int a=0;a<5;++a) {
                p[s][v[a]]/=metric.sqrtg()*multiplicity;
                old[5*s+a]=p[s][v[a]];
            }
            eas[s][KAL]=this->_aux(VEC(i,j,k),m1_kappaa_idx<sp>(),q);
            eas[s][KSL]=this->_aux(VEC(i,j,k),m1_kappas_idx<sp>(),q);
            eas[s][KANL]=this->_aux(VEC(i,j,k),m1_kappaan_idx<sp>(),q);
            eas[s][ETAL]=this->_aux(VEC(i,j,k),m1_eta_idx<sp>(),q)/multiplicity;
            eas[s][ETANL]=this->_aux(VEC(i,j,k),m1_etan_idx<sp>(),q)/multiplicity;
        };
        read(std::integral_constant<int,sa>{},0); read(std::integral_constant<int,sb>{},1);
        pairs::material const mat{this->_aux(VEC(i,j,k),PAIR_T_,q),
            this->_aux(VEC(i,j,k),PAIR_MUE_,q),this->_aux(VEC(i,j,k),PAIR_NB_,q),
            this->_aux(VEC(i,j,k),PAIR_YN_,q),this->_aux(VEC(i,j,k),PAIR_YP_,q)};
        double scale=mat.T;
        for(int s=0;s<2;++s) {
            m1_closure_t cl{p[s],metric}; cl.update_closure(0);
            if(old[5*s+4]>0) {
                double const mean=cl.J*cl.Gamma/old[5*s+4]*pairs::number_unit/pairs::energy_unit;
                scale=Kokkos::fmax(scale,Kokkos::sqrt(mat.T*mean/3));
            }
        }
        pairs::kernel kernel;
        pairs::diagnostic diag;
        auto const fail=[&]() {
            auto r=pairs::make_failure_record(mat,diag,q,VEC(i,j,k));
            r.active=bits; r.scale=scale; r.order=order;
            r.species_a=sa; r.species_b=sb; r.multiplicity=multiplicity;
            pair_failures.save(r);
        };
        if(!kernel.init(mat,{bool(bits&1),bool(bits&2),bool(bits&4)},order,scale,sa==sb,&diag)) {
            fail();
            return;
        }
        double residual=0;
        if(!pairs::implicit_update(kernel,metric,p,eas,old,out,dt*dtfact,residual)) {
            diag.code=pairs::failure_code::implicit_solve;
            diag.residual=residual;
            fail();
            return;
        }
        this->_aux(VEC(i,j,k),PAIR_RES_,q)=Kokkos::fmax(this->_aux(VEC(i,j,k),PAIR_RES_,q),residual);
        auto const write=[&](auto species, int s) {
            constexpr int sp=decltype(species)::value;
            constexpr int v[5]={m1_erad_idx<sp>(),m1_fradx_idx<sp>(),m1_frady_idx<sp>(),m1_fradz_idx<sp>(),m1_nrad_idx<sp>()};
            for(int a=0;a<5;++a)
                state_new(VEC(i,j,k),v[a],q)=metric.sqrtg()*multiplicity*out[5*s+a];
        };
        write(std::integral_constant<int,sa>{},0);
        if constexpr(sa!=sb) write(std::integral_constant<int,sb>{},1);
    }
    #endif

    #if GRACE_M1_NU_SPECIES >= 5
    KOKKOS_INLINE_FUNCTION void compute_leptonic_implicit_update(
        int q,VEC(int i,int j,int k),scalar_array_t<GRACE_NSPACEDIM> const idx,
        var_array_t state_new,double dt,double dtfact,int order) const {
        if(this->_aux(VEC(i,j,k),PAIR_DECAY_,q)==0) {
            compute_implicit_update<0>(q,VEC(i,j,k),idx,state_new,dt,dtfact);
            compute_implicit_update<1>(q,VEC(i,j,k),idx,state_new,dt,dtfact);
            compute_pair_implicit_update<2,3>(q,VEC(i,j,k),idx,state_new,dt,dtfact,order);
            return;
        }
        metric_array_t metric; FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k));
        m1_prims_array_t p[4]; m1_eas_array_t eas[4]; double old[20],out[20];
        auto const read=[&](auto species) {
            constexpr int s=decltype(species)::value;
            FILL_M1_PRIMS_ARRAY(p[s],this->_state,this->_aux,q,s,VEC(i,j,k));
            constexpr int v[5]={ERADL,FXL,FYL,FZL,NRADL};
            for(int a=0;a<5;++a) { p[s][v[a]]/=metric.sqrtg(); old[5*s+a]=p[s][v[a]]; }
            eas[s][KAL]=this->_aux(VEC(i,j,k),m1_kappaa_idx<s>(),q);
            eas[s][KSL]=this->_aux(VEC(i,j,k),m1_kappas_idx<s>(),q);
            eas[s][KANL]=this->_aux(VEC(i,j,k),m1_kappaan_idx<s>(),q);
            eas[s][ETAL]=this->_aux(VEC(i,j,k),m1_eta_idx<s>(),q);
            eas[s][ETANL]=this->_aux(VEC(i,j,k),m1_etan_idx<s>(),q);
        };
        read(std::integral_constant<int,0>{}); read(std::integral_constant<int,1>{});
        read(std::integral_constant<int,2>{}); read(std::integral_constant<int,3>{});
        pairs::material const mat{this->_aux(VEC(i,j,k),PAIR_T_,q),
            this->_aux(VEC(i,j,k),PAIR_MUE_,q),this->_aux(VEC(i,j,k),PAIR_NB_,q),
            this->_aux(VEC(i,j,k),PAIR_YN_,q),this->_aux(VEC(i,j,k),PAIR_YP_,q)};
        double scale=Kokkos::fmax(mat.T,pairs::muon_mass/3);
        for(int s=0;s<4;++s) if(old[5*s+4]>0) {
            m1_closure_t cl{p[s],metric}; cl.update_closure(0);
            double const mean=cl.J*cl.Gamma/old[5*s+4]*pairs::number_unit/pairs::energy_unit;
            scale=Kokkos::fmax(scale,Kokkos::sqrt(mat.T*mean/3));
        }
        int const bits=int(this->_aux(VEC(i,j,k),PAIR_ACTIVE_,q));
        pairs::kernel thermal; pairs::decay_kernel decay;
        if(!thermal.init(mat,{bool(bits&1),bool(bits&2),bool(bits&4)},order,scale)
           || !decay.init(thermal.g,mat.T,mat.mu_e,this->_aux(VEC(i,j,k),PAIR_MUMU_,q),
                          int(this->_aux(VEC(i,j,k),PAIR_DORDER_,q))))
            Kokkos::abort("Invalid four-species leptonic kernel");
        double residual=0;
        if(!pairs::leptonic_implicit_update(thermal,decay,bits!=0,metric,p,eas,old,out,dt*dtfact,residual))
            Kokkos::abort("Coupled leptonic source solve failed: check timestep and quadrature");
        this->_aux(VEC(i,j,k),PAIR_RES_,q)=Kokkos::fmax(
            this->_aux(VEC(i,j,k),PAIR_RES_,q),residual);
        auto const write=[&](auto species) {
            constexpr int s=decltype(species)::value;
            constexpr int v[5]={m1_erad_idx<s>(),m1_fradx_idx<s>(),m1_frady_idx<s>(),m1_fradz_idx<s>(),m1_nrad_idx<s>()};
            for(int a=0;a<5;++a) state_new(VEC(i,j,k),v[a],q)=metric.sqrtg()*out[5*s+a];
        };
        write(std::integral_constant<int,0>{}); write(std::integral_constant<int,1>{});
        write(std::integral_constant<int,2>{}); write(std::integral_constant<int,3>{});
    }

    // A single convex factor preserves EVERY reaction's stoichiometry, also
    // when both Ye and Ymu are close to EOS bounds. Tau pairs share the energy
    // acceptance for simplicity. No change to the historical no-decay limiter.
    template<typename eos_t>
    KOKKOS_INLINE_FUNCTION void add_leptonic_backreaction(int q,VEC(int i,int j,int k),
        var_array_t state_new,eos_t const& eos) const {
        auto const e=species_exchange<0>(q,VEC(i,j,k),state_new);
        auto const eb=species_exchange<1>(q,VEC(i,j,k),state_new);
        auto const m=species_exchange<2>(q,VEC(i,j,k),state_new);
        auto const mb=species_exchange<3>(q,VEC(i,j,k),state_new);
        auto const x=species_exchange<4>(q,VEC(i,j,k),state_new);
        double const D=state_new(VEC(i,j,k),DENS_,q);
        if(!(D>0) || !Kokkos::isfinite(D))
            Kokkos::abort("Invalid baryon density in leptonic backreaction");
        double const ye=state_new(VEC(i,j,k),YESTAR_,q)/D;
        double const ym=state_new(VEC(i,j,k),YMUSTAR_,q)/D;
        double const dye=(e.N-eb.N)/D,dym=(m.N-mb.N)/D;
        double const dE=e.E+eb.E+m.E+mb.E+x.E;
        double const tau=state_new(VEC(i,j,k),TAU_,q);
        if(!Kokkos::isfinite(dE+dye+dym+tau+e.Sx+eb.Sx+m.Sx+mb.Sx+x.Sx
                            +e.Sy+eb.Sy+m.Sy+mb.Sy+x.Sy+e.Sz+eb.Sz+m.Sz+mb.Sz+x.Sz))
            Kokkos::abort("Non-finite leptonic exchange");
        double theta=1;
        auto const bound=[&](double v,double dv,double lo,double hi) {
            if(v<lo || v>hi || !Kokkos::isfinite(v+dv)) { theta=0; return; }
            if(dv>0 && v+dv>hi) theta=Kokkos::fmin(theta,(hi-v)/dv*(1-1e-10));
            if(dv<0 && v+dv<lo) theta=Kokkos::fmin(theta,(lo-v)/dv*(1-1e-10));
        };
        bound(ye,dye,eos.get_c2p_ye_min(),eos.get_c2p_ye_max());
        bound(ym,dym,eos.get_c2p_ymu_min(),eos.get_c2p_ymu_max());
        bound(ye+ym,dye+dym,0,eos.get_c2p_ye_max());
        if(!(tau>0)) theta=0;
        if(dE<0 && tau+dE<=0) theta=Kokkos::fmin(theta,tau/(-dE)*(1-1e-10));
        theta=Kokkos::fmax(0.0,theta);
        state_new(VEC(i,j,k),TAU_,q)+=theta*dE;
        state_new(VEC(i,j,k),SX_,q)+=theta*(e.Sx+eb.Sx+m.Sx+mb.Sx+x.Sx);
        state_new(VEC(i,j,k),SY_,q)+=theta*(e.Sy+eb.Sy+m.Sy+mb.Sy+x.Sy);
        state_new(VEC(i,j,k),SZ_,q)+=theta*(e.Sz+eb.Sz+m.Sz+mb.Sz+x.Sz);
        state_new(VEC(i,j,k),YESTAR_,q)+=theta*(e.N-eb.N);
        state_new(VEC(i,j,k),YMUSTAR_,q)+=theta*(m.N-mb.N);
        if(theta<1) {
            blend_species<0>(q,VEC(i,j,k),state_new,theta); blend_species<1>(q,VEC(i,j,k),state_new,theta);
            blend_species<2>(q,VEC(i,j,k),state_new,theta); blend_species<3>(q,VEC(i,j,k),state_new,theta);
            blend_species<4>(q,VEC(i,j,k),state_new,theta);
        }
        #ifdef GRACE_M1_DIAGNOSTICS
        this->_aux(VEC(i,j,k),M1_HEATCOOL_,q)+=theta*dE/D;
        this->_aux(VEC(i,j,k),M1_LEPTON_SOURCE_,q)+=theta*dye;
        this->_aux(VEC(i,j,k),M1_MUON_SOURCE_,q)+=theta*dym;
        if(theta<1) this->_aux(VEC(i,j,k),M1_BR_REJECT_,q)=
            double(int(this->_aux(VEC(i,j,k),M1_BR_REJECT_,q))|16);
        #endif
    }
    #endif

    template<int ispec>
    KOKKOS_INLINE_FUNCTION double transport_opacity(int q,VEC(int i,int j,int k)) const {
        return m1_transport_opacity<ispec>(this->_aux,q,VEC(i,j,k));
    }

    // ----------------------------------------------------------------------
    // Backreaction limiter policy (override with -DGRACE_M1_BACKREACT_HARDSTOP=0)
    //   1 = HARD STOP: every species is accepted or rejected whole (E, F, N).
    //       A lepton pair that would push Ye/Ymu off the table is rejected
    //       first; the rest is accepted iff tau + sum(dE) > 0.  In a
    //       near-atmosphere cell it does nothing rather than drain the fluid.
    //   0 = conservative SCALED limiter: throttle both sides by a shared factor
    //       so the fluid absorbs what it can and the radiation keeps the rest
    //       (energy/lepton conserved, but pins the low-density fluid to floor).
    // ----------------------------------------------------------------------
    #ifndef GRACE_M1_BACKREACT_HARDSTOP
    #define GRACE_M1_BACKREACT_HARDSTOP 1
    #endif

    template< typename eos_t >
    void KOKKOS_INLINE_FUNCTION
    add_backreaction( const int q
                         , VEC( const int i
                         ,      const int j
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t const state_new
                         // NB: must be the LOADED EOS, fetched on the host and
                         // captured into the kernel (same pattern as the grmhd
                         // system's _eos).  The historic `eos_t eos;` local had
                         // uninitialized Ye/Ymu bounds (eos_base_t() = default,
                         // bare double members), so the composition limiter
                         // below compared against indeterminate values.
                         , eos_t const& eos
                         // Density cutoff (FIL M1_rho_floor): skip the WHOLE
                         // coupling below this rho.  The halo/atmosphere-adjacent
                         // cells hold ~zero baryons, so the collision's dE/dN --
                         // however small in absolute terms -- become huge per
                         // baryon (dE/tau, dN/D) and drive the fluid primitives
                         // and composition regardless of the in-bounds limiters.
                         // The limiters (scaled OR hard-stop) only act at the
                         // TABLE EDGES; an in-bounds drift (e.g. Ymu -> 0.02) is
                         // applied in full by both.  Only skipping the coupling
                         // in these cells preserves the halo.  <= 0 disables.
                         , double const rho_min
                         // Hard stop only: a muon pair that would leave the Ymu table is
                         // accepted in the fraction that lands Ymu on the bound (E, F, N alike).
                         , bool const muon_partial = false ) const
    {
        using namespace grace  ;
        using namespace Kokkos ;
        // NB: the exchanges are differences of densitized fields (sqrtg
        // cancels), so no metric is needed anywhere in this routine.

        // Per-cell density cutoff: leave near-empty cells untouched.
        if ( rho_min > 0.0 && this->_aux(VEC(i,j,k),RHO_,q) < rho_min ) return ;
        (void)muon_partial ;   // hard stop with 5 species only

        #if !defined(GRACE_FREEZE_HYDRO) && GRACE_M1_NU_SPECIES >= 5
        if(this->_aux(VEC(i,j,k),PAIR_DECAY_,q)>0) {
            add_leptonic_backreaction(q,VEC(i,j,k),state_new,eos);
            return;
        }
        #endif

        #if GRACE_M1_BACKREACT_HARDSTOP
        #if !defined(GRACE_FREEZE_HYDRO) && GRACE_M1_NU_SPECIES >= 3
        // Old - new per species.  A species the implicit solve skipped has
        // new == old bitwise, so it contributes exactly zero: no rate gate needed.
        auto const x_e    = species_exchange<0>(q,VEC(i,j,k),state_new) ;
        auto const x_ebar = species_exchange<1>(q,VEC(i,j,k),state_new) ;
        #if GRACE_M1_NU_SPECIES >= 5
        auto const x_mu    = species_exchange<2>(q,VEC(i,j,k),state_new) ;
        auto const x_mubar = species_exchange<3>(q,VEC(i,j,k),state_new) ;
        auto const x_x     = species_exchange<4>(q,VEC(i,j,k),state_new) ;
        #else
        auto const x_x     = species_exchange<2>(q,VEC(i,j,k),state_new) ;
        #endif
        double const D = state_new(VEC(i,j,k),DENS_,q) ;

        // Lepton pairs first; the Y* that is checked is the Y* that is written.
        double const yes_new = state_new(VEC(i,j,k),YESTAR_,q) + ( x_e.N - x_ebar.N ) ;
        double const ye_new  = yes_new / D ;
        bool   const acc_e   = ( ye_new >= eos.get_c2p_ye_min() && ye_new <= eos.get_c2p_ye_max() ) ;
        #if GRACE_M1_NU_SPECIES >= 5
        double const ymus_new = state_new(VEC(i,j,k),YMUSTAR_,q) + ( x_mu.N - x_mubar.N ) ;
        double const ymu_new  = ymus_new / D ;
        bool   const acc_mu   = ( ymu_new >= eos.get_c2p_ymu_min() && ymu_new <= eos.get_c2p_ymu_max() ) ;
        // Partial acceptance (muon_partial): the fraction f_mu of the whole pair exchange
        // that lands Ymu just inside the bound it would cross; 0 means a full revert.
        double f_mu = 0.0 ;
        if ( !acc_mu && muon_partial ) {
            double const ymu_old = state_new(VEC(i,j,k),YMUSTAR_,q) / D ;
            double const bound   = ( ymu_new < eos.get_c2p_ymu_min() ) ? eos.get_c2p_ymu_min()
                                                                       : eos.get_c2p_ymu_max() ;
            double const f = ( bound - ymu_old ) / ( ymu_new - ymu_old ) * ( 1.0 - 1.0e-10 ) ;
            f_mu = ( Kokkos::isfinite(f) && f > 0.0 ) ? Kokkos::fmin(f, 1.0) : 0.0 ;
        }
        bool const part_mu = !acc_mu && f_mu > 0.0 ;
        #endif

        // One energy decision on the summed exchange of the surviving species
        // (order independent, as in FIL); the momentum follows it.
        double dE = x_x.E, dSx = x_x.Sx, dSy = x_x.Sy, dSz = x_x.Sz ;
        if ( acc_e ) {
            dE  += x_e.E  + x_ebar.E  ; dSx += x_e.Sx + x_ebar.Sx ;
            dSy += x_e.Sy + x_ebar.Sy ; dSz += x_e.Sz + x_ebar.Sz ;
        }
        #if GRACE_M1_NU_SPECIES >= 5
        if ( acc_mu ) {
            dE  += x_mu.E  + x_mubar.E  ; dSx += x_mu.Sx + x_mubar.Sx ;
            dSy += x_mu.Sy + x_mubar.Sy ; dSz += x_mu.Sz + x_mubar.Sz ;
        } else if ( part_mu ) {
            dE  += f_mu * ( x_mu.E  + x_mubar.E  ) ; dSx += f_mu * ( x_mu.Sx + x_mubar.Sx ) ;
            dSy += f_mu * ( x_mu.Sy + x_mubar.Sy ) ; dSz += f_mu * ( x_mu.Sz + x_mubar.Sz ) ;
        }
        #endif
        double const tau_new = state_new(VEC(i,j,k),TAU_,q) + dE ;
        bool   const acc_E   = ( tau_new > 0.0 ) ;

        if ( acc_E ) {
            state_new(VEC(i,j,k),TAU_,q) = tau_new ;
            state_new(VEC(i,j,k),SX_,q) += dSx ;
            state_new(VEC(i,j,k),SY_,q) += dSy ;
            state_new(VEC(i,j,k),SZ_,q) += dSz ;
            if ( acc_e ) state_new(VEC(i,j,k),YESTAR_,q) = yes_new ;
            #if GRACE_M1_NU_SPECIES >= 5
            if ( acc_mu ) state_new(VEC(i,j,k),YMUSTAR_,q) = ymus_new ;
            else if ( part_mu )
                state_new(VEC(i,j,k),YMUSTAR_,q) += f_mu * ( x_mu.N - x_mubar.N ) ;
            #endif
        }
        // Rejected species go back to their pre-collision E, F and N.
        if ( !( acc_E && acc_e ) ) {
            revert_species<0>(q,VEC(i,j,k),state_new) ;
            revert_species<1>(q,VEC(i,j,k),state_new) ;
        }
        #if GRACE_M1_NU_SPECIES >= 5
        if ( acc_E && part_mu ) {           // radiation keeps the complementary fraction
            blend_species<2>(q,VEC(i,j,k),state_new,f_mu) ;
            blend_species<3>(q,VEC(i,j,k),state_new,f_mu) ;
        } else if ( !( acc_E && acc_mu ) ) {
            revert_species<2>(q,VEC(i,j,k),state_new) ;
            revert_species<3>(q,VEC(i,j,k),state_new) ;
        }
        if ( !acc_E ) revert_species<4>(q,VEC(i,j,k),state_new) ;
        #else
        if ( !acc_E ) revert_species<2>(q,VEC(i,j,k),state_new) ;
        #endif

        #ifdef GRACE_M1_DIAGNOSTICS
        // What was applied, summed raw over the step's implicit stages (not
        // IMEX-weighted); reset every step.  FIL: m1_heatcool, m1_lepton_source.
        if ( acc_E ) {
            this->_aux(VEC(i,j,k),M1_HEATCOOL_,q) += dE / D ;
            if ( acc_e ) this->_aux(VEC(i,j,k),M1_LEPTON_SOURCE_,q) += ( x_e.N - x_ebar.N ) / D ;
            #if GRACE_M1_NU_SPECIES >= 5
            if ( acc_mu ) this->_aux(VEC(i,j,k),M1_MUON_SOURCE_,q) += ( x_mu.N - x_mubar.N ) / D ;
            else if ( part_mu )
                this->_aux(VEC(i,j,k),M1_MUON_SOURCE_,q) += f_mu * ( x_mu.N - x_mubar.N ) / D ;
            #endif
        }
        int rejected = ( acc_e ? 0 : 1 ) | ( acc_E ? 0 : 4 ) ;
        #if GRACE_M1_NU_SPECIES >= 5
        // 8: the muon pair was accepted only in part (muon_partial), 2: not at all.
        rejected |= acc_mu ? 0 : ( ( acc_E && part_mu ) ? 8 : 2 ) ;
        #endif
        if ( rejected != 0 ) {
            double& flag = this->_aux(VEC(i,j,k),M1_BR_REJECT_,q) ;
            flag = static_cast<double>( static_cast<int>(flag) | rejected ) ;
        }
        #endif // GRACE_M1_DIAGNOSTICS
        #endif // !GRACE_FREEZE_HYDRO && GRACE_M1_NU_SPECIES >= 3

        #else // scaled limiter
        #if GRACE_M1_NU_SPECIES >= 1
        #if (GRACE_M1_NU_SPECIES >= 5)
        constexpr int n_species = 5;
        #elif (GRACE_M1_NU_SPECIES >= 3)
        constexpr int n_species = 3;
        #else
        constexpr int n_species = 1;
        #endif

        #ifndef GRACE_FREEZE_HYDRO
            // ── Accumulate dE and dS over all species ────────────────────────────
            double dE = 0., dSx = 0., dSy = 0., dSz = 0. ;
            #pragma unroll
            for( int ispec = 0; ispec < n_species; ++ispec ) {
                dE  += this->_state(VEC(i,j,k),ERAD1_+ispec*GRACE_N_M1_VARS,q) - state_new(VEC(i,j,k),ERAD1_+ispec*GRACE_N_M1_VARS,q) ;
                dSx += this->_state(VEC(i,j,k),FRADX1_+ispec*GRACE_N_M1_VARS,q) - state_new(VEC(i,j,k),FRADX1_+ispec*GRACE_N_M1_VARS,q) ;
                dSy += this->_state(VEC(i,j,k),FRADY1_+ispec*GRACE_N_M1_VARS,q) - state_new(VEC(i,j,k),FRADY1_+ispec*GRACE_N_M1_VARS,q) ;
                dSz += this->_state(VEC(i,j,k),FRADZ1_+ispec*GRACE_N_M1_VARS,q) - state_new(VEC(i,j,k),FRADZ1_+ispec*GRACE_N_M1_VARS,q) ;
            }

            // ── Energy positivity check ──────────────────────────────────────────
            double const tau_old = state_new(VEC(i,j,k),TAU_,q) ;
            bool const energy_good = ( tau_old + dE > 0. ) ;

            double const factor_tau = ( dE < 0.0 ) ? ( -tau_old / dE ) : 1.0 ;
            double const limiting_factor_E = energy_good ? 1.0 :
                                             ( factor_tau >= 0.0 && factor_tau <= 1.0 ) ? factor_tau * (1.0 - 1e-10) :
                                             1.0 ;

            // Apply scaled backreaction to hydro
            state_new(VEC(i,j,k),TAU_,q) += limiting_factor_E * dE  ;
            state_new(VEC(i,j,k),SX_,q)  += limiting_factor_E * dSx ;
            state_new(VEC(i,j,k),SY_,q)  += limiting_factor_E * dSy ;
            state_new(VEC(i,j,k),SZ_,q)  += limiting_factor_E * dSz ;

            // The fluid absorbed limiting_factor_E of the exchange; the
            // radiation keeps the complementary fraction so energy AND momentum
            // are conserved (mirror of the Ye/Ymu number limiter below).
            // Branchless convex blend of the post-collision (new) and
            // pre-collision (old) states: bounded, causality-preserving, and a
            // no-op when energy_good (keep == 0).  Reverting to old while the
            // fluid keeps a partial deposit would DESTROY energy.
            const double keep = 1.0 - limiting_factor_E ;
            #pragma unroll
            for( int ispec = 0; ispec < n_species; ++ispec ) {
                const int off = ispec*GRACE_N_M1_VARS ;
                state_new(VEC(i,j,k),ERAD1_ +off,q) = limiting_factor_E * state_new(VEC(i,j,k),ERAD1_ +off,q) + keep * this->_state(VEC(i,j,k),ERAD1_ +off,q) ;
                state_new(VEC(i,j,k),FRADX1_+off,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADX1_+off,q) + keep * this->_state(VEC(i,j,k),FRADX1_+off,q) ;
                state_new(VEC(i,j,k),FRADY1_+off,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADY1_+off,q) + keep * this->_state(VEC(i,j,k),FRADY1_+off,q) ;
                state_new(VEC(i,j,k),FRADZ1_+off,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADZ1_+off,q) + keep * this->_state(VEC(i,j,k),FRADZ1_+off,q) ;
            }
        #endif // GRACE_FREEZE_HYDRO
        #endif // GRACE_M1_NU_SPECIES >= 1

        #if GRACE_M1_NU_SPECIES >= 3
        // Baryon density (densitized): shared by the Ye and Ymu channels.
        double const D = state_new(VEC(i,j,k),DENS_,q) ;

        // We define dN in sense Ye. Old - New. Less nrad_e more ye.
        const double dN1 = this->_state(VEC(i,j,k),NRAD1_,q)
           - state_new(VEC(i,j,k),NRAD1_,q) ;
        const double dN2 = this->_state(VEC(i,j,k),NRAD2_,q)
           - state_new(VEC(i,j,k),NRAD2_,q) ;

        // Ye bounds check
        double yemax = eos.get_c2p_ye_max();
        double yemin = eos.get_c2p_ye_min();
        double const dye_old  = state_new(VEC(i,j,k),YESTAR_,q) ;
        double const ye_old   = dye_old / D ;
        double const dye_new  = dye_old + dN1 - dN2 ;
        double const ye_new   = dye_new / D ;
        bool const number_e_good = ( ye_new >= yemin && ye_new <= yemax ) ;

        double const factor_max = (yemax - ye_old) * D / (dN1 - dN2) ;
        double const factor_min = (yemin - ye_old) * D / (dN1 - dN2) ;

        // Pick the tightest limiting factor, defaulting to 1 if in bounds
        double const limiting_factor = (factor_max >= 0.0 && factor_max <= 1.0) ? factor_max * (1.0 - 1e-10) :
                                          (factor_min >= 0.0 && factor_min <= 1.0) ? factor_min * (1.0 - 1e-10) :
                                          1.0 ;

        if ( number_e_good ) {
            state_new(VEC(i,j,k),YESTAR_,q) = dye_new ;
        }
        else {
            // Scale back the neutrino number updates by the limiting factor
            state_new(VEC(i,j,k),NRAD1_,q) = this->_state(VEC(i,j,k),NRAD1_,q)
                                                - limiting_factor * dN1 ;
            state_new(VEC(i,j,k),NRAD2_,q) = this->_state(VEC(i,j,k),NRAD2_,q)
                                                - limiting_factor * dN2 ;
            state_new(VEC(i,j,k),YESTAR_,q) = dye_old + limiting_factor * (dN1 - dN2) ;
        }
        #endif // GRACE_M1_NU_SPECIES >= 3

        #if GRACE_M1_NU_SPECIES >= 5
        const double dN3 = this->_state(VEC(i,j,k),NRAD3_,q)
           - state_new(VEC(i,j,k),NRAD3_,q) ;
        const double dN4 = this->_state(VEC(i,j,k),NRAD4_,q)
           - state_new(VEC(i,j,k),NRAD4_,q) ;

        double const ymumax     = eos.get_c2p_ymu_max() ;
        double const ymumin     = eos.get_c2p_ymu_min() ;
        double const dymu_old   = state_new(VEC(i,j,k),YMUSTAR_,q) ;
        double const ymu_old    = dymu_old / D ;
        double const dymu_new   = dymu_old + (dN3 - dN4) ;
        double const ymu_new    = dymu_new / D ;
        bool const number_mu_good = ( ymu_new >= ymumin && ymu_new <= ymumax ) ;

        double const fac_max_mu = (ymumax - ymu_old) * D / (dN3 - dN4) ;
        double const fac_min_mu = (ymumin - ymu_old) * D / (dN3 - dN4) ;

        double const limiting_factor_mu = (fac_max_mu >= 0.0 && fac_max_mu <= 1.0) ? fac_max_mu * (1.0 - 1e-10) :
                                           (fac_min_mu >= 0.0 && fac_min_mu <= 1.0) ? fac_min_mu * (1.0 - 1e-10) :
                                           1.0 ;

        if ( number_mu_good ) {
            state_new(VEC(i,j,k),YMUSTAR_,q) = dymu_new ;
        } else {
            state_new(VEC(i,j,k),NRAD3_,q)   = this->_state(VEC(i,j,k),NRAD3_,q)
                                              - limiting_factor_mu * dN3 ;
            state_new(VEC(i,j,k),NRAD4_,q)   = this->_state(VEC(i,j,k),NRAD4_,q)
                                              - limiting_factor_mu * dN4 ;
            state_new(VEC(i,j,k),YMUSTAR_,q) = dymu_old + limiting_factor_mu * (dN3 - dN4) ;
        }
        #endif // GRACE_M1_NU_SPECIES >= 5
        #endif // GRACE_M1_BACKREACT_HARDSTOP
    }

    #ifdef GRACE_M1_PHOTONS
    /**
     * @brief Photon backreaction onto the hydro state.
     *
     * Photons exchange energy and momentum with the fluid but carry no
     * lepton number: there is deliberately no analogue of the dN -> Ye/Ymu
     * coupling of the neutrino add_backreaction.  The energy-positivity
     * limiter mirrors the neutrino one; on failure the photon block is
     * reverted to its pre-collision state.
     */
    void KOKKOS_INLINE_FUNCTION
    add_backreaction_photons( const int q
                         , VEC( const int i
                         ,      const int j
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const /*idx*/
                         , grace::var_array_t const state_new ) const
    {
        using namespace grace  ;

        #ifndef GRACE_FREEZE_HYDRO
        double const dE  = this->_state(VEC(i,j,k),ERADPH_,q)  - state_new(VEC(i,j,k),ERADPH_,q)  ;
        double const dSx = this->_state(VEC(i,j,k),FRADXPH_,q) - state_new(VEC(i,j,k),FRADXPH_,q) ;
        double const dSy = this->_state(VEC(i,j,k),FRADYPH_,q) - state_new(VEC(i,j,k),FRADYPH_,q) ;
        double const dSz = this->_state(VEC(i,j,k),FRADZPH_,q) - state_new(VEC(i,j,k),FRADZPH_,q) ;

        // Energy positivity check (same limiter as the neutrino version).
        double const tau_old = state_new(VEC(i,j,k),TAU_,q) ;
        bool const energy_good = ( tau_old + dE > 0. ) ;

        double const factor_tau = ( dE < 0.0 ) ? ( -tau_old / dE ) : 1.0 ;
        double const limiting_factor_E = energy_good ? 1.0 :
                                         ( factor_tau >= 0.0 && factor_tau <= 1.0 ) ? factor_tau * (1.0 - 1e-10) :
                                         1.0 ;

        #if GRACE_M1_BACKREACT_HARDSTOP
        if ( energy_good ) {
            // accept: full deposit to the fluid, photon block kept at new
            state_new(VEC(i,j,k),TAU_,q) += dE  ;
            state_new(VEC(i,j,k),SX_,q)  += dSx ;
            state_new(VEC(i,j,k),SY_,q)  += dSy ;
            state_new(VEC(i,j,k),SZ_,q)  += dSz ;
        } else {
            // hard stop: revert the photon block, fluid untouched (no deposit)
            state_new(VEC(i,j,k),ERADPH_,q)  = this->_state(VEC(i,j,k),ERADPH_,q)  ;
            state_new(VEC(i,j,k),NRADPH_,q)  = this->_state(VEC(i,j,k),NRADPH_,q)  ;
            state_new(VEC(i,j,k),FRADXPH_,q) = this->_state(VEC(i,j,k),FRADXPH_,q) ;
            state_new(VEC(i,j,k),FRADYPH_,q) = this->_state(VEC(i,j,k),FRADYPH_,q) ;
            state_new(VEC(i,j,k),FRADZPH_,q) = this->_state(VEC(i,j,k),FRADZPH_,q) ;
        }
        #else
        state_new(VEC(i,j,k),TAU_,q) += limiting_factor_E * dE  ;
        state_new(VEC(i,j,k),SX_,q)  += limiting_factor_E * dSx ;
        state_new(VEC(i,j,k),SY_,q)  += limiting_factor_E * dSy ;
        state_new(VEC(i,j,k),SZ_,q)  += limiting_factor_E * dSz ;
        // Radiation keeps the fraction the fluid did not absorb — branchless
        // convex blend of post-collision (new) and pre-collision (old); a no-op
        // when energy_good (keep == 0).  Conserves energy/momentum; reverting
        // to old while the fluid keeps a partial deposit would destroy it.  N
        // rides the same factor so the photon number stays consistent with E.
        const double keep = 1.0 - limiting_factor_E ;
        state_new(VEC(i,j,k),ERADPH_,q)  = limiting_factor_E * state_new(VEC(i,j,k),ERADPH_,q)  + keep * this->_state(VEC(i,j,k),ERADPH_,q)  ;
        state_new(VEC(i,j,k),NRADPH_,q)  = limiting_factor_E * state_new(VEC(i,j,k),NRADPH_,q)  + keep * this->_state(VEC(i,j,k),NRADPH_,q)  ;
        state_new(VEC(i,j,k),FRADXPH_,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADXPH_,q) + keep * this->_state(VEC(i,j,k),FRADXPH_,q) ;
        state_new(VEC(i,j,k),FRADYPH_,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADYPH_,q) + keep * this->_state(VEC(i,j,k),FRADYPH_,q) ;
        state_new(VEC(i,j,k),FRADZPH_,q) = limiting_factor_E * state_new(VEC(i,j,k),FRADZPH_,q) + keep * this->_state(VEC(i,j,k),FRADZPH_,q) ;
        #endif // GRACE_M1_BACKREACT_HARDSTOP
        #endif
    }
    #endif /* GRACE_M1_PHOTONS */

    /**
     * @brief Compute maximum absolute value eigenspeed.
     *
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     * @return double Maximum eigenspeed of GRMHD equations.
     */
    template< int ispec >
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_max_eigenspeed( VEC( const int i
                          ,      const int j
                          ,      const int k)
                          , int64_t q ) const
    {
        using namespace grace;
        using namespace Kokkos ;
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        // read in eas
        m1_eas_array_t eas ;
        eas[KAL]   = this->_aux(VEC(i,j,k),m1_kappaa_idx<ispec>(),q) ;
        eas[KSL]   = this->_aux(VEC(i,j,k),m1_kappas_idx<ispec>(),q) ;
        eas[ETAL]  = this->_aux(VEC(i,j,k),m1_eta_idx<ispec>(),q) ;
        eas[ETANL] = this->_aux(VEC(i,j,k),m1_etan_idx<ispec>(),q) ;
        eas[KANL]  = this->_aux(VEC(i,j,k),m1_kappaan_idx<ispec>(),q) ;
        /**************************************************************************************************/
        // construct closure and update
        m1_prims_array_t prims ;
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,ispec,VEC(i,j,k)) ;
        prims[ERADL] /= metric.sqrtg() ;
        prims[NRADL] /= metric.sqrtg() ;
        prims[FXL] /= metric.sqrtg();
        prims[FYL] /= metric.sqrtg();
        prims[FZL] /= metric.sqrtg();

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        /**************************************************************************************************/
        double cmax=0. ;
        {
            double cp, cm ;
            compute_cp_cm<0>(cp,cm,cl,metric) ;
            cmax = Kokkos::fmax(cmax,Kokkos::fmax(Kokkos::fabs(cp),Kokkos::fabs(cm))) ;
        }
        {
            double cp, cm ;
            compute_cp_cm<1>(cp,cm,cl,metric) ;
            cmax = Kokkos::fmax(cmax,Kokkos::fmax(Kokkos::fabs(cp),Kokkos::fabs(cm))) ;
        }
        {
            double cp, cm ;
            compute_cp_cm<2>(cp,cm,cl,metric) ;
            cmax = Kokkos::fmax(cmax,Kokkos::fmax(Kokkos::fabs(cp),Kokkos::fabs(cm))) ;
        }
        return cmax ;
    }
    private:
    /***********************************************************************/
    //! Number of reconstructed variables.
    static constexpr unsigned int M1_NUM_RECON_VARS = 7 ;

    //! Parameters for atmosphere
    m1_atmo_params_t atmo_params;
    //! Parameters for excision
    m1_excision_params_t excision_params;
    //! Parameters for backreaction
    m1_backreaction_params_t backreaction_params;

    //! Pre- minus post-collision change of one radiation block (densitised,
    //! so sqrtg cancels): what the fluid receives if the block is accepted.
    struct m1_exchange_t { double E, Sx, Sy, Sz, N ; } ;

    template< int ispec >
    m1_exchange_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    species_exchange( int const q, VEC(int const i, int const j, int const k)
                    , grace::var_array_t const state_new ) const
    {
        return { this->_state(VEC(i,j,k),m1_erad_idx<ispec>(),q)  - state_new(VEC(i,j,k),m1_erad_idx<ispec>(),q)
               , this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) - state_new(VEC(i,j,k),m1_fradx_idx<ispec>(),q)
               , this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) - state_new(VEC(i,j,k),m1_frady_idx<ispec>(),q)
               , this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) - state_new(VEC(i,j,k),m1_fradz_idx<ispec>(),q)
               , this->_state(VEC(i,j,k),m1_nrad_idx<ispec>(),q)  - state_new(VEC(i,j,k),m1_nrad_idx<ispec>(),q) } ;
    }

    //! Hard-stop revert: the block gets its pre-collision E, F and N back.
    template< int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    revert_species( int const q, VEC(int const i, int const j, int const k)
                  , grace::var_array_t const state_new ) const
    {
        state_new(VEC(i,j,k),m1_erad_idx<ispec>(),q)  = this->_state(VEC(i,j,k),m1_erad_idx<ispec>(),q) ;
        state_new(VEC(i,j,k),m1_fradx_idx<ispec>(),q) = this->_state(VEC(i,j,k),m1_fradx_idx<ispec>(),q) ;
        state_new(VEC(i,j,k),m1_frady_idx<ispec>(),q) = this->_state(VEC(i,j,k),m1_frady_idx<ispec>(),q) ;
        state_new(VEC(i,j,k),m1_fradz_idx<ispec>(),q) = this->_state(VEC(i,j,k),m1_fradz_idx<ispec>(),q) ;
        state_new(VEC(i,j,k),m1_nrad_idx<ispec>(),q)  = this->_state(VEC(i,j,k),m1_nrad_idx<ispec>(),q) ;
    }

    //! Partial acceptance: the block keeps old + f (post - old) of E, F and N alike,
    //! so the fluid's share f of the exchange conserves energy and lepton number.
    template< int ispec >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    blend_species( int const q, VEC(int const i, int const j, int const k)
                 , grace::var_array_t const state_new, double const f ) const
    {
        auto const blend = [&](int const v) {
            state_new(VEC(i,j,k),v,q) = this->_state(VEC(i,j,k),v,q)
                                      + f * ( state_new(VEC(i,j,k),v,q) - this->_state(VEC(i,j,k),v,q) ) ;
        } ;
        blend(m1_erad_idx<ispec>()) ; blend(m1_fradx_idx<ispec>()) ; blend(m1_frady_idx<ispec>()) ;
        blend(m1_fradz_idx<ispec>()) ; blend(m1_nrad_idx<ispec>()) ;
    }
    /***********************************************************************/
    /***********************************************************************/
    /**
     * @brief Compute fluxes for m1 equations.
     *
     * @tparam idir Direction the fluxes are computed in.
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @param i zero-offset x cell index.
     * @param j zero-offset y cell index.
     * @param k zero-offset z cell index.
     * @param q quadrant index.
     * @param ngz Number of ghost-zones.
     * @param fluxes Flux array.
     */
    template< int idir
            , int ispec
            , typename recon_t   >
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE void
    getflux(  VEC( const int i
            ,      const int j
            ,      const int k)
            , const int64_t q
            , grace::flux_array_t const fluxes
            , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
            , double const dt
            , double const dtfact ) const
    {
        /***********************************************************************/
        /* Initialize reconstructor and riemann solver                         */
        /***********************************************************************/
        recon_t reconstructor{} ;
        /***********************************************************************/
        /* 4-point Lagrange interpolation of the metric at the cell interface, */
        /* with pair-symmetric summation so the face values are bit-mirror     */
        /* under the discrete symmetries (see grmhd_helpers.hh).               */
        /***********************************************************************/
        auto const metric_face = compute_face_metric(this->_state, VEC(i,j,k), q, idir);
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        std::array<int, 5>
            recon_indices{
                  m1_erad_idx<ispec>()
                , m1_nrad_idx<ispec>()
                , m1_fradx_idx<ispec>()
                , m1_frady_idx<ispec>()
                , m1_fradz_idx<ispec>()
            } ;
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 5>
            recon_indices_loc{
                  ERADL
                , NRADL
                , FXL
                , FYL
                , FZL
            } ;
        /* Reconstruction                                  */
        m1_prims_array_t primL, primR ;
        #pragma unroll 5
        for( int ivar=0; ivar<5; ++ivar) {
            auto u = Kokkos::subview( this->_state
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                                    , recon_indices[ivar]
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_loc[ivar]]
                         , primR[recon_indices_loc[ivar]]
                         , idir) ;
        }
        // now we need to reconstruct zvec from the hydro
        std::array<int, 3>
            recon_indices_aux{
                  ZVECX_
                , ZVECY_
                , ZVECZ_
            } ;
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 3>
            recon_indices_aux_loc{
                  ZXL
                , ZYL
                , ZZL
            } ;
        #pragma unroll 3
        for( int ivar=0; ivar<3; ++ivar) {
            auto u = Kokkos::subview( this->_aux
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                                    , recon_indices_aux[ivar]
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_aux_loc[ivar]]
                         , primR[recon_indices_aux_loc[ivar]]
                         , idir) ;
        }
        // note that at this stage F is actually F/E, we need to fix that here
        for( int ii=0; ii<3; ++ii) {
            primL[FXL+ii] *= primL[ERADL] ;
            primR[FXL+ii] *= primR[ERADL] ;
        }
        // ditto for N
        primL[NRADL] *= primL[ERADL] ;
        primR[NRADL] *= primR[ERADL] ;
        // closures
        m1_closure_t cl{
            primL[ERADL],
            {primL[FXL], primL[FYL], primL[FZL]},
            {primL[ZXL], primL[ZYL], primL[ZZL]},
            metric_face
        },
        cr{
            primR[ERADL],
            {primR[FXL], primR[FYL], primR[FZL]},
            {primR[ZXL], primR[ZYL], primR[ZZL]},
            metric_face
        };

        cl.update_closure(0) ; cr.update_closure(0) ;
        cl.compute_pressure(); cr.compute_pressure() ;
        // compute P^i_j
        int imap[3][3] = {
            {0,1,2}, {1,3,4}, {2,4,5}
        } ;
        auto const PUU_l = cl.PUU ; auto const PUU_r = cr.PUU ;
        auto const PUD_l = metric_face.lower(
            {PUU_l[idir][0], PUU_l[idir][1],PUU_l[idir][2]}
        ) ;
        auto const PUD_r = metric_face.lower(
            {PUU_r[idir][0], PUU_r[idir][1],PUU_r[idir][2]}
        ) ;
        // A factor for the asymptotic flux correction.  Face opacity is the geometric
        // mean of the two adjacent cells (as FIL): mirror symmetric, unlike cell i alone.
        int const im = i - utils::delta(0,idir) ;
        int const jm = j - utils::delta(1,idir) ;
        #ifdef GRACE_3D
        int const km = k - utils::delta(2,idir) ;
        #endif
        double const kappa_R = transport_opacity<ispec>(q,VEC(i,j,k));
        double const kappa_L = transport_opacity<ispec>(q,VEC(im,jm,km));
        double const _dx = dx(idir,q);
        // the fmax clamps A to [0,1] and guards the division
        double const A = 1./( _dx * Kokkos::fmax(Kokkos::sqrt(kappa_L*kappa_R),1./_dx) ) ;
        // compute one component of the upper-index flux for the E flux

        double FUd_l = metric_face.invgamma(imap[idir][0]) * primL[FXL]
                     + metric_face.invgamma(imap[idir][1]) * primL[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primL[FZL] ;
        double FUd_r = metric_face.invgamma(imap[idir][0]) * primR[FXL]
                     + metric_face.invgamma(imap[idir][1]) * primR[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primR[FZL] ;

        // compute wave speeds
        double cmin, cmax ;
        double cpr, cmr, cpl, cml;
        compute_cp_cm<idir>(cpl,cml, cl, metric_face) ;
        compute_cp_cm<idir>(cpr,cmr, cr, metric_face) ;
        cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ;
        cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ;
        /* Add some diffusion in weakly hyperbolic limit */
        if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }

        // compute the fluxes
        // E
        double E_l = primL[ERADL] * metric_face.sqrtg() ;
        double E_r = primR[ERADL] * metric_face.sqrtg() ;
        double f_E_l = metric_face.sqrtg() * (metric_face.alp() * FUd_l - metric_face.beta(idir) * primL[ERADL]) ;
        double f_E_r = metric_face.sqrtg() * (metric_face.alp() * FUd_r - metric_face.beta(idir) * primR[ERADL]) ;
        //fluxes(VEC(i,j,k),m1_erad_idx<ispec>(),idir,q) = (cmax*f_E_l + cmin*f_E_r - A * cmax * cmin * (E_r-E_l))/(cmax+cmin) ;
        double f_E_HLLE = (cmax*f_E_l + cmin*f_E_r - A * cmax * cmin * (E_r-E_l))/(cmax+cmin) ;
        // Fx
        double Fx_l = primL[FXL] * metric_face.sqrtg() ;
        double Fx_r = primR[FXL] * metric_face.sqrtg() ;
        double f_Fx_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[0] - metric_face.beta(idir) * primL[FXL]) ;
        double f_Fx_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[0] - metric_face.beta(idir) * primR[FXL]) ;
        //fluxes(VEC(i,j,k),m1_fradx_idx<ispec>(),idir,q) = (SQR(A)*(cmax*f_Fx_l + cmin*f_Fx_r) - A * cmax * cmin * (Fx_r-Fx_l))/(cmax+cmin)
        //                            + (1-SQR(A)) * 0.5 * (f_Fx_l+f_Fx_r);
        double f_Fx_HLLE = (SQR(A)*(cmax*f_Fx_l + cmin*f_Fx_r) - A * cmax * cmin * (Fx_r-Fx_l))/(cmax+cmin)
                                    + (1-SQR(A)) * 0.5 * (f_Fx_l+f_Fx_r);
        // Fy
        double Fy_l = primL[FYL] * metric_face.sqrtg() ;
        double Fy_r = primR[FYL] * metric_face.sqrtg() ;
        double f_Fy_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[1] - metric_face.beta(idir) * primL[FYL]) ;
        double f_Fy_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[1] - metric_face.beta(idir) * primR[FYL]) ;
        //fluxes(VEC(i,j,k),m1_frady_idx<ispec>(),idir,q) = (SQR(A)*(cmax*f_Fy_l + cmin*f_Fy_r) - A * cmax * cmin * (Fy_r-Fy_l))/(cmax+cmin)
        //                            + (1-SQR(A)) * 0.5 * (f_Fy_l+f_Fy_r);
        double f_Fy_HLLE = (SQR(A)*(cmax*f_Fy_l + cmin*f_Fy_r) - A * cmax * cmin * (Fy_r-Fy_l))/(cmax+cmin)
                                    + (1-SQR(A)) * 0.5 * (f_Fy_l+f_Fy_r);
        // Fz
        double Fz_l = primL[FZL] * metric_face.sqrtg() ;
        double Fz_r = primR[FZL] * metric_face.sqrtg() ;
        double f_Fz_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[2] - metric_face.beta(idir) * primL[FZL]) ;
        double f_Fz_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[2] - metric_face.beta(idir) * primR[FZL]) ;
        //fluxes(VEC(i,j,k),m1_fradz_idx<ispec>(),idir,q) = (SQR(A)*(cmax*f_Fz_l + cmin*f_Fz_r) - A * cmax * cmin * (Fz_r-Fz_l))/(cmax+cmin)
        //                            + (1-SQR(A)) * 0.5 * (f_Fz_l+f_Fz_r);
        double f_Fz_HLLE = (SQR(A)*(cmax*f_Fz_l + cmin*f_Fz_r) - A * cmax * cmin * (Fz_r-Fz_l))/(cmax+cmin)
                                    + (1-SQR(A)) * 0.5 * (f_Fz_l+f_Fz_r);
        // Nrad
        double N_l = primL[NRADL] *  metric_face.sqrtg() ;
        double N_r = primR[NRADL] *  metric_face.sqrtg() ;
        // N_l/r already include sqrt(gamma): densitize the number current once,
        // just as for energy. A second factor changes the transport velocity.
        double f_N_l = metric_face.alp() * N_l/cl.Gamma * ( cl.W * (cl.vU[idir]-metric_face.beta(idir)/metric_face.alp()) + cl.HU[idir]/cl.J ) ;
        double f_N_r = metric_face.alp() * N_r/cr.Gamma * ( cr.W * (cr.vU[idir]-metric_face.beta(idir)/metric_face.alp()) + cr.HU[idir]/cr.J ) ;
        //fluxes(VEC(i,j,k),m1_nrad_idx<ispec>(),idir,q) = (cmax*f_N_l + cmin*f_N_r - A * cmax * cmin * (N_r-N_l))/(cmax+cmin) ;
        double f_N_HLLE = (cmax*f_N_l + cmin*f_N_r - A * cmax * cmin * (N_r-N_l))/(cmax+cmin) ;

        //#define M1_USE_PPLIM   // superseded by M1-aware FOFC; also mis-wired (a2CFL uses dt<-t). See flag_fofc_cells.
        #ifdef M1_USE_PPLIM
        {
            // Proper LLF flux: Godunov (0th-order) reconstruction gives cell-centred
            // primitives; full closures are built from those to compute exact physical
            // fluxes for both cells at this face.
            godunov_reconstructor_t gd_reconstruction{} ;
            m1_prims_array_t primL_LLF, primR_LLF ;
            #pragma unroll 5
            for( int ivar=0; ivar<5; ++ivar) {
                auto u = Kokkos::subview( this->_state
                                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                                        , recon_indices[ivar], q ) ;
                gd_reconstruction( u, VEC(i,j,k)
                                 , primL_LLF[recon_indices_loc[ivar]]
                                 , primR_LLF[recon_indices_loc[ivar]]
                                 , idir ) ;
            }
            #pragma unroll 3
            for( int ivar=0; ivar<3; ++ivar) {
                auto u = Kokkos::subview( this->_aux
                                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                                        , recon_indices_aux[ivar], q ) ;
                gd_reconstruction( u, VEC(i,j,k)
                                 , primL_LLF[recon_indices_aux_loc[ivar]]
                                 , primR_LLF[recon_indices_aux_loc[ivar]]
                                 , idir ) ;
            }
            for( int ii=0; ii<3; ++ii) {
                primL_LLF[FXL+ii] *= primL_LLF[ERADL] ;
                primR_LLF[FXL+ii] *= primR_LLF[ERADL] ;
            }
            primL_LLF[NRADL] *= primL_LLF[ERADL] ;
            primR_LLF[NRADL] *= primR_LLF[ERADL] ;

            // Build closures from cell-centred states
            m1_closure_t cl_LLF{
                primL_LLF[ERADL],
                {primL_LLF[FXL], primL_LLF[FYL], primL_LLF[FZL]},
                {primL_LLF[ZXL], primL_LLF[ZYL], primL_LLF[ZZL]},
                metric_face
            } ;
            m1_closure_t cr_LLF{
                primR_LLF[ERADL],
                {primR_LLF[FXL], primR_LLF[FYL], primR_LLF[FZL]},
                {primR_LLF[ZXL], primR_LLF[ZYL], primR_LLF[ZZL]},
                metric_face
            } ;
            cl_LLF.update_closure(0) ; cr_LLF.update_closure(0) ;
            cl_LLF.compute_pressure() ; cr_LLF.compute_pressure() ;

            auto const PUD_l_LLF = metric_face.lower(
                {cl_LLF.PUU[idir][0], cl_LLF.PUU[idir][1], cl_LLF.PUU[idir][2]}
            ) ;
            auto const PUD_r_LLF = metric_face.lower(
                {cr_LLF.PUU[idir][0], cr_LLF.PUU[idir][1], cr_LLF.PUU[idir][2]}
            ) ;

            double const FUd_l_LLF = metric_face.invgamma(imap[idir][0]) * primL_LLF[FXL]
                                   + metric_face.invgamma(imap[idir][1]) * primL_LLF[FYL]
                                   + metric_face.invgamma(imap[idir][2]) * primL_LLF[FZL] ;
            double const FUd_r_LLF = metric_face.invgamma(imap[idir][0]) * primR_LLF[FXL]
                                   + metric_face.invgamma(imap[idir][1]) * primR_LLF[FYL]
                                   + metric_face.invgamma(imap[idir][2]) * primR_LLF[FZL] ;

            // Cell-centred conserved variables and physical fluxes; LF uses c=1
            double const E_l_LLF   = primL_LLF[ERADL] * metric_face.sqrtg() ;
            double const E_r_LLF   = primR_LLF[ERADL] * metric_face.sqrtg() ;
            double const f_E_l_LLF = metric_face.sqrtg() * (metric_face.alp() * FUd_l_LLF - metric_face.beta(idir) * primL_LLF[ERADL]) ;
            double const f_E_r_LLF = metric_face.sqrtg() * (metric_face.alp() * FUd_r_LLF - metric_face.beta(idir) * primR_LLF[ERADL]) ;
            double const f_E_LF    = 0.5*(f_E_l_LLF + f_E_r_LLF) - 0.5*(E_r_LLF - E_l_LLF) ;

            double const Fx_l_LLF   = primL_LLF[FXL] * metric_face.sqrtg() ;
            double const Fx_r_LLF   = primR_LLF[FXL] * metric_face.sqrtg() ;
            double const f_Fx_l_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_l_LLF[0] - metric_face.beta(idir) * primL_LLF[FXL]) ;
            double const f_Fx_r_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_r_LLF[0] - metric_face.beta(idir) * primR_LLF[FXL]) ;
            double const f_Fx_LF    = 0.5*(f_Fx_l_LLF + f_Fx_r_LLF) - 0.5*(Fx_r_LLF - Fx_l_LLF) ;

            double const Fy_l_LLF   = primL_LLF[FYL] * metric_face.sqrtg() ;
            double const Fy_r_LLF   = primR_LLF[FYL] * metric_face.sqrtg() ;
            double const f_Fy_l_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_l_LLF[1] - metric_face.beta(idir) * primL_LLF[FYL]) ;
            double const f_Fy_r_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_r_LLF[1] - metric_face.beta(idir) * primR_LLF[FYL]) ;
            double const f_Fy_LF    = 0.5*(f_Fy_l_LLF + f_Fy_r_LLF) - 0.5*(Fy_r_LLF - Fy_l_LLF) ;

            double const Fz_l_LLF   = primL_LLF[FZL] * metric_face.sqrtg() ;
            double const Fz_r_LLF   = primR_LLF[FZL] * metric_face.sqrtg() ;
            double const f_Fz_l_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_l_LLF[2] - metric_face.beta(idir) * primL_LLF[FZL]) ;
            double const f_Fz_r_LLF = metric_face.sqrtg() * (metric_face.alp() * PUD_r_LLF[2] - metric_face.beta(idir) * primR_LLF[FZL]) ;
            double const f_Fz_LF    = 0.5*(f_Fz_l_LLF + f_Fz_r_LLF) - 0.5*(Fz_r_LLF - Fz_l_LLF) ;

            double const N_l_LLF   = primL_LLF[NRADL] * metric_face.sqrtg() ;
            double const N_r_LLF   = primR_LLF[NRADL] * metric_face.sqrtg() ;
            double const f_N_l_LLF = metric_face.alp() * N_l_LLF/cl_LLF.Gamma
                                   * ( cl_LLF.W * (cl_LLF.vU[idir] - metric_face.beta(idir)/metric_face.alp())
                                     + cl_LLF.HU[idir]/cl_LLF.J ) ;
            double const f_N_r_LLF = metric_face.alp() * N_r_LLF/cr_LLF.Gamma
                                   * ( cr_LLF.W * (cr_LLF.vU[idir] - metric_face.beta(idir)/metric_face.alp())
                                     + cr_LLF.HU[idir]/cr_LLF.J ) ;
            double const f_N_LF    = 0.5*(f_N_l_LLF + f_N_r_LLF) - 0.5*(N_r_LLF - N_l_LLF) ;

            // Positivity check and theta computation (GRMHD-style continuous blend)
            double const a2CFL       = 6. * (dt * dtfact / _dx) ;
            double const E_atmo_cons = atmo_params.E_fl * metric_face.sqrtg() ;
            double const E_m         = E_r_LLF + a2CFL * f_E_HLLE ;   // E in right cell after this flux
            double const E_p         = E_l_LLF - a2CFL * f_E_HLLE ;   // E in left cell after this flux

            if ( E_m < E_atmo_cons || E_p < E_atmo_cons ) {
                double const denom = a2CFL * (f_E_HLLE - f_E_LF) ;
                double theta_m = 1., theta_p = 1. ;
                if ( E_m < E_atmo_cons && fabs(denom) > 0. )
                    theta_m = Kokkos::min(1., Kokkos::max(0.,  (E_atmo_cons - (E_r_LLF + a2CFL*f_E_LF)) / denom )) ;
                if ( E_p < E_atmo_cons && fabs(denom) > 0. )
                    theta_p = Kokkos::min(1., Kokkos::max(0., -(E_atmo_cons - (E_l_LLF - a2CFL*f_E_LF)) / denom )) ;
                double theta = Kokkos::min(theta_m, theta_p) ;
                if ( std::isnan(theta) ) theta = 0. ;
                double const phi = (1. - theta) * A ;
                    fluxes(VEC(i,j,k),m1_erad_idx<ispec>(),idir,q) = (1.-phi)*f_E_HLLE  + phi*f_E_LF  ;
                    fluxes(VEC(i,j,k),m1_nrad_idx<ispec>(),idir,q) = (1.-phi)*f_N_HLLE  + phi*f_N_LF  ;
                    fluxes(VEC(i,j,k),m1_fradx_idx<ispec>(),idir,q) = (1.-phi)*f_Fx_HLLE + phi*f_Fx_LF ;
                    fluxes(VEC(i,j,k),m1_frady_idx<ispec>(),idir,q) = (1.-phi)*f_Fy_HLLE + phi*f_Fy_LF ;
                    fluxes(VEC(i,j,k),m1_fradz_idx<ispec>(),idir,q) = (1.-phi)*f_Fz_HLLE + phi*f_Fz_LF ;
                } else {
                    fluxes(VEC(i,j,k),m1_erad_idx<ispec>(),idir,q) = f_E_HLLE  ;
                    fluxes(VEC(i,j,k),m1_nrad_idx<ispec>(),idir,q) = f_N_HLLE  ;
                    fluxes(VEC(i,j,k),m1_fradx_idx<ispec>(),idir,q) = f_Fx_HLLE ;
                    fluxes(VEC(i,j,k),m1_frady_idx<ispec>(),idir,q) = f_Fy_HLLE ;
                    fluxes(VEC(i,j,k),m1_fradz_idx<ispec>(),idir,q) = f_Fz_HLLE ;
                }
            }
        #else
                fluxes(VEC(i,j,k),m1_erad_idx<ispec>(),idir,q) = f_E_HLLE  ;
                fluxes(VEC(i,j,k),m1_nrad_idx<ispec>(),idir,q) = f_N_HLLE  ;
                fluxes(VEC(i,j,k),m1_fradx_idx<ispec>(),idir,q) = f_Fx_HLLE ;
                fluxes(VEC(i,j,k),m1_frady_idx<ispec>(),idir,q) = f_Fy_HLLE ;
                fluxes(VEC(i,j,k),m1_fradz_idx<ispec>(),idir,q) = f_Fz_HLLE ;
        #endif
    }

    template< size_t idir >
    GRACE_HOST_DEVICE void compute_cp_cm(
        double& cp, double &cm, m1_closure_t const& cl, metric_array_t const& metric
    ) const
    {

        int const icomp = (idir==0)*0 + (idir==1)*3 + (idir==2)*5 ;

        double dthin = cl.chi * 1.5 - 0.5 ;
        double dthick = 1.5 - cl.chi * 1.5 ;
        m1_wavespeeds(
            cl.W, dthin, dthick, cl.F, metric.alp(),
            cl.vU[idir], metric.invgamma(icomp),
            metric.beta(idir),cl.FU[idir],
            &cm, &cp
        ) ;

    }

} ;

/**************************************************************************************************/
/* Standalone functions for m1 initial data and eas calculations                                  */
/**************************************************************************************************/
//! Per-step count of beta-equilibrium solver failures (see m1.cpp).
void report_betaeq_failures() ;
#ifdef GRACE_M1_DIAGNOSTICS
//! Log how the implicit collision solves of this step ended (from m1_implicit_err/res).
void report_m1_implicit_failures() ;
#endif

template < typename eos_t >
void set_m1_eas(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux
) ;

template < typename eos_t >
void set_m1_eas() ;

template < typename eos_t >
void set_m1_initial_data() ;

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)        \
extern template                          \
void set_m1_initial_data<EOS>( );        \
extern template                          \
void set_m1_eas<EOS>(                    \
      grace::var_array_t&                \
    , grace::staggered_variable_arrays_t&\
    , grace::var_array_t&                \
);                                       \
extern template                          \
void set_m1_eas<EOS>()


INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::leptonic_eos_4d_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/
} /* namespace grace */

#endif /*GRACE_PHYSICS_M1_HH*/
