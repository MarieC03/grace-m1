/**
 * @file m1.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief
 * @date 2024-11-24
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

// configuration
#include <grace_config.h>

// general headers
#include <grace/utils/device.h>
#include <grace/utils/inline.h>

// config
#include <grace/config/config_parser.hh>

// grid
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>

// utilities
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

// m1 includes
#include <grace/physics/m1.hh>
#include <grace/physics/m1_helpers.hh>
#include <grace/utils/reductions.hh>
#include <grace/physics/m1_trigger.hh>
#include <grace/physics/eas_kinds.hh>
#include <grace/physics/eas_policies.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <grace/physics/id/m1_initial_data.hh>
#include <grace/physics/grace_weakhub_table.hh>
#ifdef GRACE_HAVE_BNS_NURATES
#include <grace/physics/bns_nurates_grace.hh>
#endif

// grmhd + eos includes
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>

// Kokkos
#include <Kokkos_Core.hpp>

// STL
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

namespace grace {

void trace_pair_moments(var_array_t state, char const* label, int stage, double factor) {
    int const steps=get_param<int>("m1","eas","pair_trace_iterations");
    if(steps==0 || get_iteration()-get_initial_iteration()>=static_cast<size_t>(steps)
       || !m1_is_active() || get_pair_treatment()!=pair_treatment_t::evolved) return;
    DECLARE_GRID_EXTENTS;
    using namespace Kokkos;
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> policy(
        {VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq});
    int const rank=parallel::mpi_comm_rank();
    for(int sp=0;sp<GRACE_M1_NU_SPECIES;++sp) {
        double minE=0,minN=0,maxMean=0;
        int bad=0;
        parallel_reduce("pair_trace_moments",policy,
            KOKKOS_LAMBDA(VEC(int const& i,int const& j,int const& k),int const& q,
                          double& e,double& n,double& mean,int& invalid) {
                int const off=sp*GRACE_N_M1_VARS;
                double const E=state(VEC(i,j,k),ERAD1_+off,q);
                double const N=state(VEC(i,j,k),NRAD1_+off,q);
                e=Kokkos::fmin(e,E); n=Kokkos::fmin(n,N);
                if(!Kokkos::isfinite(E) || !Kokkos::isfinite(N) || E<0 || N<0 || ((E==0)!=(N==0))) ++invalid;
                else if(N>0) mean=Kokkos::fmax(mean,E/N*pairs::number_unit/pairs::energy_unit);
            },Min<double>(minE),Min<double>(minN),Max<double>(maxMean),Sum<int>(bad));
        maxMean=std::max(0.0,maxMean); // no positive-N cells (e.g. exact vacuum)
        std::fprintf(stderr,"[PAIR_TRACE] rank=%d iteration=%zu phase=%s stage=%d factor=%.17e "
                     "species=%d min_densitized_E=%.17e min_densitized_N=%.17e "
                     "max_lab_mean_MeV=%.17e invalid_EN_cells=%d\n",
                     rank,get_iteration(),label,stage,factor,sp,minE,minN,maxMean,bad);
    }
    std::fflush(stderr);
}

void check_pair_failures(pairs::failure_buffer const& failures, var_array_t state,
                         var_array_t aux, char const* phase, double stage_h) {
    if(!failures.count.data()) return;
    // Synchronizes the producing kernel, including all writes to its records.
    auto const count=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),failures.count);
    if(count()==0) return;
    int const n=std::min(count(),pairs::failure_buffer::capacity);
    auto records=failures.records;
    auto coords=coordinate_system::get().get_device_coord_system();
    Kokkos::parallel_for("pair_failure_context",n,KOKKOS_LAMBDA(int slot) {
        auto& r=records(slot);
        int const i=r.i,j=r.j,k=r.k;
        auto const q=r.q;
        coords.get_physical_coordinates(i,j,k,q,r.xyz);
        r.hydro[0]=aux(VEC(i,j,k),RHO_,q);
        r.hydro[1]=aux(VEC(i,j,k),TEMP_,q);
        r.hydro[2]=aux(VEC(i,j,k),YE_,q);
        #ifdef GRACE_ENABLE_MUONS
        r.hydro[3]=aux(VEC(i,j,k),YMU_,q);
        #endif
        metric_array_t metric; FILL_METRIC_ARRAY(metric,state,q,VEC(i,j,k));
        for(int s=0;s<2;++s) {
            int const sp=s==0?r.species_a:r.species_b;
            if(sp<0) continue;
            int const offset=sp*GRACE_N_M1_VARS;
            int const v[5]={ERAD1_,FRADX1_,FRADY1_,FRADZ1_,NRAD1_};
            int const pidx[5]={ERADL,FXL,FYL,FZL,NRADL};
            m1_prims_array_t p{};
            for(int a=0;a<5;++a) {
                r.moments[s][a]=state(VEC(i,j,k),v[a]+offset,q)/(metric.sqrtg()*r.multiplicity);
                p[pidx[a]]=r.moments[s][a];
            }
            p[ZXL]=aux(VEC(i,j,k),ZVECX_,q);
            p[ZYL]=aux(VEC(i,j,k),ZVECY_,q);
            p[ZZL]=aux(VEC(i,j,k),ZVECZ_,q);
            m1_closure_t cl{p,metric}; cl.update_closure(0);
            r.fluid[s][0]=p[NRADL]/cl.Gamma/pairs::number_unit;
            r.fluid[s][1]=cl.J/pairs::energy_unit;
        }
    });
    auto const host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),records);
    int const rank=parallel::mpi_comm_rank();
    std::ostringstream os;
    os << std::scientific << std::setprecision(17);
    os << "[PAIR_FAILURE] rank=" << rank << " iteration=" << get_iteration()
       << " phase=" << phase << " stage_h_code=" << stage_h
       << " failures=" << count() << " samples=" << n << '\n';
    for(int slot=0;slot<n;++slot) {
        auto const& r=host(slot); auto const& d=r.detail;
        os << "[PAIR_FAILURE] rank=" << rank << " sample=" << slot
           << " status=" << pairs::failure_name(d.code)
           << " q=" << r.q << " ijk=" << r.i << ',' << r.j << ',' << r.k
           << " xyz_code=" << r.xyz[0] << ',' << r.xyz[1] << ',' << r.xyz[2]
           << " species_zero_based=" << r.species_a << ',' << r.species_b
           << " multiplicity=" << r.multiplicity << '\n';
        os << "  rho_code=" << r.hydro[0] << " T_MeV=" << r.T
           << " hydro_T_MeV=" << r.hydro[1] << " Ye=" << r.hydro[2] << " Ymu=" << r.hydro[3]
           << " mu_e_MeV=" << r.mu_e << " mu_mu_MeV=" << r.mu_mu
           << " nb_cm^-3=" << r.nb << " Xn=" << r.yn << " Xp=" << r.yp
           << " active_bits=" << r.active << " order=" << r.order << " scale_MeV=" << r.scale << '\n';
        for(int s=0;s<2;++s) {
            os << "  input_species=" << (s==0?r.species_a:r.species_b)
               << " E_Fx_Fy_Fz_N_code_per_species=";
            for(double v:r.moments[s]) os << v << ' ';
            os << " n_cm^-3=" << r.fluid[s][0] << " J_MeV_cm^-3=" << r.fluid[s][1]
               << " mean_MeV=" << (r.fluid[s][0]>0?r.fluid[s][1]/r.fluid[s][0]:0) << '\n';
        }
        os << "  reconstruction_species=" << d.species << " iteration=" << d.iteration
           << " residual=" << d.residual << " n_cm^-3=" << d.n << " J_MeV_cm^-3=" << d.J
           << " grid_emin_MeV=" << d.emin << " grid_emax_MeV=" << d.emax
           << " kernel_nodes=" << d.node << ',' << d.partner_node
           << " kernel_energies_MeV=" << d.energy << ',' << d.partner_energy
           << " kernel_value=" << d.kernel_value << '\n';
    }
    // stderr is explicit on EVERY failing rank; the usual console logger may
    // only emit rank zero. Flush before throwing/aborting the MPI job.
    auto const message=os.str();
    std::fputs(message.c_str(),stderr);
    std::fflush(stderr);
    ERROR(message);
}
//**************************************************************************************************
/**
 * @brief Report how many cells failed the beta-equilibrium solve this step.
 *
 * The solver's failures are otherwise invisible: every call site is
 * `if (eq_ok) {...}` with no else, so a run where it never converges behaves
 * exactly like `betaeq_policy: off` while claiming to equilibrate.  Modelled on
 * check_nans_and_act_if_due: reduce the per-cell aux(BETAEQ_ERR_) bitmask,
 * MPI-sum, emit one warning.
 */
void report_betaeq_failures() {
    using namespace grace ;
    using namespace Kokkos ;

    // Only meaningful when a solver actually runs.
    if ( get_betaeq_mode() == betaeq_mode_t::off ) return ;

    DECLARE_GRID_EXTENTS ;
    auto aux = grace::variable_list::get().getaux() ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ;

    // Two counters in one pass: total failures, and the subset that was merely
    // pinned on an EOS-table bound.  The split is the whole point -- an
    // at-bound cell is at the best equilibrium the table can represent, not a
    // broken solve, and the two used to be indistinguishable in the log.
    uint64_t constexpr at_bound_mask = uint64_t(1) << BETAEQ_AT_BOUND ;
    uint64_t constexpr res_large_mask = uint64_t(1) << BETAEQ_RESIDUAL_LARGE ;
    // The electron-only fallback bit marks a rescued cell, not a failure.
    #ifdef GRACE_ENABLE_MUONS
    uint64_t constexpr floor_mask   = uint64_t(1) << BETAEQ_YMU_FLOOR_HOLD ;
    uint64_t constexpr rescued_mask = ( uint64_t(1) << BETAEQ_PARTIAL_E_FALLBACK ) | floor_mask ;
    #else
    uint64_t constexpr floor_mask   = 0 ;
    uint64_t constexpr rescued_mask = 0 ;
    #endif
    int64_t local_failed = 0, local_bound = 0, local_res = 0, local_rescued = 0, local_floor = 0 ;
    parallel_reduce( GRACE_EXECUTION_TAG("DIAG","betaeq_failure_count")
                   , policy
                   , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                                   int const& q, int64_t& acc, int64_t& acc_b,
                                   int64_t& acc_r, int64_t& acc_e, int64_t& acc_f)
    {
        double const f = aux(VEC(i,j,k),BETAEQ_ERR_,q) ;
        if ( f != 0.0 ) {
            uint64_t const bits = static_cast<uint64_t>(f) ;
            if ( bits & rescued_mask ) ++acc_e ;
            if ( bits & floor_mask ) ++acc_f ;
            if ( bits & ~rescued_mask ) {
                ++acc ;
                if ( bits & at_bound_mask ) ++acc_b ;
                if ( bits & res_large_mask ) ++acc_r ;
            }
        }
    }, local_failed, local_bound, local_res, local_rescued, local_floor ) ;

    uint64_t local_u64[5]  = { static_cast<uint64_t>(local_failed)
                             , static_cast<uint64_t>(local_bound)
                             , static_cast<uint64_t>(local_res)
                             , static_cast<uint64_t>(local_rescued)
                             , static_cast<uint64_t>(local_floor) } ;
    uint64_t global_u64[5] = { 0, 0, 0, 0, 0 } ;
    parallel::mpi_allreduce(local_u64, global_u64, 5, sc_MPI_SUM) ;

    if ( global_u64[3] > 0 )
        GRACE_INFO("Beta-equilibrium: the electron-only fallback (Ymu held) equilibrated "
                   "{} cells at iteration {} where the joint solve failed ({} of them with "
                   "Ymu held at the table floor).",
                   global_u64[3], grace::get_iteration(), global_u64[4]) ;
    if ( global_u64[0] == 0 ) return ;
    GRACE_WARN("Beta-equilibrium solver failed in {} cells at iteration {} "
               "({} ran into a variable bound; {} of those settled there but "
               "kept a large residual, i.e. no root exists inside the bounds).  "
               "Those cells kept their un-equilibrated rates; decode the "
               "per-cell cause from the betaeq_err bitmask (betaeq_err_enum_t).",
               global_u64[0], grace::get_iteration(),
               global_u64[1], global_u64[2]) ;
}


#ifdef GRACE_M1_DIAGNOSTICS
//! Per-step report of the implicit collision solves: failures per species (from the
//! sticky m1_implicit_err mask) against the cells with a live collision term, and the
//! largest estimated error.  Silent when every solve converged to a physical state.
void report_m1_implicit_failures() {
    using namespace grace ;
    using namespace Kokkos ;

    if ( !m1_is_active() ) return ;

    DECLARE_GRID_EXTENTS ;
    auto aux = grace::variable_list::get().getaux() ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ;

    #if   GRACE_M1_NU_SPECIES >= 5
    static char const* const names[] = { "nue", "nuebar", "numu", "numubar", "nux", "photon" } ;
    #elif GRACE_M1_NU_SPECIES >= 3
    static char const* const names[] = { "nue", "nuebar", "nux", "photon" } ;
    #elif GRACE_M1_NU_SPECIES >= 1
    static char const* const names[] = { "nu", "photon" } ;
    #else
    static char const* const names[] = { "photon" } ;
    #endif
    #ifdef GRACE_M1_PHOTONS
    int constexpr nslots = GRACE_M1_NU_SPECIES + 1 ;
    #else
    int constexpr nslots = GRACE_M1_NU_SPECIES ;
    #endif

    // Per slot: live, first Newton failed, roundoff, small step, max iter, linear,
    // non-physical, explicit step taken.
    enum { LIVE=0, FAILED, ROUNDOFF, SMALLSTEP, MAXITER, LINEAR, NONPHYS, EXPLICIT, NCOUNT=8 } ;
    double local_cnt[ 6*NCOUNT ] = {0.} , global_cnt[ 6*NCOUNT ] = {0.} ;
    for ( int s=0; s<nslots; ++s ) {
        #ifdef GRACE_M1_PHOTONS
        bool const is_photon = ( s == GRACE_M1_NU_SPECIES ) ;
        #else
        bool const is_photon = false ;
        #endif
        int ka = 0 ;
        #if GRACE_M1_NU_SPECIES >= 1
        if ( !is_photon ) ka = KAPPAA1_ + s*GRACE_N_M1_AUX ;
        #endif
        #ifdef GRACE_M1_PHOTONS
        if ( is_photon )  ka = KAPPAAPH_ ;
        #endif
        int const shift = M1_IMPLICIT_ERR_STRIDE * s ;
        array_sum_t<double,8> cnt ;
        parallel_reduce( GRACE_EXECUTION_TAG("DIAG","m1_implicit_failure_count")
                       , policy
                       , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                                       int const& q, array_sum_t<double,8>& acc)
        {
            // the five rates of a species are contiguous, in the order of m1_eas_array_t
            bool live = false ;
            for ( int r=0; r<GRACE_N_M1_AUX; ++r )
                live = live || !( aux(VEC(i,j,k),ka+r,q) < M1_EAS_NEGLIGIBLE ) ;
            if ( live ) acc.data[LIVE] += 1.0 ;
            unsigned const bits =
                ( static_cast<unsigned>( aux(VEC(i,j,k),M1_IMPLICIT_ERR_,q) ) >> shift ) & 31u ;
            if ( bits & 7u )                      acc.data[FAILED]    += 1.0 ;
            if ( bits & M1_IMPLICIT_ROUNDOFF )    acc.data[ROUNDOFF]  += 1.0 ;
            if ( bits & M1_IMPLICIT_SMALLSTEP )   acc.data[SMALLSTEP] += 1.0 ;
            if ( bits & M1_IMPLICIT_MAXITER )     acc.data[MAXITER]   += 1.0 ;
            if ( bits & M1_IMPLICIT_LINEAR )      acc.data[LINEAR]    += 1.0 ;
            if ( bits & M1_IMPLICIT_NONPHYSICAL ) acc.data[NONPHYS]   += 1.0 ;
            if ( ( static_cast<unsigned>( aux(VEC(i,j,k),M1_EXPLICIT_STEP_,q) ) >> s ) & 1u )
                acc.data[EXPLICIT] += 1.0 ;
        }, Kokkos::Sum<array_sum_t<double,8>>(cnt) ) ;
        for ( int c=0; c<NCOUNT; ++c ) local_cnt[ s*NCOUNT + c ] = cnt.data[c] ;
    }
    parallel::mpi_allreduce(local_cnt, global_cnt, 6*NCOUNT, sc_MPI_SUM) ;

    double local_res = 0.0, global_res = 0.0 ;
    parallel_reduce( GRACE_EXECUTION_TAG("DIAG","m1_implicit_max_residual")
                   , policy
                   , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                                   int const& q, double& acc)
    {
        acc = Kokkos::fmax( acc, aux(VEC(i,j,k),M1_IMPLICIT_RES_,q) ) ;
    }, Kokkos::Max<double>(local_res) ) ;
    parallel::mpi_allreduce(&local_res, &global_res, 1, sc_MPI_MAX) ;

    double tot[NCOUNT] = {0.} ;
    for ( int s=0; s<nslots; ++s )
        for ( int c=0; c<NCOUNT; ++c ) tot[c] += global_cnt[ s*NCOUNT + c ] ;
    if ( tot[FAILED] == 0.0 && tot[NONPHYS] == 0.0 ) return ;

    std::ostringstream by_species ;
    by_species << std::fixed << std::setprecision(2) ;
    for ( int s=0; s<nslots; ++s ) {
        double const live = std::max( global_cnt[s*NCOUNT+LIVE], 1.0 ) ;
        by_species << (s ? ", " : "") << names[s] << " "
                   << 100.*global_cnt[s*NCOUNT+FAILED]/live << "/"
                   << 100.*global_cnt[s*NCOUNT+LINEAR]/live ;
    }
    GRACE_WARN("M1 implicit solve at iteration {}: first Newton attempt failed in {:.0f} of "
               "{:.0f} live species-cells ({:.0f} roundoff, {:.0f} small step, {:.0f} max "
               "iterations); {:.0f} took the linear fallback; {:.0f} ended non-physical; "
               "{:.0f} took the explicit step; max estimated relative error {:.3e}.  "
               "By species, % failed / % linear: {}.",
               grace::get_iteration(), tot[FAILED], tot[LIVE], tot[ROUNDOFF], tot[SMALLSTEP],
               tot[MAXITER], tot[LINEAR], tot[NONPHYS], tot[EXPLICIT], global_res,
               by_species.str()) ;
}
#endif


//**************************************************************************************************
//! Put every radiation field on its floor, with zero flux.  Called once, when the M1 trigger
//! fires: M1 was idle until then, so the fields still hold the t = 0 data, and whatever of
//! that sits ABOVE the floor is never reset again and starts to free-fall (a spurious,
//! steadily growing inward luminosity).  Interior and ghost cells alike.
void reset_m1_radiation_to_floor() {
    using namespace grace ;
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;
    auto state = grace::variable_list::get().getstate() ;
    auto const atmo = get_m1_atmo_params() ;
    auto coords = grace::coordinate_system::get().get_device_coord_system() ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL","m1_reset_radiation_to_floor"), policy
                , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k), int const& q)
    {
        double rtp[3] ;
        coords.get_physical_coordinates_sph(i,j,k,q,rtp) ;
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric,state,q,VEC(i,j,k)) ;
        double const E = metric.sqrtg() * atmo.E_floor(rtp[0]) ;
        double const N = metric.sqrtg() * atmo.N_floor(rtp[0]) ;
        for ( int s = 0 ; s < GRACE_M1_NU_SPECIES ; ++s ) {
            int const o = s*GRACE_N_M1_VARS ;
            state(VEC(i,j,k),ERAD1_+o,q)  = E ;   state(VEC(i,j,k),NRAD1_+o,q)  = N ;
            state(VEC(i,j,k),FRADX1_+o,q) = 0.0 ; state(VEC(i,j,k),FRADY1_+o,q) = 0.0 ;
            state(VEC(i,j,k),FRADZ1_+o,q) = 0.0 ;
        }
        #ifdef GRACE_M1_PHOTONS
        state(VEC(i,j,k),ERADPH_,q)  = E ;   state(VEC(i,j,k),NRADPH_,q)  = N ;
        state(VEC(i,j,k),FRADXPH_,q) = 0.0 ; state(VEC(i,j,k),FRADYPH_,q) = 0.0 ;
        state(VEC(i,j,k),FRADZPH_,q) = 0.0 ;
        #endif
    }) ;
    Kokkos::fence() ;
}

template < typename eos_t >
void set_m1_eas() {
    auto& state = grace::variable_list::get().getstate() ;
    auto& sstate = grace::variable_list::get().getstaggeredstate() ;
    auto& aux = grace::variable_list::get().getaux() ;
    set_m1_eas<eos_t>(state,sstate,aux) ;
}


template < typename eos_t >
void set_m1_eas(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux
)
{
    using namespace grace  ;
    using namespace Kokkos ;

    DECLARE_GRID_EXTENTS ;

    // M1 idle: without the diagnostics build there is nothing to produce here,
    // so skip the whole EAS pass.  With it, the providers still run in
    // fugacity-only mode (see neutrinos_eas_op::diagnostics_only).
    #ifndef GRACE_M1_DIAGNOSTICS
    if ( !m1_is_active() ) return ;
    #endif

    auto eos = eos::get().get_eos<eos_t>() ;

    auto const eas = get_eas_selection() ;
    auto const pair_mode = get_pair_treatment(); // validate before any device launch

    // Launch-bounds switch: undefined leaves the policy and tile as-is.
    // The tile product must equal MaxThreadsPerBlock or HIP rejects the launch.
#ifdef GRACE_M1_EAS_LB
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space,GRACE_M1_EAS_LB>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq},{VEC(16,4,4),1}) ;
#else
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
#endif

    // Run every selected provider, in parfile order.  Which providers may
    // coexist is validated centrally in get_eas_selection().
    for ( auto const eas_kind : eas.kinds )
    switch ( eas_kind ) {

    case eas_kind_t::test : {
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        test_eas_op op(aux) ;
        parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_eas"), policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
            {
                double xyz[3] = {
                    cart_pcoords(VEC(i,j,k),0,q),
                    cart_pcoords(VEC(i,j,k),1,q),
                    cart_pcoords(VEC(i,j,k),2,q)
                } ;
                op(VEC(i,j,k),q,xyz) ;
            }
        );
        break ;
    }

    case eas_kind_t::photon_rates : {
        auto coords = grace::coordinate_system::get().get_device_coord_system() ;
        photon_eas_op op(aux) ;
        parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_eas"), policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
            {
                double xyz[3] ;
                coords.get_physical_coordinates(i,j,k,q,xyz) ;
                op(VEC(i,j,k),q,xyz) ;
            }
        );
        break ;
    }

    case eas_kind_t::neutrino_weakhub :
        // Weakhub = the analytic neutrino flow with table-backed opacities;
        // the loader is internally guarded, so repeated calls are no-ops.
        weakhub::initialize_weakhub_from_params() ;
        [[fallthrough]] ;
    case eas_kind_t::neutrino_analytic : {
        auto coords = grace::coordinate_system::get().get_device_coord_system() ;
        neutrinos_eas_op<eos_t> op(state, aux) ;
        if(pair_mode==pair_treatment_t::evolved) op.pair_failures.allocate();
        auto const evaluate = KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
            {
                double xyz[3] ;
                coords.get_physical_coordinates(i,j,k,q,xyz) ;
                op(VEC(i,j,k),q,xyz) ;
            };
        if(pair_mode!=pair_treatment_t::legacy) {
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> pair_policy(
                {VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq},{VEC(4,2,2),1});
            parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_pair_eas"),pair_policy,evaluate);
        } else parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_eas"),policy,evaluate);
        check_pair_failures(op.pair_failures,state,aux,"transport");
        break ;
    }

    case eas_kind_t::bns_nurates :
        #ifdef GRACE_HAVE_BNS_NURATES
        // Currently being done in auxiliaries.cpp and bns_nurates.hpp
        set_m1_eas_bns_nurates<eos_t>(state, aux);
        #else
        // Unreachable: get_eas_selection() rejects this kind when the
        // submodule is absent.  Kept as a defensive backstop.
        ERROR("m1.eas kind 'bns_nurates' selected but GRACE was built "
              "without the bns_nurates submodule.") ;
        #endif
        break ;
    }
    // No default: get_eas_kind() validates the string, and -Wswitch flags
    // any enumerator added without a case here.
}


template < typename id_kernel_t >
static void set_m1_initial_data_impl(
    id_kernel_t id_kernel
)
{
    using namespace grace  ;
    using namespace Kokkos ;

    DECLARE_GRID_EXTENTS ;

    auto& state = grace::variable_list::get().getstate() ;
    auto& sstate = grace::variable_list::get().getstaggeredstate() ;
    auto& aux = grace::variable_list::get().getaux() ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    parallel_for(
        GRACE_EXECUTION_TAG("ID","set_m1_id"),
        policy,
        KOKKOS_LAMBDA (VEC(int const i, int const j, int const k), int const q) {
            auto id = id_kernel(i,j,k,q) ;
            // metric
            metric_array_t metric ;
            FILL_METRIC_ARRAY(metric,state,q,i,j,k) ;
            // set id
            #if GRACE_M1_NU_SPECIES >= 1
            state(VEC(i,j,k),ERAD1_,q)  = metric.sqrtg() * id.erad1 ;
            state(VEC(i,j,k),NRAD1_,q)  = metric.sqrtg() * id.nrad1 ;
            state(VEC(i,j,k),FRADX1_,q) = metric.sqrtg() * id.fradx1 ;
            state(VEC(i,j,k),FRADY1_,q) = metric.sqrtg() * id.frady1 ;
            state(VEC(i,j,k),FRADZ1_,q) = metric.sqrtg() * id.fradz1 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 3
            state(VEC(i,j,k),ERAD2_,q)  = metric.sqrtg() * id.erad2 ;
            state(VEC(i,j,k),NRAD2_,q)  = metric.sqrtg() * id.nrad2 ;
            state(VEC(i,j,k),FRADX2_,q) = metric.sqrtg() * id.fradx2 ;
            state(VEC(i,j,k),FRADY2_,q) = metric.sqrtg() * id.frady2 ;
            state(VEC(i,j,k),FRADZ2_,q) = metric.sqrtg() * id.fradz2 ;
            state(VEC(i,j,k),ERAD3_,q)  = metric.sqrtg() * id.erad3 ;
            state(VEC(i,j,k),NRAD3_,q)  = metric.sqrtg() * id.nrad3 ;
            state(VEC(i,j,k),FRADX3_,q) = metric.sqrtg() * id.fradx3 ;
            state(VEC(i,j,k),FRADY3_,q) = metric.sqrtg() * id.frady3 ;
            state(VEC(i,j,k),FRADZ3_,q) = metric.sqrtg() * id.fradz3 ;
            #endif
            #if GRACE_M1_NU_SPECIES >= 5
            state(VEC(i,j,k),ERAD4_,q)  = metric.sqrtg() * id.erad4 ;
            state(VEC(i,j,k),NRAD4_,q)  = metric.sqrtg() * id.nrad4 ;
            state(VEC(i,j,k),FRADX4_,q) = metric.sqrtg() * id.fradx4 ;
            state(VEC(i,j,k),FRADY4_,q) = metric.sqrtg() * id.frady4 ;
            state(VEC(i,j,k),FRADZ4_,q) = metric.sqrtg() * id.fradz4 ;
            state(VEC(i,j,k),ERAD5_,q)  = metric.sqrtg() * id.erad5 ;
            state(VEC(i,j,k),NRAD5_,q)  = metric.sqrtg() * id.nrad5 ;
            state(VEC(i,j,k),FRADX5_,q) = metric.sqrtg() * id.fradx5 ;
            state(VEC(i,j,k),FRADY5_,q) = metric.sqrtg() * id.frady5 ;
            state(VEC(i,j,k),FRADZ5_,q) = metric.sqrtg() * id.fradz5 ;
            #endif
            #ifdef GRACE_M1_PHOTONS
            // Photon block: seeded from the species-1 ID profile so the
            // existing radiation test setups (beam, scattering, vacuum)
            // drive the photon fields identically.
            state(VEC(i,j,k),ERADPH_,q)  = metric.sqrtg() * id.eradph ;
            state(VEC(i,j,k),NRADPH_,q)  = metric.sqrtg() * id.nradph ;
            state(VEC(i,j,k),FRADXPH_,q) = metric.sqrtg() * id.fradxph ;
            state(VEC(i,j,k),FRADYPH_,q) = metric.sqrtg() * id.fradyph ;
            state(VEC(i,j,k),FRADZPH_,q) = metric.sqrtg() * id.fradzph ;
            #endif
        }
    ) ;
}

template < typename eos_t >
void set_m1_initial_data() {
    using namespace grace  ;
    using namespace Kokkos ;

    DECLARE_GRID_EXTENTS ;


    auto eos = eos::get().get_eos<eos_t>() ;
    auto id_type = grace::get_param<std::string>("m1","id_type") ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto device_coord_system = coord_system.get_device_coord_system() ;

    m1_excision_params_t m1_excision_params = get_m1_excision_params() ;
    m1_atmo_params_t m1_atmo_params = get_m1_atmo_params() ;

    if (get_betaeq_mode() == betaeq_mode_t::gieg) {
        // Physics caveat: the two-timescale scheme relaxes matter and radiation
        // to (partial) EQUILIBRIUM states wherever the local equilibration time
        // is short; it does NOT evolve or capture genuine non-equilibrium
        // neutrino distributions (gray, LTE-equilibrium approximation).  Results
        // in the partially-trapped / transition regime and under rapid dynamics
        // are therefore approximate.
        GRACE_WARN("m1.eas.betaeq_policy='gieg': the two-timescale equilibration "
                   "assumes matter and radiation relax to (partial) equilibrium "
                   "states and does NOT account for non-equilibrium neutrino "
                   "distribution effects.  Treat the partially-trapped/transition "
                   "regime as approximate.") ;

        // Redundancy: gieg decides trapping from the LOCAL equilibration
        // timescale tau_beta, not from a path-integrated optical depth.  Running
        // it together with the eikonal optical depth is redundant, and the
        // tau-driven degeneracy rescaling (make_fugacity_state) is inconsistent
        // with the gieg equilibration.
        if (get_tau_policy_kind() == tau_policy_kind_t::eikonal) {
            GRACE_WARN("m1.eas.betaeq_policy='gieg' is being used together with "
                       "tau_policy='eikonal'.  The two-timescale (gieg) scheme "
                       "decides trapping from the local equilibration timescale "
                       "and makes the eikonal optical depth redundant; the "
                       "tau-driven degeneracy rescaling is also inconsistent "
                       "with it.  Consider setting m1.eas.tau_policy='none'.") ;
        }
    }

    GRACE_VERBOSE("Setting M1 initial data of type {}", id_type) ;
    if ( id_type == "straight_beam" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ;
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        straight_beam_m1_id_t id(
            m1_atmo_params, m1_excision_params, cart_pcoords
        ) ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "crossed_beams" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ;
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        crossed_beams_m1_id_t id(
            m1_atmo_params, m1_excision_params, cart_pcoords,
            grace::get_param<double>("m1","crossed_beams_test","half_width"),
            grace::get_param<double>("m1","crossed_beams_test","start")
        ) ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "scattering") {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ;
        if ( grace::get_param<bool>("m1","scattering_test","is_static_background")) {
            auto ks = grace::get_param<double>("m1","scattering_test","k_s") ;
            auto t0 = grace::get_param<double>("m1","scattering_test","t_0") ;
            coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
            grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
            scattering_diffusion_m1_id_t id(
                m1_atmo_params, m1_excision_params, cart_pcoords, ks, t0
            ) ;
            set_m1_initial_data_impl(id) ;
        } else {
            auto v0 = grace::get_param<double>("grmhd","vacuum","velocity_x") ;
            ASSERT(v0!=0.0, "0 velocity but test is not static") ;
            coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
            grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
            moving_scattering_diffusion_m1_id_t id(
                m1_atmo_params, m1_excision_params, cart_pcoords, v0
            ) ;
            set_m1_initial_data_impl(id) ;
        }
    } else if ( id_type == "shadow" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ;
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        straight_beam_m1_id_t id(
            m1_atmo_params, m1_excision_params, cart_pcoords
        ) ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "emitting_sphere") {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ;
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ;
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        #if 0
        emitting_sphere_m1_id_t id{
            m1_atmo_params, m1_excision_params, cart_pcoords
        } ;
        set_m1_initial_data_impl(id) ;
        #endif
        zero_m1_id_t id{
            m1_atmo_params, m1_excision_params, cart_pcoords
        } ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "zero" or id_type == "coupling_test" ) {
        coord_array_t<GRACE_NSPACEDIM> sph_pcoords ;
        grace::fill_physical_coordinates(sph_pcoords,grace::STAG_CENTER,/*spherical coords*/ true) ;
        zero_m1_id_t id{
            m1_atmo_params, m1_excision_params, sph_pcoords
        } ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "curved_beam" ) {
        auto& coord_system = coordinate_system::get() ;
        ASSERT(coord_system.get_is_cks(), "Curved beam requires cks coordinates!") ;
        coord_array_t<GRACE_NSPACEDIM> sph_pcoords ;
        grace::fill_physical_coordinates(sph_pcoords,grace::STAG_CENTER,/*spherical coords*/ false) ;
        auto& state = variable_list::get().getstate() ;
        curved_beam_m1_id_t id(
            m1_atmo_params, m1_excision_params, sph_pcoords, state,
            grace::get_param<double>("m1","curved_beam_test","z_min"),
            grace::get_param<double>("m1","curved_beam_test","z_max")
        ) ;
        set_m1_initial_data_impl(id) ;
    } else if ( id_type == "equilibrium" ) {
        // Radiation in equilibrium with the hydro ID (works on any matter
        // background, e.g. a TOV star).  The kernel reads kappa/eta from aux,
        // which the set_m1_eas call at the END of this function would fill too
        // late — evaluate the EAS from the hydro state first.
        set_m1_eas<eos_t>() ;
        // The evolved solver's ordinary EAS intentionally excludes pairs.
        // For LTE-blended initial data only, construct the equilibrium pair
        // coefficients; the final set_m1_eas below restores evolved-mode EAS.
        if(m1_is_active() && get_pair_treatment()==pair_treatment_t::evolved) {
            DECLARE_GRID_EXTENTS;
            auto state=variable_list::get().getstate();
            auto aux=variable_list::get().getaux();
            auto coords=coordinate_system::get().get_device_coord_system();
            neutrinos_eas_op<eos_t> op(state,aux);
            op.pair_treatment=pair_treatment_t::equilibrium;
            Kokkos::parallel_for(GRACE_EXECUTION_TAG("ID","pair_equilibrium_rates"),
                Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>>(
                    {VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq},{VEC(4,2,2),1}),
                KOKKOS_LAMBDA(VEC(int const& i,int const& j,int const& k),int const& q) {
                    double xyz[3]; coords.get_physical_coordinates(i,j,k,q,xyz);
                    op(VEC(i,j,k),q,xyz);
                });
        }
        coord_array_t<GRACE_NSPACEDIM> sph_pcoords ;
        grace::fill_physical_coordinates(sph_pcoords,grace::STAG_CENTER,/*spherical coords*/ true) ;
        equil_m1_id_t id(
            m1_atmo_params, m1_excision_params,
            variable_list::get().getaux(),
            variable_list::get().getspacings(),
            sph_pcoords
        ) ;
        set_m1_initial_data_impl(id) ;
    }

    #ifdef GRACE_M1_OPTICAL_DEPTH
    // Seed the eikonal optical depth from the cold-NS fit before the first
    // EAS evaluation (aux RHO_ is filled by the GRMHD ID that ran first).
    {
        auto& state = grace::variable_list::get().getstate() ;
        auto& aux   = grace::variable_list::get().getaux()   ;
        init_m1_optical_depth(state, aux) ;
    }
    #endif

    // now set eas
    set_m1_eas<eos_t>() ;

}

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)        \
template                                \
void set_m1_initial_data<EOS>( );        \
template                                \
void set_m1_eas<EOS>(                    \
      grace::var_array_t&                \
    , grace::staggered_variable_arrays_t&\
    , grace::var_array_t&                \
);                                       \
template                                 \
void set_m1_eas<EOS>()


INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::leptonic_eos_4d_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/

} // namespace grace
