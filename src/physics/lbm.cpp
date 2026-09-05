/**
 * @file lbm.cpp
 * @brief Lattice-Boltzmann radiation transport: stencil loader, startup
 *        check, initial data and the per-step driver.
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

#include <grace_config.h>

#ifdef GRACE_ENABLE_LBM

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/execution_tag.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/m1_trigger.hh>
#include <grace/physics/lbm_stencil.hh>
#include <grace/physics/lbm.hh>

#include <Kokkos_Core.hpp>

#include <string>
#include <tuple>

namespace grace { namespace lbm {

//**************************************************************************************************
stencil_t const& get_stencil()
{
    static stencil_t const st = [] {
        std::string dir = grace::get_param<std::string>("lbm","stencil_dir") ;
        if ( dir.empty() ) dir = GRACE_LBM_DATA_DIR ;
        return load_stencil(dir) ;
    }() ;
    return st ;
}

//**************************************************************************************************
namespace {
// Wall-clock split of the LBM sweep, accumulated when lbm.report_timings is on.
struct timings_t { double stream = 0., collide = 0., copy = 0., cells = 0. ; long steps = 0 ; } ;
timings_t g_timings ;
bool timings_enabled()
{
    static bool const on = grace::get_param<bool>("lbm","report_timings") ;
    return on ;
}
}

void report_timings()
{
    auto const& t = g_timings ;
    if ( !timings_enabled() || t.steps == 0 ) return ;
    double const tot = t.stream + t.collide + t.copy ;
    GRACE_INFO("LBM sweep timings over {} steps: stream {:.2f} s ({:.0f}%), collide+moments {:.2f} s "
               "({:.0f}%), register copy {:.2f} s ({:.0f}%); {:.1f} ms per step, {:.2f} Mcells/s, "
               "{:.1f} Mpopulations/s (fenced; excludes ghost exchange and the MoL stepper).",
               t.steps, t.stream, 100*t.stream/tot, t.collide, 100*t.collide/tot, t.copy, 100*t.copy/tot,
               1e3*tot/t.steps, 1e-6*t.cells/tot, 1e-6*t.cells*GRACE_LBM_NDIR*GRACE_LBM_NSPECIES/tot) ;
}

//**************************************************************************************************
void startup_check()
{
    // Reflection symmetries: the populations are registered as plain scalars,
    // but under a reflection I_d must map to the population of the mirrored
    // direction.  Not implemented yet -- refuse rather than run wrong.
    for ( const char* ax : {"x","y","z"} ) {
        if ( grace::get_param<bool>("amr","reflection_symmetries",ax) ) {
            ERROR("LBM: amr.reflection_symmetries." << ax << " is true, but the intensity "
                  "populations have no reflection parity yet (a reflection permutes the "
                  "stencil directions).  Run the full domain.") ;
        }
    }
    // Straight-line streaming is exact in flat space only.
    auto const hydro_id = grace::get_param<std::string>("grmhd","id_type") ;
    if ( hydro_id != "minkowski_vacuum" ) {
        ERROR("LBM: lbm.streaming = flat_fixed pulls populations back along straight "
              "lines, which is only right on a flat background; grmhd.id_type = '"
              << hydro_id << "' is not minkowski_vacuum.  Geodesic streaming is the "
              "next milestone.") ;
    }

    auto const& st = get_stencil() ;   // loads and validates

    size_t nx, ny, nz ;
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto const ngz = grace::amr::get_n_ghosts() ;
    double const cells = double(nx+2*ngz)*double(ny+2*ngz)*double(nz+2*ngz) ;
    double const mb_per_copy = cells * 8.0 * double(GRACE_LBM_NDIR) * double(GRACE_LBM_NSPECIES) / 1048576.0 ;
    GRACE_INFO("LBM radiation transport: stencil {} ({} directions, sum w = 1), {} species, "
               "{} populations per cell; {:.1f} MB per quadrant per state copy "
               "(three copies live for rk3/rk4/imex222).",
               GRACE_LBM_STENCIL_NAME, st.ndir, GRACE_LBM_NSPECIES,
               GRACE_LBM_NDIR*GRACE_LBM_NSPECIES, mb_per_copy) ;
}

//**************************************************************************************************
template < typename eos_t >
void set_initial_data()
{
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;

    auto& state = grace::variable_list::get().getstate() ;
    auto& aux   = grace::variable_list::get().getaux() ;
    system_t sys{ state, state, aux, get_stencil(),
                  grace::variable_list::get().getspacings(), get_params() } ;

    // Full padded extent: the outer ghosts get the isotropic floor, i.e. no
    // incoming radiation, which the streaming pull-back then respects.
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({0,0,0,0},{nx+2*ngz,ny+2*ngz,nz+2*ngz,nq}) ;
    parallel_for(GRACE_EXECUTION_TAG("LBM","initial_data"), policy,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
    {
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, state, q, i,j,k) ;
        double const oosg = 1.0 / metric.sqrtg() ;
        for ( int s = 0; s < nspecies(); ++s ) {
            double const E = Kokkos::fmax(state(i,j,k,erad_idx(s),q) * oosg, sys.p.I_fl) ;
            std::array<double,3> const Fd {
                state(i,j,k,fradx_idx(s)  ,q) * oosg,
                state(i,j,k,fradx_idx(s)+1,q) * oosg,
                state(i,j,k,fradx_idx(s)+2,q) * oosg } ;
            auto const Fu = metric.raise(Fd) ;
            double const Fn = Kokkos::sqrt(Kokkos::fmax(0.0, Fu[0]*Fd[0] + Fu[1]*Fd[1] + Fu[2]*Fd[2])) ;
            double const sigma = sigma_of_relative_flux(Fn / E) ;
            double fh[3] = {0.,0.,1.} ;
            if ( Fn > 1e-300 ) { fh[0] = Fu[0]/Fn ; fh[1] = Fu[1]/Fn ; fh[2] = Fu[2]/Fn ; }
            for ( int d = 0; d < ndir(); ++d ) {
                double const ndotf = sys.st.cx(d)*fh[0] + sys.st.cy(d)*fh[1] + sys.st.cz(d)*fh[2] ;
                state(i,j,k,idx(s,d),q) = Kokkos::fmax(vmf_intensity(sigma, E, ndotf), sys.p.I_fl) ;
            }
        }
        // Moment slots consistent with the quadrature from t = 0.
        sys.write_moments(i,j,k,q) ;
    }) ;
    Kokkos::fence() ;
}

//**************************************************************************************************
namespace {
// Copy the evolved-variable range [lo,hi) of src into dst, all cells.
void copy_block(var_array_t const& src, var_array_t& dst, int lo, int hi)
{
    using Kokkos::ALL ;
    auto s = Kokkos::subview(src, ALL(), ALL(), ALL(), Kokkos::make_pair(lo,hi), ALL()) ;
    auto d = Kokkos::subview(dst, ALL(), ALL(), ALL(), Kokkos::make_pair(lo,hi), ALL()) ;
    Kokkos::deep_copy(d, s) ;
}
}

//**************************************************************************************************
template < typename eos_t >
void step( var_array_t& I_old, var_array_t& I_new, var_array_t& aux, double const dt )
{
    using namespace Kokkos ;
    if ( !grace::m1_is_active() ) return ;   // activation trigger, shared with M1
    DECLARE_GRID_EXTENTS ;

    system_t sys{ I_old, I_new, aux, get_stencil(),
                  grace::variable_list::get().getspacings(), get_params() } ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        interior({ngz,ngz,ngz,0},{nx+ngz,ny+ngz,nz+ngz,nq}) ;

    bool const timed = timings_enabled() ;
    Kokkos::Timer timer ;

    parallel_for(GRACE_EXECUTION_TAG("LBM","stream"), interior,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
    {
        sys.stream(i,j,k,q,dt) ;
    }) ;
    if ( timed ) { Kokkos::fence() ; g_timings.stream += timer.seconds() ; timer.reset() ; }

    parallel_for(GRACE_EXECUTION_TAG("LBM","collide"), interior,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
    {
        int const n = sys.collide(i,j,k,q,dt) ;
        sys.write_moments(i,j,k,q) ;
        aux(i,j,k,LBM_NITER_,q) = static_cast<double>(n) ;
    }) ;
    if ( timed ) { Kokkos::fence() ; g_timings.collide += timer.seconds() ; timer.reset() ; }

    // Register invariant (see lbm.hh step()): both y^n registers hold I^{n+1}
    // and its moments before the stepper starts.
    copy_block(I_new, I_old, LBM_I0_, LBM_IEND_+1) ;
    for ( int s = 0; s < nspecies(); ++s )
        copy_block(I_new, I_old, erad_idx(s), erad_idx(s)+GRACE_N_M1_VARS) ;
    Kokkos::fence() ;
    if ( timed ) {
        g_timings.copy += timer.seconds() ; ++g_timings.steps ;
        g_timings.cells += double(nx)*double(ny)*double(nz)*double(nq) ;
    }
}

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                   \
template void set_initial_data<EOS>() ;                             \
template void step<EOS>( var_array_t&, var_array_t&, var_array_t&, double )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::leptonic_eos_4d_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE

}} // namespace grace::lbm

#endif // GRACE_ENABLE_LBM
