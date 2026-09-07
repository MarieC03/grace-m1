/**
 * @file lbm.cpp
 * @brief Lattice-Boltzmann radiation transport: stencil and table loaders,
 *        startup checks, background classification, the curved-streaming
 *        geometry pass, initial data and the per-step driver.
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
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/m1_trigger.hh>
#include <grace/physics/lbm_stencil.hh>
#include <grace/physics/lbm_geometry.hh>
#include <grace/physics/lbm_geodesic.hh>
#include <grace/physics/lbm_velocity_mesh.hh>
#include <grace/physics/lbm.hh>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>
#include <tuple>

namespace grace { namespace lbm {

//**************************************************************************************************
namespace {
std::string stencil_dir()
{
    std::string dir = grace::get_param<std::string>("lbm","stencil_dir") ;
    if ( dir.empty() ) dir = GRACE_LBM_DATA_DIR ;
    return dir ;
}
params_t const& params()
{
    static params_t const p = get_params() ;
    return p ;
}
// Compile-time cap on the streaming quadrature.  This sizes seven per-thread
// arrays in the geometry kernel (7*QMAX doubles), which on a GPU is private
// memory and sets the occupancy of that kernel: 16 costs 0.9 kB per thread,
// 64 costs 3.6 kB.  Lebedev5, the default, needs 14.  Raise it here if a
// larger streaming stencil is ever wanted, and re-measure the geometry pass.
constexpr int QMAX = 16 ;
}

stencil_t const& get_stencil()
{
    static stencil_t const st = load_stencil(stencil_dir()) ;
    return st ;
}

//! The low-order quadrature that samples the geodesic map; checked to make the l <= 2 basis orthonormal.
quadrature_t const& get_streaming_quadrature()
{
    static quadrature_t const qd = [] {
        auto q = load_quadrature(stencil_dir() + "/" + grace::get_param<std::string>("lbm","streaming_stencil")) ;
        if ( q.n > QMAX ) ERROR("LBM: streaming stencil has " << q.n << " directions, more than the compiled cap QMAX = "
                                << QMAX << " (src/physics/lbm.cpp; it sizes the geometry kernel's per-thread arrays).") ;
        auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q.w) ;
        auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q.c) ;
        double G[sh_ncoef][sh_ncoef] = {} ;
        for ( int d = 0; d < q.n; ++d ) {
            double Y[sh_ncoef] ; sh_basis(c(d,0), c(d,1), c(d,2), Y) ;
            for ( int i = 0; i < sh_ncoef; ++i ) for ( int j = 0; j < sh_ncoef; ++j ) G[i][j] += 4.0*M_PI*w(d)*Y[i]*Y[j] ;
        }
        for ( int i = 0; i < sh_ncoef; ++i ) for ( int j = 0; j < sh_ncoef; ++j )
            if ( std::abs(G[i][j] - (i==j ? 1.0 : 0.0)) > 1e-10 )
                ERROR("LBM: the streaming stencil does not integrate the l <= 2 harmonics exactly (G[" << i << "][" << j << "] = " << G[i][j] << "); use Lebedev5 or higher.") ;
        return q ;
    }() ;
    return qd ;
}

velocity_interp_t const& get_velocity_interp()
{
    static velocity_interp_t const vi = build_velocity_interp(get_stencil(), params().lut_nth, params().lut_nph) ;
    return vi ;
}

//**************************************************************************************************
namespace {
// Wall-clock split of the LBM sweep, accumulated when lbm.report_timings is on.
struct timings_t { double stream = 0., collide = 0., copy = 0., geometry = 0., cells = 0. ; long steps = 0, passes = 0 ; } ;
timings_t g_timings ;
bool timings_enabled()
{
    static bool const on = grace::get_param<bool>("lbm","report_timings") ;
    return on ;
}
// Curved-streaming geometry state: the aux block is valid for one dt, one grid
// and (Z4) one slice -- see step() for the rebuild policy.
double g_geom_dt    = -1.0 ;
bool   g_geom_valid = false ;
long   g_geom_iter  = -1 ;
int    g_background = -1 ;   // -1 not yet classified, 0 flat, 1 curved
double g_background_dev = 0. ;
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
    if ( t.passes > 0 )
        GRACE_INFO("LBM geometry passes (derivatives, rays, SH fit{}): {} totalling {:.2f} s ({:.1f} ms per pass, {:.1f} ms per step on average).",
                   params().conservative ? ", claim weights" : "", t.passes, t.geometry, 1e3*t.geometry/t.passes, 1e3*t.geometry/t.steps) ;
}

//**************************************************************************************************
void startup_check()
{
    auto const& prm = params() ;
    auto const& st  = get_stencil() ;   // loads and validates (incl. the mirror table)

    size_t nx, ny, nz ;
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto const ngz = grace::amr::get_n_ghosts() ;
    if ( prm.streaming == 1 ) {
        // The pull-back needs one ghost layer, the conservative remap's claim
        // band two more (its maps must be exact there: one-sided derivatives
        // are used only in the two outermost padded layers).
        int const need = prm.conservative ? 4 : 1 ;
        if ( static_cast<int>(ngz) < need )
            ERROR("LBM curved_fixed: amr.n_ghostzones = " << ngz << " but the pull-back band needs at least " << need << ".") ;
        double const cfl = grace::get_param<double>("evolution","cfl_factor") ;
        if ( cfl > 1.0 )
            ERROR("LBM curved_fixed: evolution.cfl_factor = " << cfl << " > 1 would let the geodesic pull-back leave the ghost layer.") ;
        auto const& qd = get_streaming_quadrature() ;
        auto const& vi = get_velocity_interp() ;
        GRACE_INFO("LBM curved streaming ({} remap): {}-direction streaming stencil, {} spherical-harmonic coefficients per quantity, "
                   "velocity mesh of {} triangles with a {}x{} lookup table, geodesic tolerance {:.1e}.",
                   prm.conservative ? "conservative" : "pointwise", qd.n, sh_ncoef, vi.ntri, vi.nth, vi.nph, prm.geodesic_tol) ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
        GRACE_INFO("LBM curved streaming on the Z4c metric: the geodesic map is rebuilt from the current slice "
                   "every {} step(s) (slice frozen over the step, first order in time).", prm.geom_every) ;
        #endif
    }

    double const cells = double(nx+2*ngz)*double(ny+2*ngz)*double(nz+2*ngz) ;
    double const mb_per_copy = cells * 8.0 * double(GRACE_LBM_NDIR) * double(GRACE_LBM_NSPECIES) / 1048576.0 ;
    GRACE_INFO("LBM stencil {}: measured exactness degree {}, so the sharpest beam it carries is "
               "sigma_max = {:.1f} (|F|/E = {:.4f}), an opening half-angle of about {:.1f} degrees.",
               GRACE_LBM_STENCIL_NAME, st.degree, st.sigma_max, flux_of_sigma(st.sigma_max),
               180./M_PI/Kokkos::sqrt(st.sigma_max)) ;
    GRACE_INFO("LBM radiation transport: stencil {} ({} directions, sum w = 1), {} species, "
               "{} populations per cell; {:.1f} MB per quadrant per state copy "
               "(three copies live for rk3/rk4/imex222); streaming = {}.",
               GRACE_LBM_STENCIL_NAME, st.ndir, GRACE_LBM_NSPECIES,
               GRACE_LBM_NDIR*GRACE_LBM_NSPECIES, mb_per_copy, prm.streaming == 1 ? "curved_fixed" : "flat_fixed") ;
}

//**************************************************************************************************
void prepare_background()
{
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;
    auto& state = grace::variable_list::get().getstate() ;
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        interior({ngz,ngz,ngz,0},{nx+ngz,ny+ngz,nz+ngz,nq}) ;
    double dev_local = 0. ;
    parallel_reduce(GRACE_EXECUTION_TAG("LBM","background"), interior,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q, double& dev)
    {
        metric_array_t m ; FILL_METRIC_ARRAY(m, state, q, i,j,k) ;
        double d = Kokkos::fabs(m.alp() - 1.0) ;
        for ( int c = 0; c < 6; ++c ) {
            double const delta = (c==0||c==3||c==5) ? 1.0 : 0.0 ;
            d = Kokkos::fmax(d, Kokkos::fabs(m._g[c] - delta)) ;
        }
        for ( int a = 0; a < 3; ++a ) d = Kokkos::fmax(d, Kokkos::fabs(m._beta[a])) ;
        dev = Kokkos::fmax(dev, d) ;
    }, Max<double>(dev_local)) ;
    double dev = dev_local ;
    parallel::mpi_allreduce(&dev_local, &dev, 1, sc_MPI_MAX) ;
    g_background = dev > 1e-12 ? 1 : 0 ;
    g_background_dev = dev ;
    g_geom_valid = false ;
    // A curved background with flat_fixed is refused at the first step (not here:
    // tests that only build initial data on a curved background must still run).
    GRACE_INFO("LBM background: {} (max deviation from flat {:.3e}), streaming = {}.",
               g_background == 1 ? "curved" : "flat", dev, params().streaming == 1 ? "curved_fixed" : "flat_fixed") ;
}

void on_regrid() { g_geom_valid = false ; }

//**************************************************************************************************
namespace {
/**
 * The geometry pass of curved streaming: cell-centred metric derivatives, then
 * per cell the 14 backward rays of the streaming quadrature and the l <= 2 fit
 * of (S, departure displacement, departure direction) on the current slice.
 * Reused until dt or the grid changes and, under Z4, for
 * lbm.geometry_update_every steps.
 */
void compute_geometry(double const dt)
{
    using namespace Kokkos ;
    Kokkos::Timer timer ;
    DECLARE_GRID_EXTENTS ;
    auto& state = grace::variable_list::get().getstate() ;
    auto& aux   = grace::variable_list::get().getaux() ;
    auto const dx  = grace::variable_list::get().getspacings() ;
    auto const idx = grace::variable_list::get().getinvspacings() ;
    int const NPx = nx + 2*ngz, NPy = ny + 2*ngz, NPz = nz + 2*ngz ;

    compute_metric_derivatives(state, aux, idx, NPx, NPy, NPz, nq) ;

    auto const qd = get_streaming_quadrature() ;
    auto const coords = coordinate_system::get().get_device_coord_system() ;
    metric_field_t const fld{ state, aux, dx, {NPx, NPy, NPz} } ;
    excision_t const ex = params().ex ;
    double const tol = params().geodesic_tol ;

    // The conservative remap claims from a two-cell band of ghost arrivals too
    // (their map is exact there, LBM_METRIC_DER_ORDER); counts are interior only.
    int const band = params().conservative ? 2 : 0 ;
    int const ng = static_cast<int>(ngz), NXi = static_cast<int>(nx), NYi = static_cast<int>(ny), NZi = static_cast<int>(nz) ;
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        region({ng-band,ng-band,ng-band,0},{NXi+ng+band,NYi+ng+band,NZi+ng+band,static_cast<int>(nq)}) ;
    long nexc_l = 0, ntrunc_l = 0 ;
    parallel_reduce(GRACE_EXECUTION_TAG("LBM","geometry"), region,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q, long& nexc, long& ntrunc)
    {
        bool const in = i >= ng && i < NXi+ng && j >= ng && j < NYi+ng && k >= ng && k < NZi+ng ;
        double xc[3] ; coords.get_physical_coordinates(i,j,k,q,xc) ;
        metric_array_t m ; FILL_METRIC_ARRAY(m, state, q, i,j,k) ;
        triad_t const tr(m) ;
        int flag = GEOM_OK ;
        if ( ex.inside(xc, m.alp()) ) { flag = GEOM_EXCISED ; if ( in ) ++nexc ; }
        double fS[QMAX], fD[3][QMAX], fC[3][QMAX] ;
        for ( int d = 0; d < qd.n; ++d ) {
            double const c[3] = { qd.c(d,0), qd.c(d,1), qd.c(d,2) } ;
            double v0[3] ; tr.to_coord(c, v0) ;
            // default: straight ray in coordinates (excised cell, or truncated ray)
            fS[d] = 1.0 ;
            for ( int a = 0; a < 3; ++a ) { fD[a][d] = -v0[a]*dt ; fC[a][d] = c[a] ; }
            if ( flag == GEOM_EXCISED ) continue ;
            auto const r = integrate_backward(fld, q, i,j,k, xc, v0, dt, tol, ex) ;
            if ( r.flag != GEOM_OK ) { flag = GEOM_TRUNCATED ; continue ; }
            metric_sample_t sd ;
            fld.sample(q, i,j,k, r.disp, sd) ;
            metric_array_t const md{ {sd.g[0],sd.g[1],sd.g[2],sd.g[3],sd.g[4],sd.g[5]}, {sd.beta[0],sd.beta[1],sd.beta[2]}, sd.alp } ;
            triad_t const trd(md) ;
            double C[3] ; trd.to_triad(r.v, C) ;
            double const cn = Kokkos::sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]) ;
            fS[d] = r.S ;
            for ( int a = 0; a < 3; ++a ) { fD[a][d] = r.disp[a] ; fC[a][d] = C[a]/cn ; }
        }
        if ( flag == GEOM_TRUNCATED && in ) ++ntrunc ;
        double coef[sh_ncoef] ;
        sh_fit(qd.n, qd.w, qd.c, fS, coef) ;
        for ( int c = 0; c < sh_ncoef; ++c ) aux(i,j,k,LBM_SH_ + sh_ncoef*SH_S + c, q) = coef[c] ;
        for ( int a = 0; a < 3; ++a ) {
            sh_fit(qd.n, qd.w, qd.c, fD[a], coef) ;
            for ( int c = 0; c < sh_ncoef; ++c ) aux(i,j,k,LBM_SH_ + sh_ncoef*(SH_DX+a) + c, q) = coef[c] ;
            sh_fit(qd.n, qd.w, qd.c, fC[a], coef) ;
            for ( int c = 0; c < sh_ncoef; ++c ) aux(i,j,k,LBM_SH_ + sh_ncoef*(SH_CX+a) + c, q) = coef[c] ;
        }
        aux(i,j,k,LBM_GEOM_FLAG_,q) = static_cast<double>(flag) ;
    }, nexc_l, ntrunc_l) ;
    Kokkos::fence() ;
    if ( params().conservative ) {
        // Claim weights of the conservative remap: a property of the map, so
        // computed here (once per background under Cowling, per map update under
        // Z4).  Interior plus a two-cell ghost band, in 27 colours (stride 3 per
        // axis) so that no two concurrent arrivals write the same source bin:
        // deterministic, no atomics.
        system_t sys{ state, state, aux, get_stencil(), dx, get_params() } ;
        sys.coords = coords ;
        sys.vi = get_velocity_interp() ;
        int const NQ = static_cast<int>(nq) ;
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
            padded({0,0,0,0},{NPx,NPy,NPz,NQ}) ;
        parallel_for(GRACE_EXECUTION_TAG("LBM","zero_claims"), padded,
            KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
        {
            for ( int d = 0; d < GRACE_LBM_NDIR; ++d ) aux(i,j,k,LBM_WCLAIM_+d,q) = 0. ;
        }) ;
        int const lo = ng - 2, nb[3] = { NXi + 4, NYi + 4, NZi + 4 } ;
        for ( int col = 0; col < 27; ++col ) {
            int const cx = col % 3, cy = (col/3) % 3, cz = col/9 ;
            MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
                colour({0,0,0,0},{(nb[0]-cx+2)/3, (nb[1]-cy+2)/3, (nb[2]-cz+2)/3, NQ}) ;
            parallel_for(GRACE_EXECUTION_TAG("LBM","claim"), colour,
                KOKKOS_LAMBDA (int const ii, int const jj, int const kk, int const q)
            {
                sys.claim_curved(lo + cx + 3*ii, lo + cy + 3*jj, lo + cz + 3*kk, q) ;
            }) ;
        }
        Kokkos::fence() ;
    }
    long nexc = nexc_l, ntrunc = ntrunc_l ;
    parallel::mpi_allreduce(&nexc_l, &nexc, 1, sc_MPI_SUM) ;
    parallel::mpi_allreduce(&ntrunc_l, &ntrunc, 1, sc_MPI_SUM) ;
    g_geom_dt = dt ; g_geom_valid = true ; g_geom_iter = static_cast<long>(grace::get_iteration()) ;
    g_timings.geometry += timer.seconds() ; ++g_timings.passes ;
    if ( g_timings.passes == 1 )
        GRACE_INFO("LBM geometry pass for dt = {:.6e}: {} rays per cell, {:.2f} s; {} excised cells, {} cells with a truncated ray.",
                   dt, qd.n, timer.seconds(), nexc, ntrunc) ;
    else
        GRACE_VERBOSE("LBM geometry pass {} (iteration {}): {:.2f} s; {} excised cells, {} cells with a truncated ray.",
                      g_timings.passes, g_geom_iter, timer.seconds(), nexc, ntrunc) ;
}
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
        triad_t const tr(metric) ;
        double const oosg = 1.0 / metric.sqrtg() ;
        for ( int s = 0; s < nspecies(); ++s ) {
            double const E = Kokkos::fmax(state(i,j,k,erad_idx(s),q) * oosg, sys.p.I_fl) ;
            std::array<double,3> const Fd {
                state(i,j,k,fradx_idx(s)  ,q) * oosg,
                state(i,j,k,fradx_idx(s)+1,q) * oosg,
                state(i,j,k,fradx_idx(s)+2,q) * oosg } ;
            auto const Fu = metric.raise(Fd) ;
            double const Fn = Kokkos::sqrt(Kokkos::fmax(0.0, Fu[0]*Fd[0] + Fu[1]*Fd[1] + Fu[2]*Fd[2])) ;
            double const sigma = sigma_of_relative_flux(Fn / E, sys.st.sigma_max) ;
            // Beam axis in the triad frame, where the stencil directions live.
            double const Fuc[3] = {Fu[0], Fu[1], Fu[2]} ;
            double Ft[3] ; tr.to_triad(Fuc, Ft) ;
            double fh[3] = {0.,0.,1.} ;
            if ( Fn > 1e-300 ) { fh[0] = Ft[0]/Fn ; fh[1] = Ft[1]/Fn ; fh[2] = Ft[2]/Fn ; }
            for ( int d = 0; d < ndir(); ++d ) {
                double const ndotf = sys.st.cx(d)*fh[0] + sys.st.cy(d)*fh[1] + sys.st.cz(d)*fh[2] ;
                state(i,j,k,idx(s,d),q) = Kokkos::fmax(vmf_intensity(sigma, E, ndotf), sys.p.I_fl) ;
            }
        }
        // Moment slots consistent with the quadrature from t = 0.
        sys.write_moments(i,j,k,q) ;
    }) ;
    Kokkos::fence() ;

    prepare_background() ;
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

    bool const curved = params().streaming == 1 ;
    if ( g_background < 0 ) prepare_background() ;   // restart: no initial-data call
    if ( g_background == 1 && !curved )
        ERROR("LBM: the background is curved (max deviation of gamma_ij, beta^i, alpha from flat = "
              << g_background_dev << ") but lbm.streaming = flat_fixed streams along straight lines.  "
              "Set lbm.streaming: curved_fixed.") ;
    bool rebuild = !g_geom_valid || Kokkos::fabs(dt - g_geom_dt) > 1e-12*dt ;
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    // Dynamical metric: the map of the y^n slice holds for lbm.geometry_update_every steps.
    rebuild = rebuild || static_cast<long>(grace::get_iteration()) - g_geom_iter >= params().geom_every ;
    #endif
    if ( curved && rebuild ) compute_geometry(dt) ;

    system_t sys{ I_old, I_new, aux, get_stencil(),
                  grace::variable_list::get().getspacings(), get_params() } ;
    sys.coords = coordinate_system::get().get_device_coord_system() ;
    if ( curved ) sys.vi = get_velocity_interp() ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        interior({ngz,ngz,ngz,0},{nx+ngz,ny+ngz,nz+ngz,nq}) ;

    bool const timed = timings_enabled() ;
    Kokkos::Timer timer ;

    if ( curved ) {
        parallel_for(GRACE_EXECUTION_TAG("LBM","stream_curved"), interior,
            KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
        {
            sys.stream_curved(i,j,k,q,dt) ;
        }) ;
    } else {
        parallel_for(GRACE_EXECUTION_TAG("LBM","stream"), interior,
            KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
        {
            sys.stream(i,j,k,q,dt) ;
        }) ;
    }
    if ( timed ) { Kokkos::fence() ; g_timings.stream += timer.seconds() ; timer.reset() ; }

    parallel_for(GRACE_EXECUTION_TAG("LBM","collide"), interior,
        KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
    {
        int n = 0 ;
        if ( static_cast<int>(aux(i,j,k,LBM_GEOM_FLAG_,q)) == GEOM_EXCISED ) {
            for ( int s = 0; s < nspecies(); ++s ) for ( int d = 0; d < ndir(); ++d ) sys.I_new(i,j,k,idx(s,d),q) = sys.p.I_fl ;
        } else {
            n = sys.collide(i,j,k,q,dt) ;
        }
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
