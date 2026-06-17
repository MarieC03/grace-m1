/**
 * @file eas_optical_depth.cpp
 * @brief Grid kernels for the eikonal neutrino optical-depth solver
 *        (Neilsen et al. 2014): cold-fit seeding (init_m1_optical_depth) and
 *        the once-per-aux interior min-path relaxation sweep
 *        (update_m1_optical_depth).  Declarations in eas_optical_depth.hh.
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE).
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
#include <grace_config.h>

#ifdef GRACE_M1_OPTICAL_DEPTH

#include <grace/physics/eas_optical_depth.hh>

#include <grace/amr/amr_functions.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/physics/m1_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {

namespace {

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

    // Seed each accumulator with the cell's OWN old tau, so the min-path
    // relaxation is tau_new = min(tau_old_self, min_neighbours(tau_j + kbar*ds))
    // (Neilsen 2014 / Cactus frankfurt_m1).  Including the self term makes the
    // sweep MONOTONE (tau can only fall, from the large cold-fit seed toward the
    // true min path).  Omitting it lets tau spuriously INCREASE when all
    // neighbours are larger -> non-monotone -> over-trapping -> unphysical.
    double const kc0 = aux(VEC(i,j,k),m1_kappaa_idx<0>(),q)+aux(VEC(i,j,k),m1_kappas_idx<0>(),q) ;
    double b0 = state(VEC(i,j,k),m1_optd_idx<0>(),q) ;
    #ifdef M1_NU_THREESPECIES
    double const kc1 = aux(VEC(i,j,k),m1_kappaa_idx<1>(),q)+aux(VEC(i,j,k),m1_kappas_idx<1>(),q) ;
    double const kc2 = aux(VEC(i,j,k),m1_kappaa_idx<2>(),q)+aux(VEC(i,j,k),m1_kappas_idx<2>(),q) ;
    double b1 = state(VEC(i,j,k),m1_optd_idx<1>(),q), b2 = state(VEC(i,j,k),m1_optd_idx<2>(),q) ;
    #endif
    #ifdef M1_NU_FIVESPECIES
    double const kc3 = aux(VEC(i,j,k),m1_kappaa_idx<3>(),q)+aux(VEC(i,j,k),m1_kappas_idx<3>(),q) ;
    double const kc4 = aux(VEC(i,j,k),m1_kappaa_idx<4>(),q)+aux(VEC(i,j,k),m1_kappas_idx<4>(),q) ;
    double b3 = state(VEC(i,j,k),m1_optd_idx<3>(),q), b4 = state(VEC(i,j,k),m1_optd_idx<4>(),q) ;
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

        {
            double const kn = aux(VEC(ii,jj,kk),m1_kappaa_idx<0>(),q)+aux(VEC(ii,jj,kk),m1_kappas_idx<0>(),q) ;
            b0 = Kokkos::fmin(b0, 0.5*(kc0+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<0>(),q)) ;
        }
        #ifdef M1_NU_THREESPECIES
        {
            double const kn = aux(VEC(ii,jj,kk),m1_kappaa_idx<1>(),q)+aux(VEC(ii,jj,kk),m1_kappas_idx<1>(),q) ;
            b1 = Kokkos::fmin(b1, 0.5*(kc1+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<1>(),q)) ;
        }
        {
            double const kn = aux(VEC(ii,jj,kk),m1_kappaa_idx<2>(),q)+aux(VEC(ii,jj,kk),m1_kappas_idx<2>(),q) ;
            b2 = Kokkos::fmin(b2, 0.5*(kc2+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<2>(),q)) ;
        }
        #endif
        #ifdef M1_NU_FIVESPECIES
        {
            double const kn = aux(VEC(ii,jj,kk),m1_kappaa_idx<3>(),q)+aux(VEC(ii,jj,kk),m1_kappas_idx<3>(),q) ;
            b3 = Kokkos::fmin(b3, 0.5*(kc3+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<3>(),q)) ;
        }
        {
            double const kn = aux(VEC(ii,jj,kk),m1_kappaa_idx<4>(),q)+aux(VEC(ii,jj,kk),m1_kappas_idx<4>(),q) ;
            b4 = Kokkos::fmin(b4, 0.5*(kc4+kn)*ds + state(VEC(ii,jj,kk),m1_optd_idx<4>(),q)) ;
        }
        #endif
    }

    tau_out[0] = Kokkos::fmax(0.0, b0) ;
    #ifdef M1_NU_THREESPECIES
    tau_out[1] = Kokkos::fmax(0.0, b1) ;
    tau_out[2] = Kokkos::fmax(0.0, b2) ;
    #endif
    #ifdef M1_NU_FIVESPECIES
    tau_out[3] = Kokkos::fmax(0.0, b3) ;
    tau_out[4] = Kokkos::fmax(0.0, b4) ;
    #endif
}

} // anonymous namespace

void init_m1_optical_depth(grace::var_array_t& state, grace::var_array_t& aux)
{
    using namespace grace ;
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;

    parallel_for(GRACE_EXECUTION_TAG("ID","init_m1_optd"), policy,
        KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        double const rho_cgs = aux(VEC(i,j,k),RHO_,q) / nu_constants::RHOGF ;
        double const tau0    = compute_analytic_tau_from_rho_cgs(rho_cgs) ;
        state(VEC(i,j,k), m1_optd_idx<0>(), q) = tau0 ;
        #ifdef M1_NU_THREESPECIES
        state(VEC(i,j,k), m1_optd_idx<1>(), q) = tau0 ;
        state(VEC(i,j,k), m1_optd_idx<2>(), q) = tau0 ;
        #endif
        #ifdef M1_NU_FIVESPECIES
        state(VEC(i,j,k), m1_optd_idx<3>(), q) = tau0 ;
        state(VEC(i,j,k), m1_optd_idx<4>(), q) = tau0 ;
        #endif
    }) ;
}

void update_m1_optical_depth(
    grace::var_array_t const& state_read,
    grace::var_array_t&       state_write,
    grace::var_array_t const& aux )
{
    using namespace grace ;
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;

    auto idx = grace::variable_list::get().getinvspacings() ;

    // Interior only — the stencil reaches one cell into the ghost layer, and
    // the exchanged ghost OPTD (boundary condition) must stay intact.  Clean
    // Jacobi: read tau/metric from state_read (valid ghosts), write the
    // relaxed tau to state_write's interior.
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(ngz,ngz,ngz),0},
               {VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ;

    parallel_for(GRACE_EXECUTION_TAG("EVOL","update_m1_optd"), policy,
        KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        double const dx0 = 1.0/idx(0,q), dx1 = 1.0/idx(1,q), dx2 = 1.0/idx(2,q) ;
        double tau_out[5] = {0,0,0,0,0} ;
        relax_cell(state_read, aux, VEC(i,j,k), q, dx0, dx1, dx2, tau_out) ;

        state_write(VEC(i,j,k), m1_optd_idx<0>(), q) = tau_out[0] ;
        #ifdef M1_NU_THREESPECIES
        state_write(VEC(i,j,k), m1_optd_idx<1>(), q) = tau_out[1] ;
        state_write(VEC(i,j,k), m1_optd_idx<2>(), q) = tau_out[2] ;
        #endif
        #ifdef M1_NU_FIVESPECIES
        state_write(VEC(i,j,k), m1_optd_idx<3>(), q) = tau_out[3] ;
        state_write(VEC(i,j,k), m1_optd_idx<4>(), q) = tau_out[4] ;
        #endif
    }) ;
}

} // namespace grace

#endif /* GRACE_M1_OPTICAL_DEPTH */
