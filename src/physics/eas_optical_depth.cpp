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
        #if GRACE_M1_NU_SPECIES >= 1
        state(VEC(i,j,k), m1_optd_idx<0>(), q) = tau0 ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 3
        state(VEC(i,j,k), m1_optd_idx<1>(), q) = tau0 ;
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
        optd_detail::relax_cell(state_read, aux, VEC(i,j,k), q, dx0, dx1, dx2, tau_out) ;

        #if GRACE_M1_NU_SPECIES >= 1
        state_write(VEC(i,j,k), m1_optd_idx<0>(), q) = tau_out[0] ;
        #endif
        #if GRACE_M1_NU_SPECIES >= 3
        state_write(VEC(i,j,k), m1_optd_idx<1>(), q) = tau_out[1] ;
        #endif
    }) ;
}

} // namespace grace

#endif /* GRACE_M1_OPTICAL_DEPTH */
