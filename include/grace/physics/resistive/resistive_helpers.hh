/**
 * @file resistive_helpers.hh
 * @author Marcel Ruff (ruff@itp.uni-frankfurt.de)
 * @brief Helpers for resistive.
 * @date 2026-11-10
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
#ifndef GRACE_PHYSICS_RES_HELPERS_HH
#define GRACE_PHYSICS_RES_HELPERS_HH

#include "Kokkos_MathematicalFunctions.hpp"
#include "grace/utils/grace_utils.hh"
#include "grace/utils/inline.h"
#include "grace/utils/math.hh"
#include <grace_config.h>
#include <grace/config/config_parser.hh>
/**
 * @brief Resistive parameters
 *
 * Follows Tenet2026 v2.
 * */
struct resistive_params_t {
    double res_0;     //!< Peak Resistivity
    double res_f;    //!< Lowest resistivity value
    double rho_th;   //!< Critical transition density (see Tenet2026 v2 fig. 4)
    double lambda;   //!< Logaritmic width for the resistivity profile
} ;

/** @brief Get resistive parameters from the parameter store.
 */
GRACE_ALWAYS_INLINE
resistive_params_t get_resistive_params()
{
    resistive_params_t resistive_params;

    resistive_params.res_0   = grace::get_param<double>("resistive", "res_0");
    resistive_params.res_f  = grace::get_param<double>("resistive", "res_f");
    resistive_params.rho_th = grace::get_param<double>("resistive", "rho_th");
    resistive_params.lambda = grace::get_param<double>("resistive", "lambda");

    return resistive_params;
}

namespace grace {

/**
 * @brief Device function for resistivity res following Tenes2026 v2 eq. 46.
 * */
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
compute_res(double const rho, resistive_params_t const& resistive_params){
    double const res_0   = resistive_params.res_0;
    double const res_f  = resistive_params.res_f;
    double const rho_th = resistive_params.rho_th;
    double const lambda = resistive_params.lambda;

    double const res    = ((res_0 - res_f) / 2) * (1 + Kokkos::tanh(Kokkos::log10(rho / rho_th) / lambda));

    return res;
}

/**
 * @brief Device function for second oder averaging onto edges 
 * using 4 cell centred values along direction idir. 
 * Returns following:
 * idir = 0 (i): lower edge along j
 * idir = 1 (j): lower edge along k
 * idir = 2 (k): lower edge along i
 * */
template <typename View>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
edge_interp_2(View const& view,
               VEC(int const i, int const j, int const k),
               int const ivar, int const q, int const idir) {
    int const di = utils::delta(0, idir);
    int const dj = utils::delta(1, idir);
#ifdef GRACE_3D
    int const dk = utils::delta(2, idir);
#elif 
    int constexpr dk = 0;
#endif

    double const view_1 = view(VEC(i , j, k), ivar, q);
    double const view_2 = view(VEC(i - di, j - dj, k - dk), ivar, q);
    double const view_3 = view(VEC(i - di - dj, j - dj - dk, k - dk - di), ivar, q);
    double const view_4 = view(VEC(i - dj, j - dk, k - di), ivar, q);

    return 0.25 * (view_1 + view_2 + view_3 + view_4);
}

}
#endif
