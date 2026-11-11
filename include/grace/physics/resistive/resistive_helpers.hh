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
    double eta0;     //!< Peak Resistivity
    double eta_f;    //!< Lowest resistivity value
    double rho_th;   //!< Critical transition density (see Tenet2026 v2 fig. 4)
    double lambda;   //!< Logaritmic width for the resistivity profile
} ;

/** @brief Get resistive parameters from the parameter store.
 */
GRACE_ALWAYS_INLINE
resistive_params_t get_resistive_params()
{
    resistive_params_t resistive_params;

    resistive_params.eta0   = grace::get_param<double>("resistive", "eta0");
    resistive_params.eta_f  = grace::get_param<double>("resistive", "eta_f");
    resistive_params.rho_th = grace::get_param<double>("resistive", "rho_th");
    resistive_params.lambda = grace::get_param<double>("resistive", "lambda");

    return resistive_params;
}

namespace grace {

/**
 * @brief Device function for resistivity eta following Tenes2026 v2 eq. 46.
 * */
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
compute_eta(double const rho, resistive_params_t const& resistive_params){
    double const eta0   = resistive_params.eta0;
    double const eta_f  = resistive_params.eta_f;
    double const rho_th = resistive_params.rho_th;
    double const lambda = resistive_params.lambda;

    double const eta    = ((eta0 - eta_f) / 2) * (1 + Kokkos::tanh(Kokkos::log10(rho / rho_th) / lambda));

    return eta;
}

}
#endif
