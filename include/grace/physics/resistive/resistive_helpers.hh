/**
 * @file resistive_helpers.hh
 * @author Marcel Ruff (ruff@itp.uni-frankfurt.de)
 * @brief Helpers for resistive.
 * @date 2024-06-17
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
#endif
