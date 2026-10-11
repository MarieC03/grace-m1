/**
 * @file m1_number_flux.hh
 * @brief Physical flux of the M1 neutrino number density through a face.
 *
 * @date 2026-10-09
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Volume methods
 * to simulate relativistic spacetimes and plasmas
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

#ifndef GRACE_PHYSICS_M1_NUMBER_FLUX_HH
#define GRACE_PHYSICS_M1_NUMBER_FLUX_HH

#include <grace/utils/device.h>

namespace grace {

//! Radice+2021 eq. (28): sqrt(gamma) alpha n (W (v^i - beta^i/alpha) + H^i/J), written with the
//! densitised number N = sqrt(gamma) Gamma n, so sqrt(gamma) must not appear again.
GRACE_HOST_DEVICE inline double
m1_number_flux(double const N, double const alp, double const beta, double const Gamma,
               double const W, double const v, double const H, double const J)
{
    return alp * N/Gamma * ( W * (v - beta/alp) + H/J ) ;
}

} /* namespace grace */

#endif /* GRACE_PHYSICS_M1_NUMBER_FLUX_HH */
