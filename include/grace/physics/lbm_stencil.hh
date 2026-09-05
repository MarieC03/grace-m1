/**
 * @file lbm_stencil.hh
 * @brief Direction quadrature (Lebedev) for the Lattice-Boltzmann radiation
 *        module, resident on device.
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

#ifndef GRACE_PHYSICS_LBM_STENCIL_HH
#define GRACE_PHYSICS_LBM_STENCIL_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_LBM

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/errors/error.hh>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace grace { namespace lbm {

/**
 * @brief The direction stencil: a Lebedev quadrature on the unit sphere.
 * \ingroup lbm
 *
 * `ndir` directions with weights `w_d` (summing to 1) and unit vectors
 * `c_d`, exact for real spherical harmonics up to the table's order.  This is
 * the reference code's `LebedevStencil` reduced to what the fixed-stencil
 * scheme needs: the host-side convex hull / Voronoi machinery only matters for
 * the adaptive (rotated) stencil and stays out until then.  Loaded once from
 * `data/lbm/<GRACE_LBM_STENCIL_NAME>` (see lbm.cpp), copied by value into
 * kernels.
 */
struct stencil_t {
    static constexpr int ndir = GRACE_LBM_NDIR ;

    Kokkos::View<double*>  w ;   //!< quadrature weights, sum to 1
    Kokkos::View<double**> c ;   //!< unit directions, c(d,0..2)

    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE weight(int d) const { return w(d) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cx(int d) const { return c(d,0) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cy(int d) const { return c(d,1) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cz(int d) const { return c(d,2) ; }
} ;

/**
 * @brief Load the compiled-in stencil table into device views.
 *
 * Reads `<dir>/<GRACE_LBM_STENCIL_NAME>` where `<dir>` is `lbm.stencil_dir`
 * or, if that is empty, the source-tree `data/lbm` (GRACE_LBM_DATA_DIR).
 * Errors out if the file's direction count differs from GRACE_LBM_NDIR (the
 * constant the state vector was sized with), if the weights do not sum to 1,
 * or if a direction is not a unit vector.  Header-only so the unit tests,
 * which do not link the physics objects, can load a table too.
 */
inline stencil_t load_stencil(std::string const& dir)
{
    std::string const path = dir + "/" + GRACE_LBM_STENCIL_NAME ;
    std::ifstream in(path) ;
    if ( !in.is_open() ) {
        ERROR("LBM: cannot open stencil table '" << path << "'.  Set lbm.stencil_dir "
              "to the directory holding " << GRACE_LBM_STENCIL_NAME << ".") ;
    }

    // Reference format (3dRadiation stencils/): '#' comment lines, the
    // direction count, then one "w,cx,cy,cz,theta,phi" row per direction.
    std::vector<double> w, cx, cy, cz ;
    long count = -1 ;
    std::string line ;
    while ( std::getline(in, line) ) {
        if ( line.empty() || line[0] == '#' ) continue ;
        if ( count < 0 ) {
            count = std::stol(line) ;
            w.reserve(count) ; cx.reserve(count) ; cy.reserve(count) ; cz.reserve(count) ;
            continue ;
        }
        std::istringstream row(line) ;
        std::string tok ;
        double v[6] ; int ncol = 0 ;
        while ( ncol < 6 && std::getline(row, tok, ',') ) v[ncol++] = std::stod(tok) ;
        if ( ncol < 4 ) ERROR("LBM: malformed row in stencil table '" << path << "': " << line) ;
        w.push_back(v[0]) ; cx.push_back(v[1]) ; cy.push_back(v[2]) ; cz.push_back(v[3]) ;
    }

    if ( count != static_cast<long>(GRACE_LBM_NDIR) || static_cast<long>(w.size()) != count ) {
        ERROR("LBM: stencil table '" << path << "' has " << count << " directions ("
              << w.size() << " rows) but this build was configured for GRACE_LBM_NDIR = "
              << GRACE_LBM_NDIR << " (GRACE_LBM_STENCIL = " << GRACE_LBM_STENCIL_NAME
              << ").  Reconfigure, or point lbm.stencil_dir at the matching table.") ;
    }

    // Quadrature sanity: weights sum to 1, directions are unit vectors.
    double wsum = 0. ;
    for ( long d = 0; d < count; ++d ) {
        wsum += w[d] ;
        double const n2 = cx[d]*cx[d] + cy[d]*cy[d] + cz[d]*cz[d] ;
        if ( std::abs(n2 - 1.0) > 1e-10 )
            ERROR("LBM: direction " << d << " of '" << path << "' is not a unit vector (|c|^2 = " << n2 << ").") ;
    }
    if ( std::abs(wsum - 1.0) > 1e-10 )
        ERROR("LBM: weights of '" << path << "' sum to " << wsum << ", expected 1.") ;

    stencil_t st ;
    st.w = Kokkos::View<double*> ("lbm_w", count) ;
    st.c = Kokkos::View<double**>("lbm_c", count, 3) ;
    auto hw = Kokkos::create_mirror_view(st.w) ;
    auto hc = Kokkos::create_mirror_view(st.c) ;
    for ( long d = 0; d < count; ++d ) {
        hw(d)   = w[d] ;
        hc(d,0) = cx[d] ; hc(d,1) = cy[d] ; hc(d,2) = cz[d] ;
    }
    Kokkos::deep_copy(st.w, hw) ;
    Kokkos::deep_copy(st.c, hc) ;
    return st ;
}

/**
 * @brief The process-wide stencil, loaded on first use.
 */
stencil_t const& get_stencil() ;

}} // namespace grace::lbm

#endif // GRACE_ENABLE_LBM
#endif // GRACE_PHYSICS_LBM_STENCIL_HH
