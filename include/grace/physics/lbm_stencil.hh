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
#include <grace/physics/lbm_geometry.hh>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

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
    Kokkos::View<int**>    mirror ; //!< mirror(d,a): direction with component a of c_d flipped
    Kokkos::View<double**> Y ;      //!< Y(d,i): real spherical harmonics (l <= 2) at direction d
    int    degree = 0 ;        //!< measured exactness degree L of the quadrature
    double sigma_max = 0. ;    //!< sharpest von Mises-Fisher the quadrature carries (see quadrature_degree)

    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE weight(int d) const { return w(d) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cx(int d) const { return c(d,0) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cy(int d) const { return c(d,1) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE cz(int d) const { return c(d,2) ; }
    int    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE mirror_of(int d, int a) const { return mirror(d,a) ; }
    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE Ysh(int d, int i) const { return Y(d,i) ; }
} ;

/**
 * @brief Exactness degree L of a direction quadrature, measured rather than
 *        assumed: the largest L with sum_d w_d P_l(n_d . a) = 0 for all
 *        1 <= l <= L and every probe axis a.  Probe axes are deliberately
 *        off-symmetry, so the octahedral symmetry of a Lebedev set cannot mask
 *        a failure.
 */
inline int quadrature_degree(std::vector<double> const& w,
                             std::vector<std::array<double,3>> const& c,
                             double const tol = 1e-10, int const lmax = 200)
{
    // Probe axes: irrational directions, none aligned with a symmetry plane.
    std::vector<std::array<double,3>> axes ;
    for ( int k = 1; k <= 5; ++k ) {
        double const a = 0.7137*k, b = 1.3119*k, g = 2.1013*k ;
        double const x = std::cos(a)*std::sin(b), y = std::sin(a)*std::sin(b), z = std::cos(b)*std::cos(g/3) ;
        double const n = std::sqrt(x*x+y*y+z*z) ;
        axes.push_back({x/n, y/n, z/n}) ;
    }
    for ( int l = 1; l <= lmax; ++l ) {
        for ( auto const& a : axes ) {
            double sum = 0. ;
            for ( size_t d = 0; d < w.size(); ++d ) {
                double const x = c[d][0]*a[0] + c[d][1]*a[1] + c[d][2]*a[2] ;
                double p0 = 1., p1 = x ;                 // Legendre by recurrence
                for ( int m = 1; m < l; ++m ) {
                    double const p2 = ((2*m+1)*x*p1 - m*p0)/(m+1) ;
                    p0 = p1 ; p1 = p2 ;
                }
                sum += w[d] * (l == 0 ? 1. : p1) ;
            }
            if ( std::fabs(sum) > tol ) return l - 1 ;
        }
    }
    return lmax ;
}

/**
 * @brief Sharpest von Mises-Fisher distribution a degree-L quadrature carries.
 *
 * exp(sigma n.f) has Legendre content i_l(sigma)/i_0(sigma) ~ exp(-l(l+1)/2sigma),
 * so the first degree the rule cannot integrate, L+1 ~ L, is negligible when
 *   exp(-L(L+1)/(2 sigma)) <= eps   =>   sigma <= L(L+1) / (2 ln(1/eps)).
 * The beam it seeds then has an opening half-angle of about 1/sqrt(sigma), which
 * therefore narrows as 1/L: 7.2 deg for Lebedev29, 4.0 deg for Lebedev53.
 */
inline double vmf_sigma_max(int const degree, double const eps = 1e-3)
{
    if ( degree < 2 ) return 0. ;
    return double(degree)*double(degree+1) / (2.*std::log(1./eps)) ;
}

/**
 * @brief A direction quadrature of runtime size: weights (sum 1) and unit
 *        directions.  Used for the Lebedev-5 "streaming stencil" that samples
 *        the geodesic map for the spherical-harmonic fit (lbm_geometry.hh).
 */
struct quadrature_t {
    int n = 0 ;
    Kokkos::View<double*>  w ;
    Kokkos::View<double**> c ;
} ;

/**
 * @brief Parse a reference-format stencil table ('#' comments, the direction
 *        count, then "w,cx,cy,cz,theta,phi" rows) and validate it: unit
 *        directions, weights summing to 1.
 */
inline quadrature_t load_quadrature(std::string const& path)
{
    std::ifstream in(path) ;
    if ( !in.is_open() ) ERROR("LBM: cannot open stencil table '" << path << "'.") ;
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
    if ( count <= 0 || static_cast<long>(w.size()) != count )
        ERROR("LBM: stencil table '" << path << "' announces " << count << " directions but has " << w.size() << " rows.") ;
    double wsum = 0. ;
    for ( long d = 0; d < count; ++d ) {
        wsum += w[d] ;
        double const n2 = cx[d]*cx[d] + cy[d]*cy[d] + cz[d]*cz[d] ;
        if ( std::abs(n2 - 1.0) > 1e-10 )
            ERROR("LBM: direction " << d << " of '" << path << "' is not a unit vector (|c|^2 = " << n2 << ").") ;
    }
    if ( std::abs(wsum - 1.0) > 1e-10 )
        ERROR("LBM: weights of '" << path << "' sum to " << wsum << ", expected 1.") ;

    quadrature_t qd ;
    qd.n = static_cast<int>(count) ;
    qd.w = Kokkos::View<double*> ("lbm_qw", count) ;
    qd.c = Kokkos::View<double**>("lbm_qc", count, 3) ;
    auto hw = Kokkos::create_mirror_view(qd.w) ;
    auto hc = Kokkos::create_mirror_view(qd.c) ;
    for ( long d = 0; d < count; ++d ) { hw(d) = w[d] ; hc(d,0) = cx[d] ; hc(d,1) = cy[d] ; hc(d,2) = cz[d] ; }
    Kokkos::deep_copy(qd.w, hw) ;
    Kokkos::deep_copy(qd.c, hc) ;
    return qd ;
}

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

    // Mirror table: under a reflection of axis a the population of direction d
    // must be read from the direction with component a flipped.  Lebedev sets
    // are closed under coordinate reflections; verify it rather than assume it.
    std::vector<int> mirror(3*count, -1) ;
    for ( int a = 0; a < 3; ++a ) {
        for ( long d = 0; d < count; ++d ) {
            double const t[3] = { a==0 ? -cx[d] : cx[d], a==1 ? -cy[d] : cy[d], a==2 ? -cz[d] : cz[d] } ;
            long found = -1 ;
            for ( long e = 0; e < count; ++e ) {
                if ( std::abs(cx[e]-t[0]) < 1e-10 && std::abs(cy[e]-t[1]) < 1e-10 && std::abs(cz[e]-t[2]) < 1e-10 ) {
                    if ( found >= 0 ) ERROR("LBM: stencil '" << path << "' has two directions within 1e-10 of each other (" << found << ", " << e << ").") ;
                    found = e ;
                }
            }
            if ( found < 0 )
                ERROR("LBM: stencil '" << path << "' is not closed under the reflection of axis " << a
                      << " (direction " << d << " has no mirror image); reflection symmetries need a mirror-closed stencil.") ;
            if ( std::abs(w[found] - w[d]) > 1e-14 )
                ERROR("LBM: mirror directions " << d << " and " << found << " of '" << path << "' carry different weights.") ;
            mirror[3*d+a] = static_cast<int>(found) ;
        }
        for ( long d = 0; d < count; ++d )
            if ( mirror[3*mirror[3*d+a]+a] != d ) ERROR("LBM: mirror table of axis " << a << " is not an involution.") ;
    }

    stencil_t st ;
    st.w      = Kokkos::View<double*> ("lbm_w", count) ;
    st.c      = Kokkos::View<double**>("lbm_c", count, 3) ;
    st.mirror = Kokkos::View<int**>   ("lbm_mirror", count, 3) ;
    st.Y      = Kokkos::View<double**>("lbm_Y", count, sh_ncoef) ;
    auto hw = Kokkos::create_mirror_view(st.w) ;
    auto hc = Kokkos::create_mirror_view(st.c) ;
    auto hm = Kokkos::create_mirror_view(st.mirror) ;
    auto hY = Kokkos::create_mirror_view(st.Y) ;
    for ( long d = 0; d < count; ++d ) {
        hw(d)   = w[d] ;
        hc(d,0) = cx[d] ; hc(d,1) = cy[d] ; hc(d,2) = cz[d] ;
        hm(d,0) = mirror[3*d] ; hm(d,1) = mirror[3*d+1] ; hm(d,2) = mirror[3*d+2] ;
        double Yd[sh_ncoef] ; sh_basis(cx[d], cy[d], cz[d], Yd) ;
        for ( int i = 0; i < sh_ncoef; ++i ) hY(d,i) = Yd[i] ;
    }
    Kokkos::deep_copy(st.w, hw) ;
    Kokkos::deep_copy(st.c, hc) ;
    Kokkos::deep_copy(st.mirror, hm) ;
    Kokkos::deep_copy(st.Y, hY) ;

    std::vector<std::array<double,3>> cvec(count) ;
    for ( long d = 0; d < count; ++d ) cvec[d] = { cx[d], cy[d], cz[d] } ;
    st.degree    = quadrature_degree(w, cvec) ;
    st.sigma_max = vmf_sigma_max(st.degree) ;
    if ( st.sigma_max <= 0. )
        ERROR("LBM: stencil table '" << path << "' integrates no harmonic beyond l = 1 (measured degree "
              << st.degree << "); it cannot represent a beam.") ;
    return st ;
}

/// Host copy of the mirror table, flat [3*d + a] (for the ghost-exchange setup and tests).
inline std::vector<int> mirror_table_host(stencil_t const& st)
{
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.mirror) ;
    std::vector<int> out(3*st.ndir) ;
    for ( int d = 0; d < st.ndir; ++d ) for ( int a = 0; a < 3; ++a ) out[3*d+a] = hm(d,a) ;
    return out ;
}

/**
 * @brief The process-wide stencil, loaded on first use.
 */
stencil_t const& get_stencil() ;

}} // namespace grace::lbm

#endif // GRACE_ENABLE_LBM
#endif // GRACE_PHYSICS_LBM_STENCIL_HH
