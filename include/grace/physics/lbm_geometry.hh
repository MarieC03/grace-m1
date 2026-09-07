/**
 * @file lbm_geometry.hh
 * @brief Geometry helpers of the Lattice-Boltzmann radiation module: the
 *        Eulerian triad, null coordinate speed, Killing energy density and
 *        the real spherical-harmonic basis used to fit the geodesic map.
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
#ifndef GRACE_PHYSICS_LBM_GEOMETRY_HH
#define GRACE_PHYSICS_LBM_GEOMETRY_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/metric_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace lbm {

/**
 * @brief Orthonormal triad of the Eulerian observer, e^i_a with
 *        gamma_ij e^i_a e^j_b = delta_ab, and its inverse E^a_i.
 *
 * Built from the Cholesky factor of the spatial metric, gamma = L L^T:
 * E^a_i = L_ia (so gamma_ij = delta_ab E^a_i E^b_j) and e = L^{-T}.  The
 * stencil directions live in this frame -- the reference code's "IF" -- and
 * the triad is the identity, bit-exactly, in flat space.  Lower-triangular
 * like the reference tetrad (Metric.cpp:128-169); any smooth choice works as
 * long as initial data, moments and the geodesic map agree, which they do
 * because all three go through this struct.
 */
struct triad_t {
    double e[3][3] ;   //!< e[i][a] = e^i_a : coordinate components of triad leg a
    double E[3][3] ;   //!< E[a][i] = E^a_i : inverse, E^a_i e^i_b = delta^a_b

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    explicit triad_t(metric_array_t const& m)
    {
        double const gxx = m.gamma(0), gxy = m.gamma(1), gxz = m.gamma(2),
                     gyy = m.gamma(3), gyz = m.gamma(4), gzz = m.gamma(5) ;
        double const L11 = Kokkos::sqrt(gxx) ;
        double const L21 = gxy / L11, L31 = gxz / L11 ;
        double const L22 = Kokkos::sqrt(gyy - L21*L21) ;
        double const L32 = (gyz - L31*L21) / L22 ;
        double const L33 = Kokkos::sqrt(gzz - L31*L31 - L32*L32) ;
        // E^a_i = L_ia
        E[0][0] = L11 ; E[0][1] = L21 ; E[0][2] = L31 ;
        E[1][0] = 0.  ; E[1][1] = L22 ; E[1][2] = L32 ;
        E[2][0] = 0.  ; E[2][1] = 0.  ; E[2][2] = L33 ;
        // e = (L^T)^{-1}, upper triangular
        double const u11 = 1./L11, u22 = 1./L22, u33 = 1./L33 ;
        e[0][0] = u11 ; e[0][1] = -L21/(L11*L22) ; e[0][2] = (L21*L32 - L31*L22)/(L11*L22*L33) ;
        e[1][0] = 0.  ; e[1][1] = u22 ;            e[1][2] = -L32/(L22*L33) ;
        e[2][0] = 0.  ; e[2][1] = 0.  ;            e[2][2] = u33 ;
    }

    //! v^i = e^i_a c^a
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE to_coord(double const c[3], double v[3]) const
    {
        for ( int i = 0; i < 3; ++i ) v[i] = e[i][0]*c[0] + e[i][1]*c[1] + e[i][2]*c[2] ;
    }
    //! c^a = E^a_i v^i
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE to_triad(double const v[3], double c[3]) const
    {
        for ( int a = 0; a < 3; ++a ) c[a] = E[a][0]*v[0] + E[a][1]*v[1] + E[a][2]*v[2] ;
    }
    //! P^ij = e^i_a e^j_b P^ab for a symmetric tensor stored as (xx,xy,xz,yy,yz,zz).
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE to_coord_sym(double const P[6], double out[6]) const
    {
        double const M[3][3] = { {P[0],P[1],P[2]}, {P[1],P[3],P[4]}, {P[2],P[4],P[5]} } ;
        int n = 0 ;
        for ( int i = 0; i < 3; ++i )
        for ( int j = i; j < 3; ++j ) {
            double s = 0. ;
            for ( int a = 0; a < 3; ++a )
            for ( int b = 0; b < 3; ++b ) s += e[i][a]*e[j][b]*M[a][b] ;
            out[n++] = s ;
        }
    }
} ;

//! Largest coordinate speed of a null ray seen by the grid, max_i (alpha sqrt(gamma^ii) + |beta^i|).
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
null_coordinate_speed(metric_array_t const& m)
{
    double const a = m.alp() ;
    double const sx = a*Kokkos::sqrt(m.invgamma(0)) + Kokkos::fabs(m.beta(0)) ;
    double const sy = a*Kokkos::sqrt(m.invgamma(3)) + Kokkos::fabs(m.beta(1)) ;
    double const sz = a*Kokkos::sqrt(m.invgamma(5)) + Kokkos::fabs(m.beta(2)) ;
    return Kokkos::fmax(sx, Kokkos::fmax(sy, sz)) ;
}

/**
 * @brief Killing energy density sqrt(gamma) (alpha E - beta_i F^i) of the
 *        Eulerian moments E, F^i (coordinate components).  Its volume
 *        integral is conserved for free-streaming radiation on a stationary
 *        background; it reduces to sqrt(gamma) E for beta = 0.
 */
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
killing_energy_density(metric_array_t const& m, double const E, double const Fu[3])
{
    auto const bd = m.lower({m.beta(0), m.beta(1), m.beta(2)}) ;
    return m.sqrtg() * (m.alp()*E - (bd[0]*Fu[0] + bd[1]*Fu[1] + bd[2]*Fu[2])) ;
}

//---------------------------------------------------------------------------
// Real spherical harmonics up to l = 2 in Cartesian form, orthonormal on the
// unit sphere with the dOmega measure, indexed i = l(l+1) + m:
//   0 | 1,2,3 = (y, z, x) | 4..8 = (xy, yz, 3z^2-1, xz, x^2-y^2).
// With Lebedev weights summing to 1 the projection is c_i = 4 pi sum_d w_d f_d Y_i(c_d).
//---------------------------------------------------------------------------
constexpr int sh_ncoef = 9 ;

void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
sh_basis(double const x, double const y, double const z, double Y[sh_ncoef])
{
    constexpr double pi = 3.14159265358979323846 ;
    double const c0 = 0.5 / Kokkos::sqrt(pi) ;              // sqrt(1/4pi)
    double const c1 = Kokkos::sqrt(3.0/(4.0*pi)) ;
    double const c2 = 0.5*Kokkos::sqrt(15.0/pi) ;
    double const c3 = 0.25*Kokkos::sqrt(5.0/pi) ;
    double const c4 = 0.25*Kokkos::sqrt(15.0/pi) ;
    Y[0] = c0 ;
    Y[1] = c1*y ; Y[2] = c1*z ; Y[3] = c1*x ;
    Y[4] = c2*x*y ; Y[5] = c2*y*z ; Y[6] = c3*(3.0*z*z - 1.0) ; Y[7] = c2*x*z ; Y[8] = c4*(x*x - y*y) ;
}

//! Quadrature projection of samples f_d at directions c(d,0..2) with weights w_d (sum 1).
template < typename wview_t, typename cview_t >
void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
sh_fit(int const n, wview_t const& w, cview_t const& c, double const* f, double coef[sh_ncoef])
{
    constexpr double fourpi = 4.0*3.14159265358979323846 ;
    for ( int i = 0; i < sh_ncoef; ++i ) coef[i] = 0. ;
    for ( int d = 0; d < n; ++d ) {
        double Y[sh_ncoef] ;
        sh_basis(c(d,0), c(d,1), c(d,2), Y) ;
        double const wf = fourpi * w(d) * f[d] ;
        for ( int i = 0; i < sh_ncoef; ++i ) coef[i] += wf * Y[i] ;
    }
}

double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
sh_eval(double const coef[sh_ncoef], double const Y[sh_ncoef])
{
    double s = 0. ;
    for ( int i = 0; i < sh_ncoef; ++i ) s += coef[i]*Y[i] ;
    return s ;
}

}} // namespace grace::lbm

#endif // GRACE_PHYSICS_LBM_GEOMETRY_HH
