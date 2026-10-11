/**
 * @file tabeos_free_energy.hh
 * @brief Thermodynamically consistent interpolation of a 3D (rho, T, Y_e) table
 *        from the free energy per unit mass f = eps - T s.
 *
 * f is interpolated bicubic-Hermite in (x = ln rho, y = ln T) on each Y_e node and
 * linearly in Y_e.  The node data are f, f_x = P/rho and f_y = -T s (exact) plus f_xy
 * (finite differences).  P = rho f_x, eps = f - f_y and s = -f_y/T all derive from the
 * one C1 interpolant, so de = T ds + (P/rho^2) drho holds between the nodes.
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
#ifndef GRACE_PHYSICS_TABEOS_FREE_ENERGY_HH
#define GRACE_PHYSICS_TABEOS_FREE_ENERGY_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace grace {

struct tabeos_free_energy_t {

    // node data, the derivatives pre-scaled by the axis spacings (dx = d ln rho, dy = d ln T)
    enum FE_VIDX : int { FE_F = 0, FE_FX_DX, FE_FY_DY, FE_FXY_DXDY, N_FE_VARS } ;

    struct derivs_t { double f, fx, fy, fxx, fxy, fyy ; } ;

    tabeos_free_energy_t() = default ;

    tabeos_free_energy_t(
        Kokkos::View<double ****> fe,
        Kokkos::View<double *> ar,
        Kokkos::View<double *> at,
        Kokkos::View<double *> ay,
        double idr_, double idt_, double idy_,
        double baryon_mass_MeV
    ) : _fe(fe), _logrho(ar), _logT(at), _ye(ay)
      , idr(idr_), idt(idt_), idy(idy_), mb_MeV(baryon_mass_MeV)
    {}

    // f and its first and second (x, y) derivatives at (x = ln rho, y = ln T, ye)
    void KOKKOS_INLINE_FUNCTION
    eval(double x, double y, double ye, derivs_t& d) const
    {
        int const i = clamp_index((x - _logrho(0)) * idr, _logrho.extent(0)) ;
        int const j = clamp_index((y - _logT(0))   * idt, _logT.extent(0)) ;
        int const k = clamp_index((ye - _ye(0))    * idy, _ye.extent(0)) ;
        double const u = (x  - _logrho(i)) * idr ;
        double const v = (y  - _logT(j))   * idt ;
        double const w = (ye - _ye(k))     * idy ;
        double hu[4], du[4], ddu[4], hv[4], dv[4], ddv[4] ;
        basis(u, hu, du, ddu) ;
        basis(v, hv, dv, ddv) ;
        // contract the x direction first: A, B, C carry the x-basis h, h', h''
        // and are indexed by the y-basis slot (value / derivative at j, j+1)
        double A[4] = {0.,0.,0.,0.}, B[4] = {0.,0.,0.,0.}, C[4] = {0.,0.,0.,0.} ;
        for ( int a = 0 ; a < 2 ; ++a ) {
            for ( int b = 0 ; b < 2 ; ++b ) {
                double g[N_FE_VARS] ;
                for ( int n = 0 ; n < N_FE_VARS ; ++n )
                    g[n] = (1. - w) * _fe(i+a, j+b, k, n) + w * _fe(i+a, j+b, k+1, n) ;
                A[2*b]   += hu[2*a]  * g[FE_F]    + hu[2*a+1]  * g[FE_FX_DX] ;
                A[2*b+1] += hu[2*a]  * g[FE_FY_DY] + hu[2*a+1]  * g[FE_FXY_DXDY] ;
                B[2*b]   += du[2*a]  * g[FE_F]    + du[2*a+1]  * g[FE_FX_DX] ;
                B[2*b+1] += du[2*a]  * g[FE_FY_DY] + du[2*a+1]  * g[FE_FXY_DXDY] ;
                C[2*b]   += ddu[2*a] * g[FE_F]    + ddu[2*a+1] * g[FE_FX_DX] ;
                C[2*b+1] += ddu[2*a] * g[FE_FY_DY] + ddu[2*a+1] * g[FE_FXY_DXDY] ;
            }
        }
        d.f   = dot(A, hv) ;
        d.fy  = dot(A, dv)  * idt ;
        d.fyy = dot(A, ddv) * idt * idt ;
        d.fx  = dot(B, hv)  * idr ;
        d.fxy = dot(B, dv)  * idr * idt ;
        d.fxx = dot(C, hv)  * idr * idr ;
    }

    // cubic Hermite basis [h00, h10, h01, h11] and its first and second derivatives
    static void KOKKOS_INLINE_FUNCTION
    basis(double u, double* h, double* dh, double* ddh)
    {
        double const u2 = u * u, u3 = u2 * u ;
        h[0]   =  2.*u3 - 3.*u2 + 1. ;  h[1]   = u3 - 2.*u2 + u ;
        h[2]   = -2.*u3 + 3.*u2 ;       h[3]   = u3 - u2 ;
        dh[0]  =  6.*u2 - 6.*u ;        dh[1]  = 3.*u2 - 4.*u + 1. ;
        dh[2]  = -6.*u2 + 6.*u ;        dh[3]  = 3.*u2 - 2.*u ;
        ddh[0] =  12.*u - 6. ;          ddh[1] = 6.*u - 4. ;
        ddh[2] = -12.*u + 6. ;          ddh[3] = 6.*u - 2. ;
    }

    static double KOKKOS_INLINE_FUNCTION
    dot(double const* a, double const* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] ; }

    // same clamp as clamp_table_index in tabulated_eos.hh (kept local to avoid an include cycle)
    static int KOKKOS_INLINE_FUNCTION
    clamp_index(double fidx, size_t npoints) {
        fidx = Kokkos::fmin( Kokkos::fmax(fidx, 0.0), static_cast<double>(npoints - 2) ) ;
        return static_cast<int>(fidx) ;
    }

    Kokkos::View<double ****> _fe ;
    readonly_view_t<double> _logrho, _logT, _ye ;
    double idr{0.}, idt{0.}, idy{0.} ;
    double mb_MeV{0.} ;   //!< baryon mass of the rho axis, converts T s per baryon to per mass
} ;

// In-place sum of two components' f and derivatives (free energies are additive).
KOKKOS_INLINE_FUNCTION void
fe_add(tabeos_free_energy_t::derivs_t& d, tabeos_free_energy_t::derivs_t const& c)
{
    d.f += c.f ; d.fx += c.fx ; d.fy += c.fy ; d.fxx += c.fxx ; d.fxy += c.fxy ; d.fyy += c.fyy ;
}

// ln T with value(ln T) = target at fixed (rho, composition), given vmin < target < vmax at
// logT(0), logT(nT-1).  value(y, v, dv) returns the quantity and its d/d ln T.  Bisects the
// T nodes for the bracketing slab, then safeguarded Newton inside it.
template< typename value_fn_t, typename axis_t >
KOKKOS_INLINE_FUNCTION double
fe_invert_ltemp(value_fn_t const& value, axis_t const& logT, int const nT
               , double const target, double const vmin, double const vmax)
{
    int il = 0, ih = nT - 1 ;
    double vl = vmin, vh = vmax, dum ;
    while ( (ih - il) > 1 ) {
        int const im = (il + ih) / 2 ;
        double vm ;
        value(logT(im), vm, dum) ;
        if ( vm > target ) { ih = im ; vh = vm ; }
        else               { il = im ; vl = vm ; }
    }
    double a = logT(il), b = logT(ih) ;
    double y = a + (target - vl) * (b - a) / (vh - vl) ;
    for ( int it = 0 ; it < 60 ; ++it ) {
        double v, dv ;
        value(y, v, dv) ;
        double const g = v - target ;
        if ( g == 0. ) break ;
        if ( g > 0. ) b = y ; else a = y ;
        double yn = y - g / dv ;
        if ( !(dv > 0.) || !(yn > a && yn < b) ) yn = 0.5 * (a + b) ;
        bool const done = Kokkos::fabs(yn - y) < 1e-14 || (b - a) < 1e-14 ;
        y = yn ;
        if ( done ) break ;
    }
    return y ;
}

// Host-side builder of the node data f, f_x dx, f_y dy, f_xy dx dy, with x = ln rho, y = ln T:
// f = eps - T s/m_b, f_x = P/rho, f_y = -T s/m_b, and f_xy the mean of the two Maxwell estimates
// d(P/rho)/dy and d(-T s/m_b)/dx (2nd-order centred differences, 1st order at the edges).
// eps_of [per mass], p_over_rho_of, entropy_of [k_B/baryon] are callables (i, j, k) on the grid
// logrho x logtemp x axis3; axis3 is stored as given (Ye, or ln Ymu) and must be uniform.
template< typename eps_fn_t, typename por_fn_t, typename ent_fn_t >
tabeos_free_energy_t
build_free_energy_table( std::string const& label
                       , std::vector<double> const& logrho
                       , std::vector<double> const& logtemp
                       , std::vector<double> const& axis3
                       , double const mb_MeV
                       , eps_fn_t const& eps_of
                       , por_fn_t const& p_over_rho_of
                       , ent_fn_t const& entropy_of )
{
    using F = tabeos_free_energy_t::FE_VIDX ;
    int const nr = logrho.size(), nt = logtemp.size(), ny = axis3.size() ;
    double const dx = logrho[1] - logrho[0], dy = logtemp[1] - logtemp[0] ;
    Kokkos::View<double ****> fe(label, nr, nt, ny, F::N_FE_VARS) ;
    auto h = Kokkos::create_mirror_view(fe) ;
    auto const fy = [&] (int i, int j, int k) {
        return -std::exp(logtemp[j]) * entropy_of(i,j,k) / mb_MeV ;
    } ;
    auto const diff = [] (auto const& g, int n, int m, double hh) {
        if ( m == 0 )     return (g(1) - g(0)) / hh ;
        if ( m == n - 1 ) return (g(n-1) - g(n-2)) / hh ;
        return (g(m+1) - g(m-1)) / (2. * hh) ;
    } ;
    for ( int k = 0 ; k < ny ; ++k )
    for ( int j = 0 ; j < nt ; ++j )
    for ( int i = 0 ; i < nr ; ++i ) {
        double const fxy_a = diff([&](int jj){ return p_over_rho_of(i,jj,k) ; }, nt, j, dy) ;
        double const fxy_b = diff([&](int ii){ return fy(ii,j,k) ; }, nr, i, dx) ;
        h(i,j,k,F::FE_F)        = eps_of(i,j,k) + fy(i,j,k) ;
        h(i,j,k,F::FE_FX_DX)    = p_over_rho_of(i,j,k) * dx ;
        h(i,j,k,F::FE_FY_DY)    = fy(i,j,k) * dy ;
        h(i,j,k,F::FE_FXY_DXDY) = 0.5 * (fxy_a + fxy_b) * dx * dy ;
    }
    Kokkos::deep_copy(fe, h) ;
    Kokkos::View<double *> lr(label + "_logrho", nr), lt(label + "_logtemp", nt), a3(label + "_axis3", ny) ;
    grace::deep_copy_vec_to_view(lr, logrho) ;
    grace::deep_copy_vec_to_view(lt, logtemp) ;
    grace::deep_copy_vec_to_view(a3, axis3) ;
    return tabeos_free_energy_t(fe, lr, lt, a3, 1./dx, 1./dy, 1./(axis3[1] - axis3[0]), mb_MeV) ;
}

} /* namespace grace */
#endif /* GRACE_PHYSICS_TABEOS_FREE_ENERGY_HH */
