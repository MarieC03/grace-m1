/**
 * @file lbm_geodesic.hh
 * @brief Curved streaming for the Lattice-Boltzmann radiation module: metric
 *        sampling off the grid, the 3+1 null-geodesic equations and their
 *        backward integration over one time step.  Header-only so the unit
 *        tests can drive it without the physics objects.
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
#ifndef GRACE_PHYSICS_LBM_GEODESIC_HH
#define GRACE_PHYSICS_LBM_GEODESIC_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/metric_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/fd_subexpressions.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/lbm_geometry.hh>

#include <Kokkos_Core.hpp>

#include <array>

namespace grace { namespace lbm {

//! FD order of the LBM's own metric derivatives: 4, centred where the stencil
//! fits and one-sided in the two outermost padded layers, so that every padded
//! cell has a geodesic map and the claim weights of the conservative remap are
//! computed identically from both sides of a quadrant face (a truncated ray
//! would pollute the whole spherical-harmonic fit of its cell).
constexpr int LBM_METRIC_DER_ORDER = 4 ;

//! d/dx_a of state slot iv at (i,j,k): 5-point centred, or one-sided at the edges.
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
lbm_d1(var_array_t const& v, int const i, int const j, int const k, int const iv, int64_t const q,
       int const a, double const ooh, int const n)
{
    int const m = (a == 0) ? i : (a == 1) ? j : k ;
    auto f = [&](int const s) { return (a == 0) ? v(i+s,j,k,iv,q) : (a == 1) ? v(i,j+s,k,iv,q) : v(i,j,k+s,iv,q) ; } ;
    if ( m >= 2 && m + 2 < n ) return ooh*( f(-2) - 8.*f(-1) + 8.*f(1) - f(2) )/12. ;
    if ( m < 2 )               return ooh*( -25.*f(0) + 48.*f(1) - 36.*f(2) + 16.*f(3) - 3.*f(4) )/12. ;
    return ooh*( 25.*f(0) - 48.*f(-1) + 36.*f(-2) - 16.*f(-3) + 3.*f(-4) )/12. ;
}

//! Fitted quantities of the geodesic map, in the order of the LBM_SH_ block.
enum sh_quantity_t : int { SH_S = 0, SH_DX = 1, SH_DY = 2, SH_DZ = 3, SH_CX = 4, SH_CY = 5, SH_CZ = 6 } ;
constexpr int sh_nquant = 7 ;

//! Per-cell geometry flag (aux LBM_GEOM_FLAG_).
enum geom_flag_t : int { GEOM_OK = 0, GEOM_EXCISED = 1, GEOM_TRUNCATED = 2 } ;

//! Excision region (grmhd.excision): by coordinate radius or by lapse.
struct excision_t {
    bool   by_radius = false ;
    double r_ex = -1.0, alp_ex = -1.0 ;
    bool GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE inside(double const x[3], double const alpha) const
    {
        return by_radius ? (x[0]*x[0] + x[1]*x[1] + x[2]*x[2] < r_ex*r_ex) : (alpha < alp_ex) ;
    }
} ;

//! Metric, inverse, lapse, shift, their first derivatives and K_ij at one point.
struct metric_sample_t {
    double g[6], ginv[6], alp, beta[3] ;
    double dalp[3] ;      //!< d_a alpha
    double dbeta[9] ;     //!< d_a beta^i at 3a + i
    double dgamma[18] ;   //!< d_a gamma_c at 6a + c, c = xx,xy,xz,yy,yz,zz
    double K[6] ;
} ;

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE constexpr int sym6(int const i, int const j)
{
    // (0,0)=0 (0,1)=1 (0,2)=2 (1,1)=3 (1,2)=4 (2,2)=5
    return (i <= j) ? (i == 0 ? j : (i == 1 ? 2 + j : 5)) : sym6(j, i) ;
}

/**
 * @brief Trilinear sampler of the metric slots (state) and the cell-centred
 *        derivative slots LBM_DALP_.. (aux) of one quadrant.  Under Z4 the ADM
 *        gamma_ij and K_ij are rebuilt from the interpolated Z4c variables
 *        exactly as m1.hh does (gamma = gt/W^2, K = At/W^2 + (Khat+2Theta) gamma/3).
 *
 * `sample(q, i,j,k, disp, s)` evaluates at the point displaced by `disp`
 * (physical units) from the centre of cell (i,j,k); returns false when the
 * 8-point footprint leaves the padded block.
 */
struct metric_field_t {
    var_array_t state, aux ;
    scalar_array_t<GRACE_NSPACEDIM> dx ;
    int nmax[3] ;   //!< padded extents

    bool GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    sample(int64_t const q, int const i, int const j, int const k, double const disp[3], metric_sample_t& s) const
    {
        double const ooh = 1.0 / dx(0,q) ;
        double const u[3] = { i + disp[0]*ooh, j + disp[1]*ooh, k + disp[2]*ooh } ;
        int    i0[3] ; double f[3] ;
        for ( int a = 0; a < 3; ++a ) {
            i0[a] = static_cast<int>(Kokkos::floor(u[a])) ;
            f[a]  = u[a] - i0[a] ;
            if ( i0[a] < 0 || i0[a] + 1 >= nmax[a] ) return false ;   // derivative slots exist on every padded cell
        }
        double const w[8] = {
            (1.-f[0])*(1.-f[1])*(1.-f[2]), (1.-f[0])*(1.-f[1])*f[2], (1.-f[0])*f[1]*(1.-f[2]), (1.-f[0])*f[1]*f[2],
                f[0] *(1.-f[1])*(1.-f[2]),     f[0] *(1.-f[1])*f[2],     f[0] *f[1]*(1.-f[2]),     f[0] *f[1]*f[2] } ;
        auto tri = [&](var_array_t const& v, int const iv) {
            return w[0]*v(i0[0]  ,i0[1]  ,i0[2]  ,iv,q) + w[1]*v(i0[0]  ,i0[1]  ,i0[2]+1,iv,q)
                 + w[2]*v(i0[0]  ,i0[1]+1,i0[2]  ,iv,q) + w[3]*v(i0[0]  ,i0[1]+1,i0[2]+1,iv,q)
                 + w[4]*v(i0[0]+1,i0[1]  ,i0[2]  ,iv,q) + w[5]*v(i0[0]+1,i0[1]  ,i0[2]+1,iv,q)
                 + w[6]*v(i0[0]+1,i0[1]+1,i0[2]  ,iv,q) + w[7]*v(i0[0]+1,i0[1]+1,i0[2]+1,iv,q) ;
        } ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        for ( int c = 0; c < 6; ++c ) s.g[c]    = tri(state, GXX_ + c) ;
        for ( int c = 0; c < 6; ++c ) s.K[c]    = tri(state, KXX_ + c) ;
        #else
        // CHI_ holds W = gamma^{-1/6} (m1.hh, grmhd.hh): gamma = gt/W^2, K = At/W^2 + (Khat + 2 Theta) gamma/3
        double const ooWsqr = 1.0 / SQR(Kokkos::fmax(1e-15, tri(state, CHI_))) ;
        double const Ktr    = tri(state, KHAT_) + 2.0*tri(state, THETA_) ;
        for ( int c = 0; c < 6; ++c ) s.g[c]    = ooWsqr * tri(state, GTXX_ + c) ;
        for ( int c = 0; c < 6; ++c ) s.K[c]    = ooWsqr * tri(state, ATXX_ + c) + Ktr * s.g[c] / 3.0 ;
        #endif
        for ( int a = 0; a < 3; ++a ) s.beta[a] = tri(state, BETAX_ + a) ;
        s.alp = tri(state, ALP_) ;
        for ( int a = 0; a < 3;  ++a ) s.dalp[a]   = tri(aux, LBM_DALP_ + a) ;
        for ( int m = 0; m < 9;  ++m ) s.dbeta[m]  = tri(aux, LBM_DBETA_ + m) ;
        for ( int m = 0; m < 18; ++m ) s.dgamma[m] = tri(aux, LBM_DGAMMA_ + m) ;
        metric_array_t const m{ {s.g[0],s.g[1],s.g[2],s.g[3],s.g[4],s.g[5]}, {s.beta[0],s.beta[1],s.beta[2]}, s.alp } ;
        for ( int c = 0; c < 6; ++c ) s.ginv[c] = m.invgamma(c) ;
        return true ;
    }
} ;

/**
 * @brief Right-hand side of the 3+1 null geodesic equations in coordinate
 *        time (Vincent, Gourgoulhon & Novak 2012; reference
 *        GeodesicEquationSolver.cpp:5-52): x^k the position, v^k the photon
 *        3-velocity measured by the Eulerian observer (coordinate components,
 *        gamma_ij v^i v^j = 1), nu its Eulerian frequency.
 */
void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
geodesic_rhs(metric_sample_t const& s, double const v[3], double const nu,
             double dxdt[3], double dvdt[3], double& dnudt)
{
    double const a = s.alp ;
    double vda = 0. ;
    for ( int i = 0; i < 3; ++i ) vda += v[i]*s.dalp[i] ;
    double Kv[3] = {0.,0.,0.}, vKv = 0. ;
    for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; ++j ) Kv[i] += s.K[sym6(i,j)]*v[j] ;
        vKv += v[i]*Kv[i] ;
    }
    for ( int k = 0; k < 3; ++k ) {
        dxdt[k] = a*v[k] - s.beta[k] ;
        double d = v[k]*vda - a*v[k]*vKv ;
        for ( int i = 0; i < 3; ++i ) {
            d -= s.ginv[sym6(k,i)]*s.dalp[i] ;
            d -= v[i]*s.dbeta[3*i + k] ;
            d += 2.0*a*s.ginv[sym6(i,k)]*Kv[i] ;
        }
        // - (alpha/2) gamma^{kl} (d_i gamma_lj + d_j gamma_il - d_l gamma_ij) v^i v^j
        double chris = 0. ;
        for ( int l = 0; l < 3; ++l ) {
            double t = 0. ;
            for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                t += ( s.dgamma[6*i + sym6(l,j)] + s.dgamma[6*j + sym6(i,l)] - s.dgamma[6*l + sym6(i,j)] ) * v[i]*v[j] ;
            chris += s.ginv[sym6(k,l)]*t ;
        }
        dvdt[k] = d - 0.5*a*chris ;
    }
    dnudt = nu*(-vda + a*vKv) ;
}

//! Result of one backward ray: departure displacement, direction, frequency factor and flag.
struct ray_result_t {
    double disp[3] ;   //!< departure point minus the cell centre (physical units)
    double v[3] ;      //!< Eulerian 3-velocity at the departure point (coordinate components)
    double S ;         //!< nu_arrival / nu_departure = 1/nu (nu_arrival = 1)
    int    flag ;      //!< GEOM_OK or GEOM_TRUNCATED
} ;

/**
 * @brief Integrate the ray arriving at the centre of cell (i,j,k) of quadrant
 *        q with Eulerian velocity v0 backward in coordinate time by dt.
 *
 * Any field type with `sample(q, i,j,k, disp, metric_sample_t&)` works (the
 * grid sampler above, or an analytic metric in the tests).  The metric slice
 * is frozen over the step (the arrival-time one), so backward integration is
 * the forward system with the sign of the right-hand side flipped; on a Z4c
 * metric that is first order in time.  A ray that leaves the sampler's range
 * or enters the excision is frozen and flagged.
 */
template < typename field_t >
ray_result_t GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
integrate_backward(field_t const& fld, int64_t const q, int const i, int const j, int const k,
                   double const xc[3], double const v0[3], double const dt, double const tol,
                   excision_t const& ex)
{
    // Cash-Karp embedded RK4(5), error measured in units of atol + rtol |y|
    // (both = tol), step accepted when that normalised error is below 1.
    constexpr int NV = 7 ;
    constexpr double a21 = 1./5., a31 = 3./40., a32 = 9./40., a41 = 3./10., a42 = -9./10., a43 = 6./5.,
        a51 = -11./54., a52 = 5./2., a53 = -70./27., a54 = 35./27.,
        a61 = 1631./55296., a62 = 175./512., a63 = 575./13824., a64 = 44275./110592., a65 = 253./4096. ;
    constexpr double b5[6] = { 37./378., 0., 250./621., 125./594., 0., 512./1771. } ;
    constexpr double b4[6] = { 2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 1./4. } ;

    double y[NV] = { 0.0, 0.0, 0.0, v0[0], v0[1], v0[2], 1.0 } ;
    int flag = GEOM_OK ;
    // backward in coordinate time on the frozen slice: the forward system with the sign flipped
    auto rhs = [&](double const yy[NV], double d[NV]) {
        for ( int n = 0; n < NV; ++n ) d[n] = 0. ;
        if ( flag != GEOM_OK ) return ;
        metric_sample_t s ;
        double const disp[3] = { yy[0], yy[1], yy[2] } ;
        if ( !fld.sample(q, i,j,k, disp, s) ) { flag = GEOM_TRUNCATED ; return ; }
        double const x[3] = { xc[0] + yy[0], xc[1] + yy[1], xc[2] + yy[2] } ;
        if ( ex.inside(x, s.alp) ) { flag = GEOM_TRUNCATED ; return ; }
        double const v[3] = { yy[3], yy[4], yy[5] } ;
        double dx[3], dv[3], dnu ;
        geodesic_rhs(s, v, yy[6], dx, dv, dnu) ;
        for ( int a = 0; a < 3; ++a ) { d[a] = -dx[a] ; d[3+a] = -dv[a] ; }
        d[6] = -dnu ;
    } ;

    double t = 0., h = dt ;
    double const hmin = 1e-6*dt ;
    for ( int it = 0; it < 100000 && t < dt && flag == GEOM_OK; ++it ) {
        h = Kokkos::fmin(h, dt - t) ;
        double k1[NV], k2[NV], k3[NV], k4[NV], k5[NV], k6[NV], tmp[NV] ;
        rhs(y, k1) ;
        for ( int n = 0; n < NV; ++n ) tmp[n] = y[n] + h*(a21*k1[n]) ;
        rhs(tmp, k2) ;
        for ( int n = 0; n < NV; ++n ) tmp[n] = y[n] + h*(a31*k1[n] + a32*k2[n]) ;
        rhs(tmp, k3) ;
        for ( int n = 0; n < NV; ++n ) tmp[n] = y[n] + h*(a41*k1[n] + a42*k2[n] + a43*k3[n]) ;
        rhs(tmp, k4) ;
        for ( int n = 0; n < NV; ++n ) tmp[n] = y[n] + h*(a51*k1[n] + a52*k2[n] + a53*k3[n] + a54*k4[n]) ;
        rhs(tmp, k5) ;
        for ( int n = 0; n < NV; ++n ) tmp[n] = y[n] + h*(a61*k1[n] + a62*k2[n] + a63*k3[n] + a64*k4[n] + a65*k5[n]) ;
        rhs(tmp, k6) ;
        if ( flag != GEOM_OK ) break ;
        double y5[NV], err = 0. ;
        for ( int n = 0; n < NV; ++n ) {
            double const inc5 = b5[0]*k1[n] + b5[2]*k3[n] + b5[3]*k4[n] + b5[5]*k6[n] ;
            double const inc4 = b4[0]*k1[n] + b4[2]*k3[n] + b4[3]*k4[n] + b4[4]*k5[n] + b4[5]*k6[n] ;
            y5[n] = y[n] + h*inc5 ;
            double const sc = tol*(1.0 + Kokkos::fmax(Kokkos::fabs(y[n]), Kokkos::fabs(y5[n]))) ;
            double const e = h*(inc5 - inc4)/sc ;
            err += e*e ;
        }
        err = Kokkos::sqrt(err/NV) ;
        if ( err <= 1.0 || h <= hmin ) {
            t += h ;
            for ( int n = 0; n < NV; ++n ) y[n] = y5[n] ;
            h *= (err > 1e-30) ? Kokkos::fmin(5.0, 0.9*Kokkos::pow(err, -0.2)) : 5.0 ;
        } else {
            h *= Kokkos::fmax(0.1, 0.9*Kokkos::pow(err, -0.2)) ;
        }
        h = Kokkos::fmax(h, hmin) ;
    }
    ray_result_t r ;
    for ( int a = 0; a < 3; ++a ) { r.disp[a] = y[a] ; r.v[a] = y[3+a] ; }
    r.S = 1.0 / y[6] ;
    r.flag = flag ;
    return r ;
}

/**
 * @brief Cell-centred metric derivatives into the aux slots LBM_DALP_..
 *        LBM_DGAMMA_END_ on every padded cell (lbm_d1), plus sqrt(gamma) and
 *        the shift in the triad, beta_i e^i_a, for the conservative remap.
 *        Non-finite values (excised interior) become 0.
 */
inline void compute_metric_derivatives(var_array_t const& state, var_array_t& aux,
                                       scalar_array_t<GRACE_NSPACEDIM> const& idx,
                                       int const nx, int const ny, int const nz, int const nq)
{
    Kokkos::MDRangePolicy<Kokkos::Rank<4>> pol({0,0,0,0},{nx,ny,nz,nq}) ;
    Kokkos::parallel_for("lbm_metric_derivatives", pol, KOKKOS_LAMBDA (int const i, int const j, int const k, int const q)
    {
        int const n[3] = {nx, ny, nz} ;
        double const ooh = idx(0,q) ;
        double da[3], db[9], dg[18] ;
        for ( int a = 0; a < 3; ++a ) {
            da[a] = lbm_d1(state, i,j,k, ALP_, q, a, ooh, n[a]) ;
            for ( int c = 0; c < 3; ++c ) db[3*a+c] = lbm_d1(state, i,j,k, BETAX_+c, q, a, ooh, n[a]) ;
            #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
            for ( int c = 0; c < 6; ++c ) dg[6*a+c] = lbm_d1(state, i,j,k, GXX_+c, q, a, ooh, n[a]) ;
            #else
            // gamma = gt/W^2 with CHI_ = W: d gamma = d gt / W^2 - 2 gamma dW / W  (m1.hh)
            double const ooW = 1.0 / Kokkos::fmax(1e-15, state(i,j,k,CHI_,q)), ooWsqr = SQR(ooW) ;
            double const dW = lbm_d1(state, i,j,k, CHI_, q, a, ooh, n[a]) ;
            for ( int c = 0; c < 6; ++c )
                dg[6*a+c] = ooWsqr * ( lbm_d1(state, i,j,k, GTXX_+c, q, a, ooh, n[a]) - 2.0 * ooW * dW * state(i,j,k,GTXX_+c,q) ) ;
            #endif
        }
        for ( int a = 0; a < 3;  ++a ) aux(i,j,k,LBM_DALP_+a,q)   = Kokkos::isfinite(da[a]) ? da[a] : 0.0 ;
        for ( int m = 0; m < 9;  ++m ) aux(i,j,k,LBM_DBETA_+m,q)  = Kokkos::isfinite(db[m]) ? db[m] : 0.0 ;
        for ( int m = 0; m < 18; ++m ) aux(i,j,k,LBM_DGAMMA_+m,q) = Kokkos::isfinite(dg[m]) ? dg[m] : 0.0 ;
        metric_array_t m ; FILL_METRIC_ARRAY(m, state, q, i,j,k) ;
        triad_t const tr(m) ;
        auto const bl = m.lower(m._beta) ;
        aux(i,j,k,LBM_SQRTG_,q) = m.sqrtg() ;
        for ( int a = 0; a < 3; ++a ) {
            double const ba = bl[0]*tr.e[0][a] + bl[1]*tr.e[1][a] + bl[2]*tr.e[2][a] ;
            aux(i,j,k,LBM_BTRIAD_+a,q) = Kokkos::isfinite(ba) ? ba : 0.0 ;
        }
    }) ;
    Kokkos::fence() ;
}

}} // namespace grace::lbm

#endif // GRACE_PHYSICS_LBM_GEODESIC_HH
