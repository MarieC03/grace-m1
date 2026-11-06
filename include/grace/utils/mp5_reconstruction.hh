/**
 * @file mp5_reconstruction.hh
 * @brief Fifth-order monotonicity-preserving (MP5) reconstruction, Suresh & Huynh (1997).
 *
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */
#include <grace/utils/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>
#include <grace/utils/limiters.hh>
#include <grace/utils/matrix_helpers.tpp>
#ifndef GRACE_UTILS_MP5_RECONSTRUCTION_HH
#define GRACE_UTILS_MP5_RECONSTRUCTION_HH

namespace grace {

/**
 * @brief MP5 reconstruction (Suresh & Huynh 1997, J. Comput. Phys. 136, 83).
 *
 * The fifth-order face value of the five-cell stencil is kept where it is monotone and
 * otherwise limited to the interval spanned by the neighbours, the upwind-extrapolated value
 * and the minmod-limited curvatures (their eqs. 2.12-2.16, alpha = 4).  Unlike a TVD slope
 * limiter it preserves smooth extrema (the mirror plateau at a reflection plane); unlike
 * WENO it stays monotone across a discontinuity (a beam edge keeps |F|/E = 1).  Same stencil
 * and ghost-zone requirement (three cells) as weno_reconstructor_t<5>, same convention:
 * uL and uR are the values at i-1/2 from the left and from the right.
 */
struct mp5_reconstructor_t
{
  private:
    static constexpr double alpha   = 4.0 ;     //!< MP bound; strict monotonicity for CFL <= 1/(1+alpha)
    static constexpr double eps_rel = 1.0e-10 ; //!< smoothness-test tolerance relative to u^2 (scale-free)

    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE sgn( double const x ) { return Kokkos::copysign(1.0, x) ; }

    //! minmod of two arguments; copysign keeps it bit-exactly antisymmetric (as limiters.hh)
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE mm2( double const a, double const b )
    {
        return 0.5 * ( sgn(a) + sgn(b) ) * Kokkos::fmin( Kokkos::fabs(a), Kokkos::fabs(b) ) ;
    }
    //! minmod of four arguments, Suresh & Huynh eq. (2.11)
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE mm4( double const a, double const b, double const c, double const d )
    {
        double const sa = sgn(a) ;
        return 0.125 * ( sa + sgn(b) ) * Kokkos::fabs( ( sa + sgn(c) ) * ( sa + sgn(d) ) )
             * Kokkos::fmin( Kokkos::fmin(Kokkos::fabs(a), Kokkos::fabs(b)), Kokkos::fmin(Kokkos::fabs(c), Kokkos::fabs(d)) ) ;
    }
    /**
     * @brief Face value between u0 and up1, biased towards u0, from the five cells
     *        (um2, um1, u0, up1, up2) ordered in the direction of the face.
     */
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    face( double const um2, double const um1, double const u0, double const up1, double const up2 )
    {
        double const u5  = ( 2.0*um2 - 13.0*um1 + 47.0*u0 + 27.0*up1 - 3.0*up2 ) / 60.0 ;
        double const ump = u0 + mm2( up1 - u0, alpha * ( u0 - um1 ) ) ;
        if ( ( u5 - u0 ) * ( u5 - ump ) <= eps_rel * u0 * u0 ) return u5 ;
        double const dm1  = um2 - 2.0*um1 + u0 ;
        double const d0   = um1 - 2.0*u0  + up1 ;
        double const dp1  = u0  - 2.0*up1 + up2 ;
        double const dM4p = mm4( 4.0*d0 - dp1, 4.0*dp1 - d0, d0, dp1 ) ;   // curvature at the face ahead
        double const dM4m = mm4( 4.0*dm1 - d0, 4.0*d0 - dm1, dm1, d0 ) ;   // curvature at the face behind
        double const uul  = u0 + alpha * ( u0 - um1 ) ;
        double const umd  = 0.5 * ( u0 + up1 ) - 0.5 * dM4p ;
        double const ulc  = u0 + 0.5 * ( u0 - um1 ) + (4.0/3.0) * dM4m ;
        double const umin = Kokkos::fmax( Kokkos::fmin( Kokkos::fmin(u0, up1), umd ), Kokkos::fmin( Kokkos::fmin(u0, uul), ulc ) ) ;
        double const umax = Kokkos::fmin( Kokkos::fmax( Kokkos::fmax(u0, up1), umd ), Kokkos::fmax( Kokkos::fmax(u0, uul), ulc ) ) ;
        return u5 + mm2( umin - u5, umax - u5 ) ;   // median(u5, umin, umax)
    }

  public:
    /**
     * @brief MP5 reconstruction of a cell-centred variable at the face i-1/2.
     * @param u Variable view (cells i-3 .. i+2 along idir are read).
     * @param uL Value at i-1/2 from the left cell i-1.
     * @param uR Value at i-1/2 from the right cell i.
     * @param idir Direction of reconstruction.
     */
    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (
          ViewT& u
        , VEC( int const i
             , int const j
             , int const k)
        , double& uL
        , double& uR
        , int8_t idir )
    {
        auto U = [&](int ii, int jj, int kk) { return u(VEC(ii,jj,kk)) ; } ;
        int const im3=i-3*utils::delta(0,idir), im2=i-2*utils::delta(0,idir), im1=i-1*utils::delta(0,idir) ;
        int const ip1=i+1*utils::delta(0,idir), ip2=i+2*utils::delta(0,idir) ;
        int const jm3=j-3*utils::delta(1,idir), jm2=j-2*utils::delta(1,idir), jm1=j-1*utils::delta(1,idir) ;
        int const jp1=j+1*utils::delta(1,idir), jp2=j+2*utils::delta(1,idir) ;
        #ifdef GRACE_3D
        int const km3=k-3*utils::delta(2,idir), km2=k-2*utils::delta(2,idir), km1=k-1*utils::delta(2,idir) ;
        int const kp1=k+1*utils::delta(2,idir), kp2=k+2*utils::delta(2,idir) ;
        #else
        int const km3=0, km2=0, km1=0, kp1=0, kp2=0 ;
        #endif
        double const um3 = U(im3,jm3,km3), um2 = U(im2,jm2,km2), um1 = U(im1,jm1,km1) ;
        double const u0  = U(i,j,k),       up1 = U(ip1,jp1,kp1), up2 = U(ip2,jp2,kp2) ;
        uR = face( up2, up1, u0,  um1, um2 ) ;   // from cell i, the face lies towards i-1
        uL = face( um3, um2, um1, u0,  up1 ) ;   // from cell i-1, the face lies towards i
    }
} ;

} /* namespace grace */
#endif /* GRACE_UTILS_MP5_RECONSTRUCTION_HH */
