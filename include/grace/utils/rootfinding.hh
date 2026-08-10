/**
 * @file rootfinding.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Host/device 1D root-finding routines (bisection, Newton, Brent) operating on user-supplied callables.
 * @date 2024-06-10
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

#ifndef GRACE_UTILS_ROOTFINDING_HH
#define GRACE_UTILS_ROOTFINDING_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/LU_utils.hh>

namespace utils {

template< typename F >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
bisection(F&& func, double const& a, double const& b, double const& tol)
{
    double xa{a}, xb{b}, xc; 
    double fa{func(a)}, fb{func(b)}, fc ; 
    if ( fa * fb > 0 ) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }
    if ( fa == 0  ) {
        return a ;
    } else if ( fb == 0 ) {
        return b ; 
    }
    do {
        xc = 0.5 * ( xa + xb ) ; 
        fc = func(xc) ; 
        if( fa * fc < 0 ) { 
            fb = fc ; 
            xb = xc ; 
        } else if(fb*fc < 0) {
            fa = fc ; 
            xa = xc ;
        } else if ( fa == 0 ) {
            return xa ; 
        } else if ( fb == 0 ) {
            return xb ; 
        } else if ( fc == 0 ) {
            return xc ; 
        }
    } while( math::abs(xa-xb) > tol ) ; 
    return xc ; 
}


template <typename F>
double GRACE_HOST_DEVICE
brent(F& f, const double &a, const double &b, const double t)

//****************************************************************************80
//
//  Purpose:
//
//    ZERO seeks the root of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    The interval [A,B] must be a change of sign interval for F.
//    That is, F(A) and F(B) must be of opposite signs.  Then
//    assuming that F is continuous implies the existence of at least
//    one value C between A and B for which F(C) = 0.
//
//    The location of the zero is determined to within an accuracy
//    of 6 * MACHEPS * abs ( C ) + 2 * T.
//
//    Thanks to Thomas Secretin for pointing out a transcription error in the
//    setting of the value of P, 11 February 2013.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2013
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the change of sign interval.
//
//    Input, double T, a positive error tolerance.
//
//    Input, func_base& F, the name of a user-supplied c++ functor
//    whose zero is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double ZERO, the estimated value of a zero of
//    the function F.
//
{
  double c;
  double d;
  double e;
  double fa;
  double fb;
  double fc;
  double m;
  double p;
  double q;
  double r;
  double s;
  double sa;
  double sb;
  double tol;
  //
  //  Make local copies of A and B.
  //
  sa = a;
  sb = b;
  fa = f(sa);
  fb = f(sb);

  c = sa;
  fc = fa;
  e = sb - sa;
  d = e;

  constexpr double macheps = std::numeric_limits<double>::epsilon();

  for (;;) {
    if (std::fabs(fc) < std::fabs(fb)) {
      sa = sb;
      sb = c;
      c = sa;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    tol = 2.0 * macheps * std::fabs(sb) + t;
    m = 0.5 * (c - sb);

    if (std::fabs(m) <= tol || fb == 0.0) {
      break;
    }

    if (std::fabs(e) < tol || std::fabs(fa) <= std::fabs(fb)) {
      e = m;
      d = e;
    } else {
      s = fb / fa;

      if (sa == c) {
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (0.0 < p) {
        q = -q;
      } else {
        p = -p;
      }

      s = e;
      e = d;

      if (2.0 * p < 3.0 * m * q - std::fabs(tol * q) &&
          p < std::fabs(0.5 * s * q)) {
        d = p / q;
      } else {
        e = m;
        d = e;
      }
    }
    sa = sb;
    fa = fb;

    if (tol < std::fabs(d)) {
      sb = sb + d;
    } else if (0.0 < m) {
      sb = sb + tol;
    } else {
      sb = sb - tol;
    }

    fb = f(sb);

    if ((0.0 < fb && 0.0 < fc) || (fb <= 0.0 && fc <= 0.0)) {
      c = sa;
      fc = fa;
      e = sb - sa;
      d = e;
    }
  }
  return sb;
}
//****************************************************************************80
//****************************************************************************80

  /** @brief Newton-Raphson root finding algorithm. 
   *  @tparam F  callable object (struct with the suitable operator() defined, a lambda, an std::function object ...)
   *  @tparam DF : callable object (struct with the suitable operator() defined, a lambda, an std::function object ...)
   *  @param x0 : Initial guess
   *  @param a  : Low end of the bracket, feel free to set to a very high negative value for unconstrained newton raphson
   *  @param b  : High end of the bracket, feel free to set to a very high value for unconstrained newton raphson
   *  @param f  : object of class F representing the function 
   *  @param df  : object of class F representing the function's derivative 
   *  @param tol : Tolerance
   *  @param iter: initially set to the max iteration count, upon return contains the number of iterations the code went through 
   *  @return : a double which results from a Newton Rapshon step s.t. xk-1, xk satisfy the stopif criterion or whatever the last computed value is if iter > maxiter 
   *  Removed feature: noexcept (The function never throws. The user is responsible to check for failure by verifying that iter < maxiter.)
   * @details: 
   * Two template parameters are necessary in case when lambdas enter as f and df.
   * In that case, each automatically deduced lambda type is different,
   * and that necessitates F and DF.
   * The callable objects as passed by forward referencing to the std::invoke
   * In this way, we are not restricting ourselves to lvalues and can invoke this function also 
   * on lambdas 
   */ 
  template <typename F>
double KOKKOS_FUNCTION
rootfind_newton_raphson(double sa, double sb,
                        F&& f, unsigned long const maxiter, double tol, int& err)
    {

      constexpr const double macheps = std::numeric_limits<double>::epsilon() ;
      double a{sa}, b{sb} ;       
      double fa,fb,dummy ; 
      f(a,fa,dummy); 
      f(b,fb,dummy); 
      if ( fa == 0 ) {
        err = 0 ; 
        return a ; 
      } else if ( fb == 0 ) {
        err = 0 ; 
        return b; 
      }

      // check the root is bracketed 
      if ( fa * fb > 0 ) { 
        err = 1 ;
        return 0 ; 
      }
      double x = ( fb * a - fa * b ) / ( fb - fa ) ;    
      int iter = 0 ; 
      
      double t, xold, fx, dfx; 
      do {
        xold = x ; 
        f(x,fx,dfx) ; 
        if ( fx * fb > 0 ) {
          fb = fx ; 
          b = x ;
        } else if ( fx * fa > 0 ) {
          fa = fx ; 
          a = x ; 
        }
        x -= fx / dfx ; 
        t = 2. * macheps * fabs(x) + tol;
        if ( fabs(x-xold) < t || fx == 0 ) {
          err = 0 ; 
          return x ; 
        }
        if ( x > b or x < a) {
          x = 0.5 * ( b + a ) ; 
        }
        iter ++ ;
      } while(  iter < maxiter  ) ;
      err = 1 ; 
      return 0 ; 
}

  template <typename F>
double KOKKOS_FUNCTION
rootfind_newton_raphson_unsafe(double x0, F&& f, unsigned long const maxiter, double tol, int& err)
    {

      constexpr const double macheps = std::numeric_limits<double>::epsilon() ;
      double x, fx, dfx ; 
      x = x0; 
      f(x,fx,dfx) ; 
      if ( fx == 0 ) {
        err = 0 ;
        return x0 ; 
      } 
 
      int iter = 0 ; 
      
      double t, xold; 
      do {
        if (fabs(dfx) < 10 * macheps) {
            err = 2;
            return x;
        }

        xold = x ; 
        x -= fx / dfx ; 
        f(x,fx,dfx) ; 
        // Robust convergence check
        t = 2. * macheps * fabs(x) + tol;
        if (fabs(x-xold) < t || fabs(fx) < tol) {
            err = 0;
            return x;
        }
        iter ++ ;
      } while(  iter < maxiter  ) ;
      err = 1 ; 
      return 0 ; 
}
//****************************************************************************
enum nr_err_t {
  SUCCESS=0,
  ERR_ROUNDOFF,
  ERR_SMALLSTEP,
  ERR_STAGNATION
} ; 

template < size_t ND,  typename FT  >
void GRACE_HOST_DEVICE
lnsrch( 
  FT&& func, double (&xold)[ND], double fold, double (&g)[ND], double (&p)[ND], 
  double (&x)[ND], double *f, double stpmax, int *check )
{
    static constexpr double TOLX = 5. * std::numeric_limits<double>::epsilon()  ; 
    static constexpr double ALF = 1e-4 ; 
    *check=0; 
    double sum = 0 ; 
    for( int i=0; i<ND; ++i) sum+=SQR(p[i]) ;
    sum = sqrt(sum) ;  
    if ( sum > stpmax ) {
        // CAP the step at stpmax: p *= stpmax/|p|.  Dividing instead multiplies
        // by |p|/stpmax > 1 and lengthens the very step being limited.
        for( int i=0; i<ND; ++i) p[i] *= stpmax/sum ; 
    }
    double slope = 0 ; 
    for( int i=0; i<ND; ++i) slope += g[i]*p[i] ; 
    if (slope>=0) {
        *check=ERR_ROUNDOFF;
        return ;
    }
    double test=0.;
    for( int i=0; i<ND; ++i) {
        test = fmax(test, fabs(p[i])/fmax(fabs(xold[i]),1.)) ; 
    }
    double alamin = TOLX/test ;
    double alam = 1.0;
    // alam2/f2 carry the PREVIOUS trial step into the cubic branch and must
    // outlive the backtracking loop; declared inside it they reset to 0 each
    // pass, so SQR(alam2)=0 made the cubic inf/NaN and it silently degraded to
    // a flat 0.1 backtrack.
    double alam2 = 0.0, f2 = 0.0 ;
    int nfix = 0; 
    do {
        for( int i=0; i<ND; ++i) x[i] = xold[i] + alam * p[i]; 
        *f = func(x) ; 
        if (*f <= fold + ALF * alam * slope) { 
            return ; // good enough 
        } else if ( alam < alamin ) {
            for ( int i=0; i<ND; ++i) x[i] = xold[i] ; 
            *check = ERR_SMALLSTEP;
            return ;
        } else {
            double tmplam{0} ;
            if ( nfix == 0 ) {
                tmplam = - slope / (2.*(*f-fold-slope)) ; 
            } else {
                double r1 = *f-fold-alam*slope;
                double r2 = f2-fold-alam2*slope;
                double a = (r1/SQR(alam)-r2/SQR(alam2)/(alam-alam2));
                double b = (-alam2*r1/SQR(alam)+alam*r2/SQR(alam2)/(alam-alam2));
                if ( a==0 ) {
                tmplam = -slope/(2.*b) ; 
                } else {
                double disc = SQR(b) - 3. * a * slope ;
                if ( disc < 0. ){ 
                    tmplam = 0.5 * alam ; 
                } else if ( b<=0 ){ 
                    tmplam=(-b+sqrt(disc))/(3.*a) ; 
                } else {
                    tmplam = -slope/(b+sqrt(disc)) ; 
                }
                tmplam = fmin(tmplam,0.5*alam) ; 
                }
            } // not first rodeo
            alam2 = alam ; 
            f2 = *f ; 
            alam = fmax(tmplam,0.1*alam) ; 
            nfix ++ ; 
        }
    } while(true) ; 
}
//****************************************************************************

//****************************************************************************
template< size_t ND, typename FT, typename DFT >
void inline GRACE_HOST_DEVICE
rootfind_nd_newton_raphson(FT&& func, DFT&& dfunc, double (&x)[ND], unsigned long maxiter, double t, int& err)
{
    int iter = 0 ; 
    static constexpr double macheps = std::numeric_limits<double>::epsilon() ; 
    double dx[ND], J[ND][ND], F[ND], g[ND], xold[ND] ; 
    int piv[ND+1] ; 
    double tol ; 
    /* f -> 1/2 F^i F_j */
    auto const fmin = [&] (double (&xL)[ND]) {
        func(xL,F) ; 
        double sum = 0 ;
        for (int i=0; i<ND; ++i) sum += SQR(F[i]);
        return 0.5 * sum ; 
    } ; 

    double f = fmin(x) ; 
    double test = 0.0 ; 
    double xmax = 0; 
    for ( int i=0; i<ND; ++i ) {
      test = fmax(test, fabs(F[i])) ; 
      xmax = fmax(xmax, fabs(x[i])) ; 
    }
    
    tol = 2.0 * macheps * xmax + t;
    if ( test < tol ) {
      err = SUCCESS ; 
      return ; 
    }
    double sum=0. ; 
    for( int i=0; i<ND; ++i ) sum += SQR(x[i]) ; 
    double stpmax = 100 * fmax(sqrt(sum), static_cast<double>(ND)) ; 

    do {
        dfunc(x, F, J) ;
        for( int i=0; i<ND; ++i) {
            sum = 0 ; 
            // compute grad f
            for( int j=0; j<ND; ++j) {
                sum += F[j] * J[j][i] ; 
            }
            g[i] = sum ; 
        }
        for( int i=0; i<ND; ++i) {
            xold[i] = x[i] ; 
        }
        double fold = f ; 
        for( int i=0; i<ND; ++i ) dx[i] = -F[i] ;
        LUPDecompose<ND>(J,1e-15,piv) ; 
        LUPSolve<ND>(J,piv,dx) ; 
        int check ; 
        // this function fills x, f, and F
        lnsrch(fmin,xold,fold,g,dx,x,&f,stpmax,&check);
        if ( check == ERR_ROUNDOFF ) {
            err = ERR_ROUNDOFF ; 
            return ; 
        } else if ( check == ERR_SMALLSTEP ) {
            test = 0. ; 
            double den = fmax(f,0.5*static_cast<double>(ND));
            for( int i=0; i<ND; ++i)
                test = fmax(test, fabs(g[i])*fmax(fabs(x[i]),1.0));
            tol = 2.0 * macheps * den + 100 * t;
            if ( test < tol ) {
                err = SUCCESS ; 
                return ; 
            } else {
#ifdef GRACE_M1_COUNT_IMPLICIT
                { double fr=0.; for(int i=0;i<ND;++i) fr=fmax(fr,fabs(F[i]));
                  double xs=0.; for(int i=0;i<ND;++i) xs=fmax(xs,fabs(x[i]));
                  printf("[SS] %.6e %.6e %.6e\n", fr, xs, fr/fmax(xs,1e-300)) ; }
#endif
                err = ERR_SMALLSTEP ; 
                return ; 
            }
        }
        // test for convergence 
        test = 0. ; 
        double ftest = 0. ; 
        double scale = 0 ;
        for( int i=0; i<ND; ++i){
            test = fmax(test, fabs(x[i]-xold[i])) ; 
            scale = fmax(scale, fabs(x[i])) ; 
            ftest = fmax(ftest, fabs(F[i])) ; 
        }
        tol = 2.0 * macheps * scale + t;
        // Accept a small RESIDUAL as well as a small step, mirroring the entry
        // test above.  `ftest == 0.0` demanded a bit-zero residual and so never
        // fired, leaving the step test as the only reachable exit.
        if ( test < tol || ftest < tol ) {
#ifdef GRACE_M1_COUNT_IMPLICIT
            printf("[IT] %d\n", iter+1) ;
#endif
            err = SUCCESS ; 
            return ; 
        }
        iter++ ; 
    } while( iter<maxiter );
    err = ERR_STAGNATION ; 
    return ; 
}
//****************************************************************************
/** @brief MINPACK/GSL `hybridsj`-style scaled dogleg trust region, device-side.
 *
 * Faithful to the algorithm FIL and THC_M1 use (Powell's Hybrid, GSL
 * gsl_multiroot_fdfsolver_hybridsj; Radice+2022 sec 3.2).  The pieces that
 * matter and that a plain dogleg lacks:
 *   - diagonal scaling D from the Jacobian column norms, monotonically
 *     non-decreasing, so the trust region is |D p| <= delta rather than |p|;
 *   - the dogleg constructed in that scaled metric;
 *   - MINPACK's radius update (shrink to 0.5|Dp| below ratio 0.1, grow to
 *     2|Dp| above 0.5) and its acceptance threshold 1e-4;
 *   - delta0 = factor |D x|, factor = 100 as in GSL.
 * On return x holds the best iterate found, whatever err reports.
 */
template< size_t ND, typename FT, typename DFT >
void inline GRACE_HOST_DEVICE
rootfind_nd_dogleg(FT&& func, DFT&& dfunc, double (&x)[ND],
                   unsigned long maxiter, double t, int& err)
{
    static constexpr double macheps = std::numeric_limits<double>::epsilon() ;
    static constexpr double FACTOR  = 100.0 ;
    static constexpr double ACCEPT  = 1.0e-4 ;
    double F[ND], Fn[ND], J[ND][ND], Jlu[ND][ND], g[ND], gs[ND],
           pN[ND], p[ND], xn[ND], Jp[ND], diag[ND] ;
    int piv[ND+1] ;

    func(x, F) ;
    double fnorm = 0., xmax = 0. ;
    for (size_t i=0; i<ND; ++i) {
        fnorm += SQR(F[i]) ; xmax = fmax(xmax, fabs(x[i])) ;
    }
    fnorm = sqrt(fnorm) ;
    { double fm = 0. ;
      for (size_t i=0; i<ND; ++i) fm = fmax(fm, fabs(F[i])) ;
      if ( fm < 2.0*macheps*xmax + t ) { err = SUCCESS ; return ; } }

    for (size_t i=0; i<ND; ++i) diag[i] = 1.0 ;
    double delta = 0. ;
    bool  have_delta = false ;

    for (unsigned long iter = 0 ; iter < maxiter ; ++iter) {
        dfunc(x, F, J) ;

        // --- scaling: diag_j = max(diag_j, |column j of J|), never decreasing
        for (size_t j=0; j<ND; ++j) {
            double cn = 0. ;
            for (size_t i=0; i<ND; ++i) cn += SQR(J[i][j]) ;
            cn = sqrt(cn) ;
            if ( cn == 0. ) cn = 1.0 ;
            diag[j] = fmax(diag[j], cn) ;
        }
        if ( !have_delta ) {                      // delta0 = FACTOR |D x|
            double dx = 0. ;
            for (size_t i=0; i<ND; ++i) dx += SQR(diag[i]*x[i]) ;
            dx = sqrt(dx) ;
            delta = FACTOR * ( dx > 0. ? dx : 1.0 ) ;
            have_delta = true ;
        }

        // --- gradient  g = J^T F  and its scaled form  gs_j = g_j/diag_j
        for (size_t j=0; j<ND; ++j) {
            double sum = 0. ;
            for (size_t i=0; i<ND; ++i) sum += F[i]*J[i][j] ;
            g[j] = sum ; gs[j] = sum/diag[j] ;
        }
        // --- Gauss-Newton step  pN = -J^{-1} F
        for (size_t i=0; i<ND; ++i)
            for (size_t j=0; j<ND; ++j) Jlu[i][j] = J[i][j] ;
        for (size_t i=0; i<ND; ++i) pN[i] = -F[i] ;
        int const lu_ok = LUPDecompose<ND>(Jlu, 1e-300, piv) ;
        if ( lu_ok ) LUPSolve<ND>(Jlu, piv, pN) ;
        else         for (size_t i=0; i<ND; ++i) pN[i] = 0. ;

        double qnorm = 0. ;                       // |D pN|
        for (size_t i=0; i<ND; ++i) qnorm += SQR(diag[i]*pN[i]) ;
        qnorm = sqrt(qnorm) ;

        // ---------------- MINPACK dogleg in the scaled metric ---------------
        if ( lu_ok && qnorm <= delta ) {
            for (size_t i=0; i<ND; ++i) p[i] = pN[i] ;
        } else {
            double gnorm = 0. ;
            for (size_t i=0; i<ND; ++i) gnorm += SQR(gs[i]) ;
            gnorm = sqrt(gnorm) ;
            if ( gnorm == 0. ) {
                double const sc = ( qnorm > 0. ) ? delta/qnorm : 0. ;
                for (size_t i=0; i<ND; ++i) p[i] = sc*pN[i] ;
            } else {
                // Cauchy point along -D^{-1} gs
                double jg2 = 0. ;
                for (size_t i=0; i<ND; ++i) {
                    double sum = 0. ;
                    for (size_t j=0; j<ND; ++j) sum += J[i][j]*(gs[j]/diag[j]) ;
                    jg2 += SQR(sum) ;
                }
                double const alpha  = ( jg2 > 0. ) ? (gnorm*gnorm)/jg2 : 0. ;
                double const sgnorm = alpha*gnorm ;          // |D pC|
                if ( sgnorm >= delta || !lu_ok ) {           // boundary, steepest descent
                    double const sc = delta/gnorm ;
                    for (size_t i=0; i<ND; ++i) p[i] = -sc*gs[i]/diag[i] ;
                } else {
                    // blend pC -> pN so that |D p| = delta
                    double pC[ND], dvec[ND] ;
                    for (size_t i=0; i<ND; ++i) pC[i] = -alpha*gs[i]/diag[i] ;
                    double dd = 0., cd = 0. ;
                    for (size_t i=0; i<ND; ++i) {
                        dvec[i] = pN[i]-pC[i] ;
                        dd += SQR(diag[i]*dvec[i]) ;
                        cd += (diag[i]*pC[i])*(diag[i]*dvec[i]) ;
                    }
                    double tau = 0. ;
                    if ( dd > 0. ) {
                        double const disc = cd*cd + dd*(delta*delta - sgnorm*sgnorm) ;
                        tau = ( disc > 0. ) ? (-cd + sqrt(disc))/dd : 0. ;
                        tau = fmax(0., fmin(1., tau)) ;
                    }
                    for (size_t i=0; i<ND; ++i) p[i] = pC[i] + tau*dvec[i] ;
                }
            }
        }

        double pnorm = 0. ;                        // |D p|
        for (size_t i=0; i<ND; ++i) pnorm += SQR(diag[i]*p[i]) ;
        pnorm = sqrt(pnorm) ;

        for (size_t i=0; i<ND; ++i) xn[i] = x[i] + p[i] ;
        func(xn, Fn) ;
        double fnnorm = 0. ;
        for (size_t i=0; i<ND; ++i) fnnorm += SQR(Fn[i]) ;
        fnnorm = sqrt(fnnorm) ;

        // MINPACK ratios, expressed relative to |F|
        double ared = -1.0 ;
        if ( fnnorm < fnorm ) ared = 1.0 - SQR(fnnorm/fnorm) ;
        for (size_t i=0; i<ND; ++i) {
            double sum = 0. ;
            for (size_t j=0; j<ND; ++j) sum += J[i][j]*p[j] ;
            Jp[i] = F[i] + sum ;
        }
        double lnorm = 0. ;
        for (size_t i=0; i<ND; ++i) lnorm += SQR(Jp[i]) ;
        lnorm = sqrt(lnorm) ;
        double const prered = ( lnorm < fnorm ) ? 1.0 - SQR(lnorm/fnorm) : 0. ;
        double const ratio  = ( prered > 0. ) ? ared/prered : 0. ;

        // --- MINPACK trust-radius update
        if ( ratio < 0.1 ) {
            delta = 0.5*pnorm ;
        } else if ( ratio >= 0.5 || ared >= prered ) {
            delta = fmax(delta, 2.0*pnorm) ;
        }

        if ( ratio >= ACCEPT ) {                   // accept the step
            for (size_t i=0; i<ND; ++i) { x[i] = xn[i] ; F[i] = Fn[i] ; }
            fnorm = fnnorm ;
            double fm = 0., xm = 0., dxn = 0. ;
            for (size_t i=0; i<ND; ++i) {
                fm  = fmax(fm, fabs(F[i])) ; xm = fmax(xm, fabs(x[i])) ;
                dxn += SQR(diag[i]*x[i]) ;
            }
            dxn = sqrt(dxn) ;
            if ( fm < 2.0*macheps*xm + t ) { err = SUCCESS ; return ; }
            if ( pnorm < t*fmax(dxn,1.0) ) { err = SUCCESS ; return ; }   // xtol
        }
        if ( delta <= macheps*1.0e3*fmax(xmax,1.0) ) { err = ERR_SMALLSTEP ; return ; }
    }
    err = ERR_STAGNATION ;
    return ;
}

//****************************************************************************
// Plain secant. Does not bracket, does not cap, does not line-search.
// Caller is expected to apply downstream physical sanity gates on the
// returned x; non-finite f is the only mid-iteration failure path.
//
// Error codes:
//   0  converged (|dx| < rtol|x| + atol  or  |f| < ftol)
//   1  initial sample non-finite
//   2  zero secant slope (f == fl)
//   3  trial step produced non-finite f
//   5  ran out of iterations
template< typename F >
KOKKOS_INLINE_FUNCTION
double rootfind_secant(F&& func,
                        double x1, double x2,
                        double atol, double rtol, double ftol,
                        size_t maxit, int& err)
{
    err = 0;

    double xl = x1, x = x2;
    double fl = func(xl);
    double f  = func(x);

    if (!Kokkos::isfinite(fl) || !Kokkos::isfinite(f)) {
        err = 1; return x;
    }

    for (size_t j = 0; j < maxit; ++j) {
        if (f == 0.0) return x;

        double df = f - fl;
        if (df == 0.0) { err = 2; return x; }

        double dx    = -f * (x - xl) / df;
        double x_new = x + dx;
        double f_new = func(x_new);

        if (!Kokkos::isfinite(f_new)) { err = 3; return x; }

        xl = x;     fl = f;
        x  = x_new; f  = f_new;

        if (Kokkos::fabs(dx) < rtol * Kokkos::fabs(x) + atol ||
            Kokkos::fabs(f)  < ftol) {
            return x;
        }
    }

    err = 5;
    return x;
}

} /* namespace utils */

#endif /* GRACE_UTILS_ROOTFINDING_HH */