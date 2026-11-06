/**
 * @file test_reconstruction.cpp
 * @brief MP5 reconstruction: exact for quartic cell averages, keeps a smooth extremum that sits on a
 *        face (the mirror plateau of a reflection plane, which TVD slopes clip), monotone across a
 *        step, bit-exactly mirror symmetric and scale-free.  Header only.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/utils/mp5_reconstruction.hh>
#include <grace/utils/reconstruction.hh>
#include <cmath>
#include <vector>
#include <algorithm>

using Catch::Matchers::WithinRel ;
using Catch::Matchers::WithinAbs ;

namespace {
// a 1D line of cell values behind the reconstructor's view interface (direction 0)
struct line_t {
    std::vector<double> v ;
    double operator() ( VEC(int const i, int const, int const) ) const { return v[static_cast<size_t>(i)] ; }
} ;
template< typename recon_t >
std::pair<double,double> at_face( line_t& u, int const i ) {   // values at i-1/2 from the left and from the right
    recon_t r{} ; double uL = 0., uR = 0. ;
    r( u, VEC(i, 0, 0), uL, uR, 0 ) ;
    return { uL, uR } ;
}
// cell average of p over [x-h/2, x+h/2] for a quartic p with coefficients c[0..4]
double quartic( double const x, double const (&c)[5] ) { return c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*x*x*x*x ; }
double quartic_avg( double const x, double const h, double const (&c)[5] ) {
    double const p2 = 2.*c[2] + 6.*c[3]*x + 12.*c[4]*x*x, p4 = 24.*c[4] ;
    return quartic(x, c) + p2 * h*h/24. + p4 * h*h*h*h/1920. ;
}
}

TEST_CASE("MP5 reproduces the face values of quartic cell averages", "[reconstruction][mp5]")
{
    double const c[5] = { 1.0, 2.0, -3.0, 0.5, -0.25 } ; double const h = 0.1 ;
    line_t u ; for ( int i = 0 ; i < 20 ; ++i ) u.v.push_back( quartic_avg( (i + 0.5) * h, h, c ) ) ;
    for ( int i = 4 ; i < 17 ; ++i ) {
        auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, i ) ;
        double const exact = quartic( i * h, c ) ;      // face i-1/2 sits at x = i h
        CHECK_THAT( uL, WithinRel(exact, 1e-12) ) ;
        CHECK_THAT( uR, WithinRel(exact, 1e-12) ) ;
    }
}

TEST_CASE("MP5 keeps a smooth maximum that sits on a face; MC2 clips it", "[reconstruction][mp5]")
{
    // cos(x) sampled as cell averages with the maximum x = 0 exactly on the face between cells 7 and 8
    double const h = 0.1 ; line_t u ;
    for ( int i = 0 ; i < 16 ; ++i ) { double const x = (i - 8 + 0.5) * h ; u.v.push_back( 2. * std::sin(h/2.) / h * std::cos(x) ) ; }
    auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, 8 ) ;
    double const cell = u.v[8] ;                        // = u.v[7], the mirror plateau
    CHECK_THAT( uL, WithinAbs(1.0, 2e-6) ) ;
    CHECK_THAT( uR, WithinAbs(1.0, 2e-6) ) ;
    CHECK( uL > cell ) ; CHECK( uR > cell ) ;           // above the plateau, which a TVD slope cannot give
    auto const [mL, mR] = at_face<grace::slope_limited_reconstructor_t<grace::MCbeta>>( u, 8 ) ;
    CHECK( mL == cell ) ; CHECK( mR == cell ) ;         // MC2 returns the cell value at the plateau
}

TEST_CASE("MP5 is monotone across a step", "[reconstruction][mp5]")
{
    line_t u ; for ( int i = 0 ; i < 16 ; ++i ) u.v.push_back( i < 8 ? 0.0 : 1.0 ) ;
    for ( int i = 4 ; i < 13 ; ++i ) {
        auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, i ) ;
        CHECK( uL >= 0.0 ) ; CHECK( uL <= 1.0 ) ; CHECK( uR >= 0.0 ) ; CHECK( uR <= 1.0 ) ;
        CHECK( uL <= uR + 1e-15 ) ;                     // no inversion at any face
    }
    auto const [jL, jR] = at_face<grace::mp5_reconstructor_t>( u, 8 ) ;   // the jump face
    CHECK( jL == 0.0 ) ; CHECK( jR == 1.0 ) ;
}

TEST_CASE("MP5 is bit-exactly mirror symmetric", "[reconstruction][mp5]")
{
    line_t u ; for ( int i = 0 ; i < 20 ; ++i ) u.v.push_back( std::sin(0.7 * i) + 0.3 * std::cos(2.1 * i * i) + (i == 11 ? 2.0 : 0.0) ) ;
    line_t w ; w.v.assign( u.v.rbegin(), u.v.rend() ) ;   // w[i] = u[19 - i]
    int const n = static_cast<int>( u.v.size() ) ;
    for ( int i = 4 ; i < n - 3 ; ++i ) {
        auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, i ) ;
        auto const [wL, wR] = at_face<grace::mp5_reconstructor_t>( w, n - i ) ;   // the same physical face
        CHECK( wL == uR ) ; CHECK( wR == uL ) ;
    }
}

TEST_CASE("MP5 is scale-free", "[reconstruction][mp5]")
{
    line_t u ; for ( int i = 0 ; i < 20 ; ++i ) u.v.push_back( 1.0 + 0.5 * std::sin(0.9 * i) + (i > 12 ? 3.0 : 0.0) ) ;
    for ( double const scale : { 1.0e-13, 1.0e13 } ) {
        line_t s ; for ( double const x : u.v ) s.v.push_back( scale * x ) ;
        for ( int i = 4 ; i < 17 ; ++i ) {
            auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, i ) ;
            auto const [sL, sR] = at_face<grace::mp5_reconstructor_t>( s, i ) ;
            CHECK_THAT( sL, WithinRel(scale * uL, 1e-12) ) ;
            CHECK_THAT( sR, WithinRel(scale * uR, 1e-12) ) ;
        }
    }
}

TEST_CASE("MP5 reconstructs a constant exactly", "[reconstruction][mp5]")
{
    line_t u ; u.v.assign( 16, 3.7e-9 ) ;
    auto const [uL, uR] = at_face<grace::mp5_reconstructor_t>( u, 8 ) ;
    CHECK( uL == 3.7e-9 ) ; CHECK( uR == 3.7e-9 ) ;
}
