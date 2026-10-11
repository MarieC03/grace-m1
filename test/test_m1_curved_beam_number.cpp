/**
 * @file test_m1_curved_beam_number.cpp
 * @brief Reads the xz plane written by m1_curved_beam_run (Kerr-Schild, M = 1) and checks that
 *        neutrino number and energy travel together along the bent beam.
 *
 * Along a null geodesic the energy at infinity per neutrino is conserved; the normal observer
 * measures E_n = E_inf / (alpha - beta_i l^i), l^i the unit propagation direction (from F).
 * So q = (E/N)(alpha - beta_i l^i) must be the same in every beam cell.  The extra sqrt(gamma)
 * that used to sit in the number flux spread q over 0.11 .. 1.84 (median 0.82); fixed: +-3.5 %.
 */
#include <catch2/catch_test_macros.hpp>

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_dataset(hid_t file, char const* name, hid_t memtype, std::vector<hsize_t>& dims)
{
    hid_t const ds = H5Dopen2(file, name, H5P_DEFAULT) ;
    REQUIRE( ds >= 0 ) ;
    hid_t const sp = H5Dget_space(ds) ;
    int const rank = H5Sget_simple_extent_ndims(sp) ;
    dims.assign(rank, 0) ;
    H5Sget_simple_extent_dims(sp, dims.data(), nullptr) ;
    hsize_t n = 1 ; for ( auto d : dims ) n *= d ;
    std::vector<T> out(n) ;
    REQUIRE( H5Dread(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0 ) ;
    H5Sclose(sp) ; H5Dclose(ds) ;
    return out ;
}

double percentile(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end()) ;
    return v[static_cast<size_t>(std::lround(p / 100. * (v.size() - 1)))] ;
}

} // namespace

TEST_CASE("M1 curved beam: number and energy move together in Kerr-Schild", "[m1][number_flux][integration]")
{
    char const* path = std::getenv("M1_CURVED_BEAM_PLANE") ;
    REQUIRE( path != nullptr ) ;
    hid_t const file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT) ;
    REQUIRE( file >= 0 ) ;
    std::vector<hsize_t> dp, dc, de, dn, df ;
    auto const P = read_dataset<double>(file, "Points", H5T_NATIVE_DOUBLE, dp) ;
    auto const C = read_dataset<uint64_t>(file, "Cells", H5T_NATIVE_UINT64, dc) ;
    auto const E = read_dataset<double>(file, "Erad1", H5T_NATIVE_DOUBLE, de) ;
    auto const N = read_dataset<double>(file, "Nrad1", H5T_NATIVE_DOUBLE, dn) ;
    auto const F = read_dataset<double>(file, "Frad1", H5T_NATIVE_DOUBLE, df) ;
    H5Fclose(file) ;
    size_t const ncell = dc[0], nvert = dc[1] ;
    REQUIRE( E.size() == ncell ) ; REQUIRE( N.size() == ncell ) ; REQUIRE( F.size() == 3 * ncell ) ;

    double const Emax = *std::max_element(E.begin(), E.end()) ;
    std::vector<double> q ;
    for ( size_t c = 0 ; c < ncell ; ++c ) {
        std::array<double,3> x {0., 0., 0.} ;
        for ( size_t v = 0 ; v < nvert ; ++v )
            for ( int d = 0 ; d < 3 ; ++d ) x[d] += P[3 * C[c * nvert + v] + d] / nvert ;
        double const r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) ;
        double const H = 1. / r ;                                   // M/r, M = 1
        std::array<double,3> const l {x[0]/r, x[1]/r, x[2]/r} ;
        // Kerr-Schild: gamma^ij = delta^ij - 2H/(1+2H) l^i l^j, beta_i = 2H l_i, alpha = (1+2H)^-1/2
        double const lF = l[0]*F[3*c] + l[1]*F[3*c+1] + l[2]*F[3*c+2] ;
        std::array<double,3> Fu ;
        for ( int d = 0 ; d < 3 ; ++d ) Fu[d] = F[3*c+d] - 2.*H/(1.+2.*H) * l[d] * lF ;
        double const Fn = std::sqrt(Fu[0]*F[3*c] + Fu[1]*F[3*c+1] + Fu[2]*F[3*c+2]) ;
        if ( !(E[c] > 1e-2 * Emax) || !(Fn > 0.9 * E[c]) ) continue ;   // forward-peaked beam cells
        double const beta_l = 2. * H * (l[0]*Fu[0] + l[1]*Fu[1] + l[2]*Fu[2]) / Fn ;
        q.push_back( E[c] / N[c] * ( 1. / std::sqrt(1. + 2.*H) - beta_l ) ) ;
    }
    INFO( "beam cells: " << q.size() ) ;
    REQUIRE( q.size() >= 50 ) ;
    double const med = percentile(q, 50.), lo = percentile(q, 1.), hi = percentile(q, 99.) ;
    INFO( "q median " << med << ", p1 " << lo << ", p99 " << hi ) ;
    CHECK( lo > 0.94 * med ) ;
    CHECK( hi < 1.06 * med ) ;
}
