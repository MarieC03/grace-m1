#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <grace/IO/cell_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <string>
#include <utility>
#include <stdexcept>

#include <grace/IO/spherical_surfaces.hh>

TEST_CASE("Spherical-surface 4πr² integration", "[spherical_surface]")
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    Kokkos::fence() ; 

    double r = 1.0;
    std::string name{"pippo"} ; 
    std::array<double,3> c{0,0,0} ; 
    size_t npt = 33 ; 
    // create a spherical surface 
    auto surf = std::make_unique<spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>>(
                spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>(name,r,c,npt)
            ); 
    auto& state = grace::variable_list::get().getstate() ; 
    auto state_h = create_mirror_view(state) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            state_h(i,j,k,0,q) = 1.0  ; 
        }, {false,false,false}, true 
    ) ; 
    Kokkos::deep_copy(state,state_h) ;

    Kokkos::View<double**, grace::default_space> interp    ("test",      2048, 1);
    // Second output buffer for the (empty) aux-variable list — required by
    // the interpolate_on_sphere signature even when aux_idx_h is empty.
    Kokkos::View<double**, grace::default_space> interp_aux("test_aux",   2048, 0);
    interpolate_on_sphere(*surf, std::vector<int>{0}, std::vector<int>{}, interp, interp_aux) ;
    auto iv = Kokkos::create_mirror_view(interp) ; 
    Kokkos::deep_copy(iv,interp) ; 

    auto npoints = surf->intersecting_points_h.size() ; 
    double resL = 0 ; 
    for( int i=0; i<npoints; ++i) {
        auto ip = surf->intersecting_points_h[i] ; 
        double domega = surf->weights_h[ip] ; 
        GRACE_VERBOSE("Iv {} domega {}", iv(i,0), domega) ; 
        resL += domega * iv(i,0) ; 
    }

    double res ; 
    parallel::mpi_allreduce(
        &resL,
        &res,
        1,
        MPI_SUM
    ) ; 


    // 33-point sphere quadrature integrates a constant field to the
    // accumulated FP round-off floor, not bit-exact.  Empirical headroom
    // is O(1e-13) — give it 1e-12.
    REQUIRE( fabs(res - 4*M_PI*r*r) < 1e-12) ;

}

// A quadrature point on a reflection plane is shared with its mirror image, so
// it must be half-weighted per plane for the x2-per-reflection sym-multiplier to
// reconstruct the full sphere.  Without that, a constant integrated over an
// octant overshoots 4*pi by ~4%.  This locks the normalization for every
// symmetry fold (no-sym / z-only / quadrant / octant).
TEST_CASE("Spherical-surface quadrature normalization under reflection symmetry",
          "[spherical_surface]")
{
    using namespace grace ;
    size_t const res    = 33 ;
    size_t const ntheta = res, nphi = 2*res ;

    // Full-sphere integral of unity = sum over the SAMPLED (owned) sub-domain,
    // scaled by the reflection multiplier.  Ownership mirrors the grid: a cell
    // exists where the reflected coordinate is >= 0.
    auto integral_of_unity = [&](bool xs, bool ys, bool zs) {
        auto w = uniform_sampler_t::get_quadrature_weights(1.0, res, xs, ys, zs) ;
        double const mu_min = zs ? 0.0 : -1.0, mu_max = 1.0 ;
        int const sym = (xs?2:1)*(ys?2:1)*(zs?2:1) ;
        double sum = 0.0 ;
        for (size_t iphi = 0; iphi < nphi; ++iphi) {
            double const phi = 2*M_PI/nphi * iphi ;
            for (size_t it = 0; it < ntheta; ++it) {
                double const mu  = mu_max - (mu_max-mu_min)/(ntheta-1)*it ;
                double const sth = std::sqrt(std::max(0.0, 1.0 - mu*mu)) ;
                double const x = sth*std::cos(phi), y = sth*std::sin(phi), z = mu ;
                bool const owned = (!xs || x >= -1e-12)
                                && (!ys || y >= -1e-12)
                                && (!zs || z >= -1e-12) ;
                if (owned) sum += w[iphi*ntheta + it] ;
            }
        }
        return sum * sym ;
    };

    REQUIRE( fabs(integral_of_unity(false,false,false) - 4*M_PI) < 1e-11 ); // full sphere
    REQUIRE( fabs(integral_of_unity(false,false,true ) - 4*M_PI) < 1e-11 ); // z-only (BNS/GW)
    REQUIRE( fabs(integral_of_unity(true ,true ,false) - 4*M_PI) < 1e-11 ); // quadrant (x,y)
    REQUIRE( fabs(integral_of_unity(true ,true ,true ) - 4*M_PI) < 1e-11 ); // octant
}