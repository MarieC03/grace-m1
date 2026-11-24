/**
 * @file test_resistive_helpers.cpp
 * @author Marcel Ruff (ruff@itp.uni-frankfurt.de)
 * @brief Unit tests for resistive helpers.
 * @date 2026-11-20
 */
#include "Kokkos_Core.hpp"
#include "grace/amr/amr_functions.hh"
#include "grace/coordinates/coordinate_systems.hh"
#include "grace/data_structures/variables.hh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/physics/resistive/resistive_helpers.hh>

namespace {
    // Function to fill the grid (state) with. xyz are physical coords.
    double test_func(double x, double y, double z) {
        return x + y + z;
    }

    //---------------------------------------------------------------------
    // Fill the state View with `f(x,y,z)` inside the valid ghost-zone
    // budget and stomp NaN everywhere outside it. Each axis is treated
    // independently: a cell counts as valid iff all three of its global
    // indices fall in [ngz - valid_gz, n_axis + ngz + valid_gz - 1].
    // ---
    // Code taken from test_grid_interpolator.cpp
    //---------------------------------------------------------------------
    template <typename Func>
    void fill_state_with_polynomial_and_nan_fence(
        int var_slot, int valid_gz, Func const& f
    ) {
        using namespace grace;
        DECLARE_GRID_EXTENTS;
        auto& state = variable_list::get().getstate();
        auto state_h = Kokkos::create_mirror_view(state);

        long const lo_valid_x = static_cast<long>(ngz) - valid_gz;
        long const hi_valid_x = static_cast<long>(nx + ngz) + valid_gz;
        long const lo_valid_y = static_cast<long>(ngz) - valid_gz;
        long const hi_valid_y = static_cast<long>(ny + ngz) + valid_gz;
        long const lo_valid_z = static_cast<long>(ngz) - valid_gz;
        long const hi_valid_z = static_cast<long>(nz + ngz) + valid_gz;

        double const nan = std::nan("");
        auto& coord_system = coordinate_system::get();

        for (size_t q = 0; q < static_cast<size_t>(nq); ++q) {
            for (size_t i = 0; i < nx + 2*ngz; ++i)
            for (size_t j = 0; j < ny + 2*ngz; ++j)
            for (size_t k = 0; k < nz + 2*ngz; ++k) {
                bool const valid =
                    static_cast<long>(i) >= lo_valid_x && static_cast<long>(i) < hi_valid_x &&
                    static_cast<long>(j) >= lo_valid_y && static_cast<long>(j) < hi_valid_y &&
                    static_cast<long>(k) >= lo_valid_z && static_cast<long>(k) < hi_valid_z;
                if (valid) {
                    auto pc = coord_system.get_physical_coordinates(
                        {i, j, k}, q, {0.5, 0.5, 0.5}, /*use_ghostzones=*/true
                    );
                    state_h(i, j, k, var_slot, q) = f(pc[0], pc[1], pc[2]);
                } else {
                    state_h(i, j, k, var_slot, q) = nan;
                }
            }
        }
        Kokkos::deep_copy(state, state_h);
    }

}
TEST_CASE("edge_interp_2 function","[resistive][grid_interpolator]") {
   /*
    template <typename View>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double
    edge_interp_2(View const& view,
                   VEC(int const i, int const j, int const k),
                   int const ivar, int const q, int const idir)
    */

    using namespace grace;
    DECLARE_GRID_EXTENTS;
    REQUIRE(nq > 0);

    // Fetch states and set const vars.
    auto& state = variable_list::get().getstate();
    constexpr int q = 0, ivar = 0;

    // Here valid_gz = ngz.
    fill_state_with_polynomial_and_nan_fence(ivar, ngz, test_func);

    // Interpolate edges at the physical domain corner.
    const unsigned long i = ngz, j = ngz, k = ngz;
    constexpr int idir_y = 0; // 0 => interp lower edge along y
    constexpr int idir_x = 2; // 2 => interp lower edge along x
    constexpr int idir_z = 1; // 1 => interp lower edge along z

    // Set host state
    auto state_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state);

    // Set interp'd edges
    const double edge_y = edge_interp_2(state_h, i, j, k, ivar, q, idir_y);
    const double edge_z = edge_interp_2(state_h, i, j, k, ivar, q, idir_z);
    const double edge_x = edge_interp_2(state_h, i, j, k, ivar, q, idir_x);

    // Calculate expected (_c) edge values.
    // Along y
    const auto pc_edge_y = get_physical_coordinates(
            {i, j, k}, q, 
            {0.0, 0.5, 0.0} // coord at y edge
            , /*include_gzs=*/true);
    const auto pc_edge_z = get_physical_coordinates(
            {i, j, k}, q, 
            {0.0, 0.0, 0.5} // cord at z edge
            , /*include_gzs=*/true);
    const auto pc_edge_x = get_physical_coordinates(
            {i, j, k}, q, 
            {0.5, 0.0, 0.0} // coord at x edge
            , /*include_gzs=*/true
            );
    const double edge_y_c = test_func(pc_edge_y[0], pc_edge_y[1], pc_edge_y[2]);
    const double edge_z_c = test_func(pc_edge_z[0], pc_edge_z[1], pc_edge_z[2]);
    const double edge_x_c = test_func(pc_edge_x[0], pc_edge_x[1], pc_edge_x[2]);

    // Compare:
    //           actual                            expected
    REQUIRE_THAT(edge_y,Catch::Matchers::WithinAbs(edge_y_c, 1e-6));
    REQUIRE_THAT(edge_z,Catch::Matchers::WithinAbs(edge_z_c, 1e-6));
    REQUIRE_THAT(edge_x,Catch::Matchers::WithinAbs(edge_x_c, 1e-6));
}
