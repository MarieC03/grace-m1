/**
 * @file test_weakhub_golden.cpp
 * @brief Golden-node test of the production Weakhub table loader + lookup
 *        against the raw HDF5 contents of a REAL opacity table.
 *
 * The synthetic-table unit tests (test_weakhub_table.cpp) prove the slot
 * mapping and interpolation math on data whose answer is known by
 * construction; they cannot catch a loader that reads the real file with
 * the wrong axis ordering, a transposed reorder loop, or a unit slip in
 * the axis metadata.  This test closes that gap:
 *
 *   1. grace::initialize (parfile selects m1.eas.kinds=[neutrino_weakhub])
 *      loads the table through the PRODUCTION path
 *      (weakhub::initialize_weakhub_from_params, lazy inside set_m1_eas).
 *   2. The test re-reads the SAME file with plain HDF5 calls and its own
 *      index arithmetic: the kappa datasets are rank-5, C-order
 *      {species, ymu, ye, temp, rho}, so
 *          raw[(((s*nymu + l)*nye + k)*ntemp + j)*nrho + i]
 *      is the value at node (s, l, k, j, i) -- INDEPENDENT of the loader's
 *      reorder loop.
 *   3. device_handle::lookup is queried exactly AT grid nodes; multilinear
 *      interpolation at a node returns the node value (up to the exp/log
 *      round-trip of the query coordinates, ~1 ulp), so the production
 *      output must match the raw file contents to tight relative tolerance,
 *      modulo the documented 1e-60 positivity floor.
 *
 * Table paths are machine-local (see configs/weakhub_golden_test.yaml) --
 * this is a local/manual check, not a CI-portable test.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>
#include <hdf5.h>

#include <grace_config.h>
#include <grace/config/config_parser.hh>
#include <grace/physics/grace_weakhub_table.hh>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::WithinRel;

namespace {

using namespace grace;

// lookup() floors non-positive opacities to this value before returning.
constexpr double kFloor = 1.0e-60;

// Independent image of the file: raw arrays in the HDF5 dataset's own
// C-order {species, ymu, ye, temp, rho}.
struct golden_table_t {
    int nspec = 0, nymu = 0, nye = 0, ntemp = 0, nrho = 0;
    std::vector<double> logrho, logtemp, ye, logymu;
    std::vector<double> ae, an, s;

    size_t raw_idx(int s_, int l, int k, int j, int i) const {
        return (((size_t(s_) * nymu + l) * nye + k) * ntemp + j) * nrho + i;
    }
};

void read_1d(hid_t file, char const* name, std::vector<double>& out)
{
    hid_t ds = H5Dopen(file, name, H5P_DEFAULT);
    REQUIRE(ds >= 0);
    hid_t sp = H5Dget_space(ds);
    hsize_t n = 0;
    REQUIRE(H5Sget_simple_extent_ndims(sp) == 1);
    H5Sget_simple_extent_dims(sp, &n, nullptr);
    out.resize(n);
    REQUIRE(H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    out.data()) >= 0);
    H5Sclose(sp);
    H5Dclose(ds);
}

golden_table_t read_golden(std::string const& path)
{
    golden_table_t g;
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    // Take the shape from the kappa dataset's OWN dataspace -- not from the
    // IV* scalars the loader trusts -- so a metadata/dataspace mismatch in
    // the file is caught rather than inherited.
    hid_t ds = H5Dopen(file, "kappa_a_en_grey_table", H5P_DEFAULT);
    REQUIRE(ds >= 0);
    hid_t sp = H5Dget_space(ds);
    REQUIRE(H5Sget_simple_extent_ndims(sp) == 5);
    hsize_t dims[5];
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    H5Sclose(sp);
    H5Dclose(ds);
    g.nspec = int(dims[0]);
    g.nymu  = int(dims[1]);
    g.nye   = int(dims[2]);
    g.ntemp = int(dims[3]);
    g.nrho  = int(dims[4]);

    read_1d(file, "logrho_IVtable",  g.logrho);
    read_1d(file, "logtemp_IVtable", g.logtemp);
    read_1d(file, "ye_IVtable",      g.ye);
    read_1d(file, "logymu_IVtable",  g.logymu);
    REQUIRE(g.logrho.size()  == size_t(g.nrho));
    REQUIRE(g.logtemp.size() == size_t(g.ntemp));
    REQUIRE(g.ye.size()      == size_t(g.nye));

    auto read_kappa = [&](char const* name, std::vector<double>& out) {
        hid_t d = H5Dopen(file, name, H5P_DEFAULT);
        REQUIRE(d >= 0);
        out.resize(size_t(g.nspec) * g.nymu * g.nye * g.ntemp * g.nrho);
        REQUIRE(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        out.data()) >= 0);
        H5Dclose(d);
    };
    read_kappa("kappa_a_en_grey_table",  g.ae);
    read_kappa("kappa_a_num_grey_table", g.an);
    read_kappa("kappa_s_grey_table",     g.s);

    H5Fclose(file);
    return g;
}

/// Run device_handle::lookup in a Kokkos kernel (production call site is
/// device code) and copy the 3x5 outputs back to the host.
std::array<double, 15> device_lookup(
    weakhub::device_handle const& h,
    double rho_code, double temp_mev, double yle, double ymu)
{
    Kokkos::View<double*> out("whg_out", 15);
    Kokkos::parallel_for("whg_lookup", 1, KOKKOS_LAMBDA(int) {
        const weakhub::interp_outputs r = h.lookup(rho_code, temp_mev, yle, ymu);
        for (int s = 0; s < 5; ++s) {
            out(s)      = r.kappa_a_en[s];
            out(5 + s)  = r.kappa_a_num[s];
            out(10 + s) = r.kappa_s[s];
        }
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(m, out);
    std::array<double, 15> a;
    for (int i = 0; i < 15; ++i) a[i] = m(i);
    return a;
}

double floored(double raw) { return raw > 0.0 ? raw : kFloor; }

}  // namespace


TEST_CASE("Weakhub production lookup reproduces the raw HDF5 table at grid "
          "nodes", "[weakhub][golden]")
{
    // The parfile selects neutrino_weakhub, so grace::initialize already
    // loaded the table through the production path.
    REQUIRE(weakhub::is_initialized());
    auto const& h = weakhub::get_device_handle();

    auto const path =
        grace::get_param<std::string>("m1", "eas", "weakhub_table");
    auto const g = read_golden(path);

    INFO("table: " << path);
    INFO("shape: nspec=" << g.nspec << " nymu=" << g.nymu << " nye=" << g.nye
         << " ntemp=" << g.ntemp << " nrho=" << g.nrho);

    // --- Loader metadata must match the dataset's own shape ---------------
    REQUIRE(h.n_species_table == g.nspec);
    REQUIRE(h.nrho  == g.nrho);
    REQUIRE(h.ntemp == g.ntemp);
    REQUIRE(h.nye   == g.nye);
    REQUIRE(h.nymu  == std::max(g.nymu, 1));
    REQUIRE_THAT(h.logrho_min,  WithinRel(g.logrho.front(),  1e-14));
    REQUIRE_THAT(h.logrho_max,  WithinRel(g.logrho.back(),   1e-14));
    REQUIRE_THAT(h.logtemp_min, WithinRel(g.logtemp.front(), 1e-14));
    REQUIRE_THAT(h.logtemp_max, WithinRel(g.logtemp.back(),  1e-14));

    // Axes must be strictly monotone or find_bracket misbehaves silently.
    for (int i = 1; i < g.nrho;  ++i) REQUIRE(g.logrho[i]  > g.logrho[i-1]);
    for (int j = 1; j < g.ntemp; ++j) REQUIRE(g.logtemp[j] > g.logtemp[j-1]);
    for (int k = 1; k < g.nye;   ++k) REQUIRE(g.ye[k]      > g.ye[k-1]);

    // Output-slot mapping for the species count of THIS file.  The local
    // golden table is a 3-species (nue, anue, nux) npe table; extend the
    // mapping here if a 5/6-species golden file is dropped in.
    REQUIRE(g.nspec == 3);   // premise of the slot mapping below
    constexpr int slot_of_species3[3] = {0, 1, 4};

    // --- Node lattice ------------------------------------------------------
    // Include both endpoints of every axis (find_bracket boundary paths) and
    // a stride through the interior.  ~8 x 8 x 6 nodes x 3 species x 3
    // tables of assertions.
    auto lattice = [](int n, int target) {
        std::vector<int> idx;
        int const step = std::max(1, (n - 1) / std::max(1, target - 1));
        for (int i = 0; i < n; i += step) idx.push_back(i);
        if (idx.back() != n - 1) idx.push_back(n - 1);
        return idx;
    };
    auto const ii = lattice(g.nrho, 8);
    auto const jj = lattice(g.ntemp, 8);
    auto const kk = lattice(g.nye, 6);

    size_t n_checked = 0;
    for (int l = 0; l < std::max(g.nymu, 1); ++l) {
        // 3D tables carry a single (degenerate) ymu plane; the production
        // handle clamps any queried ymu to it.
        double const ymu_q =
            (g.nymu > 1 ? std::exp(g.logymu[l]) : 0.0);
        for (int k : kk) {
            for (int j : jj) {
                for (int i : ii) {
                    auto const r = device_lookup(
                        h,
                        std::exp(g.logrho[i]),
                        std::exp(g.logtemp[j]),
                        g.ye[k], ymu_q);
                    for (int s = 0; s < g.nspec; ++s) {
                        int const slot = slot_of_species3[s];
                        double const e_ae = floored(g.ae[g.raw_idx(s, l, k, j, i)]);
                        double const e_an = floored(g.an[g.raw_idx(s, l, k, j, i)]);
                        double const e_s  = floored(g.s [g.raw_idx(s, l, k, j, i)]);
                        INFO("node (s=" << s << ", l=" << l << ", k=" << k
                             << ", j=" << j << ", i=" << i << ")  slot=" << slot
                             << "  logrho=" << g.logrho[i]
                             << " logtemp=" << g.logtemp[j]
                             << " ye=" << g.ye[k]);
                        // Node queries re-enter through exp/log of the
                        // coordinates (~1 ulp off the node), so allow a
                        // tight relative band rather than bit equality.
                        REQUIRE_THAT(r[slot],      WithinRel(e_ae, 1e-10));
                        REQUIRE_THAT(r[5 + slot],  WithinRel(e_an, 1e-10));
                        REQUIRE_THAT(r[10 + slot], WithinRel(e_s,  1e-10));
                    }
                    ++n_checked;
                }
            }
        }
    }
    INFO("nodes checked: " << n_checked);
    REQUIRE(n_checked > 0);
}
