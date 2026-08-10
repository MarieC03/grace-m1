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
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::WithinRel;

namespace {

using namespace grace;

// lookup() maps non-positive / non-finite opacities to this value before
// returning; the 1e-60 positivity floor is applied at the end of the rates.
constexpr double kFloor = weakhub::kappa_zero_cgs;

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


// ---------------------------------------------------------------------------
//  Table-agnostic bounds check.
//
//  The round-trip test above is bound to the 3-species DD2 table (it asserts
//  g.nspec == 3 for the slot mapping), so a 5/6-species production table never
//  reaches its logrho_min assertion.  This case makes the same comparison for
//  WHATEVER table the parfile selects, and prints the domain next to the
//  atmosphere floor -- lookup() now floors rather than clamps below the table's
//  rho/T range, so where that boundary sits relative to rho_fl decides how much
//  of the star loses its table opacities.
//
//  Run against a production parfile:
//    ./weakhub_golden_test "[bounds]" --grace-parfile <your parfile>
// ---------------------------------------------------------------------------
TEST_CASE("Weakhub domain is read exactly from the file", "[weakhub][bounds]")
{
    REQUIRE(weakhub::is_initialized());
    auto const& h = weakhub::get_device_handle();
    std::string const path =
        grace::get_param<std::string>("m1", "eas", "weakhub_table");
    auto const g = read_golden(path);

    printf("\n=============== Weakhub domain as loaded ===============\n");
    printf("file: %s\n", path.c_str());
    printf("shape (from the kappa dataspace): nspec=%d nymu=%d nye=%d ntemp=%d nrho=%d\n",
           g.nspec, g.nymu, g.nye, g.ntemp, g.nrho);
    printf("               %-22s %-22s\n", "file", "device_handle");
    printf("  logrho_min   %-22.15g %-22.15g\n", g.logrho.front(),  h.logrho_min);
    printf("  logrho_max   %-22.15g %-22.15g\n", g.logrho.back(),   h.logrho_max);
    printf("  logtemp_min  %-22.15g %-22.15g\n", g.logtemp.front(), h.logtemp_min);
    printf("  logtemp_max  %-22.15g %-22.15g\n", g.logtemp.back(),  h.logtemp_max);
    printf("  ye_min       %-22.15g %-22.15g\n", g.ye.front(),      h.ye_min);
    printf("  ye_max       %-22.15g %-22.15g\n", g.ye.back(),       h.ye_max);
    if (g.nymu > 1) {
        printf("  logymu_min   %-22.15g %-22.15g\n", g.logymu.front(), h.logymu_min);
        printf("  logymu_max   %-22.15g %-22.15g\n", g.logymu.back(),  h.logymu_max);
    }
    printf("\nlinear domain (axes are natural log; rho in CODE units, T in MeV):\n");
    printf("  rho  [%.6e, %.6e]\n", std::exp(h.logrho_min),  std::exp(h.logrho_max));
    printf("  T    [%.6e, %.6e]\n", std::exp(h.logtemp_min), std::exp(h.logtemp_max));
    printf("  ye   [%.6f, %.6f]\n", h.ye_min, h.ye_max);
    if (h.nymu > 1)
        printf("  ymu  [%.6e, %.6e]\n", std::exp(h.logymu_min), std::exp(h.logymu_max));

    // Where the table floor sits relative to the fluid atmosphere.  Below the
    // table's rho the opacities are floored, so a table floor well above rho_fl
    // means a shell of the star gets no table contribution at all.
    double const rho_fl   = grace::get_param<double>("grmhd","atmosphere","rho_fl");
    double const rho_tmin = std::exp(h.logrho_min);
    double const rho_tmax = std::exp(h.logrho_max);

    // The EOS and the weakhub table are independent files with independent
    // domains.  Nothing forces them to agree, and for SFHo they do not: the
    // EOS runs several decades further down.  Fluid can therefore exist at
    // densities where no opacity data exists at all.
    printf("\ndomain comparison (code units):\n");
    printf("  %-26s [%.6e, %.6e]\n", "weakhub table rho", rho_tmin, rho_tmax);
    if (grace::get_param<std::string>("eos","eos_type") == "leptonic") {
        auto const eos = grace::eos::get().get_eos<grace::leptonic_eos_4d_t>();
        double const rho_emin = eos.density_minimum();
        double const rho_emax = eos.density_maximum();
        printf("  %-26s [%.6e, %.6e]\n", "EOS rho", rho_emin, rho_emax);
        printf("  %-26s [%.6e, %.6e]  (%.4f MeV)\n", "EOS T",
               eos.temperature_minimum(), eos.temperature_maximum(),
               eos.temperature_minimum());
        printf("  %-26s [%.6f, %.6f]\n", "EOS ye",
               eos.get_c2p_ye_min(), eos.get_c2p_ye_max());
        if (rho_emin < rho_tmin)
            printf("  ==> the EOS reaches %.1fx lower in rho than the weakhub table:\n"
                   "      rho in [%.3e, %.3e] is valid fluid with NO opacity data.\n",
                   rho_tmin / rho_emin, rho_emin, rho_tmin);
    }
    printf("  %-26s %.6e\n", "atmosphere rho_fl", rho_fl);
    if (rho_fl < rho_tmin)
        printf("  ==> rho_fl sits BELOW the weakhub floor by %.1fx: every cell in\n"
               "      [%.3e, %.3e] is outside the table and gets floored rates.\n",
               rho_tmin / rho_fl, rho_fl, rho_tmin);
    else
        printf("  ==> the evolved density range starts inside the weakhub table.\n");
    printf("=======================================================\n");

    // The actual assertions: every bound and every axis node, exactly.
    REQUIRE(h.n_species_table == g.nspec);
    REQUIRE(h.nrho  == g.nrho);
    REQUIRE(h.ntemp == g.ntemp);
    REQUIRE(h.nye   == g.nye);
    REQUIRE(h.nymu  == std::max(g.nymu, 1));
    REQUIRE_THAT(h.logrho_min,  WithinRel(g.logrho.front(),  1e-14));
    REQUIRE_THAT(h.logrho_max,  WithinRel(g.logrho.back(),   1e-14));
    REQUIRE_THAT(h.logtemp_min, WithinRel(g.logtemp.front(), 1e-14));
    REQUIRE_THAT(h.logtemp_max, WithinRel(g.logtemp.back(),  1e-14));
    REQUIRE_THAT(h.ye_min,      WithinRel(g.ye.front(),      1e-14));
    REQUIRE_THAT(h.ye_max,      WithinRel(g.ye.back(),       1e-14));
    if (g.nymu > 1) {
        REQUIRE_THAT(h.logymu_min, WithinRel(g.logymu.front(), 1e-14));
        REQUIRE_THAT(h.logymu_max, WithinRel(g.logymu.back(),  1e-14));
    }

    // The bounds are only meaningful if the axis VIEWS agree too -- find_bracket
    // reads those, not the scalars.
    auto ax = [](Kokkos::View<double*> const& v) {
        auto m = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(m, v);
        std::vector<double> o(v.extent(0));
        for (size_t i = 0; i < o.size(); ++i) o[i] = m(i);
        return o;
    };
    auto const a_lr = ax(h.logrho_axis);
    auto const a_lt = ax(h.logtemp_axis);
    auto const a_ye = ax(h.ye_axis);
    for (int i = 0; i < g.nrho;  ++i) REQUIRE_THAT(a_lr[i], WithinRel(g.logrho[i],  1e-14));
    for (int j = 0; j < g.ntemp; ++j) REQUIRE_THAT(a_lt[j], WithinRel(g.logtemp[j], 1e-14));
    for (int k = 0; k < g.nye;   ++k) REQUIRE_THAT(a_ye[k], WithinRel(g.ye[k],      1e-14));
    if (g.nymu > 1) {
        auto const a_lm = ax(h.logymu_axis);
        for (int l = 0; l < g.nymu; ++l)
            REQUIRE_THAT(a_lm[l], WithinRel(g.logymu[l], 1e-14));
    }
}
