/**
 * @file test_weakhub_halo.cpp
 * @brief Evaluate the production EAS chain at the exact halo state reported by
 *        the [BR diag] instrumentation, and account for every factor between
 *        the raw Weakhub table entry and the kappa the run actually applied.
 *
 * Context: the halo cells sit at rho ~ 1.07e-12 code units, but the SFHo npemu
 * Weakhub table's rho axis starts at 1e-9 -- three decades higher.  device_
 * handle::lookup clamps into range (grace_weakhub_table.hh clamp_state), so
 * those cells are evaluated at the table's lowest-density row.  Since
 * kappa ~ rho (measured slope 1.000 over the first decade), that alone is a
 * ~10^3 overstatement.  But the run reported kappa_s ~ 3e-3 where the table
 * holds ~2e-8 at that corner, a further ~10^5 that the clamp does NOT explain.
 *
 * kappa_a is expected to exceed its table value: pair / plasmon / brems feed
 * kappa_a through Kirchhoff (compute_all_species_weakhub).  kappa_s is NOT fed
 * by them -- it is assigned straight from the table -- so any excess there has
 * a different cause.  This probe prints each stage so the arithmetic is
 * visible rather than argued about.
 *
 * Bound to configs/weakhub_halo_test.yaml (2 GB weakhub + leptonic 4D EOS).
 */
#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/grace_weakhub_table.hh>
#include <grace/physics/eas_optical_depth.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>

#include <array>
#include <cmath>
#include <cstdio>

#if defined(GRACE_ENABLE_M1) && GRACE_M1_NU_SPECIES >= 5

namespace {
using namespace grace;

// The halo cell reported by [BR diag] at iteration 140 (rank=2 q=9), together
// with the kappa/eta that run applied there.
constexpr double kRho = 1.0809e-12;   // code units
constexpr double kT   = 2.0938e-01;   // MeV
constexpr double kYe  = 5.0000e-01;
constexpr double kYmu = 5.0017e-04;

struct ref_t { int s; double ka, ks, eta; };
constexpr ref_t kRun[] = {
    {0, 7.248464e-08, 8.973243e-03, 1.506212e-23},
    {1, 1.245348e-08, 2.046313e-04, 2.587865e-24},
    {2, 1.476518e-25, 3.080229e-03, 1.421345e-80},
    {3, 1.476518e-25, 8.326522e-04, 1.421345e-80},
    {4, 6.219588e-13, 7.065500e-05, 2.584860e-28},
};
constexpr char const* kName[5] = {"nue","anue","numu","anumu","nux"};

}  // namespace


TEST_CASE("Weakhub halo-state rate breakdown", "[weakhub][halo][probe]")
{
    REQUIRE(weakhub::is_initialized());
    auto const& h = weakhub::get_device_handle();

    printf("\n================ Weakhub halo-state probe ================\n");
    printf("state: rho=%.5e (code)  T=%.5f MeV  Ye=%.5f  Ymu=%.5e\n",
           kRho, kT, kYe, kYmu);

    // ---- 1. table domain, and whether this state is inside it --------------
    printf("\n[1] table domain vs the halo state\n");
    printf("    rho : table [%.5e, %.5e]   state %.5e  -> %s\n",
           std::exp(h.logrho_min), std::exp(h.logrho_max), kRho,
           (kRho < std::exp(h.logrho_min)) ? "BELOW FLOOR (clamped)" : "in range");
    printf("    T   : table [%.5e, %.5e]   state %.5e  -> %s\n",
           std::exp(h.logtemp_min), std::exp(h.logtemp_max), kT,
           (kT < std::exp(h.logtemp_min) || kT > std::exp(h.logtemp_max))
               ? "OUT OF RANGE (clamped)" : "in range");
    printf("    Ye  : table [%.5f, %.5f]   state %.5f  -> %s\n",
           h.ye_min, h.ye_max, kYe,
           (kYe > h.ye_max) ? "AT/ABOVE CEILING (clamped)" : "in range");
    printf("    Ymu : table [%.5e, %.5e]   state %.5e  -> %s\n",
           std::exp(h.logymu_min), std::exp(h.logymu_max), kYmu,
           (kYmu < std::exp(h.logymu_min)) ? "AT/BELOW FLOOR (clamped)" : "in range");
    if (kRho < std::exp(h.logrho_min))
        printf("    ==> rho clamp factor = %.1fx too dense\n",
               std::exp(h.logrho_min) / kRho);

    // ---- 2. raw table lookup (production device path) ----------------------
    Kokkos::View<double*> out("wh_halo_out", 15);
    Kokkos::parallel_for("wh_halo_lookup", 1, KOKKOS_LAMBDA(int) {
        const weakhub::interp_outputs r = h.lookup(kRho, kT, kYe, kYmu);
        for (int s = 0; s < 5; ++s) {
            out(s)      = r.kappa_a_en[s];
            out(5 + s)  = r.kappa_a_num[s];
            out(10 + s) = r.kappa_s[s];
        }
    });
    Kokkos::fence();
    auto om = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(om, out);

    printf("\n[2] RAW weakhub lookup (after clamping), per species\n");
    printf("    %-6s %14s %14s %14s\n", "spec", "kappa_a_en", "kappa_a_num", "kappa_s");
    for (int s = 0; s < 5; ++s)
        printf("    %-6s %14.6e %14.6e %14.6e\n",
               kName[s], om(s), om(5+s), om(10+s));

    // ---- 3. full assembled rates, extras ON vs OFF -------------------------
    auto eos = eos::get().get_eos<leptonic_eos_4d_t>();
    double const xyz[3] = {1.0, 0.0, 0.0};

    // Assemble twice: with the extra emission processes and without, so their
    // contribution to kappa_a (and their absence from kappa_s) is explicit.
    Kokkos::View<double*> rout("wh_halo_rates", 40);
    Kokkos::parallel_for("wh_halo_rates", 1, KOKKOS_LAMBDA(int) {
        tau_policy_none tp{};
        double const eps_rad[NUMSPECIES] = {0,0,0,0,0};
        fugacity_state F = make_fugacity_state(eos, kRho, kT, kYe, kYmu, 1.0, xyz, tp);
        for (int pass = 0; pass < 2; ++pass) {
            bool const extras = (pass == 1);
            auto r = compute_all_species_weakhub(
                h, F, extras, extras, extras, xyz, tp,
                /*apply_temp_correction=*/false, eps_rad);
            for (int s = 0; s < 5; ++s) {
                rout(pass*20 + s)      = r.out[s].kappa_a;
                rout(pass*20 + 5 + s)  = r.out[s].kappa_s;
                rout(pass*20 + 10 + s) = r.out[s].eta_E;
                rout(pass*20 + 15 + s) = r.out[s].kappa_n;
            }
        }
    });
    Kokkos::fence();
    auto rm = Kokkos::create_mirror_view(rout);
    Kokkos::deep_copy(rm, rout);

    printf("\n[3] assembled rates, temp-correction OFF\n");
    printf("    %-6s | %13s %13s | %13s %13s\n",
           "spec", "ka (no extra)", "ka (+extras)", "ks (no extra)", "ks (+extras)");
    for (int s = 0; s < 5; ++s)
        printf("    %-6s | %13.6e %13.6e | %13.6e %13.6e\n",
               kName[s], rm(s), rm(20+s), rm(5+s), rm(25+s));

    // ---- 4. compare against what the run actually applied ------------------
    printf("\n[4] vs the values the Hunter run applied at this cell\n");
    printf("    %-6s | %13s %13s %8s | %13s %13s %8s\n",
           "spec", "ka here", "ka in run", "ratio", "ks here", "ks in run", "ratio");
    for (auto const& r : kRun) {
        double const ka_here = rm(20 + r.s);
        double const ks_here = rm(25 + r.s);
        printf("    %-6s | %13.6e %13.6e %8.2e | %13.6e %13.6e %8.2e\n",
               kName[r.s], ka_here, r.ka, r.ka / (ka_here > 0 ? ka_here : 1e-300),
               ks_here, r.ks, r.ks / (ks_here > 0 ? ks_here : 1e-300));
    }
    // Which rate path ran?  The analytic fallback (weakhub.valid == false)
    // computes kappa_s from the local state with no table domain, so it is
    // finite at any density -- a completely different number from either the
    // clamped table value or the floor.
    Kokkos::View<double*> aout("wh_halo_analytic", 10);
    Kokkos::parallel_for("wh_halo_analytic", 1, KOKKOS_LAMBDA(int) {
        tau_policy_none tp{};
        double const eps_rad[NUMSPECIES] = {0,0,0,0,0};
        fugacity_state F = make_fugacity_state(eos, kRho, kT, kYe, kYmu, 1.0, xyz, tp);
        auto r = compute_all_species(F, true, true, true, true, xyz, tp, false, eps_rad);
        for (int s = 0; s < 5; ++s) {
            aout(s)     = r.out[s].kappa_a;
            aout(5 + s) = r.out[s].kappa_s;
        }
    });
    Kokkos::fence();
    auto am = Kokkos::create_mirror_view(aout);
    Kokkos::deep_copy(am, aout);
    printf("\n[5] ANALYTIC path (the silent fallback when weakhub.valid==false)\n");
    printf("    %-6s %14s %14s\n", "spec", "kappa_a", "kappa_s");
    for (int s = 0; s < 5; ++s)
        printf("    %-6s %14.6e %14.6e\n", kName[s], am(s), am(5+s));

    printf("\n    NB kappa_s is assigned straight from the table\n"
           "       (rates.kappa_s[s] = tbl.kappa_s[s]) and receives NOTHING\n"
           "       from pair/plasmon/brems, so a ks ratio far from 1 is not\n"
           "       explained by the extra emission processes.\n");
    printf("==========================================================\n");

    REQUIRE(true);
}

#endif  // GRACE_ENABLE_M1 && GRACE_M1_NU_SPECIES >= 5
