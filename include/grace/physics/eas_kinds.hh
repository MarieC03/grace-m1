/**
 * @file eas_kinds.hh
 * @brief Strongly-typed, composable selection of M1 EAS rate providers.
 *
 * The parfile is parsed exactly once, here, into a set of enums.  All
 * branching elsewhere (set_m1_eas dispatch, weakhub gating) compares enum
 * values — the valid strings exist in one place only, unknown values fail
 * loudly at parse time with the full list of options, and a `switch` over
 * the enum gets compiler exhaustiveness checking.
 *
 * Parfile interface (either form):
 *   m1: { eas: { kind:  "neutrino_analytic" } }            # single provider
 *   m1: { eas: { kinds: ["photon_rates",
 *                        "neutrino_analytic"] } }           # composed
 *
 * Each provider fills the aux rate slots for the species it owns; the
 * combination rules below say which providers may coexist.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Volume methods to
 * simulate relativistic spacetimes and plasmas.
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
#ifndef GRACE_PHYSICS_EAS_KINDS_HH
#define GRACE_PHYSICS_EAS_KINDS_HH

#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>

#include <algorithm>
#include <string>
#include <vector>

namespace grace {

enum class eas_kind_t {
    test,               ///< synthetic rates for transport tests
    photon_rates,       ///< grey photon opacities
    neutrino_analytic,  ///< analytic neutrino rates (Ruffert+ style)
    neutrino_weakhub,   ///< Weakhub opacity table + analytic thermal extras
    bns_nurates         ///< bns_nurates library (set in auxiliaries.cpp)
};

/// Parse one provider name.  Unknown values abort with the list of valid
/// options — a parfile typo can never silently select a default.
inline eas_kind_t parse_eas_kind(std::string const& kind)
{
    if (kind == "test")              return eas_kind_t::test ;
    if (kind == "photon_rates")      return eas_kind_t::photon_rates ;
    if (kind == "neutrino_analytic") return eas_kind_t::neutrino_analytic ;
    if (kind == "neutrino_weakhub")  return eas_kind_t::neutrino_weakhub ;
    if (kind == "bns_nurates")       return eas_kind_t::bns_nurates ;
    ERROR("m1.eas kind '" << kind << "' is not one of: test, photon_rates, "
          "neutrino_analytic, neutrino_weakhub, bns_nurates") ;
}

/// The set of active EAS providers, in parfile order.
struct eas_selection_t {
    std::vector<eas_kind_t> kinds ;

    bool contains(eas_kind_t k) const {
        return std::find(kinds.begin(), kinds.end(), k) != kinds.end() ;
    }
} ;

/// Read and validate the EAS provider selection.  Host-only.
///
/// Accepts `m1.eas.kinds` (list) or, if absent, the legacy scalar
/// `m1.eas.kind`.  Combination rules — these encode which providers write
/// the same rate slots, not dispatch limitations:
///   - "test" fills every slot synthetically: must be the sole provider.
///   - at most one NEUTRINO provider (analytic / weakhub / bns_nurates),
///     since they all own the neutrino species' slots.
///   - photons + neutrinos: blocked until the M1 variable layout gives
///     photons their own species block — today photon_eas_op writes the
///     same ETA1_/KAPPA*1_ slots as neutrino species 1, so composing them
///     would mean last-writer-wins.  Lift the check here once photon slots
///     exist; the dispatch loop already handles the combination.
inline eas_selection_t get_eas_selection()
{
    eas_selection_t sel ;
    try {
        // Preferred: list form.
        for (auto const& name :
             get_param<std::vector<std::string>>("m1","eas","kinds"))
            sel.kinds.push_back(parse_eas_kind(name)) ;
    } catch (...) {
        // Missing `kinds` key (the parser offers no "has_param") — handled
        // by the empty-list fallback below.
    }

    // The schema default for `kinds` is the empty list: empty means
    // "use the scalar m1.eas.kind".
    if (sel.kinds.empty())
        sel.kinds.push_back(
            parse_eas_kind(get_param<std::string>("m1","eas","kind"))) ;

    for (size_t a = 0; a < sel.kinds.size(); ++a)
        for (size_t b = a + 1; b < sel.kinds.size(); ++b)
            if (sel.kinds[a] == sel.kinds[b])
                ERROR("m1.eas.kinds lists the same provider twice.") ;

    if (sel.contains(eas_kind_t::test) && sel.kinds.size() > 1)
        ERROR("m1.eas: 'test' fills all rate slots and cannot be combined "
              "with other providers.") ;

    int const n_neutrino =
          int(sel.contains(eas_kind_t::neutrino_analytic))
        + int(sel.contains(eas_kind_t::neutrino_weakhub))
        + int(sel.contains(eas_kind_t::bns_nurates)) ;
    if (n_neutrino > 1)
        ERROR("m1.eas: neutrino_analytic, neutrino_weakhub and bns_nurates "
              "all own the neutrino rate slots; select exactly one.") ;

    #ifndef GRACE_M1_PHOTONS
    if (sel.contains(eas_kind_t::photon_rates) && n_neutrino > 0)
        ERROR("m1.eas: photon_rates + a neutrino provider requires the "
              "dedicated photon variable block: reconfigure with "
              "-DGRACE_M1_PHOTONS=ON.") ;
    #endif

    #ifndef GRACE_HAVE_BNS_NURATES
    if (sel.contains(eas_kind_t::bns_nurates))
        ERROR("m1.eas: 'bns_nurates' selected, but GRACE was built without "
              "the bns_nurates submodule.  Run "
              "'git submodule update --init extern/bns_nurates' and "
              "reconfigure.") ;
    #endif

    return sel ;
}

// ---------------------------------------------------------------------------
// Secondary m1.eas mode selectors (consumed by neutrinos_eas_op).
// Same contract as eas_kind_t: strings parsed once, here; a MISSING key
// selects the documented default; an unknown VALUE aborts with the list of
// options.  The enums are plain ints underneath and safe to carry into
// device kernels.
// ---------------------------------------------------------------------------

enum class betaeq_mode_t : int {
    off,        ///< use the evolved Ye/Ymu as-is
    chemical,   ///< reset Ye to neutrinoless chemical equilibrium per cell
    timescale   ///< relax (T, Ye) toward radiation-matter equilibrium when
                ///< the equilibration time tau_beta is shorter than dt
};

enum class tau_policy_kind_t : int {
    none,              ///< tau = 0 everywhere (optically thin fugacities)
    analytic_density,  ///< Deaton+ 2013 density fit (cold NS only!)
    local_spherical,   ///< kappa * (r_outer - r) with kappa from the rate
                       ///< evaluation itself (init seeded by the cold fit)
    local_kappa,       ///< per-species (kappa_a + kappa_s) * (r_outer - r)
                       ///< using the PREVIOUS step's opacities from aux —
                       ///< pointwise, species-dependent, no cold-fit seed
    eikonal            ///< Neilsen+ 2014 min-path optical depth, relaxed in
                       ///< the OPTD_* state fields (needs -DGRACE_M1_OPTICAL_DEPTH)
};

namespace detail {
/// Optional string parameter under m1.eas.  The parser has no "has_param";
/// a missing key throws, which selects the fallback.  Scoped narrowly so
/// only the lookup itself is caught — parse errors still abort loudly.
inline std::string get_eas_param_or(std::string fallback, char const* key)
{
    try { return get_param<std::string>("m1","eas",key) ; }
    catch (...) { return fallback ; }
}
} // namespace detail

inline betaeq_mode_t get_betaeq_mode()
{
    auto const mode = detail::get_eas_param_or("off", "betaeq_policy") ;
    if (mode == "off")       return betaeq_mode_t::off ;
    if (mode == "chemical")  return betaeq_mode_t::chemical ;
    if (mode == "timescale") return betaeq_mode_t::timescale ;
    ERROR("m1.eas.betaeq_policy = '" << mode
          << "' is not one of: off, chemical, timescale") ;
}

inline tau_policy_kind_t get_tau_policy_kind()
{
    auto const mode = detail::get_eas_param_or("none", "tau_policy") ;
    if (mode == "none")             return tau_policy_kind_t::none ;
    if (mode == "analytic_density") return tau_policy_kind_t::analytic_density ;
    if (mode == "local_spherical")  return tau_policy_kind_t::local_spherical ;
    if (mode == "local_kappa")      return tau_policy_kind_t::local_kappa ;
    if (mode == "eikonal") {
        #ifndef GRACE_M1_OPTICAL_DEPTH
        ERROR("m1.eas.tau_policy = 'eikonal' requires a build with "
              "-DGRACE_M1_OPTICAL_DEPTH=ON.") ;
        #endif
        return tau_policy_kind_t::eikonal ;
    }
    ERROR("m1.eas.tau_policy = '" << mode
          << "' is not one of: none, analytic_density, local_spherical, "
             "local_kappa, eikonal") ;
}

} // namespace grace

#endif /* GRACE_PHYSICS_EAS_KINDS_HH */
