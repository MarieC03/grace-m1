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
#include <fstream>
#include <sstream>
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
    timescale,  ///< relax (T, Ye, Ymu) toward radiation-matter equilibrium with a
                ///< SINGLE global timescale tau_beta_min (min over all species),
                ///< applying the full joint (Ye,Ymu,T) equilibrium together
    gieg        ///< per-flavor TWO-timescale equilibration (Gieg+ 2026, Table 3):
                ///< independent tau_e, tau_mu -> 9-state dispatch with partial
                ///< equilibration that holds the free-streaming flavour fixed
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
    if (mode == "gieg")      return betaeq_mode_t::gieg ;
    ERROR("m1.eas.betaeq_policy = '" << mode
          << "' is not one of: off, chemical, timescale, gieg") ;
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

// What the Weakhub grey table already carries, so that nothing is counted twice: charged current
// and scattering only (none), e+e- pairs and NN bremsstrahlung folded into kappa_a (thermal), and
// inverse lepton decay on top (thermal_decay).
enum class weakhub_content_t : int { none, thermal, thermal_decay };

inline char const* weakhub_content_name(weakhub_content_t c)
{
    return c==weakhub_content_t::none ? "none" : c==weakhub_content_t::thermal ? "thermal" : "thermal+decay";
}

namespace detail {
// One "<label> ... : T/F" flag of a Weakhub tableinfo log; false when no line carries both words.
inline bool tableinfo_flag(std::string const& text, char const* word_a, char const* word_b, bool& value)
{
    std::istringstream lines(text);
    for(std::string line; std::getline(lines,line);) {
        if(line.find(word_a)==std::string::npos || line.find(word_b)==std::string::npos) continue;
        auto const colon=line.find(':');
        if(colon==std::string::npos) continue;
        auto const v=line.find_first_not_of(" \t",colon+1);
        if(v==std::string::npos) continue;
        value=(line[v]=='T' || line[v]=='t');
        return true;
    }
    return false;
}
}

// Resolved once from m1.eas.weakhub_table_content; "auto" reads Weakhub's companion file
// <weakhub_table minus .h5>_tableinfo.log, which is shipped next to every table.  Only meaningful
// with a Weakhub table selected.  refresh re-reads the parfile (tests).
inline weakhub_content_t get_weakhub_table_content(bool refresh=false)
{
    static bool resolved=false;
    static weakhub_content_t cached=weakhub_content_t::none;
    if(resolved && !refresh) return cached;
    std::string const mode=detail::get_eas_param_or("auto","weakhub_table_content");
    std::string source;
    if(mode=="none") cached=weakhub_content_t::none;
    else if(mode=="thermal") cached=weakhub_content_t::thermal;
    else if(mode=="thermal+decay") cached=weakhub_content_t::thermal_decay;
    else if(mode=="auto") {
        std::string log=detail::get_eas_param_or("","weakhub_table");
        if(log.size()>3 && log.compare(log.size()-3,3,".h5")==0) log.erase(log.size()-3);
        log+="_tableinfo.log";
        std::ifstream in(log);
        std::string const text((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        bool pair=false,brems=false,decay=false;
        if(!in.is_open() || !detail::tableinfo_flag(text,"pair","(Emiss)",pair)
           || !detail::tableinfo_flag(text,"Brems","(Emiss)",brems))
            ERROR("m1.eas.weakhub_table_content: auto needs Weakhub's companion log " << log
                  << " next to the table (copy it from where the table came from), or set "
                     "m1.eas.weakhub_table_content: none | thermal | thermal+decay by hand");
        detail::tableinfo_flag(text,"Inverse lepton decay","",decay);   // absent in grey logs
        cached=decay ? weakhub_content_t::thermal_decay
             : (pair || brems) ? weakhub_content_t::thermal : weakhub_content_t::none;
        source=" (from "+log+")";
    }
    else ERROR("m1.eas.weakhub_table_content must be auto, none, thermal or thermal+decay (got " << mode << ")");
    resolved=true;
    GRACE_INFO("Weakhub table content: {}{}", weakhub_content_name(cached), source);
    return cached;
}

// standard = the table/analytic grey rates GRACE always had; "legacy" is accepted as an alias.
enum class pair_treatment_t : int { standard, equilibrium, evolved };

inline bool get_muon_decay()
{
    try { return get_param<bool>("m1","eas","muon_decay"); } catch(...) { return false; }
}

inline int get_muon_decay_kernel_order()
{
    int n=24;
    try { n=get_param<int>("m1","eas","muon_decay_kernel_order"); } catch(...) {}
    if(n<8 || n>64) ERROR("muon_decay_kernel_order must be between 8 and 64");
    return n;
}

inline pair_treatment_t get_pair_treatment()
{
    auto name=detail::get_eas_param_or("standard", "pair_treatment");
    if(name=="legacy") name="standard";
    if(name!="standard" && name!="equilibrium" && name!="evolved")
        ERROR("m1.eas.pair_treatment must be standard, equilibrium or evolved");
    // With pair_treatment: standard, muon_decay selects the vacuum decay sink of the standard
    // rates (checked in check_m1_setup); the kernel decay below belongs to the kernel modes.
#ifndef GRACE_M1_PAIR_KERNELS
    // The spectral kernels are compiled out: this build has the standard grey rates only.
    if(name!="standard")
        ERROR("m1.eas.pair_treatment: " << name << " needs a build with -DGRACE_M1_PAIR_KERNELS=ON; "
              "set pair_treatment: standard for this build");
    return pair_treatment_t::standard;
#else
    if(name=="standard") return pair_treatment_t::standard;
    bool const decay=get_muon_decay();
    if(decay) {
#if GRACE_M1_NU_SPECIES < 5 || !defined(GRACE_ENABLE_MUONS)
        ERROR("muon_decay requires five neutrino species and GRACE_ENABLE_MUONS");
#endif
        if(get_param<std::string>("eos","eos_type")!="leptonic")
            ERROR("muon_decay requires the leptonic EOS");
        if(get_param<bool>("eos","leptonic","dilute_muon_suppression")
           || !get_param<bool>("eos","leptonic","add_ele_contribution"))
            ERROR("muon_decay requires dilute_muon_suppression=false and add_ele_contribution=true with an electron-free baryonic EOS");
        if(get_eas_selection().contains(eas_kind_t::neutrino_weakhub)
           && get_weakhub_table_content()==weakhub_content_t::thermal_decay)
            ERROR("The kernel muon decay needs a Weakhub table without inverse lepton decay (weakhub_table_content thermal+decay)");
        (void)get_muon_decay_kernel_order();
    }
#if GRACE_M1_NU_SPECIES < 3 || !defined(GRACE_HAVE_BNS_NURATES)
    ERROR("New pair treatments require 3/5 neutrino species and the pinned bns_nurates submodule");
#endif
    auto const sel=get_eas_selection();
    if(!sel.contains(eas_kind_t::neutrino_analytic) && !sel.contains(eas_kind_t::neutrino_weakhub))
        ERROR("New pair treatments require neutrino_analytic or neutrino_weakhub EAS");
    if(sel.contains(eas_kind_t::neutrino_weakhub) && get_weakhub_table_content()!=weakhub_content_t::none)
        ERROR("The kernel pair treatments need a Weakhub table without pairs and bremsstrahlung (weakhub_table_content none)");
    if(get_betaeq_mode()!=betaeq_mode_t::off)
        ERROR("New pair treatments currently require betaeq_policy: off");
    if(get_param<bool>("m1","eas","plasmon_decay")
       && detail::get_eas_param_or("none","pair_plasmon_model")!="transverse_mass")
        ERROR("New plasmon pairs require pair_plasmon_model: transverse_mass; see the documented approximation, or disable plasmon_decay");
    if(decay && name=="equilibrium") {
        static bool warned=false;
        if(!warned) {
            GRACE_WARN("Muon decay with equilibrium partners is a comparison approximation: "
                       "it does not preserve equal daughter counts away from LTE. "
                       "Use evolved partners for event conservation; monitor decay_equilibrium_charge_defect.");
            warned=true;
        }
    }
    return name=="equilibrium" ? pair_treatment_t::equilibrium : pair_treatment_t::evolved;
#endif  // GRACE_M1_PAIR_KERNELS
}

inline int get_pair_quadrature_order()
{
    int n=16;
    try { n=get_param<int>("m1","eas","pair_quadrature_order"); } catch(...) {}
    if(n<8 || n>32) ERROR("pair_quadrature_order must be between 8 and 32");
    return n;
}

// Parfile against build, EOS and table, once on both start paths (grace_initialize): what the
// Weakhub table already carries switches the analytic extras off and forbids a second decay; the
// standard-path muon decay (the vacuum sink) needs the muonic leptonic EOS.  The kernel modes
// check themselves in get_pair_treatment().
inline void check_m1_setup()
{
    bool const decay=get_muon_decay();
    bool const standard=get_pair_treatment()==pair_treatment_t::standard;
    if(get_eas_selection().contains(eas_kind_t::neutrino_weakhub)) {
        auto const content=get_weakhub_table_content();
        if(content!=weakhub_content_t::none
           && (get_param<bool>("m1","eas","pair_annihilation") || get_param<bool>("m1","eas","bremsstrahlung"))) {
            GRACE_INFO("Weakhub table carries pairs and bremsstrahlung: the analytic pair_annihilation "
                       "and bremsstrahlung extras are switched off");
            config_parser::get()["m1"]["eas"]["pair_annihilation"]=false;
            config_parser::get()["m1"]["eas"]["bremsstrahlung"]=false;
        }
        if(content==weakhub_content_t::thermal_decay && decay)
            ERROR("m1.eas.muon_decay: the Weakhub table already carries inverse lepton decay "
                  "(weakhub_table_content thermal+decay); set muon_decay: false");
    }
    if(!decay || !standard) return;
#if GRACE_M1_NU_SPECIES < 5 || !defined(GRACE_ENABLE_MUONS)
    ERROR("m1.eas.muon_decay needs five neutrino species with GRACE_ENABLE_MUONS (set muon_decay: false)");
#endif
#ifdef GRACE_FREEZE_HYDRO
    ERROR("m1.eas.muon_decay with GRACE_FREEZE_HYDRO would emit from a frozen Ymu forever (set muon_decay: false)");
#endif
    if(get_param<std::string>("eos","eos_type")!="leptonic")
        ERROR("m1.eas.muon_decay needs eos.eos_type: leptonic (the muon fraction and its chemical potential)");
    if(!get_param<bool>("eos","leptonic","dilute_muon_suppression"))
        ERROR("m1.eas.muon_decay with pair_treatment: standard needs eos.leptonic.dilute_muon_suppression: true "
              "(the decaying population is the muon excess over the EOS's dilute band)");
}

} // namespace grace

#endif /* GRACE_PHYSICS_EAS_KINDS_HH */
