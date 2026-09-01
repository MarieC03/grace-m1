/**
 * @file m1_trigger.cpp
 * @brief Runtime activation trigger for the M1 radiation sector.
 * @date 2026-08-30
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

#include <grace_config.h>

#include <cmath>

#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>
#include <grace/system/grace_system.hh>
#include <grace/system/runtime_functions.hh>

#include <grace/IO/diagnostics/co_tracker.hh>
#include <grace/physics/m1_trigger.hh>

namespace grace {

//**************************************************************************************************
m1_trigger_params_t get_m1_trigger_params() {
    m1_trigger_params_t p ;
    p.enabled    = grace::get_param<bool>  ("m1", "trigger", "enabled") ;
    p.separation = grace::get_param<double>("m1", "trigger", "separation") ;
    return p ;
}

//**************************************************************************************************
void m1_trigger_startup_check() {
    auto const p = get_m1_trigger_params() ;

    if ( !p.enabled ) {
        // Unarmed: M1 runs from t=0, exactly as before the trigger existed.
        // Deliberately overrides any latch restored from a checkpoint -- a
        // parfile without a trigger must never inherit an idle M1 from a file
        // written by a different (triggered) run.
        m1_set_active(true) ;
        return ;
    }

    // Armed but unusable is the dangerous case: cur_distance would never be
    // updated, M1 would stay off for the whole run, and the job would look
    // like it succeeded.  Refuse to start instead.
    auto const& tracker = grace::co_tracker::get() ;
    if ( !tracker.is_active() ) {
        ERROR("m1.trigger.enabled is true but the compact-object tracker is inactive "
              "(co_tracker.n_cos = " << tracker.get_n_cos() << ", co_tracker.update_every <= 0).  "
              "The trigger reads its separation from that tracker, so M1 would never activate.  "
              "Configure co_tracker with two objects and a positive update_every, or set "
              "m1.trigger.enabled = false.") ;
    }
    if ( tracker.get_n_cos() < 2 ) {
        ERROR("m1.trigger.enabled is true but co_tracker.n_cos = " << tracker.get_n_cos()
              << ".  A separation needs two tracked objects, so M1 would never activate.  "
                 "Set m1.trigger.enabled = false for single-object runs.") ;
    }

    if ( detail::_m1_latch_restored ) {
        // Restarted: keep whatever the checkpoint said.  Resetting to idle here
        // would silently un-activate M1 for a job resumed after merger.
        GRACE_INFO("M1 activation trigger armed; latch restored from checkpoint "
                   "(active = {}).", m1_is_active()) ;
        return ;
    }

    // Fresh start, armed and usable: start idle and let m1_update_trigger() decide.
    m1_set_active(false) ;
    GRACE_INFO("M1 activation trigger armed: M1 stays idle until the compact-object "
               "separation drops below {} (co_tracker.merge_distance = {} is separate).",
               p.separation, grace::get_param<double>("co_tracker", "merge_distance")) ;
}

//**************************************************************************************************
void m1_update_trigger() {
    // Latched: once active, never re-evaluated.  After merger the separation of
    // a single remnant is meaningless.
    if ( m1_is_active() ) return ;

    auto const p = get_m1_trigger_params() ;
    if ( !p.enabled ) return ;

    auto const& tracker = grace::co_tracker::get() ;
    double const sep = tracker.get_distance() ;

    // A non-finite separation means the tracker is not locked on -- e.g. an
    // ns_tracker whose search sphere encloses no matter divides 0/0 in its
    // centre-of-mass.  `nan < threshold` is false, so this would otherwise be
    // an invisible "M1 never activates" run.  Warn once and keep checking:
    // early transients before the tracker settles are legitimate, a permanent
    // NaN is a misconfigured tracker.
    if ( !std::isfinite(sep) ) {
        static bool warned = false ;
        if ( !warned ) {
            warned = true ;
            GRACE_WARN("M1 trigger: the compact-object tracker reports a non-finite "
                       "separation ({}), so M1 cannot activate.  If this persists the "
                       "tracker is misconfigured -- check that each co_tracker.cos entry "
                       "has a radius enclosing actual matter.", sep) ;
        }
        return ;
    }

    // The tracker only refreshes every co_tracker.update_every iterations; in
    // between, get_distance() returns the last computed value, which is the
    // right thing to test.
    if ( sep < p.separation ) {
        size_t const it = grace::get_iteration() ;
        m1_set_active(true, it) ;
        GRACE_INFO("M1 ACTIVATED at iteration {} (t = {}): compact-object separation {} "
                   "dropped below m1.trigger.separation = {}.",
                   it, grace::get_simulation_time(), sep, p.separation) ;
    }
}

} // namespace grace
