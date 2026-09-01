/**
 * @file m1_trigger.hh
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

#ifndef GRACE_PHYSICS_M1_TRIGGER_HH
#define GRACE_PHYSICS_M1_TRIGGER_HH

#include <grace_config.h>

#include <cstddef>
#include <limits>
#include <string>

namespace grace {

/**
 * \defgroup m1_trigger M1 activation trigger
 *
 * Neutrinos only matter from merger onward, but M1 is ~64% of a timestep, so a
 * BNS inspiral otherwise pays in full for transport that does nothing.  This
 * keeps M1 compiled in but IDLE until the compact objects come close enough,
 * mirroring FIL's `Trigger` thorn steering `frankfurt_m1::evolve_radiation`
 * off `BNSTrackerGen::bns_sep_tot`.
 *
 * The condition is evaluated host-side, once per co_tracker update, and the
 * result is used to skip whole Kokkos launches -- never to branch inside a
 * kernel.  It is a pure compute gate: M1 does not constrain dt in Z4 runs
 * (find_stable_timestep uses cmax = 1 there), so an idle M1 cannot change the
 * inspiral evolution.
 *
 * The flag is CONSISTENT ACROSS RANKS without extra communication: it derives
 * from co_tracker's separation, which is itself computed from globally
 * allreduced centre-of-mass positions.
 */

//**************************************************************************************************
/**
 * @brief Parfile-configured trigger settings (m1.trigger).
 * \ingroup m1_trigger
 */
struct m1_trigger_params_t {
    //! Arm the trigger.  False => M1 active from t=0 (the default, and the
    //! behaviour of every parfile written before the trigger existed).
    bool   enabled{false} ;
    //! Coordinate separation [code units] below which M1 activates.  Chosen
    //! independently of (and normally larger than) co_tracker.merge_distance,
    //! so neutrinos switch on during late inspiral rather than only once the
    //! cores touch.
    double separation{0.0} ;
};

#ifdef GRACE_ENABLE_M1

namespace detail {
//! Latched activation state.  Defaults to ACTIVE so that a build without a
//! trigger in the parfile behaves exactly as before.
inline bool   _m1_active               = true ;
//! Iteration at which the trigger fired (only meaningful once latched).
inline size_t _m1_activation_iteration = 0 ;
//! Set when the latch was restored from a checkpoint, so the startup check
//! knows not to reset an already-fired trigger back to idle.
inline bool   _m1_latch_restored       = false ;
} // namespace detail

//**************************************************************************************************
/**
 * @brief Is the M1 sector currently being evolved?
 * \ingroup m1_trigger
 *
 * Host-only.  Call this to decide whether to LAUNCH M1 work; do not call it
 * from inside a kernel.
 */
inline bool m1_is_active() { return detail::_m1_active ; }

//**************************************************************************************************
/**
 * @brief Iteration at which the trigger fired (0 if it never did / was off).
 * \ingroup m1_trigger
 */
inline size_t m1_activation_iteration() { return detail::_m1_activation_iteration ; }

//**************************************************************************************************
/**
 * @brief Force the latch (checkpoint restore).
 * \ingroup m1_trigger
 */
inline void m1_set_active(bool active, size_t activation_iteration = 0) {
    detail::_m1_active               = active ;
    detail::_m1_activation_iteration = activation_iteration ;
}

//**************************************************************************************************
/**
 * @brief Restore the latch from a checkpoint.
 * \ingroup m1_trigger
 *
 * Distinct from m1_set_active so m1_trigger_startup_check(), which runs after
 * the restore, can tell "already fired in a previous job" from "fresh start".
 */
inline void m1_restore_latch(bool active, size_t activation_iteration) {
    m1_set_active(active, activation_iteration) ;
    detail::_m1_latch_restored = true ;
}

//**************************************************************************************************
/**
 * @brief Read the m1.trigger block from the parfile.
 * \ingroup m1_trigger
 */
m1_trigger_params_t get_m1_trigger_params() ;

//**************************************************************************************************
/**
 * @brief Validate the trigger configuration and set the initial state.
 * \ingroup m1_trigger
 *
 * Called once at startup, AFTER co_tracker::initialize().  Errors out if the
 * trigger is armed but the tracker cannot supply a separation -- otherwise the
 * condition would never be evaluated and M1 would stay silently off for the
 * whole run, which looks like a successful job but is physically wrong.
 */
void m1_trigger_startup_check() ;

//**************************************************************************************************
/**
 * @brief Evaluate the trigger against the current co_tracker separation.
 * \ingroup m1_trigger
 *
 * One-way: once active, never deactivated.  After merger the "separation" of a
 * single remnant is meaningless, which is why co_tracker latches its own
 * `merged` flag the same way.
 */
void m1_update_trigger() ;

#else // !GRACE_ENABLE_M1

//! Without M1 compiled in there is nothing to gate.
inline constexpr bool   m1_is_active() { return false ; }
inline constexpr size_t m1_activation_iteration() { return 0 ; }
inline void m1_set_active(bool, size_t = 0) {}
inline void m1_restore_latch(bool, size_t) {}
inline void m1_trigger_startup_check() {}
inline void m1_update_trigger() {}

#endif // GRACE_ENABLE_M1

//**************************************************************************************************
/**
 * @brief Is this output variable part of the M1 sector, i.e. meaningless while
 *        the trigger has not fired?
 * \ingroup m1_trigger
 *
 * Used to drop the M1 datasets from cell output during the inspiral.  Each
 * variable is its own named HDF5 dataset, so the files stay self-describing --
 * M1 is simply absent from pre-activation output.
 *
 * Deliberately does NOT gate the diagnostics group (`eta_nu*`, `mu_*`,
 * `mu_delta_*`, `X_*`, `Abar`/`Zbar`, `beta_eq_tscale`): those keep being
 * written from the fugacity state while M1 is idle, which is the whole point
 * of the fugacity-only EAS mode.  Note the near-collision between the gated
 * rate field `eta1` and the diagnostic `eta_nu1`.
 */
inline bool is_m1_gated_output_name(std::string const& name) {
    auto starts = [&name](char const* pre) {
        return name.rfind(pre, 0) == 0 ;
    } ;
    // Radiation fields and per-species optical depths.
    if ( starts("Erad") || starts("Nrad") || starts("Frad") || starts("optd") ) return true ;
    // Rate fields.
    if ( starts("kappa_a") || starts("kappa_s") || starts("kappa_n") ) return true ;
    if ( starts("eta") ) return !starts("eta_nu") ;   // eta1..5 / eta_n1..5 / eta*_ph
    return false ;
}

} // namespace grace

#endif // GRACE_PHYSICS_M1_TRIGGER_HH
