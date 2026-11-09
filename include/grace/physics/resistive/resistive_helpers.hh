/**
 * Not sure about the policy on these headers. Will ask.
 * */
#ifndef GRACE_PHYSICS_RES_HELPERS_HH
#define GRACE_PHYSICS_RES_HELPERS_HH

#include <grace_config.h>
#include <grace/config/config_parser.hh>
/**
 * @brief Resistive parameters
 *
 * Follows Tenet2026 v2.
 * */
struct resistive_params_t {
    double eta0;     //!< Peak Resistivity
    double eta_f;    //!< Lowest resistivity value
    double rho_th;   //!< Critical transition density (see Tenet2026 v2 fig. 4)
    double lambda;   //!< Logaritmic width for the resistivity profile
} ;

/** @brief Get resistive parameters from the parameter store.
 */
GRACE_ALWAYS_INLINE
resistive_params_t get_resistive_params()
{
    resistive_params_t resistive_params;

    resistive_params.eta0   = grace::get_param<double>("resistive", "eta0");
    resistive_params.eta_f  = grace::get_param<double>("resistive", "eta_f");
    resistive_params.rho_th = grace::get_param<double>("resistive", "rho_th");
    resistive_params.lambda = grace::get_param<double>("resistive", "lambda");

    return resistive_params;
}
#endif
