/**
 * @file hot_tov.hh
 * @brief Hot-TOV initial data (FIL-style): the hydrostatic STRUCTURE comes from
 *        the EOS cold slice (compact if that slice is generated cold), while the
 *        THERMAL state is set hot -- the temperature is fixed to T_id inside the
 *        star.  This mirrors FIL's RNS + Margherita approach: a cold .rns
 *        structure with a hot (T=20 MeV) temperature slice laid on top.
 *
 *        Composition handling is EOS-aware and works for ANY EOS:
 *          - leptonic_eos_4d_t: (Ye, Ymu) are solved from a hot beta
 *            equilibrium at (rho, T_id) via `betaeq_ye_ymu__rho_temp`, so the
 *            composition shifts with temperature (the muon-capable path).
 *          - every other EOS (tabulated 3D, hybrid, ideal gas): no hot
 *            beta-eq solver exists, so the composition is taken from the cold
 *            slice (`ye_cold__press` / `ymu_cold__press`, always available via
 *            the CRTP base) and only the temperature is raised to T_id.
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE), Copyright (C) 2023-2026 GRACE Contributors.
 * GPLv3 or later; see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_PHYSICS_ID_HOT_TOV_HH
#define GRACE_PHYSICS_ID_HOT_TOV_HH

#include <grace_config.h>
#include <grace/physics/id/tov.hh>   // reuse solve_tov + tov_id_t machinery
#include <grace/physics/eos/leptonic_eos_4d.hh>  // leptonic_eos_4d_t (hot beta-eq dispatch)

#include <type_traits>

namespace grace {

/**
 * @brief Hot-TOV initial-data kernel.
 * \ingroup initial_data
 *
 * Inherits the 1D TOV integration and metric/interpolation helpers from
 * tov_id_t (the structure is built from the cold slice via press_cold__rho).
 * Only the per-cell hydro assignment is overridden: inside the star the
 * temperature is fixed to T_id and (Ye, Ymu) are solved from beta equilibrium
 * at (rho, T_id); pressure and eps are then taken from the full (hot) EOS.
 */
template < typename eos_t >
struct hot_tov_id_t : public tov_id_t<eos_t> {
    using base_t = tov_id_t<eos_t> ;

    hot_tov_id_t(
          eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords
        , atmo_params_t atmo_params
        , double rhoC, double press_floor, double dr, double pert_amp
        , double T_id )
        : base_t(eos, pcoords, atmo_params, rhoC, press_floor, dr, pert_amp)
        , _T_id(T_id)
    {
        GRACE_INFO("Hot-TOV: structure from the cold slice, thermal state "
                   "(temp, Ye, Ymu) set at hot beta-eq with T = {} MeV.", _T_id) ;
    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q) const
    {
        double const x = this->_pcoords(VEC(i,j,k),0,q);
        double const y = this->_pcoords(VEC(i,j,k),1,q);
        #ifdef GRACE_3D
        double const z = this->_pcoords(VEC(i,j,k),2,q);
        #else
        double const z = 0. ;
        #endif
        double const rL = Kokkos::max(Kokkos::sqrt(EXPR(
            math::int_pow<2>(x),
            + math::int_pow<2>(y),
            + math::int_pow<2>(z)
        )),  1e-45) ;

        auto sol = this->get_solution(rL) ;
        auto rs  = this->get_r_schwarzschild(rL) ;

        grmhd_id_t id ;
        eos_err_t err ;

        double ye_atm   = this->_atmo_params.ye_fl  ;
        double ymu_atm  = this->_atmo_params.ymu_fl ;
        double rho_atm  = this->_atmo_params.rho_fl ;
        double temp_atm = this->_atmo_params.temp_fl ;
        double const press_atm = this->_eos.press_cold__rho(rho_atm, err) ;

        if ( sol[0] > 1.001 * press_atm ) {
            // ---- inside the star ----
            // STRUCTURE: density from the cold-slice pressure profile.
            id.rho  = this->_eos.rho__press_cold(sol[0], err) ;
            // THERMAL state: T = T_id in the bulk, smoothly tapered to the
            // atmosphere temperature over the outer (1 - taper_frac) of the
            // stellar radius.  Rationale: at T_id the tenuous surface is
            // radiation/pair dominated -- specific energy eps ~ 1/rho blows
            // past the c2p eps_maximum, so the hot edge floors/atmospheres
            // every step.  Cooling the outer envelope keeps eps bounded and
            // the surface well-behaved, at negligible cost to the (low-mass)
            // edge.  taper_frac = 0.98 -> the final 2% of R_iso.
            constexpr double taper_frac = 0.95 ;
            {
                double w = (rL / this->_R_iso - taper_frac) / (1.0 - taper_frac) ;
                w = Kokkos::fmin(1.0, Kokkos::fmax(0.0, w)) ;   // 0 bulk -> 1 surface
                id.temp = _T_id + (temp_atm - _T_id) * w ;
            }
            // Composition is EOS-aware (see file header): the leptonic EOS
            // solves a hot beta-equilibrium so (Ye, Ymu) shift with T; every
            // other EOS lacks a hot beta-eq solver, so the composition is
            // taken from the cold slice and only the temperature is raised.
            // NB: beta-eq uses the (tapered) id.temp so the envelope stays
            // self-consistent with its local temperature.
            if constexpr (std::is_same_v<eos_t, leptonic_eos_4d_t>) {
                double ye = 0.1, ymu = 0.0 ;
                this->_eos.betaeq_ye_ymu__rho_temp(id.rho, id.temp, ye, ymu) ;
                // Margherita fix_ymu_for_too_high_yp: at the hot, low-density
                // surface the beta-eq can give ye + ymu > yemax (thermal muons
                // + proton-rich).  Drop the muons there so the stored
                // composition matches what the EOS evaluates (total_press etc.
                // apply the same rule), keeping the initial data self-consistent.
                if ( ye + ymu > this->_eos.get_c2p_ye_max() )
                    ymu = this->_eos.get_c2p_ymu_min() ;
                id.ye  = ye ;
                id.ymu = ymu ;
            } else {
                id.ye  = this->_eos.ye_cold__press(sol[0], err) ;
                id.ymu = this->_eos.ymu_cold__press(sol[0], err) ;
            }
            // press/eps/entropy consistent with the HOT state.  NB: this is the
            // deliberate FIL-style cold-structure + hot-state mismatch (the star
            // is not in exact hydrostatic equilibrium with the hot eps; it
            // relaxes during the first evolution steps).
            double dummy ;
            id.press = this->_eos.press_eps_csnd2_entropy__temp_rho_ye_ymu(
                id.eps, dummy, id.entropy, id.temp, id.rho, id.ye, id.ymu, err) ;
            // radial velocity perturbation
            double s[3] = { x/rL, y/rL, z/rL } ;
            id.vx = this->_pert_amp * s[0] * rL ;
            id.vy = this->_pert_amp * s[1] * rL ;
            id.vz = this->_pert_amp * s[2] * rL ;
        } else {
            // ---- atmosphere ----
            id.rho  = rho_atm ;
            id.ye   = ye_atm  ;
            id.ymu  = ymu_atm ;
            id.temp = temp_atm ;
            double dummy ;
            id.press = this->_eos.press_eps_csnd2_entropy__temp_rho_ye_ymu(
                id.eps, dummy, id.entropy, id.temp, id.rho, id.ye, id.ymu, err) ;
            id.vx = id.vy = id.vz = 0.0 ;
        }

        double const nuL = sol[1] ;
        id.bx = id.by = id.bz = 0 ;
        id.alp = Kokkos::exp(nuL) ;
        id.betax = id.betay = id.betaz = 0. ;

        double const psi4 = rL>0 ? SQR((rs/rL)) : 1.0 ;
        id.gxx = id.gyy = id.gzz = psi4 ;
        id.gxy = id.gxz = id.gyz = 0 ;
        id.kxx = id.kxy = id.kxz = id.kyy = id.kyz = id.kzz = 0. ;

        return std::move(id) ;
    }

    double _T_id ;   //!< Fixed thermal-state temperature [MeV]
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_HOT_TOV_HH */
