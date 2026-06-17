/**
 * @file hot_tov.hh
 * @brief Hot-TOV initial data (FIL-style): the hydrostatic STRUCTURE comes from
 *        the EOS cold slice (compact if that slice is generated cold), while the
 *        THERMAL state (temperature, Ye, Ymu) is set to a hot beta equilibrium
 *        at a fixed temperature T_id.  This mirrors FIL's RNS + Margherita
 *        approach: a cold .rns structure with a hot (T=20 MeV) Ye/temperature
 *        slice laid on top.  Only the leptonic EOS supports the hot beta-eq
 *        solve (`betaeq_ye_ymu__rho_temp`).
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE), Copyright (C) 2023-2026 GRACE Contributors.
 * GPLv3 or later; see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_PHYSICS_ID_HOT_TOV_HH
#define GRACE_PHYSICS_ID_HOT_TOV_HH

#include <grace_config.h>
#include <grace/physics/id/tov.hh>   // reuse solve_tov + tov_id_t machinery

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
            // THERMAL state: fixed temperature + hot beta-equilibrium Ye/Ymu.
            id.temp = _T_id ;
            double ye = 0.1, ymu = 0.0 ;
            this->_eos.betaeq_ye_ymu__rho_temp(id.rho, _T_id, ye, ymu) ;
            id.ye  = ye ;
            id.ymu = ymu ;
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
