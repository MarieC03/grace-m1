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
 *          - tabulated_eos_t (npe): Ye from the hot beta equilibrium at
 *            (rho, T_id), as FIL's hot slice: mu_e + mu_p - mu_n - Qnp = 0.
 *          - hybrid, ideal gas: no chemical potentials, so the composition is
 *            taken from the cold slice (`ye_cold__press` / `ymu_cold__press`,
 *            always available via the CRTP base) and only T is raised to T_id.
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
#include <grace/physics/eos/tabulated_eos.hh>    // tabulated_eos_t (npe hot beta-eq)
#include <grace/utils/rootfinding.hh>

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
        , _rho_cut( grace::get_param<double>("grmhd","tov","rho_cut") )
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

        // The whole TOV solution goes on the grid (sol[0] = 0 outside R_iso); rho_cut > 0
        // writes everything at or below it as atmosphere (FIL's ID_converter rho_threshold).
        double const rho_tov = ( sol[0] > 0.0 ) ? this->_eos.rho__press_cold(sol[0], err) : 0.0 ;
        if ( rho_tov > Kokkos::fmax(_rho_cut, 0.0) ) {
            // ---- inside the star ----
            // STRUCTURE: density from the cold-slice pressure profile.
            id.rho  = rho_tov ;
            // THERMAL state: T_id above rho_cut.  Without a cut, T_id is tapered to temp_atm over
            // the outer 5% of R_iso (at T_id the tenuous surface overruns the c2p eps_maximum).
            if ( _rho_cut > 0.0 ) {
                id.temp = _T_id ;
            } else {
                constexpr double taper_frac = 0.95 ;
                double w = (rL / this->_R_iso - taper_frac) / (1.0 - taper_frac) ;
                w = Kokkos::fmin(1.0, Kokkos::fmax(0.0, w)) ;   // 0 bulk -> 1 surface
                id.temp = _T_id + (temp_atm - _T_id) * w ;
            }
            // Composition is EOS-aware (see file header): the leptonic EOS and
            // the 3D table solve a hot beta-equilibrium so the composition
            // shifts with T; the other EOSs take it from the cold slice.
            // NB: beta-eq uses the (tapered) id.temp so the envelope stays
            // self-consistent with its local temperature.
            if constexpr (std::is_same_v<eos_t, leptonic_eos_4d_t>) {
                double ye = 0.1, ymu = 0.0 ;
                this->_eos.betaeq_ye_ymu__rho_temp(id.rho, id.temp, ye, ymu) ;
                // At the hot, low-density surface the beta-eq can give
                // ye + ymu > yemax (thermal muons + proton-rich).  No charge-
                // budget fix here: id.ye/id.ymu are passed by reference into
                // press_eps_csnd2_entropy__temp_rho_ye_ymu below, whose
                // limit_ymu(ye, ymu, err) applies the identical cap and writes
                // the corrected Ymu straight back into id.ymu -- so the stored
                // composition is guaranteed to match what the EOS evaluated,
                // by construction rather than by a second, separately
                // maintained copy of the same rule.
                id.ye  = ye ;
                id.ymu = ymu ;
            } else if constexpr (std::is_same_v<eos_t, tabulated_eos_t>) {
                id.ye  = betaeq_ye_npe(id.rho, id.temp) ;
                id.ymu = 0.0 ;
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

    //! npe hot beta-eq Ye of a 3D table, FIL's hot slice (Margherita EOS_Tabulated::press_eps_yle_ymu__beta_eq__rho_temp);
    //! no root in the table's Ye range -> the end with the smaller residual.
    double GRACE_HOST_DEVICE
    betaeq_ye_npe(double rho, double temp) const
    {
        constexpr double Qnp = 1.29333236 ;   // MeV, as the leptonic beta-eq (FIL 1.293333)
        auto f = [&] (double ye) {
            double r = rho, t = temp, y = ye, ymu = 0., mumu, mup, mun, Xa, Xh, Xn, Xp, Abar, Zbar ;
            eos_err_t err ;
            double const mue = this->_eos.mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu(
                mumu, mup, mun, Xa, Xh, Xn, Xp, Abar, Zbar, t, r, y, ymu, err) ;
            return mue + mup - mun - Qnp ;
        } ;
        double const lo = this->_eos.get_c2p_ye_min(), hi = this->_eos.get_c2p_ye_max() ;
        double const flo = f(lo), fhi = f(hi) ;
        if ( flo * fhi > 0.0 ) return ( Kokkos::fabs(flo) < Kokkos::fabs(fhi) ) ? lo : hi ;
        return utils::brent_bracketed(f, lo, hi, flo, fhi, 1e-14) ;
    }

    double _rho_cut ; //!< atmosphere at or below this rho (FIL's rho_threshold) [code]; <=0 = no cut, radius taper
    double _T_id ;   //!< Fixed thermal-state temperature [MeV]
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_HOT_TOV_HH */
