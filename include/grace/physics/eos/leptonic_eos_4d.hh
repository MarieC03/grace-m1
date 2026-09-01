/**
 * @file leptonic_eos_4d.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @author Keneth Miler (miler@itp.uni-frankfurt.de)
 * @brief  GRACE wrapper around the Margherita-style additive leptonic EOS.
 *
 *         The total thermodynamic state is built additively from three
 *         independent 3D tables:
 *
 *             baryon     (rho, T, yp)               -- existing GRACE tabulated_eos
 *                        yp = Y_le + Y_mu for a no-electron table, else Y_le
 *             electronic (rho, T, Y_le)             -- from the leptonic HDF5
 *             muonic     (rho, T, Y_mu)             -- from the leptonic HDF5
 *
 *         where yp is the proton (charge) fraction (charge neutrality)
 *         clamped to the baryon-table Y_e axis bounds.
 *
 *         `ye` ALWAYS means the true electron fraction in this class; it is
 *         never reinterpreted as yp.  The baryon charge axis is always
 *         yp = ye + ymu (see raw_yp), independent of add_ele_contribution.
 *
 *         Semantics:
 *           - add_ele_contribution = true (electron-free baryon table, the
 *             configuration every shipped parfile uses): the electronic
 *             table's P / eps / entropy ARE added into the totals, and mu_e
 *             comes from that table at Y_le = ye.
 *           - add_ele_contribution = false (with-electron baryon table): the
 *             electronic table is NOT used at all -- not for P/eps/entropy
 *             and not for mu_e, since the baryon table's own TABMUE is
 *             already self-consistent with its mu_p/mu_n.  Note the baryon
 *             table's baked-in electron gas is then evaluated at yp rather
 *             than ye, i.e. over-counted by Y_mu; that is the deliberate
 *             trade for keeping the nucleon sector on the correct charge
 *             fraction.
 *           - yp is clamped to the baryon table's [yemin, yemax] before
 *             every baryon lookup (charge neutrality + Y_e axis range).
 *           - when yp >= yemax, mu_e is taken from the baryon table (the
 *             leptonic mu_e is unreliable in that corner) and mu_mu falls
 *             back to the muon rest mass -- which then goes through the
 *             dilute-Ymu ramp like any other raw value (see mumu_core).
 *           - sound speed comes from the baryon table alone.
 *           - the Y_le axis is linear; the Y_mu axis is log-spaced
 *             (ymu_table in the HDF5 stores log(Ymu)).
 *
 *         The class follows the existing GRACE CRTP / eos_base_t pattern.
 *
 * @date   2026-05-15
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE).
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas.
 * Copyright (C) 2026 Marie Cassing, Keneth Miler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_PHYSICS_EOS_LEPTONIC_4D_HH
#define GRACE_PHYSICS_EOS_LEPTONIC_4D_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/bitset.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/tabulated_eos.hh>   // tabeos_linterp_t, cold_eos_linterp_t

#include <Kokkos_Core.hpp>

namespace grace {

class leptonic_eos_4d_t
    : public eos_base_t<leptonic_eos_4d_t>
{
    using err_t  = eos_err_t ;
    using base_t = eos_base_t<leptonic_eos_4d_t> ;

  public:

    // ----------------------------------------------------------
    //  Baryon table variable indices (same as tabulated_eos_t).
    // ----------------------------------------------------------
    enum TEOS_VIDX : int {
        TABPRESS = 0,
        TABEPS,
        TABCSND2,
        TABENTROPY,
        TABMUE,
        TABMUP,
        TABMUN,
        TABXA,
        TABXH,
        TABXN,
        TABXP,
        TABABAR,
        TABZBAR,
        N_TAB_VARS_BARYON
    } ;

    // ----------------------------------------------------------
    //  Electronic table (9 vars, Margherita EELE ordering).
    // ----------------------------------------------------------
    enum ELE_VIDX : int {
        TABMUELE = 0,
        TABYLE_MINUS,
        TABYLE_PLUS,
        TABPRESS_E_MINUS,
        TABPRESS_E_PLUS,
        TABEPS_E_MINUS,
        TABEPS_E_PLUS,
        TABS_E_MINUS,
        TABS_E_PLUS,
        N_TAB_VARS_ELE
    } ;

    // ----------------------------------------------------------
    //  Muonic table (9 vars, Margherita EMUON ordering).
    // ----------------------------------------------------------
    enum MUON_VIDX : int {
        TABMUMU = 0,
        TABYMU_MINUS,
        TABYMU_PLUS,
        TABPRESS_MU_MINUS,
        TABPRESS_MU_PLUS,
        TABEPS_MU_MINUS,
        TABEPS_MU_PLUS,
        TABS_MU_MINUS,
        TABS_MU_PLUS,
        N_TAB_VARS_MUON
    } ;

    // ----------------------------------------------------------
    //  Cold-slice file layout: 8 columns
    //   log(rho)  temp  ye_cold  ymu_cold  press  eps  cs2  entropy
    // ----------------------------------------------------------
    enum COLD_VIDX : int {
        CTABTEMP = 0,
        CTABYE,
        CTABYMU,
        CTABPRESS,
        CTABEPS,
        CTABCSND2,
        CTABENTROPY,
        N_CTAB_VARS
    } ;

    leptonic_eos_4d_t() = default ;

    /**
     * @brief Construct from pre-built interpolators.
     *
     * The baryon-table 4D view, axes and energy_shift come from the
     * GRACE read_eos_table() pipeline; the lepton views and Y_e/Y_mu
     * axes come from the leptonic HDF5.  rho and T axes are *always*
     * those of the baryon table (the reader asserts the leptonic file
     * lives on the same (rho,T) grid).
     */
    leptonic_eos_4d_t(
        // baryon table (shape and layout of tabulated_eos_t::tables)
        Kokkos::View<double ****, grace::default_space> tab_baryon,
        Kokkos::View<double *,    grace::default_space> bar_logrho,
        Kokkos::View<double *,    grace::default_space> bar_logT,
        Kokkos::View<double *,    grace::default_space> bar_ye,
        // electronic table  (axis = linear Y_le)
        Kokkos::View<double ****, grace::default_space> tab_ele,
        Kokkos::View<double *,    grace::default_space> ele_yle,
        // muonic table      (axis = log Y_mu, as stored in HDF5 ymu_table)
        Kokkos::View<double ****, grace::default_space> tab_muon,
        Kokkos::View<double *,    grace::default_space> mu_ymu,
        // cold slice  [nrho, N_CTAB_VARS]
        Kokkos::View<double **,   grace::default_space> cold_tab,
        Kokkos::View<double *,    grace::default_space> cold_logrho,
        // thermodynamic ranges
        double rhomax,   double rhomin,
        double tempmax,  double tempmin,
        double yemax,    double yemin,
        double ymumax,   double ymumin,
        double baryon_mass,
        double energy_shift_,
        double c2p_epsmin, double c2p_epsmax,
        double c2p_hmin,   double c2p_hmax,
        double c2p_temp_atm,
        double c2p_ye_atm,
        double c2p_ymu_atm,
        bool   atmo_is_beta_eq,
        // When false (default): the baryon table already includes the
        // electron contribution; the electronic table is used only for
        // mu_e.  Set true only if the baryon table was generated
        // without electrons and the electronic table must be added.
        bool   add_ele_contribution_ = false
    )
    : base_t( rhomax, rhomin,
              tempmax, tempmin,
              yemax, yemin,
              ymumax, ymumin,
              baryon_mass,
              c2p_epsmin, c2p_epsmax,
              c2p_hmin,   c2p_hmax,
              c2p_temp_atm,
              c2p_ye_atm,
              c2p_ymu_atm,
              atmo_is_beta_eq )
    , baryon_table(tab_baryon, bar_logrho, bar_logT, bar_ye)
    , ele_table   (tab_ele,    bar_logrho, bar_logT, ele_yle)
    , muon_table  (tab_muon,   bar_logrho, bar_logT, mu_ymu)
    , cold_table  (cold_tab,   cold_logrho)
    , nrho(bar_logrho.size()), nT(bar_logT.size())
    , nye(ele_yle.size()),     nymu(mu_ymu.size())
    , energy_shift(energy_shift_)
    , add_ele_contribution(add_ele_contribution_)
    {
        lrhomin  = bar_logrho[0] ; lrhomax  = bar_logrho[bar_logrho.size()-1] ;
        ltempmin = bar_logT[0]   ; ltempmax = bar_logT[bar_logT.size()-1]    ;
    }

    // ===========================================================
    //  CRTP _impl methods
    // ===========================================================

    double GRACE_HOST_DEVICE
    press__eps_rho_ye_ymu_impl(double& eps, double& rho,
                               double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye_ymu_impl(double& temp, double& eps, double& rho,
                                    double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press__temp_rho_ye_ymu_impl(double& temp, double& rho,
                                double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        return total_press(Kokkos::log(rho), Kokkos::log(temp), ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    eps__temp_rho_ye_ymu_impl(double& temp, double& rho,
                              double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        return total_eps(Kokkos::log(rho), Kokkos::log(temp), ye, ymu) ;
    }

    void GRACE_HOST_DEVICE
    eps_range__rho_ye_ymu_impl(double& epsmin, double& epsmax,
                               double& rho, double& ye, double& ymu,
                               err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho = Kokkos::log(rho) ;
        epsmin = total_eps(lrho, ltempmin, ye, ymu) ;
        epsmax = total_eps(lrho, ltempmax, ye, ymu) ;
    }

    void GRACE_HOST_DEVICE
    entropy_range__rho_ye_ymu_impl(double& smin, double& smax,
                                   double& rho, double& ye, double& ymu,
                                   err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho = Kokkos::log(rho) ;
        smin = total_entropy(lrho, ltempmin, ye, ymu) ;
        smax = total_entropy(lrho, ltempmax, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye_ymu_impl( double& h, double& csnd2, double& eps,
                                        double& rho, double& ye, double& ymu,
                                        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        double const press = total_press(lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye_ymu_impl( double& h, double& csnd2, double& temp,
                                         double& rho, double& ye, double& ymu,
                                         err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        double const press = total_press(lrho, ltemp, ye, ymu) ;
        double const eps   = total_eps  (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_ymu_impl(double& eps, double& csnd2,
                                          double& temp, double& rho,
                                          double& ye, double& ymu,
                                          err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        eps   = total_eps  (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& entropy,
        double& eps, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        double const press = total_press(lrho, ltemp, ye, ymu) ;
        csnd2   = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        h = 1. + eps + press / rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& csnd2, double& entropy, double& temp,
        double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        csnd2   = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        return total_eps(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& eps, double& csnd2, double& entropy,
        double& temp, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        eps     = total_eps    (lrho, ltemp, ye, ymu) ;
        csnd2   = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& eps,
        double& entropy, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        limit_entropy_rho_ye_ymu(entropy, rho, ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__entropy_lrho_ye_ymu(entropy, lrho, ye, ymu) ;
        temp = Kokkos::exp(ltemp) ;
        double const press = total_press   (lrho, ltemp, ye, ymu) ;
        eps   = total_eps    (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press / rho ;
        return press ;
    }

    /**
     * @brief press -> (eps, h, csnd2, T, s) inversion at fixed (rho, ye, ymu).
     *
     * Needed by the vacuum/atmosphere initial-data kernels, which specify
     * the background thermodynamic state through the pressure.  The total
     * pressure is monotone in T over the table range at fixed composition,
     * so a single Brent in log T suffices (same pattern as the eps and
     * entropy inversions below).  press is clamped to the reachable
     * [P(Tmin), P(Tmax)] range, mirroring ltemp__eps_lrho_ye_ymu.
     */
    double GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& entropy,
        double& press, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__press_lrho_ye_ymu(press, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        double const eps = total_eps(lrho, ltemp, ye, ymu) ;
        csnd2   = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        h = 1. + eps + press / rho ;
        return eps ;
    }

    /**
     * @brief Chemical potentials and composition.
     *
     * mu_p, mu_n, X_*, Abar, Zbar are taken from the baryon table at
     * yp = table_yp(ye, ymu) (see there).
     * mu_e comes from the electronic table at Y_le = ye when
     * add_ele_contribution is true and yp has not saturated; otherwise from
     * the baryon table's own TABMUE -- either because the with-electron
     * baryon table already carries a self-consistent mu_e, or because the
     * leptonic mu_e becomes unreliable at yp >= yemax (Margherita's
     * convention).
     */
    double GRACE_HOST_DEVICE
    mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu_impl(
        double &mumu, double& mup, double& mun, double& Xa, double& Xh,
        double& Xn,  double& Xp,  double& Abar, double& Zbar,
        double& temp, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        double const yp    = table_yp(ye, ymu) ;

        mup  = baryon_table.interp(lrho,ltemp,yp,TABMUP)  ;
        mun  = baryon_table.interp(lrho,ltemp,yp,TABMUN)  ;
        Xa   = baryon_table.interp(lrho,ltemp,yp,TABXA)   ;
        Xh   = baryon_table.interp(lrho,ltemp,yp,TABXH)   ;
        Xn   = baryon_table.interp(lrho,ltemp,yp,TABXN)   ;
        Xp   = baryon_table.interp(lrho,ltemp,yp,TABXP)   ;
        Abar = baryon_table.interp(lrho,ltemp,yp,TABABAR) ;
        Zbar = baryon_table.interp(lrho,ltemp,yp,TABZBAR) ;

        mumu = mumu_core(lrho, ltemp, ye, ymu) ;
        // mu_e source: with a with-electron baryon table (add_ele_contribution
        // = false), the baryon table's own TABMUE is already self-consistent
        // with the mu_p/mu_n above -- the standalone ele_table describes an
        // independently-computed electron gas and must NOT be mixed in here
        // (same gating total_press/total_eps/total_entropy already apply to
        // ele_table's P/eps/S).  With a no-electron baryon table
        // (add_ele_contribution = true) the baryon table has no meaningful
        // mu_e of its own, so ele_table is the only valid source -- except in
        // the yp-saturation corner, where the leptonic mu_e itself becomes
        // unreliable (Margherita's convention) and mumu_core already reported
        // mu_mu as its rest mass, so fall back to the baryon table there too.
        if (!add_ele_contribution || raw_yp(ye, ymu) >= this->eos_yemax) {
            return baryon_table.interp(lrho,ltemp,yp,TABMUE)  ;
        }
        return ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABMUELE) ;
    }

    /**
     * @brief Muon chemical potential (dilute-Ymu blocked, see mumu_core).
     *        Not part of the standard CRTP API.  Public entry point for
     *        callers outside this class that need mu_mu on its own.
     */
    double GRACE_HOST_DEVICE
    mumu__temp_rho_ye_ymu(double& temp, double& rho,
                          double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ye, ymu,  err) ;
        limit_temp(temp, err) ;
        return mumu_core(Kokkos::log(rho), Kokkos::log(temp), ye, ymu) ;
    }

    // ----- Cold-slice accessors -----
    double GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABPRESS)) ;
    }
    double GRACE_HOST_DEVICE
    eps_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABEPS))
               - energy_shift ;
    }
    double GRACE_HOST_DEVICE
    ye_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYE) ;
    }
    double GRACE_HOST_DEVICE
    ymu_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYMU) ;
    }
    double GRACE_HOST_DEVICE
    temp_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABTEMP)) ;
    }
    double GRACE_HOST_DEVICE
    entropy_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABENTROPY) ;
    }

    // -----------------------------------------------------------------------
    //  Hot beta-equilibrium: solve for (ye, ymu) at a given (rho, T).
    //  npe-mu beta equilibrium in FIL/Margherita parity (Qnp subtracted),
    //  identical conditions to the cold-table generator solve_muon_beta_eq:
    //     (1)  mu_e(ye) = mu_mu(ymu)                 [inner, Qnp cancels]
    //     (2)  mu_n - mu_p - mu_mu + Qnp = 0          [outer, FIL parity]
    //  Device-callable.  Used by hot_tov initial data to set the THERMAL state
    //  (Ye/Ymu at T) independently of the cold (structure) slice, mirroring
    //  FIL's cold .rns structure + hot Margherita slice.
    // -----------------------------------------------------------------------
    void GRACE_HOST_DEVICE
    betaeq_ye_ymu__rho_temp(double rho, double temp, double& ye, double& ymu) const
    {
        constexpr double Qnp = 1.29333236 ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = Kokkos::log(temp) ;
        double const ye_lo = this->eos_yemin ;
        double const ye_hi = this->eos_yemax ;

        auto mue = [&](double y){ return ele_table.interp(lrho, ltemp, y, ELE_VIDX::TABMUELE) ; } ;
        auto dnp = [&](double yp){   // mu_n - mu_p at clamped charge fraction
            double const yc = Kokkos::fmin(ye_hi, Kokkos::fmax(ye_lo, yp)) ;
            return baryon_table.interp(lrho, ltemp, yc, TABMUN)
                 - baryon_table.interp(lrho, ltemp, yc, TABMUP) ;
        } ;
        // inner solve: ye such that mu_e(ye) = mm_target
        double mm_target = 0.0 ;
        auto inner = [&](double y){ return mue(y) - mm_target ; } ;
        auto find_ye = [&](double mm) -> double {
            mm_target = mm ;
            double const lo = mue(ye_lo), hi = mue(ye_hi) ;
            if ( mm <= Kokkos::fmin(lo,hi) ) return ye_lo ;
            if ( mm >= Kokkos::fmax(lo,hi) ) return ye_hi ;
            return utils::brent(inner, ye_lo, ye_hi, 1e-12) ;
        } ;

        // No muons at this point -> pure npe: mu_e + mu_p - mu_n - Qnp = 0.
        // Shared by the "no bracket" case below and the dilute-Ymu nachcheck:
        // a root landing inside the blocked band [ymumin, Y_mu,0] solved
        // dnp(yel+ymu) = -Qnp (mu_mu pinned to 0 in that band by block_mumu),
        // NOT the true npe condition dnp(yel) = mue(yel) - Qnp -- re-solve
        // properly rather than trust a root found where mu_mu no longer
        // tracked real muon physics.
        auto solve_npe = [&]() {
            ymu = this->eos_ymumin ;
            auto npe = [&](double y){ return mue(y) - dnp(y) - Qnp ; } ;
            double const a = npe(ye_lo), b = npe(ye_hi) ;
            ye = ( a*b < 0.0 ) ? utils::brent(npe, ye_lo, ye_hi, 1e-12)
                               : ( Kokkos::fabs(a) < Kokkos::fabs(b) ? ye_lo : ye_hi ) ;
        } ;

        double const lym_lo = Kokkos::log(this->eos_ymumin) ;
        double const lym_hi = Kokkos::log(this->eos_ymumax) ;
        // mumu_core needs ye (for its own saturation branch), but ye here is
        // an OUTPUT of find_ye(mm) -- chicken-and-egg.  Apply block_mumu
        // directly to the raw table lookup instead; the saturation branch is
        // for the ye+ymu>=yemax corner, unreachable from this solve since ye
        // is bounded by [ye_lo,ye_hi]=[yemin,yemax] and ymu by [ymumin,ymumax]
        // with the outer root only ever queried inside those brackets.
        auto outer = [&](double lym){
            double const ymu_loc = Kokkos::exp(lym) ;
            double const mm_raw  = muon_table.interp(lrho, ltemp, lym, MUON_VIDX::TABMUMU) ;
            double const mm      = block_mumu(ymu_loc, mm_raw) ;
            double const yel     = find_ye(mm) ;
            double const yp      = raw_yp(yel, ymu_loc) ;
            return dnp(yp) - mm + Qnp ;          // mu_n - mu_p - mu_mu + Qnp
        } ;

        if ( outer(lym_lo) * outer(lym_hi) > 0.0 ) {
            solve_npe() ;
            return ;
        }
        double const lym = utils::brent(outer, lym_lo, lym_hi, 1e-12) ;
        ymu = Kokkos::exp(lym) ;
        if ( ymu <= dilute_ymu0_ ) {
            solve_npe() ;
            return ;
        }
        double const mm_raw = muon_table.interp(lrho, ltemp, lym, MUON_VIDX::TABMUMU) ;
        ye = find_ye( block_mumu(ymu, mm_raw) ) ;
    }
    double GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, err_t& err) const
    {
        double const lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double const lrmin = cold_table._logrho(0) ;
        double const lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        return Kokkos::exp(utils::brent(rootfun, lrmin, lrmax, 1e-14)) ;
    }
    double GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, err_t& err) const
    {
        int const n = cold_table._logrho.size() ;
        double const eps_min = Kokkos::exp(cold_table._tables(0,   CTABEPS)) - energy_shift ;
        double const eps_max = Kokkos::exp(cold_table._tables(n-1, CTABEPS)) - energy_shift ;
        double const e_min   = (1.+eps_min) * Kokkos::exp(cold_table._logrho(0))   ;
        double const e_max   = (1.+eps_max) * Kokkos::exp(cold_table._logrho(n-1)) ;
        if (e_cold < e_min) { e_cold = e_min ; err.set(EOS_EPS_TOO_LOW)  ; return Kokkos::exp(cold_table._logrho(0))   ; }
        if (e_cold > e_max) { e_cold = e_max ; err.set(EOS_EPS_TOO_HIGH) ; return Kokkos::exp(cold_table._logrho(n-1)) ; }
        auto rootfun = [this, e_cold] (double lrho) {
            double eps = Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
            return (1.+eps) * Kokkos::exp(lrho) - e_cold ;
        } ;
        return Kokkos::exp(utils::brent(rootfun,
            cold_table._logrho(0), cold_table._logrho(n-1), 1e-14)) ;
    }
    double GRACE_HOST_DEVICE
    energy_cold__press_cold_impl(double& press_cold, err_t& err) const
    {
        double const lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double const lrmin = cold_table._logrho(0) ;
        double const lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        double const lrho  = utils::brent(rootfun, lrmin, lrmax, 1e-14) ;
        double const eps   = Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
        return Kokkos::exp(lrho) * (1.+eps) ;
    }
    double GRACE_HOST_DEVICE
    ye_cold__press_impl(double& press, err_t& err) const {
        double rho = rho__press_cold_impl(press, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYE) ;
    }
    double GRACE_HOST_DEVICE
    ymu_cold__press_impl(double& press, err_t& err) const {
        double rho = rho__press_cold_impl(press, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYMU) ;
    }

    // ===========================================================
    //  Public data members (captured into kernels by value)
    // ===========================================================
    tabeos_linterp_t   baryon_table ;  ///< rho, T, yp (see table_yp)
    tabeos_linterp_t   ele_table    ;  ///< rho, T, Y_le (linear axis)
    tabeos_linterp_t   muon_table   ;  ///< rho, T, log(Y_mu) axis — queries must pass log(ymu)
    cold_eos_linterp_t cold_table   ;

    int nrho, nT, nye, nymu ;
    double energy_shift ;
    double lrhomin, lrhomax ;
    double ltempmin, ltempmax ;
    // If true, add electronic pressure/eps/entropy on top of the
    // baryon table.  Keep false when the baryon table already
    // contains the electron contribution (the usual case for SFHo,
    // DD2, etc.).  Mirrors Margherita's add_ele_contribution flag.
    bool add_ele_contribution = false ;

  private:

    // ----------------------------------------------------------
    //  Clamp a charge fraction into the baryon table's range.
    // ----------------------------------------------------------
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double clamp_yp(double yp_raw) const {
        if (yp_raw < this->eos_yemin) return this->eos_yemin ;
        if (yp_raw > this->eos_yemax) return this->eos_yemax ;
        return yp_raw ;
    }

    // Charge coordinate for the baryon table, before clamping.  ALWAYS
    // yp = ye + ymu: `ye` means the true electron fraction everywhere in this
    // class -- it is never reinterpreted as yp -- and charge neutrality
    // n_p = n_e + n_mu then fixes the baryon table's axis.
    //
    // This is independent of add_ele_contribution, which controls only whether
    // the separate electron table's P/eps/s/mu_e are added on top.  With a
    // with-electron baryon table the consequence is that its baked-in electron
    // gas is evaluated at yp rather than ye, i.e. over-counted by Y_mu; that is
    // the deliberate trade, because it keeps the NUCLEON sector (X_n/X_p, which
    // drive the neutrino rates) on the correct charge fraction.  Use an
    // electron-free baryon table + add_ele_contribution=true to avoid the trade.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    raw_yp(double ye, double ymu) const {
        return ye + ymu ;
    }

    // Dilute-Ymu thresholds (GMUNU muon microphysics note): a numerically
    // unresolved Y_mu at/near the table floor represents physical zero, not
    // real matter.  Named here once so block_mumu and the beta-eq nachcheck
    // in betaeq_ye_ymu__rho_temp cannot drift apart.
    static constexpr double dilute_ymu0_ = 6.0e-4 ;
    static constexpr double dilute_dymu_ = 5.0e-5 ;

    // Ramp the raw tabulated mu_mu smoothly to zero below Y_mu,0 instead of
    // carrying it through: at the Y_mu floor the raw table value sits near
    // the muon rest mass (105.66 MeV -- see [[muon-table-low-ymu-numbers]]),
    // and fed to neutrino microphysics as eta_numu = mu_mu/T that produces an
    // O(100) spurious degeneracy even at T ~ 1 MeV.  Continuous (tanh, not a
    // hard step) so root-finders (beta-eq here, and the M1 rate solves
    // downstream) never see a jump.
    //   mu_mu^used = 0                                    Y_mu <= Y_mu,0
    //              = mu_mu^tab * tanh[(Y_mu-Y_mu,0)/dY_mu]  Y_mu >  Y_mu,0
    //
    // Deliberately blocks ONLY mu_mu -- NOT the muon P/eps/entropy terms in
    // total_press/total_eps/total_entropy, which stay exact: measured that at
    // this same Y_mu floor the muon PAIR population can carry real thermal
    // energy (eps_mu up to ~14 at rho=1e11, T=50 MeV) even while the NET Y_mu
    // sits at the floor -- only the net chemical potential is pathological
    // there, not the thermodynamics.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    block_mumu(double ymu, double mm_raw) const {
        return ( ymu <= dilute_ymu0_ ) ? 0.0
             : mm_raw * Kokkos::tanh((ymu - dilute_ymu0_) / dilute_dymu_) ;
    }

    // Full mu_mu accessor: table lookup + atmosphere/saturation fallback +
    // dilute-Ymu blocking, all in one place so mue_mumu_mup_mun_... and
    // mumu__temp_rho_ye_ymu can never disagree on what "mu_mu" means.
    // lrho/ltemp/ye/ymu are assumed already limited by the caller (this is
    // the hot path -- do not call limit_* again in here).  Not usable by the
    // beta-eq solver: it needs mu_mu before ye is known (chicken-and-egg with
    // the saturation branch below), so that caller applies block_mumu
    // directly to its own raw table lookup instead.
    // The saturation corner supplies a RAW value that then goes through
    // block_mumu exactly like the tabulated one -- it must not short-circuit
    // past the ramp.  A ye-saturated atmosphere cell typically also sits at the
    // Y_mu floor, so the ramp applies there: measured at the recorded halo cell
    // (Ye=0.5, Y_mu=5.0017e-4 vs dilute_ymu0_=6e-4, T=0.209 MeV) the bare rest
    // mass gives eta_numu = 527, the ramp gives 22.8 -- a 23x reduction.
    //
    // NB the residual 22.8 is NOT a muon artifact: with mu_mu blocked it is
    // (mu_p - mu_n - Qnp)/T, and mu_p - mu_n = +6.07 MeV is real nucleon
    // physics at Ye=0.5.  So the +/-5 eta clamp in make_fugacity_state stays
    // load-bearing here (both 527 and 22.8 sit on the rail, and the rates at
    // this particular cell are unchanged by this fix); what this removes is the
    // muon-rest-mass contribution, which is the part that can push eta far
    // enough negative for the Kirchhoff FD denominators to underflow.
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    mumu_core(double lrho, double ltemp, double ye, double ymu) const {
        double const mm_raw = ( raw_yp(ye, ymu) >= this->eos_yemax )
            ? 105.6583755              // atmosphere / saturation corner
            : muon_table.interp(lrho, ltemp, Kokkos::log(ymu), MUON_VIDX::TABMUMU) ;
        return block_mumu(ymu, mm_raw) ;
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    table_yp(double ye, double ymu) const {
        return clamp_yp(raw_yp(ye, ymu)) ;
    }

    // ----------------------------------------------------------
    //  Total quantities: baryon + muon, optionally + electron.
    //  add_ele_contribution = false (default): the baryon table
    //  already includes electrons; the electronic table is used
    //  only for mu_e.  Set true only if the baryon table was built
    //  without electrons.  Mirrors Margherita's add_ele_contribution.
    // ----------------------------------------------------------
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_press(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = table_yp(ye, ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        // The baryon table is loaded with linear_pressure=true (read_leptonic),
        // so TABPRESS is the SIGNED linear pressure — not log(P).  It is
        // negative in the nuclear spinodal (sub-saturation); the (positive)
        // lepton pressures below make the additive total positive.
        double const pb   = baryon_table.interp(lrho, ltemp, yp, TABPRESS) ;
        double const pmm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABPRESS_MU_MINUS) ;
        double const pmp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABPRESS_MU_PLUS)  ;
        double const pe   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABPRESS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABPRESS_E_PLUS)
            : 0.0 ;
        return pb + pmm + pmp + pe ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_eps(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = table_yp(ye, ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        double const eb   = Kokkos::exp(baryon_table.interp(lrho, ltemp, yp, TABEPS)) - energy_shift ;
        double const emm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABEPS_MU_MINUS) ;
        double const emp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABEPS_MU_PLUS)  ;
        double const ee   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABEPS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABEPS_E_PLUS)
            : 0.0 ;
        return eb + emm + emp + ee ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_entropy(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = table_yp(ye, ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        double const sb   = baryon_table.interp(lrho, ltemp, yp, TABENTROPY) ;
        double const smm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABS_MU_MINUS) ;
        double const smp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABS_MU_PLUS)  ;
        double const se   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABS_E_PLUS)
            : 0.0 ;
        return sb + smm + smp + se ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    baryon_csnd2(double lrho, double ltemp, double ye, double ymu) const
    {
        return baryon_table.interp(lrho, ltemp,
                                   table_yp(ye, ymu),
                                   TABCSND2) ;
    }

    // ----------------------------------------------------------
    //  Bounds clamping
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION void limit_rho(double& rho, err_t& err) const {
        if ( rho < this->eos_rhomin ) { rho = (1.+1e-5)*this->eos_rhomin ; err.set(EOS_RHO_TOO_LOW)  ; }
        if ( rho > this->eos_rhomax ) { rho = (1.-1e-5)*this->eos_rhomax ; err.set(EOS_RHO_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_ye(double& ye, err_t& err) const {
        if ( ye < this->eos_yemin ) { ye = this->eos_yemin ; err.set(EOS_YE_TOO_LOW)  ; }
        if ( ye > this->eos_yemax ) { ye = this->eos_yemax ; err.set(EOS_YE_TOO_HIGH) ; }
    }
    // ye must already be limited (every call site runs limit_ye immediately
    // before this).  Besides the axis clamp, this also enforces the charge
    // budget yp = ye + ymu <= eos_yemax: cap ymu to the remaining budget
    // (yemax - ye), floored at ymumin, so yp saturates exactly at yemax while
    // the muon keeps as much of the budget as fits -- instead of dropping
    // straight to its floor (Margherita's fix_ymu_for_too_high_yp does the
    // latter; we deliberately diverge here, see [[leptonic-yp-axis-convention]]).
    //
    // Folded into the single by-reference choke point (rather than applied
    // ad hoc at each use site) so every caller -- c2p, hot_tov, the rates,
    // the beta-eq solver -- observes the same corrected Ymu written back
    // through its own reference, instead of each call site silently
    // reconciling a local copy that the caller never sees.
    //
    // No-op with < 5 species (ymu pinned at its floor, no muon charge to
    // reconcile).  Applies for every table type: yp = ye + ymu unconditionally
    // (see raw_yp), so the muon always consumes part of the axis' budget.
    KOKKOS_INLINE_FUNCTION void limit_ymu(double const ye, double& ymu, err_t& err) const {
        if ( ymu < this->eos_ymumin ) {
            ymu = this->eos_ymumin ;
            #ifdef GRACE_ENABLE_MUONS
            err.set(EOS_YMU_TOO_LOW) ;
            #else
            err.set(EOS_YE_TOO_LOW) ;
            #endif
        }
        if ( ymu > this->eos_ymumax ) {
            ymu = this->eos_ymumax ;
            #ifdef GRACE_ENABLE_MUONS
            err.set(EOS_YMU_TOO_HIGH) ;
            #else
            err.set(EOS_YE_TOO_HIGH) ;
            #endif
        }
        #ifdef GRACE_ENABLE_MUONS
        if ( raw_yp(ye, ymu) > this->eos_yemax ) {
            ymu = Kokkos::fmax(this->eos_yemax - ye, this->eos_ymumin) ;
        }
        #else
        (void) ye ;
        #endif
    }
    KOKKOS_INLINE_FUNCTION void limit_temp(double& temp, err_t& err) const {
        double const tmin = Kokkos::exp(ltempmin) ;
        double const tmax = Kokkos::exp(ltempmax) ;
        if ( temp < tmin ) { temp = tmin ; err.set(EOS_TEMPERATURE_TOO_LOW)  ; } //(1.+1e-2)*tmin
        if ( temp > tmax ) { temp = tmax ; err.set(EOS_TEMPERATURE_TOO_HIGH) ; } //(1.-1e-2)*tmax
    }
    KOKKOS_INLINE_FUNCTION void limit_entropy_rho_ye_ymu(double& entropy,
        double& rho, double& ye, double& ymu, err_t& err) const
    {
        double smin, smax ;
        entropy_range__rho_ye_ymu_impl(smin, smax, rho, ye, ymu, err) ;
        if ( entropy < smin ) { entropy = smin ; err.set(EOS_ENTROPY_TOO_LOW)  ; }
        if ( entropy > smax ) { entropy = smax ; err.set(EOS_ENTROPY_TOO_HIGH) ; }
    }

    // ----------------------------------------------------------
    //  Brent on the additive totals.  eps and entropy are monotone
    //  in T over the table range, so a single Brent in lT works.
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    ltemp__eps_lrho_ye_ymu(double& eps, double lrho,
                           double ye, double ymu, err_t& err) const
    {
        double const eps_lo = total_eps(lrho, ltempmin, ye, ymu) ;
        double const eps_hi = total_eps(lrho, ltempmax, ye, ymu) ;
        if (eps <= eps_lo) { eps = eps_lo ; err.set(EOS_EPS_TOO_LOW)  ; return ltempmin ; }
        if (eps >= eps_hi) { eps = eps_hi ; err.set(EOS_EPS_TOO_HIGH) ; return ltempmax ; }
        auto rootfun = [this, lrho, ye, ymu, eps] (double lt) {
            return total_eps(lrho, lt, ye, ymu) - eps ;
        } ;
        return utils::brent(rootfun, ltempmin, ltempmax, 1e-12) ;
    }

    KOKKOS_INLINE_FUNCTION double
    ltemp__press_lrho_ye_ymu(double& press, double lrho,
                             double ye, double ymu, err_t& err) const
    {
        double const p_lo = total_press(lrho, ltempmin, ye, ymu) ;
        double const p_hi = total_press(lrho, ltempmax, ye, ymu) ;
        if (press <= p_lo) { press = p_lo ; err.set(EOS_PRESS_TOO_LOW)  ; return ltempmin ; }
        if (press >= p_hi) { press = p_hi ; err.set(EOS_PRESS_TOO_HIGH) ; return ltempmax ; }
        auto rootfun = [this, lrho, ye, ymu, press] (double lt) {
            return total_press(lrho, lt, ye, ymu) - press ;
        } ;
        return utils::brent(rootfun, ltempmin, ltempmax, 1e-12) ;
    }

    KOKKOS_INLINE_FUNCTION double
    ltemp__entropy_lrho_ye_ymu(double entropy, double lrho,
                               double ye, double ymu) const
    {
        double const s_lo = total_entropy(lrho, ltempmin, ye, ymu) ;
        double const s_hi = total_entropy(lrho, ltempmax, ye, ymu) ;
        if (entropy <= s_lo) return ltempmin ;
        if (entropy >= s_hi) return ltempmax ;
        auto rootfun = [this, lrho, ye, ymu, entropy] (double lt) {
            return total_entropy(lrho, lt, ye, ymu) - entropy ;
        } ;
        return utils::brent(rootfun, ltempmin, ltempmax, 1e-12) ;
    }

    KOKKOS_INLINE_FUNCTION double
    cold_lpress__press_limited(double& press_cold, err_t& err) const
    {
        int const n = cold_table._logrho.size() ;
        double const p_min = Kokkos::exp(cold_table._tables(0,   CTABPRESS)) ;
        double const p_max = Kokkos::exp(cold_table._tables(n-1, CTABPRESS)) ;
        if (press_cold < p_min) { press_cold = p_min * (1.+1e-10) ; err.set(EOS_RHO_TOO_LOW)  ; }
        if (press_cold > p_max) { press_cold = p_max * (1.-1e-10) ; err.set(EOS_RHO_TOO_HIGH) ; }
        return Kokkos::log(press_cold) ;
    }

} ; // class leptonic_eos_4d_t

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_LEPTONIC_4D_HH */
