/**
 * @file leptonic_eos_4d.hh
 * @brief  4D tabulated EOS with muon fraction Y_mu as a fourth independent
 *         variable, consistent with the GRACE EOS framework (CRTP / eos_base_t)
 *         and Kokkos.
 * @date   2025
 *
 * @copyright This file is part of GRACE. GPL-3 or later.
 */

#ifndef GRACE_PHYSICS_EOS_LEPTONIC_4D_HH
#define GRACE_PHYSICS_EOS_LEPTONIC_4D_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/bitset.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/utils/rootfinding.hh>

#include <Kokkos_Core.hpp>



namespace grace {

// ============================================================
//  4-D table interpolator
//  Axes: log(rho), log(T), Y_e, Y_mu  (uniform spacing assumed
//  for every axis, identical layout to tabeos_linterp_t)
// ============================================================
struct tabeos_linterp_4d_t {

    tabeos_linterp_4d_t() = default ;

    tabeos_linterp_4d_t(
        Kokkos::View<double *****> tabs,   ///< [irho,iT,iye,iymu,ivar]
        Kokkos::View<double *>     ar,     ///< log(rho) axis
        Kokkos::View<double *>     at,     ///< log(T)   axis
        Kokkos::View<double *>     ay,     ///< Y_e      axis
        Kokkos::View<double *>     aym     ///< Y_mu     axis
    ) : _logrho(ar), _logT(at), _ye(ay), _ymu(aym), _tables(tabs)
    {
        idr  = 1./(_logrho(1) - _logrho(0)) ;
        idt  = 1./(_logT(1)   - _logT(0))   ;
        idy  = 1./(_ye(1)     - _ye(0))     ;
        idym = 1./(_ymu(1)    - _ymu(0))    ;
    }

    KOKKOS_INLINE_FUNCTION double lrho (int i) const { return _logrho(i) ; }
    KOKKOS_INLINE_FUNCTION double ltemp(int i) const { return _logT(i)   ; }
    KOKKOS_INLINE_FUNCTION double ye   (int i) const { return _ye(i)     ; }
    KOKKOS_INLINE_FUNCTION double ymu  (int i) const { return _ymu(i)    ; }

    KOKKOS_INLINE_FUNCTION
    double interp(double lrho, double ltemp, double ye, double ymu, int idx) const
    {
        std::array<double,1> res ;
        std::array<int,1>    _i{idx} ;
        interp<1>(lrho,ltemp,ye,ymu,_i,res) ;
        return res[0] ;
    }

    template< int N >
    KOKKOS_INLINE_FUNCTION
    void interp(double lrho, double ltemp, double ye, double ymu,
                std::array<int,N> const& idx,
                std::array<double,N>&    res) const
    {
        for( int iv=0; iv<N; ++iv) res[iv] = 0.0 ;
        int ir, it, iy, iym ;
        getidx(lrho,ltemp,ye,ymu, ir,it,iy,iym) ;
        double wr[2], wt[2], wy[2], wym[2] ;
        getw(lrho ,ir ,_logrho,idr ,wr ) ;
        getw(ltemp,it ,_logT  ,idt ,wt ) ;
        getw(ye   ,iy ,_ye    ,idy ,wy ) ;
        getw(ymu  ,iym,_ymu   ,idym,wym) ;
        for( int ii=0; ii<2; ++ii)
        for( int jj=0; jj<2; ++jj)
        for( int kk=0; kk<2; ++kk)
        for( int ll=0; ll<2; ++ll) {
            double w = wr[ii]*wt[jj]*wy[kk]*wym[ll] ;
            for( int iv=0; iv<N; ++iv)
                res[iv] += w * _tables(ir+ii, it+jj, iy+kk, iym+ll, idx[iv]) ;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void getidx(double lr, double lt, double ye_, double ym_,
                int& ir, int& it, int& iy, int& iym) const
    {
        ir  = Kokkos::max(0UL,
              Kokkos::min( static_cast<size_t>((lr  - _logrho(0))*idr ),
                           _logrho.extent(0)-2 )) ;
        it  = Kokkos::max(0UL,
              Kokkos::min( static_cast<size_t>((lt  - _logT(0)  )*idt ),
                           _logT.extent(0)-2   )) ;
        iy  = Kokkos::max(0UL,
              Kokkos::min( static_cast<size_t>((ye_ - _ye(0)    )*idy ),
                           _ye.extent(0)-2     )) ;
        iym = Kokkos::max(0UL,
              Kokkos::min( static_cast<size_t>((ym_ - _ymu(0)   )*idym),
                           _ymu.extent(0)-2    )) ;
    }

    KOKKOS_INLINE_FUNCTION
    void getw(double x, int i,
              Kokkos::View<const double*> ax, double ih, double* w) const
    {
        double lam = (x - ax(i)) * ih ;
        w[0] = 1. - lam ;
        w[1] = lam      ;
    }

    Kokkos::View<const double*> _logrho, _logT, _ye, _ymu ;
    Kokkos::View<double *****>  _tables ;
    double idr, idt, idy, idym ;
} ;

// ============================================================
//  Cold-slice 1-D table  (rho, T_cold, Y_e_cold, Y_mu_cold,
//                         press_cold, eps_cold, cs2_cold, s_cold)
// ============================================================
struct cold_eos_linterp_4d_t {
    cold_eos_linterp_4d_t() = default ;
    cold_eos_linterp_4d_t(
        Kokkos::View<double **> tabs,
        Kokkos::View<double *>  ar
    ) : _logrho(ar), _tables(tabs)
    { idr = 1./(_logrho(1)-_logrho(0)) ; }

    KOKKOS_INLINE_FUNCTION double interp(double lrho, int idx) const {
        int i = Kokkos::max(0UL,
                Kokkos::min( static_cast<size_t>((lrho-_logrho(0))*idr),
                             _logrho.extent(0)-2 )) ;
        double lam = (lrho - _logrho(i)) * idr ;
        return (1.-lam)*_tables(i,idx) + lam*_tables(i+1,idx) ;
    }
    Kokkos::View<const double*> _logrho ;
    Kokkos::View<double **>     _tables  ;
    double idr ;
} ;

// ============================================================
//  leptonic_eos_4d_t
//  Concrete EOS type: 4D tabulated (rho, T, Y_e, Y_mu).
//  Inherits the standard GRACE CRTP interface from eos_base_t.
// ============================================================
class leptonic_eos_4d_t
    : public eos_base_t<leptonic_eos_4d_t>
{
    using err_t  = eos_err_t ;
    using base_t = eos_base_t<leptonic_eos_4d_t> ;

  public:

    // ----------------------------------------------------------
    //  Table variable indices – baryon table
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
    //  Table variable indices – electron lepton table
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
    //  Table variable indices – muon lepton table
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
    //  Cold-slice table indices. File layout is:
    //   col 0  : log(rho)         (stripped into a separate axis view)
    //   col 1+ : the entries indexed by COLD_VIDX below.
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

    // ----------------------------------------------------------
    //  Default / constructors
    // ----------------------------------------------------------
    leptonic_eos_4d_t() = default ;

    leptonic_eos_4d_t(
        // 4-D baryon table  [irho,iT,iye,iymu,ivar]
        Kokkos::View<double *****, grace::default_space> tab_baryon,
        Kokkos::View<double *,     grace::default_space> logrho,
        Kokkos::View<double *,     grace::default_space> logT,
        Kokkos::View<double *,     grace::default_space> ye_ax,
        Kokkos::View<double *,     grace::default_space> ymu_ax,
        // 4-D lepton tables
        Kokkos::View<double *****, grace::default_space> tab_ele,
        Kokkos::View<double *****, grace::default_space> tab_muon,
        // Cold slice  [irho, ivar]
        Kokkos::View<double **,    grace::default_space> cold_tab,
        Kokkos::View<double *,     grace::default_space> cold_logrho,
        // Thermodynamic range parameters
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
        bool   atmo_is_beta_eq
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
    , tables     (tab_baryon, logrho, logT, ye_ax, ymu_ax)
    , tables_ele (tab_ele,    logrho, logT, ye_ax, ymu_ax)
    , tables_muon(tab_muon,   logrho, logT, ye_ax, ymu_ax)
    , cold_table (cold_tab,   cold_logrho)
    , nrho(logrho.size()), nT(logT.size())
    , nye(ye_ax.size()),   nymu(ymu_ax.size())
    , energy_shift(energy_shift_)
    {
        lrhomin  = logrho[0] ; lrhomax  = logrho[logrho.size()-1] ;
        ltempmin = logT[0]   ; ltempmax = logT[logT.size()-1]     ;
    }

    // ===========================================================
    //  CRTP implementation methods (called by eos_base_t)
    //
    //  All names end in "_ymu_impl" to match eos_base_t's CRTP
    //  dispatchers in the muon-extended API.
    // ===========================================================

    // -----------------------------------------------------------
    //  press given eps, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press__eps_rho_ye_ymu_impl(double& eps, double& rho,
                               double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho  = Kokkos::log(rho) ;
        double ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        return press__lrho_ltemp_ye_ymu(lrho, ltemp, ye, ymu) ;
    }

    // -----------------------------------------------------------
    //  press + temp given eps, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye_ymu_impl(double& temp, double& eps, double& rho,
                                    double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho  = Kokkos::log(rho) ;
        double ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        return press__lrho_ltemp_ye_ymu(lrho, ltemp, ye, ymu) ;
    }

    // -----------------------------------------------------------
    //  press given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press__temp_rho_ye_ymu_impl(double& temp, double& rho,
                                double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        return press__lrho_ltemp_ye_ymu(lrho, ltemp, ye, ymu) ;
    }

    // -----------------------------------------------------------
    //  eps given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    eps__temp_rho_ye_ymu_impl(double& temp, double& rho,
                              double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        return Kokkos::exp(tables.interp(lrho,ltemp,ye,ymu,TABEPS)) - energy_shift ;
    }

    // -----------------------------------------------------------
    //  eps range  (min/max over temperature at fixed rho,ye,ymu)
    // -----------------------------------------------------------
    void GRACE_HOST_DEVICE
    eps_range__rho_ye_ymu_impl(double& epsmin, double& epsmax,
                               double& rho, double& ye, double& ymu,
                               err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho = Kokkos::log(rho) ;
        epsmin = Kokkos::exp(tables.interp(lrho,ltempmin,ye,ymu,TABEPS)) - energy_shift ;
        epsmax = Kokkos::exp(tables.interp(lrho,ltempmax,ye,ymu,TABEPS)) - energy_shift ;
    }

    // -----------------------------------------------------------
    //  entropy range
    // -----------------------------------------------------------
    void GRACE_HOST_DEVICE
    entropy_range__rho_ye_ymu_impl(double& smin, double& smax,
                                   double& rho, double& ye, double& ymu,
                                   err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho = Kokkos::log(rho) ;
        smin = tables.interp(lrho, ltempmin, ye, ymu, TABENTROPY) ;
        smax = tables.interp(lrho, ltempmax, ye, ymu, TABENTROPY) ;
    }

    // -----------------------------------------------------------
    //  press + h + cs2  given eps, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye_ymu_impl( double& h, double& csnd2, double& eps,
                                        double& rho, double& ye, double& ymu,
                                        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho  = Kokkos::log(rho) ;
        double ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        double press = press__lrho_ltemp_ye_ymu(lrho, ltemp, ye, ymu) ;
        csnd2 = tables.interp(lrho,ltemp,ye,ymu,TABCSND2) ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    // -----------------------------------------------------------
    //  press + h + cs2 given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye_ymu_impl( double& h, double& csnd2, double& temp,
                                         double& rho, double& ye, double& ymu,
                                         err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        std::array<int,3> _v{ TABPRESS, TABCSND2, TABEPS } ;
        std::array<double,3> res ;
        tables.interp<3>(lrho, ltemp, ye, ymu, _v, res) ;
        double press = Kokkos::exp(res[0]) ;
        double eps   = Kokkos::exp(res[2]) - energy_shift ;
        csnd2 = res[1] ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    // -----------------------------------------------------------
    //  press + eps + cs2 given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_ymu_impl(double& eps, double& csnd2,
                                          double& temp, double& rho,
                                          double& ye, double& ymu,
                                          err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        std::array<int,3> _v{ TABPRESS, TABEPS, TABCSND2 } ;
        std::array<double,3> res ;
        tables.interp<3>(lrho, ltemp, ye, ymu, _v, res) ;
        double press = Kokkos::exp(res[0]) ;
        eps   = Kokkos::exp(res[1]) - energy_shift ;
        csnd2 = res[2] ;
        return press ;
    }

    // -----------------------------------------------------------
    //  press + h + cs2 + temp + entropy  given eps, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& entropy,
        double& eps, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double lrho  = Kokkos::log(rho) ;
        double ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;

        std::array<int,3> _v{ TABPRESS, TABCSND2, TABENTROPY } ;
        std::array<double,3> res ;
        tables.interp<3>(lrho, ltemp, ye, ymu, _v, res) ;

        double press = Kokkos::exp(res[0]) ;
        csnd2   = res[1] ;
        entropy = res[2] ;
        h = 1. + eps + press / rho ;
        return press ;
    }

    // -----------------------------------------------------------
    //  eps + cs2 + entropy given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& csnd2, double& entropy, double& temp,
        double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        std::array<int,3> _v{ TABEPS, TABCSND2, TABENTROPY } ;
        std::array<double,3> res ;
        tables.interp<3>(lrho, ltemp, ye, ymu, _v, res) ;
        csnd2   = res[1] ;
        entropy = res[2] ;
        return Kokkos::exp(res[0]) - energy_shift ;
    }

    // -----------------------------------------------------------
    //  press + eps + cs2 + entropy  given temp, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& eps, double& csnd2, double& entropy,
        double& temp, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        std::array<int,4> _v{ TABPRESS, TABEPS, TABCSND2, TABENTROPY } ;
        std::array<double,4> res ;
        tables.interp<4>(lrho, ltemp, ye, ymu, _v, res) ;
        double press = Kokkos::exp(res[0]) ;
        eps     = Kokkos::exp(res[1]) - energy_shift ;
        csnd2   = res[2] ;
        entropy = res[3] ;
        return press ;
    }

    // -----------------------------------------------------------
    //  press + h + cs2 + temp + eps  given entropy, rho, ye, ymu
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& eps,
        double& entropy, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        limit_entropy_rho_ye_ymu(entropy, rho, ye, ymu, err) ;
        double lrho  = Kokkos::log(rho) ;
        double ltemp = ltemp__entropy_lrho_ye_ymu(entropy, lrho, ye, ymu) ;
        temp = Kokkos::exp(ltemp) ;

        std::array<int,3> _v{ TABPRESS, TABCSND2, TABEPS } ;
        std::array<double,3> res ;
        tables.interp<3>(lrho, ltemp, ye, ymu, _v, res) ;

        double press = Kokkos::exp(res[0]) ;
        csnd2 = res[1] ;
        eps   = Kokkos::exp(res[2]) - energy_shift ;
        h     = 1. + eps + press / rho ;
        return press ;
    }

    // -----------------------------------------------------------
    //  Not implemented for tabulated EOSs.
    // -----------------------------------------------------------
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp,
        double& entropy, double& press, double& rho,
        double& ye, double& ymu, err_t& err) const
    {
        Kokkos::abort("eps_h_csnd2_temp_entropy__press_rho_ye_ymu_impl"
                      " is not implemented for leptonic_eos_4d_t.") ;
        return 0. ;
    }

    // -----------------------------------------------------------
    //  Chemical potentials + composition  (baryon table)
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu_impl(
        double& mup, double& mun, double& Xa, double& Xh,
        double& Xn,  double& Xp,  double& Abar, double& Zbar,
        double& temp, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        double mue = tables.interp(lrho,ltemp,ye,ymu,TABMUE)  ;
        mup  = tables.interp(lrho,ltemp,ye,ymu,TABMUP)  ;
        mun  = tables.interp(lrho,ltemp,ye,ymu,TABMUN)  ;
        Xa   = tables.interp(lrho,ltemp,ye,ymu,TABXA)   ;
        Xh   = tables.interp(lrho,ltemp,ye,ymu,TABXH)   ;
        Xn   = tables.interp(lrho,ltemp,ye,ymu,TABXN)   ;
        Xp   = tables.interp(lrho,ltemp,ye,ymu,TABXP)   ;
        Abar = tables.interp(lrho,ltemp,ye,ymu,TABABAR) ;
        Zbar = tables.interp(lrho,ltemp,ye,ymu,TABZBAR) ;
        return mue ;
    }

    // ===========================================================
    //  Cold-slice accessors  (CRTP impls)
    // ===========================================================
    double GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return Kokkos::exp(cold_table.interp(lrho, CTABPRESS)) ;
    }

    double GRACE_HOST_DEVICE
    eps_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
    }

    double GRACE_HOST_DEVICE
    ye_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return cold_table.interp(lrho, CTABYE) ;
    }

    double GRACE_HOST_DEVICE
    ymu_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return cold_table.interp(lrho, CTABYMU) ;
    }

    double GRACE_HOST_DEVICE
    temp_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return Kokkos::exp(cold_table.interp(lrho, CTABTEMP)) ;
    }

    double GRACE_HOST_DEVICE
    entropy_cold__rho_impl(double& rho, err_t& err) const
    {
        limit_rho(rho, err) ;
        double lrho = Kokkos::log(rho) ;
        return cold_table.interp(lrho, CTABENTROPY) ;
    }

    // -----------------------------------------------------------
    //  rho given cold pressure  (root-find on the cold slice)
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, err_t& err) const
    {
        double lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double lrmin = cold_table._logrho(0) ;
        double lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        return Kokkos::exp(utils::brent(rootfun, lrmin, lrmax, 1e-14)) ;
    }

    // -----------------------------------------------------------
    //  rho given cold energy density  e = rho (1+eps)
    // -----------------------------------------------------------
    double GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, err_t& err) const
    {
        int n = cold_table._logrho.size() ;
        double eps_min = Kokkos::exp(cold_table._tables(0,   CTABEPS)) - energy_shift ;
        double eps_max = Kokkos::exp(cold_table._tables(n-1, CTABEPS)) - energy_shift ;
        double e_min   = (1.+eps_min) * Kokkos::exp(cold_table._logrho(0))   ;
        double e_max   = (1.+eps_max) * Kokkos::exp(cold_table._logrho(n-1)) ;
        if (e_cold < e_min) {
            e_cold = e_min ;
            err.set(EOS_EPS_TOO_LOW) ;
            return Kokkos::exp(cold_table._logrho(0)) ;
        }
        if (e_cold > e_max) {
            e_cold = e_max ;
            err.set(EOS_EPS_TOO_HIGH) ;
            return Kokkos::exp(cold_table._logrho(n-1)) ;
        }
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
        double lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double lrmin = cold_table._logrho(0) ;
        double lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        double lrho  = utils::brent(rootfun, lrmin, lrmax, 1e-14) ;
        double eps   = Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
        return Kokkos::exp(lrho) * (1.+eps) ;
    }

    double GRACE_HOST_DEVICE
    ye_cold__press_impl(double& press, err_t& err) const
    {
        double rho = rho__press_cold_impl(press, err) ;
        double lrho = Kokkos::log(rho) ;
        return cold_table.interp(lrho, CTABYE) ;
    }

    double GRACE_HOST_DEVICE
    ymu_cold__press_impl(double& press, err_t& err) const
    {
        double rho = rho__press_cold_impl(press, err) ;
        double lrho = Kokkos::log(rho) ;
        return cold_table.interp(lrho, CTABYMU) ;
    }

    // ===========================================================
    //  Public extras (specific to the 4D leptonic EOS)
    // ===========================================================

    /// Muon chemical potential  mu_mu(rho, T, ye, ymu)
    KOKKOS_INLINE_FUNCTION double
    mumu__temp_rho_ye_ymu(double& temp, double& rho, double& ye, double& ymu,
                          err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double lrho  = Kokkos::log(rho)  ;
        double ltemp = Kokkos::log(temp) ;
        return tables_muon.interp(lrho, ltemp, ye, ymu, TABMUMU) ;
    }

    /// Host-side beta-equilibrium solve (declared here, defined in .cpp).
    void
    press_eps_ye_ymu__beta_eq__rho_temp(
        double& press, double& eps,
        double& ye, double& ymu,
        double& rho, double& temp, err_t& err) const ;

    // ===========================================================
    //  Public data members (captured by value into GPU kernels)
    // ===========================================================
    tabeos_linterp_4d_t tables      ;
    tabeos_linterp_4d_t tables_ele  ;
    tabeos_linterp_4d_t tables_muon ;
    cold_eos_linterp_4d_t cold_table ;

    int nrho, nT, nye, nymu ;
    double energy_shift ;
    double lrhomin, lrhomax ;
    double ltempmin, ltempmax ;

  private:

    // ----------------------------------------------------------
    //  Clamping helpers
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION void limit_rho(double& rho, err_t& err) const {
        if ( rho < this->eos_rhomin ) { rho = (1.+1e-5)*this->eos_rhomin ; err.set(EOS_RHO_TOO_LOW)  ; }
        if ( rho > this->eos_rhomax ) { rho = (1.-1e-5)*this->eos_rhomax ; err.set(EOS_RHO_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_ye(double& ye, err_t& err) const {
        if ( ye < this->eos_yemin ) { ye = this->eos_yemin ; err.set(EOS_YE_TOO_LOW)  ; }
        if ( ye > this->eos_yemax ) { ye = this->eos_yemax ; err.set(EOS_YE_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_ymu(double& ymu, err_t& err) const {
        if ( ymu < this->eos_ymumin ) {
            ymu = this->eos_ymumin ;
            #ifdef M1_NU_FIVESPECIES
            err.set(EOS_YMU_TOO_LOW) ;
            #else
            err.set(EOS_YE_TOO_LOW) ;
            #endif
        }
        if ( ymu > this->eos_ymumax ) {
            ymu = this->eos_ymumax ;
            #ifdef M1_NU_FIVESPECIES
            err.set(EOS_YMU_TOO_HIGH) ;
            #else
            err.set(EOS_YE_TOO_HIGH) ;
            #endif
        }
    }
    KOKKOS_INLINE_FUNCTION void limit_temp(double& temp, err_t& err) const {
        double tmin = Kokkos::exp(ltempmin) ;
        double tmax = Kokkos::exp(ltempmax) ;
        if ( temp < tmin ) { temp = (1.+1e-2)*tmin ; err.set(EOS_TEMPERATURE_TOO_LOW)  ; }
        if ( temp > tmax ) { temp = (1.-1e-2)*tmax ; err.set(EOS_TEMPERATURE_TOO_HIGH) ; }
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
    //  Root-find for log(T) given eps – 4D path.
    //  Uses the same bisect-then-linear-slab approach as
    //  tabulated_eos_t::ltemp__eps_lrho_ye for efficiency.
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    ltemp__eps_lrho_ye_ymu(double& eps, double lrho,
                           double ye, double ymu, err_t& err) const
    {
        double leps    = Kokkos::log(eps + energy_shift) ;
        double lepsmin = tables.interp(lrho, ltempmin, ye, ymu, TABEPS) ;
        double lepsmax = tables.interp(lrho, ltempmax, ye, ymu, TABEPS) ;
        if ( leps <= lepsmin ) {
            eps = Kokkos::exp(lepsmin) - energy_shift ;
            err.set(EOS_EPS_TOO_LOW) ;
            return ltempmin ;
        }
        if ( leps >= lepsmax ) {
            eps = Kokkos::exp(lepsmax) - energy_shift ;
            err.set(EOS_EPS_TOO_HIGH) ;
            return ltempmax ;
        }
        // Bisect on the logT grid axis.
        int il = 0, ih = nT - 1 ;
        double el = lepsmin, eh = lepsmax ;
        while ( (ih - il) > 1 ) {
            int im = (il + ih) / 2 ;
            double em = tables.interp(lrho, tables.ltemp(im), ye, ymu, TABEPS) ;
            if ( em > leps ) { ih = im ; eh = em ; }
            else             { il = im ; el = em ; }
        }
        double ltl = tables.ltemp(il) ;
        double lth = tables.ltemp(ih) ;
        return ltl + (leps - el) * (lth - ltl) / (eh - el) ;
    }

    // ----------------------------------------------------------
    //  Root-find for log(T) given entropy – 4D path.
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    ltemp__entropy_lrho_ye_ymu(double entropy, double lrho,
                               double ye, double ymu) const
    {
        double smin = tables.interp(lrho, ltempmin, ye, ymu, TABENTROPY) ;
        double smax = tables.interp(lrho, ltempmax, ye, ymu, TABENTROPY) ;
        if ( entropy <= smin ) return ltempmin ;
        if ( entropy >= smax ) return ltempmax ;
        int il = 0, ih = nT - 1 ;
        double sl = smin, sh = smax ;
        while ( (ih - il) > 1 ) {
            int im = (il + ih) / 2 ;
            double sm = tables.interp(lrho, tables.ltemp(im), ye, ymu, TABENTROPY) ;
            if ( sm > entropy ) { ih = im ; sh = sm ; }
            else                { il = im ; sl = sm ; }
        }
        double ltl = tables.ltemp(il) ;
        double lth = tables.ltemp(ih) ;
        return ltl + (entropy - sl) * (lth - ltl) / (sh - sl) ;
    }

    // ----------------------------------------------------------
    //  Direct pressure lookup
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    press__lrho_ltemp_ye_ymu(double lrho, double ltemp,
                             double ye,   double ymu) const
    {
        return Kokkos::exp(tables.interp(lrho, ltemp, ye, ymu, TABPRESS)) ;
    }

    // ----------------------------------------------------------
    //  Limit cold pressure to table range, return log(p).
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    cold_lpress__press_limited(double& press_cold, err_t& err) const
    {
        int n = cold_table._logrho.size() ;
        double p_min = Kokkos::exp(cold_table._tables(0,   CTABPRESS)) ;
        double p_max = Kokkos::exp(cold_table._tables(n-1, CTABPRESS)) ;
        if (press_cold < p_min) {
            press_cold = p_min * (1.+1e-10) ;
            err.set(EOS_RHO_TOO_LOW) ;
        }
        if (press_cold > p_max) {
            press_cold = p_max * (1.-1e-10) ;
            err.set(EOS_RHO_TOO_HIGH) ;
        }
        return Kokkos::log(press_cold) ;
    }

} ; // class leptonic_eos_4d_t

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_LEPTONIC_4D_HH */
