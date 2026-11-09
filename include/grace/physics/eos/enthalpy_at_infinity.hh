/**
 * @file enthalpy_at_infinity.hh
 * @author Keneth Miler
 * @brief Bernoulli h_inf(Y_p): the specific enthalpy matter of proton fraction Y_p has once it
 *        is cold and dilute, built once from the EOS table nodes for the outflow diagnostic.
 * @date 2026-11-09
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

#ifndef GRACE_PHYSICS_EOS_ENTHALPY_AT_INFINITY_HH
#define GRACE_PHYSICS_EOS_ENTHALPY_AT_INFINITY_HH

#include <grace_config.h>

#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/physics/eos/tabulated_cold_eos.hh>
#include <grace/physics/eos/leptonic_eos_4d.hh>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace grace {

/**
 * @brief h_inf of the Bernoulli test -h u_t > h_inf as a function of Y_p (host only).
 *
 * One value per Y_p node of the baryon table: the minimum over density of the total
 * h = 1 + eps + P/rho on the table's coldest isotherm.  Radiation raises h at low density and
 * electron pressure at high density, so the minimum is cold, dilute matter of that Y_p.
 * Muons decay far out, so the caller passes Y_p = Y_e + Y_mu and no muon terms enter.
 */
struct enthalpy_at_infinity_t {
    std::vector<double> h ;   //!< one value per Y_p node; empty for an EOS without a composition axis
    double yp0     = 0.0 ;    //!< first Y_p node
    double inv_dyp = 0.0 ;    //!< 1 / spacing of the (uniform) Y_p axis
    double h_const = 1.0 ;    //!< returned when h is empty
    int    n_edge  = 0 ;      //!< Y_p nodes whose minimum sits on the first or last density node
    double axis_dev = 0.0 ;   //!< largest distance of a Y_p node from uniform spacing, in steps

    //! Linear in Y_p between the nodes, constant beyond the ends of the axis.
    double operator()(double const yp) const {
        int const n = static_cast<int>(h.size()) ;
        if ( n == 0 ) return h_const ;
        if ( n == 1 ) return h[0] ;
        double const x = std::clamp((yp - yp0) * inv_dyp, 0.0, static_cast<double>(n - 1)) ;
        int const k = std::min(static_cast<int>(x), n - 2) ;
        double const w = x - k ;
        return h[k] + w * (h[k+1] - h[k]) ;
    }
} ;

namespace detail {
//! Minimum over density (first index) of the node enthalpies hs(i,k), for every Y_p node k.
inline enthalpy_at_infinity_t
hinf_minimum_over_density(Kokkos::View<double**> const& hs_d, readonly_view_t<double> const& yp_d)
{
    auto const hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hs_d) ;
    auto const yp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), yp_d) ;
    int const nrho = static_cast<int>(hs.extent(0)) ;
    int const nyp  = static_cast<int>(hs.extent(1)) ;
    enthalpy_at_infinity_t out ;
    out.h.assign(nyp, 0.0) ;
    for ( int k = 0 ; k < nyp ; ++k ) {
        int imin = 0 ;
        for ( int i = 1 ; i < nrho ; ++i ) if ( hs(i,k) < hs(imin,k) ) imin = i ;
        out.h[k] = hs(imin,k) ;
        if ( imin == 0 || imin == nrho - 1 ) ++out.n_edge ;
    }
    double const dyp = nyp > 1 ? (yp(nyp-1) - yp(0)) / (nyp - 1) : 1.0 ;
    out.yp0     = yp(0) ;
    out.inv_dyp = 1.0 / dyp ;
    for ( int k = 0 ; k < nyp ; ++k )
        out.axis_dev = std::max(out.axis_dev, std::abs(yp(k) - (yp(0) + k * dyp)) / dyp) ;
    return out ;
}
} /* namespace detail */

/**
 * @brief h_inf(Y_p) of a 3D table that carries electrons and photons itself (eos_type tabulated).
 * TABPRESS holds ln P and TABEPS ln(eps + energy_shift) there.
 */
inline enthalpy_at_infinity_t enthalpy_at_infinity_from(tabulated_eos_t const& eos)
{
    using T = tabulated_eos_t ;
    auto const bt = eos.tables ;
    double const shift = eos.energy_shift ;
    int const nrho = static_cast<int>(bt._logrho.extent(0)) ;
    int const nyp  = static_cast<int>(bt._ye.extent(0)) ;
    Kokkos::View<double**> hs("hinf_isotherm", nrho, nyp) ;
    Kokkos::parallel_for("hinf_isotherm_tabulated",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrho,nyp}),
        KOKKOS_LAMBDA (int const i, int const k) {
            double const rho   = Kokkos::exp(bt._logrho(i)) ;
            double const eps   = Kokkos::exp(bt._tables(i,0,k,T::TABEPS)) - shift ;
            double const press = Kokkos::exp(bt._tables(i,0,k,T::TABPRESS)) ;
            hs(i,k) = 1.0 + eps + press / rho ;
        }) ;
    return detail::hinf_minimum_over_density(hs, bt._ye) ;
}

/**
 * @brief h_inf(Y_p) of the leptonic EOS: the baryon table plus, with add_ele_contribution, the
 * e-/e+ table (photons included) at Y_e = Y_p, exactly as total_eps/total_press combine them.
 * Electron nodes are matched to the Y_p nodes; if the two axes differ they are blended linearly.
 */
inline enthalpy_at_infinity_t enthalpy_at_infinity_from(leptonic_eos_4d_t const& eos)
{
    using L = leptonic_eos_4d_t ;
    auto const bt = eos.baryon_table ;
    auto const et = eos.ele_table ;
    double const shift     = eos.energy_shift ;
    bool   const add_ele   = eos.add_ele_contribution ;
    bool   const log_press = eos.log_baryon_press ;
    int const nrho = static_cast<int>(bt._logrho.extent(0)) ;
    int const nyp  = static_cast<int>(bt._ye.extent(0)) ;
    // Electron-table nodes j0, j1 and weight w bracketing each Y_p node (load time only).
    auto const yp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bt._ye) ;
    auto const yl_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), et._ye) ;
    int const nyl = static_cast<int>(yl_h.extent(0)) ;
    Kokkos::View<int*>    j0("hinf_ele_j0", nyp), j1("hinf_ele_j1", nyp) ;
    Kokkos::View<double*> wj("hinf_ele_w",  nyp) ;
    auto j0_h = Kokkos::create_mirror_view(j0) ;
    auto j1_h = Kokkos::create_mirror_view(j1) ;
    auto wj_h = Kokkos::create_mirror_view(wj) ;
    for ( int k = 0 ; k < nyp ; ++k ) {
        double const y = std::clamp(yp_h(k), yl_h(0), yl_h(nyl-1)) ;
        int j = 0 ;
        while ( j < nyl - 2 && yl_h(j+1) <= y ) ++j ;
        j0_h(k) = j ;
        j1_h(k) = std::min(j + 1, nyl - 1) ;
        wj_h(k) = j1_h(k) > j ? (y - yl_h(j)) / (yl_h(j+1) - yl_h(j)) : 0.0 ;
    }
    Kokkos::deep_copy(j0, j0_h) ; Kokkos::deep_copy(j1, j1_h) ; Kokkos::deep_copy(wj, wj_h) ;

    Kokkos::View<double**> hs("hinf_isotherm", nrho, nyp) ;
    Kokkos::parallel_for("hinf_isotherm_leptonic",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrho,nyp}),
        KOKKOS_LAMBDA (int const i, int const k) {
            double const rho = Kokkos::exp(bt._logrho(i)) ;
            double eps = Kokkos::exp(bt._tables(i,0,k,L::TABEPS)) - shift ;
            // log mode: TABPRESS = ln(P_b + P_e(Y_e=Y_q)), already the total at Y_e = Y_p
            double press = log_press ? Kokkos::exp(bt._tables(i,0,k,L::TABPRESS))
                                     : bt._tables(i,0,k,L::TABPRESS) ;
            if ( add_ele ) {
                auto const ele = [&] (int const v) {
                    return (1.0 - wj(k)) * et._tables(i,0,j0(k),v) + wj(k) * et._tables(i,0,j1(k),v) ;
                } ;
                eps += ele(L::TABEPS_E_MINUS) + ele(L::TABEPS_E_PLUS) ;
                if ( !log_press ) press += ele(L::TABPRESS_E_MINUS) + ele(L::TABPRESS_E_PLUS) ;
            }
            hs(i,k) = 1.0 + eps + press / rho ;
        }) ;
    return detail::hinf_minimum_over_density(hs, bt._ye) ;
}

/**
 * @brief h_inf of a hybrid EOS with a tabulated cold backbone: the minimum of h = 1 + eps + P/rho
 * over the cold table's rows.  The backbone is in beta equilibrium, so h_inf is one number.
 */
inline enthalpy_at_infinity_t enthalpy_at_infinity_from(tabulated_cold_eos_t const& cold)
{
    using T = tabulated_eos_t ;
    auto const tab = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cold.table()._tables) ;
    auto const lr  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cold.table()._logrho) ;
    int const n = static_cast<int>(lr.extent(0)) ;
    enthalpy_at_infinity_t out ;
    int imin = 0 ;
    for ( int i = 0 ; i < n ; ++i ) {
        double const h = 1.0 + std::exp(tab(i,T::CTABEPS)) - cold.energy_shift
                       + std::exp(tab(i,T::CTABPRESS)) / std::exp(lr(i)) ;
        if ( i == 0 || h < out.h_const ) { out.h_const = h ; imin = i ; }
    }
    if ( n > 0 && (imin == 0 || imin == n - 1) ) out.n_edge = 1 ;
    return out ;
}

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_ENTHALPY_AT_INFINITY_HH */
