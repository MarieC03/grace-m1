/**
 * @file outflow_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Diagnostic that integrates mass, energy and momentum outflow fluxes across registered extraction spheres.
 * @date 2026-01-15
 *
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#ifndef GRACE_IO_OUTFLOW_DIAGNOSTICS_HH
#define GRACE_IO_OUTFLOW_DIAGNOSTICS_HH


#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/IO/output_diagnostics.hh>
#include <grace/IO/diagnostics/diagnostic_base.hh>
#include <grace/errors/error.hh>

#include <cmath>
#include <grace/IO/spherical_surfaces.hh>

#include <grace/data_structures/variable_indices.hh>

#include <array>
#include <vector>
#include <string>


namespace grace {

struct outflows:
    public diagnostic_base_t<outflows>
{
    using base_t = diagnostic_base_t<outflows>;

    #if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
    enum loc_var_idx_t : int {
        GXXL=0, GXYL, GXZL, GYYL, GYZL, GZZL,
        BETAXL, BETAYL, BETAZL, ALPL, NUM_VARS
    } ;
    #else
    enum loc_var_idx_t : int {
        GTXXL=0, GTXYL, GTXZL, GTYYL, GTYZL, GTZZL, CHIL,
        BETAXL, BETAYL, BETAZL, ALPL, NUM_VARS
    } ;
    #endif
    enum loc_aux_idx_t : int {
        RHOL=0, EPSL, PRESSL, ZXL, ZYL, ZZL, YEL,
        #ifdef GRACE_ENABLE_MUONS
        YMUL,
        #endif
        NUM_AUX
    };
    enum diag_var_idx_t : int {
        GEO_UNBOUND=0, BERN_UNBOUND, TOT, N_DIAG_VARS
    } ;

    // ---- Y_e-binned ejecta -------------------------------------------------
    // The three scalars above say HOW MUCH unbound mass crosses the sphere;
    // these say WHAT IT IS.  The r-process yield is a steep function of Y_e
    // (Y_e <~ 0.25 -> heavy third-peak elements and a red kilonova, Y_e >~ 0.3
    // -> a blue one), and neutrino irradiation is precisely what raises Y_e in
    // the ejecta -- so this histogram is the observable the M1 sector moves.
    // Two runs with very different neutrino physics can share an identical
    // Mdot_unbound.
    //
    // Bins the BERNOULLI-unbound flux, the usual ejecta criterion, so the bins
    // sum to Mdot_unbound_bern by construction (that identity is the test).
    // The COUNT is compile time because n_fluxes must be constexpr; the RANGE is read
    // from outflows.ye_bin_min/max (default [0, 0.6): round edges that do not depend
    // on the EOS table, with the table bounds 0.01 and 0.5 in bins of their own).
    static constexpr int    n_ye_bins = 30 ;
    static double ye_bin_lo ;
    static double ye_bin_hi ;
    //! First histogram column; bin b lives at YE_BIN0 + b.
    static constexpr int    YE_BIN0   = static_cast<int>(N_DIAG_VARS) ;

    #ifdef GRACE_ENABLE_MUONS
    // With muons the same flux is also binned by Y_mu and by Y_p = Y_e + Y_mu (the
    // axis of the baryon table).  Y_p uses the Y_e edges.  Y_mu spans decades, from
    // the table floor (5e-4) to ~0.1, so its bins are LOGARITHMIC between
    // outflows.ymu_bin_min/max (default 10^-3.5 .. 10^-0.5, i.e. 8 bins per decade).
    static constexpr int    n_ymu_bins      = 24 ;
    static double log_ymu_bin_lo ;                            //!< log10 of ymu_bin_min
    static double log_ymu_bin_hi ;                            //!< log10 of ymu_bin_max
    static constexpr int    n_yp_bins       = n_ye_bins ;     //!< same edges as Y_e
    static constexpr int    YMU_BIN0 = YE_BIN0  + n_ye_bins ;
    static constexpr int    YP_BIN0  = YMU_BIN0 + n_ymu_bins ;
    static constexpr size_t n_fluxes =
        static_cast<size_t>(N_DIAG_VARS + n_ye_bins + n_ymu_bins + n_yp_bins) ;
    #else
    static constexpr size_t n_fluxes =
        static_cast<size_t>(N_DIAG_VARS + n_ye_bins) ;
    #endif

    //! Bin of x in n equal bins over [lo,hi).  Clamped, not dropped, at both ends (and
    //! for NaN), so the bins of one quantity always sum to Mdot_unbound_bern.
    static constexpr int clamped_bin(double x, double lo, double hi, int n) {
        if ( !(x > lo) ) return 0 ;
        if ( !(x < hi) ) return n - 1 ;
        int const b = static_cast<int>((x - lo) / ((hi - lo) / n)) ;
        return b < 0 ? 0 : (b >= n ? n - 1 : b) ;
    }

    static std::vector<std::string> flux_names ;
    //! "Mdot_ye_0.000_0.020" style labels for the current range: the edges travel with the data.
    static std::vector<std::string> make_flux_names() ;

    outflows()
        : base_t("outflows")
    {
        ye_bin_lo = get_param<double>("outflows","ye_bin_min") ;
        ye_bin_hi = get_param<double>("outflows","ye_bin_max") ;
        if ( !(ye_bin_hi > ye_bin_lo) )
            ERROR("outflows: ye_bin_max (" << ye_bin_hi << ") must exceed ye_bin_min (" << ye_bin_lo << ")") ;
        #ifdef GRACE_ENABLE_MUONS
        double const ymu_lo = get_param<double>("outflows","ymu_bin_min") ;
        double const ymu_hi = get_param<double>("outflows","ymu_bin_max") ;
        if ( !(ymu_lo > 0.0) || !(ymu_hi > ymu_lo) )
            ERROR("outflows: need 0 < ymu_bin_min (" << ymu_lo << ") < ymu_bin_max (" << ymu_hi << "), the Y_mu bins are logarithmic") ;
        log_ymu_bin_lo = std::log10(ymu_lo) ;
        log_ymu_bin_hi = std::log10(ymu_hi) ;
        #endif
        flux_names = make_flux_names() ;
        #if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
        this->var_interp_idx = std::vector<int>({GXX_, GXY_, GXZ_, GYY_, GYZ_, GZZ_, BETAX_, BETAY_, BETAZ_, ALP_});
        #else
        this->var_interp_idx = std::vector<int>({GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_, CHI_, BETAX_, BETAY_, BETAZ_, ALP_});
        #endif
        this->aux_interp_idx = std::vector<int>({RHO_,EPS_,PRESS_,ZVECX_,ZVECY_,ZVECZ_,YE_
                                                 #ifdef GRACE_ENABLE_MUONS
                                                 ,YMU_
                                                 #endif
                                                 });
    }

    std::array<double,n_fluxes>
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d,
        Kokkos::View<double**> ivals_aux_d,
        spherical_surface_iface const& detector
    )  ;
};


#ifdef GRACE_ENABLE_M1
// ---------------------------------------------------------------------------
/**
 * @brief Radiation energy and number luminosity through spherical detectors.
 *
 * For each registered detector sphere the diagnostic integrates
 *   L_E^(s) = ∮ (alpha F^i - beta^i E) n_i r² dΩ,   F^i = gamma^ij F_j,
 * per neutrino species s.  The conserved variables ERAD/FRADX/Y/Z carry the
 * factor sqrt(gamma) and FRAD holds the LOWERED flux, so the 3-metric is
 * interpolated on the sphere to raise the index before the projection on the
 * coordinate normal n_i = x_i/r.  On a stationary metric this is the flux of
 * the conserved Killing energy.
 *
 * Uses the same "outflows" parameter block (detector_names) as the mass
 * outflow diagnostic.
 */
// ---------------------------------------------------------------------------
struct m1_outflows :
    public diagnostic_base_t<m1_outflows>
{
    using base_t = diagnostic_base_t<m1_outflows>;

    // Local index into ivals (state variables interpolated to sphere)
    enum loc_var_idx_t : int {
        // Metric
#if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
        GXXL=0, GXYL, GXZL, GYYL, GYZL, GZZL,
#else
        GTXXL=0, GTXYL, GTXZL, GTYYL, GTYZL, GTZZL, CHIL,
#endif
        BETAXL, BETAYL, BETAZL, ALPL,
#if GRACE_M1_NU_SPECIES >= 1
        E1L, FX1L, FY1L, FZ1L,
#endif
#if GRACE_M1_NU_SPECIES >= 3
        E2L, FX2L, FY2L, FZ2L,
        E3L, FX3L, FY3L, FZ3L,
#endif
#if GRACE_M1_NU_SPECIES >= 5
        E4L, FX4L, FY4L, FZ4L,
        E5L, FX5L, FY5L, FZ5L,
#endif
#ifdef GRACE_M1_PHOTONS
        EPHL, FXPHL, FYPHL, FZPHL,
#endif
        NUM_VARS
    };

    // No aux variables needed
    enum loc_aux_idx_t : int { NUM_AUX = 0 };

    // One luminosity entry per species
#ifdef GRACE_M1_PHOTONS
    static constexpr size_t photon_flux = 1;
#else
    static constexpr size_t photon_flux = 0;
#endif
#if (GRACE_M1_NU_SPECIES >= 5)
    static constexpr size_t n_fluxes = 5 + photon_flux;
#elif (GRACE_M1_NU_SPECIES >= 3)
    static constexpr size_t n_fluxes = 3 + photon_flux;
#elif (GRACE_M1_NU_SPECIES >= 1)
    static constexpr size_t n_fluxes = 1 + photon_flux;
#else
    static constexpr size_t n_fluxes = 0 + photon_flux;
#endif

    static std::vector<std::string> flux_names;

    m1_outflows()
        : base_t("outflows")   // reuse outflows detector list
    {
        // Radiation flux components per species (state array indices)
        this->var_interp_idx = {
#if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
            GXX_, GXY_, GXZ_, GYY_, GYZ_, GZZ_,
#else
            GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_, CHI_,
#endif
            BETAX_, BETAY_, BETAZ_, ALP_
#if GRACE_M1_NU_SPECIES >= 1
            , ERAD1_, FRADX1_, FRADY1_, FRADZ1_
#endif
#if GRACE_M1_NU_SPECIES >= 3
            , ERAD2_, FRADX2_, FRADY2_, FRADZ2_
            , ERAD3_, FRADX3_, FRADY3_, FRADZ3_
#endif
#if GRACE_M1_NU_SPECIES >= 5
            , ERAD4_, FRADX4_, FRADY4_, FRADZ4_
            , ERAD5_, FRADX5_, FRADY5_, FRADZ5_
#endif
#ifdef GRACE_M1_PHOTONS
            , ERADPH_, FRADXPH_, FRADYPH_, FRADZPH_
#endif
        };
        this->aux_interp_idx = {};
    }

    std::array<double, n_fluxes>
    compute_local_fluxes(
        Kokkos::View<double**> ivals_d,
        Kokkos::View<double**> ivals_aux_d,
        spherical_surface_iface const& detector
    );
};
#endif /* GRACE_ENABLE_M1 */


}

#endif /*GRACE_IO_OUTFLOW_DIAGNOSTICS_HH*/
