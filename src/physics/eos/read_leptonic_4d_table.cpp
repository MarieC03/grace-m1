/**
 * @file read_leptonic_4d_table.cpp
 * @brief  Implementation of the 4D leptonic EOS table reader for GRACE.
 *         Reads the 4-axis (rho, T, Y_e, Y_mu) tables in the native
 *         GRACE leptonic-4D HDF5 format.
 * @date   2025
 *
 * @copyright This file is part of GRACE.  GPL-3 or later.
 */

#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/physics/eos/read_leptonic_4d_table.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh>

#include <hdf5.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  HDF5 helpers (consistent style with read_eos_table.cpp)
// ---------------------------------------------------------------------------
#define HDF5_CALL(ret, fn_call) \
    do { (ret) = (fn_call) ; \
         if ((ret) < 0) { ERROR("HDF5 call failed: " #fn_call) ; } \
    } while(0)

// Read a top-level dataset from `file` into VAR with element type TYPE,
// using memory dataspace MEM. (H5S_ALL is fine when VAR matches the file.)
#define READ_4D_EOS_HDF5(NAME, VAR, TYPE, MEM)                              \
    do {                                                                    \
        hid_t _ds ;                                                         \
        HDF5_CALL(_ds, H5Dopen(file, NAME, H5P_DEFAULT)) ;                  \
        HDF5_CALL(h5err, H5Dread(_ds, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR));\
        HDF5_CALL(h5err, H5Dclose(_ds)) ;                                   \
    } while(0)

// Read one variable slice from a packed (NTAB, NP4) HDF5 dataset into
// alltables_temp (nrho*ntemp*nye*nymu doubles, starting at OFF).
// `mem5` and `var5` (= {1, NP4}) must already be set up by the caller.
#define READ_4D_EOSTABLE_HDF5(NAME, OFF)                                    \
    do {                                                                    \
        hsize_t offset5[2] = {(hsize_t)(OFF), 0} ;                          \
        H5Sselect_hyperslab(mem5, H5S_SELECT_SET, offset5, NULL, var5, NULL);\
        READ_4D_EOS_HDF5(NAME, alltables_temp, H5T_NATIVE_DOUBLE, mem5) ;   \
    } while(0)

namespace grace {

// ============================================================
//  Helper: read the 1-D cold-slice table.
//  Format per row (after a 2-line header):
//      log(rho)  temp  ye_cold  ymu_cold  press  eps  cs2  entropy
// ============================================================
static void read_leptonic_cold_table(
    const std::string& filename,
    Kokkos::View<double**, grace::default_space>& d_data,
    Kokkos::View<double*,  grace::default_space>& d_rho)
{
    std::ifstream f(filename) ;
    ASSERT(f.is_open(), "Cannot open leptonic cold table: " << filename) ;

    std::string line ;
    std::getline(f, line) ;                  // description
    std::getline(f, line) ;                  // number of rows
    std::istringstream iss_n(line) ;
    size_t nrows ;
    iss_n >> nrows ;
    ASSERT(iss_n, "Failed to read nrows in leptonic cold table") ;

    constexpr int NCOLS = leptonic_eos_4d_t::COLD_VIDX::N_CTAB_VARS ;

    Kokkos::realloc(d_data, nrows, NCOLS) ;
    Kokkos::realloc(d_rho,  nrows) ;
    auto h_data = Kokkos::create_mirror_view(d_data) ;
    auto h_rho  = Kokkos::create_mirror_view(d_rho)  ;

    for (size_t i = 0; i < nrows; ++i) {
        ASSERT(std::getline(f,line),
               "Unexpected EOF in leptonic cold table at row " << i) ;
        std::istringstream iss(line) ;
        double logrho_val ;
        iss >> logrho_val ;
        h_rho(i) = logrho_val ;
        for (int j = 0; j < NCOLS; ++j) {
            double val ;
            iss >> val ;
            h_data(i,j) = val ;
        }
    }
    Kokkos::deep_copy(d_data, h_data) ;
    Kokkos::deep_copy(d_rho,  h_rho)  ;
}

// ============================================================
//  Main reader
// ============================================================
grace::leptonic_eos_4d_t read_leptonic_4d_table()
{
    using namespace grace::physical_constants ;

    // CGS-to-geometric unit conversion factors (same idiom as
    // read_eos_table.cpp). uconv.pressure converts CGS pressure to
    // geometric pressure; uconv.velocity^2 is the eps factor; etc.
    auto const uconv = CGS_units / GEOM_units ;

    auto const fname      = grace::get_param<std::string>("eos","leptonic_4d","table_filename") ;
    auto const cold_fname = grace::get_param<std::string>("eos","leptonic_4d","cold_table_filename") ;
    auto const tab_format = grace::get_param<std::string>("eos","leptonic_4d","table_format") ;

    bool const do_energy_shift =
        grace::get_param<bool>("eos","leptonic_4d","do_energy_shift") ;
    bool const use_muonic_eos =
        grace::get_param<bool>("eos","leptonic_4d","use_muonic_eos") ;
    bool const atm_beta_eq =
        grace::get_param<bool>("grmhd","atmosphere","atmosphere_is_beta_eq") ;

    GRACE_INFO("Reading 4D leptonic EOS table: {}", fname) ;
    GRACE_INFO("Format: {}", tab_format) ;

    herr_t h5err ;
    hid_t  file  ;
    HDF5_CALL(file, H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) ;

    // -------------------------------------------------------
    //  Grid sizes
    // -------------------------------------------------------
    int nrho, ntemp, nye, nymu ;
    READ_4D_EOS_HDF5("pointsrho",  &nrho,  H5T_NATIVE_INT, H5S_ALL) ;
    READ_4D_EOS_HDF5("pointstemp", &ntemp, H5T_NATIVE_INT, H5S_ALL) ;
    READ_4D_EOS_HDF5("pointsye",   &nye,   H5T_NATIVE_INT, H5S_ALL) ;
    READ_4D_EOS_HDF5("pointsymu",  &nymu,  H5T_NATIVE_INT, H5S_ALL) ;

    GRACE_INFO("4D EOS grid: nrho={} nT={} nye={} nymu={}",
               nrho, ntemp, nye, nymu) ;

    // -------------------------------------------------------
    //  Allocate the flat temp buffer used for every hyperslab read.
    // -------------------------------------------------------
    const size_t NTAB_B = leptonic_eos_4d_t::TEOS_VIDX::N_TAB_VARS_BARYON ;
    const size_t NTAB_E = leptonic_eos_4d_t::ELE_VIDX ::N_TAB_VARS_ELE   ;
    const size_t NTAB_M = leptonic_eos_4d_t::MUON_VIDX::N_TAB_VARS_MUON  ;
    const size_t NP4    = static_cast<size_t>(nrho)*ntemp*nye*nymu ;

    std::vector<double> buf(NP4) ;
    double* alltables_temp = buf.data() ;

    // HDF5 dataspace describing the on-disk layout (NTAB, NP4).
    // We re-set its first extent before reading each of the three groups
    // (baryon, ele, muon) so that hyperslab selection on row OFF is valid.
    hsize_t table_dims[2] = { NTAB_B, NP4 } ;
    hsize_t var5[2]       = { 1, NP4 } ;
    hid_t   mem5 = H5Screate_simple(2, table_dims, NULL) ;

    // -------------------------------------------------------
    //  Axes
    // -------------------------------------------------------
    std::vector<double> logrho(nrho), logtemp(ntemp), yes(nye), ymus(nymu) ;
    double energy_shift_raw{0.} ;

    READ_4D_EOS_HDF5("logrho",       logrho.data(),       H5T_NATIVE_DOUBLE, H5S_ALL) ;
    READ_4D_EOS_HDF5("logtemp",      logtemp.data(),      H5T_NATIVE_DOUBLE, H5S_ALL) ;
    READ_4D_EOS_HDF5("ye",           yes.data(),          H5T_NATIVE_DOUBLE, H5S_ALL) ;
    READ_4D_EOS_HDF5("ymu",          ymus.data(),         H5T_NATIVE_DOUBLE, H5S_ALL) ;
    READ_4D_EOS_HDF5("energy_shift", &energy_shift_raw,   H5T_NATIVE_DOUBLE, H5S_ALL) ;

    double baryon_mass = mn_MeV * MeV_to_g * uconv.mass ;
    if (H5Lexists(file, "/mass_factor", H5P_DEFAULT) > 0) {
        hid_t mb_ds ;
        HDF5_CALL(mb_ds, H5Dopen(file, "mass_factor", H5P_DEFAULT)) ;
        H5Dread(mb_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &baryon_mass) ;
        H5Dclose(mb_ds) ;
        GRACE_INFO("Baryon mass from file: {}", baryon_mass) ;
    } else {
        GRACE_INFO("Using default baryon mass: {}", baryon_mass) ;
    }

    // energy_shift is read in CGS-eps units; convert to geometric.
    double energy_shift = do_energy_shift ? energy_shift_raw * SQR(uconv.velocity) : 0.0 ;

    // -------------------------------------------------------
    //  Convert axes: log10 → loge + geometric units
    // -------------------------------------------------------
    for (int i=0; i<nrho;  ++i) logrho[i]  = logrho[i]  * std::log(10.) + std::log(uconv.mass_density) ;
    for (int i=0; i<ntemp; ++i) logtemp[i] = logtemp[i] * std::log(10.) ;

    // -------------------------------------------------------
    //  Allocate 5-D Kokkos views  [irho,iT,iye,iymu,ivar]
    // -------------------------------------------------------
    using view5d = Kokkos::View<double*****, grace::default_space> ;
    view5d v_baryon("eos4d_baryon", nrho, ntemp, nye, nymu, NTAB_B) ;
    view5d v_ele   ("eos4d_ele",    nrho, ntemp, nye, nymu, NTAB_E) ;
    view5d v_muon  ("eos4d_muon",   nrho, ntemp, nye, nymu, NTAB_M) ;

    auto h_baryon = Kokkos::create_mirror_view(v_baryon) ;
    auto h_ele    = Kokkos::create_mirror_view(v_ele)    ;
    auto h_muon   = Kokkos::create_mirror_view(v_muon)   ;

    // Reorder the flat slice we just read into 5D view slot (i,j,k,lm,iv).
    auto store_slot = [&](auto& hv, size_t iv)
    {
        for (int lm=0; lm<nymu; ++lm)
        for (int k=0;  k<nye;   ++k)
        for (int j=0;  j<ntemp; ++j)
        for (int i=0;  i<nrho;  ++i)
        {
            // Convention: the flattening in the file is
            //   index = i + nrho*(j + ntemp*(k + nye*lm))
            // for a single variable slice.
            size_t indold = i + (size_t)nrho*(j + (size_t)ntemp*(k + (size_t)nye*lm)) ;
            hv(i,j,k,lm,iv) = alltables_temp[indold] ;
        }
    } ;

    // -------------------------------------------------------
    //  Read baryon table
    // -------------------------------------------------------
    table_dims[0] = NTAB_B ;
    H5Sset_extent_simple(mem5, 2, table_dims, NULL) ;

    #define READ_BAR(NAME, IDX) \
        do { READ_4D_EOSTABLE_HDF5(NAME, IDX) ; store_slot(h_baryon, IDX) ; } while(0)

    READ_BAR("logpress",  leptonic_eos_4d_t::TEOS_VIDX::TABPRESS) ;
    READ_BAR("logenergy", leptonic_eos_4d_t::TEOS_VIDX::TABEPS) ;
    READ_BAR("entropy",   leptonic_eos_4d_t::TEOS_VIDX::TABENTROPY) ;
    READ_BAR("cs2",       leptonic_eos_4d_t::TEOS_VIDX::TABCSND2) ;
    READ_BAR("mu_e",      leptonic_eos_4d_t::TEOS_VIDX::TABMUE) ;
    READ_BAR("mu_p",      leptonic_eos_4d_t::TEOS_VIDX::TABMUP) ;
    READ_BAR("mu_n",      leptonic_eos_4d_t::TEOS_VIDX::TABMUN) ;
    READ_BAR("Xa",        leptonic_eos_4d_t::TEOS_VIDX::TABXA) ;
    READ_BAR("Xh",        leptonic_eos_4d_t::TEOS_VIDX::TABXH) ;
    READ_BAR("Xn",        leptonic_eos_4d_t::TEOS_VIDX::TABXN) ;
    READ_BAR("Xp",        leptonic_eos_4d_t::TEOS_VIDX::TABXP) ;
    READ_BAR("Abar",      leptonic_eos_4d_t::TEOS_VIDX::TABABAR) ;
    READ_BAR("Zbar",      leptonic_eos_4d_t::TEOS_VIDX::TABZBAR) ;
    #undef READ_BAR

    // -------------------------------------------------------
    //  Electron lepton table
    // -------------------------------------------------------
    table_dims[0] = NTAB_E ;
    H5Sset_extent_simple(mem5, 2, table_dims, NULL) ;

    #define READ_ELE(NAME, IDX) \
        do { READ_4D_EOSTABLE_HDF5(NAME, IDX) ; store_slot(h_ele, IDX) ; } while(0)

    READ_ELE("ele/mu_ele",        leptonic_eos_4d_t::ELE_VIDX::TABMUELE) ;
    READ_ELE("ele/yle_minus",     leptonic_eos_4d_t::ELE_VIDX::TABYLE_MINUS) ;
    READ_ELE("ele/yle_plus",      leptonic_eos_4d_t::ELE_VIDX::TABYLE_PLUS) ;
    READ_ELE("ele/press_e_minus", leptonic_eos_4d_t::ELE_VIDX::TABPRESS_E_MINUS) ;
    READ_ELE("ele/press_e_plus",  leptonic_eos_4d_t::ELE_VIDX::TABPRESS_E_PLUS) ;
    READ_ELE("ele/eps_e_minus",   leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_MINUS) ;
    READ_ELE("ele/eps_e_plus",    leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_PLUS) ;
    READ_ELE("ele/s_e_minus",     leptonic_eos_4d_t::ELE_VIDX::TABS_E_MINUS) ;
    READ_ELE("ele/s_e_plus",      leptonic_eos_4d_t::ELE_VIDX::TABS_E_PLUS) ;
    #undef READ_ELE

    // -------------------------------------------------------
    //  Muon lepton table  (optional)
    // -------------------------------------------------------
    if (use_muonic_eos) {
        table_dims[0] = NTAB_M ;
        H5Sset_extent_simple(mem5, 2, table_dims, NULL) ;

        #define READ_MU(NAME, IDX) \
            do { READ_4D_EOSTABLE_HDF5(NAME, IDX) ; store_slot(h_muon, IDX) ; } while(0)

        READ_MU("muon/mu_mu",          leptonic_eos_4d_t::MUON_VIDX::TABMUMU) ;
        READ_MU("muon/ymu_minus",      leptonic_eos_4d_t::MUON_VIDX::TABYMU_MINUS) ;
        READ_MU("muon/ymu_plus",       leptonic_eos_4d_t::MUON_VIDX::TABYMU_PLUS) ;
        READ_MU("muon/press_mu_minus", leptonic_eos_4d_t::MUON_VIDX::TABPRESS_MU_MINUS) ;
        READ_MU("muon/press_mu_plus",  leptonic_eos_4d_t::MUON_VIDX::TABPRESS_MU_PLUS) ;
        READ_MU("muon/eps_mu_minus",   leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_MINUS) ;
        READ_MU("muon/eps_mu_plus",    leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_PLUS) ;
        READ_MU("muon/s_mu_minus",     leptonic_eos_4d_t::MUON_VIDX::TABS_MU_MINUS) ;
        READ_MU("muon/s_mu_plus",      leptonic_eos_4d_t::MUON_VIDX::TABS_MU_PLUS) ;
        #undef READ_MU
    }

    H5Sclose(mem5) ;
    H5Fclose(file) ;

    // -------------------------------------------------------
    //  Unit conversion over the baryon table.
    //  Stored convention (raw, from disk):
    //    logpress  : log10(P_cgs)
    //    logenergy : log10(eps_cgs * c^2)  with shift
    //    cs2       : (cm/s)^2
    //    mu_e/p/n  : MeV/baryon
    //    entropy   : k_B / baryon (dimensionless)
    //    Xa/h/n/p, Abar, Zbar : dimensionless
    // -------------------------------------------------------
    double hmin{std::numeric_limits<double>::max()} ;
    double hmax{std::numeric_limits<double>::lowest()} ;
    double epsmin{std::numeric_limits<double>::max()} ;
    double epsmax_val{std::numeric_limits<double>::lowest()} ;

    for (int lm=0; lm<nymu; ++lm)
    for (int k=0;  k<nye;   ++k)
    for (int j=0;  j<ntemp; ++j)
    for (int i=0;  i<nrho;  ++i)
    {
        double rhoL = std::exp(logrho[i]) ;

        // pressure: log10 -> loge + log(uconv.pressure)
        h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABPRESS) =
            h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABPRESS) * std::log(10.)
            + std::log(uconv.pressure) ;
        double pressL = std::exp(h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABPRESS)) ;

        // eps: 10^(logenergy) * c^2 (geometric) gives eps_shifted
        double leps_raw = h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABEPS) ;
        double epsT     = std::pow(10., leps_raw) * SQR(uconv.velocity) ;
        double epsL     = epsT - energy_shift ;
        h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABEPS) = std::log(epsT) ;

        epsmin     = std::min(epsmin,     epsL) ;
        epsmax_val = std::max(epsmax_val, epsL) ;

        // sound speed: (cm/s)^2 -> dimensionless, then divide by h.
        double& cs2 = h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABCSND2) ;
        cs2 *= SQR(uconv.velocity) ;
        if (cs2 < 0.) cs2 = 0. ;
        double hL = 1. + epsL + pressL / rhoL ;
        cs2 /= hL ;
        cs2 = std::min(std::max(cs2, 1e-6), 1.-1e-10) ;

        // chemical potentials: MeV/baryon -> dimensionless geometric
        // mu_geom = mu_MeV * MeV_to_g * uconv.mass / baryon_mass
        // (baryon_mass is already in geometric mass units from above.)
        constexpr double MeV2g = MeV_to_g ;
        double const mu_factor = MeV2g * uconv.mass / baryon_mass ;
        h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABMUE) *= mu_factor ;
        h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABMUP) *= mu_factor ;
        h_baryon(i,j,k,lm, leptonic_eos_4d_t::TEOS_VIDX::TABMUN) *= mu_factor ;

        hmin = std::min(hmin, hL) ;
        hmax = std::max(hmax, hL) ;
    }

    // Convert lepton-table pressure and eps to geometric units.
    auto convert_lepton_view = [&](auto& hv, int p1, int p2, int e1, int e2)
    {
        for (int lm=0; lm<nymu; ++lm)
        for (int k=0;  k<nye;   ++k)
        for (int j=0;  j<ntemp; ++j)
        for (int i=0;  i<nrho;  ++i)
        {
            hv(i,j,k,lm,p1) *= uconv.pressure ;
            hv(i,j,k,lm,p2) *= uconv.pressure ;
            hv(i,j,k,lm,e1) *= SQR(uconv.velocity) ;
            hv(i,j,k,lm,e2) *= SQR(uconv.velocity) ;
        }
    } ;
    convert_lepton_view(h_ele,
        leptonic_eos_4d_t::ELE_VIDX::TABPRESS_E_MINUS,
        leptonic_eos_4d_t::ELE_VIDX::TABPRESS_E_PLUS,
        leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_MINUS,
        leptonic_eos_4d_t::ELE_VIDX::TABEPS_E_PLUS) ;
    if (use_muonic_eos) {
        convert_lepton_view(h_muon,
            leptonic_eos_4d_t::MUON_VIDX::TABPRESS_MU_MINUS,
            leptonic_eos_4d_t::MUON_VIDX::TABPRESS_MU_PLUS,
            leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_MINUS,
            leptonic_eos_4d_t::MUON_VIDX::TABEPS_MU_PLUS) ;
    }

    // -------------------------------------------------------
    //  Deep copy to device
    // -------------------------------------------------------
    Kokkos::deep_copy(v_baryon, h_baryon) ;
    Kokkos::deep_copy(v_ele,    h_ele)    ;
    Kokkos::deep_copy(v_muon,   h_muon)   ;

    // -------------------------------------------------------
    //  Axes
    // -------------------------------------------------------
    using view1d = Kokkos::View<double*, grace::default_space> ;
    view1d v_logrho ("eos4d_logrho",  nrho)  ;
    view1d v_logT   ("eos4d_logT",    ntemp) ;
    view1d v_ye     ("eos4d_ye",      nye)   ;
    view1d v_ymu    ("eos4d_ymu",     nymu)  ;
    auto h_lr = Kokkos::create_mirror_view(v_logrho) ;
    auto h_lt = Kokkos::create_mirror_view(v_logT)   ;
    auto h_ye = Kokkos::create_mirror_view(v_ye)     ;
    auto h_ym = Kokkos::create_mirror_view(v_ymu)    ;
    for (int i=0; i<nrho;  ++i) h_lr(i) = logrho[i]  ;
    for (int i=0; i<ntemp; ++i) h_lt(i) = logtemp[i] ;
    for (int i=0; i<nye;   ++i) h_ye(i) = yes[i]     ;
    for (int i=0; i<nymu;  ++i) h_ym(i) = ymus[i]    ;
    Kokkos::deep_copy(v_logrho, h_lr) ;
    Kokkos::deep_copy(v_logT,   h_lt) ;
    Kokkos::deep_copy(v_ye,     h_ye) ;
    Kokkos::deep_copy(v_ymu,    h_ym) ;

    // -------------------------------------------------------
    //  Cold slice
    // -------------------------------------------------------
    Kokkos::View<double**, grace::default_space> cold_tabs ;
    Kokkos::View<double*,  grace::default_space> cold_lrho ;
    GRACE_INFO("Reading 4D leptonic cold table: {}", cold_fname) ;
    read_leptonic_cold_table(cold_fname, cold_tabs, cold_lrho) ;

    // -------------------------------------------------------
    //  EOS limits
    // -------------------------------------------------------
    double rhomax  = std::exp(logrho[nrho-1])  ;
    double rhomin  = std::exp(logrho[0])       ;
    double tempmax = std::exp(logtemp[ntemp-1]) ;
    double tempmin = std::exp(logtemp[0])       ;
    double yemax   = yes[nye-1]                 ;
    double yemin   = yes[0]                     ;
    double ymumax  = ymus[nymu-1]               ;
    double ymumin  = ymus[0]                    ;

    auto usr_epsmax = grace::get_param<double>("eos","eps_maximum") ;
    if (usr_epsmax < epsmax_val) epsmax_val = usr_epsmax ;

    // Atmosphere defaults: floor temperature; ye/ymu either from parfile
    // or, when requested, found by a host-side beta-equilibrium solve
    // *after* construction (see press_eps_ye_ymu__beta_eq__rho_temp).
    double temp_floor = grace::get_param<double>("grmhd","atmosphere","temp_fl") ;
    double tmin_safe  = std::exp(logtemp[1]) ;
    if (temp_floor < tmin_safe) {
        GRACE_WARN("Atmosphere T_fl below second table point ({}); overriding.", tmin_safe) ;
        temp_floor = tmin_safe ;
    }
    double ye_atm  = grace::get_param<double>("grmhd","atmosphere","ye_fl") ;
    double ymu_atm = ymumin ;

    if (atm_beta_eq) {
        GRACE_INFO("Atmosphere: beta-equilibrium Y_e, Y_mu will be solved "
                   "host-side after EOS construction.") ;
    }

    GRACE_INFO("4D leptonic EOS rho [{:.4e}, {:.4e}]  T [{:.4e}, {:.4e}]  "
               "Ye [{:.3f}, {:.3f}]  Ymu [{:.3f}, {:.3f}]",
               rhomin, rhomax, tempmin, tempmax, yemin, yemax, ymumin, ymumax) ;

    return leptonic_eos_4d_t(
        v_baryon, v_logrho, v_logT, v_ye, v_ymu,
        v_ele, v_muon,
        cold_tabs, cold_lrho,
        rhomax, rhomin,
        tempmax, tempmin,
        yemax,  yemin,
        ymumax, ymumin,
        baryon_mass, energy_shift,
        epsmin, epsmax_val,
        hmin,   hmax,
        temp_floor,
        ye_atm, ymu_atm,
        atm_beta_eq
    ) ;
}

} /* namespace grace */
