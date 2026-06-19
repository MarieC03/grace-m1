# GRMHD is the only evolved-equations module GRACE currently supports.
# (Burgers and scalar-advection were dropped years ago; their source
# files no longer exist.)  Optional add-ons sit alongside it.
#option(GRACE_ENABLE_GRMHD "Enable GRMHD equation module" ON)

# M1 radiation transport.
#
# GRACE_M1_NU_SPECIES selects the grey neutrino scheme:
#   0 -> no neutrinos (photon-only transport; requires GRACE_M1_PHOTONS)
#   1 -> nue
#   3 -> nue, nuebar, nux
#   5 -> nue, nuebar, numu, numubar, nux
# The species form strict supersets, so the source tests it with
# `#if GRACE_M1_NU_SPECIES >= {1,3,5}`.  Any neutrino species turns on the
# shared M1 infrastructure (GRACE_ENABLE_M1); the photon block does too.
option(GRACE_ENABLE_M1 "Enable M1 radiation transport" OFF)
set(GRACE_M1_NU_SPECIES "0" CACHE STRING "Grey neutrino species evolved by M1 (0, 1, 3, or 5)")
set_property(CACHE GRACE_M1_NU_SPECIES PROPERTY STRINGS 0 1 3 5)
if(NOT GRACE_M1_NU_SPECIES MATCHES "^(0|1|3|5)$")
    message(FATAL_ERROR "GRACE_M1_NU_SPECIES must be 0, 1, 3, or 5 (got '${GRACE_M1_NU_SPECIES}').")
endif()
if(GRACE_M1_NU_SPECIES GREATER 0)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 neutrino transport: ${GRACE_M1_NU_SPECIES}-species.")
endif()

# Photon M1 transport: a single, explicitly-addressed radiation block with
# its own variables and rates, decoupled from the neutrino species (no
# lepton-number coupling).  Implies the M1 infrastructure.
option(GRACE_M1_PHOTONS "Enable photon M1 transport block" OFF)
if(GRACE_M1_PHOTONS)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 photon transport enabled.")
endif()

# Eikonal optical-depth solver (Neilsen+ 2014): per-species neutrino optical
# depths stored as inert (zero-flux) evolved variables so they inherit ghost
# exchange + AMR prolongation + BCs, updated by a once-per-step min-path
# relaxation sweep.  Implies the M1 infrastructure.  Off by default — the
# tau fields cost flux-buffer memory, only paid when the eikonal tau policy
# is used.
option(GRACE_M1_OPTICAL_DEPTH "Enable the eikonal neutrino optical-depth solver" OFF)
if(GRACE_M1_OPTICAL_DEPTH)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 eikonal optical-depth solver enabled.")
endif()

# Debug: write EAS-rate diagnostic fields into dedicated aux slots so they can
# be dumped via the "rates" output group and compared cell-by-cell against the
# reference (FIL).  Covers the per-species equilibrium fugacity eta_nu = mu_nu/T
# (eta_nu1..5) and the matter chemical potentials feeding it (mu_e, mu_mu, mu_p,
# mu_n) -- the latter localise where mu_n-mu_p collapses so eta_nu is nonzero at
# beta equilibrium.  Off by default; costs a handful of extra aux scalars when
# enabled.  Implies the M1 infrastructure.
option(GRACE_M1_DEBUG_EAS "Output EAS-rate diagnostics (eta_nu, mu_e/mu_mu/mu_p/mu_n) to aux" OFF)
if(GRACE_M1_DEBUG_EAS)
    set(GRACE_ENABLE_M1 ON)
    message(STATUS "M1 EAS-rate diagnostics enabled.")
endif()

# Consistency: M1 must transport something.  Catches an enabled M1 build (e.g.
# a stale GRACE_ENABLE_M1=ON in the cache, or photon/debug flags alone) that
# selects zero neutrino species and no photons.
if(GRACE_ENABLE_M1 AND GRACE_M1_NU_SPECIES EQUAL 0 AND NOT GRACE_M1_PHOTONS)
    message(FATAL_ERROR
        "M1 is enabled but GRACE_M1_NU_SPECIES=0 and GRACE_M1_PHOTONS=OFF — "
        "nothing to transport.  Set -DGRACE_M1_NU_SPECIES={1,3,5} and/or "
        "-DGRACE_M1_PHOTONS=ON.")
endif()

# bns_nurates is an optional header-only submodule providing one of the M1
# EAS providers.  Build fine without it: the provider is compiled out and
# selecting m1.eas kind "bns_nurates" in a parfile errors at startup.
set(GRACE_HAVE_BNS_NURATES OFF)
if(GRACE_ENABLE_M1)
    if(EXISTS "${CMAKE_SOURCE_DIR}/extern/bns_nurates/include/bns_nurates.hpp")
        set(GRACE_HAVE_BNS_NURATES ON)
        message(STATUS "bns_nurates submodule found: EAS provider enabled.")
    else()
        message(STATUS "bns_nurates submodule NOT found — building M1 without "
                       "the bns_nurates EAS provider.  To enable it: "
                       "git submodule update --init extern/bns_nurates")
    endif()
endif()
option(GRACE_FREEZE_HYDRO "Freeze hydrodynamics evolution" OFF)

# First-Order Flux Correction.  Stage-3 flagger + stage-4 donor-cell/LLF
# recompute at faces of cells whose tentative HO update would have required
# c2p flooring.  Default ON — it's a safety net with negligible cost on
# clean states.  Disable for symmetry-preservation diagnostics or when
# bisecting a flux-related bug.
option(GRACE_ENABLE_FOFC "Enable First-Order Flux Correction" ON)

# GRMHD Riemann solver selection (compile-time).
#   HLL — 2-wave HLLE (default).
#   ADV — "advanced": HLLD for MHD, HLLC for pure-hydro states; HLLE fallback.
#   LLF — Local Lax-Friedrichs (Rusanov): symmetric HLL with cmax/cmin
#         clamped to the largest local |fast magnetosonic speed|.
set(GRACE_RIEMANN_SOLVER "HLL" CACHE STRING
    "GRMHD Riemann solver (HLL|ADV|LLF)")
set_property(CACHE GRACE_RIEMANN_SOLVER PROPERTY STRINGS HLL ADV LLF)
if(NOT GRACE_RIEMANN_SOLVER MATCHES "^(HLL|ADV|LLF)$")
    message(FATAL_ERROR
        "GRACE_RIEMANN_SOLVER=${GRACE_RIEMANN_SOLVER} is not one of HLL, ADV, LLF.")
endif()
message(STATUS "GRMHD Riemann solver: ${GRACE_RIEMANN_SOLVER}")

# EMF reconstruction scheme for constrained transport (compile-time).
#   GS  — Gardiner-Stone EMF (default): edge EMFs reconstructed in-kernel from
#         face fluxes during the directional flux sweep.
#   UCT — Upwind constrained transport: edge EMFs computed in a separate pass
#         from face-centered vtilde and the staggered B field.
set(GRACE_EMF_SCHEME "GS" CACHE STRING
    "CT EMF reconstruction scheme (GS|UCT)")
set_property(CACHE GRACE_EMF_SCHEME PROPERTY STRINGS GS UCT)
if(NOT GRACE_EMF_SCHEME MATCHES "^(GS|UCT)$")
    message(FATAL_ERROR
        "GRACE_EMF_SCHEME=${GRACE_EMF_SCHEME} is not one of GS, UCT.")
endif()
message(STATUS "EMF scheme: ${GRACE_EMF_SCHEME}")

if( GRACE_ENABLE_FUKA )
    message(STATUS "FUKA module enabled.")
endif()
