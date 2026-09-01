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
# NB: GRACE_ENABLE_M1 is NOT a user option -- it is derived from the flags
# below and published as a hidden INTERNAL cache entry further down.  It used
# to be an option() that the implication blocks overwrote with a plain set(),
# which shadowed the cache instead of writing to it: every M1 build tree then
# reported GRACE_ENABLE_M1=OFF in ccmake while M1 was actually compiled in.
set(GRACE_M1_NU_SPECIES "0" CACHE STRING "Grey neutrino species evolved by M1 (0, 1, 3, or 5)")
set_property(CACHE GRACE_M1_NU_SPECIES PROPERTY STRINGS 0 1 3 5)
if(NOT GRACE_M1_NU_SPECIES MATCHES "^(0|1|3|5)$")
    message(FATAL_ERROR "GRACE_M1_NU_SPECIES must be 0, 1, 3, or 5 (got '${GRACE_M1_NU_SPECIES}').")
endif()
if(GRACE_M1_NU_SPECIES GREATER 0)
    message(STATUS "M1 neutrino transport: ${GRACE_M1_NU_SPECIES}-species.")
endif()

# Muon sector: evolve the muon fraction Y_mu as an independent fluid
# composition variable (conserved YMUSTAR_, primitive/aux ymu, c2p muon
# resets, muonic backreaction).  Orthogonal to M1 -- valid at any species
# count and with M1 off entirely (pure GRMHD + muonic EOS).
#
# FORCED ON at 5 species: numu/numubar transport is meaningless without a
# muon sector, so the invariant GRACE_M1_NU_SPECIES==5 => GRACE_ENABLE_MUONS
# holds by construction (grace_config.h.in #errors if it is ever violated).
# Deliberately NOT auto-cleared when the species count drops back below 5 --
# that would defeat "explicitly switchable"; turn it off by hand.
#
# The forcing writes through with CACHE ... FORCE so ccmake shows the real
# value and a reconfigure is idempotent.
option(GRACE_ENABLE_MUONS "Evolve the muon fraction Y_mu (forced ON when GRACE_M1_NU_SPECIES=5)" OFF)
if(GRACE_M1_NU_SPECIES EQUAL 5 AND NOT GRACE_ENABLE_MUONS)
    set(GRACE_ENABLE_MUONS ON CACHE BOOL
        "Evolve the muon fraction Y_mu (forced ON when GRACE_M1_NU_SPECIES=5)" FORCE)
    message(STATUS "Muon sector: ON (forced by GRACE_M1_NU_SPECIES=5).")
elseif(GRACE_ENABLE_MUONS)
    message(STATUS "Muon sector: ON (explicitly enabled, ${GRACE_M1_NU_SPECIES}-species).")
endif()

# Photon M1 transport: a single, explicitly-addressed radiation block with
# its own variables and rates, decoupled from the neutrino species (no
# lepton-number coupling).  Implies the M1 infrastructure.
option(GRACE_M1_PHOTONS "Enable photon M1 transport block" OFF)
if(GRACE_M1_PHOTONS)
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
    message(STATUS "M1 eikonal optical-depth solver enabled.")
endif()

# Write M1 diagnostic fields into dedicated aux slots, dumped via the
# "diagnostics" output group.  All of them are read straight out of the
# fugacity_state that neutrinos_eas_op already builds every substep, so they
# cost stores but no extra EOS lookups or root-finds (contrast FIL's
# compute_m1_diagnostics.cc, which recomputes everything at analysis time).
# Covers: the per-species equilibrium fugacity eta_nu = mu_nu/T (eta_nu1..5),
# the matter chemical potentials feeding it (mu_e, mu_mu, mu_p, mu_n), the raw
# beta-equilibrium offsets (mu_delta_npe, mu_delta_npmu), the nuclear
# composition already interpolated by the same EOS call (X_n, X_p, X_a, X_h,
# Abar, Zbar) and the beta-equilibration timescale ratio (beta_eq_tscale).
# Off by default; costs a handful of extra aux scalars when enabled.
# Implies the M1 infrastructure.
option(GRACE_M1_DIAGNOSTICS "Output M1 diagnostics (eta_nu, chemical potentials, composition) to aux" OFF)
if(GRACE_M1_DIAGNOSTICS)
    message(STATUS "M1 diagnostics enabled.")
endif()

# GRACE_ENABLE_M1 is fully DERIVED from the flags above -- any neutrino
# species, the photon block, the optical-depth solver or the diagnostics all
# require the shared M1 infrastructure.  Published as CACHE INTERNAL so it is
# hidden from ccmake (no knob left to display a stale value) and recomputed
# from scratch on every configure (no way for a stale ON to survive).
if(GRACE_M1_NU_SPECIES GREATER 0 OR GRACE_M1_PHOTONS OR GRACE_M1_OPTICAL_DEPTH OR GRACE_M1_DIAGNOSTICS)
    set(GRACE_ENABLE_M1 ON  CACHE INTERNAL "Derived: M1 infrastructure required" FORCE)
else()
    set(GRACE_ENABLE_M1 OFF CACHE INTERNAL "Derived: M1 infrastructure required" FORCE)
endif()

# Consistency: M1 must transport something.  Now only reachable by enabling
# the optical-depth solver or the diagnostics with no species and no photons.
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

# Recompute the CT edge EMF at FOFC-flagged edges (donor+LLF faces, then GS
# edge reassembly).  This is the original (pre-2026-06) behaviour, but the
# partial flagged-only EMF recompute breaks bit-exact discrete symmetry (the
# flux correction itself does not).  OFF => hydro-only FOFC: the flux
# correction still applies, but B evolves on the unmodified main-pass EMF —
# conservative, div-B-preserving, and symmetry-preserving.  Enable only to
# reproduce legacy results.  No effect unless GRACE_ENABLE_FOFC is also ON.
option(GRACE_FOFC_CORRECT_EMF "Recompute CT edge EMF at FOFC-flagged edges (legacy; breaks discrete symmetry)" OFF)

# Diagnostic: after each full step, dump conserved + face fluxes + fofc flags
# for cells with tau/D > 1 or eps > 0.5 to hot_flux_dump.<rank>.dat.  Used to
# pin the superheated-atmosphere-cell mechanism (does the cell drain?).  OFF by
# default; adds a full-grid scan + per-step host copy when ON.
option(GRACE_DUMP_HOT_CELLS "Dump fluxes at hot (high tau/D or eps) cells" OFF)

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
