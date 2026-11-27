Harry spectral tables in five-species grey M1
===========================================

The 23 November 2026 SFHo and DD2 releases supply both ILD moments with HDF5
shape (272,35,35,35). Each charge has 136 triangular energy pairs on a
16-bin grid. The 14 November releases lack these complete arrays. Startup
checks both shapes when tabulated decay is requested.

Provider selection
------------------

``m1.eas.weakhub_format: auto`` detects spectral versus historical grey HDF5
schemas during setup. Each MPI rank retains its selection until the tables
are finalized. An unreadable file during automatic selection is an error.
``pair_treatment: auto`` chooses evolved partners for spectral tables
and the historical standard treatment otherwise. Explicit historical modes
remain available with the historical providers. Harry tables require a
five-species pair-kernel build, leptonic EOS, ``dilute_muon_cc_target: true``,
``add_ele_contribution: true``, and ``dilute_muon_suppression: false``. The
baryonic EOS must be electron-free and consistent with the rate tables.
For these releases use the corresponding
``sfho_compose_noele_0-5ye.h5`` or ``dd2_compose_noele_0-5ye.h5`` with
the matching November lepton file. Harry used the electron-inclusive
CompOSE files to obtain nuclear quantities during rate generation.
With the 13 November lepton tables and the electron-free baryon tables,
set ``eos.leptonic.add_photon_gas: true`` for LTE photon pressure,
specific energy and entropy. The option defaults to false for existing
EOS combinations, which can already contain photons. The photon free
energy is independent of charge and leaves the lepton chemical potentials
unchanged. Use ``eos.tabulated_eos.interpolation: linear`` with the
13 November SFHo lepton file. At 1e11 g/cm3, 20 MeV, Ye = 0.1 and
Ymu = 0.0005 its Hermite free-energy interpolant yields a nonmonotonic
specific energy and an ambiguous temperature. The native muon entropy
also disagrees with the ideal-gas reference in this regime. The linear
interpolant still uses the supplied entropy and chemical potentials.
CC and ES lookups convert the evolved rest-mass density to the
release convention at fixed baryon number, using the loaded EOS baryon mass.
GRACE reads each rate process on its own axes. For different baryon and
lepton resolutions the EOS uses native, regular logarithmic density and
temperature coordinates for each component. The lepton axes must cover the
baryon domain in an identifiable unit convention. Equal-resolution legacy
tables retain their row registration. The charge grids have independent sizes.

``harry_decay_source: table`` is the default for this provider. For controlled
comparisons only, ``harry_decay_source: spectral`` explicitly substitutes the
existing analytic spectral decay kernel. This is a mixed-provider model, not
validation of Harry's ILD table. The ``muon_decay`` master switch is unchanged.
``guo_response_table`` gives the path to the author's ``S_Table_Tv.dat`` for NN
bremsstrahlung; it is not included in the WeakHub HDF5 files.

Physics conventions
-------------------

The reader interpolates opacities and Phi0 in their stored natural-log units.
Charged-current raw absorption is converted to stimulated absorption before
grey averaging, using stable logarithmic factors. Emission is reconstructed
from the charged-current equilibrium chemical potential. The dilute-muon
potential regularization applies to the reaction target; EOS thermodynamics
are unchanged. Grey FD normalization uses the continuum moments, while the
rate integrals use the supplied finite energy grid. This is a grey closure,
not a multi-group Boltzmann evolution.

FD2 and FD3 for this provider use fixed-cost Chebyshev evaluations of the
scaled integrals with reflection identities for positive degeneracy. The
historical provider's approximate fits remain unchanged. This avoids amplifying
their small normalization/mean-energy errors near muon thresholds. Tests use
independently generated high-precision values, and the separate CC audit uses
adaptive integration rather than the production coefficients.

The supplied raw opacity floor is 1e-300. Logarithmic arithmetic prevents
intermediate overflow but cannot recover information lost below this floor.
Strong-degeneracy tests of arithmetic finiteness do not establish physical
accuracy when detailed balance amplifies a floored absorption coefficient.

Thermal pairs include electron, muon and tau flavours. Pair kernels enter the
existing conservative reaction contraction with Phi0/2 (the angular-average
convention of WeakHub Eq.45). The tau field represents nu_tau+antinu_tau;
opacities are averaged rather than multiplied by the species count twice.
The current isotropic pair and decay models use Phi0. The reader checks the
shapes of their companion Phi1 datasets without allocating their values.
The four raw absorption arrays supply only charged-current reactions.
Elastic scattering uses six energy spectra at (rho, T, Ye+Ymu). The tau
opacity is the arithmetic mean of its two spectra. Guo's response supplies
the NN kernel on the fly. Detailed balance relates the microscopic kernels.
The grey emissivity and loss are separate contractions with the evolved
partner occupations, including final-state blocking.

Bounds and validation limits
---------------------------

Active CC lookups reject out-of-domain compositions instead of combining a
clamped opacity with chemical potentials from another state. Below the
flavour's density/temperature threshold CC is zero, following the supplied
reader. Upper density/temperature violations are errors. Pair/ILD kernel
thermodynamic bounds and finite-grid realizability are checked explicitly.

The Guo response domain is T=2..50 MeV, free-nucleon density=1e-4..1 fm^-3,
Yp=0..0.5, and omega approximately 0.0501..398.1 MeV. As in Harry's supplied
routine, its contribution is zero outside that support. This can omit NN
emission in hot or dilute matter and at high pair energies. It must be
reported in comparisons; it is not an extrapolation or a proof of BNS-wide
accuracy.

Validation records, source snapshots, EOS/data provenance and comparisons
are archived in GRACE_runs/HOTTOV_5spec/harry_tables_20261117. Mixed-provider
closed-box runtimes are source-update timings, not full GRACE/BNS step costs.
The follow-up EOS/source/independent-CC audit is archived under
GRACE_runs/HOTTOV_5spec/cc_boundary_solver_20261118. Numerical fixes do not
extend the thermodynamic coverage of the supplied EOS or interaction tables.

To validate the November 23 files, configure a test build with
``-DGRACE_HARRY_RELEASE_DIR=/path/to/releases`` and run
``ctest --output-on-failure -R harry_release`` in the build directory. Both
files must be present with their published campaign checksums. The ordinary
synthetic reader tests run independently of the availability of these files.

Pair and atmosphere boundary policy
----------------------------------

For electron-positron pairs, GRACE holds the amplitude at the nearest
tabulated electron degeneracy when the nonnegative physical value lies
outside the grid (0.05 to 100 in this release). The inverse/forward ratio
uses exp[-(E+Eprime)/T] at the actual matter temperature. This preserves the
LTE fixed point. The absolute rate beyond either face is an approximation.
Negative degeneracy and out-of-range matter temperature are rejected. The reader
does not reproduce every boundary approximation in Harry's supplied reader.
For electron-positron pairs, his non-debug reader sets the kernel to zero
below its temperature minimum, clamps higher temperature and electron
chemical degeneracy, and uses the clamped temperature in detailed balance.
For tabulated ILD GRACE follows Harry's non-debug evaluator. A temperature
or either charged-lepton degeneracy below its minimum gives a zero ILD
kernel for both charges. An upper excursion uses the maximum of the
affected axis in both the lookup and detailed-balance exponent. These
edge prescriptions approximate the rate outside the tabulated domain.
The other reactions keep their own domains. The in-domain exponent for
mu-minus decay is (mu_mu - mu_e - E1 - E2)/T. The mu-plus channel reverses
the chemical-potential difference.

True GRACE atmosphere cells already bypass the rate calculation: below
rho_fl*(1+atmo_tol), ordinary rates are floored and PAIR_ACTIVE is zero.
With the campaign's positive hot-TOV rho_cut, initial data below that cut are
written as atmosphere; the remaining stellar cells start at T=20 MeV. Thus
cold atmosphere alone does not imply startup failure. Non-atmosphere cells
near interfaces or during evolution can still leave the pair domain and
will be rejected. A later channel-specific boundary extension must preserve
other active channels and use a consistent state for each forward/inverse
reaction pair. Silently disabling the entire thermal network is inappropriate.

Radiation-spectrum selection
----------------------------

With ``fermi_fofc: true``, the reconstructed faces and the predicted transport
update use the same finite energy support as the evolved pair spectra.
At fixed comoving number, the minimum energy comes from filling bins from
the bottom up. A constant occupation gives the maximum mean energy of a
positive-temperature Fermi spectrum. The limiter can correct a shared flux.
An IMEX stage can also leave this domain through its source combination.

Harry charged-current grey averages always obtain the evolved fluid-frame
radiation mean energies, even when temperature_correction is false. That
switch controls the historical multiplicative prescription; Harry's provider
uses the radiation spectrum directly. Matter emissivity remains independent
of the radiation mean energy. Empty/floor-reset radiation retains the existing
matter-temperature fallback. Provider auto-selection is a configuration
default, not a statement that this treatment is validated for general BNS use.
