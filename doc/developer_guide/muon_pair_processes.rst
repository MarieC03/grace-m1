.. _grace-muon-pair-processes:

Muon-neutrino pair processes in grey M1
=======================================

Scope and conclusion
--------------------

This note audits the treatment of

.. math::

   e^-+e^+ \rightleftarrows \nu_\mu+\bar\nu_\mu

in the five-species EAS/M1 implementation.  The same transport issue occurs for
nucleon--nucleon bremsstrahlung and any other reaction that creates or destroys
a neutrino and its antineutrino partner together.

GRACE already *produces* both muon species: ``add_pair_process_emission`` puts
one single-species share of the pair rate in ``NUMU`` and the same share in
``NUMUBAR``.  It also gives the combined tau block ``NUX`` twice that share.
The missing physics is the inverse reaction.  The code currently turns each
species' emission into an independent absorption opacity with Kirchhoff's law
and advances each species in an independent implicit solve.  A genuine pair
collision term instead depends on the distribution and angular moments of the
*partner* species.

The recommended implementation order is:

1. Separate pair contributions from ordinary one-particle EAS coefficients.
2. Replace ``Q_pair / B`` by finite number- and energy-averaged opacities
   computed from an isotropic annihilation kernel, following Gieg et al.
   (2026).  This is the best first production implementation because it fits
   GRACE's existing EAS interface.
3. Add the inexpensive partner-density correction of Betranhandy & O'Connor
   (2025) outside trapped regions.
4. Ultimately advance ``NUMU`` and ``NUMUBAR`` in one cell-local implicit
   solve.  The number part of that solve must use exactly the same pair source
   for both species, so that the pair process cannot change muon lepton number.

The existing fugacity clamp and the smooth ``muon_rate_gate`` are stabilizers,
not a derivation of the pair opacity.  In particular, a gate intended for
reactions that require charged muons must not suppress neutral-current
production of a muon-neutrino pair.


Process inventory: do not mix three meanings of "muonic"
--------------------------------------------------------

The neutral-current thermal reactions relevant to the five-species transport
are

.. math::

   e^-+e^+ &\rightleftarrows \nu_i+\bar\nu_i,\\
   \gamma^*_{T,L,A,M} &\rightleftarrows \nu_i+\bar\nu_i,\\
   N+N &\rightleftarrows N+N+\nu_i+\bar\nu_i,

where :math:`i=e,\mu,\tau`.  These reactions produce muon-flavour neutrinos;
they do *not* require a charged muon in the initial or final matter state.
Consequently their :math:`\nu_\mu\bar\nu_\mu` rates must remain active when
:math:`Y_\mu\rightarrow0`.

There are separate reactions which genuinely involve charged muons:

.. math::

   \nu_\mu+n &\rightleftarrows p+\mu^-,\\
   \bar\nu_\mu+p &\rightleftarrows n+\mu^+,\\
   \mu^- &\rightleftarrows e^-+\nu_\mu+\bar\nu_e,\\
   \mu^+ &\rightleftarrows e^++\bar\nu_\mu+\nu_e.

The last two are pair-like because two different neutrino species are coupled,
but they conserve a different combination of electron and muon lepton number.
They need a cross-flavour pair descriptor, not the
``NUMU``--``NUMUBAR`` descriptor used by the neutral-current reactions.
The low-density/low-temperature cutoff of Gieg et al. applies to these
charged-muon reactions, not to electron-pair annihilation or nucleon
bremsstrahlung.

A still different, optional channel is
:math:`\mu^-+\mu^+\rightleftarrows\nu_i+\bar\nu_i`.  It is not present in the
current analytic rates, the current GRACE WeakHub table interface, or the Gieg
reaction set.  It requires the massive charged-lepton kernel and separate
:math:`\mu^-` and :math:`\mu^+` distributions; replacing :math:`m_e` by
:math:`m_\mu` in the ultrarelativistic electron formula is not valid.  It is a
lower-priority extension for merger temperatures because of the
:math:`2m_\mu c^2` threshold and the usually small antimuon population.


Exact collision operator
------------------------

Let :math:`f=f_{\nu_\mu}(\epsilon,\Omega)` and
:math:`\bar f=f_{\bar\nu_\mu}(\bar\epsilon,\bar\Omega)`.  In the fluid frame,
the pair contribution to the Boltzmann equation is

.. math::

   \mathcal C_\nu[f,\bar f] =
   (1-f)\int d\Pi_{\bar\nu}\,
       R^{\rm pro}(\epsilon,\bar\epsilon,\cos\Theta)(1-\bar f)
   -f\int d\Pi_{\bar\nu}\,
       R^{\rm ann}(\epsilon,\bar\epsilon,\cos\Theta)\bar f,

where

.. math::

   d\Pi_{\bar\nu} =
   \frac{\bar\epsilon^2\,d\bar\epsilon\,d\bar\Omega}{(hc)^3}.

The antineutrino equation is obtained by interchanging barred and unbarred
quantities.  This equation shows three facts that any approximation should
retain:

* production is Pauli blocked by *both* final states;
* annihilation of one species is proportional to the partner distribution;
* every microscopic event changes the two particle numbers equally.

For electrons and positrons in thermal equilibrium, detailed balance gives

.. math::

   R^{\rm pro} =
   \exp[-(\epsilon+\bar\epsilon)/T]R^{\rm ann}.

Consequently the operator vanishes for Fermi--Dirac distributions satisfying
:math:`\eta_{\nu_\mu}+\eta_{\bar\nu_\mu}=0`.  Pair reactions alone constrain
the *sum* of the chemical potentials; they do not erase the conserved muon
lepton asymmetry.

Expanding the kernels in Legendre polynomials,

.. math::

   R(\epsilon,\bar\epsilon,\cos\Theta)
   = \sum_{\ell}(2\ell+1)R_\ell(\epsilon,\bar\epsilon)
     P_\ell(\cos\Theta),

makes the coupling between moments explicit.  The :math:`\ell=0` term couples
the energy density of one species to the energy density of its partner.  The
:math:`\ell=1` term also couples their fluxes; higher terms couple pressure and
higher moments.  Keeping only :math:`R_0` is therefore an isotropic
emission/absorption approximation, not the full pair operator.


Isotropic spectral and grey opacities
-------------------------------------

Neglecting final-state blocking for the purpose of constructing an effective
coefficient, the isotropic collision term can be written

.. math::

   \mathcal C^{(0)}_\nu(\epsilon)
   = j_p(\epsilon)-\kappa_p(\epsilon)f_\nu(\epsilon),

with

.. math::

   j_p(\epsilon) = \frac{4\pi}{(hc)^3}
      \int d\bar\epsilon\,\bar\epsilon^2
      R^{\rm pro}_0(\epsilon,\bar\epsilon),

.. math::

   \kappa_p(\epsilon) = \frac{4\pi}{(hc)^3}
      \int d\bar\epsilon\,\bar\epsilon^2
      R^{\rm ann}_0(\epsilon,\bar\epsilon)\bar f(\bar\epsilon).

The second equation is the essential one: pair opacity is a functional of the
partner radiation field.  It is not a material-only opacity.

Gieg et al. use the controlled approximation
:math:`\bar f\rightarrow\bar f_{\rm eq}` and define separate grey number
(:math:`i=0`) and energy (:math:`i=1`) opacities,

.. math::

   \kappa_{p,i}^{\rm LTE} =
   \frac{\int d\epsilon\,\epsilon^{2+i}\kappa_p(\epsilon)
                      f_{\nu,\rm eq}(\epsilon)}
        {\int d\epsilon\,\epsilon^{2+i}f_{\nu,\rm eq}(\epsilon)}.

The corresponding grey emission is

.. math::

   Q_{p,i}=c\,\kappa_{p,i}^{\rm LTE}B_i(T,\eta_\nu),
   \qquad
   B_i=\frac{4\pi}{(hc)^3}T^{3+i}\mathcal F_{2+i}(\eta_\nu).

This construction remains finite when :math:`\eta_\nu\ll-1`: the common
:math:`\exp(\eta_\nu)` factor cancels between numerator and denominator of the
grey average.  The partner distribution inside :math:`\kappa_p` also suppresses
the opposite, highly degenerate limit.  This is why kernel-first averaging is
better behaved than dividing a separately fitted, unblocked total emission by
a nearly vanishing blackbody integral.

Gieg et al. also introduce the smooth cutoff

.. math::

   \kappa_{a,i}\rightarrow
   \frac{\kappa_{a,i}}
   {[1+(\rho_{\rm th}/\rho)^5][1+(T_{\rm th}/T)^6]},
   \quad \rho_{\rm th}=10^{11}\ {\rm g\,cm^{-3}},\quad
   T_{\rm th}=2.5\ {\rm MeV}.

for reactions that contain a charged muon: muonic beta reactions and inverse
(anti)muon decay.  It is preferable to an abrupt zero and must be applied
consistently to the associated emissivity through detailed balance.  It must
*not* be applied to
:math:`e^-e^+\rightleftarrows\nu_\mu\bar\nu_\mu` or
:math:`NN\rightleftarrows NN\nu_\mu\bar\nu_\mu`; those neutral-current
channels do not require charged muons.  GRACE currently applies
``muon_rate_gate`` after the thermal rates have been added, and therefore gates
these channels incorrectly.


Channel-specific kernels
------------------------

Electron--positron annihilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`e^-e^+\rightleftarrows\nu_i\bar\nu_i`, the isotropic and dipole
kernels are obtained by integrating the electron energy with the electron and
positron occupation/blocking factors.  The weak couplings are flavour
dependent.  In the notation used by the existing analytic routine,

.. math::

   C_{ee}=(C_V-C_A)^2+(C_V+C_A)^2,

for electron flavour, whereas

.. math::

   C_{x}=(C_V-C_A)^2+(C_V+C_A-2)^2

for muon and tau flavour.  Thus ``add_pair_process_emission`` uses the correct
kind of coupling split, but its integrated blocking-factor fit is only a
production approximation.  It does not supply the partner-dependent inverse
kernel.  The kernel is not exactly symmetric under
:math:`\epsilon\leftrightarrow\bar\epsilon`; a table should therefore store the
oriented ``NUMU`` and ``NUMUBAR`` energy averages separately even though the
integrated *number* of particles created in the two equations must be equal.

Plasma process
~~~~~~~~~~~~~~

The Ruffert-style ``add_plasmon_decay_emission`` is a compact fit to the
transverse contribution.  A kernel treatment should instead sum the
transverse, longitudinal, axial and mixed vector--axial modes and store both
production and annihilation kernels.  WeakHub does this because the plasma
mode occupation and its inverse process have their own detailed-balance
bookkeeping.  The mixed term can also distinguish the oriented neutrino and
antineutrino energy spectra.

For an npe\ :math:`\mu` plasma, charged muons can contribute to the photon
polarization tensor.  This is not obtained by substituting
:math:`\eta_e\rightarrow\eta_\mu` in the present ``gamma`` fit.  A complete
extension needs
:math:`\Pi^{\alpha\beta}=\Pi_e^{\alpha\beta}+\Pi_\mu^{\alpha\beta}` with the
finite muon mass retained.  Since the plasma channel is normally subdominant
in merger remnants, the first implementation can retain the electron plasma
kernel and document that approximation.

Nucleon--nucleon bremsstrahlung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`NN\rightleftarrows NN\nu_i\bar\nu_i`, the commonly used isotropic
kernel is a sum over :math:`nn`, :math:`pp` and :math:`np` pairs.  The WeakHub
implementation takes the smaller of its degenerate and non-degenerate kernel
approximations.  This kernel depends on
:math:`\epsilon+\bar\epsilon` and is symmetric in the two neutrino energies,
which makes exact equality of the two LTE number rates easier to retain.
Bremsstrahlung is independent of charged-muon abundance and can dominate
heavy-lepton pair equilibration in dense, relatively cool matter.

The compact formula presently used by ``add_brems_emission`` comes from an
expression whose published normalization is for all four heavy-lepton species
combined.  GRACE treats its local ``Q_brems`` and ``R_brems`` as a
single-species value and then assigns four shares.  The coefficient therefore
needs a dedicated normalization comparison against the original FIL routine
or a kernel-table reference before this path is used for quantitative work;
the existing multiplicity test checks internal consistency but cannot detect a
common factor error.

Electron-flavour channels
~~~~~~~~~~~~~~~~~~~~~~~~~

All three neutral-current processes are physically allowed for
:math:`\nu_e\bar\nu_e` as well.  Omitting them is an approximation justified
only when charged-current beta reactions dominate.  The analytic path
currently comments out electron-flavour pair production, disables analytic
electron-flavour plasmon production, and never adds electron-flavour
bremsstrahlung.  A channel-resolved WeakHub/kernel implementation should allow
all flavours and make the approximation explicit rather than encode it in
commented assignments.


Partner correction outside LTE
------------------------------

The LTE-partner opacity is exact only when the partner field is in LTE.
Betranhandy & O'Connor propose the inexpensive correction

.. math::

   \kappa^*_{p,i,\nu} = F_{\bar\nu}\kappa^{\rm LTE}_{p,i,\nu},
   \qquad
   F_{\bar\nu} = \min\left(1,
      \frac{J_{\bar\nu}}{J_{\bar\nu,\rm eq}}\right)

in a grey implementation.  In a multigroup implementation the ratio contains
the energy-bin sums.  The correction captures the leading fact that a depleted
partner field cannot provide the LTE annihilation opacity.  It is explicit,
cheap, and the required comoving :math:`J` is already available from GRACE's M1
closure.

This is still an approximation.  It discards spectral correlations and the
:math:`R_1` angular term, and independent species updates do not enforce equal
number changes away from equilibrium.  It should be presented as an
intermediate model, not as the exact pair collision term.

The same physical point motivates the improved Monte-Carlo treatment of
Foucart et al. (2026), which evaluates rates using the simulated neutrino
energy distribution, including blocking factors.  That result supports making
the partner field an explicit input to the rate evaluation; it is not by
itself a grey-M1 closure because its remaining angular approximation and
packet representation differ from GRACE's moments.


Pair-conservative grey source
-----------------------------

For the number equations, a useful grey closure that preserves the defining
invariant is

.. math::

   S_N^{\rm pair} = R_p
   \left[1-
     \frac{N_{\nu_\mu}N_{\bar\nu_\mu}}
          {N^{\rm eq}_{\nu_\mu}N^{\rm eq}_{\bar\nu_\mu}}
   \right],

.. math::

   \left.\frac{dN_{\nu_\mu}}{dt}\right|_{\rm pair}
   =\left.\frac{dN_{\bar\nu_\mu}}{dt}\right|_{\rm pair}
   =S_N^{\rm pair}.

This bilinear expression is a grey closure, not a replacement for tabulated
kernels, but it has the right vacuum limit, vanishes on the equilibrium
manifold, permits a nonzero conserved neutrino asymmetry, and guarantees

.. math::

   \left.\frac{d}{dt}
   (N_{\nu_\mu}-N_{\bar\nu_\mu})\right|_{\rm pair}=0.

An implementation based directly on :math:`R_0` should use the kernel-weighted
analogue of this expression.  The common number source must be assembled once
and written to both equations from the same floating-point value.  Averaging
two independently computed number sources after the fact is less robust.

This closure has a cheap positivity-preserving backward-Euler update.  Write

.. math::

   \dot N_\nu=\dot N_{\bar\nu}=P-A N_\nu N_{\bar\nu},
   \qquad A=\frac{P}{N_\nu^{\rm eq}N_{\bar\nu}^{\rm eq}},

and let :math:`a,b` be the two densities at the start of a comoving source
substep :math:`\Delta\tau`.  With
:math:`N_\nu^{n+1}=a+\delta` and
:math:`N_{\bar\nu}^{n+1}=b+\delta`, the common increment obeys

.. math::

   (\Delta\tau A)\delta^2+
   [1+\Delta\tau A(a+b)]\delta+
   \Delta\tau(Aab-P)=0.

The root continuous at :math:`\Delta\tau\rightarrow0` can be evaluated without
subtractive cancellation as

.. math::

   \delta=\frac{-2C}{B+\sqrt{B^2-4A_qC}},\qquad
   A_q=\Delta\tau A,\quad B=1+\Delta\tau A(a+b),\quad
   C=\Delta\tau(Aab-P).

For :math:`A=0`, use :math:`\delta=\Delta\tau P`.  This is a useful first
``compute_implicit_muon_pair_number_update`` routine: it is scalar, has no
Newton iteration, preserves :math:`N_\nu-N_{\bar\nu}` to roundoff, and can be
operator-split from the ordinary one-particle number sources.  ``P`` and ``A``
should be obtained from the same kernel quadrature and tabulated directly;
constructing ``A`` at runtime by dividing two underflowing equilibrium
densities would recreate the current numerical problem.

For energy and momentum, the isotropic moment source has the form

.. math::

   S^\alpha_{\nu,\rm pair} =
   (\eta_{E,\nu}-\kappa_{E,\nu}[\bar J]J_\nu)u^\alpha
   -\kappa_{E,\nu}[\bar J]H^\alpha_\nu,

with a barred companion equation.  A production implementation should obtain
the two energy emissivities and opacities from the same kernel quadrature.  An
:math:`R_1` upgrade adds the partner-flux terms.  The sum of both radiation
energy changes is deposited in the fluid by the existing backreaction path;
the equal number changes cancel from the ``YMU`` update.


Current GRACE audit
-------------------

The current code has several correct building blocks:

* Five-species indexing is
  ``NUE, NUEBAR, NUMU, NUMUBAR, NUX``.
* In five-species mode, the analytic electron--positron rate is split as one
  share for ``NUMU``, one for ``NUMUBAR``, and two for the combined tau block.
  The total heavy-lepton multiplicity is therefore four.
* Distinct equilibrium degeneracies are constructed for ``NUMU`` and
  ``NUMUBAR``.
* The backreaction uses ``dN3-dN4`` for ``YMU``, which is exactly the invariant
  that a pair-conservative source should leave unchanged.

The gaps are:

* ``add_kirchhoff_absorption_opacity_from_QR`` converts the pair emission of
  each species independently into ``kappa_a`` and ``kappa_n``.
* ``compute_implicit_update<2>`` and ``compute_implicit_update<3>`` are called
  separately.  The number update is the scalar one-particle formula
  :math:`N^{n+1}=(N^n+\Delta t\eta_N)/(1+\Delta t\kappa_N/\Gamma)`.
* The rate bundle does not identify which part of ``eta`` and ``kappa`` came
  from a pair reaction, so the solver cannot enforce pair invariants.
* Muon degeneracies are clipped to :math:`[-5,5]` to prevent the
  ``Q/B`` inversion from producing enormous finite opacities.  This hides the
  divergent-opacity symptom instead of constructing finite opacities.
* ``muon_rate_gate`` now uses the smooth Gieg functional form, but it is applied
  to the *total* ``NUMU`` and ``NUMUBAR`` emissivity and opacity after
  electron-pair, plasmon and bremsstrahlung additions.  It therefore suppresses
  neutral-current pair creation when charged muons are dilute.  The gate must
  be moved to the charged-muon channel contribution before the neutral-current
  thermal rates are added.
* The WeakHub HDF5 loader reads only total ``kappa_a_en``, ``kappa_a_num`` and
  ``kappa_s`` arrays.  It reads no process mask or provenance metadata.
  ``compute_all_species_weakhub`` then adds analytic pair/plasmon/brems rates
  unconditionally when their run-time flags are enabled.  This is safe only for
  tables intentionally generated without those pair processes (for example a
  table named ``noPP``); a table containing them would double count them.
* A total-opacity WeakHub table cannot be used to rescale only the pair opacity
  by the evolved partner moment.  Channel-resolved tables or separate pair
  datasets are required.
* The comment above ``add_plasmon_decay_emission`` says that the WeakHub path
  enables electron flavour, but both current production call sites pass
  ``include_electron_flavour=false``.  Tests exercise ``true`` directly, so
  they do not detect this call-site/documentation disagreement.
* The optional ``bns_nurates`` adapter presently exposes only the library's
  four slots ``nue, anue, nux, anux``.  In a five-species GRACE build it writes
  the combined heavy result into ``KAPPAA3/ETAN3`` (which is ``NUMU``), then
  explicitly zeroes species 4 and 5.  It therefore does not yet implement a
  five-species muon transport path.  Moreover, opacities of aggregated species
  must be averaged, while emissivities are summed; summing both opacities and
  emissivities applies the species multiplicity twice in the absorption term.


Suggested EAS data model
------------------------

The present ``nu_rates_out`` is sufficient only for a linear one-particle
source.  Pair provenance must survive until the implicit update.  A minimal
interface is conceptually

.. code-block:: cpp

   enum class pair_channel { ee, plasmon, nn_brems, mu_decay, antimu_decay };

   struct pair_rates_out {
     // Oriented energy source for each evolved species.
     std::array<double, NUMSPECIES> eta_E{};
     std::array<double, NUMSPECIES> kappa_E_lte{};

     // Common-number bilinear model for each evolved pair.
     double P_N_numu_numubar{0.0};
     double A_N_numu_numubar{0.0};

     // Optional cross-flavour pairs for inverse muon decay.
     double P_N_numu_anue{0.0}, A_N_numu_anue{0.0};
     double P_N_numubar_nue{0.0}, A_N_numubar_nue{0.0};
   };

   struct nu_rates_all_out {
     std::array<nu_rates_out, NUMSPECIES> ordinary;
     pair_rates_out pair;
   };

The exact member layout can be flattened for Kokkos, but the separation is
essential.  ``ordinary`` contains beta absorption/emission and scattering.
``pair`` contains electron-pair annihilation, plasma decay and bremsstrahlung,
with optional cross-flavour muon-decay channels.  A channel mask in the table
metadata states which processes are present.

Recommended kernel-table datasets are, per channel and oriented species,
``pair_eta_E``, ``pair_kappa_E_lte``, ``pair_P_N`` and ``pair_A_N`` on the
``rho,T,Ye,Ymu`` axes.  Storing ``P_N`` and ``A_N`` directly avoids divisions
by exponentially small black-body moments.  If dipole kernels are added, also
store the coefficients needed for the partner-flux term.  The table should
carry ``n_species``, species labels, statistical weights, process flags,
kernel version, units and whether stimulated absorption is included.


Implementation map
------------------

Phase 1: kernel-derived coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add the pair bundle above and keep electron-pair, plasma and bremsstrahlung
   contributions separate from beta and scattering.
2. Generate tables of :math:`R_0^{\rm prod/ann}` for electron-pair
   annihilation, the plasma modes and nucleon bremsstrahlung using WeakHub or
   the Bruenn/Pons/Hannestad--Raffelt kernels.  Offline quadrature should
   tabulate the grey integrals over the same thermodynamic axes as the EAS
   source.  Store separate oriented energy averages for ``NUMU`` and
   ``NUMUBAR`` and one common number-pair rate.
3. Interpolate those coefficients in ``neutrinos_eas_op``.  Apply the smooth
   low-:math:`\rho`, low-:math:`T` cutoff only to channels involving a charged
   muon.  Do not gate neutral-current pair, plasma or bremsstrahlung rates.
4. Add unit tests for finiteness over at least
   :math:`\eta_{\nu_\mu}\in[-100,100]`, LTE detailed balance, and the
   :math:`\eta_{\bar\nu_\mu}=-\eta_{\nu_\mu}` symmetry.

The energy part of this phase can feed the existing per-species implicit
solver.  The number part should already use the scalar common-increment solve
above; it is small and removes an otherwise avoidable lepton-number error.
The energy closure still implements the Gieg et al. LTE-partner approximation,
not the exact coupled operator.

Phase 2: partner-aware opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Always compute ``fluid_frame_J<2>`` and ``fluid_frame_J<3>`` when pair
   reactions are enabled.
2. Compute the matching equilibrium energy densities with the same
   temperature, degeneracy, units and species weights used by the tables.
3. Multiply only the pair part of each opacity by the partner ratio
   :math:`F_{\bar\nu}`.  Do not scale beta or scattering coefficients.
4. Lag the ratio from the beginning of the implicit substep.  This keeps the
   existing solve linear in the partner field and is the least invasive stable
   implementation.

Tests should cover vacuum (:math:`F=0`), LTE (:math:`F=1`), boundedness, and
partner exchange equivariance.

Phase 3: coupled implicit pair solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In the cell-local source kernel, replace the separate species-2/species-3
   calls by one ``compute_implicit_muon_pair_update`` call.
2. Keep ordinary beta/scattering coefficients separate from pair coefficients.
   A split ordinary step followed by a pair step is acceptable initially;
   Strang splitting can be added if convergence tests require it.
3. Solve the two energy/flux systems with opacities evaluated from their
   partner's new moments.  The existing M1 closure supplies :math:`J`,
   :math:`H^\alpha` and derivatives needed by the block Jacobian.
4. Solve the two number equations with one shared pair increment.  A simple
   bilinear grey closure reduces this part to a positivity-preserving quadratic;
   a kernel closure can be included in the same Newton system.
5. Pass the resulting states to the existing aggregate backreaction.  Assert in
   tests that a pair-only update changes ``NRAD3`` and ``NRAD4`` by exactly the
   same value and leaves ``YMUSTAR`` bitwise unchanged.

The :math:`\nu_\tau+\bar\nu_\tau` ``NUX`` block cannot represent a tau lepton
asymmetry.  Its current combined treatment is consistent only with the assumed
symmetry :math:`f_{\nu_\tau}=f_{\bar\nu_\tau}`.  That is acceptable for the
present five-species model, but its opacity and emissivity must retain the
two-species multiplicity exactly once.


Files and routines to change
----------------------------

``include/grace/physics/eas_neutrino_rates_analytic.hh``
  Split ``rates_accum`` into ordinary and pair pieces.  Replace
  ``add_pair_process_emission``, ``add_plasmon_decay_emission`` and
  ``add_brems_emission`` by channel routines returning ``pair_rates_out``.
  Keep the present analytic formulas as an explicitly named
  ``analytic_emission_only`` fallback.  Do not pass their output through
  ``add_kirchhoff_absorption_opacity_from_QR`` for muon species.  Move
  ``muon_rate_gate`` so it acts only on charged-muon beta/muon-decay
  contributions.  Audit the all-four-species bremsstrahlung normalization.

``include/grace/physics/grace_weakhub_table.hh`` and
``src/physics/grace_weakhub_setup.cpp``
  Load channel-resolved pair datasets and HDF5 process metadata.  Reject an
  ambiguous table/configuration combination instead of silently adding an
  analytic channel to a table that may already contain it.  Preserve the
  existing rule that an aggregated ``NUX`` opacity is an average, whereas its
  emissivity is a sum.

``include/grace/physics/eas_policies.hh``
  Compute ``J`` (and later ``H``) for both partners, evaluate the optional
  partner correction, and write pair coefficients to dedicated auxiliary
  fields.  The ordinary totals written to ``ETA*``, ``KAPPAA*``, ``ETAN*`` and
  ``KAPPAAN*`` must exclude the pair-number contribution if that contribution
  is handled by the coupled update.

``include/grace/data_structures/variable_indices.hh`` and
``src/data_structures/variable_indices.cpp``
  Register the pair auxiliary fields.  A minimal first version needs common
  ``PAIR_PN_MU`` and ``PAIR_AN_MU`` plus oriented pair energy emissivity and
  opacity for species 3 and 4.  Add cross-flavour fields only when inverse
  muon decay is enabled.

``include/grace/physics/m1_helpers.hh``
  Add the stable quadratic common-increment routine.  Keep all quantities in
  either the comoving or lab convention throughout; do not mix the current
  ``Gamma`` factor into only one side of the bilinear term.

``include/grace/physics/m1.hh`` and ``src/evolution/evolve.cpp``
  Split ordinary and pair number updates.  Replace separate pair-number parts
  of ``compute_implicit_update<2>`` and ``<3>`` by
  ``compute_implicit_muon_pair_number_update``.  A later full implementation
  replaces their energy/flux calls by an eight-variable partner-coupled solve.
  The existing aggregate backreaction can remain, because equal number
  increments cancel exactly in ``dN3-dN4``.

``include/grace/physics/bns_nurates_grace.hh``
  Do not advertise this path as five-species capable until the library
  interface exposes independent ``numu``, ``anumu``, ``nutau`` and ``anutau``
  slots and accepts :math:`\mu_\mu`/``Ymu`` in its distribution model.  Fix the
  current species-3 mapping and sum-versus-average issue first.  BNS_NURATES
  already provides electron-pair and several bremsstrahlung kernels and can
  use reconstructed M1 distributions, so it is a useful implementation base;
  it does not presently provide the WeakHub plasma-process kernel.

``parameters/m1.yaml``
  Replace the ambiguous booleans with, or supplement them by, a source selector
  such as ``thermal_pair_provider = analytic | weakhub_table | kernel_table``
  and channel flags.  Record whether electron flavours are included and whether
  the charged-muon cutoff is enabled.

``test/test_m1_analytic_rates.cpp`` and new pair-update tests
  Add reference-node tests for each physical kernel, a low-:math:`Y_\mu` test
  proving that neutral-current ``NUMU`` production is not gated, exact common
  pair-number increments for unequal initial fields, WeakHub double-counting
  rejection, and correct ``NUX`` aggregation.  Compare normalization against
  independent WeakHub or BNS_NURATES values rather than only testing internal
  species ratios.


Acceptance tests
----------------

At minimum, the completed implementation should satisfy:

* **Pair number:** pair-only
  :math:`\Delta N_{\nu_\mu}=\Delta N_{\bar\nu_\mu}` for arbitrary unequal
  initial fields.
* **Muon lepton number:** pair-only ``YMUSTAR`` is unchanged.
* **Detailed balance:** all pair energy, momentum and number sources vanish for
  LTE Fermi--Dirac moments with
  :math:`\eta_{\bar\nu_\mu}=-\eta_{\nu_\mu}`.
* **Vacuum:** empty radiation fields receive positive, equal particle-number
  emission.
* **Inverse reaction:** overpopulated neutrino and antineutrino fields give net
  annihilation and heat the fluid.
* **No partner:** annihilation tends to zero when either partner field tends to
  zero.
* **Finiteness:** coefficients remain finite and non-negative for the full EOS
  range without clipping :math:`\eta_\nu` merely to avoid division by zero.
* **Multiplicity:** in five-species mode the tau ``NUX`` emission is twice one
  muon-species emission; the total heavy-lepton production equals four
  single-species shares.
* **Convergence:** the partner-lagged and coupled implementations agree as the
  source substep is refined; a multigroup/kernel reference should be used for
  physical validation.


Primary references
------------------

* Shibata, Kiuchi, Sekiguchi & Suwa, *Truncated Moment Formalism for Radiation
  Hydrodynamics in Numerical Relativity* (2011),
  `arXiv:1104.3937 <https://arxiv.org/abs/1104.3937>`_.
* Musolino & Rezzolla, *A practical guide to a moment approach for neutrino
  transport in numerical relativity* (2024),
  `MNRAS 528, 5952 <https://doi.org/10.1093/mnras/stae224>`_.
* Ng et al., *Accurate muonic interactions in neutron-star mergers and impact
  on heavy-element nucleosynthesis* (2025),
  `arXiv:2411.19178 <https://arxiv.org/abs/2411.19178>`_.
* Gieg et al., *Consistent Treatment of Muons in Binary Neutron Star Mergers*
  (2026), `Phys. Rev. D 114, 043036
  <https://doi.org/10.1103/qmb8-q3cx>`_.
* Betranhandy & O'Connor, *A new approximation for heavy-lepton neutrino pair
  processes in simulations of core-collapse supernovae* (2025),
  `Phys. Rev. D 111, 103049 <https://doi.org/10.1103/fv13-5jtg>`_.
* Cheong et al., *General-relativistic Radiation Transport Scheme in Gmunu.
  II. Implementation of Novel Microphysical Library for Neutrino
  Radiation--Weakhub* (2024),
  `arXiv:2309.03526 <https://arxiv.org/abs/2309.03526>`_.
* Chiesa et al., *An open-source library for performance-portable neutrino
  reaction rates: Application to neutron star mergers* (2025),
  `arXiv:2412.04570 <https://arxiv.org/abs/2412.04570>`_.
* Pons, Miralles & Ibanez, *Legendre expansion of the neutrino--antineutrino
  pair kernel: influence of high order terms* (1998),
  `arXiv:astro-ph/9802333 <https://arxiv.org/abs/astro-ph/9802333>`_.
* Hannestad & Raffelt, *Supernova neutrino opacity from nucleon--nucleon
  bremsstrahlung and related processes* (1998),
  `arXiv:astro-ph/9711132 <https://arxiv.org/abs/astro-ph/9711132>`_.
* Guo et al., *Charged-Current Muonic Reactions in Core-Collapse Supernovae*
  (2020), `arXiv:2006.12051 <https://arxiv.org/abs/2006.12051>`_.
* Sugiura et al., *Leptonic and semi-leptonic neutrino interactions with muons
  in proto-neutron-star cooling* (2022),
  `arXiv:2211.06944 <https://arxiv.org/abs/2211.06944>`_.
* Foucart, *Neutrino transport in general relativistic neutron star merger
  simulations* (2023),
  `Living Reviews in Computational Astrophysics 9, 1
  <https://doi.org/10.1007/s41115-023-00016-y>`_.
* Foucart et al., *Impact of neutrino--electron scattering and an improved
  treatment of pair processes on binary neutron star mergers* (2026),
  `arXiv:2606.27425 <https://arxiv.org/abs/2606.27425>`_.
* Rath et al., *Assessing the Relative Importance of Neutrino Matter
  Interaction Channels in Post-Merger Remnants of Binary Neutron Stars*
  (2026), `arXiv:2605.29187 <https://arxiv.org/abs/2605.29187>`_.
