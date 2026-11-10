Muon decay and inverse decay in grey M1
=======================================

Scope and recommended mode
--------------------------

The decay prescription can be selected independently of the thermal-pair
partner treatment with ``m1.eas.muon_decay_treatment``:

* ``auto`` (default) preserves the historical choice: ``standard`` pairs use
  the vacuum sink; ``equilibrium`` and ``evolved`` pairs use spectral decay.
* ``vacuum`` selects the existing emission-only approximation even with
  equilibrium or evolved thermal-pair kernels. It requires
  ``dilute_muon_suppression: true`` and retains the ordinary-rate muon gate
  and fugacity clamp. Thermal pairs still use their kernel EOS potentials.
  No spectral-decay matrix or connected four-flavour decay solve is enabled.
* ``spectral`` requires an equilibrium/evolved kernel pair mode and the
  unsuppressed additive EOS described below. Prefer evolved partners.

``muon_decay: false`` remains the master off switch. Kernel pair modes
continue to require ``betaeq_policy: off`` for either decay prescription.
Thus switching from Ken's timescale baseline also changes the rate closure;
it is not a single-process comparison. The optional vacuum/kernel combination
is a controlled approximation, not a substitute for spectral inverse decay.

This document describes the spectral decay kernel of builds configured with
``-DGRACE_M1_PAIR_KERNELS=ON`` (``pair_treatment: equilibrium`` or ``evolved``).
``m1.eas.muon_decay: true`` adds both charge-conjugate reactions:

.. math::

   \mu^- \rightleftharpoons e^-+\nu_\mu+\bar\nu_e,\qquad
   \mu^+ \rightleftharpoons e^++\bar\nu_\mu+\nu_e.

This is a leptonic charged-current interaction, not nucleon beta decay,
electron-positron annihilation or creation of a same-flavour neutrino pair.
The inverse reactions create charged muons. No extra evolved charged-muon
field is needed: the existing net ``Ymu`` and thermal leptonic EOS describe
the charged-lepton bath. The EOS determines both signs of charged leptons.

Use ``pair_treatment: evolved`` for the event-conserving treatment.
``equilibrium`` is a deliberately retained, literal equilibrium-partner
comparison approximation. It has the correct LTE fixed point but NOT equal
daughter counts away from LTE. A startup warning and the diagnostic described
below expose this limitation. It must not be advertised as an equally
conservative decay prescription.

The kernels use finite electron/muon masses, tree-level weak interactions,
charged-lepton blocking and an isotropic angular average (R0). They do not
include radiative corrections, flavour-exchange scattering, oscillations or
the other crossed charged-lepton pair-annihilation channels. Existing thermal
pair kernels are unchanged. This is a grey closure, not energy-dependent
Boltzmann transport and not an exact angular annihilation calculation.

Literature and choice of normalization
-------------------------------------

* Guo et al., `Charged-Current Muonic Reactions in Core-Collapse Supernovae
  <https://arxiv.org/html/2006.12051>`_, section II.2 and Appendix A: finite-mass
  inverse-decay collision kernel and chemical-potential detailed balance.
* Sugiura et al., `Leptonic and semi-leptonic neutrino interactions with muons
  in proto-neutron star cooling <https://doi.org/10.1093/ptep/ptac118>`_,
  section 2.1 and Appendix A.3: charged-lepton phase-space integrals and
  partner-dependent opacity. PNS opacity rankings are not evidence of the
  quantitative importance of decay in a merger.
* Gieg et al., `Consistent Treatment of Muons in Binary Neutron Star Mergers
  <https://arxiv.org/html/2604.14225v2>`_, Table 1 and Appendix A: a merger
  treatment including muon decay and equilibrium-partner grey rates.
* Ng et al., `Accurate muonic interactions in neutron star mergers and impact
  on heavy-element nucleosynthesis <https://arxiv.org/html/2411.19178>`_,
  Table 1 and the microphysics discussion: five-species merger transport with
  muonic nucleon beta reactions. Muon decay is not in that listed reaction
  set; charged-muon creation by beta reactions should not be confused with
  inverse muon decay.

The implementation below independently reduces the phase-space integral.
Its spin normalization and daughter orientation are checked against the
tree-level vacuum width and the different Michel energy moments. No claim
that a named implementation is wrong is inferred merely from its use of an
equilibrium-partner approximation.

Setup
-----

Example additions to a physically compatible simulation parfile:

.. code-block:: yaml

   eos:
     eos_type: leptonic
     leptonic:
       add_ele_contribution: true
       dilute_muon_suppression: false
       # Requires an ELECTRON-FREE baryonic table and matching lepton tables.
   m1:
     eas:
       kind: neutrino_weakhub    # or neutrino_analytic
       betaeq_policy: off
       muon_decay: true
       pair_treatment: evolved  # equilibrium only for comparison
       pair_quadrature_order: 16
       muon_decay_kernel_order: 24
       weakhub_table_content: none   # or auto: read from the table's companion _tableinfo.log
       plasmon_decay: false     # or explicitly select transverse_mass model

``weakhub_table_content`` is read from the table's companion log (``auto``) or
declared; it is a statement of what the table carries, NOT a filter.  The
kernel modes need ``none``: inclusive grey opacities cannot be decomposed by
this interface. The analytic
ordinary-rate provider does not acquire muonic nucleon charged-current rates
from this change; those still require the appropriate WeakHub input.
The explicit thermal-pair framework remains heavy-flavour only: this change
does not add the electron-flavour thermal pairs missing from the analytic
baseline. Electron pair channels supplied inside an ordinary WeakHub table
retain that table's grey treatment. The content check concerns the
heavy-flavour channels added explicitly by the pair framework.

Required build: five neutrino species, ``GRACE_ENABLE_MUONS``, and the existing
``bns_nurates`` support used by the new pair framework. Other EAS providers,
standard pair treatment and beta-equilibrium remapping policies are rejected.
``muon_decay`` defaults to false; historical EOS suppression defaults to true.
The default standard/no-decay path retains its previous behaviour. Existing
``evolved`` thermal-pair runs also receive the transport-opacity correction
described below, even when muon decay is disabled.

Why the EOS option matters: the historical dilute-muon prescription sets or
ramps the muon chemical potential to zero and suppresses muon pressure,
energy and entropy below a net-Ymu threshold. That is not a thermal muon gas
with the chemical potential used by the decay integral. Disabling suppression
restores the tabulated terms, including at the floor; it also avoids the
historical charge-axis chemical-potential substitutions. The loader sets this
choice before generating its cold EOS slice. It does not extend the EOS table
domain: positive Ymu floors and composition boundaries remain limitations.

In the required additive, unsuppressed configuration the physical sums are

.. math::

   P=P_b(\rho,T,Y_e+Y_\mu)+P_e(\rho,T,Y_e)+P_\mu(\rho,T,Y_\mu),

.. math::

   \epsilon=\epsilon_b+\epsilon_e+\epsilon_\mu,\qquad
   s=s_b+s_e+s_\mu,

with the original EOS energy shift retained, and mu_e, mu_mu taken from the
corresponding lepton tables rather than the dilute-muon substitutions.

Microscopic kernel and equations
-------------------------------

Use natural units in the following derivation. Charged-lepton chemical
potentials include rest masses; neutrinos are massless and have one helicity.
For either charge, the FIRST daughter is the muon-flavour neutrino, the second
the electron-flavour neutrino. Write their energies as e, e'. Let c=+1 for
mu- decay and c=-1 for mu+ decay, and define

.. math::

   f_{\ell,c}(E)=[\exp((E-c\mu_\ell)/T)+1]^{-1},\quad
   \Delta_c=c(\mu_\mu-\mu_e),\quad \omega=e+e'.

For the minus reaction the spin-averaged squared amplitude is

.. math::

   \overline{|\mathcal M|^2}
   =64G_F^2(p_e\cdot q_{\nu_\mu})(p_\mu\cdot q_{\bar\nu_e}).

Charge conjugation keeps this oriented energy assignment. The two initial
muon spin states are included in the phase-space normalization below.
The matter occupation factors are

.. math::

   P_c=f_{\mu,c}(E_e+\omega)[1-f_{e,c}(E_e)],\qquad
   A_c=f_{e,c}(E_e)[1-f_{\mu,c}(E_e+\omega)].

They imply MICROSCOPIC detailed balance, independently of the radiation:

.. math::

   R^p_c(e,e')=e^{-(e+e'-\Delta_c)/T}R^a_c(e,e').

For completeness the actual positive two-dimensional integration in
``decay_total`` is specified here. M and m are the muon and electron masses.

.. math::

   s_{\max}=\min(4ee',(M-m)^2),\quad Q=\sqrt{\omega^2-s},\quad
   A=(M^2-m^2-s)/2,\quad B=\sqrt{A^2-sm^2},

.. math::

   E_-={A^2+Q^2m^2\over\omega A+QB},\qquad
   E_+={\omega A+QB\over s}.

The first endpoint is rationalized to avoid subtraction at small s. With

.. math::

   u={sE_e-\omega A\over QB},\quad a={e-e'\over Q},\quad
   t=ua,\quad t_2=u^2a^2+\tfrac12(1-u^2)(1-a^2),

the azimuth-averaged matrix element is

.. math::

   \langle\overline{|\mathcal M|^2}\rangle_\phi
   =16G_F^2[A(A+s)-sBt-B^2t_2].

The angle-averaged kernel sum is

.. math::

   R_c^{\rm tot}=R_c^p+R_c^a
   ={(\hbar c)^3\over\hbar}
   \int_0^{s_{\max}}ds\int_{E_-}^{E_+}dE_e\,
   {\langle\overline{|\mathcal M|^2}\rangle_\phi\over64\pi e^2e'^2Q}
   (P_c+A_c).

Energies are supplied in MeV and the displayed unit factor makes the kernel
cm^3/s. Electron and muon holes are evaluated directly as Fermi functions of
the negated argument, avoiding subtraction from one. The charged-lepton tail
is truncated at

.. math::

   E_{\rm cut}=\max[m,|\mu_e|,|\mu_\mu|-\omega]+48T.

Beyond that point both relevant occupation tails are exponentially small.
The inner Gauss-Legendre order controls BOTH the s and electron-energy
integrals. This finite-order quadrature must be converged for the matter
states of interest; detailed balance alone does not prove rate accuracy.

Instead of multiplying an underflowed absorption rate by a large exponential,
the code partitions the positive sum with a bounded logistic function:

.. math::

   x=(e+e'-\Delta_c)/T,\quad
   R_c^p={R_c^{\rm tot}\over1+e^x},\quad
   R_c^a={R_c^{\rm tot}\over1+e^{-x}}.

One bounded exponential evaluates both fractions without overflow or loss
of a small fraction through subtraction. No rate is switched off merely
because the net muon abundance is small; inverse decay can produce muons.

Spectral collision term and effective opacity
--------------------------------------------

Let ``w_i`` include the one-helicity measure
``dPi = e^2 de / (2 pi^2 (hbar c)^3)``. For a trial pair of spectra,

.. math::

   j_i=\sum_jw_jR^p_{ij}(1-f'_j),\qquad
   a_i=\sum_jw_jR^a_{ij}f'_j,\qquad
   C_i=j_i(1-f_i)-a_if_i=j_i-(j_i+a_i)f_i.

The effective spectral damping is j+a, not absorption a alone: j f is the
target-neutrino Pauli-blocking term. For one matrix entry define

.. math::

   I_{ij}=w_iw_j[R^p_{ij}(1-f_i)(1-f'_j)-R^a_{ij}f_if'_j].

The two daughters share a SINGLE event rate, but not an energy rate:

.. math::

   r=\sum_{ij}I_{ij},\quad
   Q_0=\sum_{ij}e_iI_{ij},\quad Q_1=\sum_{ij}e_jI_{ij}.

For each species, the unblocked emission moments and damping moments are

.. math::

   \eta_N=\sum_iw_ij_i,\quad \eta_E=\sum_iw_ie_ij_i,\quad
   L_N=\sum_iw_i(j_i+a_i)f_i,\quad L_E=\sum_iw_ie_i(j_i+a_i)f_i.

Thus r=eta_N-L_N and Q=eta_E-L_E. All physical-to-GRACE conversions reuse
``pairs::number_unit``, ``energy_unit`` and ``time_unit``; GRACE's number
variable is baryon-mass weighted. Kernels and source sums stay in CGS/MeV
until the final conversion, consistent with the existing pair implementation.

Equilibrium-partner option
--------------------------

Use matter-temperature FD spectra and the unsuppressed chemical potentials

.. math::

   \mu_{\nu_e}=\mu_e+\mu_p-\mu_n-Q_{np},\quad
   \mu_{\nu_\mu}=\mu_\mu+\mu_p-\mu_n-Q_{np},\quad
   \mu_{\bar\nu_\ell}=-\mu_{\nu_\ell}.

Consequently mu_numu+mu_antinue=mu_mu-mu_e, exactly the decay balance
condition. Optical-depth suppression or independent clipping would break
this identity. With decay enabled the four lepton-carrying species therefore
all use these raw potentials, also in the ordinary EAS coefficients.

Freeze the partner at that LTE spectrum. Evaluate the preceding moments with
both spectra at LTE, and define

.. math::

   B_N=\sum_iw_if_i^{\rm eq},\quad B_E=\sum_iw_ie_if_i^{\rm eq},\quad
   \kappa_N={L_N^{\rm eq}\over B_N},\quad
   \kappa_E={L_E^{\rm eq}\over B_E},\quad
   \eta_w=\kappa_wB_w.

The implementation evaluates BOTH emission and loss from the kernel; the
last equality is a checked consequence of balance, not a prescription for
the evolved-partner case. The grey source is eta_w-kappa_w M_w. For example,
if one daughter remains at LTE and the other is depleted, one source is zero
and the other positive: they cannot represent a common decay event count.

``decay_equilibrium_charge_defect`` reports the decay-only source
``m_b * d[n_b(Ye+Ymu)]/dproper_time`` in code units, evaluated at the
pre-update radiation state. It is not the integrated error after the source
solve or limiting. The evolved mode leaves this diagnostic zero.

Evolved-partner option and local implicit update
-----------------------------------------------

Reconstruct each of the four trial spectra on the SAME positive energy grid:

.. math::

   f_s(e)=[\exp(e/T_{\nu,s}-\eta_s)+1]^{-1},\quad
   \sum_iw_if_{s,i}=n_s,\quad \sum_iw_ie_if_{s,i}=J_s.

``n_s=N_s/Gamma_s`` and J_s come from the trial M1 closure, not the old
radiation state. All four reconstructions and all three reaction edges
(two decays and same-flavour muonic thermal pairs) participate in the same
nonlinear residual. Matter and the energy grid are held fixed during an
IMEX diagonal stage; the expensive kernel matrices are built once outside
Newton, not for each residual evaluation.

The R0 grey angular source uses

.. math::

   G_s^\alpha=Q_su^\alpha-\lambda_sH_s^\alpha,\qquad
   \lambda_s=L_{E,s}/J_s.

For the undensitized GRACE variables this contributes

.. math::

   S_{E,s}=\alpha W[Q_s-\lambda_s(E_s-v^iF_{i,s}-J_s)],\quad
   S_{F_i,s}=\alpha[Wv_iQ_s-\lambda_sH_{i,s}],\quad
   S_{N,s}=\alpha r_s.

Ordinary absorption/emission/scattering sources are added inside the same
twenty-variable solve:

.. math::

   U^{n+1}-U^*=h[S_{\rm ordinary}(U^{n+1})+
                  S_{\rm pairs}(U^{n+1})+S_{\rm decay}(U^{n+1})].

The existing scaled damped Newton routine is reused with N=20 and a numerical
Jacobian. If necessary, continuation solves the same equation at increasing
h, always against the ORIGINAL U*: this is not time subcycling. Positivity,
causality and FD reconstructibility are checked. Failure aborts with an
explicit error; the solver does not silently substitute LTE partners.

Small Newton/roundoff number defects are projected onto the reaction
stoichiometry AFTER removing the ordinary source contribution. Define

.. math::

   q_s=(1+h\alpha\kappa_{N,s}/\Gamma_s)N_s^{n+1}
       -(N_s^*+h\alpha\eta_{N,s}).

In order (nue, antinue, numu, antinumu), pure decay requires q0=q3 and
q1=q2. With muonic thermal pairs included, q0-q1+q2-q3=0. The projection
uses these linear constraints, restores N and checks the FULL residual
again. It is not a physical rate correction. No extra independent electron
source update runs after this connected solve. The tau aggregate retains
its existing symmetric thermal-pair update and multiplicity two.

Transport opacity is distinct from collision EAS
------------------------------------------------

Leaving evolved-mode EAS ordinary-only must NOT remove pair/decay damping
from the optically thick flux correction. ``prepare_pair_transport`` therefore
reconstructs current-state spectra during EAS evaluation and stores separate
``pair_transport_kappa1`` through ``pair_transport_kappa5`` fields. With code
unit conversion understood, the opacity used by transport is

.. math::

   \kappa_{{\rm tr},s}=\kappa_{{\rm a,ordinary},s}
      +\kappa_{{\rm s,ordinary},s}+\sum_{r}{L_{E,s}^{(r)}\over J_s}.

This uses the SAME R0 damping as the collision residual, evaluated at the
current rather than Newton-trial moments. At J=0 its contribution is zero
(the radiation flux is then zero). Aggregate moments are divided by their
multiplicity before reconstruction; opacities are not multiplied by it.
The existing face correction becomes

.. math::

   A_{\rm face}=\min\left(1,
        {1\over\Delta x\sqrt{\kappa_{{\rm tr},L}\kappa_{{\rm tr},R}}}\right).

The same summed-opacity accessor serves the lagged local optical-depth
estimate and the existing electron-flavour eikonal update. The latter keeps
its pre-existing one-EAS-evaluation lag. Density-fit/spherical optical-depth
prescriptions remain approximate. In equilibrium and standard modes, the extra
fields are zero because the corresponding damping is already in EAS.
Ordinary collision reads NEVER use the transport-only additions.

This corrects the pre-existing evolved thermal-pair opacity omission as well
as connecting the new decay channel. EAS kernel evaluations use the small-tile
launch in BOTH new partner modes. Transport preparation evaluates the kernels
once per EAS call, in addition to the once-per-local-source-stage kernel
construction; matrices are not retained in a large per-grid-cell allocation.

Matter backreaction and acceptance
---------------------------------

For decay alone, in the fluid frame,

.. math::

   n_b\dot Y_e=r_- - r_+,\qquad
   n_b\dot Y_\mu=-r_- + r_+,\qquad \dot{(Y_e+Y_\mu)}=0.

The implementation takes matter exchange from the actual old-minus-new
densitized radiation fields, including ordinary reactions:

.. math::

   \delta\tau=\sum_s(E_s^*-E_s^{n+1}),\quad
   \delta S_i=\sum_s(F_{i,s}^*-F_{i,s}^{n+1}),

.. math::

   \delta(DY_e)=\delta N_{\nu_e}-\delta N_{\bar\nu_e},\qquad
   \delta(DY_\mu)=\delta N_{\nu_\mu}-\delta N_{\bar\nu_\mu}.

Do NOT add a separate (m_mu-m_e)*r heating term: the full leptonic EOS
already carries the charged-lepton rest energy, and the neutrino energy
exchange above includes the energy lost by matter.

For a decay-enabled cell, one convex acceptance fraction theta in [0,1]
limits Ye, Ymu, Ye+Ymu and the existing positive-tau guard:

.. math::

   U_{\rm rad,accepted}=U^*+\theta(U_{\rm rad,trial}-U^*),\quad
   \delta U_{\rm matter,accepted}=\theta\delta U_{\rm matter,trial}.

For any constrained scalar y with increment dy, the admissible bound is
(ymax-y)/dy for a positive increment, or (ymin-y)/dy for a negative one.
Take the minimum over constraints and one, with the existing small safety
margin at crossed bounds. This replaces independent flavour acceptance ONLY
in decay-enabled cells; it also includes tau species in the shared energy
acceptance. ``muon_partial_at_bound`` does not override this connected
limiter. ``m1_br_reject`` bit 32 marks connected limiting when diagnostics
are compiled. At theta<1 the accepted result is conservative but no longer
the full unmodified implicit source solution; timestep convergence and
monitoring bound contact remain necessary. The tau guard is not a full EOS
primitive-recovery admissibility proof.

Workflow
--------

Common preparation:

1. Validate species, EOS, table provenance and reaction/mode options.
2. Read physical matter state and unsuppressed chemical potentials from EOS.
3. Evaluate ordinary analytic/WeakHub EAS without the thermal-pair additions.
   Form ordinary emissivities using its existing Kirchhoff prescription;
   ordinary temperature corrections stay in this step. With decay enabled,
   the historical muonic low-density/low-temperature rate gate is disabled
   (its multiplier becomes one). Optical-depth
   estimates may still be diagnostic inputs, but do not suppress the four
   physical equilibrium chemical potentials.
4. Prepare physical matter auxiliaries and the decay activation/order.

Equilibrium partners:

5. Build thermal and decay R0 kernels; integrate with LTE partner spectra.
6. Add their eta_E, eta_N, kappa_E and kappa_N to ordinary EAS. Do not apply
   another fugacity, abundance, optical-depth or temperature multiplier to
   these new kernel contributions.
7. Use the existing independent E,F,N implicit species updates.
8. Apply the connected conservative backreaction limiter. This cannot repair
   the equilibrium-partner approximation's intrinsic event-count defect.

Evolved partners:

5. Leave stored EAS coefficients ordinary-only; store the thermal-channel bits
   and current-state pair/decay damping in the separate transport-opacity fields.
6. In the local source update, build fixed-matter kernels and solve the four
   lepton-carrying species together, reconstructing all four trial spectra.
7. Update tau pairs separately. Skip the ordinary electron/muon updates that
   would otherwise double-apply their sources.
8. Apply connected conservative backreaction; subsequent primitive recovery
   and EAS evaluation update T, Ye, Ymu and the charged-lepton bath.

Transport flux formulas and geometric M1 terms are unchanged, but the opacity
supplied to their thick-limit correction now includes evolved pair/decay
damping. Evolved-variable counts and the EOS table format are unchanged. The
auxiliary layout gains mu_mu, activation, inner order, charge-defect and
transport-opacity fields; old checkpoints should be treated with
the normal variable-layout compatibility checks, not assumed interchangeable.

Code map
--------

Experimental dilute-target and lagged-partner options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``m1.eas.dilute_muon_cc_target: true`` regularizes the WeakHub muon CC
emissivity target toward the electron chemical potential on the demuonizing
branch, below Ymu=6e-4 with transition width 5e-5. It requires leptonic EOS,
noPP WeakHub rates, and ``betaeq_policy: off``. The raw EOS chemical potential
is still available for thermodynamic diagnostics. When spectral decay is
active, its reaction potential and muon-neutrino target use this same
regularized value: otherwise CC and decay impose incompatible affinities.
This is a phenomenological dilute-floor regularization, not a new physical
EOS. Existing grey absorption averages are retained; exact equivalence to
reintegrating a spectral WeakHub table is not claimed.
The new CC target uses the unclamped effective chemical potential. Therefore
standard-mode cells previously on the legacy +/-5 degeneracy rail can also
change outside the dilute transition; this option is not merely a change to
the floor value.

``m1.eas.fermi_fofc: true`` checks the flux cone and necessary continuum Fermi
bounds in both lab and fluid frames, with per-helicity species multiplicity.
It limits reconstructed face states toward donor states and triggers existing
shared-face donor flux correction for inadmissible predicted transport states.
It requires pair kernels and FOFC. It does not prove full discrete-quadrature
realizability or positivity of arbitrary IMEX stage right-hand sides. In
particular, reducing the CFL can remain necessary. Radiation moments are not
independently clipped to manufacture a reconstructible spectrum.

``m1.eas.muon_decay_treatment: lagged`` requires evolved partners and the
same EOS setup as ``spectral``. Frozen partner spectra yield four independent
five-moment solves. Outer iterations update their coefficients, and the
coupled residual and stoichiometry remain the acceptance criterion. Failure,
including failed acceptance of a stationary lagged iterate, retries the
reference coupled solver. This conservative prototype differs from a single
unconditionally accepted lagged Boltzmann update. It needs no new tables,
but retains the existing finite-mass kernel quadrature; it does not promise
a speedup in stiff matter. All three new options are opt-in.

Single-pass decay experiment and profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``muon_decay_treatment: singlepass`` freezes the transported partner spectra
once per implicit stage. Grey opacity weighting also freezes the target
spectral shape, an extra approximation required by grey M1. Production and
stimulated absorption are integrated independently from the existing decay
kernel, without a prescribed equilibrium target. Four-moment energy/flux
solves and an analytic number update replace the four-species decay solve;
the existing ten-moment conservative thermal-pair solve remains for muonic
thermal pairs. Tau transport and thermal-pair physics are unchanged.

There is no outer decay iteration, four-species coupled acceptance, coupled
fallback, or decay stoichiometric projection. Failure of positivity or FD
reconstruction stops the run. This is the single-pass lagging idea of Ng
et al. Sec. III C adapted to the decay channel in grey M1, not a copy of
their energy-group/angle transport or their treatment of every reaction.
Both the old ``lagged`` mode and ``spectral`` remain available as controls.

The two decay daughters need not acquire equal number increments in this
approximation. Matter backreaction still balances radiation energy and each
flavour lepton number, but the mismatch can spuriously change Ye+Ymu through
decay. It is therefore necessary to measure both conservation of exchange
and reaction stoichiometry; the former alone is insufficient.

For CPU diagnostics, compile with ``-DGRACE_LEPTON_PROFILE``. RAII timers
record inclusive and exclusive per-thread elapsed work, with call counts and
lagged fallback/iteration counts. Only exclusive categories may be added;
their sum covers the instrumented microphysics, not total wall time. Run an
otherwise identical build without this definition for runtime comparisons.
The profiler additionally writes ``source_budget_rank<N>.dat`` at each
implicit stage, over interior cells only. After t and diagonal-stage h,
columns b0--b9 contain baryon integral, L1 energy-exchange defect, L1 electron
and muon lepton-exchange defects, signed and L1 decay charge defect after
ordinary-rate removal and shared backreaction acceptance, absolute matter
energy/electron/muon exchanges, and signed matter charge exchange. Repeated
IMEX-stage sums are diagnostics, not a time-integrated final conservation
budget. No atmosphere, boundary-flux or gravitational energy accounting is
implied. The auxiliary field ``decay_number_increment_defect`` is registered
in all builds but populated only in diagnostic builds;
it is reset by rate preparation, so final plane values need not retain the
source-stage defect. Use the rank-local files for the audit.

Exact CPU decay-kernel reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``m1.eas.muon_decay_kernel_cache: true`` enables bounded, exact memoization
of the finite-mass decay kernel. The default is false. Source, transport
and equilibrium-rate preparation use the same process-local cache. It
does not cache opacities or source solutions: current neutrino spectra,
blocking, closure and the conservative connected solve are still evaluated.

The key includes bit representations of T, mu_e, the effective reaction
mu_mu, inner quadrature order, active grid order, mapping scale, energies
and weights. Every input must match exactly, without rounding, tolerance,
interpolation or freezing matter. Consequently a changed state misses the
cache; no cell indices, AMR topology or checkpoint state are involved.
The cache starts empty after restart. Bounded eviction changes only runtime.

Each MPI rank lazily allocates approximately 70 MiB for at most 4096 entries
in four-way buckets. Bucket mutexes protect lookup/copy/insertion across
OpenMP threads. Integration on a miss occurs outside the lock; concurrent
identical misses may duplicate work but cannot return a partially written
kernel. Only successfully integrated kernels are stored. GPU builds reject
this option; the uncached device implementation is unchanged.

``SOURCE_CACHE`` profile records give per-rank hits and misses. The existing
``decay_kernel`` timer now counts actual integrations, including duplicate
concurrent misses. Lookup/copy overhead remains in measured timestep runtime.
The achievable gain depends on exact state repetition and working-set size;
a symmetric hot-TOV benchmark cannot predict BNS cache hit rates.

Code locations
~~~~~~~~~~~~~~

* ``neutrino_muon_decay.hh``: inner quadrature, finite-mass kernels, stable
  detailed balance, common-event moments and LTE EAS coefficients.
* ``neutrino_leptonic_update.hh``: four-species implicit residual, FD spectral
  reconstruction, continuation and stoichiometric residual check.
* ``eas_policies.hh``: material preparation, physical chemical potentials,
  equilibrium coefficients/diagnostic, evolved ordinary-only EAS and separate
  current-state transport damping.
* ``eas_neutrino_rates_analytic.hh``: explicit optional charged-muon gate;
  both ordinary analytic and WeakHub paths preserve the old default.
* ``m1.hh``: density/metric conversions, connected solve wrapper and shared
  matter/radiation acceptance; ``pair_residual`` retains the maximum across
  connected and tau source solves rather than being overwritten by the latter.
* ``evolve.cpp``: dispatch connected update and avoid duplicate source steps.
* ``m1_helpers.hh``, ``eas_optical_depth.hh`` and ``m1.cpp``: shared transport
  opacity, optical-depth consumers and low-scratch-pressure EAS launch choice.
* ``eas_kinds.hh`` and parameter schemas: selection, compatibility checks and
  equilibrium-mode warning; variable-index files register auxiliary fields.
* ``leptonic_eos_4d.hh`` and its reader: optional unsuppressed thermodynamics
  and chemical potentials, applied before cold-slice generation.

Validation and remaining limitations
------------------------------------

The tests cover the finite-mass vacuum width

.. math::

   \Gamma_0={G_F^2M^5\over192\pi^3\hbar}
   [1-8r+8r^3-r^4-12r^2\ln r],\qquad r=m^2/M^2,

including the thermal muon time-dilation average. The daughter energy checks
use the massless-electron Michel means 0.35 M (muon flavour) and 0.30 M
(electron flavour), with a stated quadrature/finite-mass tolerance. These are
tree-level checks, not a claim of reproducing the radiatively corrected
experimental lifetime exactly.

Additional tests check microscopic balance, charge conjugation, inverse decay
without an abundance gate, quadrature refinement, LTE Kirchhoff coefficients,
the equilibrium closure's off-LTE defect, moving-fluid LTE, stiff connected
updates with ordinary rates, emission from vacuum, actual GRACE species and
densitization wiring, transport-opacity addition without source duplication,
composition/energy limiter conservation, EOS compatibility,
and rejected incompatible configurations. Existing thermal-pair, ordinary-rate
and backreaction tests are rerun as disabled-mode regressions.

Scientific boundaries remain:

* No grey method can recover an arbitrary spectrum from just J and n. The FD
  reconstruction and R0/grey flux source are explicit closure assumptions.
* This follows GRACE's FIXED-MATTER diagonal stage. It is not a monolithic
  matter-temperature/composition/radiation nonlinear solve and is not claimed
  asymptotic preserving at arbitrarily stiff matter-coupling timesteps.
* Finite energy and inner quadrature need convergence checks in representative
  hot, cold and degenerate merger cells. Underflow/unreconstructible states
  can abort; not every EOS table state is numerically resolved by order 16/24.
* EOS floors, atmosphere source cutoffs, disabled backreaction and frozen
  hydro retain their usual meanings. Conservation statements refer to the
  coupled source step with backreaction active, before unrelated atmosphere
  resets or other projections.
* Number conversion retains GRACE's existing fixed-nucleon-mass convention;
  this change does not redesign the legacy baryon-density/mass normalization
  shared by the EOS and ordinary-rate interfaces.
* Kokkos device functions, fixed-size scratch arrays and cached matter kernels
  avoid host transfers, device allocation and vendor-specific code. They do
  not constitute GPU performance validation: a 20-variable numerical Jacobian
  and per-cell kernel integration are substantial work. CUDA/HIP/SYCL builds,
  register/local-memory profiling and merger-scale benchmarks are still
  required before claiming efficient production operation on those devices.

See ``muon_decay_plan.rst`` for the reviewed implementation sequence and the
record of actual validation runs.

Reversible lifetime-based grey decay
====================================

``muon_decay_treatment: grey_decay`` now selects a representative-energy
reaction, not the 7 November spectral interpolation prototype. No decay
collision matrix is integrated, cached or refined. ``grey_decay_kernel_nodes``
is ignored. The full spectral and lagged prescriptions are unchanged.

For each muon charge, a fixed 16-point one-dimensional Fermi-gas moment
integral supplies density n and mean total energy U. The rest lifetime is
2.197 microseconds; the lab-frame grey lifetime uses the factor M/U.
Daughter energies are 0.35 U (same-flavour neutrino), 0.30 U (other-flavour
antineutrino), and 0.35 U (charged electron). These recover the massless-
electron Michel means for a nonrelativistic, dilute parent. This is a
representative-energy and time-dilation approximation, not a finite-mass
microscopic decay calculation.

With f_mu evaluated at U, f_e at 0.35 U, and holes h=1-f evaluated stably,
the common amplitude is C=(n/f_mu) M/(U tau). Define P=C f_mu h_e and
A=C h_mu f_e. All coefficients are evaluated in logarithmic form, so an
underflowing forward population does not erase a finite inverse rate.
Their ratio is exp[-(epsilon_0+epsilon_1 - sign*(mu_mu-mu_e))/T].
The event rate is R=P h_0 h_1 - A f_0 f_1, using the actual M1 populations'
moment-matched FD fits at the two representative energies. Both daughter
number sources equal R and their energy sources equal epsilon_s R.
Thus inverse decay requires both incoming partners, and chemical LTE is
a fixed point. Charge conjugation uses the opposite matter potentials.

The conservative four-species source solver, radiation-to-matter exchange,
realizability checks, and source acceptance remain active. Numerical solver
fallbacks evaluate this SAME cheap reaction; they cannot switch to spectral
physics. The analytic/block optimization still applies to thermal pairs;
the cheap grey reaction is evaluated directly during Jacobian probes.
Initial-data equilibrium rates and transport damping use the same model.
Profile category ``grey_rates`` measures coefficient preparation; grey
runs must have zero ``decay_kernel`` calls and zero decay-cache accesses.

This model requires evolved partners and dilute_muon_suppression=false.
It does not invert the existing vacuum prescription's density floor, gate,
or finite-timestep saturation. Consequently it agrees with vacuum decay
only in its dilute, unblocked, nonrelativistic limit, not bitwise in dense
matter. The existing vacuum option also requires a different EOS suppression
policy; composition differences in that comparison include this distinction.
No new external tables are needed. Detailed balance and conservation alone
do not establish dense-matter accuracy; compare against spectral before use.

The wider-domain Calea validation archived in
``GRACE_runs/HOTTOV_5spec/grey_domain_20261108/final_sweep`` found this
prescription unsuitable as a general BNS replacement for spectral decay.
Among 4831 active non-LTE reaction samples passing a 5-percent spectral
quadrature convergence screen, 52 percent had net-rate errors exceeding
10 percent of the reference emission-plus-loss scale; 65 had the wrong
net reaction sign. These are unweighted test-grid statistics, not BNS
occurrence frequencies. Matched closed-box evolutions to 10 ms differed
by up to 0.00910 in Ye/Ymu and 13.69 MeV in temperature, despite relative
energy conservation better than 5.1e-13 at completed checkpoints.
Representative energies miss spectral tails and energy-dependent blocking;
chemical detailed balance does not determine the correct relaxation rate.
Keep this option experimental. Smaller timesteps do not remove its model
error. See the archive's ``validation_report.rst`` for coverage, numerical
convergence checks, incomplete trajectories and coupled hot-TOV results.

The discarded interpolation prototype remains archived under
``GRACE_runs/HOTTOV_5spec/grey_decay_20261107``. Every grey8 build refined
there, explaining its 3.015 s step median versus spectral 3.017 s and vacuum
1.008 s. Those measurements do NOT describe the new lifetime model.
