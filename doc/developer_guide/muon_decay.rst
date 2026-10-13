Muon decay and inverse decay in grey M1
=======================================

Scope and recommended mode
--------------------------

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
       weakhub_pair_content: none
       weakhub_muon_decay_content: none
       plasmon_decay: false     # or explicitly select transverse_mass model

The table-content declarations are assertions of provenance, NOT filters.
Only use ``none`` after verifying the supplied tables exclude those reactions.
Inclusive grey opacities cannot be decomposed by this interface. The analytic
ordinary-rate provider does not acquire muonic nucleon charged-current rates
from this change; those still require the appropriate WeakHub input.
The explicit thermal-pair framework remains heavy-flavour only: this change
does not add the electron-flavour thermal pairs missing from the analytic
baseline. Electron pair channels supplied inside an ordinary WeakHub table
retain that table's grey treatment. ``weakhub_pair_content`` concerns the
heavy-flavour channels added explicitly by the pair framework.

Required build: five neutrino species, ``GRACE_ENABLE_MUONS``, and the existing
``bns_nurates`` support used by the new pair framework. Other EAS providers,
legacy pair treatment and beta-equilibrium remapping policies are rejected.
``muon_decay`` defaults to false; historical EOS suppression defaults to true.
The default legacy/no-decay path retains its previous behaviour. Existing
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
prescriptions remain approximate. In equilibrium and legacy modes, the extra
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
limiter. ``m1_br_reject`` bit 16 marks connected limiting when diagnostics
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
