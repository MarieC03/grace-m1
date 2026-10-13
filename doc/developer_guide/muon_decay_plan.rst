Muon decay implementation plan
==============================

Scope
-----

Add mu- <-> e- + numu + antinue and its charge conjugate, selected by
``m1.eas.muon_decay`` (default false). Preserve the default legacy/no-decay path.
Use finite charged-lepton masses, both charged-lepton blocking factors,
tree-level weak matrix elements, and the isotropic R0 angular approximation.
Do not add lepton-flavour scattering, radiative corrections or oscillations.

Primary references
------------------

* Guo et al., https://arxiv.org/html/2006.12051, II.2 and Appendix A.
* Sugiura et al., https://doi.org/10.1093/ptep/ptac118, muon decay kernels.
* Gieg et al., https://arxiv.org/html/2604.14225v2, Table 1 and Appendix A.

Review gates before implementation
---------------------------------

1. Fix species orientation with the vacuum Michel spectra, not merely a
   symmetric total-rate test. Check spin factors against the vacuum width.
2. Use Rp/Ra = exp[-(e+ep-mu_mu+mu_e)/T], not the existing thermal-pair factor.
   Avoid overflowing an exponential times an underflowed rate.
3. Keep decay separate from same-flavour thermal pairs. The evolved network
   connects nue, antinue, numu and antinumu: solve all twenty radiation moments
   together and never run the separate electron update on them afterwards.
4. Preserve equal number increments per reaction and both flavour lepton
   numbers. Decay alone must leave Ye+Ymu unchanged, including at limiter bounds.
5. Literal equilibrium-partner EAS is a comparison approximation, not an
   event-conserving closure away from LTE. Expose its charge-source defect;
   do not claim that independent Kirchhoff relaxation is exact decay physics.
6. Require a thermodynamically consistent muonic EOS and declared decay-free
   external opacity tables. Do not quietly use an artificial muon chemical
   potential or switch off inverse decay just because few muons are present.

Implementation sequence
-----------------------

* Implement a finite-mass phase-space kernel and independent normalization,
  balance, CP and spectral-orientation tests first.
* Extend setup, matter auxiliaries, EAS separation and LTE coefficients.
* Extend the local coupled source solve and conservative acceptance of the
  connected four-species exchange; retain existing lagged-matter stage semantics.
* Check EOS dilute-muon compatibility explicitly and reject unsupported setups.
* Build and run kernel, source, backreaction and existing EAS tests. Repeat
  review after testing, checking disabled-mode compatibility and setup failures.
* Document equations, both workflows, changed routines and actual validation.

Numerical boundaries
--------------------

Grey FD reconstruction and R0 angular closure are approximations. A local
fixed-matter stage is not a simultaneous matter/EOS nonlinear solve. Production
use requires energy/inner-kernel quadrature and timestep convergence and GPU
compilation/performance measurements. Unit tests are not merger validation.

Review iterations completed
---------------------------

1. Kernel review: checked the finite-mass normalization against the vacuum
   width and the two DIFFERENT daughter Michel moments. Added microscopic
   detailed balance, charge-conjugation, inverse-decay and quadrature tests.
   Used direct occupation/hole factors and bounded exponentials to avoid
   cancellation or overflow in degenerate matter.
2. Coupling review: replaced independent four-species decay updates with a
   connected radiation solve in evolved mode. Added the ordinary sources to
   the same residual, removed duplicate update calls and checked actual
   metric/densitization/species wiring. Added a stoichiometric correction for
   converged numerical roundoff, followed by a full residual check.
3. Matter/EOS review: identified incompatible dilute-muon substitutions and
   added an explicitly selected unsuppressed EOS path. Replaced independent
   flavour acceptance by a common conservative limiter for decay-enabled
   cells. Tested composition bounds, energy exchange and flavour lepton
   conservation. Added setup rejection tests and made the equilibrium
   approximation's off-LTE event-count defect explicit rather than concealing it.
4. Transport review: found that ordinary-only evolved EAS omitted thermal-pair
   damping from the thick-limit flux correction and optical-depth consumers.
   Added separate current-state transport opacities, including decay, without
   adding them to the ordinary collision coefficients. Added regressions for
   this separation, including the existing no-decay evolved-pair path.

Validation record
-----------------

Validation uses local GCC/Kokkos Serial Debug builds. No CUDA, HIP or SYCL
compiler is available in this environment. No production EOS/WeakHub table
set or merger evolution has been validated by these tests.

On 30 September 2026:

* The complete five-species ``grace`` executable compiled and linked after
  the final transport-opacity changes, including all EOS/EAS instantiations
  in that configured build.
* Five species, muons ON, 3D Cowling: all six selected CTest entries passed
  (``muon_decay_test``, ``muon_decay_setup_rejection``, ``m1_pair_test``,
  ``m1_analytic_rates_test``, ``m1_backreaction_test``,
  ``leptonic_logpress_test``). The five Catch2 executables passed 5,081
  assertions in 105 cases; the setup harness additionally checked seven
  incompatible configurations and their expected error messages.
* Three species, muons OFF, optical depth ON, 3D Cowling: all four selected
  CTest entries passed (decay kernel/network, thermal pairs, backreaction,
  and optical-depth tests). This checks the disabled-mode build and the
  eikonal consumer of the separately stored transport opacity.
* ``git diff --check`` passed. These are targeted microphysics/source tests,
  not a claim that every test in GRACE's complete CTest suite was run.

Reproduction (from the repository root, for the configured build directories):

.. code-block:: sh

   cmake --build build-pairs-cpu --target muon_decay_test m1_pair_test \
     m1_analytic_rates_test m1_backreaction_test leptonic_logpress_test grace -j4
   ctest --test-dir build-pairs-cpu --output-on-failure \
     -R 'muon_decay|m1_pair_test|m1_analytic_rates_test|m1_backreaction_test|leptonic_logpress_test'
   cmake --build build-decay-3sp --target muon_decay_test m1_pair_test \
     m1_backreaction_test m1_optical_depth_test -j4
   ctest --test-dir build-decay-3sp --output-on-failure \
     -R 'muon_decay_test|m1_pair_test|m1_backreaction_test|m1_optical_depth_test'
