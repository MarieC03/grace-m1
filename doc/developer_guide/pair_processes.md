# Grey heavy-flavour pair processes

## Scope and status

This implementation adds selectable `equilibrium` and `evolved` treatments;
`legacy` remains the default. It treats **neutrino pairs**, not creation of
charged muons:

* electron-positron annihilation, `e- + e+ <-> nu + antinu`;
* nucleon bremsstrahlung, `N + N <-> N + N + nu + antinu`;
* an explicitly optional transverse-plasmon model, `gamma* <-> nu + antinu`.

These thermal channels do not require charged muons to be present. They must
not be multiplied by the charged-muon abundance/density/temperature gate.
Muonic nucleon charged-current reactions remain the responsibility of the
ordinary EAS provider. Muon decay/inverse decay is a separate, optional
extension documented in [muon_decay.rst](muon_decay.rst); it connects four
species and changes their source/backreaction workflow. The thermal-pair
module described here does not add muon-antimuon annihilation, flavour
conversion, or missing electron-flavour pair channels in the baseline.

This is an **isotropic-kernel grey closure**, not an exact Boltzmann solution.
Conservation and detailed balance can be checked independently of the accuracy
of its spectral/angular assumptions. In particular, it is not a validated
polar-funnel annihilation/heating model and is not claimed to outperform a
spectral Monte Carlo calculation. Production use requires quadrature,
resolution, timestep and target-GPU validation.

## Configuration

Build with 3 or 5 neutrino species and initialize the repository's existing
`extern/bns_nurates` submodule. Its pinned revision is
`efbabe0d4dfa6935c9ecbbd59a30c69b7cd1b91e`; no new revision is selected here.

```yaml
m1:
  eas:
    kind: neutrino_weakhub        # alternatively neutrino_analytic
    pair_treatment: evolved      # legacy | equilibrium | evolved
    pair_quadrature_order: 16     # 8..32; compare 16, 24, 32
    weakhub_pair_content: none    # ONLY after checking table provenance
    betaeq_policy: off
    pair_annihilation: true
    bremsstrahlung: true
    plasmon_decay: true
    pair_plasmon_model: transverse_mass
```

No spectral WeakHub tables are needed: the new pair kernels are evaluated
locally. The ordinary grey WeakHub table **must exclude these heavy-flavour
thermal pair channels**. `weakhub_pair_content: none` is a declaration, not a
filter: the code cannot subtract unknown pair contributions from an inclusive
grey table. `unknown` and `included` are rejected in the new modes. The old
standalone `bns_nurates` EAS adapter is not used by these modes.

The new modes reject algebraic `betaeq_policy` replacements of the actual
matter state. Ordinary temperature corrections remain confined to ordinary
rates. Pair rates receive neither an extra temperature-square multiplier nor
an optical-depth fugacity suppression. The historical muonic `eta=+/-5` clamp
is retained in legacy mode only: using it in ordinary charged-current
Kirchhoff factors but not in the new pair calculation would introduce
incompatible equilibrium targets. GRACE's existing EOS chemical-potential
convention is retained:

\[
 \mu_{\nu_\mu}=\mu_\mu+\mu_p-\mu_n-Q_{np},\qquad
 \mu_{\bar\nu_\mu}=-\mu_{\nu_\mu}.
\]

## Collision operator and grey equations

In the matter frame, per physical helicity species, define

\[
 d\Pi=D(\epsilon)d\epsilon,
 \quad D(\epsilon)=\frac{4\pi\epsilon^2}{(hc)^3},
 \quad n=\int f\,d\Pi,\quad J=\int\epsilon f\,d\Pi.
\]

`R_a(e,ep)` is the angle-independent annihilation kernel with the first
argument belonging to the neutrino, the second to the antineutrino. Kernel
units are cm^3/s. For these thermal channels the matter bath has zero net
chemical potential for the event, so

\[
 R_p(\epsilon,\epsilon')
 =e^{-(\epsilon+\epsilon')/T}R_a(\epsilon,\epsilon'). \tag{1}
\]

For a specified partner distribution,

\[
 j_\nu(\epsilon)=\int R_p(1-\bar f')\,d\Pi',\qquad
 a_\nu(\epsilon)=\int R_a\bar f'\,d\Pi', \tag{2}
\]
\[
 C_\nu=(1-f)j_\nu-fa_\nu
       =j_\nu-\lambda_\nu f,\qquad \lambda_\nu=j_\nu+a_\nu. \tag{3}
\]

The effective opacity is `lambda/c`, **not** `a/c`. The antineutrino equation
uses the same event kernel with the roles of the arguments transposed.
Electron-pair kernels are not generally symmetric under energy exchange.

For weights `w=1` (number) or `w=epsilon` (energy), coefficients for a specified
spectral shape can be written

\[
 Q_w=\int w j\,d\Pi,\qquad
 c\kappa_w=\frac{\int w\lambda f\,d\Pi}{\int w f\,d\Pi}.
 \tag{4}
\]

Here `Q` is spontaneous emission *with partner blocking*, and own-species
blocking is contained in `lambda`. Thus both emissivity and opacity are
computed from the kernel. Calling both of them independent physical inputs
would also be misleading: they obey microscopic detailed balance (1).

### Equilibrium partner

Use `f_eq=[exp((epsilon-mu)/T)+1]^-1` and the partner with `-mu`. Then

\[
 j(\epsilon)=\lambda(\epsilon)f^{eq}(\epsilon),\quad
 B_w=\int w f^{eq}\,d\Pi,\quad
 Q_w=c\kappa_w B_w, \tag{5}
\]

where the grey opacity in (4) is weighted with `f_eq`. The code evaluates
both sides from the same quadrature. Equation (5) is valid here because the
blocking and averaging conventions agree. It does not justify dividing an
arbitrary unblocked emissivity by a degenerate FD blackbody.

The independent grey sources are `S_N=Q_N-c*kappa_N*n` and
`S_E=Q_E-c*kappa_E*J`. They vanish at the prescribed LTE target but **need not
give equal neutrino and antineutrino number changes away from LTE**. This is
an intentional limitation of the equilibrium-partner comparison mode.

### Evolved partner

At each trial of the implicit solve, GRACE's M1 closure supplies `J`, `H^alpha`
and `Gamma`. The comoving number density is `n=N_lab/Gamma`, with
`Gamma=W*(E-v^i F_i)/J` in the existing grey number-current closure. Reconstruct:

\[
 f_i=[\exp(a+b\epsilon_i/\langle\epsilon\rangle)+1]^{-1},\quad b>0,
 \qquad
 \sum_i W_i f_i=n,\quad \sum_i W_i\epsilon_i f_i=J. \tag{6}
\]

Both moments are matched on the **same** positive quadrature that evaluates
the collisions, separately for each partner. `W_i` includes `D(epsilon_i)`.
This is a two-parameter FD/maximum-entropy spectral closure, not a claim that
the evolved spectrum is actually thermal. Invalid or unrepresentable moments
are rejected; the solver does not secretly replace the partner by matter LTE.

For every ordered pair of energy nodes, accumulate one event rate:

\[
 I_{ij}=W_iW_j\{R^p_{ij}(1-f_i)(1-\bar f_j)
                         -R^a_{ij}f_i\bar f_j\}, \tag{7}
\]
\[
 S_N=\bar S_N=\sum_{ij}I_{ij},\qquad
 S_E=\sum_{ij}\epsilon_i I_{ij},\qquad
 \bar S_E=\sum_{ij}\epsilon_j I_{ij}. \tag{8}
\]

Consequently `d(nu-number minus antinu-number)/dt=0` for pair-only sources.
The two energies need not be equal. The whole FD family with common matter
temperature and opposite chemical potentials is a pair-equilibrium family;
pair interactions alone do not select beta equilibrium or erase a preexisting
lepton-number difference. There is no matter-Kirchhoff reset of (7)-(8).

The grey flux spectrum is assumed proportional to the energy spectrum. With
`lambda_E=integral(epsilon*lambda*f*dPi)/J`, the covariant pair source is

\[
 G^\alpha=S_E u^\alpha-\lambda_E H^\alpha. \tag{9}
\]

In GRACE's undensitized Eulerian variables this contributes

\[
 \dot E=\alpha W\{S_E-\lambda_E(E-v^iF_i-J)\},\quad
 \dot F_i=\alpha(Wv_iS_E-\lambda_E H_i),\quad
 \dot N=\alpha S_N. \tag{10}
\]

Ordinary charged-current/emission/scattering sources are added to (10) inside
one coupled ten-variable diagonal IMEX solve:

\[
 U-U_* - h\,[S_{ordinary}(U)+S_{pair}(U)]=0,
 \quad U=(E,F_x,F_y,F_z,N,\bar E,\bar F_x,\bar F_y,\bar F_z,\bar N),
 \quad h=dt\,dtfact. \tag{11}
\]

Matter and ordinary coefficients are held fixed over this local stage, as in
the existing GRACE source treatment. This is not a simultaneous EOS+matter
Newton solve. The existing conservative backreaction deposits the accepted
radiation energy/momentum and lepton-number changes into the fluid. Existing
backreaction limiters may throttle/reject those changes.

The solver uses scaled finite-difference Newton steps, positivity/cone checks
and line search. A continuation retry solves smaller coefficients of the same
original residual and ends at the full `h`; it is not a sequence of hidden
time substeps. Failure aborts explicitly, instead of accepting an unrelated
linear fallback. A final projection makes the two solved number equations
share a common pair increment; ordinary number absorption remains implicit.
The full nonlinear residual is checked again after this projection.
`pair_residual` records the normalized residual of the accepted state.

### Aggregate species and units

Five species: indices 2/3 are separate `nu_mu/antinu_mu`; index 4 represents
`nu_tau+antinu_tau` with multiplicity 2. Three species: index 2 has multiplicity
4. Divide aggregate `E,F,N` and ordinary emissivities by that multiplicity
before reconstructing a one-helicity spectrum. For a charge-symmetric
aggregate, use `(R(e,ep)+R(ep,e))/2`, evolve one representative spectrum and
multiply the output by the multiplicity. Do **not** multiply its opacity by
the multiplicity.

Physical sources are converted using the same constants as existing EAS:

\[
 J_{code}=C_E J,\quad n_{code}=C_N n,\quad
 Q_{E,code}=C_E Q_E/TIMEGF,\quad
 Q_{N,code}=C_N Q_N/TIMEGF,\quad
 \lambda_{code}=\lambda/TIMEGF,
\]

`C_E=mev_to_erg*RHOGF*EPSGF`, `C_N=mnuc_cgs*RHOGF`. Conserved variables
are divided by `sqrt(gamma)` before solving, multiplied back afterwards.

## Kernel implementation and approximations

Electron pairs use the pinned BNS_NURATES Pons/Bruenn `PairPsi` kernel,
including its relativistic/massless-electron approximation (not an exact
finite-electron-mass kernel at arbitrarily low temperature),
algebraically evaluating its absorption coefficient without the unstable
product `exp((e+ep)/T)*R_p`. In the library's symbols the implemented expression is

\[
 R_a^{ee}=-\frac{C T^2}{2[1-e^{-(y+z)}]}
 [\alpha_1^2\Psi_0(y,z,\eta_e)+\alpha_2^2\Psi_0(z,y,\eta_e)],
 \quad y=e/T,\ z=e'/T,
\]

with `C=kBS_Pair_Phi` and the heavy-flavour weak couplings from the pinned
library; the result is converted from nm^3/s to cm^3/s.
NN pairs use its unchanged Hannestad-Raffelt 1998 kernel,
without the optional Fischer medium correction. BNS_NURATES uses nm units:
`nb_nm^-3=nb_cm^-3*1e-21`, `R_cm^3/s=R_nm^3/s*1e-21`.
The electron-pair expression retains the library's numerical nonnegative-rate
guard; this is not an independent revalidation of the library's fit over every
EOS state. Extremely degenerate states require explicit convergence checks.

The optional plasma approximation uses two transverse polarisations, residue
`Z=1`, `omega^2=k^2+m_gamma^2`,

\[
 m_\gamma^2=\frac{4\alpha_{EM}}{3\pi}
 (\mu_e^2+\pi^2 T^2/3),\quad
 \Gamma_T(\omega)=\frac{G_F^2 C_V^2 m_\gamma^6}
 {48\pi^2\alpha_{EM}\hbar\omega},\quad C_V=-\tfrac12+2\sin^2\theta_W.
 \tag{12}
\]

For `omega=e+ep`, `k^2=omega^2-m_gamma^2`, the allowed region is
`(e-ep)^2<=k^2`, with the transverse decay distribution
`dP/de=3[1+(e-ep)^2/k^2]/(4k)`. Transforming the thermal photon phase space
gives the joint event kernel

\[
 D(e)D(e')R_a=
 2\frac{4\pi}{(hc)^3}\,\omega\Gamma_T(\omega)
 [1+f_{BE}(\omega)]\,\frac34
 \left[1+\frac{(e-e')^2}{k^2}\right]. \tag{13}
\]

`R_p=exp(-omega/T)R_a` includes `f_BE`; the inverse rate includes
`1+f_BE`, not `1-f_BE`. This approximation neglects longitudinal/axial/mixed
modes, charged-muon polarisation, finite electron mass corrections and the
momentum-dependent transverse dispersion. It is not quantitatively reliable
for all cool/degenerate states; opt-in is mandatory. Plasma thresholds can
converge more slowly than smooth ee/NN kernels on this tensor quadrature.

## EAS and evolution workflows

These steps describe `muon_decay: false`; see the separate decay guide for
the connected four-species extension.

Equilibrium mode:

1. Read EOS and ordinary analytic/WeakHub rates; suppress old thermal extras.
2. Use the consistent muonic chemical potentials in ordinary Kirchhoff factors;
   apply the existing *ordinary* gate and temperature corrections.
3. Construct thermal pair kernels and opposite-chemical-potential LTE spectra.
4. Compute both grey `Q` and stimulated-absorption `kappa` from (2)-(5).
5. Add these pair terms after ordinary corrections; preserve scattering.
6. Use the existing per-species `E,F,N` implicit update and fluid backreaction.

Evolved mode:

1. Perform the same ordinary EAS calculation, excluding old thermal extras.
2. Store physical `T,mu_e,nb,Xn,Xp` and enabled channels in pair auxiliaries.
   Ordinary EAS slots deliberately contain **no pair terms**.
   Store current-state pair damping separately in `pair_transport_kappa*`;
   flux corrections and optical-depth estimates use ordinary plus pair damping,
   while the collision solve reads only ordinary EAS coefficients.
3. In each implicit stage construct/cache one kernel matrix per physical pair.
4. Reconstruct both spectra from each trial `J,N/Gamma`; compute (7)-(10).
5. Solve (11), including ordinary EAS, simultaneously for the two partners.
   Treat the charge-symmetric residual heavy aggregate with its multiplicity.
6. Run unchanged electron/photon source updates, then existing backreaction.

Equilibrium-blended initial data temporarily use equilibrium pair EAS even
when the simulation selects evolved partners. The usual final EAS pass then
restores ordinary-only slots and material snapshots for evolution. Atmosphere
cells disable pair channels and clear their material/residual auxiliaries.

## Code map, review and validation

* `neutrino_pair_collision.hh`: quadrature, kernels, FD reconstruction, common
  blocked collision integrals and units.
* `neutrino_pair_update.hh`: coupled M1 residual and local nonlinear solver.
* `eas_kinds.hh`, `parameters/m1.yaml`: setup selectors and incompatible-mode checks.
* `eas_policies.hh`: ordinary/pair separation, matching muonic LTE targets,
  equilibrium coefficients and evolved material snapshots.
* `variable_indices.hh/.cpp`: material/channel/residual auxiliary registration.
* `m1.hh`: one-cell species mapping, densitization and aggregate factors.
* `evolve.cpp`: pair-stage launch before ordinary sources/backreaction.
* `m1.cpp`: setup validation, small-tile EAS launch and initialization support.
* `CMakeLists.txt`, `test/CMakeLists.txt`, `test_m1_pairs.cpp`: headers and tests.
  The Kokkos test helper now propagates the bundled YAML/p4est/HDF5 dependencies;
  existing backreaction tests support both Cowling and Z4 metric layouts.
  Two legacy-rate assertions were corrected to test the already-existing smooth
  muon suppression factor, rather than assuming a hard cutoff. The legacy rate
  implementation itself was not changed for those assertions.

The hot path has fixed-size automatic storage and Kokkos device functions, no
host-only solver, dynamic allocation or CUDA/HIP-specific API. Kernel matrices
are computed once per stage, not once per Newton trial. Separate small tiles
avoid the ordinary kernel's GPU launch-bound assumptions. This is portable
source design, not a GPU performance measurement; register pressure, spills
and throughput still need measurement on each target backend.

Tests cover unit normalization, FD moments, kernel orientation, LTE balance,
blocking/vacuum emission, unequal populations, implicit stiffness, simultaneous
ordinary EAS, moving-fluid LTE and execution on the configured Kokkos backend.
Run `ctest --test-dir build-pairs-cpu -R m1_pair_test --output-on-failure`.
Do not equate unit-test success with a validated neutron-star evolution.

### Verified build (2026-09-29)

The complete `grace` executable built with GCC/WSL, Kokkos Serial, Debug,
3 spatial dimensions, 5 neutrino species, muons enabled and Cowling metric.
The following suites passed after the review/fix cycles:

| Suite | Cases | Assertions |
|---|---:|---:|
| `m1_pair_test` | 9 | 119 |
| `m1_analytic_rates_test` | 33 | 1655 |
| `m1_backreaction_test` | 44 | 643 |
| Total | 86 | 2417 |

```sh
cmake -S . -B build-pairs-cpu -DGRACE_USE_BUNDLED_DEPS=ON \
  -DGRACE_M1_NU_SPECIES=5 -DGRACE_3D=ON -DGRACE_METRIC_EVOL=COWLING \
  -DGRACE_PERF_TUNING=generic -DCMAKE_BUILD_TYPE=Debug
cmake --build build-pairs-cpu --target grace m1_pair_test \
  m1_analytic_rates_test m1_backreaction_test -j6
ctest --test-dir build-pairs-cpu \
  -R '^m1_(pair|analytic_rates|backreaction)_test$' --output-on-failure
```

Not verified here: GPU compilation/performance, a three-species executable,
full coupled hydrodynamic/merger evolutions, all extreme EOS states, or
quantitative accuracy against full angular/spectral transport. The standard
M1 flux cone is checked; this is not a full fermionic angular-realizability
closure. Even with matched chemical potentials, differences between finite
pair quadrature and the existing ordinary-EAS Fermi-integral fits leave a
quadrature-level LTE mismatch: check order convergence before production.
The existing generic implicit-solver counters describe the ordinary solver;
`pair_residual` is the new coupled-pair solver's diagnostic.

## Diagnosing a failed evolved-pair update

The evolved transport pass and two-species implicit pair update collect up to
four failure records per MPI rank. After the device launch completes, the host
prints `[PAIR_FAILURE]` records to stderr, flushes them, and aborts before using
the failed update. This avoids relying on device printf immediately before a
HIP device assertion. Capture stderr along with stdout in the batch-job log.
This diagnostic path adds a host synchronization to each evolved-pair launch.

Each report identifies the iteration, transport/implicit phase, rank, local
block `q`, cell indices (including ghosts), physical coordinates, species,
matter temperature/composition, kernel scale/order and enabled channels.
`active_bits` uses 1 for electron-positron annihilation, 2 for nucleon
bremsstrahlung and 4 for plasmon decay. Species indices are zero based: in a
five-species build 2/3 are numu/antinu_mu and 4 is the tau aggregate.
Radiation E, F_i and N are undensitized code values per physical species;
comoving n is in cm^-3 and J in MeV cm^-3. Aggregate moments are divided by
their recorded multiplicity.

Failure status meanings:

- `kernel-input`: invalid material, quadrature order or energy scale.
- `kernel-backend-unavailable`: requested a channel without BNS_NURATES.
- `kernel-element`: negative/nonfinite kernel value; the energy-node pair and
  value are recorded.
- `decay-kernel`: muon-decay kernel construction failed.
- `moment-input`: reconstruction received nonpositive, non-vacuum moments.
- `reconstruction-jacobian`, `reconstruction-line-search`, or
  `reconstruction-iterations`: the spectrum reconstruction failed. Its input
  n/J, iteration, residual and grid energy range are recorded. This alone does
  not establish that the continuum moments are physically impossible.
- `implicit-solve`: the coupled pair solve (including its continuation/final
  residual check) failed. The record contains its last reported residual and
  input state; it does not identify the rejected internal Newton trial.

Unused diagnostic fields have sentinel values (iteration/species/node -1).
The separate four-species muon-decay implicit solver retains its existing abort
path; the buffered implicit diagnostics cover the two-species thermal pairs.

The `zero` radiation initial data now uses E_floor(r) and N_floor(r), matching
the evolution/activation reset. Previously it used E_floor(r)/eps_fl for N,
which could impose a different spectrum before the first floor reset. This
consistency fix does not establish the cause of any particular solver failure.

Do not work around failures by skipping pairs for the first iterations or by
resetting the radiation to LTE. Vacuum emission is supported; the intended
initial-value problem should remain intact. A matter-density cut also needs
physical and convergence justification, because cold/dilute cells can contain
neutrinos transported from hotter regions. Diagnose the recorded state first:
fix invalid inputs, improve reconstruction for representable moments, or use
a conservative step-rejection/substepping strategy for a stiff source solve.

## Literature and what can actually be criticized

[Ng et al., WeakHub](https://arxiv.org/html/2309.03526v2) give the pair collision
operator and reaction kernels (Sec. 3.4). [Chiesa et al., BNS_NURATES](https://arxiv.org/html/2412.04570v2)
describe the portable kernel library and reconstruction from grey moments.
The new code reuses the kernels, not the library's complete grey closure.
[Braaten and Segel](https://arxiv.org/abs/hep-ph/9302213) give the more complete
plasma treatment; the simplified model (12)-(13) must not be confused with it.

[Gieg et al.](https://arxiv.org/html/2604.14225v2), Appendix A, drop neutrino
blocking in A4 before writing FD detailed balance in A5. That cannot be the
exact identity of the fully blocked collision operator; an additional closure
choice is involved. This implementation keeps both blocking factors instead.
[Foucart](https://arxiv.org/html/2606.27425v1), Sec. II.4, uses evolved spectral
estimates with an opaque-region equilibrium fallback and an angular correction;
it is not simply an equilibrium-partner treatment. A blanket claim that this
new grey R0 implementation is more accurate is unwarranted. No unpublished
criticism attributed to Harry is treated as evidence of a specific error.
