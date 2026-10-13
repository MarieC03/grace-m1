# Grey pair processes: implementation and verification plan

Scope: thermal electron-positron annihilation, nucleon bremsstrahlung and
plasmon decay producing muon neutrinos and antineutrinos, with the residual
tau aggregate treated consistently. Preserve the existing implementation as
`legacy`; add explicit `equilibrium` and `evolved` setup choices.

1. Audit the current, locally modified EAS, species mapping, unit conversions,
   source solver and backreaction. Preserve unrelated work.
2. Supply energy-dependent kernels on the GPU: reuse the pinned BNS_NURATES
   electron-pair and HR98 bremsstrahlung kernels. Implement and document a
   transverse-plasmon approximation; do not present it as the complete
   transverse/longitudinal/axial/mixed WeakHub plasma calculation.
3. Use one isotropic (R0) collision operator with both neutrino blocking
   factors. Reconstruct spectra from energy and number moments on a common
   quadrature. Production and annihilation obey microscopic detailed balance.
4. Equilibrium option: hold the partner at the matter FD distribution and
   calculate stimulated-absorption-corrected grey coefficients. State explicitly
   that independent LTE-partner relaxation does not conserve pair number away
   from equilibrium. Do not attribute this approximation to the evolved mode.
5. Evolved option: evaluate both partners at the implicit stage, integrate one
   common event rate for both number equations, and use separate energy weights.
   Couple ordinary EAS and pair contributions in the same local solve. Use the
   existing fluid backreaction on the resulting conservative radiation changes.
6. Keep new thermal rates outside charged-muon gates and generic temperature
   corrections. Require explicit confirmation in configuration that the ordinary
   WeakHub table excludes thermal pair channels. Reject incompatible providers.
7. Register material inputs and solver diagnostics, setup validation, new source
   calls, and tests. No device allocation, host-only numerics or vendor APIs in
   the collision path.
8. Cross-check kernel orientation, units, statistical weights, blocking,
   detailed balance, vacuum/absent-partner behavior, unequal populations,
   source stiffness, moving-fluid transformations and aggregate multiplicity.
   Build and run the CPU/Kokkos tests; report GPU validation separately.

The implemented angular approximation is R0. Flux damping assumes the same
spectral shape as the energy moment. This does not resolve annihilation geometry
in a polar funnel, spectral pinching, or crossing beams; those require an
additional spectral/angular closure and validation. Conservation and detailed
balance are properties to test, not substitutes for that validation.

## Completion and review record

Implemented both selectors, built-in kernels, material auxiliaries, equilibrium
EAS, coupled evolved-partner M1 stages, aggregate species and initialization
connections. Reviews corrected mismatched muonic equilibrium targets, the
initial-data rate path, stiff final-number projection, launch-bound assumptions,
and build/test dependencies. Legacy tests were updated for the existing smooth
muon gate and Cowling metric layout, without changing legacy production rates.

The full five-species CPU executable builds. Pair, analytic-rate and
backreaction suites pass (86 cases, 2417 assertions). The equations, code map,
workflow, reproducible commands and remaining validation limits are in
[pair_processes.md](pair_processes.md). GPU and production-physics validation
remain follow-up work, not completed checks.
