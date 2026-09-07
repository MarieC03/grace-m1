# LBM radiation: cluster-scale verification runs

Higher-resolution twins of the `test/configs/lbm_*.yaml` problems (which are sized for a laptop),
plus an M1 twin for side-by-side comparison.  Each parfile header states what it checks.
Memory: 302 populations x 3 registers plus the claim weights of the conservative remap = 134 MB per 16^3 quadrant (77 MB per 12^3 quadrant).

## Builds

Reuse the cluster configure line and change only these flags:

| build | flags |
|---|---|
| LBM, static metric | `-DGRACE_RADIATION_SCHEME=LBM -DGRACE_M1_NU_SPECIES=1 -DGRACE_METRIC_EVOL=COWLING` |
| LBM, Z4c metric    | `-DGRACE_RADIATION_SCHEME=LBM -DGRACE_M1_NU_SPECIES=1 -DGRACE_METRIC_EVOL=Z4` |
| M1 twin            | `-DGRACE_RADIATION_SCHEME=M1 -DGRACE_M1_NU_SPECIES=1 -DGRACE_METRIC_EVOL=COWLING` |

`GRACE_LBM_STENCIL=Lebedev29` is the default.  Test suites: `ctest -L fast` and `ctest -L lbm` in
the LBM builds (the `lbm` label holds the integration runs, ~1 min each on 4 cores; the run list is
split by metric mode in `test/CMakeLists.txt`); the M1 build keeps its usual suite.

## Runs

| parfile | build | quadrants | check (laptop result at the coarser twin) |
|---|---|---|---|
| `lbm_sphere_wave_ks_hr.yaml` | Cowling | 512 x 12^3, dx 0.1875, ~40 GB | `∫lbm_ekill1` (Killing energy) constant to round-off (1.6e-10 by t = 2 at dx 0.375 on 64 quadrants) until the shell reaches the excision or the boundary |
| `lbm_curved_beam_hr.yaml` | Cowling | 512 x 12^3, ~40 GB | `scripts/lbm_radial_profile.py FILE blob 7.7`: pulse centroid on the analytic null geodesic within 1 cell |
| `lbm_tov_cowling_hr.yaml`, `lbm_tov_z4_hr.yaml` | Cowling, Z4 | 64 x 16^3, dx 0.47, ~8.6 GB | `scripts/lbm_compare_scalars.py A/output_scalar B/output_scalar 1e-3`: the two agree to the Z4c truncation error (1.4e-4 at dx 0.94, 3e-5 at dx 0.625); `∫lbm_ekill1` constant to round-off (7e-13 at dx 0.94) in Cowling (conservative remap) |
| `m1_crossed_beams.yaml` | M1 | 8 x 16^3 | M1 twin of `test/configs/lbm_crossed_beams.yaml` (imex222 at cfl 0.25, same end time). Two orthogonal free-streaming pencils cross at the origin: the LBM lets them through (on-axis E decays smoothly, no feature at the crossing), M1 merges them (on-axis E collapses 100x within three cells of the crossing and the energy moves into the diagonal quadrant, where M1 carries 3x the LBM's). `scripts/lbm_radial_profile.py FILE crossed 0.125 0.0625` |
| `m1_tov_cowling.yaml` | M1 | 8 x 16^3 | M1 twin of the Cowling TOV flash (imex222, same grid and dt); M1 keeps `∫alpha Erad1 dV` (the Killing energy, beta = 0) constant to 2e-5. The LBM's default conservative remap (`lbm.curved_remap`) keeps `∫lbm_ekill1` to 7e-13 over 9 M; the pointwise sweep loses 12% at this resolution (first order in dx) |
| `lbm_puncture_z4_hr.yaml` | Z4 | 64 x 16^3, dx 0.1875, ~8.6 GB | `scripts/lbm_escape_fraction.py output_volume/volume_out_000000.h5 output_scalar/Lrad_nu1_R9.dat`: energy escaping through R9 vs the analytic Schwarzschild escape fraction; measured/predicted = 1.03 at dx 0.375 by t = 32 |
| `test/configs/lbm_shadow.yaml` | Cowling | cluster size | shadow contrast (`lbm_radial_profile.py FILE shadow 0.1 0.4`); the fixed stencil spreads the beam to ~10 deg |

The puncture and TOV Z4 runs report the geometry-pass cost at the end (`lbm.report_timings`): on
the laptop the per-step geodesic map (rays, fit and claim weights) costs about 2.5x the LBM sweep
under Z4, where it is rebuilt every step; under Cowling it is computed once.
