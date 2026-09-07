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

Submit them with `examples/lbm/submit_lbm_hunter.sh`, which runs `ctest -L lbm` as a gate and
then the parfiles below on one MI300A node (4 ranks, one per APU).  `qsub -v WHICH=z4,RUN_CTEST=0`
selects a subset.

All the high-resolution parfiles use **24^3 blocks**: with 4 ghost zones those are 32^3 padded,
367 MB per quadrant, against 90 MB for a 12^3 block that holds only a fifth of the useful cells.
Per interior cell that is 22 kB versus 52 kB.  512 quadrants is ~188 GB, comfortable on one node.

| parfile | build | grid | check (laptop result at coarser resolution) |
|---|---|---|---|
| `lbm_crossed_beams_hr.yaml` | Cowling | 192^3, dx 0.0104 | two orthogonal beams cross and pass through each other, 24 cells across each beam. `lbm_radial_profile.py FILE crossed 0.125 0.0104`. M1 merges them into one diagonal beam (`m1_crossed_beams.yaml`): on-axis E collapses 100x within three cells of the crossing |
| `lbm_shadow_hr.yaml` | Cowling | 192^3, dx 0.0052 | the geometric shadow behind an opaque sphere stays dark; core was 40-50x dimmer than the rim at dx 1/128 |
| `lbm_sphere_wave_ks_hr.yaml` | Cowling | 192^3, dx 0.09375 | `∫lbm_ekill1` constant to round-off (5.7e-12 over 11 steps measured on GPU) until the shell reaches the excision at t = 2.2, then a smooth absorption ramp |
| `lbm_curved_beam_hr.yaml` | Cowling | dx 0.09375 | `lbm_radial_profile.py FILE blob 7.7`: pulse centroid on the analytic null geodesic within a cell |
| `lbm_tov_cowling_hr.yaml`, `lbm_tov_z4_hr.yaml` | Cowling, Z4 | dx 0.15625 | `lbm_compare_scalars.py A B 1e-3`: the pair differs only by Z4c truncation error (8.6e-6 at dx 0.47). `∫lbm_ekill1` conserved to round-off in both (4.5e-13 on GPU) |
| `lbm_puncture_z4_hr.yaml` | Z4 | dx 0.125, to t = 32 | `lbm_escape_fraction.py output_volume/volume_out_000000.h5 output_scalar/Lrad_nu1_R9.dat`: energy through the r = 9 detector against the analytic Schwarzschild escape fraction, 1.04 at dx 0.375 |
| `m1_crossed_beams.yaml`, `m1_tov_cowling.yaml` | M1 | as their twins | M1 comparisons: the crossed beams merge, and M1 conserves the TOV Killing energy where the pointwise LBM sweep would have lost 12 % |

Each run writes the xy, xz and yz slices into `output_surface` at ~10 frames, a few MB per run,
which is what to download rather than the volume files.

The puncture and TOV Z4 runs report the geometry-pass cost at the end (`lbm.report_timings`): on
the laptop the per-step geodesic map (rays, fit and claim weights) costs about 2.5x the LBM sweep
under Z4, where it is rebuilt every step; under Cowling it is computed once.
