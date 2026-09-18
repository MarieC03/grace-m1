# Wrong-code report: hipcc (ROCm 7.0.2) on MI300A / gfx942 — lane-0 corruption in an inlined EOS kernel

**Reporter:** Keneth Miler (GRACE code, Goethe University Frankfurt), on HLRS Hunter
**Date:** 17 September 2026
**Severity:** silent wrong results in production kernels; run-to-run non-determinism of a
GRMHD simulation. No crash, no error code.

## Summary

A GPU kernel that evaluates a tabulated equation of state (a few 3D table interpolations,
`exp`/`log`, and a Brent root-find) returns a wrong value on **the first work-item of a
wavefront** for some launches: the returned specific energy is exactly `eps - 1.0`
while every other output of the same evaluation (pressure, sound speed, clamped inputs)
is correct. The result depends on what ran earlier in the process but is reproducible
for a fixed sequence. It is a property of the compiled code:

| same source function, same inputs, same node | result |
|---|---|
| compiled at `-O3` (production) | **wrong** |
| compiled at `-O2` | **wrong** |
| compiled at `-O1` | correct (3/3 runs) |
| `-O3` with an empty `asm volatile("" ::: "memory")` and a `volatile` copy of one double between two calls | correct |
| `-O3` under `#pragma clang optimize off` | correct |
| `-O3`, the same calls arranged differently in the kernel body | correct |
| `-O3`, identical function template instantiated for a simpler EOS class | correct |

Runtime, memory-system and hardware explanations were excluded (section 4).
At application level, two identical 8-rank GRMHD runs diverge from the first time step
when built at `-O3` and are bit-identical over 21 steps when built at `-O1` (section 5).

## 1. Environment

- HLRS Hunter, HPE Cray EX, AMD Instinct MI300A (gfx942, APU). Reproduced on nodes
  `x1001c1s4b1n0`, `x1000c3s0b0n0`, `x1001c0s3b0n0`.
- Modules `HLRS/APU/2026.1`, `rocm/7.0.2`.
  `hipcc --version`: HIP version 7.0.51831-7c9236b16; AMD clang version 20.0.0git
  (roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc), target x86_64-unknown-linux-gnu.
- Kokkos 5.0.2 built with `Kokkos_ARCH_AMD_GFX942_APU` (HIPSpace is host-accessible unified
  memory), device code compiled with `-fgpu-rdc`, `-O3 -DNDEBUG`, `HSA_XNACK=1`.
- Cray MPICH, `MPICH_GPU_SUPPORT_ENABLED=1` (the reproducer runs on one rank, one GPU).

## 2. The failing code

GRACE, branch `fix/leptonic-fil-parity`, file `test/test_c2p.cpp`. The kernel body is the
function `eval_eos_once<leptonic_eos_4d_t>`:

```cpp
// forward: press, eps, csnd2 from (rho, T, Ye, Ymu); three 3D table interpolations + exp
out[D_PRESS] = eos.press_eps_csnd2__temp_rho_ye_ymu(eps, csnd2, t, r, y, m, err) ;
out[D_EPS]   = eps ; /* ... other outputs ... */
// inverse: T from that eps by Brent on total_eps(T) - eps, then press, h, csnd2, entropy
double e2 = eps ;
out[D_KAS_PRESS] = eos.press_h_csnd2_temp_entropy__eps_rho_ye_ymu(hh, cs2, t2, s2, e2, r2, y2, m2, err2) ;
```

`total_eps` (header `include/grace/physics/eos/leptonic_eos_4d.hh`) is
`exp(baryon_table.interp(...)) - energy_shift + [muon terms if ymu > 6e-4] + [electron terms]`,
all `KOKKOS_INLINE_FUNCTION`/`always_inline`. Inside the inverse it is inlined again for two
bracket evaluations and once per Brent iteration. The wrong evaluations all have
`ymu = 5e-4` (muon branch not taken).

Launch: `Kokkos::parallel_for(RangePolicy<HIP>)`, 8192 or 13824 work-items, functor passed
through Kokkos' constant-memory launch path (`hip_parallel_launch_constant_memory`).

## 3. Observed behaviour

Test `leptonic EOS is a deterministic function of its inputs` (Catch2 tag `[determinism]`):

- **Across threads:** one input point evaluated by 512 work-items in one launch. Work-item 0
  of the point's block returns `eps = -0.99171548424738976`, work-items 1..511 return
  `+0.0082845157526101523` (difference `1.0` to 17 digits). Pressure and sound speed agree
  across all 512. The inverse's outputs of the bad work-item are consistent with having
  received `eps - 1`. Total: 4520 field mismatches per run.
- **Across launches:** 13824 points evaluated twice. 216 evaluations differ = exactly
  every flat index `≡ 0 (mod 64)`; at `eps ≈ 1.7e8` the difference is still exactly `1.0`
  (`167593368.65660062` vs `167593369.65660062`).
- The mismatch counts are identical for repeated runs with the same launch history and
  change when the number or shape of preceding launches in the process changes.
- The energy shift the device reads is correct in every evaluation; a term-by-term
  recomputation of `total_eps` in a *differently structured* kernel is correct in 8192/8192
  evaluations on 128 distinct (XCC, SE, CU, SIMD) units.
- The identical function instantiated for `tabulated_eos_t` (one interpolation per
  evaluation, no muon/electron branches) is correct.

## 4. Excluded

Each of the following leaves the failure unchanged (same counts, same values):
`AMD_SERIALIZE_KERNEL=3`, `AMD_SERIALIZE_COPY=3`, both together; `GPU_MAX_HW_QUEUES=1`
(for equal launch history); `HSA_XNACK=0`; `HSA_ENABLE_SDMA=0`; `Kokkos::fence()` or
`hipDeviceSynchronize()` immediately before the launch; allocating the output without the
zero-fill memset; a different lepton table (changes which launches are wrong, not whether).
Three different nodes fail identically. The host (OpenMP) build of the same source is
bit-identical everywhere.

ISA of the failing kernel (attached): 111 `v_writelane_b32`/145 `v_readlane_b32` SGPR spills
into `v126`/`v127` (never used as data), 234 per-lane `scratch_*` accesses with a
non-overlapping frame layout, no scalar stores, no DPP/permlane/readfirstlane. I did not
locate the faulty instruction by hand.

## 5. Application-level impact

GRACE (GRMHD + Z4c, Kokkos/p4est), two identical 8-rank runs, `GRACE_ENABLE_DETERMINISTIC_MPI=ON`,
cells differing bit-wise in a 2D output plane:

| case | build | it 1 | it 5 | it 10 | it 20 |
|---|---|---|---|---|---|
| hot TOV star, FOFC on | `-O3` | 37 | 1236 | 6227 | 27675 |
| hot TOV star, FOFC on | `-O1` | 0 | 0 | 0 | 0 |
| head-on NS binary (FUKA ID) | `-O3` | 4606 | 10375 | 13009 | 17948 |
| head-on NS binary (FUKA ID) | `-O1` | 0 | 0 | 0 | 0 |

At `-O1` the full 3D volume (8 273 920 cells) and all 164 scalar diagnostics are identical
between the two head-on runs. A five-species neutrino-transport build of the same case at
`-O1` (transport compiled in, idle by its activation trigger) is likewise bit-identical in
all 97 output fields and 228 scalars over 21 steps, except four values of the
momentum-constraint *diagnostic* (not an evolved field) at two outputs, differing by
24–896 ULP in far-atmosphere cells; the evolved state in those cells is identical and the
next output is identical again. Wall time of that build: 1.62 / 1.65 s per step at `-O1`
(no `-O3` reference run of the same build). With the transport switched on (neutrino
emissivities/opacities evaluated every step, radiation still at its floor for this cold
initial data) the pair is bit-identical in all 97 fields and all scalars over 21 steps,
at 1.83 s per step. The `-O3` FOFC run also carries a mirror-symmetry violation of
`1.7e-9` (`3.9e-14`, round-off, at `-O1`). Wall time of the head-on runs (8 ranks, 8 GPUs,
20 steps, rank-0 log timestamps): `-O3` 0.99 / 0.98 s per step, `-O1` 0.96 / 0.95 s per
step; total runtime 266.5 / 267.2 s vs 265.5 / 265.1 s. `-O1` costs nothing measurable here.

## 6. Reproducer (about 10 s on one GPU)

Source: GRACE, branch `fix/leptonic-fil-parity` (Hunter copy:
`/lustre/hpe/ws13/ws13.a/ws/xfpmiler-BHNS/grace-m1`). Needs the SFHo tables
`sfho_compose_noele_0-5ye.h5` (251 MB), `sfho_leptons_noele_0-5ye_ken_v2.h5` (216 MB) and the
two cold slices (`/zhome/projects/groups/xfp44203/common/EOS/4D/`), paths set in
`test/configs/c2p_test_replay.yaml`.

```bash
source <grace env>                       # HLRS/APU/2026.1 + rocm/7.0.2, see above
cmake --build build --target c2p_test -j
cd build/test
./c2p_test "[determinism]~[tabulated]" --grace-parfile ./configs/c2p_test_replay.yaml \
    2>&1 | grep -E "mismatches across|assertions"
```

Expected (wrong) at `-O3`: non-zero `mismatches across threads/launches` for the plain
kernels, `0` for the `asm barrier` and `optimize off` lines, `assertions: ... failed`.
Variants: `cmake -DGRACE_C2P_TEST_OPT=-O1 build` (passes), `-O2` (fails);
`./c2p_test "[tabulated]" --grace-parfile ./configs/c2p_test_tabulated.yaml` (passes).
Device assembly: `cmake -DGRACE_C2P_TEST_OPT=-save-temps=obj build`, then
`roc-obj -d -o isa build/test/c2p_test`.

## 7. Attachments

- `test/test_c2p.cpp` (failing function `eval_eos_once`, fixed variants
  `eval_eos_once_barrier`, `eval_eos_once_optnone`, the anatomy kernels),
  `test/configs/c2p_test_replay.yaml`, `test/configs/c2p_test_tabulated.yaml`.
- `isa/k_a_plain.s` (failing kernel), `isa/k_tc_barrier.s` (passing barrier variant),
  extracted from the gfx942 code object of `c2p_test`; the full 253 MB disassembly and the
  per-TU bitcode `test_c2p-hip-amdgcn-amd-amdhsa-gfx942.bc` on request.

## 8. Questions

1. Is this a known wrong-code issue in the roc-7.0.2 AMDGPU backend (SGPR spill / lane-0 /
   `s_cbranch_execz` handling), and is it fixed in a later ROCm?
2. HLRS: `rocm/6.4.1` is the default module but the 2026.1 stack builds against 7.0.2 — can
   a matched Kokkos/GTL build against 6.4.1 be provided, or is a 7.x point release planned?
3. Recommended interim: building GRACE at `-O1` is proven correct and, for the case above,
   free. We still need the fix or a narrower `-mllvm` switch, since heavier configurations
   (five-species neutrino transport with an implicit solver) have not been timed at `-O1`.
