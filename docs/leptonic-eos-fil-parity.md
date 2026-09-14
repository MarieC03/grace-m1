# Leptonic EOS: FIL parity at the table boundaries

*Sep 2026. Written so that reverting any piece of this is a five-minute job.*

Reference: `Margherita_EOS/src/4D_Table/leptonic_eos_implementation.hh` in FIL.

## The problem this fixes

`leptonic_eos_4d_t` mixed two self-consistent designs and got a broken one.
Three codes handle "the state is sitting on the bottom of the EOS table" in two
different but internally consistent ways:

| | `limit_temp` clamps T to | eps bracket flags `EPS_TOO_LOW`? | consistent? |
|---|---|---|---|
| FIL / Margherita | `eos_tempmin` exactly (impl.hh:133) | **no** — clamps and returns, no error code (impl.hh:188) | yes: being at the floor is a normal state |
| GRACE `tabulated_eos` | `(1+1e-2)*eos_tempmin` (tabulated_eos.hh:843) | yes | yes: never reaches the bracket bottom |
| GRACE `leptonic_eos_4d` **(before this change)** | `tmin` exactly | yes | **no — FIL's clamp with GRACE's flag** |

Consequence of the broken pairing: a cold star initialised at the table's
temperature floor has `eps == eps_lo` by construction, the `eps <= eps_lo`
test fires on equality, `EOS_EPS_TOO_LOW` is raised, and `c2p.hh:343` turns
that into `C2P_RESET_STILDE + C2P_RESET_TAU` — the momentum and energy
conservatives of **every cell** rewritten on **every** c2p call.

Measured on a FUKA head-on (`eos_type: leptonic`, `temp_fl: 0.1` = the table's
first T point): `SIG_EPS_TOO_LOW` in 100 % of star cells after one step,
`RESET_TAU`/`RESET_STILDE` in 100 %. The same ID under `eos_type: tabulated`
at T = 0.101: 0 %.

**Important caveat, do not lose this.** The reset is a *resync*, not corruption:
the tail of `conservs_to_prims` re-densitizes (`cons[TAUL] = sqrtg * tau`,
c2p.cpp:531), so it writes a self-consistent conservative recomputed from the
clamped primitives. When the clamp is a rounding-level nudge, so is the
rewrite. A cold TOV at exactly `T_min` runs **clean** on the pre-fix code —
verified, twice, with and without FOFC. So this pairing was wrong and noisy,
but it was **never demonstrated to be what destroyed the head-on run**. Do not
cite it as that cause without a fresh reproduction.

## What changed

### 1. The eps and press brackets clamp silently near the boundary

`leptonic_eos_4d.hh`, `ltemp__eps_lrho_ye_ymu` and `ltemp__press_lrho_ye_ymu`:

```cpp
if (eps <= eps_lo) {
    bool const significant = eps < eps_lo - eps_bracket_tol_ * fabs(eps_lo) ;
    eps = eps_lo ;
    if (significant) err.set(EOS_EPS_TOO_LOW) ;
    return ltempmin ;
}
```

`eps_bracket_tol_ = 1e-8`: far above round-off (~1e-16), far below any
physical energy deficit. Exact FIL parity would be *no* flag at all; the
tolerance keeps genuine c2p divergence detectable.

The entropy bracket (`ltemp__entropy_lrho_ye_ymu`) already had no `err`
parameter and was always silent — unchanged.

### 2. The temperature floor is the table minimum again

`set_temperature_floor(requested)`: `requested <= 0` → the table minimum
exactly (FIL parity). An explicit value is used verbatim for the EOS clamp,
the generated cold slice, and therefore the temperature the FUKA/TOV importers
write into the ID. A value below the table minimum is raised to it with a
warning.

**These two are a matched pair.** (2) is only safe because of (1). If you
restore the flag in (1), you *must* restore a margin in `limit_temp`, or you
are back to the broken pairing. The comment in `limit_temp` says so.

### 3 and 4 — unrelated to parity, listed so the diff is fully accounted for

- **dilute-Yμ**: muon P/eps/entropy are zeroed and the muon table lookup
  skipped below `dilute_ymu0_ = 6e-4`, per the GMUNU dilute-Yμ note. The tanh
  ramp still applies to `mu_mu` only. Verified surgical: the generated cold
  slice changes only in the dilute band (max |dP/P| 8.8e-4, |ds/s| 1.7e-2) and
  is **bit-identical** above the threshold. Also a speedup.
- **sound speed**: `total_csnd2` returns the baryon table's `TABCSND2`
  unchanged when `add_ele_contribution == false` (exact FIL behaviour,
  impl.hh:611 — correct, because FIL pairs an electrons-*included* table).
  With `add_ele_contribution == true` the baryon `CS2` is baryons-only and
  goes **negative** across the spinodal (2454 of 5100 grid points on
  `sfho_compose_noele_0-5ye.h5` at T = 0.1 over 2e-4 < n_b < 0.5, min −3.5e-2;
  the electrons-included table has none), so it is rebuilt from the additive
  P and eps. Clamped to (0,1] either way. **Latent, not a live bug**: the
  star's own β-equilibrium track (y_p ≈ 0.02–0.08) does not enter the negative
  region — 0 of 3800 cells measured.

## How to revert

Each item is independent except the (1)+(2) pair.

| revert | how |
|---|---|
| (1) bracket tolerance | In `ltemp__eps_lrho_ye_ymu` and `ltemp__press_lrho_ye_ymu`, replace each guarded block with the one-liner `if (eps <= eps_lo) { eps = eps_lo ; err.set(EOS_EPS_TOO_LOW) ; return ltempmin ; }`. **Then also revert (2)**, or cold stars break. |
| (2) floor default | In `set_temperature_floor`, change `temp_floor_ = (requested > tmin) ? requested : tmin ;` back to an automatic `(1.+1e-2)*tmin`, and `temp_ceil_ = (1.-1e-2)*tmax`. Safe on its own; it only costs you the ability to set `tmin` = the table minimum. |
| (3) dilute-Yμ | In `total_press` / `total_eps` / `total_entropy`, drop the `if (muons_resolved(ymu))` guard and interpolate the two muon slots unconditionally, as before. `muons_resolved` then has no callers. |
| (4) sound speed | Point the eight `csnd2 = total_csnd2(...)` call sites back at a plain `baryon_table.interp(lrho, ltemp, table_yp(ye,ymu), TABCSND2)`. Loses the negative-cs² guard. |

Full set: `git log --oneline -- include/grace/physics/eos/leptonic_eos_4d.hh`
and revert the commit carrying this file.

## Test that encodes the contract

`test/test_c2p.cpp`, section *"temp_fl strictly below exp(ltempmin): no T-floor
events"*. It previously asserted the recovered T equals the raw table minimum;
it now asserts it equals `eos.temperature_floor()`. With (2) reverted to a
margin, `temperature_floor()` is above `t_min` and the extra
`REQUIRE(eos.temperature_floor() > t_min)` still holds; with (2) as shipped,
that `REQUIRE` must be dropped since the floor *is* the table minimum by
default.

## Known-unrelated failure

`m1_analytic_rates_test` fails 2 of 1565 assertions (NUMU `eta_E`). Pre-existing,
from the hard-step → sigmoid muon gate change; verified to fail identically with
all of the above stashed. Not a regression from this work.
