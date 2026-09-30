#!/usr/bin/env python3
"""Convert a GRACE cold EOS table into a LORENE/FUKA `eos_tabul` table.

FUKA solves a cold barotrope and reads LORENE's four-column format

    index   n_B [fm^-3]   rho [g/cm^3]   p [dyn/cm^2]

where the third column is the TOTAL energy density e/c^2, *not* the rest-mass
density -- a distinction that is easy to get wrong and costs ~15% in pressure
at nuclear density if you do.  This script writes that file from the cold
table GRACE generates for itself (eos.leptonic.cold_table_output_filename),
so the star FUKA builds and the star GRACE evolves share one EOS: same
composition (including muons), same baryon mass, same crust.

Two things it fixes relative to hand-assembled tables:

  * The low-density end of a GRACE cold table is the T-floor RADIATION
    pressure, p -> a T^4 / 3, independent of rho.  That is a Gamma ~ 0
    plateau: the sound speed vanishes and FUKA's h -> rho inversion becomes
    degenerate exactly where the stellar surface sits.  We cut it off and
    replace it with a cold crust.
  * The crust is a polytrope matched to the table in p AND e, and C1 in
    Gamma, so there is no discontinuity for a spectral solver to ring on.

Usage:
    grace_cold_to_lorene.py IN.grace OUT.lorene [--rho-min 13.0] [--n-crust 200]
"""

import argparse, math, re, sys

# GRACE units (physical_constants.hh): c = G = Msun = 1, IAU 2015 nominal GMsun.
G_SI     = 6.67430e-11
C_SI     = 299792458.0
GMSUN    = 1.3271244e20
E_SI     = 1.602176634e-19
MU_MEV   = 931.49410242

MSUN_G   = (GMSUN / G_SI) * 1.0e3                  # g
L_CM     = (GMSUN / (C_SI * C_SI)) * 1.0e2         # cm
RHO_UNIT = MSUN_G / L_CM**3                        # g/cm^3 per code unit
P_UNIT   = RHO_UNIT * (C_SI * 1.0e2)**2            # dyn/cm^2 per code unit
MEV_TO_G = E_SI * 1.0e6 / (C_SI * C_SI) * 1.0e3    # g per MeV/c^2
MU_G     = MU_MEV * MEV_TO_G                       # atomic mass unit, g


def read_grace_cold(path):
    """Return (rho_code, p_code, eps, ye, ymu, meta) from a GRACE cold table."""
    meta, rows = {}, []
    for line in open(path):
        if line.startswith('#'):
            m = re.match(r'#\s*([A-Za-z_]+)\s*=\s*([0-9eE+\-.]+)', line)
            if m:
                meta[m.group(1)] = float(m.group(2))
            continue
        cols = line.split()
        if len(cols) in (7, 8):
            try:
                rows.append([float(c) for c in cols])
            except ValueError:
                pass
    if not rows:
        sys.exit("no data rows found in %s" % path)

    ncol = len(rows[0])
    if ncol == 8:      # logrho logT ye ymu logP log(eps+shift) cs2 s
        iy, im, ip, ie = 2, 3, 4, 5
    else:              # logrho logT ye     logP log(eps+shift) cs2 s
        iy, im, ip, ie = 2, None, 3, 4

    shift = meta.get('energy_shift')
    if shift is None:
        sys.exit("cold table has no 'energy_shift' metadata: col5 is "
                 "log(eps + shift) and eps cannot be recovered without it.\n"
                 "Regenerate with a GRACE build that writes it into the header.")
    mb = meta.get('baryon_mass')
    if mb is None:
        sys.exit("cold table has no 'baryon_mass' metadata.")

    rho = [math.exp(r[0]) for r in rows]
    p   = [math.exp(r[ip]) for r in rows]
    eps = [math.exp(r[ie]) - shift for r in rows]
    ye  = [r[iy] for r in rows]
    ymu = [r[im] for r in rows] if im is not None else [0.0] * len(rows)
    return rho, p, eps, ye, ymu, meta, ncol, rows[0][1]


def gamma_at(rho, p, i):
    """Local adiabatic index d ln p / d ln rho, centred where possible."""
    lo = max(i - 1, 0)
    hi = min(i + 1, len(rho) - 1)
    if hi == lo:
        return float('nan')
    return math.log(p[hi] / p[lo]) / math.log(rho[hi] / rho[lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--rho-min', type=float, default=13.0,
                    help='lowest REST-MASS density of the crust extension, g/cm^3')
    ap.add_argument('--n-crust', type=int, default=200,
                    help='number of log-spaced crust points to prepend')
    ap.add_argument('--p-over-prad', type=float, default=20.0,
                    help='trust the table from the lowest density where the pressure\n'
                         'exceeds this multiple of the T-floor radiation pressure')
    ap.add_argument('--no-causality-cut', action='store_true',
                    help='keep rows whose cs^2 exceeds 1 (default: drop them)')
    args = ap.parse_args()

    rho, p, eps, ye, ymu, meta, ncol, rows_T = read_grace_cold(args.infile)
    mb_code = meta['baryon_mass']
    n = len(rho)

    # --- consistency: does the table's baryon mass look like m_u? ------------
    mb_g = mb_code * MSUN_G
    print("cold table : %s" % args.infile)
    print("  rows        %d  (%s)" % (n, "npe-mu" if ncol == 8 else "npe, NO muons"))
    print("  baryon mass %.6e g  = %.6f m_u   (FUKA/LORENE assume m_u)"
          % (mb_g, mb_g / MU_G))
    print("  energy shift %.6e" % meta['energy_shift'])
    print("  Ymu range   %.4e .. %.4e" % (min(ymu), max(ymu)))
    if abs(mb_g / MU_G - 1.0) > 1e-6:
        print("  !! WARNING: not m_u.  FUKA assumes m_u; n_B below is rho/m_b with")
        print("     THIS m_b, so e(n_B) stays right, but GRACE must be told the")
        print("     same convention via eos.tabulated_eos.baryon_mass.")

    # --- drop the acausal tail ---------------------------------------------
    #     The cold table's own cs2 column is clamped at 1, but the (e,p) pair it
    #     stores can still imply dp/de > 1 in the last rows.  FUKA would be free
    #     to sample them, so cut unless asked not to.
    def cs2_at(i):
        e0 = rho[i-1]*(1.0+eps[i-1]); e1 = rho[i]*(1.0+eps[i])
        return (p[i]-p[i-1])/(e1-e0) if e1 > e0 else float('inf')
    top = n
    if not args.no_causality_cut:
        while top > 2 and cs2_at(top-1) > 1.0:
            top -= 1
    if top < n:
        print("\ncausality  : dropped %d top rows with cs^2 > 1 "
              "(above rho = %.4e g/cm^3)" % (n-top, rho[top-1]*RHO_UNIT))

    # --- find where the table stops being radiation-dominated ---------------
    #     At the T floor the pressure tends to a T^4/3, independent of rho: a
    #     Gamma ~ 0 plateau with no sound speed.  Cut it off.  (Do NOT use a
    #     Gamma threshold here -- the inner crust has genuine Gamma < 1 dips.)
    a_rad = 7.5657e-15                              # erg cm^-3 K^-4
    T_K   = math.exp(rows_T) * 1.1604518e10         # MeV -> K
    p_rad = (a_rad * T_K**4 / 3.0) / P_UNIT
    join = 0
    for i in range(top):
        if p[i] > args.p_over_prad * p_rad:
            join = i
            break
    else:
        sys.exit("table never rises above %g x the radiation floor"
                 % args.p_over_prad)

    rj, pj, ej = rho[join], p[join], eps[join]
    gc = gamma_at(rho, p, join)
    gc = min(max(gc, 1.05), 2.0)          # keep the crust stiff but subluminal
    print("\nradiation  : p_rad(T floor) = %.4e code" % p_rad)
    print("crust join : row %d  rho = %.4e code = %.4e g/cm^3  p/p_rad = %.1f"
          % (join, rj, rj*RHO_UNIT, pj/p_rad))
    print("  Gamma there %.4f  (C1-matched polytrope below)" % gc)
    print("  discarded  %d rows of T-floor radiation plateau below it" % join)

    # --- build the crust: p = pj (rho/rj)^gc, eps from the cold first law ----
    #     d eps = p / rho^2 d rho  =>  eps = p / (rho (gc-1)) + const
    rho_lo = args.rho_min / RHO_UNIT
    if rho_lo >= rj:
        sys.exit("--rho-min is above the join density; nothing to extend")
    crust = []
    for k in range(args.n_crust):
        f = k / float(args.n_crust)       # 0 .. just below 1
        r = rho_lo * (rj / rho_lo) ** f
        pk = pj * (r / rj) ** gc
        ek = ej + (pk / r - pj / rj) / (gc - 1.0)
        crust.append((r, pk, ek))

    rho_a = [c[0] for c in crust] + rho[join:top]
    p_a   = [c[1] for c in crust] + p[join:top]
    eps_a = [c[2] for c in crust] + eps[join:top]

    # --- convert to LORENE columns ------------------------------------------
    #     n_B  = rho_rest / m_b        (number density)
    #     'rho'= e/c^2 = rho_rest (1+eps)      <-- LORENE's third column
    nB, ecs, pcs = [], [], []
    for r, pp, ee in zip(rho_a, p_a, eps_a):
        r_g = r * RHO_UNIT
        nB.append(r_g / mb_g / 1.0e39)    # cm^-3 -> fm^-3
        ecs.append(r_g * (1.0 + ee))
        pcs.append(pp * P_UNIT)

    # --- validate ------------------------------------------------------------
    bad = []
    for i in range(1, len(nB)):
        if nB[i] <= nB[i-1]:  bad.append(("n_B not increasing", i))
        if ecs[i] <= ecs[i-1]: bad.append(("e not increasing", i))
        if pcs[i] <= pcs[i-1]: bad.append(("p not increasing", i))
    worst_g, worst_i = 0.0, 0
    for i in range(1, len(nB)):
        g = math.log(pcs[i]/pcs[i-1]) / math.log(ecs[i]/ecs[i-1])
        if g > worst_g: worst_g, worst_i = g, i
    cs2 = [ (pcs[i]-pcs[i-1]) / ((ecs[i]-ecs[i-1]) * (C_SI*100)**2)
            for i in range(1, len(nB)) ]
    print("\nvalidation  : %d points, n_B %.4e .. %.4e fm^-3" % (len(nB), nB[0], nB[-1]))
    print("  monotonic   %s" % ("OK" if not bad else "FAILED: %s" % bad[:3]))
    print("  max d ln p / d ln e   %.4f at row %d" % (worst_g, worst_i))
    print("  cs^2 range  %.4e .. %.4f  %s"
          % (min(cs2), max(cs2), "OK" if min(cs2) > 0 and max(cs2) <= 1.0 else "!! CHECK"))

    # --- write ---------------------------------------------------------------
    with open(args.outfile, 'w') as f:
        f.write("# GRACE cold table -> LORENE eos_tabul, written by grace_cold_to_lorene.py\n")
        f.write("# source: %s\n" % args.infile)
        f.write("# composition: %s beta-equilibrium, baryon mass %.6f m_u\n"
                % ("npe-mu" if ncol == 8 else "npe", mb_g / MU_G))
        f.write("# crust: polytrope Gamma=%.4f below n_B=%.6e fm^-3, C1-matched\n"
                % (gc, nB[args.n_crust]))
        f.write("#\n")
        f.write("%d\n" % len(nB))
        f.write("#\n")
        f.write("# index  n_B [fm^{-3}]  rho [g/cm^3]  p [dyn/cm^2]\n")
        f.write("#\n")
        for i in range(len(nB)):
            f.write("%d    %.16e    %.16e    %.16e\n" % (i + 1, nB[i], ecs[i], pcs[i]))
    print("\nwrote %s" % args.outfile)


if __name__ == '__main__':
    main()
