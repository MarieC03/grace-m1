#!/usr/bin/env python
"""Compare GRACE it=0 vs FIL it=0 along the radial x-axis (z=0 / y=0).

FIL  : Carpet HDF5, xz-plane slice, multiple refinement levels (rl) & components (c).
       Extract the z=0 row from each finest patch -> (x, value); finer rl wins.
GRACE: cell output, xy-plane; take the y~0 cells -> (x, value).
Both are in geometric/code units (c=G=Msun=1), so directly comparable.
"""
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILDIR = "/Users/miler/Graveyard/M1/FIL_fresh"
GRACE  = "/Users/miler/Graveyard/M1/new/surface_out_plane_xy_000000.h5"

# -------------------------------------------------- FIL Carpet reader (z=0 line)
def fil_zeq(fname):
    """Return (x, val) sampled on z=0, taking the finest refinement level
    available at each x."""
    f = h5py.File(fname, "r")
    keys = [k for k in f.keys() if " it=0 " in k and "tl=0" in k]
    by_rl = {}
    for k in keys:
        rl = int(k.split("rl=")[1].split(" ")[0])
        d  = f[k]
        ox, oz = d.attrs["origin"]          # [x, z] of corner
        dx, dz = d.attrs["delta"]
        a = np.array(d)                     # shape (nz, nx): last index = x
        nz, nx = a.shape
        zc = oz + np.arange(nz) * dz
        # nearest z=0 row, only if 0 is within this patch
        if zc.min() - 1e-9 <= 0.0 <= zc.max() + 1e-9:
            j = int(np.argmin(np.abs(zc)))
            xc = ox + np.arange(nx) * dx
            by_rl.setdefault(rl, []).append((xc, a[j, :]))
    # assemble, finest rl wins
    pts = {}
    for rl in sorted(by_rl):                # coarse -> fine, fine overwrites
        for xc, vals in by_rl[rl]:
            for x, v in zip(xc, vals):
                pts[round(x, 5)] = v
    xs = np.array(sorted(pts))
    return xs, np.array([pts[x] for x in xs])

def fil_prefix(fname):
    f = h5py.File(fname, "r")
    k = next(k for k in f.keys() if " it=0 " in k)
    return k.split(" it=")[0]

# ----------------------------------------------------- GRACE y=0 radial line
g = h5py.File(GRACE, "r")
pts = np.array(g["Points"]); cells = np.array(g["Cells"]); cen = pts[cells].mean(1)
gx, gy = cen[:, 0], cen[:, 1]
lvl = np.array(g["Level"])
def grace_line(name):
    v = np.array(g[name])
    # finest level, |y| < half a finest cell
    fine = lvl == lvl.max()
    dy = np.min(np.diff(np.unique(gy[fine])))
    m = fine & (np.abs(gy) < 0.75 * dy) & (gx > 0)
    order = np.argsort(gx[m])
    return gx[m][order], v[m][order]

# --------------------------------------------------------- field map FIL<->GRACE
# (label, FIL file, GRACE dataset, log-plot?)
FIELDS = [
    ("rho",         "rho.xz.h5",            "rho",      False),
    ("eta_nue",     "eta_nue.xz.h5",        "eta_nu1",  False),
    ("kappa_nue_a", "kappa_nue_a.xz.h5",    "kappa_a1", False),
    ("kappa_nuebar_a","kappa_nue_bar_a.xz.h5","kappa_a2",False),
    ("kappa_numu_a","kappa_numu_a.xz.h5",   "kappa_a3", False),
    ("kappa_numubar_a","kappa_numu_bar_a.xz.h5","kappa_a4",False),
    ("kappa_nux_a", "kappa_nux_a.xz.h5",    "kappa_a5", False),
    ("Qnue",        "Qnue.xz.h5",           "eta1",     False),
    ("Rnue",        "Rnue.xz.h5",           "eta_n1",   False),
]

import os
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
print(f"{'field':18} {'GRACE@core':>13} {'FIL@core':>13} {'ratio':>8}")
for ax, (label, filf, gname, logy) in zip(axes.flat, FIELDS):
    if gname not in g:
        ax.set_title(f"{label}\n(not in output — needs GRACE_M1_DEBUG_EAS)")
        ax.axis("off")
        print(f"{label:18}  SKIPPED ({gname} not in GRACE file)")
        continue
    fp = os.path.join(FILDIR, filf)
    fx, fv = fil_zeq(fp)
    gxx, gvv = grace_line(gname)
    # core values (smallest radius)
    gc = gvv[np.argmin(gxx)] if len(gxx) else np.nan
    fc = fv[np.argmin(np.abs(fx))] if len(fx) else np.nan
    ratio = gc / fc if (fc not in (0, np.nan) and np.isfinite(fc) and fc != 0) else np.nan
    print(f"{label:18} {gc:13.4e} {fc:13.4e} {ratio:8.3f}")
    if logy:
        ax.semilogy(np.abs(fx), np.abs(fv), '.', ms=3, label="FIL", alpha=.6)
        ax.semilogy(gxx, np.abs(gvv), '-', label="GRACE", lw=1.3)
    else:
        ax.plot(np.abs(fx), fv, '.', ms=3, label="FIL", alpha=.6)
        ax.plot(gxx, gvv, '-', label="GRACE", lw=1.3)
    ax.set_title(label); ax.set_xlabel("x [code]"); ax.set_xlim(0, 10); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
out = "/Users/miler/Desktop/grace_vs_fil_it0.png"
plt.savefig(out, dpi=110)
print(f"\nsaved {out}")
