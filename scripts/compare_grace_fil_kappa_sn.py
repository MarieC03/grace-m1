#!/usr/bin/env python
"""Compare GRACE vs FIL scattering (kappa_s) and number (kappa_n) opacities at it=0."""
import os, h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILDIR = "/Users/miler/Graveyard/M1/FIL_fresh"
GRACE  = "/Users/miler/Graveyard/M1/new/surface_out_plane_xy_000000.h5"

def fil_zeq(fname):
    f = h5py.File(fname, "r")
    keys = [k for k in f.keys() if " it=0 " in k and "tl=0" in k]
    by_rl = {}
    for k in keys:
        rl = int(k.split("rl=")[1].split(" ")[0])
        d  = f[k]
        ox, oz = d.attrs["origin"]
        dx, dz = d.attrs["delta"]
        a = np.array(d)
        nz, nx = a.shape
        zc = oz + np.arange(nz) * dz
        if zc.min() - 1e-9 <= 0.0 <= zc.max() + 1e-9:
            j = int(np.argmin(np.abs(zc)))
            xc = ox + np.arange(nx) * dx
            by_rl.setdefault(rl, []).append((xc, a[j, :]))
    pts = {}
    for rl in sorted(by_rl):
        for xc, vals in by_rl[rl]:
            for x, v in zip(xc, vals):
                pts[round(x, 5)] = v
    xs = np.array(sorted(pts))
    return xs, np.array([pts[x] for x in xs])

g = h5py.File(GRACE, "r")
pts = np.array(g["Points"]); cells = np.array(g["Cells"]); cen = pts[cells].mean(1)
gx, gy = cen[:, 0], cen[:, 1]
lvl = np.array(g["Level"])

def grace_line(name):
    v = np.array(g[name])
    fine = lvl == lvl.max()
    dy = np.min(np.diff(np.unique(gy[fine])))
    m = fine & (np.abs(gy) < 0.75 * dy) & (gx > 0)
    order = np.argsort(gx[m])
    return gx[m][order], v[m][order]

# (label, FIL file, GRACE dataset)
FIELDS = [
    ("kappa_nue_s",     "kappa_nue_s.xz.h5",     "kappa_s1"),
    ("kappa_nuebar_s",  "kappa_nue_bar_s.xz.h5",  "kappa_s2"),
    ("kappa_numu_s",    "kappa_numu_s.xz.h5",     "kappa_s3"),
    ("kappa_numubar_s", "kappa_numu_bar_s.xz.h5", "kappa_s4"),
    ("kappa_nux_s",     "kappa_nux_s.xz.h5",      "kappa_s5"),
    ("kappa_nue_n",     "kappa_nue_n.xz.h5",      "kappa_n1"),
    ("kappa_nuebar_n",  "kappa_nue_bar_n.xz.h5",  "kappa_n2"),
    ("kappa_numu_n",    "kappa_numu_n.xz.h5",     "kappa_n3"),
    ("kappa_numubar_n", "kappa_numu_bar_n.xz.h5", "kappa_n4"),
    ("kappa_nux_n",     "kappa_nux_n.xz.h5",      "kappa_n5"),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
print(f"{'field':22} {'GRACE@core':>13} {'FIL@core':>13} {'ratio':>8}")
for ax, (label, filf, gname) in zip(axes.flat, FIELDS):
    if gname not in g:
        ax.set_title(f"{label}\n(not in output)")
        ax.axis("off")
        print(f"{label:22}  SKIPPED ({gname} not in GRACE file)")
        continue
    fp = os.path.join(FILDIR, filf)
    fx, fv = fil_zeq(fp)
    gxx, gvv = grace_line(gname)
    gc = gvv[np.argmin(gxx)] if len(gxx) else np.nan
    fc = fv[np.argmin(np.abs(fx))] if len(fx) else np.nan
    ratio = gc / fc if (fc not in (0, np.nan) and np.isfinite(fc) and fc != 0) else np.nan
    print(f"{label:22} {gc:13.4e} {fc:13.4e} {ratio:8.3f}")
    ax.plot(np.abs(fx), fv, '.', ms=3, label="FIL", alpha=.6)
    ax.plot(gxx, gvv, '-', label="GRACE", lw=1.3)
    ax.set_title(label); ax.set_xlabel("x [code]"); ax.set_xlim(0, 10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.suptitle("kappa_s (top) and kappa_n (bottom): GRACE vs FIL at it=0", y=1.01)
plt.tight_layout()
out = "/Users/miler/Desktop/grace_vs_fil_kappa_sn.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved {out}")
