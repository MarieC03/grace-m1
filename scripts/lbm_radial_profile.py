#!/usr/bin/env python3
"""Radial profile of cell-centred variables from a GRACE volume output file.

Reads /Points, /Cells and the requested datasets (h5py if present, else the
h5dump CLI), reconstructs cell centres, and prints the shell-averaged value,
the angular RMS scatter and the analytic reference for the LBM test problems.

usage: lbm_radial_profile.py FILE.h5 {sphere R t | diffusion ks t0 t dx | shadow x0 x1 | beam z0 [M] [dx] | blob z0 [M]}
"""
import subprocess, sys, math

def _h5dump(path, dset):
    out = subprocess.run(["h5dump", "-d", dset, "-y", "-w", "100000", path],
                         capture_output=True, text=True, check=True).stdout
    body = out.split("DATA {", 1)[1].split("}", 1)[0]
    return [float(v) for v in body.replace("\n", " ").split(",") if v.strip()]

def read(path, names):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            pts = f["/Points"][...].reshape(-1, 3).tolist()
            cells = f["/Cells"][...].reshape(-1, 8).tolist()
            data = {n: f["/" + n][...].ravel().tolist() for n in names if "/" + n in f}
    except ImportError:
        p = _h5dump(path, "/Points"); pts = [p[i:i+3] for i in range(0, len(p), 3)]
        c = _h5dump(path, "/Cells");  cells = [[int(v) for v in c[i:i+8]] for i in range(0, len(c), 8)]
        data = {}
        for n in names:
            try: data[n] = _h5dump(path, "/" + n)
            except subprocess.CalledProcessError: pass   # not written by this run
    centres = [[sum(pts[k][a] for k in cell) / 8.0 for a in range(3)] for cell in cells]
    return centres, data

def main():
    path, mode = sys.argv[1], sys.argv[2]
    # the flux is written as one (ncell x 3) dataset "Frad1"; split it into components
    centres, d = read(path, ["Erad1", "Frad1"])
    names = ["Erad1", "Fradx1", "Frady1", "Fradz1"]
    if "Frad1" in d and len(d["Frad1"]) == 3*len(centres):
        F = d.pop("Frad1")
        for a, n in enumerate(names[1:]): d[n] = F[a::3]
    if mode == "sphere":
        R, t = float(sys.argv[3]), float(sys.argv[4]); dr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03125
        # flash from a uniform ball: E(r,t) = E0 (R^2 - (r-t)^2) / (4 r t) for |r-t| < R
        ana = lambda r: max(R*R - (r-t)**2, 0.0) / (4.0*r*t) if r > 0 else 0.0
        lo, hi = t - R - 3*dr, t + R + 3*dr
    elif mode == "blob":
        # a radiation packet launched tangentially at radius z0: its E-weighted centroid and its
        # peak should sit on the null geodesic z_geo(x) of impact parameter b = z0/sqrt(1-2M/z0)
        z0 = float(sys.argv[3]); M = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        u, du, phi, h = 1.0/z0, 0.0, 0.0, 1e-4; curve = []
        while phi < 1.6 and u > 1e-3:
            def f(u, du): return (du, -u + 3.0*M*u*u)
            k1 = f(u, du); k2 = f(u + 0.5*h*k1[0], du + 0.5*h*k1[1])
            k3 = f(u + 0.5*h*k2[0], du + 0.5*h*k2[1]); k4 = f(u + h*k3[0], du + h*k3[1])
            u += h*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6; du += h*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6; phi += h
            r = 1.0/u; curve.append((r*math.sin(phi), r*math.cos(phi)))
        def z_geo(x): return min(curve, key=lambda q: abs(q[0]-x))[1]
        sE = sx = sy = sz = szz = 0.0; peak = (-1.0, 0.0, 0.0, 0.0)
        for (x, y, z), E in zip(centres, d["Erad1"]):
            if E < 1e-8: continue
            sE += E; sx += E*x; sy += E*y; sz += E*z; szz += E*z*z
            if E > peak[0]: peak = (E, x, y, z)
        if sE == 0: print("no radiation above 1e-8"); return
        xc, zc = sx/sE, sz/sE; zr = math.sqrt(max(szz/sE - zc*zc, 0.0))
        print(f"sumE={sE:.4e}  centroid x={xc:.3f} y={sy/sE:.3f} z={zc:.3f}  z_geo(x_c)={z_geo(xc):.3f}  dz_c={zc-z_geo(xc):+.3f}  z_rms={zr:.3f} | "
              f"peak E={peak[0]:.3e} at x={peak[1]:.3f} z={peak[3]:.3f}  z_geo(x_p)={z_geo(peak[1]):.3f}  dz_p={peak[3]-z_geo(peak[1]):+.3f}")
        return
    elif mode == "beam":
        # E-weighted centroid of the beam per x-bin against the Schwarzschild null geodesic
        # launched tangentially at radius z0 (impact parameter b = z0/sqrt(1-2M/z0)):
        # u = 1/r, u'' = -u + 3 M u^2 in phi, r(phi=0) = z0, x = r sin(phi), z = r cos(phi).
        z0 = float(sys.argv[3]); M = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        dxb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.375
        u, du, phi, h = 1.0/z0, 0.0, 0.0, 1e-4; curve = []
        while phi < 1.6 and u > 1e-3:
            def f(u, du): return (du, -u + 3.0*M*u*u)
            k1 = f(u, du); k2 = f(u + 0.5*h*k1[0], du + 0.5*h*k1[1])
            k3 = f(u + 0.5*h*k2[0], du + 0.5*h*k2[1]); k4 = f(u + h*k3[0], du + h*k3[1])
            u += h*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6; du += h*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6; phi += h
            r = 1.0/u; curve.append((r*math.sin(phi), r*math.cos(phi)))
        def z_geo(x):
            best = min(curve, key=lambda p: abs(p[0]-x)); return best[1]
        bins = {}
        for (x, y, z), E in zip(centres, d["Erad1"]):
            if x < 0 or E < 1e-8: continue
            b = bins.setdefault(int(x/dxb), [0.0, 0.0, 0.0, 0.0])
            b[0] += E; b[1] += E*z; b[2] += E*z*z; b[3] += E*y
        print(f"{'x':>7} {'sumE':>12} {'z_centroid':>11} {'z_geodesic':>11} {'dz':>8} {'z_rms':>7} {'y_cen':>7}   (z0={z0}, b={z0/math.sqrt(1-2*M/z0):.3f})")
        for k in sorted(bins):
            sE, sz, szz, sy = bins[k]; x = (k + 0.5)*dxb; zc = sz/sE; zr = math.sqrt(max(szz/sE - zc*zc, 0.0)); zg = z_geo(x)
            print(f"{x:7.3f} {sE:12.4e} {zc:11.4f} {zg:11.4f} {zc-zg:+8.4f} {zr:7.3f} {sy/sE:7.3f}")
        return
    elif mode == "shadow":
        x0, x1 = float(sys.argv[3]), float(sys.argv[4]); dr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.015625
        bins = {}
        for (x, y, z), E in zip(centres, d["Erad1"]):
            if not (x0 <= x < x1): continue
            rho = math.sqrt(y*y + z*z)
            if rho >= 0.125: continue
            b = bins.setdefault(int(rho/dr), [0, 0.0, 0.0, 1e300])
            b[0] += 1; b[1] += E; b[2] = max(b[2], E); b[3] = min(b[3], E)
        print(f"{'rho':>7} {'ncell':>6} {'<E>':>12} {'E_max':>12} {'E_min':>12}   (x in [{x0},{x1}])")
        for k in sorted(bins):
            n, s1, mx, mn = bins[k]; print(f"{(k+0.5)*dr:7.4f} {n:6d} {s1/n:12.5e} {mx:12.5e} {mn:12.5e}")
        return
    else:
        ks, t0, t, dx = map(float, sys.argv[3:7]); dr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.03125
        # numerical diffusion coefficient of the LBM scheme (Olsen & Rezzolla 2025): D = (1 + 0.75 ks dx)/(3 ks)
        te = t0 + (1.0 + 0.75*ks*dx)*t
        ana = lambda r: (ks/t0)**1.5 * (t0/te)**1.5 * math.exp(-0.75*ks*r*r/te)
        lo, hi = 0.0, 0.8
    bins = {}
    haveF = all(n in d for n in names[1:])
    zero = [0.0]*len(centres)
    for (x, y, z), E, Fx, Fy, Fz in zip(centres, d["Erad1"], *( [d[n] for n in names[1:]] if haveF else [zero]*3 )):
        r = math.sqrt(x*x + y*y + z*z)
        if not (lo <= r < hi): continue
        fr = (x*Fx + y*Fy + z*Fz) / (r*E) if haveF and r > 0 and E > 0 else float("nan")
        b = bins.setdefault(int(r/dr), [0, 0.0, 0.0, 0.0])
        b[0] += 1; b[1] += E; b[2] += E*E; b[3] += fr
    print(f"{'r':>7} {'ncell':>6} {'<E>':>12} {'rms/<E>':>8} {'E_analytic':>12} {'ratio':>7} {'<F_r/E>':>8}"
          + ("  r^2<E>" if mode == "sphere" else ""))
    for k in sorted(bins):
        n, s1, s2, sf = bins[k]; r = (k + 0.5)*dr; m = s1/n; rms = math.sqrt(max(s2/n - m*m, 0.0))
        a = ana(r); ratio = m/a if a > 0 else float("nan")
        line = f"{r:7.4f} {n:6d} {m:12.5e} {rms/m if m>0 else 0:8.3f} {a:12.5e} {ratio:7.3f} {sf/n:8.3f}"
        if mode == "sphere": line += f"  {r*r*m:.4e}"
        print(line)

if __name__ == "__main__":
    main()
