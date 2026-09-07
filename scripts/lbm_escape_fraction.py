#!/usr/bin/env python3
"""Escaping Killing energy of an isotropic radiation ball around a Schwarzschild puncture.

Prediction from the iteration-0 volume file (isotropic coordinates, M = 1 by default,
psi = 1 + M/2r): every non-excised cell contributes alpha_static Erad1 dV f_esc(r_areal),
alpha_static = (1 - M/2r)/(1 + M/2r), Erad1 = sqrt(gamma) E, r_areal = r (1 + M/2r)^2, with
the Schwarzschild escape fraction of static-frame-isotropic emission
    sin^2 psi_c = 27 M^2 (1 - 2M/r_a) / r_a^2,
    f_esc = (1 + cos psi_c)/2 for r_a >= 3M (captured: inward cone), (1 - cos psi_c)/2 inside.
The initial slice has beta = 0 and K = 0, so the Eulerian frame the LBM ball is isotropic in
IS the static frame.  Measured value: the time integral of the detector luminosity file
Lrad_*.dat (flux r^2 dOmega (alpha F^i - beta^i E) n_i, whole sphere).  An octant volume file
is multiplied by 8 (pass the symmetry factor if different).

usage: lbm_escape_fraction.py VOLUME_it0.h5 Lrad.dat [M=1] [alp_ex=0.3] [symmetry=8]
"""
import math, subprocess, sys

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
            except subprocess.CalledProcessError: pass
    centres, vols = [], []
    for cell in cells:
        cs = [pts[k] for k in cell]
        centres.append([sum(c[a] for c in cs) / 8.0 for a in range(3)])
        vols.append(math.prod(max(c[a] for c in cs) - min(c[a] for c in cs) for a in range(3)))
    return centres, vols, data

def f_esc(ra, M):
    if ra <= 2.0*M: return 0.0
    s2 = 27.0*M*M*(1.0 - 2.0*M/ra)/(ra*ra)
    c = math.sqrt(max(0.0, 1.0 - s2))
    return 0.5*(1.0 + c) if ra >= 3.0*M else 0.5*(1.0 - c)

def main():
    vol, lrad = sys.argv[1], sys.argv[2]
    M = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    alp_ex = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    sym = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    centres, vols, d = read(vol, ["Erad1", "alp", "lbm_ekill1"])
    E_pred = E_kill = E_coord = 0.0; n_ex = 0
    for (x, y, z), dV, Er, al in zip(centres, vols, d["Erad1"], d["alp"]):
        r = math.sqrt(x*x + y*y + z*z)
        if al < alp_ex or r <= 0.5*M: n_ex += 1; continue   # excised, or inside the horizon (alpha_static < 0)
        psi = 1.0 + 0.5*M/r
        a_s = (1.0 - 0.5*M/r)/psi
        ra = r*psi*psi
        E_coord += Er*dV
        E_kill += a_s*Er*dV
        E_pred += a_s*Er*dV*f_esc(ra, M)
    E_coord *= sym; E_kill *= sym; E_pred *= sym
    print(f"cells {len(vols)} (excised {n_ex}); sym x{sym}")
    print(f"initial: int Erad1 dV = {E_coord:.6e}; Killing energy int alpha_static Erad1 dV = {E_kill:.6e}")
    if "lbm_ekill1" in d:
        ek = sym*sum(v*dV for v, dV, al, c in zip(d["lbm_ekill1"], vols, d["alp"], centres)
                     if al >= alp_ex and math.sqrt(c[0]**2 + c[1]**2 + c[2]**2) > 0.5*M)
        print(f"         int lbm_ekill1 dV (with the ID lapse psi^-2) = {ek:.6e}")
    print(f"predicted escaping Killing energy = {E_pred:.6e}  (fraction {E_pred/E_kill:.4f} of the Killing energy)")
    rows = []
    for line in open(lrad):
        parts = line.split()
        if not parts or parts[0].startswith("#"): continue
        try: v = [float(p) for p in parts]
        except ValueError: continue
        rows.append((v[-2], v[-1]))
    E_meas, prev = 0.0, None
    print("   t        L(t)        int L dt")
    for i, (t, L) in enumerate(rows):
        if prev is not None: E_meas += 0.5*(L + prev[1])*(t - prev[0])
        prev = (t, L)
        if i % max(1, len(rows)//12) == 0 or i == len(rows)-1:
            print(f"{t:8.3f}  {L:.6e}  {E_meas:.6e}")
    print(f"measured escaping energy through the detector = {E_meas:.6e}; measured/predicted = {E_meas/E_pred:.4f}")

if __name__ == "__main__":
    main()
