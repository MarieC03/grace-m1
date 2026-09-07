#!/usr/bin/env python3
"""Compare the scalar outputs (output_scalar/*.dat) of two GRACE runs.

usage: lbm_compare_scalars.py DIR_A DIR_B [rel_tol=1e-12]
Prints the largest relative difference per file and exits 1 if any exceeds the tolerance.
"""
import glob, os, sys

def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("Iteration") or not line.strip(): continue
            rows.append([float(v) for v in line.split()])
    return rows

def main():
    a, b = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
    bad = False
    files = sorted(glob.glob(os.path.join(a, "*.dat")))
    if not files:
        print(f"no .dat files in {a} -- nothing compared"); sys.exit(2)
    for fa in files:
        fb = os.path.join(b, os.path.basename(fa))
        if not os.path.exists(fb):
            print(f"{os.path.basename(fa):32s} missing in {b}"); bad = True; continue
        ra, rb = read(fa), read(fb)
        n = min(len(ra), len(rb)); worst = 0.0
        for i in range(n):
            for va, vb in zip(ra[i][1:], rb[i][1:]):
                scale = max(abs(va), abs(vb), 1e-300)
                worst = max(worst, abs(va - vb)/scale)
        flag = "OK" if worst <= tol else "DIFF"
        if worst > tol: bad = True
        print(f"{os.path.basename(fa):32s} rows {n:4d}  max rel diff {worst:.3e}  {flag}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
