#!/usr/bin/env python3
"""Lift cells out of a GRACE surface_out_plane_*.h5 for the [replay] c2p test.

One row per selected cell, 28 columns, printed with %.17g so every double
round-trips exactly:

  x y z  gt_xx gt_xy gt_xz gt_yy gt_yz gt_zz  conf_fact alp  beta_x beta_y beta_z
  rho eps press temp entropy ye ymu  z_x z_y z_z  B_x B_y B_z  c2p_err

Cells are selected by rho > --rho-min (default 1e-5: the stars), optionally
thinned to --max-cells with a uniform stride.  Then:

  GRACE_C2P_REPLAY_FILE=cells.txt ./c2p_test "[replay]" \\
      --grace-parfile ./configs/c2p_test_replay.yaml
"""
import argparse
import sys

import h5py
import numpy as np

COLUMNS = ("x y z gt_xx gt_xy gt_xz gt_yy gt_yz gt_zz conf_fact alp "
           "beta_x beta_y beta_z rho eps press temp entropy ye ymu "
           "z_x z_y z_z B_x B_y B_z c2p_err")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", help="surface_out_plane_*.h5 to read")
    ap.add_argument("-o", "--out", default="c2p_replay_cells.txt")
    ap.add_argument("--rho-min", type=float, default=1e-5,
                    help="keep cells with rho above this (default 1e-5)")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="thin the selection to at most this many cells (0 = all)")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    pts, cells = f["Points"][:], f["Cells"][:]
    ctr = pts[cells].mean(axis=1)
    rho = f["rho"][:]
    sel = np.flatnonzero(rho > args.rho_min)
    if args.max_cells and len(sel) > args.max_cells:
        sel = sel[:: int(np.ceil(len(sel) / args.max_cells))]
    if len(sel) == 0:
        sys.exit(f"no cells with rho > {args.rho_min:g}")

    def col(name):
        return f[name][:][sel]

    gt = [col(f"gamma_tilde[{i},{j}]") for i, j in
          ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    beta, zvec, bvec = col("beta"), col("zvec"), col("Bvec")
    table = np.column_stack(
        [ctr[sel, 0], ctr[sel, 1], ctr[sel, 2], *gt,
         col("conf_fact"), col("alp"), beta[:, 0], beta[:, 1], beta[:, 2],
         col("rho"), col("eps"), col("press"), col("temperature"), col("entropy"),
         col("ye"), col("ymu"), zvec[:, 0], zvec[:, 1], zvec[:, 2],
         bvec[:, 0], bvec[:, 1], bvec[:, 2], col("c2p_err")])

    with open(args.out, "w") as out:
        out.write(f"# source: {args.h5}\n")
        out.write(f"# iteration {int(f.attrs['Iteration'])}  time {float(f.attrs['Time']):.17g}"
                  f"  plane {f.attrs['PlaneAxis'].decode() if isinstance(f.attrs['PlaneAxis'], bytes) else f.attrs['PlaneAxis']}\n")
        out.write(f"# selection: rho > {args.rho_min:g}, {len(sel)} of {len(rho)} cells\n")
        out.write(f"# columns: {COLUMNS}\n")
        np.savetxt(out, table, fmt="%.17g")
    print(f"wrote {len(sel)} cells to {args.out}")


if __name__ == "__main__":
    main()
