#!/usr/bin/env python3
"""Decode GRACE's aux(C2P_ERR_) bitmask against the layout of a given build.

The bit numbering of c2p_err_enum_t (include/grace/physics/eos/c2p.hh) is NOT
fixed: C2P_RESET_YMU exists only with GRACE_ENABLE_MUONS, and
C2P_FOFC_FLOORED / C2P_FOFC_DMP only with GRACE_ENABLE_FOFC.  Each present bit
shifts every later enumerator up by one, so the same integer means different
things in different builds.  Decoding against a remembered table is how you
mis-read a run.  This script reads the build's own grace_config.h and decodes
against the layout that build actually compiled.

Usage
-----
  # decode raw values
  decode_c2p_err.py --build build-m1-test 25166094 10485823

  # decode a run's HDF5 output: histogram of the distinct values
  decode_c2p_err.py --build build-m1-test --h5 surface_out_plane_xy_000001.h5

  # ... restricted to the star, with per-bit prevalence
  decode_c2p_err.py --build build-m1-test --h5 out.h5 --rho-min 1e-4 --prevalence

Notes
-----
aux(C2P_ERR_) is a sticky OR over all RK substeps of a timestep (zeroed once
per step at the top of evolve()).  A value is the UNION of what happened, not
a snapshot.  Two consequences when reading the output:
  * C2P_RESET_ENTROPY is set unconditionally on every c2p call
    (c2p.cpp: "by default we overwrite S_star") -- it carries no information.
  * C2P_FOFC_FLOORED / _DMP are stamped by flag_fofc_cells from a DRY RUN
    that writes no state; they do not mean the cell's primitives were touched.
"""

import argparse
import re
import sys

# c2p_err_enum_t in declaration order.  (name, condition) where condition is
# None for always-present, or a macro that must be defined in grace_config.h.
ENUM = [
    ("C2P_RESET_DENS",       None),
    ("C2P_RESET_TAU",        None),
    ("C2P_RESET_STILDE",     None),
    ("C2P_RESET_ENTROPY",    None),
    ("C2P_RESET_YE",         None),
    ("C2P_RESET_YMU",        "GRACE_ENABLE_MUONS"),
    ("C2P_SIG_RHO_TOO_LOW",   None),
    ("C2P_SIG_RHO_TOO_HIGH",  None),
    ("C2P_SIG_EPS_TOO_LOW",   None),
    ("C2P_SIG_EPS_TOO_HIGH",  None),
    ("C2P_SIG_YE_TOO_LOW",    None),
    ("C2P_SIG_YE_TOO_HIGH",   None),
    ("C2P_SIG_ENT_TOO_LOW",   None),
    ("C2P_SIG_ENT_TOO_HIGH",  None),
    ("C2P_SIG_TEMP_TOO_LOW",  None),
    ("C2P_SIG_TEMP_TOO_HIGH", None),
    ("C2P_SIG_PRESS_TOO_LOW", None),
    ("C2P_SIG_PRESS_TOO_HIGH",None),
    ("C2P_SIG_VEL_TOO_HIGH",  None),
    ("C2P_SIG_SIGMA_TOO_HIGH",None),
    ("C2P_ENT_BACKUP_USED",   None),
    ("C2P_ATMO_RESET",        None),
    ("C2P_T_FLOORED",         None),
    ("C2P_FOFC_FLOORED",     "GRACE_ENABLE_FOFC"),
    ("C2P_FOFC_DMP",         "GRACE_ENABLE_FOFC"),
]

# Short human-readable meaning, and whether the bit is an ACTION or diagnostic.
MEANING = {
    "C2P_RESET_DENS":     ("action", "vars(DENS_) rewritten from the c2p's cons"),
    "C2P_RESET_TAU":      ("action", "vars(TAU_) rewritten -- the energy conservative"),
    "C2P_RESET_STILDE":   ("action", "vars(SX_..SZ_) rewritten -- the momentum"),
    "C2P_RESET_ENTROPY":  ("action", "vars(ENTROPYSTAR_) rewritten -- SET UNCONDITIONALLY, no information"),
    "C2P_RESET_YE":       ("action", "vars(YESTAR_) rewritten"),
    "C2P_RESET_YMU":      ("action", "vars(YMUSTAR_) rewritten"),
    "C2P_SIG_RHO_TOO_LOW":   ("diag", "rho below table/atmosphere bound"),
    "C2P_SIG_RHO_TOO_HIGH":  ("diag", "rho above table bound"),
    "C2P_SIG_EPS_TOO_LOW":   ("diag", "eps at/below eps(T_min) -- NORMAL for a cold star at the T floor"),
    "C2P_SIG_EPS_TOO_HIGH":  ("diag", "eps above eps(T_max) -- runaway heating"),
    "C2P_SIG_YE_TOO_LOW":    ("diag", "Ye below table bound"),
    "C2P_SIG_YE_TOO_HIGH":   ("diag", "Ye above table bound"),
    "C2P_SIG_ENT_TOO_LOW":   ("diag", "entropy below range"),
    "C2P_SIG_ENT_TOO_HIGH":  ("diag", "entropy above range"),
    "C2P_SIG_TEMP_TOO_LOW":  ("diag", "T clamped up to the EOS temperature floor"),
    "C2P_SIG_TEMP_TOO_HIGH": ("diag", "T clamped down to the ceiling"),
    "C2P_SIG_PRESS_TOO_LOW": ("diag", "P below range (P-recon hook only)"),
    "C2P_SIG_PRESS_TOO_HIGH":("diag", "P above range (P-recon hook only)"),
    "C2P_SIG_VEL_TOO_HIGH":  ("diag", "Lorentz factor clamped -- real trouble"),
    "C2P_SIG_SIGMA_TOO_HIGH":("diag", "magnetisation clamped"),
    "C2P_ENT_BACKUP_USED":   ("outcome", "energy inversion distrusted, entropy backup recovered the cell"),
    "C2P_ATMO_RESET":        ("outcome", "FULL atmosphere reset -- rest mass in this cell was DESTROYED"),
    "C2P_T_FLOORED":         ("outcome", "T-only floor branch: T,eps,p,s reset, velocity preserved"),
    "C2P_FOFC_FLOORED":      ("fofc",    "dry-run c2p would floor -> first-order flux (no state change)"),
    "C2P_FOFC_DMP":          ("fofc",    "discrete-maximum-principle violation -> first-order flux"),
}


def layout_from_config(path):
    """Return (bit -> name) for the build whose grace_config.h is at `path`."""
    try:
        src = open(path).read()
    except OSError as e:
        sys.exit("cannot read %s: %s\n"
                 "Point --build at a configured build directory." % (path, e))
    defined = set()
    for macro in ("GRACE_ENABLE_MUONS", "GRACE_ENABLE_FOFC"):
        # "#define X" counts; "/* #undef X */" does not.
        if re.search(r"^\s*#\s*define\s+%s\b" % macro, src, re.M):
            defined.add(macro)
    bits, n = {}, 0
    for name, cond in ENUM:
        if cond is not None and cond not in defined:
            continue
        bits[n] = name
        n += 1
    return bits, defined, n


def decode(value, bits):
    v = int(value)
    out, unknown = [], []
    for b in range(64):
        if not (v >> b) & 1:
            continue
        (out if b in bits else unknown).append(b)
    return out, unknown


def describe(value, bits, indent="  "):
    set_bits, unknown = decode(value, bits)
    if not set_bits and not unknown:
        return indent + "(clean -- no bits set)"
    lines = []
    for b in set_bits:
        name = bits[b]
        kind, why = MEANING.get(name, ("?", ""))
        lines.append("%s bit %-2d %-24s [%-7s] %s" % (indent, b, name, kind, why))
    for b in unknown:
        lines.append("%s bit %-2d %-24s [UNKNOWN] not in this build's layout -- "
                     "wrong --build?" % (indent, b, "?"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("values", nargs="*", help="raw c2p_err values to decode")
    ap.add_argument("--build", default="build",
                    help="build directory containing grace_config.h (default: build)")
    ap.add_argument("--h5", help="GRACE HDF5 output; histogram its c2p_err field")
    ap.add_argument("--rho-min", type=float, default=None,
                    help="with --h5, restrict to cells with rho above this")
    ap.add_argument("--top", type=int, default=12,
                    help="with --h5, how many distinct values to show (default 12)")
    ap.add_argument("--prevalence", action="store_true",
                    help="with --h5, also report per-bit prevalence")
    args = ap.parse_args()

    cfg = args.build.rstrip("/") + "/grace_config.h"
    bits, defined, nbits = layout_from_config(cfg)
    print("layout from %s" % cfg)
    print("  GRACE_ENABLE_MUONS=%s  GRACE_ENABLE_FOFC=%s  ->  %d bits\n"
          % ("GRACE_ENABLE_MUONS" in defined,
             "GRACE_ENABLE_FOFC" in defined, nbits))

    for v in args.values:
        print("value %s:" % v)
        print(describe(int(float(v)), bits))
        print()

    if args.h5:
        try:
            import h5py, numpy as np
        except ImportError:
            sys.exit("--h5 needs h5py and numpy")
        f = h5py.File(args.h5, "r")
        if "c2p_err" not in f:
            sys.exit("%s has no c2p_err field (add c2p_err to "
                     "IO.plane_surface_output_cell_variables)" % args.h5)
        err = f["c2p_err"][:]
        sel = np.ones(err.shape, dtype=bool)
        label = "all %d cells" % err.size
        if args.rho_min is not None:
            if "rho" not in f:
                sys.exit("--rho-min needs the rho field in the same file")
            sel = f["rho"][:] > args.rho_min
            label = "%d cells with rho > %g" % (sel.sum(), args.rho_min)
        err = err[sel]
        it = f.attrs.get("Iteration", "?")
        t = f.attrs.get("Time", "?")
        print("%s  (iteration %s, t = %s)\n%s" % (args.h5, it, t, label))
        if err.size == 0:
            sys.exit("no cells selected")
        vals, counts = np.unique(err, return_counts=True)
        order = np.argsort(-counts)
        print("\ndistinct values, most common first:")
        for i in order[: args.top]:
            v = int(vals[i])
            print("\n  value %d  in %d cells (%.1f%%)"
                  % (v, counts[i], 100.0 * counts[i] / err.size))
            print(describe(v, bits, indent="    "))
        if len(order) > args.top:
            print("\n  ... %d further distinct values not shown"
                  % (len(order) - args.top))
        if args.prevalence:
            print("\nper-bit prevalence over %s:" % label)
            ints = err.astype(np.int64)
            for b in sorted(bits):
                frac = 100.0 * (((ints >> b) & 1) == 1).mean()
                if frac > 0:
                    kind, _ = MEANING.get(bits[b], ("?", ""))
                    print("  bit %-2d %-24s [%-7s] %6.1f%%"
                          % (b, bits[b], kind, frac))
        f.close()

    if not args.values and not args.h5:
        print("Nothing to decode.  Pass raw values and/or --h5.  Full layout:\n")
        for b in sorted(bits):
            kind, why = MEANING.get(bits[b], ("?", ""))
            print("  bit %-2d %-24s [%-7s] %s" % (b, bits[b], kind, why))


if __name__ == "__main__":
    main()
