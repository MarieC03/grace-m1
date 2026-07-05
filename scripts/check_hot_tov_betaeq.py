#!/usr/bin/env python
"""
Reproduce GRACE's hot_tov initial data at the EOS-table level.

What hot_tov does:  pick a density rho (from the TOV structure), set T = 20 MeV,
and solve for (Ye, Ymu) such that the neutrino chemical potentials vanish:
    mu_nue  = mu_e  + mu_p - mu_n - Qnp = 0
    mu_numu = mu_mu + mu_p - mu_n - Qnp = 0
exactly the conditions in leptonic_eos_4d_t::betaeq_ye_ymu__rho_temp.

Then it RE-EVALUATES those same mu_nue / mu_numu the way make_fugacity_state
does at runtime, and checks they are actually ~0.  If they are not, the
read-in/convention of the tables is inconsistent between the beta-eq solver
and the fugacity (the suspected 170-MeV bug).

Tables (same files the run uses):
  baryon  : CompOSE  Thermo_qty/thermo  -> mu_n = mu_b, mu_p = mu_b + mu_q
  electron: electronic_eos_tables[0] = mu_e   (axis Yle)
  muon    : muonic_eos_tables[0]     = mu_mu  (axis log Ymu)
All chemical potentials are in MeV.
"""
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BARYON = "/Users/miler/Codes/eos/4D/sfho_compose_electrons_ye0-5.h5"
LEPTON = ("/Users/miler/Codes/eos/4D/"
          "sfhoele_0-5ye_newformat_1000mev_mu_max_fullfermi_allfixed_"
          "Bollig_typeIII_hotfixMuon_electron_tables.h5")
QNP = 1.29333236  # m_n - m_p [MeV]

# ---------------------------------------------------------------- load tables
lep = h5py.File(LEPTON, "r")
logrho = np.array(lep["logrho_table"])    # log10(rho_cgs), 308
logT   = np.array(lep["logtemp_table"])   # log10(T_MeV),   81
yle    = np.array(lep["yle_table"])       # linear Y_le,     60
lymu   = np.array(lep["ymu_table"])       # log(Y_mu),       60
# electronic_eos_tables / muonic_eos_tables: (var, y, T, rho) -> (rho, T, y)
mu_e_grid  = np.transpose(np.array(lep["electronic_eos_tables"])[0], (2, 1, 0))
mu_mu_grid = np.transpose(np.array(lep["muonic_eos_tables"])[0],     (2, 1, 0))

bar = h5py.File(BARYON, "r")
th   = np.array(bar["Thermo_qty/thermo"])   # (9, yq, T, nb) ; idx 2,3,4 = mu_b,mu_q,mu_l
yq   = np.array(bar["Parameters/yq"])       # charge fraction, 60
mu_b_grid = np.transpose(th[2], (2, 1, 0))  # -> (rho, T, yq)
mu_q_grid = np.transpose(th[3], (2, 1, 0))
mu_l_grid = np.transpose(th[4], (2, 1, 0))

# sanity: shared rho/T grid
assert mu_e_grid.shape == mu_b_grid.shape == (len(logrho), len(logT), len(yle))

# interpolators (clamp out-of-range to the edge, like the C++ limit_* calls)
def rgi(grid, ax3):
    return RegularGridInterpolator((logrho, logT, ax3), grid,
                                   bounds_error=False, fill_value=None)
I_mue  = rgi(mu_e_grid,  yle)
I_mumu = rgi(mu_mu_grid, lymu)
I_mub  = rgi(mu_b_grid,  yq)
I_muq  = rgi(mu_q_grid,  yq)
I_mul  = rgi(mu_l_grid,  yq)

yle_lo, yle_hi = yle[0], yle[-1]
lymu_lo, lymu_hi = lymu[0], lymu[-1]

# nucleon potentials after the CompOSE conversion (mu_n=mu_b, mu_p=mu_b+mu_q)
def mu_n(lr, lt, yp): return float(I_mub((lr, lt, yp)))
def mu_p(lr, lt, yp): return float(I_mub((lr, lt, yp)) + I_muq((lr, lt, yp)))
def mu_e(lr, lt, y):  return float(I_mue((lr, lt, y)))
def mu_mu(lr, lt, ly):return float(I_mumu((lr, lt, ly)))

# ------------------------------------------------- beta-eq solve (hot_tov port)
def find_ye_for_mue(lr, lt, target):
    lo, hi = mu_e(lr, lt, yle_lo), mu_e(lr, lt, yle_hi)
    if target <= min(lo, hi): return yle_lo
    if target >= max(lo, hi): return yle_hi
    return brentq(lambda y: mu_e(lr, lt, y) - target, yle_lo, yle_hi, xtol=1e-12)

def solve_betaeq(lr, lt):
    """Return (ye, ymu, muons_present)."""
    def outer(ly):
        mm  = mu_mu(lr, lt, ly)
        yel = find_ye_for_mue(lr, lt, mm)
        yp  = yel + np.exp(ly)
        return (mu_n(lr, lt, yp) - mu_p(lr, lt, yp)) - mm + QNP
    if outer(lymu_lo) * outer(lymu_hi) > 0.0:
        # no muons -> pure npe: mu_e(ye) - (mu_n-mu_p)(ye) - Qnp = 0
        def npe(y): return mu_e(lr, lt, y) - (mu_n(lr, lt, y) - mu_p(lr, lt, y)) - QNP
        a, b = npe(yle_lo), npe(yle_hi)
        if a * b < 0.0:
            ye = brentq(npe, yle_lo, yle_hi, xtol=1e-12)
        else:
            ye = yle_lo if abs(a) < abs(b) else yle_hi
        return ye, np.exp(lymu_lo), False
    ly = brentq(outer, lymu_lo, lymu_hi, xtol=1e-12)
    ymu = np.exp(ly)
    ye  = find_ye_for_mue(lr, lt, mu_mu(lr, lt, ly))
    return ye, ymu, True

# ----------------------------------------------------- evaluate the fugacities
def eval_point(lr, lt, ye, ymu):
    yp = ye + ymu
    me  = mu_e(lr, lt, ye)             # from ELECTRONIC table at Ye
    mp  = mu_p(lr, lt, yp)             # from BARYON table at yp=Ye+Ymu
    mn  = mu_n(lr, lt, yp)
    mm  = mu_mu(lr, lt, np.log(ymu))
    mnue  = me + mp - mn - QNP
    mnumu = mm + mp - mn - QNP
    # baryon table's OWN electron mu (mu_l - mu_q) at yp -- the consistent one
    me_bar = float(I_mul((lr, lt, yp)) - I_muq((lr, lt, yp)))
    return dict(yp=yp, mu_e=me, mu_e_baryon=me_bar, mu_p=mp, mu_n=mn,
                mu_mu=mm, munp=mn - mp, mu_nue=mnue, mu_numu=mnumu)

# ----- unit helpers -----------------------------------------------------------
# logrho_table = ln(rho_code), logtemp_table = ln(T_MeV)  (natural log!).
# rho_code = rho_cgs * RHOGF.
RHOGF = 1.61887093132742e-18      # code rho per (g/cm^3)
def lr_of_cgs(rho_cgs): return np.log(rho_cgs * RHOGF)
def cgs_of_lr(lr):      return np.exp(lr) / RHOGF

# ============================================================ core point report
T_ID = 20.0
lt = np.log(T_ID)                 # natural log
lr_core = lr_of_cgs(7.2e14)       # NS core ~ 7.2e14 g/cc
ye, ymu, has_mu = solve_betaeq(lr_core, lt)
d = eval_point(lr_core, lt, ye, ymu)
print(f"\n=== CORE  rho={cgs_of_lr(lr_core):.3e} g/cc (code {np.exp(lr_core):.3e})  T={T_ID} MeV ===")
print(f"  solved beta-eq:  Ye={ye:.5f}   Ymu={ymu:.5f}   muons={has_mu}")
print(f"  mu_e (electronic table) = {d['mu_e']:9.3f} MeV")
print(f"  mu_e (baryon mu_l-mu_q) = {d['mu_e_baryon']:9.3f} MeV   <-- consistent one")
print(f"  mu_p                    = {d['mu_p']:9.3f} MeV")
print(f"  mu_n                    = {d['mu_n']:9.3f} MeV")
print(f"  mu_n - mu_p             = {d['munp']:9.3f} MeV   (should be ~ mu_e-Qnp)")
print(f"  mu_mu                   = {d['mu_mu']:9.3f} MeV")
print(f"  --> mu_nue  = mu_e+mu_p-mu_n-Qnp = {d['mu_nue']:9.3f} MeV   (should be 0)")
print(f"  --> mu_numu = mu_mu+mu_p-mu_n-Qnp= {d['mu_numu']:9.3f} MeV   (should be 0)")
# what mu_nue would be if we used the CONSISTENT baryon-table mu_e:
mnue_consistent = d['mu_e_baryon'] + d['mu_p'] - d['mu_n'] - QNP
print(f"  --> mu_nue with baryon mu_e      = {mnue_consistent:9.3f} MeV "
      f"(diff isolates the electronic-table baseline)")

# ================================================== sweep the T=20 slice & plot
# rho from ~1e11 g/cc up to the top of the table.
lrs = np.linspace(lr_of_cgs(1e11), logrho[-1] - 1e-3, 80)
YE, YMU, MNUE, MNUMU, MUE, MUE_B, MUNP, MUMU = ([] for _ in range(8))
for lr in lrs:
    try:
        y, ym, _ = solve_betaeq(lr, lt)
        r = eval_point(lr, lt, y, ym)
    except Exception:
        y = ym = np.nan; r = dict(mu_nue=np.nan, mu_numu=np.nan, mu_e=np.nan,
                                  mu_e_baryon=np.nan, munp=np.nan, mu_mu=np.nan)
    YE.append(y); YMU.append(ym); MNUE.append(r['mu_nue']); MNUMU.append(r['mu_numu'])
    MUE.append(r['mu_e']); MUE_B.append(r['mu_e_baryon']); MUNP.append(r['munp']); MUMU.append(r['mu_mu'])

rho = cgs_of_lr(lrs)
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].semilogx(rho, YE, label="Ye"); ax[0,0].semilogx(rho, YMU, label="Ymu")
ax[0,0].set_title("beta-eq composition @ T=20"); ax[0,0].legend(); ax[0,0].set_xlabel("rho [g/cc]")
ax[0,1].semilogx(rho, MNUE, label="mu_nue"); ax[0,1].semilogx(rho, MNUMU, label="mu_numu")
ax[0,1].axhline(0, c='k', lw=.6)
ax[0,1].set_title("fugacity drivers (should be ~0 everywhere!)"); ax[0,1].legend(); ax[0,1].set_ylabel("MeV")
ax[1,0].semilogx(rho, MUE, label="mu_e (electronic table)")
ax[1,0].semilogx(rho, MUE_B, '--', label="mu_e (baryon mu_l-mu_q)")
ax[1,0].set_title("electron chemical potential — baseline check"); ax[1,0].legend(); ax[1,0].set_ylabel("MeV")
ax[1,1].semilogx(rho, MUNP, label="mu_n - mu_p"); ax[1,1].semilogx(rho, MUMU, label="mu_mu")
ax[1,1].set_title("nucleon splitting & muon mu"); ax[1,1].legend(); ax[1,1].set_ylabel("MeV")
for a in ax.flat: a.set_xlabel("rho [g/cc]"); a.grid(alpha=.3)
plt.tight_layout()
out = "/Users/miler/Desktop/hot_tov_betaeq_check.png"
plt.savefig(out, dpi=110)
print(f"\nsaved {out}")
