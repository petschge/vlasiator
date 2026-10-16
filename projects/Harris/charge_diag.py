#!/usr/bin/env python3
"""Per-step charge-imbalance diagnostic for the x-invariant double-Harris test.

Usage: python3 charge_diag.py [glob]   (default glob: bulk.*.vlsv, first file is the reference)
"""
import glob
import sys
import numpy as np
try:
    import analysator as pt
except ImportError:  # older installs
    import pytools as pt

LAM0 = 1.138553794e5   # Harris.Scale_size
QE = 1.60217653e-19    # Vlasiator's value
EPS0 = 8.85418782e-12
KB = 1.3806503e-23

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "bulk.*.vlsv"))
if not files:
    sys.exit("no files")

def open_grid(fn):
    f = pt.vlsvfile.VlsvReader(fn)
    nx, ny, nz = f.get_spatial_mesh_size()
    order = np.argsort(f.read_variable("CellID"))  # file order is per-rank, not CellID order
    return f, int(nx), int(ny), order

def vg(f, nx, ny, order, name):
    a = f.read_variable(name)[order]
    return a.reshape((ny, nx) + a.shape[1:])        # CellID-1 = i + nx*j

f0, nx, ny, order0 = open_grid(files[0])
xmin, ymin, _, xmax, ymax, _ = f0.get_spatial_mesh_extent()
dy = (ymax - ymin) / ny
yc = ymin + (np.arange(ny) + 0.5) * dy
Lh = 0.5 * (ymax - ymin)
ymid = 0.5 * (ymin + ymax)
sheets = np.array([ymid - 0.5 * Lh, ymid + 0.5 * Lh])
# signed offset from the nearest sheet, in lambda0
off = (yc[:, None] - sheets[None, :])
near = np.argmin(np.abs(off), axis=1)
off_lam = off[np.arange(ny), near] / LAM0

ne0 = vg(f0, nx, ny, order0, "electron/vg_rho").mean(axis=1)
ni0 = vg(f0, nx, ny, order0, "proton/vg_rho").mean(axis=1)
nref = ni0.max()
# sheet-1 core cell on the +y side: where the step-1 current correction is largest
cand = np.where((near == 0) & (off_lam > 0))[0]
jc = cand[np.argmin(off_lam[cand])]
jl = np.argmax(np.abs(off_lam))                      # lobe row: background only, uniform B

def temps(f, order):
    out = []
    for pop in ("electron", "proton"):
        n = vg(f, nx, ny, order, pop + "/vg_rho").mean(axis=1)
        P = vg(f, nx, ny, order, pop + "/vg_ptensor_diagonal").mean(axis=1)
        out.append(P.mean(axis=1) / (n * KB))
    return out
Te0, Ti0 = temps(f0, order0)

s1 = np.where(near == 0)[0]                          # sheet-1 rows
def sheet1_current(f, order):
    ne = vg(f, nx, ny, order, "electron/vg_rho").mean(axis=1)
    ni = vg(f, nx, ny, order, "proton/vg_rho").mean(axis=1)
    ve = vg(f, nx, ny, order, "electron/vg_v").mean(axis=1)[:, 2]
    vi = vg(f, nx, ny, order, "proton/vg_v").mean(axis=1)[:, 2]
    J = np.abs(QE * (ni * vi - ne * ve))[s1]
    return J.max(), J.sum(), J.sum() / J.max()       # peak, integral/dy, equivalent width/dy
Jp0, Ji0, W0 = sheet1_current(f0, order0)

print(f"{len(files)} files, grid {nx}x{ny}, sheets at y={sheets/LAM0} lambda0, n_ref={nref:.3e}")
print(" t[s]        dN_e/N_e   dN_i/N_i   max|ni-ne|/n_ref @y-ysheet[lam0]  Nyquist frac"
      "  max|dne|/n_ref @off   max|dni|/n_ref @off   x-spread   max|Ey|    max|Ez|"
      f"   Vz_e,Vz_i @y-y1={off_lam[jc]:+.2f}"
      f"   core dne,dni (signed)   core Te/Te0,Ti/Ti0   lobe(@{off_lam[jl]:+.1f}) Te/Te0,Ti/Ti0"
      f"   sheet1 Jpeak/J0, intJ/intJ0, width/width0")
for fn in files:
    f, _, _, order = open_grid(fn)
    ne2 = vg(f, nx, ny, order, "electron/vg_rho")
    ni2 = vg(f, nx, ny, order, "proton/vg_rho")
    ne, ni = ne2.mean(axis=1), ni2.mean(axis=1)
    xspread = max(np.ptp(ne2, axis=1).max(), np.ptp(ni2, axis=1).max()) / nref  # should stay ~0
    E = vg(f, nx, ny, order, "vg_e_vol").mean(axis=1)
    vze = vg(f, nx, ny, order, "electron/vg_v").mean(axis=1)[jc, 2]
    vzi = vg(f, nx, ny, order, "proton/vg_v").mean(axis=1)[jc, 2]
    q = ni - ne
    jq = np.argmax(np.abs(q))
    P = np.abs(np.fft.fft(q))**2
    nyq = P[ny // 2] / max(P.sum(), 1e-300)          # 1 = pure odd-even pattern, ~1/ny for a spike
    jde = np.argmax(np.abs(ne - ne0))
    jdi = np.argmax(np.abs(ni - ni0))
    t = f.read_parameter("time")
    Te, Ti = temps(f, order)
    Jp, Ji, W = sheet1_current(f, order)
    print(f"{t:10.4e}  {ne.sum()/ne0.sum()-1:+.2e}  {ni.sum()/ni0.sum()-1:+.2e}"
          f"   {abs(q[jq])/nref:.2e} @{off_lam[jq]:+6.2f}"
          f"   (rho/eps0={QE*q[jq]/EPS0:+.2e})   {nyq:.2f}"
          f"   {abs(ne-ne0)[jde]/nref:.2e} @{off_lam[jde]:+6.2f}"
          f"   {abs(ni-ni0)[jdi]/nref:.2e} @{off_lam[jdi]:+6.2f}"
          f"   {xspread:.1e}   {np.abs(E[:,1]).max():.2e}  {np.abs(E[:,2]).max():.2e}"
          f"   {vze:+.3e} {vzi:+.3e}"
          f"   {(ne-ne0)[jc]/nref:+.2e} {(ni-ni0)[jc]/nref:+.2e}"
          f"   {Te[jc]/Te0[jc]:.4f} {Ti[jc]/Ti0[jc]:.4f}   {Te[jl]/Te0[jl]:.4f} {Ti[jl]/Ti0[jl]:.4f}"
          f"   {Jp/Jp0:.4f} {Ji/Ji0:.4f} {W/W0:.4f}")
