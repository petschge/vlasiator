#!/usr/bin/env python3
"""Lobe-stripe diagnostic for the double-Harris runs (2D output; 1D output works too).

Fine structure along y is separated from the large-scale fields with a smooth spectral high-pass
along y (wavelengths below --lcut), and examined in the lobes of one sheet, 1.5 d_i <= |y - y_sheet|
<= 6.4 d_i. The geometry comes from harris_geometry.py; between conducting walls the interior
rows are evenly reflected at the walls before the FFTs. Per output it reports
  - lobe rms of the fine structure in u_ze, u_ye, u_zi, E_y, E_z, B_z, B_x, n_e and n_i - n_e
  - the dominant wavelength along y, in d_i and in cells, from x-averaged lobe spectra of dE_y
  - an E x B test: correlation and slope of du_ze (and du_zi) against the high-passed E x B drift
  - a Gauss test: correlation and slope of e(n_i - n_e)/eps0 against div E, both high-passed
and writes
  stripes.csv           per-output table
  stripes_spacetime.png y-t diagrams of du_ze and dE_y at one x, in the sheet's frame
  stripes_kw.png        k_y-omega power of dE_y in both lobes, with +k pointing away from the sheet
  stripes_amp.png       lobe rms against time, and du_ze rms against x at three times
How to read it: electrons E x B drifting with unmagnetized ions (slope ~1 for electrons, ~0 for ions)
plus a good Gauss correlation and small dB means an electrostatic oscillation perpendicular to B;
a k-omega ridge near the lower-hybrid frequency or ion-cyclotron harmonics points to a physical mode,
power piled up near the grid Nyquist wavenumber points to numerics. vg_e_vol holds the field the
Vlasov solver used, before the Gauss correction, so the Gauss test is approximate.

Usage: python3 stripes.py [--bulk "bulk.*.vlsv"] [--sheet 1|2] [--x 10] [--lcut 0.7]
                          [--every 1] [--tmax T]
"""
import argparse
import glob
import sys
import numpy as np
try:
    import analysator as pt
except ImportError:  # older installs
    import pytools as pt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harris_geometry import Geometry

LAM0 = 1.138553794e5            # Harris.Scale_size
DI = 2.0 * LAM0                 # lambda0 = d_i/2 in this setup
QE, MP, KB = 1.60217653e-19, 1.67262158e-27, 1.3806503e-23   # Vlasiator's values
ME = 0.04 * MP
EPS0 = 8.85418782e-12
LOBE = (1.5, 6.4)               # |y - y_sheet| range in d_i

ap = argparse.ArgumentParser()
ap.add_argument("--bulk", default="bulk.*.vlsv")
ap.add_argument("--sheet", type=int, default=1, help="which sheet (1 for the single sheet)")
ap.add_argument("--x", type=float, default=10.0, help="x [d_i] of the space-time diagrams")
ap.add_argument("--lcut", type=float, default=0.7, help="high-pass cutoff wavelength [d_i]")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--tmax", type=float, default=np.inf, help="last time [1/w_ci]")
args = ap.parse_args()

files = sorted(glob.glob(args.bulk))[::args.every]
if not files:
    sys.exit("no files")

# ----- grid, geometry and units from the first file -----
f0 = pt.vlsvfile.VlsvReader(files[0])
nx, ny, _ = (int(v) for v in f0.get_spatial_mesh_size())
xmin, ymin, _, xmax, ymax, _ = f0.get_spatial_mesh_extent()
dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
xc = xmin + (np.arange(nx) + 0.5) * dx
yc = ymin + (np.arange(ny) + 0.5) * dy
geo = Geometry(f0, nx, ny)
print(geo.describe())
if not 1 <= args.sheet <= len(geo.sheets):
    sys.exit(f"--sheet must be between 1 and {len(geo.sheets)}")
ysheet, flip = geo.sheets[args.sheet - 1]
flip = float(flip)                               # rotation by pi about x maps sheet 2 onto sheet 1
in_half = np.zeros(ny, bool)
in_half[geo.rows[args.sheet - 1]] = True

def reader(fn):
    f = pt.vlsvfile.VlsvReader(fn)
    order = np.argsort(f.read_variable("CellID"))  # file order is per-rank
    def get(name):
        a = f.read_variable(name)[order]
        return a.reshape((ny, nx) + a.shape[1:])    # CellID-1 = i + nx*j
    return f, get

off = yc - ysheet
dist = np.abs(off)
jmid = geo.lobe_rows[0]
_, g0 = reader(files[0])
B0 = abs(g0("vg_b_vol")[jmid, :, 0].mean())
wci = QE * B0 / MP
vA = DI * wci
nb = g0("electron/vg_rho")[jmid].mean()
vte = np.sqrt(g0("electron/vg_ptensor_diagonal")[jmid].mean(axis=0).mean() / nb / ME)
vti = np.sqrt(g0("proton/vg_ptensor_diagonal")[jmid].mean(axis=0).mean()
              / g0("proton/vg_rho")[jmid].mean() / MP)
wce = (MP / ME) * wci
wpe = np.sqrt(nb * QE**2 / (EPS0 * ME))
wlh = np.sqrt(wci * wce) / np.sqrt(1.0 + (wce / wpe)**2)
rho_e, rho_i = vte / wce, vti / wci
print(f"B0={B0:.4e} T, 1/w_ci={1/wci:.4e} s, v_A={vA:.4e} m/s, lobe n={nb:.4e} m^-3")
print(f"dy={dy/DI:.3f} d_i; rho_e={rho_e/DI:.3f} d_i (2*pi*rho_e={2*np.pi*rho_e/DI:.3f}), "
      f"rho_i={rho_i/DI:.3f} d_i, d_e={DI*np.sqrt(ME/MP)/DI:.3f} d_i")
print(f"w_lh={wlh/wci:.2f} w_ci, w_ce={wce/wci:.1f} w_ci, w_pe={wpe/wci:.1f} w_ci "
      f"(output cadence resolves w up to {np.pi/((pt.vlsvfile.VlsvReader(files[1]).read_parameter('time') - f0.read_parameter('time'))*wci) if len(files) > 1 else 0:.1f} w_ci)")

lobe = in_half & (dist >= LOBE[0] * DI) & (dist <= LOBE[1] * DI)
side = np.where(off * flip > 0, 1, -1)                       # +1: lobe above the sheet (sheet frame)
seg_up = np.where(lobe & (side > 0))[0]
seg_dn = np.where(lobe & (side < 0))[0]
seg_up = seg_up[np.argsort(dist[seg_up])]                    # ordered away from the sheet
seg_dn = seg_dn[np.argsort(dist[seg_dn])]
nseg = min(len(seg_up), len(seg_dn))
seg_up, seg_dn = seg_up[:nseg], seg_dn[:nseg]
half, ypap = geo.rows[args.sheet - 1], geo.ypap[args.sheet - 1]
ix = int(np.argmin(np.abs(xc - args.x * DI)))

# Spectral operations along y: the whole periodic column, or the interior rows evenly reflected
# at the walls (no jump at the wrap); rows outside the interior are returned as 0.
Jin = np.arange(geo.j_lo, geo.j_hi + 1)
nspec = ny if not geo.walls else 2 * len(Jin)
ky = 2 * np.pi * np.fft.fftfreq(nspec, dy)
kc = 2 * np.pi / (args.lcut * DI)
with np.errstate(divide="ignore"):
    H = 1.0 / (1.0 + (kc / np.abs(ky))**8)                   # smooth, so the sheets ring little
H[0] = 0.0

def yspec(a, mult):
    if not geo.walls:
        return np.real(np.fft.ifft(np.fft.fft(a, axis=0) * mult[:, None], axis=0))
    ai = a[Jin]
    ext = np.concatenate([ai, ai[::-1]], axis=0)
    out = np.zeros_like(a, dtype=float)
    out[Jin] = np.real(np.fft.ifft(np.fft.fft(ext, axis=0) * mult[:, None], axis=0))[:len(Jin)]
    return out

def hp(a):
    return yspec(a, H)

def corr_slope(a, b):
    a, b = a[lobe].ravel(), b[lobe].ravel()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / den if den > 0 else 0.0, (a * b).sum() / (b * b).sum() if (b * b).sum() > 0 else 0.0

QUANT = [("u_ze", vA), ("u_ye", vA), ("u_zi", vA), ("E_y", vA * B0), ("E_z", vA * B0),
         ("B_z", B0), ("B_x", B0), ("n_e", nb), ("n_i-n_e", nb)]
rows_out, st_u, st_E, kw_up, kw_dn, xprof, times = [], [], [], [], [], [], []
hann_y = np.hanning(nseg)[:, None]
for fn in files:
    f, get = reader(fn)
    t = f.read_parameter("time") * wci
    if t > args.tmax:
        break
    E, B = get("vg_e_vol"), get("vg_b_vol")
    ue, ui = get("electron/vg_v"), get("proton/vg_v")
    ne, ni = get("electron/vg_rho"), get("proton/vg_rho")
    b2 = (B**2).sum(axis=-1)
    uEz = (E[..., 0] * B[..., 1] - E[..., 1] * B[..., 0]) / b2
    # spectral d/dy: a centred difference would understate short waves by sin(k dy)/(k dy)
    divE = ((np.roll(E[..., 0], -1, 1) - np.roll(E[..., 0], 1, 1)) / (2 * dx)
            + yspec(E[..., 1], 1j * ky))
    fields = {"u_ze": flip * ue[..., 2], "u_ye": flip * ue[..., 1], "u_zi": flip * ui[..., 2],
              "E_y": flip * E[..., 1], "E_z": flip * E[..., 2], "B_z": flip * B[..., 2],
              "B_x": B[..., 0], "n_e": ne, "n_i-n_e": ni - ne}
    d = {k: hp(v) for k, v in fields.items()}
    rms = {k: np.sqrt((d[k][lobe]**2).mean()) / s for k, s in QUANT}
    ce, se = corr_slope(d["u_ze"], hp(flip * uEz))
    cI, sI = corr_slope(d["u_zi"], hp(flip * uEz))
    cg, sg = corr_slope(hp(QE * (ni - ne) / EPS0), hp(divE))
    seg = np.concatenate([d["E_y"][seg_up], d["E_y"][seg_dn]], axis=1)       # (nseg, 2*nx)
    P = (np.abs(np.fft.rfft((seg - seg.mean(axis=0)) * hann_y, axis=0))**2).mean(axis=1)
    kseg = 2 * np.pi * np.fft.rfftfreq(nseg, dy)
    kpk = kseg[1:][np.argmax(P[1:])] if nseg > 2 else np.nan
    lam = 2 * np.pi / kpk if kpk > 0 else np.nan
    rows_out.append([t, *[rms[k] for k, _ in QUANT], lam / DI, lam / dy, ce, se, cI, sI, cg, sg])
    st_u.append(d["u_ze"][half, ix] / vA)
    st_E.append(d["E_y"][half, ix] / (vA * B0))
    kw_up.append(d["E_y"][seg_up])
    kw_dn.append(d["E_y"][seg_dn])
    xprof.append(np.sqrt((d["u_ze"][lobe]**2).mean(axis=0)) / vA)
    times.append(t)

R = np.array(rows_out)
times = np.array(times)
names = [k for k, _ in QUANT]
hdr = ("t*w_ci," + ",".join(f"rms_{k}" for k in names) +
       ",lambda_d_i,lambda_cells,corr_ExB_e,slope_ExB_e,corr_ExB_i,slope_ExB_i,corr_Gauss,slope_Gauss")
np.savetxt("stripes.csv", R, delimiter=",", header=hdr, comments="")
print("\nlobe rms of the fine structure (u in v_A, E in v_A B0, B in B0, n in lobe n), "
      "dominant wavelength, E x B and Gauss tests:")
print("  t    " + " ".join(f"{k:>8s}" for k in names) + "   lam[d_i] cells  ExB_e(c,s)   ExB_i(c,s)   Gauss(c,s)")
for r in R[::max(1, len(R) // 20)]:
    print(f"{r[0]:5.1f} " + " ".join(f"{v:8.1e}" for v in r[1:10]) +
          f"   {r[10]:6.3f} {r[11]:5.1f}  {r[12]:+.2f},{r[13]:+.2f}  {r[14]:+.2f},{r[15]:+.2f}  {r[16]:+.2f},{r[17]:+.2f}")

# ----- space-time diagrams -----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, data, lab in ((axs[0], np.array(st_u), r"$\delta u_{ze}/v_A$"),
                      (axs[1], np.array(st_E), r"$\delta E_y/(v_A B_0)$")):
    data = np.where(np.abs(ypap)[None, :] < LOBE[0] * DI, np.nan, data)
    m = np.nanpercentile(np.abs(data), 99) or 1.0
    im = ax.pcolormesh(times, ypap / DI, data.T, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
    ax.set_xlabel(r"$t\,\omega_{ci}$")
    ax.set_title(f"{lab} at x={xc[ix]/DI:.1f} $d_i$ (sheet {args.sheet} frame)")
    plt.colorbar(im, ax=ax)
axs[0].set_ylabel(r"$y\ [d_i]$")
fig.tight_layout()
fig.savefig("stripes_spacetime.png", dpi=150)

# ----- k-omega power of dE_y, both lobes, +k away from the sheet -----
if len(times) > 4 and nseg > 4:
    dtw = np.mean(np.diff(times))
    win = np.hanning(len(times))[:, None, None] * np.hanning(nseg)[None, :, None]
    Pkw = 0
    for A in (np.array(kw_up), np.array(kw_dn)):                 # (t, distance, x)
        A = A - A.mean(axis=0)
        Pkw = Pkw + (np.abs(np.fft.fft2(A * win, axes=(0, 1)))**2).mean(axis=2)
    om = 2 * np.pi * np.fft.fftfreq(len(times), dtw)              # in w_ci (t is in 1/w_ci)
    kk = 2 * np.pi * np.fft.fftfreq(nseg, dy / DI)                # in 1/d_i
    # fft sign convention exp(-i(w t + k y)) for positive bins: flip k so +k means outward for w>0
    Pkw = np.fft.fftshift(Pkw)
    om, kk = np.fft.fftshift(om), -np.fft.fftshift(kk)
    pos = om >= 0
    fig, ax = plt.subplots(figsize=(7, 5))
    LP = np.log10(Pkw[pos] + 1e-300)
    im = ax.pcolormesh(kk, om[pos], LP, shading="auto", cmap="viridis", vmin=LP.max() - 6, vmax=LP.max())
    for n in range(1, int(om.max()) + 1):
        ax.axhline(n, color="w", lw=0.3, alpha=0.25)
    ax.axhline(wlh / wci, color="r", ls="--", lw=1, label=r"$\omega_{lh}$")
    for s in (-1, 1):
        ax.axvline(s * DI / rho_e, color="orange", ls=":", lw=1, label=r"$k\rho_e=1$" if s > 0 else None)
    ax.set_xlabel(r"$k_y d_i$ (+: away from the sheet)")
    ax.set_ylabel(r"$\omega/\omega_{ci}$")
    ax.set_title(r"$\log_{10}$ power of $\delta E_y$ in the lobes (thin lines: $n\,\omega_{ci}$)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("stripes_kw.png", dpi=150)
    iw, ik = np.unravel_index(np.argmax(Pkw[pos]), Pkw[pos].shape)
    print(f"\nk-omega peak of dE_y: k_y d_i = {kk[ik]:+.2f} (wavelength {2*np.pi/abs(kk[ik]) if kk[ik] else np.inf:.3f} d_i),"
          f" omega = {om[pos][iw]:.2f} w_ci")

# ----- amplitudes -----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for k in ("u_ze", "E_y", "n_i-n_e", "B_z", "u_zi"):
    axs[0].semilogy(times, R[:, 1 + names.index(k)], label=k)
axs[0].set_xlabel(r"$t\,\omega_{ci}$")
axs[0].set_ylabel("lobe rms of fine structure (normalised)")
axs[0].legend(fontsize=8)
X = np.array(xprof)
for k in sorted({0, len(times) // 4, len(times) // 2}):
    axs[1].plot(xc / DI, X[k], label=rf"$t\omega_{{ci}}$={times[k]:.1f}")
axs[1].set_xlabel(r"$x\ [d_i]$")
axs[1].set_ylabel(r"lobe rms $\delta u_{ze}/v_A$")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("stripes_amp.png", dpi=150)
print("wrote stripes.csv, stripes_spacetime.png, stripes_kw.png, stripes_amp.png")
