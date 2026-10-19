#!/usr/bin/env python3
"""Reconnected flux psi(t) along each current sheet of the 2D Harris/GEM runs (cf. paper Fig. 8):
both sheets of the periodic double sheet, or the single sheet between conducting walls (the
geometry is found from the first output by harris_geometry.py).

psi is max - min of A(x) = int_0^x By dx' along each sheet centre line. The line is periodic,
so A returns to 0 after one period, and with one island per sheet max - min is the flux between
the X- and O-lines: the paper's psi = int_0^Lx By(x, y=0) dx, with psi(0) = 2*phi0 = 0.2 B0 d_i.
Units: psi in B0*d_i, time in 1/w_ci; B0 is taken from the lobe Bx at the first output.

It also tracks the X- and O-lines along each sheet line: the zero crossings of By, an X-line where
dBy/dx has the sign of the sheet's dBx/dy (a saddle of A), an O-line otherwise. Each O-line's island
flux is |A(O) - A(X)| to the nearer of its two neighbouring X-lines, so a secondary island shows up
as a second X-line and an O-line with a small flux, and one held at the X-line by symmetry stays put.

Usage: python3 reconnected_flux.py [glob] [reference.csv]
  glob defaults to bulk.*.vlsv; reference.csv holds (t*w_ci, psi) pairs to overlay, e.g.
  points digitised from Fig. 8. Writes psi.csv, psi.png and xlines.log.
"""
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

LAM0 = 1.138553794e5          # Harris.Scale_size; d_i = 2*lambda0 in this setup
DI = 2.0 * LAM0
QE = 1.60217653e-19           # Vlasiator's values
MP = 1.67262158e-27

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "bulk.*.vlsv"))
ref = np.loadtxt(sys.argv[2], delimiter=",", ndmin=2) if len(sys.argv) > 2 else None
if not files:
    sys.exit("no files")

f0 = pt.vlsvfile.VlsvReader(files[0])
nx, ny, _ = (int(v) for v in f0.get_spatial_mesh_size())
if nx < 2:
    sys.exit("1D run (nx = 1): no reconnected flux to measure")
xmin, ymin, _, xmax, ymax, _ = f0.get_spatial_mesh_extent()
dx = (xmax - xmin) / nx
geo = Geometry(f0, nx, ny)
print(geo.describe())
sheets = [ys for ys, _ in geo.sheets]

def bvol(f):
    order = np.argsort(f.read_variable("CellID"))  # file order is per-rank
    return f.read_variable("vg_b_vol")[order].reshape(ny, nx, 3)   # CellID-1 = i + nx*j

def line_By(B, y):
    j0, j1 = geo.straddle(y)                  # sheet centres sit on cell faces
    return 0.5 * (B[j0, :, 1] + B[j1, :, 1])

B = bvol(f0)
j0, j1 = geo.lobe_rows                        # midway between double sheets, or next to a wall
B0 = 0.5 * (abs(B[j0, :, 0].mean()) + abs(B[j1, :, 0].mean()))   # x-mean removes the perturbation's dBx
wci = QE * B0 / MP
for k, y in enumerate(sheets):
    b = line_By(B, y)
    nulls = int(np.sum(np.sign(b) != np.sign(np.roll(b, -1))))
    print(f"sheet {k+1}: {nulls} By sign changes along the sheet at t=0 "
          f"(2 = one X- and one O-line; 4 = the old two-island perturbation)")
print(f"B0 = {B0:.6e} T, 1/w_ci = {1/wci:.6e} s, d_i = {DI:.6e} m")

xc = xmin + (np.arange(nx) + 0.5) * dx
Lx = xmax - xmin
ISLAND_MIN = 0.01             # B0 d_i: islands with less flux are reported but not counted

def nulls(B, y, sg):
    """X- and O-lines along the sheet line y: list of (x [m], 'X'|'O', A [T m]) and the island
    flux of each O-line [B0 d_i]."""
    b = line_By(B, y)
    A = np.concatenate(([0.0], np.cumsum(b)[:-1])) * dx + 0.5 * b * dx    # A at cell centres
    out = []
    for i in range(nx):
        j = (i + 1) % nx
        if b[i] == 0.0 or np.sign(b[i]) == np.sign(b[j]):
            continue
        frac = b[i] / (b[i] - b[j])
        x0 = xc[i] + frac * dx
        x0 = x0 - Lx if x0 > xmax else x0
        a0 = A[i] + 0.5 * frac * (b[i]) * dx                              # A grows by ~b*dx/2 to the null
        kind = "X" if np.sign(b[j] - b[i]) == sg else "O"
        out.append((x0, kind, a0))
    out.sort()
    flux = []
    xs = [k for k, n in enumerate(out) if n[1] == "X"]
    for k, n in enumerate(out):
        if n[1] != "O" or not xs:
            continue
        left = max([q for q in xs if q < k], default=xs[-1])
        right = min([q for q in xs if q > k], default=xs[0])
        flux.append(min(abs(n[2] - out[left][2]), abs(n[2] - out[right][2])) / (B0 * DI))
    return out, flux

xlog = open("xlines.log", "w")
xlog.write("# per output and sheet: t*w_ci, sheet, number of X-lines, number of islands with flux >= "
           f"{ISLAND_MIN} B0 d_i; then the nulls along x [d_i] as X or O(island flux)\n")
history = {k: [] for k in range(len(sheets))}
rows = []
for fn in files:
    f = pt.vlsvfile.VlsvReader(fn)
    B = bvol(f)
    psi = []
    for y in sheets:
        A = np.concatenate(([0.0], np.cumsum(line_By(B, y)) * dx))
        psi.append((A.max() - A.min()) / (B0 * DI))
    t = f.read_parameter("time") * wci
    rows.append((t, *psi))
    for k, (y, sg) in enumerate(geo.sheets):
        out, flux = nulls(B, y, sg)
        nX = sum(1 for n in out if n[1] == "X")
        nIsl = sum(1 for v in flux if v >= ISLAND_MIN)
        it = iter(flux)
        desc = " ".join(f"X{n[0]/DI:+.2f}" if n[1] == "X" else f"O{n[0]/DI:+.2f}({next(it):.3f})" for n in out)
        xlog.write(f"{t:7.2f} {k+1} {nX} {nIsl}  {desc}\n")
        history[k].append((t, nX, nIsl, [n[0] / DI for n in out if n[1] == "X"], max(flux, default=0.0),
                           sorted(flux)[-2] if len(flux) > 1 else 0.0))
xlog.close()
rows = np.array(rows)
print("psi(0) = " + ", ".join(f"{v:.4f}" for v in rows[0, 1:]) + " B0 d_i (expect 2*phi0 = 0.2)")
np.savetxt("psi.csv", rows, delimiter=",", comments="",
           header="t*w_ci," + ",".join(f"psi_sheet{k+1}" for k in range(len(sheets))) + " [B0 d_i]")

fig, ax = plt.subplots(figsize=(6, 4))
for k in range(len(sheets)):
    ax.plot(rows[:, 0], rows[:, 1 + k], "-" if k == 0 else "--", label=f"sheet {k+1}")
if ref is not None:
    ax.plot(ref[:, 0], ref[:, 1], "ko", ms=3, label="reference")
ax.set_xlabel(r"$t\,\omega_{ci}$")
ax.set_ylabel(r"$\psi\ [B_0 d_i]$")
ax.legend()
fig.tight_layout()
fig.savefig("psi.png", dpi=150)
print(f"wrote psi.csv and psi.png ({len(rows)} outputs, t*w_ci up to {rows[-1,0]:.2f})")
for k, h in history.items():
    print(f"sheet {k+1}: X-lines and islands (flux >= {ISLAND_MIN} B0 d_i) over time, written in full to xlines.log:")
    last = None
    for t, nX, nIsl, xX, fmax, f2 in h:
        if (nX, nIsl) != last:
            print(f"  from t = {t:6.2f}: {nX} X-line(s) at x = {', '.join(f'{v:+.2f}' for v in xX)} d_i; "
                  f"{nIsl} island(s); largest / second island flux {fmax:.3f} / {f2:.3f} B0 d_i")
            last = (nX, nIsl)
print("wrote xlines.log")
