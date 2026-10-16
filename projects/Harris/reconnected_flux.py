#!/usr/bin/env python3
"""Reconnected flux psi(t) along both sheets of the 2D double-Harris run (cf. paper Fig. 8).

psi is max - min of A(x) = int_0^x By dx' along each sheet centre line. The line is periodic,
so A returns to 0 after one period, and with one island per sheet max - min is the flux between
the X- and O-lines: the paper's psi = int_0^Lx By(x, y=0) dx, with psi(0) = 2*phi0 = 0.2 B0 d_i.
Units: psi in B0*d_i, time in 1/w_ci; B0 is taken from the lobe Bx at the first output.

Usage: python3 reconnected_flux.py [glob] [reference.csv]
  glob defaults to bulk.*.vlsv; reference.csv holds (t*w_ci, psi) pairs to overlay, e.g.
  points digitised from Fig. 8. Writes psi.csv and psi.png.
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
dy = (ymax - ymin) / ny
Lh = 0.5 * (ymax - ymin)
ymid = 0.5 * (ymin + ymax)
sheets = [ymid - 0.5 * Lh, ymid + 0.5 * Lh]

def bvol(f):
    order = np.argsort(f.read_variable("CellID"))  # file order is per-rank
    return f.read_variable("vg_b_vol")[order].reshape(ny, nx, 3)   # CellID-1 = i + nx*j

def row_pair(y):
    # the two cell rows straddling y; sheet centres and the midplane sit on cell faces
    j0 = int(np.floor((y - ymin) / dy - 0.5)) % ny
    return j0, (j0 + 1) % ny

def line_By(B, y):
    j0, j1 = row_pair(y)
    return 0.5 * (B[j0, :, 1] + B[j1, :, 1])

B = bvol(f0)
j0, j1 = row_pair(ymid)                       # lobe midway between the sheets
B0 = abs(0.5 * (B[j0, :, 0] + B[j1, :, 0]).mean())   # x-mean removes the perturbation's dBx
wci = QE * B0 / MP
for k, y in enumerate(sheets):
    b = line_By(B, y)
    nulls = int(np.sum(np.sign(b) != np.sign(np.roll(b, -1))))
    print(f"sheet {k+1}: {nulls} By sign changes along the sheet at t=0 "
          f"(2 = one X- and one O-line; 4 = the old two-island perturbation)")
print(f"B0 = {B0:.6e} T, 1/w_ci = {1/wci:.6e} s, d_i = {DI:.6e} m")

rows = []
for fn in files:
    f = pt.vlsvfile.VlsvReader(fn)
    B = bvol(f)
    psi = []
    for y in sheets:
        A = np.concatenate(([0.0], np.cumsum(line_By(B, y)) * dx))
        psi.append((A.max() - A.min()) / (B0 * DI))
    rows.append((f.read_parameter("time") * wci, *psi))
rows = np.array(rows)
print(f"psi(0) = {rows[0,1]:.4f}, {rows[0,2]:.4f} B0 d_i (expect 2*phi0 = 0.2)")
np.savetxt("psi.csv", rows, delimiter=",", header="t*w_ci,psi_sheet1,psi_sheet2 [B0 d_i]", comments="")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rows[:, 0], rows[:, 1], label="sheet 1")
ax.plot(rows[:, 0], rows[:, 2], "--", label="sheet 2")
if ref is not None:
    ax.plot(ref[:, 0], ref[:, 1], "ko", ms=3, label="reference")
ax.set_xlabel(r"$t\,\omega_{ci}$")
ax.set_ylabel(r"$\psi\ [B_0 d_i]$")
ax.legend()
fig.tight_layout()
fig.savefig("psi.png", dpi=150)
print(f"wrote psi.csv and psi.png ({len(rows)} outputs, t*w_ci up to {rows[-1,0]:.2f})")
