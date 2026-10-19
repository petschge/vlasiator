#!/usr/bin/env python3
"""Figures in the style of Liu et al. 2025 (JCP 528:113840), Figs. 9-11, for the double-Harris run,
plus a map of electron non-Maxwellianity.

  fig9.png : out-of-plane B_z [B0] and electron velocity u_ze [v_A] at one time
  fig10.png: field lines (contours of A_z) over the out-of-plane current J_z [e n0 v_A], two times
  fig11.png: isosurfaces of the electron VDF at the X-line, two times, in the paper's velocity box
             (vx, vy in [-6.5, 6.5] v_te, vz in [-5, 8] v_te)
  fig12.png: electron non-Maxwellianity with field lines, at the fig10 times (or --t12)
  fig13.png: as fig11, in the cell with the largest electron non-Maxwellianity among the cells with
             a stored VDF on the chosen sheet's half of the domain, at the fig11 times (or --t13);
             the velocity box is fitted to the distribution, since it need not sit in fig11's box

fig12 and fig13 need electron/vg_nonmaxwellianess in the bulk files (relative-entropy DRO patch; cfg line
"output = populations_vg_nonmaxwellianess"): sigma/n = (1/n) int f ln(f/f_M) d^3v, where f_M has
the same n, bulk velocity and diagonal temperatures T_xx, T_yy, T_zz in the simulation axes. It is
the Kullback-Leibler divergence of f from f_M, i.e. (s_M - s)/(n k_B) with s the kinetic entropy
density: 0 for that tri-Maxwellian and positive otherwise. Temperature anisotropy along x, y, z is
absorbed by f_M; anisotropy along B at an angle to the axes, agyrotropy, beams and non-Gaussian
tails are not. Small negative values come from velocity discretisation and sparsity truncation.

One sheet is shown in the paper's frame: x in [-Lx, Lx], y in [-6.4, 6.4] d_i around the sheet.
The geometry (periodic double sheet, or one sheet between conducting walls) is found from the
first output by harris_geometry.py; only interior rows are plotted. A sheet of the opposite
polarity (the double sheet's sheet 2) is mapped onto the paper's orientation through the setup's
symmetry (rotation by pi about x: y -> -y, (V_x, V_y, V_z) -> (V_x, -V_y, -V_z)).

Usage: python3 gem_figures.py [--bulk "bulk.*.vlsv"] [--vdf "distributions.*.vlsv"] [--sheet 1|2]
                              [--t9 16] [--t10 16 20] [--t11 16 20] [--t12 T1 T2] [--t13 T1 T2]
                              [--xline X]
                              [--nlines 16]
                              [--levels 0.05 0.2 0.5]
Times are in 1/w_ci; the nearest available output is used and reported.
Units come from the first bulk file: B0 from the lobe Bx, w_ci = e B0/m_p, v_A = d_i w_ci with
d_i = 2*lambda0, and v_te from the lobe electron temperature.
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
MU0 = 4e-7 * np.pi

ap = argparse.ArgumentParser()
ap.add_argument("--bulk", default="bulk.*.vlsv")
ap.add_argument("--vdf", default="distributions.*.vlsv")
ap.add_argument("--sheet", type=int, default=1, help="which sheet (1 for the single sheet)")
ap.add_argument("--t9", type=float, default=16.0)
ap.add_argument("--t10", type=float, nargs=2, default=(16.0, 20.0))
ap.add_argument("--t11", type=float, nargs=2, default=(16.0, 20.0))
ap.add_argument("--t12", type=float, nargs=2, default=None, help="fig12 times; default: --t10")
ap.add_argument("--t13", type=float, nargs=2, default=None, help="fig13 times; default: --t11")
ap.add_argument("--xline", type=float, default=None,
                help="x [d_i] of the X-line for fig11; default: located from B at each time")
ap.add_argument("--nlines", type=int, default=16, help="field-line spacing: (max-min of A)/nlines")
ap.add_argument("--levels", type=float, nargs="+", default=(0.05, 0.2, 0.5),
                help="isosurface levels as fractions of the VDF maximum")
args = ap.parse_args()

bulk = sorted(glob.glob(args.bulk))
if not bulk:
    sys.exit("no bulk files")

# ----- grid and units from the first bulk file -----
f0 = pt.vlsvfile.VlsvReader(bulk[0])
nx, ny, _ = (int(v) for v in f0.get_spatial_mesh_size())
if nx < 2:
    sys.exit("1D run: nothing to plot")
xmin, ymin, _, xmax, ymax, _ = f0.get_spatial_mesh_extent()
dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
geo = Geometry(f0, nx, ny)
print(geo.describe())
if not 1 <= args.sheet <= len(geo.sheets):
    sys.exit(f"--sheet must be between 1 and {len(geo.sheets)}")
ysheet, flip = geo.sheets[args.sheet - 1]
flip = float(flip)                          # sign applied to y and to the y, z components

def grid(f, name):
    order = np.argsort(f.read_variable("CellID"))       # file order is per-rank
    a = f.read_variable(name)[order]
    return a.reshape((ny, nx) + a.shape[1:])            # CellID-1 = i + nx*j

yc = ymin + (np.arange(ny) + 0.5) * dy
xc = xmin + (np.arange(nx) + 0.5) * dx
jmid = geo.lobe_rows[0]                                 # a lobe row (between sheets / next to a wall)
B = grid(f0, "vg_b_vol")
B0 = abs(B[jmid, :, 0].mean())                          # x-mean removes the perturbation's dBx
wci = QE * B0 / MP
vA = DI * wci
J0 = B0 / (MU0 * DI)                                    # = e n0 v_A for this setup
ne_l = grid(f0, "electron/vg_rho")[jmid].mean()
Pe_l = grid(f0, "electron/vg_ptensor_diagonal")[jmid].mean(axis=0).mean()
vte = np.sqrt(Pe_l / ne_l / ME)
print(f"B0={B0:.4e} T, 1/w_ci={1/wci:.4e} s, v_A={vA:.4e} m/s, v_te={vte:.4e} m/s "
      f"(v_te/v_A={vte/vA:.3f}; GEM: {np.sqrt(25/12):.3f})")

# interior rows of the chosen sheet's half-domain, mapped to the paper's y in [-Ly/2, Ly/2]
rows, ypap = geo.rows[args.sheet - 1], geo.ypap[args.sheet - 1]
in_half = np.zeros(ny, bool)
in_half[rows] = True

_times = {}
def time_of(fn):
    if fn not in _times:
        _times[fn] = pt.vlsvfile.VlsvReader(fn).read_parameter("time") * wci
    return _times[fn]

def nearest(files, t):
    fn = min(files, key=lambda f: abs(time_of(f) - t))
    print(f"  requested t={t:g}/w_ci -> {fn} (t={time_of(fn):.2f}/w_ci)")
    return fn, time_of(fn)

def flux_function(Bx, By):
    # Bx = dA/dy, By = -dA/dx by line integration over the interior rows (no periodicity in y needed)
    return geo.flux_function(Bx, By)

jc = np.argsort(np.abs(yc - ysheet))[:2]            # the two rows straddling the sheet centre
jl = int(np.argmin(np.abs(yc - (ysheet + 3.0 * DI))))  # a lobe row

def xline(A):
    """Dominant X-line of the chosen sheet from the full-domain flux function A.
    With A oriented so the lobes are positive, X-lines are maxima of A along the sheet and O-lines
    minima; the dominant X-line, whose separatrix encloses every island, is the largest maximum.
    Returns its x (parabolic sub-cell fit), the orientation sign and the oriented A there."""
    a = A[jc].mean(axis=0)
    s = np.sign(A[jl].mean() - a.mean()) or 1.0
    a = s * a
    k = int(np.argmax(a))
    am, ap_ = a[(k - 1) % nx], a[(k + 1) % nx]
    den = am - 2 * a[k] + ap_
    off = 0.5 * (am - ap_) / den if den != 0 else 0.0
    x = xmin + (xc[k] + off * dx - xmin) % (xmax - xmin)
    return x, s, a[k] - 0.25 * (am - ap_) * off

def field_lines(ax, B, color="k"):
    """Contours of A_z anchored at the dominant X-line (its separatrix is the 0 level, lobes
    positive, so both sheets get the same levels), and a marker on the X-line. Returns its x."""
    Afull = flux_function(B[..., 0], B[..., 1])
    xX, sA, AX = xline(Afull)
    A = (sA * Afull - AX)[rows]
    dA = (A.max() - A.min()) / args.nlines
    lev = dA * np.arange(np.floor(A.min() / dA), np.ceil(A.max() / dA) + 1)
    ax.contour(xc / DI, ypap / DI, A, levels=lev, colors=color, linewidths=0.7, linestyles="solid")
    # ax.plot(xX / DI, 0.0, "c+", ms=9, mew=1.5)
    return xX

extent = [xc[0] / DI, xc[-1] / DI, ypap[0] / DI, ypap[-1] / DI]

def panel(ax, data, title, cmap, symmetric=False, vmax=None):
    vmin = None
    if symmetric:
        m = np.abs(data).max()
        vmin, vmax = -m, m
    im = ax.imshow(data, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$x\ [d_i]$")
    ax.set_ylabel(r"$y\ [d_i]$")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

# ----- Fig. 9 -----
print("Fig. 9:")
fn, tt = nearest(bulk, args.t9)
f = pt.vlsvfile.VlsvReader(fn)
B = grid(f, "vg_b_vol")
ue = grid(f, "electron/vg_v")
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
panel(axs[0], flip * B[rows, :, 2] / B0, rf"(a) $B_z/B_0$, $t\omega_{{ci}}$={tt:.1f}", "RdBu_r", symmetric=True)
panel(axs[1], flip * ue[rows, :, 2] / vA, rf"(b) $u_{{ze}}/v_A$, $t\omega_{{ci}}$={tt:.1f}", "turbo")
fig.tight_layout()
fig.savefig("fig9.png", dpi=150)

# ----- Fig. 10 -----
print("Fig. 10:")
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, t, lab in zip(axs, args.t10, "ab"):
    fn, tt = nearest(bulk, t)
    f = pt.vlsvfile.VlsvReader(fn)
    B = grid(f, "vg_b_vol")
    ni, ne = grid(f, "proton/vg_rho"), grid(f, "electron/vg_rho")
    vi, ve = grid(f, "proton/vg_v"), grid(f, "electron/vg_v")
    Jz = QE * (ni * vi[..., 2] - ne * ve[..., 2])
    jz = flip * Jz[rows] / J0
    panel(ax, jz, rf"({lab}) $J_z/(e n_0 v_A)$, $t\omega_{{ci}}$={tt:.1f}", "hot", vmax=0.0)
    xX = field_lines(ax, B)
    print(f"    dominant X-line at x={xX/DI:+.2f} d_i")
fig.tight_layout()
fig.savefig("fig10.png", dpi=150)

# ----- Fig. 12: electron non-Maxwellianity -----
NM = "electron/vg_nonmaxwellianess"
print("Fig. 12:")
from matplotlib.colors import LogNorm
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
made12 = True
for ax, t, lab in zip(axs, args.t12 if args.t12 is not None else args.t10, "ab"):
    fn, tt = nearest(bulk, t)
    f = pt.vlsvfile.VlsvReader(fn)
    if not f.check_variable(NM):
        print(f"    {NM} not in {fn}; skipping fig12 (needs output = populations_vg_nonmaxwellianess)")
        made12 = False
        break
    M = grid(f, NM)
    Mh = M[rows]
    vmax = Mh.max()
    floor = max(vmax * 1e-3, 1e-12)                   # three decades; log scale needs M > 0
    im = ax.imshow(np.maximum(Mh, floor), origin="lower", extent=extent, aspect="equal", cmap="magma",
                   norm=LogNorm(vmin=floor, vmax=max(vmax, 10 * floor)))
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_xlabel(r"$x\ [d_i]$")
    ax.set_ylabel(r"$y\ [d_i]$")
    ax.set_title(rf"({lab}) electron $\sigma_{{rel}}/n$, $t\omega_{{ci}}$={tt:.1f}")
    xX = field_lines(ax, grid(f, "vg_b_vol"), color="0.8")   # light: magma is dark at the low end
    j, i = np.unravel_index(np.argmax(Mh), Mh.shape)
    ix = int(np.argmin(np.abs(xc - xX)))
    lobe = np.abs(ypap) > 3.0 * DI
    print(f"    max {vmax:.3e} at x-x_X={(xc[i]-xX)/DI:+.2f} d_i, y={ypap[j]/DI:+.2f} d_i; "
          f"at the X-line {M[jc, ix].mean():.3e}; lobe median {np.median(Mh[lobe]):.3e}; "
          f"min {Mh.min():+.1e} ({(Mh < 0).mean()*100:.1f} % of cells negative)")
if made12:
    fig.tight_layout()
    fig.savefig("fig12.png", dpi=150)

# ----- Fig. 11 -----
print("Fig. 11:")
vdfs = sorted(glob.glob(args.vdf))
if not vdfs:
    print("  no VDF files; skipping fig11")
    sys.exit(0)
try:
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    marching_cubes = None
    print("  scikit-image not found: plotting central slices instead of isosurfaces")

BOX = np.array([[-6.5, 6.5], [-6.5, 6.5], [-5.0, 8.0]])   # paper's Fig. 11 box, in v_te

def vdf_cube(f, cid, box):
    """Electron VDF of cell cid on its velocity mesh, in v_te, sheet frame; cropped to box
    ([[lo, hi]] x3 in v_te), or to auto_box() when box is None. Returns F, axes, dv, box."""
    nb = f.get_velocity_mesh_size("electron").astype(int)
    W = int(f.get_WID())
    vmn = f.get_velocity_mesh_extent("electron")[:3]
    dv = f.get_velocity_mesh_dv("electron")
    ids, data = f.read_blocks(cid, "electron")
    ids = np.asarray(ids, dtype=np.int64)
    data = np.asarray(data, dtype=float).reshape(len(ids), W, W, W)   # [block, cz, cy, cx]
    F = np.zeros(tuple(nb * W))
    bx, by, bz = ids % nb[0], (ids // nb[0]) % nb[1], ids // (nb[0] * nb[1])
    for b in range(len(ids)):
        F[bx[b]*W:(bx[b]+1)*W, by[b]*W:(by[b]+1)*W, bz[b]*W:(bz[b]+1)*W] = data[b].transpose(2, 1, 0)
    axes = [(vmn[d] + (np.arange(nb[d] * W) + 0.5) * dv[d]) / vte for d in range(3)]
    if flip < 0:                                   # (vy, vz) -> (-vy, -vz)
        F = F[:, ::-1, ::-1]
        axes = [axes[0], -axes[1][::-1], -axes[2][::-1]]
    if box is None:
        box = auto_box(F, axes)
    sel = [(a >= lo) & (a <= hi) for a, (lo, hi) in zip(axes, box)]
    F = F[np.ix_(*sel)]
    axes = [a[s] for a, s in zip(axes, sel)]
    return F, axes, dv / vte, box

def auto_box(F, axes):
    """Whole-v_te box around where f exceeds half the lowest isosurface level, plus 1 v_te."""
    m = F >= 0.5 * min(args.levels) * F.max()
    box = []
    for d in range(3):
        idx = np.where(m.any(axis=tuple(e for e in range(3) if e != d)))[0]
        box.append([np.floor(axes[d][idx[0]] - 1.0), np.ceil(axes[d][idx[-1]] + 1.0)])
    return np.array(box)

styles = [("royalblue", 0.15), ("turquoise", 0.3), ("hotpink", 0.6), ("gold", 0.8)]   # outer -> inner

def vdf_panel(fig, k, F, axes, dvn, box, title):
    if marching_cubes is None:
        ax = fig.add_subplot(1, 2, k + 1)
        iy = np.argmin(np.abs(axes[1] - axes[1][np.argmax(F.max(axis=(0, 2)))]))
        ax.pcolormesh(axes[0], axes[2], F[:, iy, :].T, shading="auto", cmap="turbo")
        ax.set_xlabel(r"$v_x/v_{te}$"); ax.set_ylabel(r"$v_z/v_{te}$")
        ax.set_title(title + rf", $v_y$={axes[1][iy]:.1f}")
        ax.set_aspect("equal")
        return
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for lev, (col, alpha) in zip(sorted(args.levels), styles):
        if not F.min() < lev * F.max() < F.max():
            continue
        verts, faces, _, _ = marching_cubes(F, lev * F.max(), spacing=tuple(dvn))
        verts = verts + np.array([axes[0][0], axes[1][0], axes[2][0]])
        ax.add_collection3d(Poly3DCollection(verts[faces], alpha=alpha, facecolor=col, edgecolor="none"))
    ax.set_xlim(*box[0]); ax.set_ylim(*box[1]); ax.set_zlim(*box[2])
    ax.set_box_aspect(tuple(box[:, 1] - box[:, 0]))
    ax.set_xlabel(r"$v_x/v_{te}$"); ax.set_ylabel(r"$v_y/v_{te}$"); ax.set_zlabel(r"$v_z/v_{te}$")
    ax.set_title(title)

fig = plt.figure(figsize=(12, 5.5))
for k, (t, lab) in enumerate(zip(args.t11, "ab")):
    fn, tt = nearest(vdfs, t)
    f = pt.vlsvfile.VlsvReader(fn)
    if args.xline is not None:
        xX = args.xline * DI
    else:
        fb, _ = nearest(bulk, tt)                     # B at the same output time
        xX = xline(flux_function(*np.moveaxis(grid(pt.vlsvfile.VlsvReader(fb), "vg_b_vol")[..., :2], -1, 0)))[0]
    cid = int(np.atleast_1d(f.get_cellid_with_vdf(np.array([xX, ysheet, 0.0]), "electron"))[0])
    cc = f.get_cell_coordinates(cid)
    ddx = (cc[0] - xX + (xmax - xmin) / 2) % (xmax - xmin) - (xmax - xmin) / 2
    print(f"    X-line at x={xX/DI:+.2f} d_i; nearest cell with a VDF: {cid} at x-x_X={ddx/DI:+.2f} d_i, "
          f"y-ysheet={(cc[1]-ysheet)/DI:+.2f} d_i")
    if abs(ddx) > 2 * dx:
        print(f"    note: that is {abs(ddx)/dx:.0f} cells from the X-line (VDF output stride)")
    F, axes, dvn, box = vdf_cube(f, cid, BOX)
    title = rf"({lab}) $f_e$ at X-line (x={cc[0]/DI:.1f} $d_i$), $t\omega_{{ci}}$={tt:.1f}"
    if f.check_variable(NM):
        title += "\n" + rf"$\sigma_{{rel}}/n$={float(np.atleast_1d(f.read_variable(NM, cellids=cid))[0]):.3g}"
    vdf_panel(fig, k, F, axes, dvn, box, title)
fig.tight_layout()
fig.savefig("fig11.png", dpi=150)

# ----- Fig. 13: VDF where the electrons are least Maxwellian -----
print("Fig. 13:")
made13 = False
fig = plt.figure(figsize=(12, 7.0))                    # fitted boxes can be tall
for k, (t, lab) in enumerate(zip(args.t13 if args.t13 is not None else args.t11, "ab")):
    fn, tt = nearest(vdfs, t)
    f = pt.vlsvfile.VlsvReader(fn)
    fb = pt.vlsvfile.VlsvReader(nearest(bulk, tt)[0])
    src = f if f.check_variable(NM) else fb
    if not src.check_variable(NM):
        print(f"    {NM} not in {fn} or the bulk file; skipping fig13")
        break
    M = grid(src, NM)
    try:
        cw = np.atleast_1d(f.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name="electron"))
        bpc = np.atleast_1d(f.read(mesh="SpatialGrid", tag="BLOCKSPERCELL", name="electron"))
    except Exception:
        cw = np.atleast_1d(f.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS"))
        bpc = np.atleast_1d(f.read(mesh="SpatialGrid", tag="BLOCKSPERCELL"))
    cw = cw[bpc > 0].astype(np.int64)
    ii, jj = (cw - 1) % nx, (cw - 1) // nx
    keep = in_half[jj]                                  # the chosen sheet's half of the domain
    cw, ii, jj = cw[keep], ii[keep], jj[keep]
    best = int(np.argmax(M[jj, ii]))
    cid, ib, jb = int(cw[best]), ii[best], jj[best]
    xX = xline(flux_function(*np.moveaxis(grid(fb, "vg_b_vol")[..., :2], -1, 0)))[0]
    wrapx = lambda d: (d + (xmax - xmin) / 2) % (xmax - xmin) - (xmax - xmin) / 2
    Mh = np.where(in_half[:, None], M, -np.inf)
    jg, ig = np.unravel_index(np.argmax(Mh), M.shape)
    print(f"    largest sigma/n with a stored VDF: {M[jb, ib]:.3e} in cell {cid} at "
          f"x-x_X={wrapx(xc[ib]-xX)/DI:+.2f} d_i, y={flip*(yc[jb]-ysheet)/DI:+.2f} d_i")
    print(f"    largest anywhere on the sheet: {M[jg, ig]:.3e} at x-x_X={wrapx(xc[ig]-xX)/DI:+.2f} d_i, "
          f"y={flip*(yc[jg]-ysheet)/DI:+.2f} d_i")
    F, axes, dvn, box = vdf_cube(f, cid, None)
    print(f"    velocity box [v_te]: vx {box[0]}, vy {box[1]}, vz {box[2]}")
    title = (rf"({lab}) $f_e$ at max $\sigma_{{rel}}/n$={M[jb, ib]:.3g}, $t\omega_{{ci}}$={tt:.1f}" + "\n"
             + rf"$x-x_X$={wrapx(xc[ib]-xX)/DI:+.2f} $d_i$, $y$={flip*(yc[jb]-ysheet)/DI:+.2f} $d_i$")
    vdf_panel(fig, k, F, axes, dvn, box, title)
    made13 = True
if made13:
    fig.tight_layout()
    fig.savefig("fig13.png", dpi=150)
print("wrote fig9.png, fig10.png, fig11.png" + (", fig12.png" if made12 else "")
      + (", fig13.png" if made13 else ""))
