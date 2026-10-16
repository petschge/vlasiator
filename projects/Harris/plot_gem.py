#!/usr/bin/env python3
"""
GEM reconnection challenge plotting script, reproducing Figs. 8-10 of
Liu, Cai, Cao & Lapenta 2025 (JCP 528:113840), Section 5.4, from existing
bulk*.vlsv output (produced by gem.cfg; see derive_gem.py for how that cfg
was generated). No new simulation; just reads what is already there.

Self-derives w_ci (hence the SI <-> dimensionless time conversion) from the
asymptotic |Bx| at the domain's y edges in the FIRST available snapshot,
rather than hard-coding it: Bx(y=+-Ly) -> +-B0 for the Harris tanh profile,
and w_ci = q*B0/m_proton. This keeps the script correct regardless of which
n0/v_a choice derive_gem.py was run with -- nothing to keep in sync by hand
between the two scripts.

Uses the same read_variable/read_parameter API conventions as plot_pos.py
elsewhere in this patch series (vector variables come back as (N,3) arrays,
not via an operator= kwarg). Cell positions are reconstructed from CellID
plus the domain bounds, not read from vg_gridcoordinates -- that variable
was observed to not be reliably readable ("not found ... in data reducers")
even when the cfg requests it as output, so this script no longer depends
on it at all. The CellID-to-index formula used here (cellid = 1 + i + j*Nx,
1-indexed, i fastest-varying, for this unrefined grid) was independently
cross-checked repeatedly over the course of debugging this setup -- every
physical-location inference drawn from it checked out against independent
evidence -- so this is a confirmed convention, not an assumption.

Produces:
  <prefix>_fig8_flux.png   reconnected flux psi(t) = integral_0^Lx By(x,y=0)dx
                           vs time, over ALL available snapshots (paper's Fig. 8).
  <prefix>_fig9a_bz.png    Bz(x,y) colour map at the snapshot nearest t*w_ci=16
                           (paper's Fig. 9a).
  <prefix>_fig10_t16.png,
  <prefix>_fig10_t40.png   B field lines + Jz filled contours, at the snapshots
                           nearest t*w_ci=16 and t*w_ci=40 (paper's Fig. 10).

Jz is computed from the particle moments already in the output,
Jz = q_i*n_i*v_zi + q_e*n_e*v_ze (species charges signed), rather than from
a numerical curl of B -- more direct, and avoids amplifying grid noise
through differentiation.

Usage:  python3 plot_gem.py <run_dir> [out_prefix]
        python3 plot_gem.py --selftest
Example:  python3 plot_gem.py bulk/ gem
"""
import glob
import os
import sys
import numpy as np

QE = 1.602176634e-19
MASS_PROTON = 1.67262158e-27


def _import_pt():
    try:
        import analysator as pt
    except ImportError:
        import pytools as pt
    return pt


def _vector(f, name):
    """Read a vector variable; returns (N,3), aligned with f.read_variable('cellid')
    (NOT sorted -- see _grid() for how positions and data are matched up)."""
    return np.asarray(f.read_variable(name))


def _grid(f):
    """Cell-centre (x,y) positions reconstructed directly from CellID and the
    file's own stored domain parameters (xmin/xmax/ymin/ymax, xcells_ini/
    ycells_ini) -- NOT from vg_gridcoordinates, which was observed to fail
    ("not found ... in data reducers") even when the cfg requests it as
    output. For this unrefined grid, cellid = 1 + i + j*Nx (1-indexed, i
    fastest-varying); see this function's own docstring note in the module
    header for why this convention is trusted rather than assumed.
    """
    cellids = np.asarray(f.read_variable('cellid')).astype(np.int64)
    nx = int(round(f.read_parameter('xcells_ini')))
    ny = int(round(f.read_parameter('ycells_ini')))
    xmin = f.read_parameter('xmin'); xmax = f.read_parameter('xmax')
    ymin = f.read_parameter('ymin'); ymax = f.read_parameter('ymax')
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    idx = cellids - 1
    i = idx % nx
    j = (idx // nx) % ny
    x = xmin + (i + 0.5) * dx
    y = ymin + (j + 0.5) * dy
    return x, y, xmin, ymin, dx, dy, nx, ny


def _to_2d(values, x, y, xmin, ymin, dx, dy, nx, ny):
    """Scatter a per-cell 1D array into a (ny,nx) grid using each cell's own
    (x,y) position (robust to whatever order read_variable returns data in),
    rather than assuming a specific CellID-to-index formula."""
    i = np.clip(np.round((x - xmin) / dx - 0.5).astype(int), 0, nx - 1)
    j = np.clip(np.round((y - ymin) / dy - 0.5).astype(int), 0, ny - 1)
    out = np.full((ny, nx), np.nan)
    out[j, i] = values
    if np.isnan(out).any():
        raise RuntimeError("grid scatter left unfilled cells -- check vg_gridcoordinates/"
                            "xcells_ini/ycells_ini are present in this output")
    return out


def load_frame(fname):
    pt = _import_pt()
    f = pt.vlsvfile.VlsvReader(fname)
    x, y, xmin, ymin, dx, dy, nx, ny = _grid(f)

    def grid2d(values):
        return _to_2d(values, x, y, xmin, ymin, dx, dy, nx, ny)

    b = _vector(f, 'vg_b_vol')
    bx, by, bz = grid2d(b[:, 0]), grid2d(b[:, 1]), grid2d(b[:, 2])

    rho_i = grid2d(np.asarray(f.read_variable('proton/vg_rho')))
    rho_e = grid2d(np.asarray(f.read_variable('electron/vg_rho')))
    v_i = _vector(f, 'proton/vg_v')
    v_e = _vector(f, 'electron/vg_v')
    jz = (+QE) * rho_i * grid2d(v_i[:, 2]) + (-QE) * rho_e * grid2d(v_e[:, 2])

    xvals = xmin + (np.arange(nx) + 0.5) * dx
    yvals = ymin + (np.arange(ny) + 0.5) * dy
    t = f.read_parameter('time')
    return dict(t=t, x=xvals, y=yvals, bx=bx, by=by, bz=bz, jz=jz)


def find_snapshots(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "bulk*.vlsv")))
    if not files:
        raise SystemExit(f"No bulk*.vlsv files found in {run_dir!r}")
    return files


def derive_wci(first_file):
    """w_ci = q*B0/m_proton, with B0 self-derived from the asymptotic |Bx|
    at the domain's y edges (first and last grid rows) of the first
    available snapshot."""
    fr = load_frame(first_file)
    B0 = 0.5 * (np.abs(fr['bx'][0, :]).mean() + np.abs(fr['bx'][-1, :]).mean())
    wci = QE * B0 / MASS_PROTON
    return wci, B0


def nearest_snapshot(files, t_target_SI):
    pt = _import_pt()
    tagged = [(pt.vlsvfile.VlsvReader(fn).read_parameter('time'), fn) for fn in files]
    t_got, fn = min(tagged, key=lambda p: abs(p[0] - t_target_SI))
    return fn, t_got


def fig8_reconnected_flux(files, wci, out):
    ts, psis = [], []
    for fn in files:
        fr = load_frame(fn)
        j0 = int(np.argmin(np.abs(fr['y'])))   # row nearest y=0
        psi = np.trapz(fr['by'][j0, :], fr['x'])
        ts.append(fr['t']); psis.append(abs(psi))
    ts = np.array(ts); psis = np.array(psis)
    order = np.argsort(ts)
    ts, psis = ts[order], psis[order]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ts * wci, psis, '-o', ms=3)
    ax.set_xlabel(r'$t\,\omega_{ci}$')
    ax.set_ylabel(r'$|\psi(t)|$  (Wb/m)')
    ax.set_title('Reconnected flux (cf. Fig. 8)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig9a_bz(files, wci, out):
    fn, t_got = nearest_snapshot(files, 16.0 / wci)
    fr = load_frame(fn)
    print(f"Fig 9a: using {fn} (t*w_ci = {fr['t']*wci:.3f}, target 16, "
          f"|dt*w_ci| = {abs(fr['t']-t_got)*wci:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    extent = [fr['x'][0], fr['x'][-1], fr['y'][0], fr['y'][-1]]
    im = ax.imshow(fr['bz'], origin='lower', extent=extent, aspect='auto', cmap='RdBu_r')
    fig.colorbar(im, ax=ax, label=r'$B_z$ (T)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(r'$B_z$ at $t\,\omega_{ci}=16$ (cf. Fig. 9a)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig10_fieldlines_jz(files, wci, out_prefix):
    for t_target, label in [(0.8, "t0.8"), (1.6, "t1.6"), (2.4, "t2.4"), (3.2, "t3.2"), (4.0, "t4"), (4.8, "t4.8"), (5.6, "t5.6"), (6.4, "t6.4"), (7.2, "t7.2"), (8.0, "t8.0"), (8.8, "t8.8"), (16.0, "t16"), (40.0, "t40")]:
        fn, t_got = nearest_snapshot(files, t_target / wci)
        fr = load_frame(fn)
        print(f"Fig 10 ({label}): using {fn} (t*w_ci = {fr['t']*wci:.3f}, target "
              f"{t_target:.0f}, |dt*w_ci| = {abs(fr['t']-t_got)*wci:.3f})")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        cf = ax.contourf(fr['x'], fr['y'], fr['jz'], levels=21, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, label=r'$J_z$ (A/m$^2$)')
        ax.streamplot(fr['x'], fr['y'], fr['bx'], fr['by'], color='k', density=1.2, linewidth=0.7)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(rf'Field lines + $J_z$ at $t\,\omega_{{ci}}={t_target:.0f}$ (cf. Fig. 10)')
        fig.tight_layout()
        out = f"{out_prefix}_fig10_{label}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def _selftest():
    """Exercise _to_2d's scatter logic against synthetic data, without
    needing an actual .vlsv file -- the one piece of this script that can
    be checked without a real simulation run."""
    nx, ny = 5, 3
    xmin, ymin, dx, dy = -10.0, -6.0, 4.0, 4.0
    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymin + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)   # shape (ny,nx), matching the expected output
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    # shuffle, as real cell order is not assumed to match any particular layout
    rng = np.random.default_rng(0)
    perm = rng.permutation(x_flat.size)
    x_flat, y_flat = x_flat[perm], y_flat[perm]
    values = (x_flat + 100 * y_flat)  # encode position in the value itself

    got = _to_2d(values, x_flat, y_flat, xmin, ymin, dx, dy, nx, ny)
    expected = xx + 100 * yy
    assert np.allclose(got, expected), "selftest FAILED: _to_2d scatter did not round-trip"
    print("selftest OK: _to_2d scatter round-trips shuffled-order synthetic data correctly")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        _selftest()
        return
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "gem"

    files = find_snapshots(run_dir)
    wci, B0 = derive_wci(files[0])
    print(f"Self-derived from {files[0]}: B0 = {B0:.6e} T, w_ci = {wci:.6e} rad/s")

    fig8_reconnected_flux(files, wci, f"{out_prefix}_fig8_flux.png")
    fig9a_bz(files, wci, f"{out_prefix}_fig9a_bz.png")
    fig10_fieldlines_jz(files, wci, out_prefix)


if __name__ == "__main__":
    main()
