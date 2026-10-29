#!/usr/bin/env python3
"""
Plasma Opening Switch, Liu, Cai, Cao & Lapenta 2025 (JCP 528:113840), Sec. 5.3.

Plots n(x), J(x), Ex(x), Bz(x) from Vlasiator bulk*.vlsv output in the layout of
the paper's Figs. 5-7, optionally overlaid with the paper's curves, and checks
per-species particle and charge conservation.

Paper normalisation (Sec. 2.1, x0 = L, T0 = Te, n0 = n_in, m0 = me):
x/L, n/n_in, J/(n_in e v0), Ex/E0, Bz/B0 with v0 = sqrt(kB Te/me),
B0 = me v0/(e L), E0 = v0 B0. Only non-boundary cells are used. n_in is read
from the run's cfg, which Vlasiator embeds in every bulk file.

Usage:
  plot_pos.py RUN_DIR T [-o OUT.png] [--paper CSV] [--nearest]
  plot_pos.py RUN_DIR --conservation
  plot_pos.py --selftest

T is in units of t0 = L/v0. Without --nearest, the script refuses to plot unless
an output lies within half an output interval of T. --paper overlays the paper
figure for the run's n_in (Fig. 5, 6 or 7) and refuses if T is not that figure's time.

Examples:
  plot_pos.py bulk_low/ 0.02 -o pos_low_fig5.png --paper paper_sec53_curves.csv
  plot_pos.py bulk_low/ --conservation
"""
import argparse
import glob
import os
import sys

import numpy as np

QE = 1.602176634e-19
ME = 9.1093837015e-31
KB = 1.380649e-23
EPS0 = 8.8541878128e-12

L = 0.2
TE_EV = 10.0
TE_K = TE_EV * QE / KB
V0 = np.sqrt(TE_EV * QE / ME)
T0 = L / V0
B0 = ME * V0 / (QE * L)
E0 = V0 * B0

NOT_SYSBOUNDARY = 1   # sysboundarytype enum, common.h
BOUNDARY_DEPTH = 2    # hard-coded in SysBoundaryCondition::determineFace

FIG_TIME = {5: 0.02, 6: 0.026, 7: 0.026}
FIG_NIN = {5: 1e16, 6: 1e18, 7: 1e20}
PAPER_LS = {"n_e": "--", "n_i": ":", "J_ex": "--", "J_ey": ":", "Ex": "--", "Bz": "--"}

_trapz = getattr(np, "trapezoid", None) or np.trapz


class Refusal(Exception):
    """Raised instead of producing output that would answer a different question."""


def debye_norm(n_in):
    return np.sqrt(EPS0 * TE_EV / (QE * n_in)) / L


def _import_pt():
    try:
        import analysator as pt
    except ImportError:
        import pytools as pt
    return pt


def _reader(fn):
    return _import_pt().vlsvfile.VlsvReader(fn)


def parse_cfg(text):
    """{section: {key: [values]}}; '#' comments stripped (they may contain 'key = value')."""
    out, sec = {"": {}}, ""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
            out.setdefault(sec, {})
        elif "=" in line:
            k, v = (t.strip() for t in line.split("=", 1))
            out[sec].setdefault(k, []).append(v)
    return out


def _cfg_scalar(cfg, sec, key, fn):
    vals = cfg.get(sec, {}).get(key, [])
    if len(vals) != 1:
        raise Refusal(f"{fn}: expected one [{sec}] {key} in the embedded cfg, found {vals}")
    return float(vals[0])


def file_n_in(fn):
    text = _reader(fn).get_config_string()
    if not text:
        raise Refusal(f"{fn}: no embedded cfg")
    cfg = parse_cfg(text)
    ne, ni = (_cfg_scalar(cfg, f"{p}_MultiPeak", "rho", fn) for p in ("electron", "ion"))
    if ne != ni:
        raise Refusal(f"{fn}: electron rho {ne:g} != ion rho {ni:g}")
    for p in ("electron", "ion"):
        for k in ("Tx", "Ty", "Tz"):
            T = _cfg_scalar(cfg, f"{p}_MultiPeak", k, fn)
            # v0, t0, B0, E0 above assume the paper's 10 eV
            if not np.isclose(T, TE_K, rtol=1e-6):
                raise Refusal(f"{fn}: {p} {k} = {T:g} K, normalisation assumes {TE_K:.6g} K")
    return ne


def run_n_in(outputs):
    vals = {fn: file_n_in(fn) for _, fn in outputs}
    first = next(iter(vals.values()))
    odd = [fn for fn, v in vals.items() if v != first]
    if odd:
        raise Refusal(f"bulk files from different runs (n_in {first:g} vs e.g. {odd[0]}: {vals[odd[0]]:g})")
    return first


def load_profile(fn):
    f = _reader(fn)
    cid = np.asarray(f.read_variable("cellid"))
    order = np.argsort(cid)  # ascending CellID is ascending x only on this 1D, non-AMR grid

    def rd(name):
        return np.asarray(f.read_variable(name))[order]

    if f.check_variable("vg_boundarytype"):
        keep = rd("vg_boundarytype") == NOT_SYSBOUNDARY
        how = "vg_boundarytype"
    else:
        keep = np.zeros(cid.size, bool)
        keep[BOUNDARY_DEPTH:cid.size - BOUNDARY_DEPTH] = True
        how = f"no vg_boundarytype in file: dropped outermost {BOUNDARY_DEPTH} cells per side"

    if f.check_variable("vg_x"):
        x = rd("vg_x") + 0.5 * rd("vg_dx")  # vg_x is the lower corner
    else:
        xmin, xmax, nx = (f.read_parameter(k) for k in ("xmin", "xmax", "xcells_ini"))
        x = xmin + (cid[order] - 0.5) * (xmax - xmin) / nx

    ne, ni = rd("electron/vg_rho"), rd("ion/vg_rho")
    J = QE * (ni[:, None] * rd("ion/vg_v") - ne[:, None] * rd("electron/vg_v"))
    E, B = rd("vg_e_vol"), rd("vg_b_vol")
    p = {"x": x, "n_e": ne, "n_i": ni, "Jx": J[:, 0], "Jy": J[:, 1], "Jz": J[:, 2],
         "Ex": E[:, 0], "Bz": B[:, 2]}
    p = {k: v[keep] for k, v in p.items()}
    p["mask"] = how
    p["time"] = f.read_parameter("time")
    return p


def list_outputs(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "bulk*.vlsv")))
    if not files:
        raise RuntimeError(f"no bulk*.vlsv files in {run_dir!r}")
    return sorted((_reader(fn).read_parameter("time"), fn) for fn in files)


def pick_output(outputs, t_si, allow_nearest=False):
    ts = np.array([t for t, _ in outputs])
    i = int(np.argmin(np.abs(ts - t_si)))
    tol = 0.5 * np.median(np.diff(ts)) if ts.size > 1 else 0.0
    if abs(ts[i] - t_si) > tol and not allow_nearest:
        raise Refusal(f"no output within {tol / T0:.2g} t0 of t = {t_si / T0:.4g} t0 "
                      f"(nearest: {ts[i] / T0:.4g} t0); use --nearest to plot it anyway")
    return outputs[i]


def read_paper(csv, fig):
    d = {}
    for line in open(csv):
        if line.startswith("#") or line.startswith("fig") or not line.strip():
            continue
        f_, q, x, y = line.strip().split(",")
        if int(f_) == fig:
            d.setdefault(q, ([], []))
            d[q][0].append(float(x))
            d[q][1].append(float(y))
    return {q: (np.array(x), np.array(y)) for q, (x, y) in d.items()}


def plot(run_dir, t_norm, out_png, paper_csv=None, allow_nearest=False):
    outputs = list_outputs(run_dir)
    n_in = run_n_in(outputs)
    fig = None
    if paper_csv:
        figs = [k for k, v in FIG_NIN.items() if np.isclose(n_in, v, rtol=1e-6)]
        if not figs:
            raise Refusal(f"n_in = {n_in:g} matches none of the paper's Figs. 5-7")
        fig = figs[0]
        # an overlay at another time would compare different physics
        if not np.isclose(t_norm, FIG_TIME[fig]):
            raise Refusal(f"Fig. {fig} is at t = {FIG_TIME[fig]} t0, requested t = {t_norm}")
    t_got, fn = pick_output(outputs, t_norm * T0, allow_nearest)
    p = load_profile(fn)
    off_time = not np.isclose(t_got / T0, t_norm, rtol=0, atol=0.5e-4)

    print(f"n_in = {n_in:.6g} m^-3 (from embedded cfg); requested t = {t_norm:.5f} t0; "
          f"using {fn} at t = {t_got / T0:.5f} t0")
    print(f"boundary cells: {p['mask']}; {p['x'].size} cells plotted")
    print(f"max |J_ez| / (n_in e v0) = {np.abs(p['Jz']).max() / (n_in * QE * V0):.3g} (zero in 1d2v)")

    jn = n_in * QE * V0
    xn = p["x"] / L
    paper = read_paper(paper_csv, fig) if paper_csv else {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figr, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0, 0], "n / n_in", "(a) density",
         [("n_e", p["n_e"] / n_in, "C0", "n_e"), ("n_i", p["n_i"] / n_in, "C1", "n_i")]),
        (axes[0, 1], "J / (n_in e v0)", "(b) current density",
         [("J_ex", p["Jx"] / jn, "C0", "J_ex"), ("J_ey", p["Jy"] / jn, "C1", "J_ey"),
          (None, p["Jz"] / jn, "C2", "J_ez")]),
        (axes[1, 0], "Ex / E0", "(c) electric field", [("Ex", p["Ex"] / E0, "C0", "Ex")]),
        (axes[1, 1], "Bz / B0", "(d) magnetic field", [("Bz", p["Bz"] / B0, "C0", "Bz")]),
    ]
    for ax, ylabel, title, curves in panels:
        for key, y, col, lab in curves:
            ax.plot(xn, y, col, label=lab)
            if key in paper:
                ax.plot(*paper[key], "k", ls=PAPER_LS[key], lw=1.0, zorder=3, label=f"{lab} (paper)")
        ax.set_xlabel("x / L")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(curves) > 1 or paper:
            ax.legend(fontsize=8)

    head = f"POS (Sec. 5.3): t = {t_got / T0:.4f} t0, n_in = {n_in:.0e} m$^{{-3}}$"
    if paper:
        head += f", black: Liu et al. Fig. {fig}"
    figr.suptitle(head, y=0.99)
    if off_time:
        figr.text(0.5, 0.945, f"NOT the requested t = {t_norm:.4f} t0", color="red",
                  ha="center", fontsize=11)
    figr.tight_layout(rect=(0, 0, 1, 0.92 if off_time else 0.96))
    figr.savefig(out_png, dpi=140)
    plt.close(figr)
    print(f"wrote {out_png}")


def conservation(run_dir):
    """Per output: particles per area of each species and the net charge, in n_in*L.
    q_n integrates the moments; q_E = lambda^2 [Ex(last) - Ex(first)]/E0 is what Gauss's
    law needs between the first and last interior cells. Sec. 5.3 physics keeps both ~0
    (no particles reach the boundaries before t ~ 0.026 t0)."""
    outputs = list_outputs(run_dir)
    n_in = run_n_in(outputs)
    lam2 = debye_norm(n_in) ** 2
    rows = []
    for t, fn in outputs:
        p = load_profile(fn)
        xn = p["x"] / L
        dxn = np.diff(xn).mean()
        rows.append((t / T0, p["n_e"].sum() * dxn / n_in, p["n_i"].sum() * dxn / n_in,
                     _trapz((p["n_i"] - p["n_e"]) / n_in, xn),
                     lam2 * (p["Ex"][-1] - p["Ex"][0]) / E0))
    ne0, ni0 = rows[0][1], rows[0][2]
    print(f"n_in = {n_in:.6g} m^-3 (from embedded cfg), lambda = {np.sqrt(lam2):.4e};  "
          f"N, q in units of n_in*L;  dN = N/N(t0) - 1")
    print(f"{'t/t0':>9} {'N_e':>12} {'N_i':>12} {'dN_e':>10} {'dN_i':>10} {'q_n':>11} {'q_E':>11}")
    for t, ne, ni, qn, qe in rows:
        print(f"{t:9.5f} {ne:12.6e} {ni:12.6e} {ne / ne0 - 1:10.2e} {ni / ni0 - 1:10.2e} "
              f"{qn:11.3e} {qe:11.3e}")
    print("q_n ~ q_E != 0: charge gained/lost by the Vlasov update; "
          "q_n ~ 0 but q_E != 0: field solver or boundary inconsistent with Gauss's law.")
    return rows


# --- self-test: synthetic outputs through the real code paths, no Vlasiator needed --------
def selftest():
    nb, nx = BOUNDARY_DEPTH, 100 + 2 * BOUNDARY_DEPTH
    dx = L / 100
    xc = -nb * dx + (np.arange(nx) + 0.5) * dx
    cid = np.arange(1, nx + 1)
    bt = np.ones(nx, int)
    bt[:nb], bt[-nb:] = 4, 3
    n_in = 1e16
    slab = ((xc > 0.05) & (xc < 0.15)) * n_in
    g = lambda c, w: np.exp(-((xc - c) / w) ** 2)
    ve = np.stack([-1e5 * g(.03, .02), 3e4 * g(.04, .02), -2e4 * g(.07, .015)], 1)
    vi = np.stack([np.zeros(nx), -5e2 * g(.04, .02), 8e1 * g(.07, .015)], 1)
    Ex = 1e5 * g(.05, .01) + 3e3 * (xc / L)
    Bz = 1e-6 * xc / L
    lost = np.zeros(nx)
    lost[40] = 0.01 * n_in
    z3 = np.zeros((nx, 3))

    def cfg(rho_e=n_in, rho_i=n_in, T=116045.181):
        # a commented-out 'rho = ...' must not be read as a value
        return (f"project = MultiPeak\n[electron_MultiPeak]\nn = 1\n# rho = 5.0e+99 (old)\n"
                f"rho = {rho_e:.1e}\nTx = {T}\nTy = {T}\nTz = {T}\n"
                f"[ion_MultiPeak]\nrho   =   {rho_i:.1e}   # same as electrons\n"
                f"Tx = {T}\nTy = {T}\nTz = {T}\n")

    def snap(t, ne, ve_, vi_, ex, bz, full=True, conf=None):
        rev = lambda a: np.asarray(a)[::-1].copy()  # files need not be CellID-ordered
        d = {"time": t, "xmin": -nb * dx, "xmax": L + nb * dx, "xcells_ini": nx,
             "cellid": rev(cid), "electron/vg_rho": rev(ne), "ion/vg_rho": rev(slab),
             "electron/vg_v": rev(ve_), "ion/vg_v": rev(vi_),
             "vg_e_vol": rev(np.stack([ex, 0 * ex, 0 * ex], 1)),
             "vg_b_vol": rev(np.stack([0 * bz, 0 * bz, bz], 1)), "_cfg": conf or cfg()}
        if full:
            d.update({"vg_boundarytype": rev(bt), "vg_x": rev(xc - dx / 2), "vg_dx": rev(np.full(nx, dx))})
        return d

    good = {"bulk.0000000.vlsv": snap(0.0, slab, z3, z3, 0 * xc, 0 * xc),
            "bulk.0000001.vlsv": snap(0.02 * T0, slab - lost, ve, vi, Ex, Bz),
            "bulk.0000002.vlsv": snap(0.04 * T0, slab - lost, ve, vi, Ex, Bz, full=False)}
    reg = dict(good)

    class Rd:
        def __init__(s, d): s.d = d
        def read_parameter(s, k): return s.d[k]
        def read_variable(s, k): return s.d[k]
        def check_variable(s, k): return k in s.d
        def get_config_string(s): return s.d["_cfg"]

    class PT:
        class vlsvfile:
            VlsvReader = staticmethod(lambda fn: Rd(reg[fn]))

    def refuses(fn, *args, **kw):
        try:
            fn(*args, **kw)
        except Refusal:
            return True
        return False

    def with_cfg(conf_last):
        reg.clear()
        reg.update(good)
        reg["bulk.0000002.vlsv"] = dict(good["bulk.0000002.vlsv"], _cfg=conf_last)

    mod = sys.modules[__name__]
    real_pt, real_glob = mod._import_pt, glob.glob
    mod._import_pt = lambda: PT
    glob.glob = lambda pattern: sorted(reg)
    interior = slice(nb, nx - nb)
    jexp = QE * (slab[:, None] * vi - (slab - lost)[:, None] * ve)
    checks = []
    try:
        p = load_profile("bulk.0000001.vlsv")
        checks += [
            ("boundary cells removed via vg_boundarytype", p["x"].size == 100 and p["mask"] == "vg_boundarytype"),
            ("x = cell centres on [0, L] from vg_x + vg_dx/2", np.allclose(p["x"], xc[interior])),
            ("unordered CellIDs sorted back to x order", np.allclose(p["n_e"], (slab - lost)[interior])),
            ("n_i passed through", np.allclose(p["n_i"], slab[interior])),
            ("J = e(n_i v_i - n_e v_e), all three components",
             all(np.allclose(p[k], jexp[interior, i]) for i, k in enumerate(("Jx", "Jy", "Jz")))),
            ("Jz nonzero and peaked at x~0.07, not Jy's 0.04",
             np.abs(p["Jz"]).max() > 0 and abs(p["x"][np.argmax(np.abs(p["Jz"]))] - 0.07) < 0.005),
            ("Ex, Bz passed through", np.allclose(p["Ex"], Ex[interior]) and np.allclose(p["Bz"], Bz[interior])),
        ]
        q = load_profile("bulk.0000002.vlsv")
        checks += [("fallbacks: xmin/CellID for x, outer 2 cells dropped",
                    np.allclose(q["x"], xc[interior]) and q["mask"].startswith("no vg_boundarytype"))]

        outs = list_outputs(".")
        checks += [("n_in read from embedded cfg, comments ignored", run_n_in(outs) == n_in),
                   ("nearest output picked", pick_output(outs, 0.019 * T0)[1] == "bulk.0000001.vlsv"),
                   ("far-off time refused", refuses(pick_output, outs, 0.1 * T0)),
                   ("far-off time allowed with allow_nearest",
                    pick_output(outs, 0.1 * T0, allow_nearest=True)[1] == "bulk.0000002.vlsv")]

        rows = conservation(".")
        lam2 = debye_norm(n_in) ** 2
        exp_q = 0.01 * dx / L
        checks += [
            ("conservation: electron loss seen as dN_e and q_n",
             np.isclose(rows[0][1] - rows[1][1], exp_q, rtol=1e-9) and np.isclose(rows[1][3], exp_q, rtol=1e-9)),
            ("conservation: ions unchanged", rows[1][2] == rows[0][2]),
            ("conservation: q_E = lambda^2 dEx/E0",
             np.isclose(rows[1][4], lam2 * (Ex[nx - nb - 1] - Ex[nb]) / E0, rtol=1e-12)),
        ]

        csv = "/tmp/pos_selftest_paper.csv"
        with open(csv, "w") as fh:
            fh.write("# test\nfig,quantity,x,y\n" + "".join(
                f"5,{k},{a},{b}\n" for k in ("n_e", "n_i", "J_ex", "J_ey", "Ex", "Bz") for a, b in ((0, 0), (1, 1))))
        png = "/tmp/pos_selftest_plot.png"
        if os.path.exists(png):
            os.remove(png)
        plot(".", 0.02, png, paper_csv=csv)
        checks += [("plot() with paper overlay (Fig. 5 from n_in) writes a PNG", os.path.getsize(png) > 0),
                   ("overlay refused at a time that is not Fig. 5's", refuses(plot, ".", 0.04, png, paper_csv=csv))]

        with_cfg(cfg(rho_e=1e18, rho_i=1e18))
        checks += [("bulk files from different runs refused", refuses(conservation, "."))]
        with_cfg(cfg(rho_i=2e16))
        checks += [("electron/ion rho mismatch refused", refuses(conservation, "."))]
        with_cfg(cfg(T=2 * 116045.181))
        checks += [("non-10 eV temperature refused", refuses(conservation, "."))]
        reg.clear()
        reg.update({k: dict(v, _cfg=cfg(3e17, 3e17)) for k, v in good.items()})
        checks += [("overlay refused for an n_in with no paper figure", refuses(plot, ".", 0.02, png, paper_csv=csv))]
    finally:
        mod._import_pt, glob.glob = real_pt, real_glob

    ok = True
    for name, cond in checks:
        print(f"  {name:62s} {'PASS' if cond else 'FAIL'}")
        ok = ok and bool(cond)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("t", nargs="?", type=float, help="comparison time in units of t0 = L/v0")
    ap.add_argument("-o", "--out", default="pos_profile.png")
    ap.add_argument("--paper", help="paper_sec53_curves.csv; figure chosen from the run's n_in")
    ap.add_argument("--nearest", action="store_true", help="plot the nearest output even if far from T")
    ap.add_argument("--conservation", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.run_dir is None:
        ap.error("RUN_DIR is required")
    if a.t is None and not a.conservation:
        ap.error("T is required unless --conservation")
    try:
        if a.conservation:
            conservation(a.run_dir)
        else:
            plot(a.run_dir, a.t, a.out, a.paper, a.nearest)
    except Refusal as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
