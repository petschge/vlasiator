#!/usr/bin/env python3
"""
Compare 1-4 LandauDamping runs with Fig. 1 of Liu, Cai, Cao & Lapenta 2025
(JCP 528:113840, Sec. 5.1): E_Ex(t) = 0.5*integral(Ex^2) dx.

Usage:
    python3 compare_landau_damping.py RUN_DIR [RUN_DIR ...] [-o OUT.png]
                                      [--fit-window TMIN TMAX]

Each RUN_DIR holds a bulk*.vlsv time series. Labels (CSL-RME-WG/WO, theta)
and all physical/grid parameters come from the run config that Vlasiator
embeds in every VLSV file, so they cannot drift from what was actually run.

Caveat: Vlasiator embeds only the --run_config file. Options given on the
command line or in global/user config files are not visible here.
"""

import argparse
import glob
import os
from fractions import Fraction

import numpy as np
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.special import wofz

try:
    import analysator as pt
except ImportError:
    import pytools as pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vlasiator's own constants (common.h) so omega_pe and lambda_D match the run
Q_E = 1.60217653e-19
M_ELECTRON = 9.10938188e-31
M_PROTON = 1.67262158e-27
K_B = 1.3806503e-23
EPS_0 = 8.85418782e-12

COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple"]

# Defaults Vlasiator applies when a key is absent (parameters.cpp,
# object_wrapper.cpp, projects/LandauDamping/LandauDamping.cpp)
DEFAULTS = {
    "fieldsolver.method": "default_fieldsolver",
    "fieldsolver.theta": "0.5",
    "fieldsolver.enforceGaussLaw": "true",
    "LandauDamping.k": "1.0",
    "_LandauDamping.rho": "1.0e6",
    "_LandauDamping.Temperature": "1.0e5",
    "_LandauDamping.densityPertRelAmp": "0.0",
    "_properties.charge": "1",
    "_properties.mass": "1",
    "_properties.mass_units": "PROTON",
}

_BOOL = {"1": True, "true": True, "yes": True, "on": True,
         "0": False, "false": False, "no": False, "off": False}


# ============================================================
# Embedded config
# ============================================================

def parse_config(text):
    """Boost.Program_options rules: '#' starts a comment anywhere on a line,
    '[sec]' prefixes later keys with 'sec.', repeated keys accumulate.
    analysator's get_config() is not used: it also parses commented-out
    'key = value' text."""
    cfg, section = {}, ""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue  # banner line Vlasiator prepends
        key, val = (s.strip() for s in line.split("=", 1))
        cfg.setdefault(f"{section}.{key}" if section else key, []).append(val)
    return cfg


def cfg_scalar(cfg, key, default_key=None):
    vals = cfg.get(key)
    if vals is None:
        return DEFAULTS[default_key or key]
    if len(vals) > 1:
        # Vlasiator would not accept this either; refuse rather than guess
        raise ValueError(f"config key {key!r} given {len(vals)} times: {vals}")
    return vals[0]


def run_settings(cfg):
    """Field-solver settings and physics of the perturbed species."""
    method = cfg_scalar(cfg, "fieldsolver.method")
    theta = float(cfg_scalar(cfg, "fieldsolver.theta"))
    gauss_str = cfg_scalar(cfg, "fieldsolver.enforceGaussLaw").lower()
    if gauss_str not in _BOOL:
        raise ValueError(f"cannot parse fieldsolver.enforceGaussLaw = {gauss_str!r}")

    pops = cfg.get("ParticlePopulations", [])
    pert = [p for p in pops
            if float(cfg_scalar(cfg, f"{p}_LandauDamping.densityPertRelAmp",
                                "_LandauDamping.densityPertRelAmp")) != 0.0]
    if len(pert) != 1:
        raise ValueError(f"expected exactly one perturbed population, found {pert} in {pops}")
    p = pert[0]

    units = cfg_scalar(cfg, f"{p}_properties.mass_units", "_properties.mass_units")
    if units not in ("PROTON", "ELECTRON"):
        raise ValueError(f"unknown mass_units {units!r} for population {p!r}")
    mass = float(cfg_scalar(cfg, f"{p}_properties.mass", "_properties.mass")) * \
        (M_PROTON if units == "PROTON" else M_ELECTRON)

    return {
        "method": method,
        "theta": theta,
        "gauss": _BOOL[gauss_str],
        "pop": p,
        "k": float(cfg_scalar(cfg, "LandauDamping.k")),
        "n": float(cfg_scalar(cfg, f"{p}_LandauDamping.rho", "_LandauDamping.rho")),
        "T": float(cfg_scalar(cfg, f"{p}_LandauDamping.Temperature",
                              "_LandauDamping.Temperature")),
        "alpha0": float(cfg_scalar(cfg, f"{p}_LandauDamping.densityPertRelAmp",
                                   "_LandauDamping.densityPertRelAmp")),
        "q": abs(int(cfg_scalar(cfg, f"{p}_properties.charge", "_properties.charge"))) * Q_E,
        "m": mass,
    }


def format_theta(theta):
    fr = Fraction(theta).limit_denominator(8)
    return str(fr) if abs(float(fr) - theta) < 1e-12 else f"{theta:g}"


def scheme_label(s):
    if s["method"] not in ("AP", "LCCL"):   # the solver was renamed to LCCL
        return f"fieldsolver={s['method']}"
    return f"CSL-RME-{'WG' if s['gauss'] else 'WO'}, θ={format_theta(s['theta'])}"


# ============================================================
# Linear theory
# ============================================================

def solve_landau_dispersion(klD, guess=(1.5, -0.5)):
    """Least-damped root of eps = 1 + [1 + zeta Z(zeta)]/(k lambda_D)^2,
    zeta = (omega/omega_pe)/(sqrt(2) k lambda_D), Fried-Conte Z.
    Returns omega/omega_pe (complex)."""
    def eqs(v):
        zeta = (v[0] + 1j * v[1]) / (klD * np.sqrt(2))
        e = 1 + (1.0 / klD**2) * (1 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta))
        return [e.real, e.imag]

    sol = root(eqs, guess, tol=1e-14)
    if not sol.success:
        raise RuntimeError(f"dispersion-relation root find failed: {sol.message}")
    return sol.x[0] + 1j * sol.x[1]


# ============================================================
# Loading
# ============================================================

def load_vg_field(f, name):
    cellids = f.read_variable("cellid")
    return f.read_variable(name)[cellids.argsort()]


def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "bulk*.vlsv")))
    if len(files) < 2:
        raise RuntimeError(f"expected >= 2 bulk*.vlsv files in {run_dir!r}, found {len(files)}")

    recs = []
    for fn in files:
        f = pt.vlsvfile.VlsvReader(fn)
        text = f.get_config_string() if hasattr(f, "get_config_string") else None
        if text is None:
            raise RuntimeError(f"{fn}: no embedded config (or analysator too old)")
        recs.append((f.read_parameter("time"), fn, f, run_settings(parse_config(text))))
    recs.sort(key=lambda r: r[0])

    # restarts can rewrite an output time; keep one file per time
    _, first = np.unique([r[0] for r in recs], return_index=True)
    recs = [recs[i] for i in sorted(first)]

    s0 = recs[0][3]
    mixed = sorted({os.path.basename(r[1]) for r in recs if r[3] != s0})
    if mixed:
        print(f"WARNING [{run_dir}]: settings differ from {os.path.basename(recs[0][1])} in "
              f"{len(mixed)} file(s), e.g. {mixed[:3]}; label uses the first file")

    f0 = recs[0][2]
    if f0.read_parameter("ycells_ini") != 1 or f0.read_parameter("zcells_ini") != 1:
        raise RuntimeError(f"{run_dir}: E_Ex integral below assumes a 1D grid (Ny=Nz=1)")
    xmin, xmax = f0.read_parameter("xmin"), f0.read_parameter("xmax")
    Nx = int(f0.read_parameter("xcells_ini"))
    dx = (xmax - xmin) / Nx

    E_Ex = np.array([0.5 * np.sum(r[2].read_variable("fg_e")[..., 0].ravel()**2) * dx
                     for r in recs])
    return {
        "dir": run_dir, "settings": s0, "mixed": bool(mixed),
        "times": np.array([r[0] for r in recs]), "E_Ex": E_Ex,
        "f0": f0, "xmin": xmin, "dx": dx, "Nx": Nx,
    }


def derived(s):
    omega_pe = np.sqrt(s["n"] * s["q"]**2 / (EPS_0 * s["m"]))
    lambda_D = np.sqrt(EPS_0 * K_B * s["T"] / (s["q"]**2 * s["n"]))
    return omega_pe, s["k"] * lambda_D


def sanity_checks(run):
    s, f0, Nx = run["settings"], run["f0"], run["Nx"]
    x = (load_vg_field(f0, "cellid") - 0.5) * run["dx"] + run["xmin"]
    rho = load_vg_field(f0, f"{s['pop']}/vg_rho")
    diff = rho - s["n"] * (1.0 + s["alpha0"] * np.sin(s["k"] * x))
    err = np.max(np.abs(diff))
    tol = s["n"] * 1e-6
    print(f"  rho(t=0) vs n(1+alpha0 sin kx): max|diff| = {err:.4e} (tol {tol:.4e}) "
          f"[{'PASS' if err < tol else 'FAIL'}]")
    # sparse pruning lowers n uniformly; separate that from a wrong profile
    err_shape = np.max(np.abs(diff - diff.mean()))
    print(f"    uniform offset {diff.mean() / s['n']:+.3e} relative (sparse pruning); "
          f"shape with offset removed: max|diff| = {err_shape:.4e} "
          f"[{'PASS' if err_shape < tol else 'FAIL'}]")

    Ex0 = f0.read_variable("fg_e")[..., 0].ravel()
    # cell-centred samples never hit k*x = 0, hence cos(pi/Nx)
    amp_pred = s["q"] * s["n"] * s["alpha0"] / (EPS_0 * s["k"]) * np.cos(np.pi / Nx)
    amp = 0.5 * (Ex0.max() - Ex0.min())
    err = abs(amp - amp_pred)
    print(f"  Ex(t=0) amplitude: actual {amp:.6e}, predicted {amp_pred:.6e}, diff {err:.4e} "
          f"(tol {amp_pred * 1e-3:.4e}) [{'PASS' if err < amp_pred * 1e-3 else 'FAIL'}]")


def fit_envelope(t_wpe, E, window):
    """gamma/omega_pe from local maxima of E_Ex inside `window`."""
    idx = np.where((t_wpe >= window[0]) & (t_wpe <= window[1]))[0]
    pk, _ = find_peaks(E[idx])
    pk = idx[pk]
    pk = pk[E[pk] > 0]
    if len(pk) < 3:
        return None, pk
    # E_Ex ~ exp(2 gamma t)
    return np.polyfit(t_wpe[pk], np.log(E[pk]), 1)[0] / 2.0, pk


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="1-4 run directories with bulk*.vlsv")
    ap.add_argument("-o", "--output", default="landau_damping_comparison.png")
    ap.add_argument("--fit-window", nargs=2, type=float, default=(0.0, 7.0),
                    metavar=("TMIN", "TMAX"),
                    help="t*omega_pe range for the envelope fit (default 0 7). Peaks "
                         "after the signal reaches the numerical floor are shoulders "
                         "of the floor, not of the damped mode.")
    args = ap.parse_args()
    if len(args.runs) > len(COLORS):
        ap.error(f"at most {len(COLORS)} runs")

    runs = [load_run(d) for d in args.runs]

    base = [scheme_label(r["settings"]) + (" [mixed cfg]" if r["mixed"] else "") for r in runs]
    labels = [f"{lab} [{os.path.basename(os.path.normpath(r['dir']))}]" if base.count(lab) > 1 else lab
              for lab, r in zip(base, runs)]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    theory = {}  # klD -> (gamma, anchoring run index)
    print()
    for i, (run, lab) in enumerate(zip(runs, labels)):
        s = run["settings"]
        omega_pe, klD = derived(s)
        t_wpe = run["times"] * omega_pe
        E = run["E_Ex"]
        print(f"=== {lab}  ({run['dir']}, {len(E)} files, t*omega_pe in "
              f"[{t_wpe[0]:.3f}, {t_wpe[-1]:.3f}]) ===")
        print(f"  method={s['method']} theta={s['theta']} enforceGaussLaw={s['gauss']} "
              f"k*lambda_D={klD:.6f} alpha0={s['alpha0']}")
        sanity_checks(run)

        key = round(klD, 9)
        if key not in theory:
            theory[key] = (solve_landau_dispersion(klD), i)
        gamma_th = theory[key][0].imag

        gamma_fit, pk = fit_envelope(t_wpe, E, args.fit_window)
        if gamma_fit is None:
            print(f"  fewer than 3 envelope peaks in {args.fit_window}: no fit")
            curve_label = lab
        else:
            print(f"  fitted gamma/omega_pe = {gamma_fit:.6f} from peaks at t*omega_pe = "
                  f"{np.round(t_wpe[pk], 2).tolist()}  (theory {gamma_th:.6f}, "
                  f"{abs(gamma_fit - gamma_th) / abs(gamma_th) * 100:.2f}% off)")
            curve_label = f"{lab}  (fit $\\gamma/\\omega_{{pe}}$={gamma_fit:.3f})"
            ax.plot(t_wpe[pk], E[pk], "^", color=COLORS[i], markersize=5)
        ax.semilogy(t_wpe, E, "-", color=COLORS[i], linewidth=1.4, label=curve_label)
        run["t_wpe"] = t_wpe

    # Theory line anchored by least squares to the first run with that k*lambda_D
    t_max = max(r["t_wpe"][-1] for r in runs)
    for klD, (om, i) in theory.items():
        r = runs[i]
        m = (r["t_wpe"] >= args.fit_window[0]) & (r["t_wpe"] <= args.fit_window[1]) & (r["E_Ex"] > 0)
        if m.sum() < 2:
            continue
        logA = np.mean(np.log(r["E_Ex"][m]) - 2 * om.imag * r["t_wpe"][m])
        tl = np.linspace(0.0, t_max, 200)
        extra = f", $k\\lambda_D$={klD:.3f}" if len(theory) > 1 else ""
        ax.semilogy(tl, np.exp(logA + 2 * om.imag * tl), "k--", linewidth=1.1,
                    label=f"linear theory: $\\gamma/\\omega_{{pe}}$={om.imag:.4f}{extra}")

    common = {(round(derived(r["settings"])[1], 6), r["settings"]["alpha0"]) for r in runs}
    sub = (f"$k\\lambda_D$={next(iter(common))[0]:.3f}, $\\beta$=2, "
           f"$\\alpha_0$={next(iter(common))[1]}") if len(common) == 1 else "parameters differ, see legend"
    ax.set_title(f"Landau damping (Sec. 5.1, Liu et al. 2025)\n{sub}", fontsize=11)
    ax.set_xlabel(r"$t \, \omega_{pe}$")
    ax.set_ylabel(r"$\mathcal{E}_{Ex} = \frac{1}{2}\int |E_x|^2\,dx$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
