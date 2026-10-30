#!/usr/bin/env python3
"""
Figures 2-4 of Liu, Cai, Cao & Lapenta, J. Comput. Phys. 528 (2025) 113840,
Sec. 5.2 (Weibel instability), from Vlasiator bulk*.vlsv output.

  Fig. 2  field energies (a) and kinetic energies (b) vs time
  Fig. 3  conservation errors |dL1|, |dPx|, |dPy|, |dE_t| vs time
          (one panel per run, e.g. fine and coarse grid)
  Fig. 4  electron f(vx,vy) at x = L/2 and t = --t-vdf (one panel per run,
          e.g. C=1 and C=10)

Each directory given on the command line is one run. Fig. 2 overlays all
runs, Figs. 3 and 4 get one panel per run.

Required output variables (per population <pop>):
    <pop>/vg_rho, <pop>/vg_v, <pop>/vg_ptensor_diagonal   (populations_vg_*)
    fg_e, fg_b
Recommended: vg_v (centre-of-mass velocity; otherwise it is rebuilt from
all populations, which needs all their masses).
Fig. 4 additionally needs VDFs written in the cell at the snapshot position.

The run's cfg is embedded in every vlsv. It is read (comments stripped) to
obtain species masses and, for the MultiPeak project, n0 and v0, and is
audited against the paper's Sec. 5.2 parameters. Use --no-cfg to disable.

MultiPeak initialises Bz = dBz*cos(2 pi x/lambda), the paper uses sin(kx), so
the paper's x = L/2 corresponds to x = L/4 here (--x-frac, set automatically).

Example:
    plot_weibel.py run_C1 run_C10 --labels "C=1" --labels "C=10"
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import analysator as pt
except ImportError:
    import pytools as pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
EPS_0 = 8.8541878128e-12
C_LIGHT = 299792458.0
K_B = 1.380649e-23

FLOOR = 1e-18  # log-axis floor so exact zeros stay visible


def open_vlsv(path):
    return pt.vlsvfile.VlsvReader(path)


# ------------------------------------------------------------------ cfg

M_P = 1.67262192369e-27


def read_cfg(f):
    """Parse the cfg embedded in a vlsv into {(section, key): [values]}.

    analysator's get_config() does not strip comments, so commented-out
    'dt = ...' lines would be returned as values; parse here instead.
    """
    text = f.get_config_string()
    if text is None:
        return None
    sec, out = "", {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
        elif "=" in line:
            k, v = (t.strip() for t in line.split("=", 1))
            out.setdefault((sec, k), []).append(v)
    return out


def cfg_get(cfg, sec, key, cast=float, default=None):
    vals = cfg.get((sec, key))
    return cast(vals[-1]) if vals else default


def cfg_charges(cfg):
    """Species charges in elementary charges, from <pop>_properties."""
    out = {}
    for pop in cfg.get(("", "ParticlePopulations"), []):
        q = cfg_get(cfg, f"{pop}_properties", "charge", float)
        if q is not None:
            out[pop] = q
    return out


def cfg_masses(cfg):
    """Species masses in electron masses, from <pop>_properties."""
    out = {}
    for pop in cfg.get(("", "ParticlePopulations"), []):
        m = cfg_get(cfg, f"{pop}_properties", "mass", float)
        unit = cfg_get(cfg, f"{pop}_properties", "mass_units", str, "PROTON")
        out[pop] = m*(1.0 if unit == "ELECTRON" else M_P/M_E)
    return out


# ---------------------------------------------------------------- units

class Units:
    """Normalisation of Sec. 2.1: reference n0, v0, x0 with m0 = m_e, q = e."""

    def __init__(self, n0, v0, x0=None):
        self.n0, self.v0 = n0, v0
        # lambda = 1 is the paper's choice for this test
        self.x0 = x0 if x0 else np.sqrt(EPS_0*M_E*v0**2/(Q_E**2*n0))
        self.t0 = self.x0/v0
        self.B0 = M_E*v0/(Q_E*self.x0)
        self.E0 = self.B0*v0
        self.lam = np.sqrt(EPS_0*M_E*v0**2/(Q_E**2*n0))/self.x0
        self.c = C_LIGHT/v0
        # Energies/momenta are per unit area; only the x-integral is normalised.
        self.N0 = n0*self.x0
        self.P0 = n0*M_E*v0*self.x0
        self.W0 = n0*M_E*v0**2*self.x0


# ----------------------------------------------------------- diagnostics

def find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "bulk*.vlsv")))
    if not files:
        raise RuntimeError(f"no bulk*.vlsv in {directory!r}")
    tagged = sorted((open_vlsv(fn).read_parameter("time"), fn) for fn in files)
    return np.array([t for t, _ in tagged]), [fn for _, fn in tagged]


def populations(f):
    return sorted(v[:-len("/vg_rho")] for v in f.get_variables()
                  if v.endswith("/vg_rho"))


def sorted_vg(f, name, order):
    return np.asarray(f.read_variable(name), dtype=float)[order]


def check_1d(f):
    nx, ny, nz = f.get_spatial_mesh_size()
    if ny != 1 or nz != 1:
        raise RuntimeError(f"expected Ny=Nz=1, got {nx}x{ny}x{nz}")
    return int(nx)


def com_velocity(f, order, masses):
    # P_ij in Vlasiator is taken about this frame, so it is needed exactly.
    if "vg_v" in f.get_variables():
        return sorted_vg(f, "vg_v", order)
    pops = populations(f)
    missing = [p for p in pops if p not in masses]
    if missing:
        raise RuntimeError(
            f"vg_v not in file and no --mass for {missing}; add 'output = vg_v' "
            "or pass --mass for every population")
    num = sum(masses[p]*M_E*sorted_vg(f, f"{p}/vg_rho", order)[:, None]
              * sorted_vg(f, f"{p}/vg_v", order) for p in pops)
    den = sum(masses[p]*M_E*sorted_vg(f, f"{p}/vg_rho", order) for p in pops)
    return num/den[:, None]


def node_sum_sq(e):
    """Sum of squared node values behind a periodic x-series stored as the average
    (node(i-1) + node(i))/2.

    Each Fourier mode is divided by the average's transfer function; the Nyquist mode,
    which the average removes, cannot be recovered and is taken as zero.
    """
    n = len(e)
    h = 0.5*(1 + np.exp(-2j*np.pi*np.fft.fftfreq(n)))
    F = np.fft.fft(e)
    ok = np.abs(h) > 1e-12
    return float(np.sum(np.abs(F[ok]/h[ok])**2)/n)


def field_momentum(E, B):
    """sum over x cells of E x B (rows = x cells), pairing co-located samples.

    AP Yee layout (ap_electric_field.cpp, Section 0): along x, E_y, E_z, B_x
    sit on the left face x_i (averages of cells i-1, i); E_x, B_y, B_z at the
    cell centre. Only the P_x pairs (E_y B_z, E_z B_y) are staggered; pairing
    them half a cell apart fakes P_x ~ k*dx*|E||B|*L/4 even for a standing
    mode whose true integral is 0, so B_y, B_z are moved to the faces with the
    same averaging. The P_y and P_z pairs are already co-located.
    """
    face = lambda v: 0.5*(v + np.roll(v, 1))
    Ex, Ey, Ez = E.T
    Bx, By, Bz = B.T
    return np.array([(Ey*face(Bz) - Ez*face(By)).sum(),
                     (Ez*Bx - Ex*Bz).sum(),
                     (Ex*By - Ey*Bx).sum()])


def e_at_centres(E):
    """fg_e rows (x cells) moved to cell centres, where the moments live.

    E_x is already there; E_y and E_z sit on the left face x_i (ap_NodesToYee
    averages cells i-1 and i), so the centre of cell i is the mean of faces
    i and i+1.
    """
    Ec = E.copy()
    Ec[:, 1:] = 0.5*(E[:, 1:] + np.roll(E[:, 1:], -1, axis=0))
    return Ec


def diagnostics(f, units, pops, masses, kin_comps, charges=None):
    nx = check_1d(f)
    ext = f.get_spatial_mesh_extent()
    dxv = (ext[3] - ext[0])/nx
    order = np.argsort(np.asarray(f.read_variable("cellid")))
    V = com_velocity(f, order, masses)

    L1 = Px = Py = 0.0
    K = np.zeros(3)
    K_pop = {}
    J_pop = {}   # SI current density per population, for the work diagnostic
    for p in populations(f):
        if p not in masses:
            continue  # only needed for --pops, which main() has checked
        m = masses[p]*M_E
        n = sorted_vg(f, f"{p}/vg_rho", order)
        u = sorted_vg(f, f"{p}/vg_v", order)
        P = sorted_vg(f, f"{p}/vg_ptensor_diagonal", order)
        # Second moment from P about V_cm: M_ii = P_ii + m n (2 u_i V_i - V_i^2)
        M = P + m*n[:, None]*(2*u*V - V**2)
        Kp = 0.5*M.sum(axis=0)*dxv
        mom = m*n[:, None]*u
        Pp = mom.sum(axis=0)*dxv
        if charges and p in charges:
            J_pop[p] = charges[p]*Q_E*n[:, None]*u
        for c, ax in enumerate("xyz"):
            K_pop[f"K:{p}:{ax}"] = Kp[c]/units.W0
            K_pop[f"P:{p}:{ax}"] = Pp[c]/units.P0
        if p in pops:
            K += Kp
            L1 += n.sum()*dxv
            Px += Pp[0]
            Py += Pp[1]

    E = np.asarray(f.read_variable("fg_e"), dtype=float).reshape(-1, 3)
    B = np.asarray(f.read_variable("fg_b"), dtype=float).reshape(-1, 3)
    dxf = f.get_fsgrid_cell_size()[0]
    # The AP solver works on cell-collocated nodes and conserves energy in that norm,
    # but fg_e holds E_y, E_z as averages of nodes i-1 and i (ap_NodesToYee), which
    # understates their energy by sin^2(k dx/2) per mode. Undo the average (periodic
    # 1D grid). E_x and B_y, B_z in fg_e/fg_b are node values.
    WE = 0.5*EPS_0*np.array([np.sum(E[:, 0]**2), node_sum_sq(E[:, 1]), node_sum_sq(E[:, 2])])*dxf
    WB = 0.5/(4e-7*np.pi)*(B**2).sum(axis=0)*dxf
    EB = EPS_0*field_momentum(E, B)*dxf

    Kt = K[kin_comps].sum()
    # Per-population number, for all populations (not only --pops).
    L1_pop = {f"L1:{p}": sorted_vg(f, f"{p}/vg_rho", order).sum()*dxv/units.N0
              for p in populations(f)}
    # Velocity-space block count per population (populations_vg_blocks).
    names = set(f.get_variables())
    blk = {f"blk:{p}": (float(np.asarray(f.read_variable(f"{p}/vg_blocks")).sum())
                        if f"{p}/vg_blocks" in names else np.nan)
           for p in populations(f)}
    fields = {"Pf:x": EB[0]/units.P0, "Pf:y": EB[1]/units.P0,
              "W:E": WE.sum()/units.W0, "W:B": WB.sum()/units.W0}
    # Work rate of E on each population, int J.E dx, in units W0 per t0. Two
    # fields: fg_e (E^{k+1}, at output time) and vg_e_vol (E^{k+theta}, the
    # field the acceleration used), to show how sensitive the check is to it.
    Ec = e_at_centres(E) if len(E) == nx else None
    Ev = sorted_vg(f, "vg_e_vol", order) if "vg_e_vol" in f.get_variables() else None
    for p, J in J_pop.items():
        fields[f"JE:{p}"] = (np.sum(J*Ec)*dxv*units.t0/units.W0) if Ec is not None else np.nan
        fields[f"JEv:{p}"] = (np.sum(J*Ev)*dxv*units.t0/units.W0) if Ev is not None else np.nan
    # In-solver acceleration energy accumulators (populations_vg_acc_energy, debug patch
    # 0005): J/m^3 per cell since the start, what the remap made and what the exact
    # velocity map makes.
    for p in populations(f):
        for key, var in (("AccA", "actual"), ("AccX", "exact")):
            name = f"{p}/vg_acc_energy_{var}"
            fields[f"{key}:{p}"] = (sorted_vg(f, name, order).sum()*dxv/units.W0
                                    if name in names else np.nan)
    return dict(
        **L1_pop, **blk, **K_pop, **fields,
        L1=L1/units.N0,
        Px=(Px + EB[0])/units.P0,
        Py=(Py + EB[1])/units.P0,
        E_E=(WE[0] + WE[1])/units.W0,
        E_Bz=WB[2]/units.W0,
        E_Kx=K[0]/units.W0, E_Ky=K[1]/units.W0, E_K=Kt/units.W0,
        E_t=(Kt + WE.sum() + WB.sum())/units.W0,
    )


def time_series(directory, tf, units, pops, masses, kin_comps, charges=None):
    times, files = tf
    print(f"[{directory}] {len(files)} files, "
          f"t/t0 in [{times[0]/units.t0:.3g}, {times[-1]/units.t0:.3g}]")
    rows = []
    for i, fn in enumerate(files):
        rows.append(diagnostics(open_vlsv(fn), units, pops, masses, kin_comps, charges))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}")
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["t"] = times/units.t0
    out["files"], out["times_si"] = files, times
    return out


# ------------------------------------------------------------ VDF (Fig 4)

def vdf_xy(f, pop, cellid, units):
    """Electron f(vx,vy) = integral of f dvz, in units n0/v0^2."""
    d = f.read_velocity_cells(cellid, pop=pop)
    if not d:
        raise RuntimeError(
            f"cell {cellid} has no VDF for {pop!r}; with "
            "system_write_distribution_stride = S only cellid % S == 0 is "
            "written (see the audit table for the cells your cfg writes)")
    ids = np.fromiter(d.keys(), dtype=np.int64)
    vals = np.fromiter(d.values(), dtype=float)
    xyz = f.get_velocity_cell_coordinates(ids, pop=pop)
    ncell = f.get_velocity_mesh_size(pop=pop)*f.get_velocity_block_size(pop=pop)
    ext = f.get_velocity_mesh_extent(pop=pop)
    dv = (ext[3:6] - ext[0:3])/ncell
    idx = np.floor((xyz - ext[0:3])/dv).astype(int)
    fxy = np.zeros((ncell[0], ncell[1]))
    np.add.at(fxy, (idx[:, 0], idx[:, 1]), vals*dv[2])
    vx = (ext[0] + (np.arange(ncell[0]) + 0.5)*dv[0])/units.v0
    vy = (ext[1] + (np.arange(ncell[1]) + 0.5)*dv[1])/units.v0
    dens = fxy.sum()*dv[0]*dv[1]  # [m^-3], for the integrity check
    return vx, vy, fxy*units.v0**2/units.n0, dens


def snapshot_cell(f, x_frac):
    """Cell containing x = x_frac*L (a cell boundary goes to the right cell)."""
    nx = check_1d(f)
    ext = f.get_spatial_mesh_extent()
    L = ext[3] - ext[0]
    dx = L/nx
    target = ext[0] + (np.floor(x_frac*nx + 1e-9) + 0.5)*dx
    cid = np.sort(np.asarray(f.read_variable("cellid")))
    xc = np.asarray(f.get_cell_coordinates(cid))[:, 0]
    i = int(np.argmin(np.abs(xc - target)))
    return int(cid[i]), xc[i], nx


# --------------------------------------------------------------- fitting

def fit_growth(t, E, window=None, frac=(1e-2, 1e-1)):
    """Least-squares exp fit of the linear phase; returns (slope, window)."""
    if window:
        m = (t >= window[0]) & (t <= window[1])
    else:
        # Rising edge between fractions of the maximum avoids the initial
        # transient and the saturated phase.
        m = (E >= frac[0]*E.max()) & (E <= frac[1]*E.max()) \
            & (t <= t[np.argmax(E)])
    if m.sum() < 5 or np.any(E[m] <= 0):
        return None
    slope, icpt = np.polyfit(t[m], np.log(E[m]), 1)
    return slope, icpt, (t[m][0], t[m][-1])


# ---------------------------------------------------------------- checks

def sanity_checks(run, units, args, f0, pops, masses, cfg=None):
    print("\n=== Normalisation / t=0 checks (first run) ===")
    print(f"  lambda = {units.lam:.4g} (paper: 1),  c/v0 = {units.c:.4g} (paper: 1)")
    print(f"  x0 = {units.x0:.6g} m,  t0 = {units.t0:.6g} s,  "
          f"B0 = {units.B0:.4g} T")
    ext = f0.get_spatial_mesh_extent()
    Lx = (ext[3] - ext[0])/units.x0
    print(f"  Lx/x0 = {Lx:.6g}; 2*pi/k = {2*np.pi/args.k:.6g}")
    B = np.asarray(f0.read_variable("fg_b"), dtype=float).reshape(-1, 3)[:, 2]
    amp = 0.5*(B.max() - B.min())/units.B0
    # Sampling at cell positions misses the crest by up to cos(pi/N).
    print(f"  Bz(t=0) amplitude/B0 = {amp:.4g}; expected alpha0 = {args.alpha0:g}")
    # Initial kinetic energy per population and component, per unit area:
    #   K_c = 1/2 (n/n0) (kB T_c/(m_e v0^2) + (m/m_e)(V_c/v0)^2) Lx/x0.
    # From the cfg temperatures for MultiPeak; otherwise the paper's beta, Tr.
    names = populations(f0)
    for pop in names:
        if f"K:{pop}:x" not in run:
            continue
        mrel = masses.get(pop, np.nan)
        exp = None
        if cfg and cfg_get(cfg, f"{pop}_MultiPeak", "n", int) == 1:
            sec = f"{pop}_MultiPeak"
            nn = cfg_get(cfg, sec, "rho")/units.n0
            exp = {ax: 0.5*nn*(K_B*cfg_get(cfg, sec, f"T{ax}")/(M_E*units.v0**2)
                               + mrel*(cfg_get(cfg, sec, f"V{ax}")/units.v0)**2)*Lx
                   for ax in "xyz"}
        elif pop == "electron":
            exp = {"x": args.beta/4*Lx, "y": args.beta*args.tr/4*Lx}
        if exp is None:
            continue
        for ax, e in exp.items():
            got = run[f"K:{pop}:{ax}"][0]
            rel = abs(got - e)/e
            print(f"  K_{ax}(0) {pop:<9} = {got:.5g}; expected {e:.5g} "
                  f"(rel. diff {rel:.2%})  [{'OK' if rel < 0.05 else 'CHECK'}]")
    print(f"  (E_K, E_t in the figures use only --pops {args.pops})")
    print("  (a mismatch here usually means --n0/--v0/--x0 or --mass differ "
          "from the run's cfg, or vz is not negligible)")


# ----------------------------------------------------------------- audit

def audit(cfg, units, args, v0_derived):
    """Compare the cfg with the paper's Sec. 5.2 setup, in the paper's units."""
    g = lambda sec, key, cast=float, d=None: cfg_get(cfg, sec, key, cast, d)
    proj = g("", "project", str, "?")
    Nx = g("gridbuilder", "x_length", int)
    Lx = g("gridbuilder", "x_max") - g("gridbuilder", "x_min")
    dx = Lx/Nx
    rows = []

    def add(name, got, paper, note=""):
        ok = abs(got - paper) <= 1e-3*abs(paper)
        rows.append((name, f"{got:.5g}", f"{paper:.5g}",
                     "OK" if ok else "DIFF", note))

    add("Nx", Nx, 128)
    add("Lx/x0 (= 2pi/k)", Lx/units.x0, 2*np.pi/args.k)
    add("lambda (norm. Debye length)", units.lam, 1.0)
    add("c/v0", units.c, 1.0)
    add("t_max/t0", g("gridbuilder", "t_max")/units.t0, 50.0)
    dt = g("gridbuilder", "dt")
    add("C = dt*4.14*v0/dx", dt*4.14*units.v0/dx, 1.0,
        "paper: dt = C*dx/vym with vym = 4.14")
    for ax, paper in (("x", 1.2), ("y", 4.14)):
        add(f"v{ax}max/v0", g("electron_vspace", f"v{ax}_max")/units.v0, paper)
        add(f"N_v{ax} (cells, WID=4)",
            4*g("electron_vspace", f"v{ax}_length", int), 128)
    if proj == "MultiPeak":
        add("k*x0 = 2pi*x0/lambda", 2*np.pi*units.x0/g("MultiPeak", "lambda"),
            args.k)
        add("alpha0 = dBz/B0", g("MultiPeak", "dBz")/units.B0, args.alpha0)
        Tx, Ty, Tz = (g("electron_MultiPeak", f"T{c}") for c in "xyz")
        if not v0_derived:
            add("beta = 2*Tx/T0", 2*Tx/(M_E*units.v0**2/K_B), args.beta)
        add("Tr = Ty/Tx", Ty/Tx, args.tr)
        add("Tz/Tx (paper is 1d2v, no vz)", Tz/Tx, 1.0,
            "1d3v run; vz decouples, f(vx,vy) is the vz-marginal")
        rows.append(("initial Bz", "dBz*cos(kx)", "alpha0*sin(kx)", "DIFF",
                     "shift by L/4: paper's x=L/2 is x=L/4 here"))
    w = max(len(r[0]) for r in rows)
    print("\n=== cfg vs paper (Sec. 5.2) ===")
    print(f"  {'quantity':<{w}}  {'cfg':>14}  {'paper':>14}")
    for name, got, paper, st, note in rows:
        print(f"  {name:<{w}}  {got:>14}  {paper:>14}  {st:<4} {note}")
    if v0_derived:
        print("  (v0 was derived from Tx assuming the paper's beta, so the "
              "beta row is omitted; c/v0 = 1 is the independent check)")
    ions = [q for q in cfg.get(("", "ParticlePopulations"), []) if q != "electron"]
    for q in ions:
        mr = cfg_masses(cfg)[q]
        Ti = g(f"{q}_MultiPeak", "Tx", float, float("nan"))
        print(f"  note: '{q}' is a Vlasov population (m/m_e = {mr:g}, "
              f"T/T0 = {Ti/(M_E*units.v0**2/K_B):.3g}); the paper uses a "
              "uniform static ion background")
    S = g("io", "system_write_distribution_stride", int, 0)
    cells = [i for i in range(1, Nx + 1) if S > 0 and i % S == 0]
    target = int(np.floor(args.x_frac*Nx + 1e-9)) + 1
    print(f"  VDFs written in cellids {cells if len(cells) <= 8 else str(cells[:8]) + '...'}"
          f"; Fig. 4 target x/L = {args.x_frac:g} -> cellid {target}: "
          f"{'written' if target in cells else 'NOT WRITTEN'}")


# ------------------------------------------------- t=0 neutrality check

def t0_check(f, cfg, units, args):
    """Densities and net charge of the initial state, per cell."""
    print("\n=== t=0 density / neutrality check (first file) ===")
    t = f.read_parameter("time")/units.t0
    if t != 0:
        print(f"  note: first file is at t/t0 = {t:g}, not 0")
    order = np.argsort(np.asarray(f.read_variable("cellid")))
    qnet = np.zeros(len(order))
    have_q = cfg is not None
    for pop in populations(f):
        n = sorted_vg(f, f"{pop}/vg_rho", order)
        ref = (cfg_get(cfg, f"{pop}_MultiPeak", "rho") if cfg else None) or units.n0
        print(f"  {pop:<10} n/n_cfg: mean {n.mean()/ref:.6f}, "
              f"min {n.min()/ref:.6f}, max {n.max()/ref:.6f}  (n_cfg = {ref:g})")
        q = cfg_get(cfg, f"{pop}_properties", "charge") if cfg else None
        if q is None:
            have_q = False
        else:
            qnet += q*n
    tol = 0.1*args.alpha0
    if have_q:
        r = qnet/units.n0
        bad = np.abs(r).max() > tol
        print(f"  net charge sum(q n)/(e n0): mean {r.mean():.3e}, "
              f"max|.| {np.abs(r).max():.3e}  "
              f"[{'CHECK' if bad else 'OK'}: want << alpha0 = {args.alpha0:g}; "
              f"threshold {tol:g}]")
    if "vg_rhoq" in f.get_variables():
        rq = sorted_vg(f, "vg_rhoq", order)/(Q_E*units.n0)
        print(f"  vg_rhoq/(e n0):           mean {rq.mean():.3e}, "
              f"max|.| {np.abs(rq).max():.3e}")
    for pop in populations(f):
        if f"{pop}/vg_blocks" in f.get_variables():
            print(f"  {pop:<10} velocity blocks (all cells): "
                  f"{int(np.asarray(f.read_variable(f'{pop}/vg_blocks')).sum())}")


# ----------------------------------------------- block count vs mass loss

def block_report(run, label):
    """Does the per-population particle number drop when blocks are removed?"""
    t = run["t"]
    print(f"\n=== block count vs particle number [{label}] ===")
    for key in sorted(k for k in run if k.startswith("blk:")):
        pop = key[4:]
        nb, L = run[key], run[f"L1:{pop}"]
        if np.all(np.isnan(nb)):
            print(f"  {pop}: no {pop}/vg_blocks in the files "
                  "(add 'output = populations_vg_blocks')")
            continue
        dL = np.diff(L)
        dN = np.diff(nb)
        print(f"  {pop}: blocks {int(nb[0])} -> {int(nb[-1])}; "
              f"particle number {L[0]:.6g} -> {L[-1]:.6g} "
              f"(relative change {(L[-1] - L[0])/L[0]:+.3e})")
        for name, m in (("blocks removed", dN < 0), ("blocks unchanged", dN == 0),
                        ("blocks added", dN > 0)):
            if m.any():
                print(f"    output intervals with {name:<17}: {m.sum():4d}, "
                      f"mean dL1/L1 = {dL[m].mean()/L[0]:+.3e}, "
                      f"sum = {dL[m].sum()/L[0]:+.3e}")
    print("  (if the loss is pruning, the sum is concentrated in the intervals "
          "where blocks are removed)")


def plot_blocks(runs, labels, outpath):
    n = len(runs)
    fig, axs = plt.subplots(2, n, figsize=(6*n, 7), squeeze=False, sharex="col")
    cols = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    for j, (r, lab) in enumerate(zip(runs, labels)):
        a, b = axs[0, j], axs[1, j]
        for i, key in enumerate(sorted(k for k in r if k.startswith("L1:"))):
            pop = key[3:]
            a.plot(r["t"], (r[key] - r[key][0])/r[key][0], c=cols[i % 4],
                   label=pop)
            nb = r.get(f"blk:{pop}")
            if nb is not None and not np.all(np.isnan(nb)):
                b.plot(r["t"], nb - nb[0], c=cols[i % 4], label=pop)
        a.set_yscale("symlog", linthresh=1e-12)
        a.set(title=lab, ylabel=r"$(L_1(t)-L_1(0))/L_1(0)$, signed")
        b.set(xlabel=r"$t$", ylabel="velocity blocks, change since t=0")
        for ax in (a, b):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


# ------------------------------------------- momentum / energy budget

def budget_parts(r, kin_comps):
    """Signed changes since t=0 of each part of Px, Py and E_t.

    Uses every population with a known mass (not only --pops), so 'sum' is
    the full particle + field total.
    """
    pops = sorted({k.split(":")[1] for k in r if k.startswith("P:")})
    d = lambda y: y - y[0]
    out = {}
    for name, comp in (("Px", "x"), ("Py", "y")):
        parts = {p: d(r[f"P:{p}:{comp}"]) for p in pops}
        parts["field"] = d(r[f"Pf:{comp}"])
        out[name] = parts
    parts = {p: d(sum(r[f"K:{p}:{'xyz'[c]}"] for c in kin_comps)) for p in pops}
    parts["E field"] = d(r["W:E"])
    parts["B field"] = d(r["W:B"])
    out["E_t"] = parts
    for q in out.values():
        q["sum"] = sum(q.values())
    return out


def budget_report(r, label, kin_comps, times=(10.0, 25.0, None)):
    b = budget_parts(r, kin_comps)
    t = r["t"]
    print(f"\n=== momentum / energy budget, signed change since t=0 [{label}] ===")
    for name, parts in b.items():
        print(f"  {name}:")
        hdr = "".join(f"{k:>13}" for k in parts)
        print(f"    {'t':>7}{hdr}")
        for tt in times:
            j = len(t) - 1 if tt is None else int(np.argmin(np.abs(t - tt)))
            row = "".join(f"{v[j]:13.3e}" for v in parts.values())
            print(f"    {t[j]:7.2f}{row}")


def plot_budget(runs, labels, kin_comps, outpath):
    n = len(runs)
    fig, axs = plt.subplots(3, n, figsize=(6.5*n, 10), squeeze=False,
                            sharex="col")
    cols = {"field": "k", "E field": "tab:blue", "B field": "k", "sum": "0.5"}
    pcols = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
    for j, (r, lab) in enumerate(zip(runs, labels)):
        b = budget_parts(r, kin_comps)
        for i, (name, parts) in enumerate(b.items()):
            ax = axs[i, j]
            k = 0
            for key, y in parts.items():
                if key in cols:
                    kw = dict(c=cols[key], lw=3 if key == "sum" else 1.5,
                              alpha=0.6 if key == "sum" else 1,
                              ls="--" if key == "E field" else "-")
                else:
                    kw = dict(c=pcols[k % len(pcols)], lw=1.5)
                    k += 1
                ax.plot(r["t"], y, label=key, **kw)
            ax.set_yscale("symlog", linthresh=1e-12)
            ax.set(ylabel={"Px": r"$\Delta P_x$", "Py": r"$\Delta P_y$",
                           "E_t": r"$\Delta E_t$"}[name] + ", signed")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        axs[0, j].set_title(lab)
        axs[-1, j].set_xlabel(r"$t$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


# --------------------------------------------------- work (J.E) budget

def cumtrapz(y, t):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5*(y[1:] + y[:-1])*np.diff(t))
    return out


def work_parts(r):
    """Kinetic and field energy changes against the work int int J.E dx dt.

    Per population: residual = dK - work (all three components: the magnetic
    force moves energy between components but does no work). Fields:
    residual = d(W_E + W_B) + total work. The residuals add up to the change of
    the total energy, so they say where an energy error arises. 'fg' uses
    fg_e (E^{k+1}), 'vol' uses vg_e_vol (E^{k+theta}).
    """
    pops = sorted({k.split(":")[1] for k in r if k.startswith("JE:")})
    if not pops:
        return None
    t = r["t"]
    out = {"t": t, "pops": pops}
    for src, key in (("fg", "JE"), ("vol", "JEv")):
        if any(np.all(np.isnan(r[f"{key}:{p}"])) for p in pops):
            continue
        tot_work = 0.0
        res = {}
        for p in pops:
            dK = sum(r[f"K:{p}:{ax}"] - r[f"K:{p}:{ax}"][0] for ax in "xyz")
            w = cumtrapz(r[f"{key}:{p}"], t)
            res[p] = (dK, w, dK - w)
            tot_work = tot_work + w
        dW = (r["W:E"] + r["W:B"]) - (r["W:E"][0] + r["W:B"][0])
        res["field"] = (dW, -tot_work, dW + tot_work)
        out[src] = res
    return out


def work_report(r, label, times=(10.0, 25.0, None)):
    W = work_parts(r)
    if W is None:
        print(f"\n[{label}] work diagnostic skipped: species charges unknown (no cfg)")
        return
    t = W["t"]
    print(f"\n=== energy vs work int J.E, signed change since t=0 [{label}] ===")
    print("  residual = change - work; the residuals add up to the total energy change")
    for src in ("fg", "vol"):
        if src not in W:
            print(f"  ({'vg_e_vol' if src == 'vol' else 'fg_e'} not in output: '{src}' skipped)")
            continue
        print(f"  E from {'fg_e (E^k+1)' if src == 'fg' else 'vg_e_vol (E^k+theta)'}:")
        print(f"    {'t':>7}" + "".join(f"{p + ' dK':>14}{'work':>11}{'resid':>11}" for p in W["pops"])
              + f"{'field dW':>12}{'-work':>11}{'resid':>11}")
        for tt in times:
            j = len(t) - 1 if tt is None else int(np.argmin(np.abs(t - tt)))
            row = f"    {t[j]:7.2f}"
            for p in W["pops"] + ["field"]:
                a, b, c = W[src][p]
                row += f"{a[j]:+14.3e}{b[j]:+11.3e}{c[j]:+11.3e}" if p != "field" else \
                       f"{a[j]:+12.3e}{b[j]:+11.3e}{c[j]:+11.3e}"
            print(row)
    print("  (an electron residual of the size of the energy error points at the Vlasov "
          "update; a field residual at the field update. Outputs are 0.1 t0 apart, so "
          "the time integral itself has a small error: compare 'fg' with 'vol'.)")


def plot_work(runs, labels, outpath):
    n = len(runs)
    fig, axs = plt.subplots(2, n, figsize=(6.5*n, 7.5), squeeze=False, sharex="col")
    cols = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
    for j, (r, lab) in enumerate(zip(runs, labels)):
        W = work_parts(r)
        a, b = axs[0, j], axs[1, j]
        a.set_title(lab)
        if W is None:
            a.text(0.5, 0.5, "charges unknown", transform=a.transAxes, ha="center")
            continue
        t = W["t"]
        for src, ls in (("fg", "-"), ("vol", "--")):
            if src not in W:
                continue
            for i, p in enumerate(W["pops"]):
                a.plot(t, W[src][p][2], ls, c=cols[i % 4], label=f"{p} ({src})")
            a.plot(t, W[src]["field"][2], ls, c="k", label=f"field ({src})")
        tot = sum(W[next(iter(k for k in ("fg", "vol") if k in W))][p][2]
                  for p in W["pops"] + ["field"])
        a.plot(t, tot, c="0.5", lw=3, alpha=0.6, label="total energy change")
        a.set_yscale("symlog", linthresh=1e-12)
        a.set(ylabel="residual: change - work, signed")
        for src, ls in (("fg", "-"), ("vol", "--")):
            if src in W and "electron" in W[src]:
                dK, w, _ = W[src]["electron"]
                if src == "fg":
                    b.plot(t, dK, c="tab:red", lw=2.5, alpha=0.5, label="electron dK")
                b.plot(t, w, ls, c="k", lw=1, label=f"electron work ({src})")
        b.set(xlabel=r"$t$", ylabel="electron energy change")
        for ax in (a, b):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


# ------------------------------------- in-solver acceleration energy budget

def acc_parts(r):
    """Exact decomposition of the energy change, from the in-solver accumulators.

    Per population: acceleration remap error = actual - exact energy change of the
    acceleration (what the remap made minus what its velocity map v -> A v + b makes),
    and translation + block adjustment = total change - actual acceleration change.
    Particle-field coupling = field energy change + exact acceleration change (the
    work the exact maps did). They add up to the total change, with no time
    integration. Kinetic energies use all three components.
    """
    pops = [k.split(":")[1] for k in r if k.startswith("AccA:")]
    pops = sorted(p for p in pops if not np.all(np.isnan(r[f"AccA:{p}"])))
    if not pops:
        return None
    d = lambda a: a - a[0]
    out = {"t": r["t"], "pops": pops}
    coupling = d(r["W:E"] + r["W:B"])
    total = d(r["W:E"] + r["W:B"])
    for p in pops:
        dK = sum(d(r[f"K:{p}:{ax}"]) for ax in "xyz")
        A, X = d(r[f"AccA:{p}"]), d(r[f"AccX:{p}"])
        out[p] = {"remap": A - X, "transl": dK - A}
        coupling = coupling + X
        total = total + dK
    out["coupling"] = coupling
    out["total"] = total
    return out


def acc_report(r, label, times=(10.0, 25.0, None)):
    D = acc_parts(r)
    if D is None:
        return
    t = D["t"]
    print(f"\n=== in-solver energy budget, signed change since t=0 [{label}] ===")
    print("  remap = acceleration error (actual - exact); transl = translation + block adjustment;")
    print("  coupling = field energy change + exact acceleration change; these add up to the total")
    head = "".join(f"{p + ' remap':>16}{p + ' transl':>16}" for p in D["pops"])
    print(f"    {'t':>7}{head}{'coupling':>12}{'total':>12}{'sum-total':>11}")
    for tt in times:
        j = len(t) - 1 if tt is None else int(np.argmin(np.abs(t - tt)))
        row = "".join(f"{D[p]['remap'][j]:+16.3e}{D[p]['transl'][j]:+16.3e}" for p in D["pops"])
        parts = sum(D[p]["remap"][j] + D[p]["transl"][j] for p in D["pops"]) + D["coupling"][j]
        print(f"    {t[j]:7.2f}{row}{D['coupling'][j]:+12.3e}{D['total'][j]:+12.3e}{parts - D['total'][j]:+11.1e}")


def plot_acc(runs, labels, outpath):
    parts = [acc_parts(r) for r in runs]
    if all(D is None for D in parts):
        return
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6.5*n, 4.5), squeeze=False)
    cols = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
    for ax, D, lab in zip(axs[0], parts, labels):
        ax.set_title(lab)
        if D is None:
            ax.text(0.5, 0.5, "no populations_vg_acc_energy", transform=ax.transAxes, ha="center")
            continue
        for i, p in enumerate(D["pops"]):
            ax.plot(D["t"], D[p]["remap"], c=cols[i % 4], label=f"{p}: acceleration remap error")
            ax.plot(D["t"], D[p]["transl"], ":", c=cols[i % 4], label=f"{p}: translation + block adjustment")
        ax.plot(D["t"], D["coupling"], c="k", label="particle-field coupling")
        ax.plot(D["t"], D["total"], c="0.5", lw=3, alpha=0.6, label="total energy change")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set(xlabel=r"$t$", ylabel="signed energy change")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


# -------------------------------------------------------------- plotting

STYLES = ["-", "--", ":", "-."]


# A growth fit is only reported if E_Bz is above round-off and actually grows.
FIT_MIN_REL = 1e-20   # min of max(E_Bz in window)/E_K(0)
FIT_MIN_GROWTH = 10.0  # min growth factor of the fitted line over the window


def positive(y):
    # Log axes cannot show <= 0; leave gaps rather than clip to a floor,
    # so the plotted level is never misleading.
    return np.where(y > 0, y, np.nan)


def plot_fig2(runs, labels, args, outpath):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (r, lab) in enumerate(zip(runs, labels)):
        ls = STYLES[i % len(STYLES)]
        a.semilogy(r["t"], positive(r["E_E"]), ls, c="tab:blue",
                   label=rf"$\mathcal{{E}}_E$ {lab}")
        a.semilogy(r["t"], positive(r["E_Bz"]), ls, c="tab:red",
                   label=rf"$\mathcal{{E}}_{{B_z}}$ {lab}")
        b.plot(r["t"], r["E_K"], ls, c="k", label=rf"$\mathcal{{E}}_K$ {lab}")
        b.plot(r["t"], r["E_Kx"], ls, c="tab:green",
               label=rf"$\mathcal{{E}}_{{K_x}}$ {lab}")
        b.plot(r["t"], r["E_Ky"], ls, c="tab:orange",
               label=rf"$\mathcal{{E}}_{{K_y}}$ {lab}")

        fit = fit_growth(r["t"], r["E_Bz"],
                         tuple(args.fit) if args.fit else None)
        if fit is None:
            print(f"[{lab}] growth fit skipped: <5 points in window or "
                  "E_Bz <= 0 there (use --fit T1 T2)")
        else:
            s, icpt, (t1, t2) = fit
            win = (r["t"] >= t1) & (r["t"] <= t2)
            level = r["E_Bz"][win].max()/abs(r["E_K"][0])
            growth = np.exp(s*(t2 - t1))
            if level < FIT_MIN_REL:
                print(f"[{lab}] growth fit skipped: E_Bz is at round-off "
                      f"level (max/E_K(0) = {level:.1e} < {FIT_MIN_REL:g})")
            elif growth < FIT_MIN_GROWTH:
                print(f"[{lab}] growth fit skipped: E_Bz grows only by "
                      f"{growth:.3g}x over t in [{t1:.3g}, {t2:.3g}] "
                      f"(< {FIT_MIN_GROWTH:g}x)")
            else:
                tt = np.linspace(t1, t2, 20)
                a.semilogy(tt, np.exp(icpt + s*tt), "k", lw=2.5, alpha=0.5)
                print(f"[{lab}] E_Bz fit on t in [{t1:.3g}, {t2:.3g}]: "
                      f"energy rate = {s:.4f}, amplitude rate = {s/2:.4f} "
                      f"(paper quotes 0.23; its convention is not stated)")
        if args.gamma_ref is not None and i == 0:
            ok = np.flatnonzero(r["E_Bz"] > FIT_MIN_REL*abs(r["E_K"][0]))
            if ok.size:
                t = r["t"]
                j = ok[0]
                ref = r["E_Bz"][j]*np.exp(2*args.gamma_ref*(t - t[j]))
                a.semilogy(t, ref, "k:", lw=1,
                           label=rf"$e^{{2\gamma t}}$, $\gamma$={args.gamma_ref}")
            else:
                print(f"[{lab}] --gamma-ref line skipped: E_Bz is never above "
                      "round-off level")
    a.set(xlabel=r"$t$", ylabel="field energy", title="(a) field energy")
    b.set(xlabel=r"$t$", ylabel="kinetic energy", title="(b) kinetic energy")
    for ax in (a, b):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


def plot_fig3(runs, labels, outpath):
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 4.6), squeeze=False)
    for ax, r, lab in zip(axs[0], runs, labels):
        t = r["t"]
        for key, kw, tex in (
                ("L1", dict(c="red", ls="-"), r"|\Delta L_1|"),
                ("Px", dict(c="lime", ls="-"), r"|\Delta P_x|"),
                ("Py", dict(c="k", ls="--"), r"|\Delta P_y|"),
                ("E_t", dict(c="olive", ls="-"), r"|\Delta E_t|")):
            ax.semilogy(t, np.maximum(np.abs(r[key] - r[key][0]), FLOOR),
                        label=f"${tex}$", **kw)
        # Red dashed/dotted: |dL1| of each population separately.
        for ls, key in zip(("--", ":", "-.", (0, (5, 1))),
                           sorted(k for k in r if k.startswith("L1:"))):
            ax.semilogy(t, np.maximum(np.abs(r[key] - r[key][0]), FLOOR),
                        c="red", ls=ls, label=rf"$|\Delta L_1|$ {key[3:]}")
        ax.set(ylim=(1e-18, 1e-2), xlim=(0., 50.), xlabel=r"$t$",
               ylabel="Conservation error", title=lab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


def other_series_files(bulk_file, t_target, units):
    """Files of the run's other output series (e.g. a separate, sparser distribution
    series written alongside bulk*.vlsv), nearest to t_target (in t0) first."""
    directory = os.path.dirname(bulk_file) or "."
    out = []
    for fn in sorted(glob.glob(os.path.join(directory, "*.vlsv"))):
        if os.path.basename(fn).startswith("bulk"):
            continue
        try:
            out.append((open_vlsv(fn).read_parameter("time")/units.t0, fn))
        except Exception:
            continue
    return [(fn, t) for t, fn in sorted(out, key=lambda p: abs(p[0] - t_target))]


def plot_fig4(runs, labels, units, args, outpath):
    panels = []
    for r, lab in zip(runs, labels):
        j = int(np.argmin(np.abs(r["t"] - args.t_vdf)))
        t_used = r["t"][j]
        f = open_vlsv(r["files"][j])
        cid, xc, nx = snapshot_cell(f, args.x_frac)
        try:
            vx, vy, fxy, dens = vdf_xy(f, args.vdf_pop, cid, units)
        except Exception:
            # No VDF in the bulk series (analysator raises IndexError for a file without any): take it from the nearest file of another series that
            # has one (e.g. distributions*.vlsv).
            for fn, t_used in other_series_files(r["files"][j], args.t_vdf, units):
                f = open_vlsv(fn)
                try:
                    vx, vy, fxy, dens = vdf_xy(f, args.vdf_pop, cid, units)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"cell {cid} has no VDF for {args.vdf_pop!r} in any output series")
        n_cell = np.asarray(f.read_variable(f"{args.vdf_pop}/vg_rho",
                                            cellids=cid)).ravel()[0]
        print(f"[{lab}] Fig.4 uses t/t0 = {t_used:.4g} (asked {args.t_vdf}), "
              f"cell {cid} at x/L = {(xc - f.get_spatial_mesh_extent()[0])/(f.get_spatial_mesh_extent()[3] - f.get_spatial_mesh_extent()[0]):.4f}; "
              f"int f dvx dvy / rho = {dens/n_cell:.4f}")
        panels.append((lab, vx, vy, fxy))
    vmax = args.vmax or max(p[3].max() for p in panels)
    levels = np.linspace(0, vmax, 101)
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.6*n, 4.8), squeeze=False)
    for ax, (lab, vx, vy, fxy) in zip(axs[0], panels):
        cs = ax.contourf(vx, vy, fxy.T, levels=levels, cmap="jet",
                         extend="max")
        ax.set(xlim=(-args.vlim, args.vlim), ylim=(-args.vlim, args.vlim),
               xlabel=r"$v_x$", ylabel=r"$v_y$", title=lab, aspect="equal")
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks(np.round(np.linspace(0, vmax, 5), 2))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("saved", outpath)


# ------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="one directory per run")
    p.add_argument("--labels", "--label", action="append",
                   help="label of one run; repeat once per directory, in order")
    p.add_argument("--n0", type=float, default=None,
                   help="reference number density [m^-3] (default: electron "
                        "MultiPeak rho from the embedded cfg)")
    p.add_argument("--v0", default=None,
                   help="reference velocity [m/s] or 'c'. Default: from the "
                        "cfg's electron Tx via Tx = (beta/2) m v0^2/kB")
    p.add_argument("--x0", type=float, default=None,
                   help="reference length [m]; default gives lambda = 1")
    p.add_argument("--pops", nargs="+", default=["electron"],
                   help="populations in E_K, L1, P (paper: electrons only)")
    p.add_argument("--mass", nargs="+", default=None,
                   help="NAME=mass in electron masses (default: from cfg)")
    p.add_argument("--no-cfg", action="store_true",
                   help="ignore the cfg embedded in the vlsv files")
    p.add_argument("--x-frac", type=float, default=None,
                   help="Fig. 4 position x/L (default 0.25 for MultiPeak, "
                        "whose cos(kx) is the paper's sin(kx) shifted by L/4; "
                        "else 0.5)")
    p.add_argument("--kz", action="store_true",
                   help="include vz kinetic energy in E_K and E_t (paper: x,y)")
    p.add_argument("--k", type=float, default=1.25)
    p.add_argument("--alpha0", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--tr", type=float, default=12.0)
    p.add_argument("--t-vdf", type=float, default=27.0,
                   help="time [t0] of the Fig. 4 snapshot")
    p.add_argument("--vdf-pop", default="electron")
    p.add_argument("--vlim", type=float, default=1.2,
                   help="half-width of the Fig. 4 velocity window [v0]")
    p.add_argument("--vmax", type=float, default=None,
                   help="colour-scale maximum in Fig. 4 (default: data max)")
    p.add_argument("--fit", nargs=2, type=float, metavar=("T1", "T2"),
                   default=[5.0, 25.0],
                   help="growth-fit window in t0 units (default: 5 25)")
    p.add_argument("--gamma-ref", type=float, default=None,
                   help="draw exp(2*gamma*t) reference in Fig. 2a")
    p.add_argument("--outdir", default=None)
    p.add_argument("--no-vdf", action="store_true", help="skip Fig. 4")
    return p.parse_args()


def main():
    args = parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(d))
                             for d in args.dirs]
    if len(labels) != len(args.dirs):
        sys.exit("--labels needs one entry per directory")

    tfs = [find_files(d) for d in args.dirs]
    f0 = open_vlsv(tfs[0][1][0])
    cfg = None if args.no_cfg else read_cfg(f0)
    if cfg is None and not args.no_cfg:
        print("no CONFIG in the vlsv; --n0, --v0 and --mass must be given")

    masses = cfg_masses(cfg) if cfg else {}
    charges = cfg_charges(cfg) if cfg else {}
    for item in args.mass or []:
        name, val = item.split("=")
        masses[name] = float(val)
    for pop in args.pops:
        if pop not in masses:
            sys.exit(f"no mass for {pop!r}: pass --mass {pop}=<m/m_e>")

    n0 = args.n0
    if n0 is None and cfg:
        n0 = cfg_get(cfg, "electron_MultiPeak", "rho")
    if n0 is None:
        sys.exit("need --n0")

    v0_derived = args.v0 is None
    if v0_derived:
        Tx = cfg_get(cfg, "electron_MultiPeak", "Tx") if cfg else None
        if Tx is None:
            sys.exit("need --v0")
        v0 = np.sqrt(2*K_B*Tx/(M_E*args.beta))
    else:
        v0 = C_LIGHT if str(args.v0).lower() == "c" else float(args.v0)
    units = Units(n0, v0, args.x0)
    print(f"n0 = {n0:g} m^-3, v0 = {v0:.6g} m/s = {v0/C_LIGHT:.6g} c"
          f"{' (derived)' if v0_derived else ''}")

    if args.x_frac is None:
        proj = cfg_get(cfg, "", "project", str, "") if cfg else ""
        args.x_frac = 0.25 if proj == "MultiPeak" else 0.5
    if cfg:
        audit(cfg, units, args, v0_derived)

    t0_check(f0, cfg, units, args)
    found = populations(f0)
    for pop in args.pops:
        if pop not in found:
            sys.exit(f"population {pop!r} not in file; found {found}")
    kin = [0, 1, 2] if args.kz else [0, 1]
    runs = [time_series(d, tf, units, args.pops, masses, kin, charges)
            for d, tf in zip(args.dirs, tfs)]

    sanity_checks(runs[0], units, args, f0, args.pops, masses, cfg)

    outdir = args.outdir or args.dirs[0]
    os.makedirs(outdir, exist_ok=True)
    plot_fig2(runs, labels, args, os.path.join(outdir, "weibel_fig2.png"))
    plot_fig3(runs, labels, os.path.join(outdir, "weibel_fig3.png"))
    for r, lab in zip(runs, labels):
        block_report(r, lab)
    plot_blocks(runs, labels, os.path.join(outdir, "weibel_blocks.png"))
    for r, lab in zip(runs, labels):
        budget_report(r, lab, kin)
    plot_budget(runs, labels, kin, os.path.join(outdir, "weibel_budget.png"))
    for r, lab in zip(runs, labels):
        work_report(r, lab)
    plot_work(runs, labels, os.path.join(outdir, "weibel_work.png"))
    for r, lab in zip(runs, labels):
        acc_report(r, lab)
    plot_acc(runs, labels, os.path.join(outdir, "weibel_acc_energy.png"))
    if not args.no_vdf:
        try:
            plot_fig4(runs, labels, units, args,
                      os.path.join(outdir, "weibel_fig4.png"))
        except RuntimeError as e:
            print("Fig. 4 skipped:", e)


if __name__ == "__main__":
    main()
