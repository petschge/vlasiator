#!/usr/bin/env python3
"""

Analyze output from the LandauDamping project and reproduce Figure 1 of Liu,
Cai, Cao & Lapenta 2025 Section 5.1 that shows time evolution of the
electric-field energy E_Ex = 0.5*integral(Ex^2) dx.

Usage:
    python3 compare_landau_damping.py [output_directory]

output_directory defaults to the current directory and should contain
the full bulk*.vlsv time series landau_damping.cfg writes (one at t=0,
then one every system_write_t_interval up to t_max).
"""

import sys
import glob
import os
import numpy as np
from scipy.special import wofz
from scipy.optimize import root
from scipy.signal import find_peaks

try:
    import analysator as pt
except ImportError:
    import pytools as pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Physical parameters -- must match landau_damping.cfg exactly
# ============================================================

Q_E = 1.602176634e-19          # C
M_ELECTRON = 9.1093837015e-31  # kg
K_B = 1.380649e-23             # J/K
EPS_0 = 8.8541878128e-12       # F/m

Te = 1.0e5                     # K  -- electron_LandauDamping.Temperature
ne = 1.0e6                     # m^-3 -- electron_LandauDamping.rho
alpha0 = 0.01                  # electron_LandauDamping.densityPertRelAmp
k_phys = 0.045824146786756774  # LandauDamping.k (m^-1)

lambda_D = np.sqrt(EPS_0*K_B*Te/(Q_E**2*ne))
omega_pe = np.sqrt(ne*Q_E**2/(EPS_0*M_ELECTRON))
klD = k_phys*lambda_D  # should be ~1 (the paper's own dimensionless k, given lambda=1)


# ============================================================
# Independent linear theory: solve the Landau-damping dispersion
# relation directly, rather than trusting the paper's stated gamma=-0.85.
# ============================================================

def solve_landau_dispersion(klD, guess=(1.5, -0.5)):
    """Standard 1D-Maxwellian Vlasov-Poisson dispersion relation,
    Fried-Conte convention:
        eps(k,omega) = 1 + (1/(k*lambdaD)^2) * [1 + zeta*Z(zeta)] = 0
        zeta = (omega/omega_pe) / (k*lambdaD*sqrt(2))
        Z(zeta) = i*sqrt(pi)*wofz(zeta)   [plasma dispersion function]
    Returns omega/omega_pe (complex) for the least-damped root near
    `guess`.
    """
    def Z(zeta):
        return 1j*np.sqrt(np.pi)*wofz(zeta)

    def dispersion(omega_over_wpe):
        zeta = omega_over_wpe/(klD*np.sqrt(2))
        return 1 + (1.0/klD**2)*(1 + zeta*Z(zeta))

    def eqs(v):
        om = v[0] + 1j*v[1]
        e = dispersion(om)
        return [e.real, e.imag]

    sol = root(eqs, guess, tol=1e-14)
    if not sol.success:
        raise RuntimeError(f"Dispersion-relation root find failed: {sol.message}")
    return sol.x[0] + 1j*sol.x[1]


# ============================================================
# Loading helpers
# ============================================================

def find_bulk_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "bulk*.vlsv")))
    if len(files) < 2:
        raise RuntimeError(
            f"Expected at least 2 bulk*.vlsv files in {directory!r}, found {len(files)}"
        )
    tagged = []
    for fn in files:
        f = pt.vlsvfile.VlsvReader(fn)
        t = f.read_parameter('time')
        tagged.append((t, fn))
    tagged.sort(key=lambda x: x[0])
    return tagged


def load_vg_field(f, name):
    cellids = f.read_variable('cellid')
    order = cellids.argsort()
    return f.read_variable(name)[order]


def cellid_to_x(f, Lx, Nx):
    """For this specific grid (Ny=Nz=1), DCCRG's CellID numbering runs
    sequentially along x (CellID = i, 1-indexed), so x(CellID) =
    (CellID-0.5)*dx directly"""
    cellids = f.read_variable('cellid')
    order = cellids.argsort()
    cellids_sorted = cellids[order]
    dx = Lx/Nx
    return (cellids_sorted - 0.5)*dx


def Ex_energy(f, dx):
    """E_Ex = 0.5*integral(Ex^2) dx, via a simple Riemann sum over the
    uniform, periodic x-grid"""
    fg_e = f.read_variable('fg_e')
    Ex = fg_e[..., 0].ravel()
    return 0.5*np.sum(Ex**2)*dx


# ============================================================
# Main
# ============================================================

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    tagged = find_bulk_files(directory)
    times = np.array([t for t, _ in tagged])
    files = [fn for _, fn in tagged]

    print(f"Found {len(files)} bulk files, t in [{times[0]:.4e}, {times[-1]:.4e}] s")

    f0 = pt.vlsvfile.VlsvReader(files[0])
    # Lx, Nx, dx known exactly from the CFG itself
    Lx = 2*np.pi/k_phys
    Nx = 64
    dx = Lx/Nx
    xg = cellid_to_x(f0, Lx, Nx)

    print(f"\n=== Independent linear theory (this script's own dispersion-relation solve) ===")
    print(f"k*lambda_D = {klD:.6f}  (should be ~1.0, the paper's own dimensionless k*lambda)")
    omega_root = solve_landau_dispersion(klD)
    gamma_theory = omega_root.imag  # in units of omega_pe
    omega_r_theory = omega_root.real
    print(f"omega/omega_pe = {omega_root.real:.6f} + {omega_root.imag:.6f}j")
    print(f"  => gamma_theory/omega_pe = {gamma_theory:.6f}  (paper's stated value: -0.85)")
    print(f"  => real oscillation frequency omega_r/omega_pe = {omega_r_theory:.6f}")
    print(f"     (E_Ex oscillates at 2*omega_r on top of the exp(2*gamma*t) envelope decay --")
    print(f"      period ~ {np.pi/omega_r_theory:.4f} in omega_pe*t units)")
    gamma_paper = -0.85

    # ---- t=0 sanity checks ----
    print(f"\n=== t=0 sanity checks ===")
    rho_e0 = load_vg_field(f0, 'electron/vg_rho')
    rho_e0_pred = ne*(1.0 + alpha0*np.sin(k_phys*xg))
    err_rho = np.max(np.abs(rho_e0 - rho_e0_pred))
    print(f"  electron rho(t=0) vs ne*(1+alpha0*sin(kx)): max|actual-predicted| = {err_rho:.4e} "
          f"(tol = {ne*1e-6:.4e})  [{'PASS' if err_rho < ne*1e-6 else 'FAIL'}]")

    fg_e0 = f0.read_variable('fg_e')
    Ex0 = fg_e0[..., 0].ravel()
    # That amplitude needs one correction: the discrete sample
    # points are CELL-CENTERED at x_i=(i-0.5)*dx, so k*x_i never actually
    # reaches k*x=0 (the true continuum peak of cos(kx)) -- the closest
    # sampled angle is k*x_1=pi/Nx, so the discrete max is
    # cos(pi/Nx)*(continuum amplitude), not the continuum amplitude
    # itself.
    Ex0_amp_continuum = Q_E*ne*alpha0/(EPS_0*k_phys)
    Ex0_amp_pred = Ex0_amp_continuum*np.cos(np.pi/Nx)
    Ex0_amp_actual = 0.5*(np.max(Ex0) - np.min(Ex0))
    err_amp = abs(Ex0_amp_actual - Ex0_amp_pred)
    print(f"  fg_e(t=0) amplitude vs Q*ne*alpha0/(eps0*k)*cos(pi/Nx): actual={Ex0_amp_actual:.6e}, "
          f"predicted={Ex0_amp_pred:.6e}, diff={err_amp:.4e} "
          f"(tol = {Ex0_amp_pred*1e-3:.4e})  [{'PASS' if err_amp < Ex0_amp_pred*1e-3 else 'FAIL'}]")

    # ---- Time series: E_Ex(t) ----
    print(f"\n=== Computing E_Ex(t) over {len(files)} files ===")
    E_Ex = np.array([Ex_energy(pt.vlsvfile.VlsvReader(fn), dx) for fn in files])
    E_Ex0 = E_Ex[0]
    print(f"  E_Ex(t=0) = {E_Ex0:.6e}")

    # ---- Fit the simulation's own decay rate from the oscillation envelope (local maxima)
    t_dimless = times*omega_pe  # paper's own time axis is in units of 1/omega_pe (t0)
    fit_mask = t_dimless < min(8.0, t_dimless[-1])
    peak_idx, _ = find_peaks(E_Ex[fit_mask])
    gamma_sim = None
    if len(peak_idx) >= 3:
        t_peaks = times[peak_idx]
        E_peaks = E_Ex[peak_idx]
        valid = E_peaks > 0
        if np.sum(valid) >= 3:
            # log(E_Ex) = log(E_Ex0') + 2*gamma*omega_pe*t  (E_Ex ~ exp(2*gamma*t))
            coeffs = np.polyfit(t_peaks[valid]*omega_pe, np.log(E_peaks[valid]), 1)
            gamma_sim = coeffs[0]/2.0  # in units of omega_pe
            print(f"  Simulation's own fitted decay rate (from {np.sum(valid)} envelope peaks): "
                  f"gamma_sim/omega_pe = {gamma_sim:.6f}")
            print(f"    vs gamma_theory = {gamma_theory:.6f}  "
                  f"(relative difference: {abs(gamma_sim-gamma_theory)/abs(gamma_theory)*100:.2f}%)")
    if gamma_sim is None:
        print("  Not enough clear envelope peaks found to fit an empirical decay rate "
              "(try a longer t_max or finer system_write_t_interval).")

    # ============================================================
    # Plot: reproduce Figure 1(a)
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(t_dimless, E_Ex, 'r-', linewidth=1.5, label='CSL-RME-WG (this run)')

    # Theoretical decay lines: E_Ex(t) = E_Ex0 * exp(2*gamma*omega_pe*t).
    # Normalization: fit each line's own prefactor by least-squares
    # matching log(E_Ex) over an early-to-mid time window
    if np.sum(fit_mask) >= 2 and np.all(E_Ex[fit_mask] > 0):
        logE_fit = np.log(E_Ex[fit_mask])

        def fit_prefactor(gamma):
            # log(E) = log(A) + 2*gamma*t  -> log(A) = mean(logE - 2*gamma*t)
            logA = np.mean(logE_fit - 2*gamma*t_dimless[fit_mask])
            return np.exp(logA)

        A_theory = fit_prefactor(gamma_theory)
        #A_paper = fit_prefactor(gamma_paper)
        t_line = np.linspace(t_dimless[0], t_dimless[-1], 200)
        ax.semilogy(t_line, A_theory*np.exp(2*gamma_theory*t_line), 'k--', linewidth=1.3, label=f'linear theory: ' f'$\\gamma/\\omega_{{pe}}$={gamma_theory:.4f}')
        #xJax.semilogy(t_line, A_paper*np.exp(2*gamma_paper*t_line), 'b:', linewidth=1.8, label=f"paper's stated $\\gamma/\\omega_{{pe}}$={gamma_paper:.2f}")

    if gamma_sim is not None:
        ax.plot(times[peak_idx]*omega_pe, E_Ex[peak_idx], 'g^', markersize=6,
                label=f'this run\'s own fitted envelope: $\\gamma/\\omega_{{pe}}$={gamma_sim:.4f}')

    ax.set_xlabel(r'$t \, \omega_{pe}$')
    ax.set_ylabel(r'$\mathcal{E}_{Ex} = \frac{1}{2}\int |E_x|^2\,dx$')
    ax.set_title('Landau damping (Section 5.1, Liu et al. 2025) -- CSL-RME-WG\n'
                  f'$k\\lambda_D$={klD:.3f}, $\\beta$=2, $\\alpha_0$={alpha0}', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    outpath = os.path.join(directory, 'landau_damping_fig1a.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved plot to {outpath}")

    print(f"\n{'='*60}")
    print(f"Summary: gamma/omega_pe")
    print(f"  paper's stated value          : {gamma_paper:.6f}")
    print(f"  this script's independent solve: {gamma_theory:.6f}  "
          f"(diff from paper: {abs(gamma_theory-gamma_paper):.6f}, "
          f"{abs(gamma_theory-gamma_paper)/abs(gamma_paper)*100:.2f}%)")
    if gamma_sim is not None:
        print(f"  this run's own empirical fit   : {gamma_sim:.6f}  "
              f"(diff from theory: {abs(gamma_sim-gamma_theory):.6f}, "
              f"{abs(gamma_sim-gamma_theory)/abs(gamma_theory)*100:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
