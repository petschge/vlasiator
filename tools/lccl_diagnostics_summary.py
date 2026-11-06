#!/usr/bin/env python3
"""
Summary of lccl_diagnostics.txt (fieldsolver.diagnosticInterval > 0) for the conservation report:
Gauss residual relative to the charge-density scale (centred and consistent divergence), div B
relative to |B|/dx, and the change of the field energy. With --plot, also a figure of the three
against time.

  python3 lccl_diagnostics_summary.py RUN_DIR/lccl_diagnostics.txt [--plot diagnostics.png]

Columns of the file: step, time, rms and max of rho/eps0 - div E (centred), the same with the
consistent divergence, max sum_s |rho_s|/eps0, max |div B|, max |div B| dx_min/max|B|, W_E, W_B, W_E + W_B.
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--plot", help="write a figure to this file")
    a = p.parse_args()
    d = np.atleast_2d(np.loadtxt(a.file, comments="#"))
    # a restarted run appends to the same file: keep the last line of every step
    _, last = np.unique(d[::-1, 0], return_index=True)
    d = d[::-1][last]
    d = d[np.argsort(d[:, 0])]
    step, t = d[:, 0], d[:, 1]
    scale = np.where(d[:, 6] > 0, d[:, 6], np.nan)
    gc_rms, gc_max = d[:, 2]/scale, d[:, 3]/scale
    gd_rms, gd_max = d[:, 4]/scale, d[:, 5]/scale
    divb = d[:, 8]
    we, wb, w = d[:, 9], d[:, 10], d[:, 11]
    print(f"{len(step)} entries, steps {int(step[0])}..{int(step[-1])}, t = {t[0]:.6g} .. {t[-1]:.6g} s")
    print("Gauss residual / max sum_s |rho_s|/eps0 (largest over time):")
    print(f"  centred divergence (Eq. 45):   rms {np.nanmax(gc_rms):.3e}   max {np.nanmax(gc_max):.3e}")
    print(f"  consistent divergence:         rms {np.nanmax(gd_rms):.3e}   max {np.nanmax(gd_max):.3e}")
    print(f"max |div B| dx_min / max |B| (largest over time): {np.max(divb):.3e}")
    dw = (w - w[0])/w[0] if w[0] != 0 else w - w[0]
    print(f"field energy W_E + W_B: {w[0]:.6e} J -> {w[-1]:.6e} J, relative change {dw[-1]:+.3e}, "
          f"largest excursion {dw[np.argmax(np.abs(dw))]:+.3e}")
    print(f"  W_E: {we[0]:.6e} -> {we[-1]:.6e} J;  W_B: {wb[0]:.6e} -> {wb[-1]:.6e} J")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        ax[0].semilogy(t, gc_rms, label="rms, centred")
        ax[0].semilogy(t, gc_max, label="max, centred")
        ax[0].semilogy(t, gd_rms, "--", label="rms, consistent")
        ax[0].semilogy(t, gd_max, "--", label="max, consistent")
        ax[0].set_ylabel(r"$|\rho/\epsilon_0-\nabla\cdot E|$ / scale")
        ax[0].legend(fontsize=8)
        ax[1].semilogy(t, np.maximum(divb, 1e-300))
        ax[1].set_ylabel(r"max$|\nabla\cdot B|\,\Delta x$/max$|B|$")
        ax[2].plot(t, dw)
        ax[2].set_ylabel(r"$\Delta W_{EM}/W_{EM}(0)$")
        ax[2].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
