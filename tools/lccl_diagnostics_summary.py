#!/usr/bin/env python3
"""
Summary of lccl_diagnostics.txt (fieldsolver.diagnosticInterval > 0) for the conservation report:
Gauss residual relative to the charge-density scale (centred and consistent divergence), div B
relative to |B|/dx, the change of the field energy and, with fieldsolver.energyAccounting, the
total energy and its budget:
  field side      R_F - D_theta + dW_corr   (Poynting remainder/boundary flux, theta damping,
                                             Gauss correction and filters)
  coupling        sum_s Wmap_s - P_pred     (work of the push on the actual distribution minus the
                                             work the field solve assumed)
  remap           sum_s (dKacc_s - Wmap_s)  (acceleration remap and sparse mesh during it)
  translation     sum_s dKtrans_s           (translations, sparse mesh, boundaries)
which add up to the change of W_total = W_E + W_B + sum_s K_s (see lccl_energy_budget.h).
With --plot, also a figure against time.

  python3 lccl_diagnostics_summary.py RUN_DIR/lccl_diagnostics.txt [--plot diagnostics.png]

The budget columns are cumulative since the start of a run; after a restart the file continues
with a new run segment, and the budget is summarised for the last segment.
"""
import argparse
import numpy as np

OLD_NAMES = ["step", "time", "gauss_centred_rms", "gauss_centred_max", "gauss_consistent_rms",
             "gauss_consistent_max", "rho_scale", "divB_max", "divB_rel", "W_E", "W_B", "W_EM"]


def read(path):
    names = None
    rows = []
    for line in open(path):
        if line.startswith("# columns:"):
            names = line.split(":", 1)[1].split()
        elif line.strip() and not line.startswith("#"):
            rows.append([float(x) for x in line.split()])
    names = names or OLD_NAMES
    width = min(len(r) for r in rows)
    d = np.array([r[:width] for r in rows])
    return names[:width], d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--plot", help="write a figure to this file")
    a = p.parse_args()
    names, raw = read(a.file)
    col = {n: i for i, n in enumerate(names)}
    # run segments: a restart appends lines whose step does not increase
    seg_start = [0] + [i for i in range(1, len(raw)) if raw[i, 0] <= raw[i - 1, 0]]
    if len(seg_start) > 1:
        print(f"note: {len(seg_start)} run segments (restarts); the budget below is for the last one")
    d = raw[seg_start[-1]:]
    # Gauss residual and div B over the whole file, keeping the last line of every step
    _, last = np.unique(raw[::-1, 0], return_index=True)
    allsteps = raw[::-1][last]
    allsteps = allsteps[np.argsort(allsteps[:, 0])]
    t = allsteps[:, col["time"]]
    scale = np.where(allsteps[:, col["rho_scale"]] > 0, allsteps[:, col["rho_scale"]], np.nan)
    gc_rms, gc_max = allsteps[:, col["gauss_centred_rms"]]/scale, allsteps[:, col["gauss_centred_max"]]/scale
    gd_rms, gd_max = allsteps[:, col["gauss_consistent_rms"]]/scale, allsteps[:, col["gauss_consistent_max"]]/scale
    divb = allsteps[:, col["divB_rel"]]
    w = allsteps[:, col["W_EM"]]
    print(f"{len(allsteps)} entries, steps {int(allsteps[0, 0])}..{int(allsteps[-1, 0])}, t = {t[0]:.6g} .. {t[-1]:.6g} s")
    print("Gauss residual / max sum_s |rho_s|/eps0 (largest over time):")
    print(f"  centred divergence (Eq. 45):   rms {np.nanmax(gc_rms):.3e}   max {np.nanmax(gc_max):.3e}")
    print(f"  consistent divergence:         rms {np.nanmax(gd_rms):.3e}   max {np.nanmax(gd_max):.3e}")
    print(f"max |div B| dx_min / max |B| (largest over time): {np.max(divb):.3e}")
    dw = (w - w[0])/w[0] if w[0] != 0 else w - w[0]
    print(f"field energy W_E + W_B: {w[0]:.6e} J -> {w[-1]:.6e} J, relative change {dw[-1]:+.3e}")

    budget = "W_total" in col
    if budget:
        species = [n[2:] for n in names if n.startswith("K_")]
        first, lastrow = d[0], d[-1]
        get = lambda row, n: row[col[n]]
        diff = lambda n: get(lastrow, n) - get(first, n)
        W0 = get(first, "W_total")
        dW = diff("W_total")
        field = diff("R_F") - diff("D_theta") + diff("dW_corr")
        coupling = sum(diff(f"Wmap_{s}") for s in species) - diff("P_pred")
        remap = sum(diff(f"dKacc_{s}") - diff(f"Wmap_{s}") for s in species)
        trans = sum(diff(f"dKtrans_{s}") for s in species)
        exch = abs(diff("P_pred"))
        print(f"\ntotal energy W_E + W_B + sum K: {W0:.9e} J -> {get(lastrow, 'W_total'):.9e} J "
              f"(steps {int(first[0])}..{int(lastrow[0])}): change {dW:+.4e} J = {dW/W0:+.3e} of the total")
        print(f"  energy exchanged with the particles (|P_pred|): {exch:.4e} J")
        rel = lambda x: f"{x:+.4e} J  ({x/W0:+.2e} of total" + (f", {x/exch:+.2e} of exchanged)" if exch > 0 else ")")
        print(f"  field side   (R_F - D_theta + dW_corr): {rel(field)}")
        print(f"     R_F {diff('R_F'):+.3e}, D_theta {diff('D_theta'):+.3e}, dW_corr {diff('dW_corr'):+.3e}")
        print(f"  coupling     (sum Wmap - P_pred):        {rel(coupling)}")
        print(f"  remap        (sum dKacc - Wmap):         {rel(remap)}")
        for s_ in species:
            print(f"     {s_}: {diff(f'dKacc_{s_}') - diff(f'Wmap_{s_}'):+.3e}")
        print(f"  translation  (sum dKtrans):              {rel(trans)}")
        for s_ in species:
            print(f"     {s_}: {diff(f'dKtrans_{s_}'):+.3e}")
        print(f"  sum of the terms {field + coupling + remap + trans:+.6e} J, change of the total {dW:+.6e} J "
              f"(closure {field + coupling + remap + trans - dW:+.2e} J)")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 4 if budget else 3
        fig, ax = plt.subplots(n, 1, figsize=(7, 2.6*n), sharex=True)
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
        if budget:
            td = d[:, col["time"]]
            W0 = d[0, col["W_total"]]
            c = lambda n: (d[:, col[n]] - d[0, col[n]])/W0
            sp = [nm[2:] for nm in names if nm.startswith("K_")]
            ax[3].plot(td, c("W_total"), "k", label="total")
            ax[3].plot(td, c("R_F") - c("D_theta") + c("dW_corr"), label="field side")
            ax[3].plot(td, sum(c(f"Wmap_{s}") for s in sp) - c("P_pred"), label="coupling")
            ax[3].plot(td, sum(c(f"dKacc_{s}") - c(f"Wmap_{s}") for s in sp), label="remap")
            ax[3].plot(td, sum(c(f"dKtrans_{s}") for s in sp), label="translation")
            ax[3].set_ylabel(r"$\Delta W_{total}/W_{total}(0)$")
            ax[3].legend(fontsize=8)
        ax[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
