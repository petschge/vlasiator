#!/usr/bin/env python3
"""
SI-unit derivation and .cfg generator for a DOUBLE Harris-sheet variant of
the GEM reconnection challenge (Liu, Cai, Cao & Lapenta 2025, JCP 528:113840,
Section 5.4), for Vlasiator's AP (CSL-RME) field solver and the
Harris.doubleSheet=1 extension (see that patch's own commit message for the
full derivation of the double-sheet B-field/density/drift/perturbation
formulas this script's outputs feed into).

----------------------------------------------------------------------------
WHY THIS VARIANT EXISTS
----------------------------------------------------------------------------
The single-sheet GEM setup (derive_gem.py) needs a wall boundary (or an open
one) at y=+-Ly, and debugging a persistent instability there kept raising the
question of whether ConductingWall itself was implicated. Rather than keep
swapping wall types (already tried: ConductingWall vs Outflow on y, periodic
vs Outflow on x -- neither changed the failure), this variant sidesteps the
question entirely: TWO Harris sheets of opposite polarity, in a domain twice
as tall as the original, fully periodic in both x and y. No wall of any
kind. If the instability survives this, ConductingWall is unambiguously
cleared; if it doesn't, that is itself strong evidence pointing back at the
wall boundary.

Sheets sit at y=+-L/2, where L is this domain's own (doubled) y half-extent
-- which places each sheet exactly where the single-sheet domain's own WALL
used to be (L/2 = Ly_single, the original domain's y half-extent), so the
spatial scale between a sheet and its nearest "edge" (now a sheet via
periodicity, not a wall) is preserved rather than arbitrarily rescaled.

All the single-sheet physics derivation below (lambda_i, B0, v_ts, v_ds,
pressure balance, etc.) is UNCHANGED -- it still describes the local
structure near either sheet exactly, since far from the other sheet its own
tanh term is saturated (see the Harris.cpp patch's own setProjectBField
comment). Only the domain geometry, boundary conditions, and the
Harris.doubleSheet flag differ from derive_gem.py; search for "DOUBLE-SHEET"
below for every place this script actually departs from it.

----------------------------------------------------------------------------
GEOMETRY
----------------------------------------------------------------------------
"double" (default): the two-sheet, fully periodic box described above.
"single": the paper's own geometry, one sheet at y=0 between conducting walls at
y=+-6.4 lambda_i, periodic in x -- half the cells. Vlasiator makes the two outermost
cell rows at each y face the wall's boundary layers, so the domain is 2 rows taller
at each end and the walls sit at the interior faces. Harris.cpp takes the
perturbation's Ly from the domain, so its By vanishes at the domain edge, 2 rows
beyond each wall, rather than on the walls.

----------------------------------------------------------------------------
SCALE CHOICE (n0, v_a)
----------------------------------------------------------------------------
The paper's setup is dimensionless (lambda_i, 1/w_ci, v_a, B0, m_i, n0) with
m_i/m_e = 25, T_i/T_e = 5 and c/v_a = 20. Vlasiator runs in SI with the true
c, so n0 is free and c/v_a = 20 fixes v_a = c/20 (T ~ 1e10 K). The paper's
scheme and Vlasiator's Vlasov solver are both non-relativistic, so this is
the paper's model. The earlier c/v_a = 749 (v_a = 4e5 m/s) put w_pe dt at ~37,
where the Gauss-corrected field reaches the particles with weight
1/(1+(theta w_pe dt)^2) ~ 2e-3: nothing held electrons and ions together.

----------------------------------------------------------------------------
STAGES (MODE, FROZEN_FIELDS)
----------------------------------------------------------------------------
1. MODE="1d", FROZEN_FIELDS=True: x-invariant sheet, no perturbation, B
   frozen and E=0. The loaded VDF is then an exact Vlasov equilibrium, so any
   change is Vlasov-solver error.
2. MODE="1d", FROZEN_FIELDS=False: same, with the AP field solve.
3. MODE="2d": full GEM run with the island perturbation.

Running this script writes the cfg named in CFG_NAME into the same directory.
KEY=VALUE arguments override free choices without editing this file, e.g.
  python3 derive_double_harris.py NY=128 DT_TCI=0.01 FROZEN_FIELDS=False CFG_TAG=_ny128
----------------------------------------------------------------------------
"""
import ast
import math
import sys
import textwrap

# ============================= FREE CHOICES =================================
# Reference (peak Harris-sheet) number density, m^-3.
n0 = 1.0e6

# Alfven speed, m/s: c/v_a = 20 as in the paper (see SCALE CHOICE above).
va = 299792458.0 / 20.0

# Velocity half-widths in v_ts. Wider than the paper's (9, 8): Vlasiator keeps one empty
# neighbour-block layer around populated blocks and bails out near the wall, so it needs
# spare blocks that the paper's code does not.
ION_VMAX_MULTIPLE = 16.0       # x v_ti (paper: 9)
ELECTRON_VMAX_MULTIPLE = 12.0  # x v_te (paper: 8)

# Target dv in v_ts; each species gets the whole number of blocks whose dv is closest.
# Lobe electron heating from the velocity-space remap falls steeply with dv_e: 12-16 %/s at
# 0.6 v_te (c/v_a=749), 0.7 % per 1/w_ci at 0.375, 0.27 % per 1/w_ci at 0.3 (1D frozen
# tests). The sheet decay did not change between 0.375 and 0.3 (it is ion/spatial).
DV_ION = 0.5
DV_ELECTRON = 0.3
PAPER_DV_ION = 2.0 * 9.0 / 26.0       # for reporting only
PAPER_DV_ELECTRON = 2.0 * 8.0 / 26.0

# minValue as a fraction of each species' BACKGROUND peak f; one fraction for both species
# keeps the initial truncation species-symmetric. The old rule (SPARSITY_SIGMA=1.4 below the
# sheet peak) put minValue above the background peak, so the t=0 block adjustment emptied
# the lobes; single-sheet runs made with it and the same n_b had the same defect. On the old
# 10-block electron grid, 1e-5 left no spare block before the wall; the check below reports it.
SPARSITY_REL_BACKGROUND = 1.0e-4

# 1 damps the resolved plasma oscillation (0.19 per period at w_pe dt ~ 0.5); 0.6 let the
# core charge oscillation heat the electrons until bailout (1D stage 2, 4.2/w_ci).
THETA = 1.0
LOW_PASS_FILTER = False # re-diffused B every step and drove the earlier bailouts
# Harris.currentLoading (needs the harris-current-loading patch). 1 loads the sheet drifts so the
# plasma current equals the AP solver's discrete curl(B)/mu0, removing the step-1 current impulse
# at the sheets that launched lower-hybrid waves; 2 also loads the in-sheet share of the island
# perturbation's current (doubleSheet); 0 is the analytic Harris loading.
CURRENT_LOADING = 1
# Harris.symmetrySeed (needs the harris-symmetry-seed patch): odd-in-x term at each X-line, relative
# to phi0, that breaks the x -> -x symmetry pinning a secondary island at x=0. 0.01 moves the X-line
# by ~0.17 d_i and adds |dB| <~ 1e-3 B0. Written to MODE="2d" cfgs only (phi0 = 0 in 1d).
SYMMETRY_SEED = 0.01
SEED_WIDTH_LAMBDA0 = 2.0   # Gaussian width in Scale_size units (2 = 1 d_i)

GEOMETRY = "double"     # "double": two sheets, periodic; "single": one sheet, conducting walls
# Cells per 25.6 lambda_i (the double-sheet height, and the width in 2d); the single sheet's
# 12.8 lambda_i interior gets NY/2 rows plus the walls' boundary layers. 256 gives
# lambda0/dy = 5; core erosion and core drift loss were first order in dy.
NY = 256
DT_TCI = 0.005          # dt in 1/w_ci; per-sub-step translation CFL < 1 is checked below

MODE = "1d"             # "1d": x-invariant equilibrium test; "2d": GEM run
X_CELLS_1D = 1          # Vlasiator skips x-translation for 1 cell; use 16 if init fails
FROZEN_FIELDS = True    # B frozen, E = 0: isolates the Vlasov solver (stage 1)
T_MAX_TCI = 10.0        # paper's GEM run: 40
BULK_EVERY_TCI = 0.1
VDF_EVERY_TCI = 1.0     # full-VDF files for wall_scan.py, MODE="1d"; 0 disables
VDF_EVERY_TCI_2D = 4.0  # MODE="2d": all-cell VDF files would be ~18 GB each at 256^2
# Every VDF_STRIDE_2D-th CellID; for Nx a multiple of 16 that is every 16th column,
# including the columns beside x = 0 (X-line) and beside the seam (O-line).
VDF_STRIDE_2D = 16
# MODE="2d": also write the VDF of every cell in the rows just above each sheet centre (and
# the domain's first row), so the X-line VDF is available wherever the X-line goes.
VDF_SHEET_LINES = True
# bailout.velocity_space_wall_block_margin. 1 stops the run when a VDF reaches the last velocity
# block; 0 lets the edge act instead: content pushed past +v is dropped, past -v it is piled
# into the edge cell (cpu_acc_map.cpp), so prefer it only where the +v edge is the one reached.
WALL_BLOCK_MARGIN = 1
RESTART_WALLTIME = 0.0  # io.restart_walltime_interval in s of walltime; 0 = no restart files
CFG_TAG = ""            # appended to the cfg file name
# ==============================================================================

# Only names defined above are accepted, and the value must match the default's type
# (strings are taken verbatim), so a typo fails instead of silently changing the run.
OVERRIDES = []
for _arg in sys.argv[1:]:
    _key, _eq, _val = _arg.partition("=")
    if not _eq or _key not in globals() or _key.startswith("_") or _key == "OVERRIDES":
        raise SystemExit(f"unknown or malformed override: {_arg}")
    _default = globals()[_key]
    try:
        _new = _val if isinstance(_default, str) else ast.literal_eval(_val)
    except (ValueError, SyntaxError):
        raise SystemExit(f"cannot parse override: {_arg}")
    _ok = (type(_new) is type(_default) or
           (isinstance(_default, float) and type(_new) is int))
    if not _ok:
        raise SystemExit(f"override {_arg}: expected {type(_default).__name__}")
    globals()[_key] = float(_new) if isinstance(_default, float) else _new
    OVERRIDES.append(f"{_key}={_val}")

# ----- Vlasiator physical constants (common.h) -- fixed, not configurable -----
C_LIGHT = 299792458.0
EPS_0   = 8.85418782e-12
MU_0    = 1.25663706e-6
K_B     = 1.3806503e-23
CHARGE  = 1.60217653e-19
MASS_PROTON = 1.67262158e-27
WID = 4

mi = MASS_PROTON          # ion species mass (SI)
me = mi / 25.0             # m_i/m_e = 25 (paper)
qi, qe = +CHARGE, -CHARGE

# ----- Derived reference quantities -----
wpi   = math.sqrt(n0 * CHARGE**2 / (EPS_0 * mi))
lam_i = C_LIGHT / wpi                  # ion inertial length (SI), using the TRUE c
lam0  = lam_i / 2.0                     # Harris half-thickness (paper: lambda0 = lambda_i/2)
B0    = va * math.sqrt(MU_0 * n0 * mi)  # from v_a = B0/sqrt(mu0 n0 mi)
wci   = CHARGE * B0 / mi
tci   = 1.0 / wci

Ti_over_Te = 5.0
Theta_i = Ti_over_Te / (1.0 + Ti_over_Te)   # 5/6
Theta_e = 1.0 / (1.0 + Ti_over_Te)          # 1/6

# v_ts = v_a * sqrt(Theta_s * (m_i/m_s) / 2)  -- Eq. 56, expanded in full (see
# the Harris.cpp patch's own comment for the m_s-cancellation used for v_ds;
# v_ts itself does NOT have m_s cancel, hence the explicit m_i/m_s factor here).
vti = va * math.sqrt(Theta_i * (mi / mi) / 2.0)
vte = va * math.sqrt(Theta_e * (mi / me) / 2.0)
Ti = mi * vti**2 / K_B
Te = me * vte**2 / K_B

# v_ds = -2*kB*Ts/(qs*B0*lambda0)  -- Eq. 55, mass cancels against v_ts^2=kB*Ts/ms
vdi = -2.0 * K_B * Ti / (qi * B0 * lam0)
vde = -2.0 * K_B * Te / (qe * B0 * lam0)

# ----- Domain / grid -----
# Square cells (Vlasiator requires dx = dy), dy = 25.6 lambda_i / NY in both geometries.
# double: Ly is the y half-extent; sheets at y=+-Ly/2 sit 6.4 lambda_i from the symmetry
# planes, the paper's sheet-to-wall distance. single: walls at y=+-6.4 lambda_i, and the
# domain extends NG boundary-layer rows beyond each (determineFace() uses 2).
assert GEOMETRY in ("double", "single"), 'GEOMETRY must be "double" or "single"'
assert NY % 4 == 0, "NY must be a multiple of 4"
NG = 2
Lx = 12.8 * lam_i
dy = 25.6 * lam_i / NY
dx = dy
if GEOMETRY == "double":
    Ly, Ny, Ly_wall = 12.8 * lam_i, NY, 0.0
    SHEETS = [(-6.4 * lam_i, +1), (+6.4 * lam_i, -1)]       # (centre, drift sign) as in Harris.cpp
else:
    Ly_wall = 6.4 * lam_i
    Ny = NY // 2 + 2 * NG
    Ly = Ly_wall + NG * dy
    SHEETS = [(0.0, +1)]
assert MODE in ("1d", "2d")
assert not (GEOMETRY == "single" and CURRENT_LOADING == 2), "CURRENT_LOADING=2 is double-sheet only"
assert WALL_BLOCK_MARGIN in (0, 1), "WALL_BLOCK_MARGIN must be 0 or 1"
assert CURRENT_LOADING in (0, 1, 2), "CURRENT_LOADING must be 0, 1 or 2"
Nx = X_CELLS_1D if MODE == "1d" else int(round(2 * Lx / dx))
Lx_run = 0.5 * Nx * dx                    # x half-extent actually written

dt    = DT_TCI * tci
t_max = T_MAX_TCI * tci

nb   = 0.2 * n0                  # background density
phi0 = 0.1 * B0 * lam_i           # island perturbation amplitude
phi0_run = phi0 if MODE == "2d" else 0.0
CFG_NAME = f"{GEOMETRY}_harris_{MODE}{'_frozen' if FROZEN_FIELDS else ''}{CFG_TAG}.cfg"

# ----- Velocity-space extents and block counts -----
ve_max = ELECTRON_VMAX_MULTIPLE * vte
vi_max = ION_VMAX_MULTIPLE * vti

def blocks_for_dv(vmax_multiple, dv_target):
    # nearest dv, which is not always the nearest block count
    n_lo = max(1, math.floor(2.0 * vmax_multiple / (dv_target * WID)))
    return min((n_lo, n_lo + 1), key=lambda n: abs(2.0 * vmax_multiple / (n * WID) - dv_target))

blocks_i = blocks_for_dv(ION_VMAX_MULTIPLE, DV_ION)
blocks_e = blocks_for_dv(ELECTRON_VMAX_MULTIPLE, DV_ELECTRON)
dv_i = 2.0 * vi_max / (blocks_i * WID)
dv_e = 2.0 * ve_max / (blocks_e * WID)

# ----- Sparsity thresholds -----
# f_peak (3D isotropic Maxwellian, v=0) = n * (m/(2 pi kB T))^1.5
def f_peak(n, T, m):
    return n * (m / (2.0 * math.pi * K_B * T))**1.5

minValue_i = SPARSITY_REL_BACKGROUND * f_peak(nb, Ti, mi)
minValue_e = SPARSITY_REL_BACKGROUND * f_peak(nb, Te, me)

# ----- Block-level sparsity check -----
# Mirrors Vlasiator at t=0, row by row in y: fillPhaseSpace samples f at velocity-cell
# centres, then adjustVelocityBlocks (grid.cpp, before any acceleration) keeps a block iff
# it or one of its 26 velocity neighbours (blockAddWidthV=1) has a cell with f >= minValue.
# Blocks kept for a spatial neighbour's content are bounded by the extremes over all rows.
# The acceleration bails out once a target block falls within WALL_MARGIN_BLOCKS of the
# wall, and any outward shift moves the outermost kept block into the next one, so at
# least one spare block is required.
WALL_MARGIN_BLOCKS = 1

def block_sparsity_check(label, mass, T, vts, vd, minValue, v_max, nblk):
    nc = nblk * WID
    dv = 2.0 * v_max / nc
    v = [-v_max + (i + 0.5) * dv for i in range(nc)]
    g = lambda u: math.exp(-u * u / (2.0 * vts * vts))
    # f = pref * g(vx) g(vy) h(vz), so per-block max and sum factorise by axis
    gx = [g(u) for u in v]
    gmax = [max(gx[b*WID:(b+1)*WID]) for b in range(nblk)]
    gsum = [sum(gx[b*WID:(b+1)*WID]) for b in range(nblk)]
    pref = (mass / (2.0 * math.pi * K_B * T))**1.5
    min_edge, max_kept, empty_rows, lost, nkept = nblk, 0, 0, [], []
    for j in range(Ny):
        y = -Ly + (j + 0.5) * dy
        ns = [(n0 / math.cosh((y - yk) / lam0)**2, sk) for yk, sk in SHEETS]
        h = [sum(nk * g(u - sk * vd) for nk, sk in ns) + nb * g(u) for u in v]
        hmax = [max(h[b*WID:(b+1)*WID]) for b in range(nblk)]
        hsum = [sum(h[b*WID:(b+1)*WID]) for b in range(nblk)]
        content = [(a, b, c) for a in range(nblk) for b in range(nblk) for c in range(nblk)
                   if pref * gmax[a] * gmax[b] * hmax[c] >= minValue]
        kept = {(a + da, b + db, c + dc) for (a, b, c) in content
                for da in (-1, 0, 1) for db in (-1, 0, 1) for dc in (-1, 0, 1)
                if 0 <= a + da < nblk and 0 <= b + db < nblk and 0 <= c + dc < nblk}
        if not kept:
            empty_rows += 1
            lost.append(1.0)
            continue
        min_edge = min(min_edge, min(min(i, nblk - 1 - i) for blk in kept for i in blk))
        max_kept = max(max_kept, len(kept)); nkept.append(len(kept))
        retained = pref * dv**3 * sum(gsum[a] * gsum[b] * hsum[c] for (a, b, c) in kept)
        lost.append(1.0 - retained / (sum(nk for nk, _ in ns) + nb))
    spare = min_edge - WALL_MARGIN_BLOCKS
    status = "PASS" if (spare >= 1 and empty_rows == 0) else "FAIL"
    print(f"  {label}: {nblk} blocks/dim, dv={dv/vts:.3f} v_ts, block={WID*dv/vts:.2f} v_ts, "
          f"spare blocks before wall margin={spare} [{status}]")
    print(f"           y-rows with no blocks={empty_rows}/{Ny}, max density truncated={max(lost):.2e}, "
          f"kept blocks/cell max {max_kept}, mean {sum(nkept)/max(len(nkept),1):.0f}")
    return status, spare, lost, sum(nkept) / max(len(nkept), 1)

# ----- Debye length, for awareness (the AP solver does not require resolving
# it, unlike an explicit scheme, but it's a useful diagnostic regardless) -----
lambda_D = vte / wpi / math.sqrt(mi / me)  # = sqrt(eps0 kB Te / (n0 q^2))

# =============================== Report ======================================
print("================ Derived reference quantities ================")
print(f"n0        = {n0:.6e} m^-3   (free choice)")
print(f"v_a       = {va:.6e} m/s    (free choice)")
print(f"lambda_i  = {lam_i:.6e} m")
print(f"lambda0   = {lam0:.6e} m")
print(f"B0        = {B0:.6e} T")
print(f"w_ci      = {wci:.6e} rad/s")
print(f"t_ci=1/w_ci = {tci:.6e} s")
print(f"c/v_a     = {C_LIGHT/va:.6f}   (paper: 20)")
print(f"Debye length = {lambda_D:.6e} m,  dx/lambda_D = {dx/lambda_D:.3f}")
print()
print("================ Consistency checks against the paper =========")
print(f"v_ti/v_a  = {vti/va:.6f}   target sqrt(5/12)  = {math.sqrt(5/12):.6f}")
print(f"v_te/v_a  = {vte/va:.6f}   target sqrt(25/12) = {math.sqrt(25/12):.6f}")
print(f"v_di/v_a  = {vdi/va:.6f}   target -5/3         = {-5/3:.6f}")
print(f"v_de/v_a  = {vde/va:.6f}   target  1/3         = {1/3:.6f}")
print(f"(w_pe/w_ci = {wpi*math.sqrt(mi/me)/wci:.6f} -- NOT an independent check: algebraically")
print(f" equal to 5*(c/v_a) given m_i/m_e=25, so it only reads the paper's stated 100")
print(f" when c/v_a=20 is specifically chosen, same as v_a=c_true/20 would give)")
print()
print("================ Species parameters (SI) =======================")
print(f"Ti        = {Ti:.6e} K")
print(f"Te        = {Te:.6e} K")
print(f"v_ti      = {vti:.6e} m/s      v_te      = {vte:.6e} m/s")
print(f"v_di      = {vdi:.6e} m/s      v_de      = {vde:.6e} m/s   (z-directed drifts)")
print(f"n0        = {n0:.6e} m^-3      nb        = {nb:.6e} m^-3")
print()
print("================ Domain / grid =================================")
print(f"MODE={MODE}, FROZEN_FIELDS={FROZEN_FIELDS}: x half-extent {Lx_run:.6e} m (Nx={Nx}, dx={dx:.6e} m)")
print(f"GEOMETRY={GEOMETRY}: Ly = {Ly:.6e} m  (Ny={Ny}, dy={dy:.6e} m)"
      + (f"; walls at y=+-{Ly_wall:.6e} m, {NG} boundary rows beyond each" if GEOMETRY == "single" else ""))
print(f"dt = {dt:.6e} s   t_max = {t_max:.6e} s   ({t_max/dt:.0f} steps)")
# AP translation runs as theta*dt then (1-theta)*dt; Vlasiator's translation moves mass at
# most one cell per call (cpu_trans_map_amr.cpp), so each sub-step needs CFL < 1.
cfl_e = ve_max * max(THETA, 1.0 - THETA) * dt / dx
print(f"fastest electrons (v={ve_max:.3e} m/s): translation CFL per sub-step = {cfl_e:.3f}"
      f"  (theta={THETA:g}; must stay < 1)")
print()
print("================ Velocity space =================================")
print(f"electron: +-{ve_max:.6e} m/s, {blocks_e*WID} cells/dim ({blocks_e} blocks), "
      f"dv={dv_e/vte:.3f} v_te (target {DV_ELECTRON:.3f}, paper {PAPER_DV_ELECTRON:.3f})")
print(f"ion:      +-{vi_max:.6e} m/s, {blocks_i*WID} cells/dim ({blocks_i} blocks), "
      f"dv={dv_i/vti:.3f} v_ti (target {DV_ION:.3f}, paper {PAPER_DV_ION:.3f})")
print(f"minValue (electron, {SPARSITY_REL_BACKGROUND:g} x background peak) = {minValue_e:.6e}")
print(f"minValue (ion,      {SPARSITY_REL_BACKGROUND:g} x background peak) = {minValue_i:.6e}")
print()
print("================ Block-level sparsity check (t=0) ===============")
status_e, spare_e, lost_e, mean_e = block_sparsity_check("electron", me, Te, vte, vde, minValue_e, ve_max, blocks_e)
status_i, spare_i, lost_i, mean_i = block_sparsity_check("ion     ", mi, Ti, vti, vdi, minValue_i, vi_max, blocks_i)
print(f"  max |n_i - n_e|/n from truncation = {max(abs(a - b) for a, b in zip(lost_i, lost_e)):.2e}")
if status_e == "FAIL" or status_i == "FAIL" or cfl_e >= 1.0:
    raise SystemExit("Check FAILED -- for no spare block, widen *_VMAX_MULTIPLE or raise "
                     "SPARSITY_REL_BACKGROUND; for empty y-rows, lower SPARSITY_REL_BACKGROUND; "
                     "for CFL >= 1, lower dt or *_VMAX_MULTIPLE. Not writing the cfg.")
print()
print("================ Field perturbation =============================")
print(f"BX0 (Harris.BX0)             = {B0:.6e} T")
print(f"perturbationAmplitude (phi0) = {phi0_run:.6e} T*m" + ("  (MODE=1d: off)" if MODE == "1d" else ""))

deviations = []
if abs(C_LIGHT / va - 20.0) > 1e-6:
    deviations.append(f"c/v_a={C_LIGHT/va:.3g} instead of 20")
if abs(dt / tci - 0.02) > 1e-9:
    deviations.append(f"dt={dt/tci:g}/w_ci instead of 0.02/w_ci")
if (ION_VMAX_MULTIPLE, ELECTRON_VMAX_MULTIPLE) != (9.0, 8.0):
    deviations.append(f"velocity boxes +-{ION_VMAX_MULTIPLE:g}v_ti and +-{ELECTRON_VMAX_MULTIPLE:g}v_te "
                      f"instead of +-9v_ti and +-8v_te")
if abs(dx / lam_i - 0.2) > 1e-9:
    deviations.append(f"dx=dy={dx/lam_i:g} lambda_i instead of 0.2")
if (abs(dv_i / vti - PAPER_DV_ION) > 1e-3) or (abs(dv_e / vte - PAPER_DV_ELECTRON) > 1e-3):
    deviations.append(f"dv={dv_i/vti:.3f}v_ti and {dv_e/vte:.3f}v_te instead of {PAPER_DV_ION:.3f} and {PAPER_DV_ELECTRON:.3f}")
if GEOMETRY == "double":
    deviations.append("two opposite-polarity sheets in a fully periodic box instead of one sheet "
                      "between conducting walls")
deviations.append(f"theta={THETA:g} (the paper does not state its value for Sec. 5.4)")
if LOW_PASS_FILTER:
    deviations.append("lowPassFilter=1, which is not in the paper")
stage = ("1D x-invariant equilibrium test, no perturbation, " + ("fields FROZEN (B fixed, E=0)" if FROZEN_FIELDS else "AP field solve on")
         if MODE == "1d" else "2D GEM run" + (", fields FROZEN" if FROZEN_FIELDS else ""))
override_note = ("# Overrides: " + " ".join(OVERRIDES) + "\n") if OVERRIDES else ""
geometry_note = ("# DOUBLE Harris-sheet variant of the GEM reconnection challenge (Liu, Cai, Cao &\n"
                 "# Lapenta 2025, JCP 528:113840, Section 5.4), fully periodic." if GEOMETRY == "double" else
                 "# GEM reconnection challenge as in Liu, Cai, Cao & Lapenta 2025, JCP 528:113840,\n"
                 "# Section 5.4: one Harris sheet between conducting walls, periodic in x.")
geometry_note += "\n# See derive_double_harris.py's module docstring."
requires_extra = ("; harris-single-sheet.patch for the face-evaluated single-sheet perturbation"
                  if GEOMETRY == "single" else "")
stage_note = textwrap.fill(f"Stage: {stage}; {Nx} x {Ny} cells, t_max = {T_MAX_TCI:g}/w_ci "
                           f"({t_max/dt:.0f} steps).", width=80, initial_indent="# ", subsequent_indent="# ")
deviation_note = textwrap.fill(
    ("Deliberate deviations from the paper: " + "; ".join(deviations) + ".") if deviations
    else "No deviations from the paper's Section 5.4 parameters.",
    width=80, initial_indent="# ", subsequent_indent="# ")

seed_lines = (f"symmetrySeed = {SYMMETRY_SEED:g}\nsymmetrySeedWidth = {SEED_WIDTH_LAMBDA0:g}\n"
              if MODE == "2d" else "")
if GEOMETRY == "double":
    boundary_block = """[boundaries]
periodic_x = yes
periodic_y = yes
periodic_z = yes
# No "boundary = ..." line: fully periodic in all three dimensions, so there
# is no open/wall face anywhere for a boundary condition to apply to.
"""
else:
    boundary_block = f"""[boundaries]
periodic_x = yes
periodic_y = no
periodic_z = yes
boundary = ConductingWall
# The {NG} outermost rows at each y face are the walls' boundary layers; the walls sit
# between them and the interior, at y = +-{Ly_wall:.9e} m (+-6.4 lambda_i).

[proton_conductingwall]
face = y+
face = y-

[electron_conductingwall]
face = y+
face = y-
"""
# rows just above each sheet centre: NY/4, 3NY/4 (double) or Ny/2 (single); row 0 comes along
YLINE = NY // 4 if GEOMETRY == "double" else Ny // 2
line_rows = len(range(0, Ny, YLINE)) if (MODE == "2d" and VDF_SHEET_LINES) else 0

def io_class(name, every_tci, stride, lines=False):
    steps = max(1, int(round(every_tci * tci / dt)))
    xl, yl, zl = (1, YLINE, 1) if lines else (0, 0, 0)
    return (f"system_write_t_interval = {steps*dt*(1-1e-9):.9e}\n"
            f"system_write_file_name = {name}\n"
            f"system_write_distribution_stride = {stride}\n"
            f"system_write_distribution_xline_stride = {xl}\n"
            f"system_write_distribution_yline_stride = {yl}\n"
            f"system_write_distribution_zline_stride = {zl}\n")
io_classes = io_class("bulk", BULK_EVERY_TCI, 0)
vdf_every = VDF_EVERY_TCI if MODE == "1d" else VDF_EVERY_TCI_2D
vdf_stride = 1 if MODE == "1d" else VDF_STRIDE_2D
if vdf_every > 0:
    io_classes += ("# VDFs: every " + f"{vdf_stride}" + "th cell" + (", plus every cell of rows "
                   + ", ".join(str(r) for r in range(0, Ny, YLINE)) if line_rows else "") + "\n"
                   + io_class("distributions", vdf_every, vdf_stride, lines=line_rows > 0))
if RESTART_WALLTIME > 0:
    io_classes += f"restart_walltime_interval = {RESTART_WALLTIME:g}\n"

# =============================== Write the cfg ===============================
cfg = f"""\
{geometry_note}
#
{stage_note}
{override_note}#
# Requires: petschge/vlasiator ca0a894 or later{requires_extra}.
#
# Generated by derive_double_harris.py -- DO NOT hand-edit the derived
# numbers below; edit the free choices (n0, v_a, *_VMAX_MULTIPLE, DV_*,
# SPARSITY_REL_BACKGROUND, THETA, LOW_PASS_FILTER, CURRENT_LOADING, SYMMETRY_SEED,
# SEED_WIDTH_LAMBDA0, GEOMETRY, NY, DT_TCI, MODE, VDF_SHEET_LINES, WALL_BLOCK_MARGIN,
# X_CELLS_1D, FROZEN_FIELDS, T_MAX_TCI, *_EVERY_TCI) at the top of that script
# and rerun instead, so this file and its derivation stay in sync.
#
{deviation_note}
#
# Dimensionless cross-checks against the paper (computed by derive_gem.py;
# these should match to machine precision for ANY choice of n0/v_a -- they
# do not depend on c, unlike c/v_a itself):
#   v_ti/v_a={vti/va:.6f} (target sqrt(5/12)={math.sqrt(5/12):.6f})
#   v_te/v_a={vte/va:.6f} (target sqrt(25/12)={math.sqrt(25/12):.6f})
#   v_di/v_a={vdi/va:.6f} (target -5/3={-5/3:.6f})
#   v_de/v_a={vde/va:.6f} (target 1/3={1/3:.6f})
# (w_pe/w_ci={wpi*math.sqrt(mi/me)/wci:.6f}, paper states 100 -- but this ratio is
#  algebraically 5*(c/v_a), so it only equals 100 when c/v_a=20 is chosen; it
#  is not an independent check of this run's n0/v_a, just of that choice)

dynamic_timestep = 0
project = Harris
ParticlePopulations = proton
ParticlePopulations = electron

propagate_field = {0 if FROZEN_FIELDS else 1}
propagate_vlasov_translation = 1
propagate_vlasov_acceleration = 1

[io]
diagnostic_write_interval = 10
write_initial_state = {1 if MODE == '1d' else 0}
{io_classes}
[gridbuilder]
x_length = {Nx}
y_length = {Ny}
z_length = 1
x_min = {-Lx_run:.9e}
x_max = {Lx_run:.9e}
y_min = {-Ly:.9e}
y_max = {Ly:.9e}
z_min = {-0.5*dx:.9e}
z_max = {0.5*dx:.9e}

dt = {dt:.9e}
t_max = {t_max:.9e}

[bailout]
min_dt = {dt*1e-3:.3e}
velocity_space_wall_block_margin = {WALL_BLOCK_MARGIN}

{boundary_block}
[fieldsolver]
method = AP
theta = {THETA:g}
enforceGaussLaw = 1
lowPassFilter = {int(LOW_PASS_FILTER)}

[Harris]
Scale_size = {lam0:.9e}
BX0 = {B0:.9e}
BY0 = 0.0
BZ0 = 0.0
perturbationAmplitude = {phi0_run:.9e}
doubleSheet = {1 if GEOMETRY == "double" else 0}
currentLoading = {CURRENT_LOADING}
{seed_lines}
[proton_Harris]
Temperature = {Ti:.9e}
rho = {n0:.9e}
rhoBackground = {nb:.9e}

[electron_Harris]
Temperature = {Te:.9e}
rho = {n0:.9e}
rhoBackground = {nb:.9e}

[proton_properties]
mass = 1.0
mass_units = PROTON
charge = 1

[electron_properties]
mass = 0.04
mass_units = PROTON
charge = -1

[proton_vspace]
# {blocks_i} blocks x WID={WID} over +-{ION_VMAX_MULTIPLE:g} v_ti: dv={dv_i/vti:.3f} v_ti (target {DV_ION:g})
vx_min = {-vi_max:.9e}
vx_max = {vi_max:.9e}
vy_min = {-vi_max:.9e}
vy_max = {vi_max:.9e}
vz_min = {-vi_max:.9e}
vz_max = {vi_max:.9e}
vx_length = {blocks_i}
vy_length = {blocks_i}
vz_length = {blocks_i}

[proton_sparse]
# {SPARSITY_REL_BACKGROUND:g} x background peak f (both species); {spare_i} spare block(s) before the wall at t=0
minValue = {minValue_i:.6e}

[electron_vspace]
# {blocks_e} blocks x WID={WID} over +-{ELECTRON_VMAX_MULTIPLE:g} v_te: dv={dv_e/vte:.3f} v_te (target {DV_ELECTRON:g})
vx_min = {-ve_max:.9e}
vx_max = {ve_max:.9e}
vy_min = {-ve_max:.9e}
vy_max = {ve_max:.9e}
vz_min = {-ve_max:.9e}
vz_max = {ve_max:.9e}
vx_length = {blocks_e}
vy_length = {blocks_e}
vz_length = {blocks_e}

[electron_sparse]
# {SPARSITY_REL_BACKGROUND:g} x background peak f (both species); {spare_e} spare block(s) before the wall at t=0
minValue = {minValue_e:.6e}

[variables]
output = populations_vg_rho
output = populations_vg_v
output = populations_vg_ptensor
output = populations_vg_blocks

output = vg_rhom
output = vg_rhoq

output = fg_e
output = fg_b
output = vg_e_vol
output = vg_b_vol

output = populations_vg_nonmaxwellianess

output = vg_rank
output = vg_boundarytype
diagnostic = populations_vg_blocks
"""

with open(CFG_NAME, "w") as f:
    f.write(cfg)
print()
print(f"Wrote {CFG_NAME}")
# ~260 B per velocity block, measured from the 128^2 2D pilot's VDF files (17.29 M blocks,
# 4.50 GB each); the initial state is written with full VDFs as well.
_all_gb = Nx * Ny * (mean_e + mean_i) * 260 / 1e9
_vdf_gb = _all_gb / vdf_stride + _all_gb * line_rows / Ny
_nvdf = int(T_MAX_TCI / vdf_every) + 1 if vdf_every > 0 else 0
_init_gb = _all_gb if MODE == "1d" else 0.0       # write_initial_state is on only in 1d
print(f"Velocity blocks at t=0: ~{Nx*Ny*(mean_e+mean_i)/1e6:.2f} M. VDF files ~{_vdf_gb:.2f} GB each x {_nvdf}"
      f" (stride {vdf_stride}{f' + {line_rows} rows' if line_rows else ''}), initial state {_init_gb:.2f} GB: ~{_vdf_gb*_nvdf+_init_gb:.1f} GB in total")
