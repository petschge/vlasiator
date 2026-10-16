#!/usr/bin/env python3
"""
SI-unit derivation and .cfg generator for the GEM reconnection challenge
(Liu, Cai, Cao & Lapenta 2025, JCP 528:113840, Section 5.4), for Vlasiator's
AP (CSL-RME) field solver.

----------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS (the c/v_a issue)
----------------------------------------------------------------------------
The paper specifies the GEM setup entirely in dimensionless form: lengths in
ion inertial lengths lambda_i, time in ion cyclotron periods 1/w_ci, velocity
in the Alfven speed v_a, B in B0, mass in the ion mass m_i, density in n0.
It also states three ratios: m_i/m_e = 25, T_i/T_e = 5, and c/v_a = 20.

Vlasiator runs in SI units with a FIXED, true speed of light (no mechanism
to reduce it, unlike many explicit PIC/Vlasov reconnection codes -- which is
almost certainly why the paper's own c/v_a is as small as 20: an artificial
light-speed reduction in THEIR code to relax its CFL condition, not a
statement about realistic plasma parameters). Reproducing c/v_a=20 exactly
with the true c forces v_a ~ 1.5e7 m/s and T ~ 1e10 K, with thermal speeds
reaching several percent of c even though Vlasiator's Vlasov solver is
non-relativistic.

So n0 and v_a are left here as free choices -- there is no way to derive
them from the paper's own dimensionless numbers, because the paper's
numbers don't fix an absolute physical scale (this is different from the
POS test in Section 5.3, where the paper states actual SI quantities).
Edit the two marked values below and rerun; every other quantity is
derived from them. The dimensionless ratios the paper DOES state (v_ti/v_a,
v_te/v_a, v_di/v_a, v_de/v_a, w_pe/w_ci) are cross-checked against their
target values at the end of the report and should match to machine
precision regardless of what you choose for n0/v_a -- if they don't, the
derivation itself is broken, not your choice of scale.

Running this script writes gem.cfg into the same directory.
----------------------------------------------------------------------------
"""
import math

# ============================= FREE CHOICES =================================
# Reference (peak Harris-sheet) number density, m^-3.
n0 = 1.0e6

# Reference Alfven speed, m/s. This is the "how fast" decision discussed
# above. Some reference points:
#   c_true/20        = 1.4990e7 m/s  -- matches the paper's c/v_a=20 exactly,
#                                        but T~1e10 K, v_te/c ~ 7%.
#   ~4e5 (default)    -- typical Earth-magnetotail-like Alfven speed;
#                         T~10^6-10^7 K, safely non-relativistic (v_te/c
#                         well under 1%); c/v_a then comes out to ~750,
#                         far above the paper's 20 -- which should, if
#                         anything, make the quasi-static reconnection
#                         physics MORE accurate, since a larger c/v_a means
#                         the displacement current is more negligible on
#                         reconnection timescales, not less.
va = 4.0e5 # 1.4990e7

# Velocity-space cells per dimension, and the ion/electron domain half-widths (in units
# of v_ti/v_te). The paper uses Nvx=Nvy=Nvz=26 cells (6.5 blocks) with vi_max=9*v_ti,
# ve_max=8*v_te -- tuned for the paper's own CSL-RME code, but NOT sufficient for
# Vlasiator's block-sparse VDF representation: findBlocksToInitialize adds a 2-block
# safety buffer around each population's own computed extent, and the acceleration
# solver separately requires content to stay >=1 block from the domain wall
# (bailout.velocity_space_wall_block_margin). With only ~7 blocks spanning the whole
# domain (28 cells), that fixed 3-block (2+1) overhead is nearly as wide as the domain
# itself, regardless of how correctly each population's own content radius is computed.
# This is not fixable by changing the sparsity threshold or by fixing block-allocation
# bugs alone; it needs a finer grid so the fixed per-block overhead becomes a small
# fraction of the domain. The values below are verified further down in this script
# (see "Velocity-space margin check") to leave >=10% headroom between each species'
# actual required extent and the configured wall, including both the init-time buffer
# and the acceleration-time wall margin -- rerun this script after changing n0/v_a or
# these multiples and confirm that check still reports PASS before trusting the cfg.
VCELLS_PER_DIM = 64          # 32 blocks/dim; must be a multiple of WID.
ION_VMAX_MULTIPLE = 12.0      # x v_ti (paper: 9; widened for Vlasiator's own margins)
ELECTRON_VMAX_MULTIPLE = 11.0  # x v_te (paper: 8; small increase for comfortable margin)

# Sparsity threshold, as a number of standard deviations of the (dominant,
# Harris-sheet) Maxwellian peak. The velocity-space domain itself extends to
# 8 v_te / 9 v_ti (paper); the threshold below controls how much of that
# domain is actually kept as populated blocks, not the domain's own extent.
# 5 sigma leaves 3-4 v_te/v_ti of clearance before the wall in each species'
# domain (same margin-based reasoning as the POS cfgs in this same series).
SPARSITY_SIGMA = 3.0
# ==============================================================================

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

# ----- Domain / grid (paper, Section 5.4) -----
Lx = 12.8 * lam_i
# Ly = 6.4 * lam_i
Nx = 128
Ny = 60             # paper: Nx = 2*Ny
dx = 2 * Lx / Nx
dy = dx
#dy = 2 * Ly / Ny
Ly = dy*Ny/2.

dt    = 0.01 * tci
t_max = 40.0 * tci

nb   = 0.2 * n0                  # background density
phi0 = 0.1 * B0 * lam_i           # island perturbation amplitude

# ----- Velocity-space extents -----
ve_max = ELECTRON_VMAX_MULTIPLE * vte
vi_max = ION_VMAX_MULTIPLE * vti
blocks_per_dim = VCELLS_PER_DIM // WID
assert VCELLS_PER_DIM % WID == 0, "VCELLS_PER_DIM must be a multiple of WID"

# ----- Sparsity thresholds: SPARSITY_SIGMA cutoff of each species' peak -----
# f_peak (3D isotropic Maxwellian, v=0) = n * (m/(2 pi kB T))^1.5
def f_peak(n, T, m):
    return n * (m / (2.0 * math.pi * K_B * T))**1.5

fpeak_e = f_peak(n0, Te, me)   # dominant (Harris-sheet) population peak
fpeak_i = f_peak(n0, Ti, mi)
minValue_e = fpeak_e * math.exp(-SPARSITY_SIGMA**2 / 2.0)
minValue_i = fpeak_i * math.exp(-SPARSITY_SIGMA**2 / 2.0)

# ----- Velocity-space margin check -----
# For each species: the content radius Vlasiator's (fixed) findBlocksToInitialize
# needs to resolve is the larger of (a) the drifting Harris-sheet population's own
# far-side extent at y=0 (where its density is full n0, centred on v_d), and (b) the
# uniform background population's own extent (density n_b, centred on 0) -- which
# dominates far from the current sheet, where the sheet's local density has decayed to
# a negligible fraction of its peak (see this patch series' findBlocksToInitialize
# fix). Both are isotropic Gaussians, so each has a closed-form radius at which its
# own density alone drops to tolerance*minValue (tolerance=0.1, matching
# TriAxisSearch's own convention):
#   f_peak * exp(-r^2/(2 v_ts^2)) = tolerance*minValue  =>  r = v_ts*sqrt(2*ln(f_peak/(tolerance*minValue)))
# This must fit inside the domain half-width together with the 2-block init buffer and
# the 1-block acceleration wall-margin (3 blocks of overhead total, each
# 2*v_max/blocks_per_dim wide) -- checked below with >=10% required headroom.
TRIAXIS_TOLERANCE = 0.1
INIT_BUFFER_BLOCKS = 2
WALL_MARGIN_BLOCKS = 1

def content_radius_and_margin(label, mass, T, vts, vd, minValue, v_max):
    threshold = TRIAXIS_TOLERANCE * minValue
    fpeak_sheet = f_peak(n0, T, mass)
    r_sheet = vts * math.sqrt(2.0 * math.log(fpeak_sheet / threshold)) if fpeak_sheet > threshold else 0.0
    fpeak_bg = f_peak(nb, T, mass)
    r_bg = vts * math.sqrt(2.0 * math.log(fpeak_bg / threshold)) if fpeak_bg > threshold else 0.0
    content = max(abs(vd) + r_sheet, r_bg)
    dvBlock = 2.0 * v_max / blocks_per_dim
    overhead = (INIT_BUFFER_BLOCKS + WALL_MARGIN_BLOCKS) * dvBlock
    required = content + overhead
    margin = (v_max - required) / v_max
    status = "PASS" if margin >= 0.10 else ("MARGINAL" if margin >= 0.0 else "FAIL")
    print(f"  {label}: content={content/vts:.3f} v_ts, +{INIT_BUFFER_BLOCKS}+{WALL_MARGIN_BLOCKS} blocks overhead="
          f"{overhead/vts:.3f} v_ts, required={required/vts:.3f} v_ts, domain={v_max/vts:.3f} v_ts "
          f"-> margin={margin*100:.1f}% [{status}]")
    return status

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
print(f"c/v_a     = {C_LIGHT/va:.6f}   (paper: 20; see header note on why this need not match)")
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
print(f"Lx = {Lx:.6e} m  (Nx={Nx}, dx={dx:.6e} m)")
print(f"Ly = {Ly:.6e} m  (Ny={Ny}, dy={dy:.6e} m)")
print(f"dt = {dt:.6e} s   t_max = {t_max:.6e} s   ({t_max/dt:.0f} steps)")
print(f"fastest electrons (v={ve_max:.3e} m/s) cross {ve_max*dt/dx:.2f} cells/step")
print(f"  (dt=0.02/w_ci is the paper's own value, not a free choice here; the")
print(f"   semi-Lagrangian translation scheme is not CFL-limited the way an")
print(f"   explicit Eulerian one would be, but a ratio above 1 is still worth")
print(f"   being aware of)")
print()
print("================ Velocity space =================================")
print(f"electron: +-{ve_max:.6e} m/s, {VCELLS_PER_DIM} cells/dim ({blocks_per_dim} blocks)")
print(f"ion:      +-{vi_max:.6e} m/s, {VCELLS_PER_DIM} cells/dim ({blocks_per_dim} blocks)")
print(f"minValue (electron, {SPARSITY_SIGMA:.0f}-sigma of peak) = {minValue_e:.6e}")
print(f"minValue (ion,      {SPARSITY_SIGMA:.0f}-sigma of peak) = {minValue_i:.6e}")
print()
print("================ Velocity-space margin check ====================")
print("(content radius vs. domain edge, including Vlasiator's own init-buffer")
print(" and acceleration wall-margin overhead -- see comment above for the derivation)")
status_e = content_radius_and_margin("electron", me, Te, vte, vde, minValue_e, ve_max)
status_i = content_radius_and_margin("ion     ", mi, Ti, vti, vdi, minValue_i, vi_max)
if status_e == "FAIL" or status_i == "FAIL":
    raise SystemExit("Velocity-space margin check FAILED -- widen ION_VMAX_MULTIPLE / "
                      "ELECTRON_VMAX_MULTIPLE or increase VCELLS_PER_DIM above, then rerun. "
                      "Not writing gem.cfg.")
print()
print("================ Field perturbation =============================")
print(f"BX0 (Harris.BX0)             = {B0:.6e} T")
print(f"perturbationAmplitude (phi0) = {phi0:.6e} T*m")

# =============================== Write gem.cfg ===============================
cfg = f"""\
# GEM reconnection challenge, reproducing Liu, Cai, Cao & Lapenta 2025
# (JCP 528:113840), Section 5.4.
#
# Requires: AP solver + boundary-condition patches (ES_FS branch, on top of
# dd574e33); gem/0001-conducting-wall.patch (new ConductingWall boundary);
# gem/0002-harris-gem-extension.patch (Harris project: y-dependent density,
# species drift, background population, island perturbation).
#
# Generated by derive_gem.py -- DO NOT hand-edit the derived numbers below;
# edit the free choices (n0, v_a, VCELLS_PER_DIM, SPARSITY_SIGMA) at the top
# of that script and rerun instead, so this file and its derivation stay in
# sync. See that script's header for why n0/v_a are free choices (the
# paper's c/v_a=20 cannot be matched using Vlasiator's true, fixed c without
# implying T~1e10 K and mildly relativistic thermal speeds -- this run uses
# {"the paper's exact c/v_a=20 ratio" if abs(C_LIGHT/va-20)<1e-6 else f"a different v_a (c/v_a={C_LIGHT/va:.3g}, chosen for non-relativistic T)"}).
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

propagate_field = 1
propagate_vlasov_translation = 1
propagate_vlasov_acceleration = 1

[io]
diagnostic_write_interval = 10
write_initial_state = 1
system_write_t_interval = {t_max/50.0:.6e}
system_write_distribution_stride = 0
system_write_distribution_xline_stride = 0
system_write_distribution_yline_stride = 0
system_write_distribution_zline_stride = 0
system_write_file_name = bulk

[gridbuilder]
x_length = {Nx}
y_length = {Ny}
z_length = 1
x_min = {-Lx:.9e}
x_max = {Lx:.9e}
y_min = {-Ly:.9e}
y_max = {Ly:.9e}
z_min = {-0.5*dx:.9e}
z_max = {0.5*dx:.9e}

dt = {dt:.9e}
t_max = {t_max:.9e}

[bailout]
min_dt = {dt*1e-3:.3e}
velocity_space_wall_block_margin = 1

[boundaries]
periodic_x = yes
periodic_y = no
periodic_z = yes
boundary = ConductingWall

[proton_conductingwall]
face = y+
face = y-

[electron_conductingwall]
face = y+
face = y-

[fieldsolver]
method = AP
theta = 1.0
enforceGaussLaw = 1
lowPassFilter = 1

[Harris]
Scale_size = {lam0:.9e}
BX0 = {B0:.9e}
BY0 = 0.0
BZ0 = 0.0
perturbationAmplitude = {phi0:.9e}

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
vx_min = {-vi_max:.9e}
vx_max = {vi_max:.9e}
vy_min = {-vi_max:.9e}
vy_max = {vi_max:.9e}
vz_min = {-vi_max:.9e}
vz_max = {vi_max:.9e}
vx_length = {blocks_per_dim}
vy_length = {blocks_per_dim}
vz_length = {blocks_per_dim}

[proton_sparse]
minValue = {minValue_i:.6e}

[electron_vspace]
vx_min = {-ve_max:.9e}
vx_max = {ve_max:.9e}
vy_min = {-ve_max:.9e}
vy_max = {ve_max:.9e}
vz_min = {-ve_max:.9e}
vz_max = {ve_max:.9e}
vx_length = {blocks_per_dim}
vy_length = {blocks_per_dim}
vz_length = {blocks_per_dim}

[electron_sparse]
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

output = vg_gridcoordinates
output = vg_rank
output = vg_boundarytype
diagnostic = populations_vg_blocks
"""

with open("gem.cfg", "w") as f:
    f.write(cfg)
print()
print("Wrote gem.cfg")
