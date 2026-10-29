#!/usr/bin/env python3
import glob
import numpy as np
import analysator
import matplotlib.pyplot as plt
import os
import sys
import argparse


# parse command line
parser = argparse.ArgumentParser(description="Plot expected and observed dispersion relations")
parser.add_argument("--title",
                    dest="title",
                    action="store",
                    type=str,
                    default="",
                    help="Title above all plots (default: %(default)s)")
parser.add_argument("--cmap",
                    dest="cmap",
                    action="store",
                    type=str,
                    default="viridis",
                    help="Name of matplotlib color map to use (default: %(default)s)")
parser.add_argument("--freqnorm",
                    dest="freqnorm",
                    action="store",
                    type=str,
                    default="wpe",
                    help="Frequency used for normalization (wpe|wpi|Wce|Wci|dt) (default: %(default)s)")
parser.add_argument("--lengthnorm",
                    dest="lengthnorm",
                    action="store",
                    type=str,
                    default="lD",
                    help="Length scale used for normalization (lD|re|ri|de|di|dx) (default: %(default)s)")
parser.add_argument("--kmax",
                    dest="kmax",
                    action="store",
                    type=str,
                    default="",
                    help="Maximum k value (default: %(default)s)")
parser.add_argument("--wmax",
                    dest="wmax",
                    action="store",
                    type=str,
                    default="",
                    help="Maximum frequency value (default: %(default)s)")
parser.add_argument("--cbrange",
                    dest="cbrange",
                    action="store",
                    type=int,
                    default=10,
                    help="Order of magnitude between cbmin and cmax (default: %(default)s)")
parser.add_argument("--onlyE",
                    dest="onlyE",
                    action="store_true",
                    help="plot only electric fields (default: %(default)s)")
parser.add_argument("--decimate",
                    dest="decimate",
                    action="store",
                    type=int,
                    default=1,
                    help="Read every N.th timestep (default: %(default)s)")
parser.add_argument("--limit",
                    dest="limit",
                    action="store",
                    type=float,
                    default=1.,
                    help="Read the first 1/N of timesteps (default: %(default)s)")
parser.add_argument("--limit-part",
                    dest="limit_part",
                    action="store",
                    type=float,
                    default=1.,
                    help="Read the first n/N of timesteps (default: %(default)s)")

parser.add_argument("--panel-label",
                    dest="panel_label",
                    action="store",
                    type=str,
                    default="",
                    help="Label such as (a) drawn inside every plot (default: none)")
parser.add_argument("--panel-label-loc",
                    dest="panel_label_loc",
                    action="store",
                    choices=["upper left", "upper right", "lower left", "lower right"],
                    default="upper left",
                    help="Corner for the panel label (default: %(default)s)")
parser.add_argument("--figsize",
                    dest="figsize",
                    action="store",
                    type=float,
                    nargs=2,
                    metavar=("WIDTH", "HEIGHT"),
                    default=None,
                    help="Figure size in inches, sets the aspect ratio (default: matplotlib's)")
parser.add_argument("--legend-loc",
                    dest="legend_loc",
                    action="store",
                    type=str,
                    default="best",
                    help="Matplotlib legend location, or 'none' to omit the legend (default: %(default)s)")
parser.add_argument("--no-numerical-dispersion",
                    dest="numerical_dispersion",
                    action="store_false",
                    help="Do not overplot the numerical dispersion relation of the LCCL field solver")

parser.add_argument("--fields",
                    action="store",
                    choices=["fg", "vg"],
                    default="fg",
                    help="Read E and B from the field grid (fg_e, fg_b; averaged onto the Yee lattice, which "
                         "attenuates power near k dx = pi) or the cell-centred Vlasov-grid output "
                         "(vg_e_vol, vg_b_vol) (default: %(default)s)")
parser.add_argument("--measure-cutoffs",
                    dest="measure_cutoffs",
                    action="store_true",
                    help="Measure the frequencies of the spectral peaks at the smallest nonzero |k| near the "
                         "predicted cutoffs, print them and write them to cutoffs.txt in the run directory")
parser.add_argument("--window",
                    dest="window",
                    action="store",
                    choices=["both", "temporal", "spatial", "none"],
                    default="both",
                    help="Hamming windows applied before the FFT. In a periodic domain the spatial window is not "
                         "needed and spreads each mode over neighbouring k bins, which biases --measure-cutoffs "
                         "for dispersive modes; use 'temporal' there (default: %(default)s)")
parser.add_argument("--cutoff-kbins",
                    dest="cutoff_kbins",
                    action="store",
                    type=int,
                    default=1,
                    help="Number of smallest nonzero |k| bins on each side averaged by --measure-cutoffs; "
                         "more bins add signal but shift the peak by the dispersion of the mode (default: %(default)s)")
parser.add_argument("--no-kinetic",
                    dest="kinetic",
                    action="store_false",
                    help="Do not overplot the kinetic (warm-plasma) Langmuir and parallel L/R curves")

parser.add_argument("--elec-bernstein",
                    dest="elec_bernstein",
                    action="store_true",
                    help="plot expectation for Electron bernstein modes (default: %(default)s)")
parser.add_argument("dirname",
                    nargs='*',
                    action="store",
                    type=str,
                    default=".",
                    help="Simulation directory (default: %(default)s)")

args = parser.parse_args()

if args.elec_bernstein:
    from bernstein import bernstein_root_in_bracket

if len(args.dirname) != 1:
    sys.stderr.write(str(args.dirname)+" is a list with more than one entry. That doesn't work currently. Please make separate calls to this tool\n")
    sys.exit(1)
else:
    args.dirname = args.dirname[0]


# load environment
ptinteractive = int(os.environ.get('PTNOINTERACTIVE', '0')) == 0
have_latex    = int(os.environ.get('PTNOLATEX',       '0')) == 0

# set up tqdm or a replacement
if ptinteractive: # interactive mode
    try:
        from tqdm import tqdm
        have_tqdm = True
    except ImportError:
        print("WARNING: Could not import tqdm")
        have_tqdm = False
else:
    have_tqdm = False
if not have_tqdm:
    tqdm = lambda x : x

# Configure windowing, can be "none", "spatial", "temporal" or "both" (--window)
do_windowing=args.window

# bunch of natural constants
class SI:
    e = 1.6022e-19 #C
    mp = 1.6726e-27 #kg
    me = 9.1094e-31 #kg
    eps0 = 8.8542e-12 # F/m
    mu0 = 4.*np.pi*1e-7 # H/m
    kB = 1.3807e-23 # J/K
    c = 2.9979e8 # m/s

# Be clever about ignorable dimensions
def auto_squeeze_vec(array):
        if len(array.shape) == 4:
            return np.average(array, axis=(1,2))
        elif len(array.shape) == 3:
            return np.average(array, axis=1)
        else:
            return array
def auto_squeeze_scalar(array):
        assert len(array.shape) < 4, "scalar fields should maximally be 3d"
        if len(array.shape) == 3:
            return np.average(array, axis=(1,2))
        elif len(array.shape) == 2:
            return np.average(array, axis=1)
        else:
            return array

def lccl_numerical_omega(k, dx, dt, wp, theta):
    """Frequency of the transverse EM mode in the AP (CSL-RME) field solver by Liu, Cai, Cao and Lapenta.

    Cold, unmagnetised plasma; K = sin(k dx)/dx is the centred collocated curl.
    Amplification factor g = (1 + i b (1-theta)) / (1 - i b theta), b = dt sqrt(wp^2 + c^2 K^2).
    """
    K = np.sin(k*dx)/dx
    b = dt*np.sqrt(wp**2 + (SI.c*K)**2)
    return (np.arctan(b*(1.-theta)) + np.arctan(b*theta))/dt

def plasma_Z(zeta):
    """Plasma dispersion function (Fried and Conte)."""
    from scipy.special import wofz
    return 1j*np.sqrt(np.pi)*wofz(zeta)

def track_roots(D, kgrid, guess, wscale, max_damping=0.5):
    """Follow one root omega(k) of D(omega, k) = 0 along kgrid, starting from guess at kgrid[0].

    Stops where the root is lost or the damping exceeds max_damping times the real frequency,
    because no ridge is visible there. Returns complex omega, NaN after the stop.
    """
    from scipy.optimize import root
    out = np.full(len(kgrid), np.nan, dtype=complex)
    prev = []
    x = complex(guess)/wscale
    for i, k in enumerate(kgrid):
        if len(prev) >= 2:   # linear extrapolation keeps the solver on the branch
            x = 2*prev[-1] - prev[-2]
        def f(v):
            d = D(complex(v[0], v[1])*wscale, k)
            return [d.real, d.imag]
        sol = root(f, [x.real, x.imag])
        z = complex(sol.x[0], sol.x[1])
        if not sol.success or z.real <= 0. or abs(z.imag) > max_damping*z.real:
            break
        out[i] = z*wscale
        prev.append(z)
    return out

def add_panel_label(ax, label, loc):
    """Draw e.g. (a) in a corner of the axes; the box keeps it readable on any colormap."""
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va, fontsize="large", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

def finish_plot(ax):
    if args.panel_label != "":
        add_panel_label(ax, args.panel_label, args.panel_label_loc)

# find timesteps that are present. Maybe we should outsource that to analysator
timesteps = []
for filename in glob.glob(args.dirname+"/bulk*vlsv"):
    parts = filename.split("/")[-1].split(".")
    if len(parts) == 3:
        t = int(parts[1])
        if t%args.decimate == 0:
            timesteps.append(t)
timesteps.sort()
if args.limit > 1:
    start = int((args.limit_part-1)*len(timesteps)/args.limit)
    stop  = int((args.limit_part  )*len(timesteps)/args.limit)
    if start < 0:
        start = 0
    if stop >= len(timesteps):
        stop = len(timesteps)-1
    timesteps = timesteps[start:stop]

tsize = len(timesteps)

# collection of output quantities present in the files
variablename = {}
variableunit = {}
variables = {}

# fg_* are averaged onto the Yee lattice (power x cos^2(k dx/2) per averaged direction); vg_*_vol are cell-centred
Ename, Bname = ("vg_e_vol", "vg_b_vol") if args.fields == "vg" else ("fg_e", "fg_b")

def read_vec(f, name):
    """Vector field along x, averaged over the ignorable directions: shape (xsize, 3)."""
    if name.startswith("fg_"):
        return auto_squeeze_vec(f.read_fsgrid_variable(name))
    cellids = f.read_variable("cellid")
    v = f.read_variable(name)[cellids.argsort()]    # CellID-1 = i + nx*j + nx*ny*k
    return np.average(v.reshape([zsize*ysize, xsize, 3]), axis=0)

# Analyze first timestep
for i,t in enumerate(timesteps[:1]):
    #f = analysator.vlsvfile.VlsvReader(args.dirname+"/bulk."+"{:07d}".format(t)+".vlsv", fsGridDecomposition=[1,1,1])
    f = analysator.vlsvfile.VlsvReader(args.dirname+"/bulk."+"{:07d}".format(t)+".vlsv")

    [xsize, ysize, zsize] = map(int,f.get_fsgrid_mesh_size()) # uint64t makes some other stuff unhappy

    if f.check_variable(Bname):
        B0vec = np.average(read_vec(f, Bname), axis=0)
        have_B = True
        if have_latex:
            variablename["B"] = f.read_variable_info(Bname).latex
            variableunit["B"] = f.read_variable_info(Bname).latexunits
        else:
            variablename["B"] = "B"
            variableunit["B"] = f.read_variable_info(Bname).units
    else:
        B0vec = np.zeros(3)
        have_B = False

    B0 = np.sqrt(np.sum(B0vec**2))

    if f.check_variable(Ename):
        have_E = True
        if have_latex:
            variablename["E"] = f.read_variable_info(Ename).latex
            variableunit["E"] = f.read_variable_info(Ename).latexunits
        else:
            variablename["E"] = "E"
            variableunit["E"] = f.read_variable_info(Ename).units
    else:
        have_E = False

    if f.check_variable("fg_phi"):
        have_Phi = True
        if have_latex:
            variablename["Phi"] = f.read_variable_info("fg_phi").latex
            variableunit["Phi"] = f.read_variable_info("fg_phi").latexunits
        else:
            variablename["Phi"] = "Phi"
            variableunit["Phi"] = f.read_variable_info("fg_phi").units
    else:
        have_Phi = False

    if f.check_variable("fg_rhoq"):
        have_rho = True
        if have_latex:
            variablename["rho"] = f.read_variable_info("fg_rhoq").latex
            variableunit["rho"] = f.read_variable_info("fg_rhoq").latexunits
        else:
            variablename["rho"] = "rho"
            variableunit["rho"] = f.read_variable_info("fg_rhoq").units
    else:
        have_rho = False

    if f.check_variable("vg_eje"):
        have_Eje = True
        if have_latex:
            variablename["Eje"] = f.read_variable_info("vg_eje").latex
            variableunit["Eje"] = f.read_variable_info("vg_eje").latexunits
        else:
            variablename["Eje"] = "Eje"
            variableunit["Eje"] = f.read_variable_info("vg_eje").units
    else:
        have_Eje = False

    if f.check_variable("electron/vg_rho"):
        have_ne = True
        if have_latex:
            variablename["ne"] = f.read_variable_info("electron/vg_rho").latex
            variableunit["ne"] = f.read_variable_info("electron/vg_rho").latexunits
            variablename["deltane"] = r"$\Delta n_e$"
            variableunit["deltane"] = f.read_variable_info("electron/vg_rho").latexunits
        else:
            variablename["ne"] = "n_e"
            variableunit["ne"] = f.read_variable_info("electron/vg_rho").units
            variablename["deltane"] = "n_e - n0"
            variableunit["deltane"] = f.read_variable_info("electron/vg_rho").units
    else:
        have_ne = False

    if f.check_variable("proton/vg_rho"):
        have_ni = True
        if have_latex:
            variablename["ni"] = f.read_variable_info("proton/vg_rho").latex
            variableunit["ni"] = f.read_variable_info("proton/vg_rho").latexunits
            variablename["deltani"] = r"$\Delta n_i$"
            variableunit["deltani"] = f.read_variable_info("proton/vg_rho").latexunits
        else:
            variablename["ni"] = "n_i"
            variableunit["ni"] = f.read_variable_info("proton/vg_rho").units
            variablename["deltani"] = "n_i - n0"
            variableunit["deltani"] = f.read_variable_info("proton/vg_rho").units
    else:
        have_ni = False

if args.onlyE:
    have_B = False
    have_Phi = False
    have_rho = False
    have_Eje = False
    have_ne = False
    have_ni = False

# Report what we found in the file
print("Found field grid with "+str(xsize)+"x"+str(ysize)+"x"+str(zsize)+" cells")

dt = f.read_parameter("dt")
config=f.get_config()

dtout = args.decimate * float(config["io"]["system_write_t_interval"][0])
if dt > dtout:
    dtout = dt
xmin = f.read_parameter("xmin")
xmax = f.read_parameter("xmax")
dx = (xmax-xmin)/xsize

fieldsolver_method = "default_fieldsolver"
theta = 0.5 # default of fieldsolver.theta in parameters.cpp
if "fieldsolver" in config:
    if "method" in config["fieldsolver"]:
        fieldsolver_method = config["fieldsolver"]["method"][0]
    if "theta" in config["fieldsolver"]:
        theta = float(config["fieldsolver"]["theta"][0])

print("Found "+str(tsize)+" timesteps")
print("B_0 = "+str(B0vec)+" T")
print("dt = "+str(dt)+" s")
print("dtout = "+str(dtout)+" s")
print("T_end = "+str(timesteps[-1]*dtout)+" s")
print("dx = "+str(dx)+" m")
print("field solver = "+fieldsolver_method+(", theta = "+str(theta) if fieldsolver_method == "LCCL" else ""))

# some projects have additional keys in the config that we can use
if "proton_Dispersion" in config and "rho" in config["proton_Dispersion"]:
    ni0 = float(config["proton_Dispersion"]["rho"][0])
elif "Dispersion" in config and "rho" in config["Dispersion"]:
    ni0 = float(config["Dispersion"]["rho"][0])
else:
    ni0 = None
if "proton_Dispersion" in config and "Temperature" in config["proton_Dispersion"]:
    Ti0 = float(config["proton_Dispersion"]["Temperature"][0])
elif "Dispersion" in config and "Temperature" in config["Dispersion"]:
    Ti0 = float(config["Dispersion"]["Temperature"][0])
else:
    Ti0 = None

if "electron_Dispersion" in config and "rho" in config["electron_Dispersion"]:
    ne0 = float(config["electron_Dispersion"]["rho"][0])
elif "Dispersion" in config and "rho" in config["Dispersion"]:
    ne0 = float(config["Dispersion"]["rho"][0])
else:
    ne0 = None
if "electron_Dispersion" in config and "Temperature" in config["electron_Dispersion"]:
    Te0 = float(config["electron_Dispersion"]["Temperature"][0])
elif "Dispersion" in config and "Temperature" in config["Dispersion"]:
    Te0 = float(config["Dispersion"]["Temperature"][0])
else:
    Te0 = None

# reasonable assumption
if ne0 is None and ni0 is not None:
    ne0 = ni0
elif ni0 is None and ne0 is not None:
    ni0 = ne0
# somewhat wild assumption
if Te0 is None and Ti0 is not None:
    Te0 = Ti0
elif Ti0 is None and Te0 is not None:
    Ti0 = Te0

print("n_p = "+str(ni0)+" m^-3")
print("n_e = "+str(ne0)+" m^-3")
print("T_p = "+str(Ti0)+" K")
print("T_e = "+str(Te0)+" K")

me = SI.me
if "electron_properties" in config:
    if "mass" in config["electron_properties"]:
        if "mass_units" in config["electron_properties"]:
            if config["electron_properties"]["mass_units"][0] == "ELECTRON":
                me = float(config["electron_properties"]["mass"][0]) * SI.me
            elif config["electron_properties"]["mass_units"][0] == "PROTON":
                me = float(config["electron_properties"]["mass"][0]) * SI.mp
            else:
                print("Don't know how to work with mass units of "+str(config["electron_properties"]["mass_units"][0]))
mp = SI.mp
if "proton_properties" in config:
    if "mass" in config["proton_properties"]:
        if "mass_units" in config["proton_properties"]:
            if config["proton_properties"]["mass_units"][0] == "ELECTRON":
                mp = float(config["proton_properties"]["mass"][0]) * SI.me
            elif config["proton_properties"]["mass_units"][0] == "PROTON":
                mp = float(config["proton_properties"]["mass"][0]) * SI.mp
            else:
                print("Don't know how to work with mass units of "+str(config["proton_properties"]["mass_units"][0]))

print("numerical me = "+str(me / SI.me)+" real me")
print("numerical mp = "+str(mp / SI.mp)+" real mp")
print("numerical mass ratio = "+str(mp/me))

# calculate derived quantities
Wci = SI.e * B0 / mp
Wce = SI.e * B0 / me
wpi = np.sqrt(ni0 * SI.e**2 / mp / SI.eps0)
wpe = np.sqrt(ne0 * SI.e**2 / me / SI.eps0)
vthi = np.sqrt(SI.kB * Ti0 / mp)
vA = B0 / np.sqrt(SI.mu0 * (me*ne0 + mp*ni0))
cS = np.sqrt(5./3. * SI.kB * Ti0 / mp)
vthe = np.sqrt(SI.kB * Te0 / me)
di = SI.c / wpi
de = SI.c / wpe
ri = vthi / Wci
re = vthe / Wce
lD = vthe / wpe

print("W_ci = "+str(Wci)+" 1/s")
print("W_ce = "+str(Wce)+" 1/s")
print("w_pi = "+str(wpi)+" 1/s")
print("w_pe = "+str(wpe)+" 1/s")
print("v_thi = "+str(vthi)+" m/s")
print("v_A = "+str(vA)+" m/s")
print("v_the = "+str(vthe)+" m/s")
print("d_i = "+str(di)+" m")
print("d_e = "+str(de)+" m")
print("r_i = "+str(ri)+" m")
print("r_e = "+str(re)+" m")
print("l_D = "+str(lD)+" m")

# The LCCL curve is derived for B = 0; fg_b of an unmagnetised run still averages to a tiny nonzero B0.
show_lccl_curve = args.numerical_dispersion and fieldsolver_method == "LCCL" and Wce < 1e-2*wpe
if args.numerical_dispersion and fieldsolver_method == "LCCL" and not show_lccl_curve:
    print("Not overplotting the LCCL numerical dispersion: only derived for an unmagnetised plasma (W_ce << w_pe)")
# The L/R curves assume k || B0; for k perpendicular to B0 the transverse mode is the X mode.

# Geometry of B0 relative to the resolved direction x decides which reference curves apply
if Wce < 1e-2*wpe:
    geometry = "unmagnetised"
elif abs(B0vec[0]) > 0.99*B0:
    geometry = "parallel"
elif abs(B0vec[0]) < 1e-2*B0:
    geometry = "perpendicular"
else:
    geometry = "oblique"
print("geometry: "+geometry)

wp = np.sqrt(wpe**2 + wpi**2)
wLcutoff = 0.5*(np.sqrt((Wci+Wce)**2+4.*wpe**2+4.*wpi**2)-(Wce-Wci))
wRcutoff = 0.5*(np.sqrt((Wci+Wce)**2+4.*wpe**2+4.*wpi**2)+(Wce-Wci))
wUH = np.sqrt(wpe**2 + Wce**2)
if geometry == "perpendicular":
    from scipy.optimize import brentq
    # The field component along B0 carries the O mode (Kilian et al. 2017, Sec. 5.3)
    perp_par_comp = "y" if abs(B0vec[1]) > abs(B0vec[2]) else "z"
    # Upper-hybrid resonance S = 0 with ions (Eq. 3.26 of Kilian et al. is the electron-only approximation)
    def _S(om):
        return 1. - wpe**2/(om**2 - Wce**2) - wpi**2/(om**2 - Wci**2)
    wUH = brentq(_S, Wce*(1. + 1e-9), wRcutoff) if Wce < wRcutoff else np.nan
    # Cold X mode n^2 = RL/S (Eq. 3.24 of Kilian et al.) and O mode n^2 = 1 - w_p^2/w^2 on a dense frequency
    # grid up to the temporal Nyquist frequency, so that cutoffs and the resonance are resolved
    wdense = np.linspace(1e-4*wpe, np.pi/dtout, 40000)
    with np.errstate(divide="ignore", invalid="ignore"):
        _R = 1. - wpe**2/(wdense*(wdense - Wce)) - wpi**2/(wdense*(wdense + Wci))
        _L = 1. - wpe**2/(wdense*(wdense + Wce)) - wpi**2/(wdense*(wdense - Wci))
        n2X = _R*_L/(0.5*(_R + _L))
        n2O = 1. - wp**2/wdense**2
    # NaN where n^2 <= 0 keeps branches from being joined across stop bands
    kXdense = np.where(np.isfinite(n2X) & (n2X > 0.), wdense/SI.c*np.sqrt(np.abs(n2X)), np.nan)
    kOdense = np.where(n2O > 0., wdense/SI.c*np.sqrt(np.abs(n2O)), np.nan)
    print("Overplotting the cold X and O modes; upper-hybrid frequency {:.5f} w_pe".format(wUH/wpe))

# Frequencies at k -> 0 (Kilian et al. 2017, Eqs. 3.18, 3.20-3.21, 3.25-3.27)
predicted_cutoffs = {"unmagnetised":  {"w_p": wp},
                     "parallel":      {"L cutoff": wLcutoff, "R cutoff": wRcutoff, "w_p": wp},
                     "perpendicular": {"L cutoff": wLcutoff, "w_UH": wUH, "R cutoff": wRcutoff, "w_p": wp},
                     "oblique":       {}}[geometry]
if args.measure_cutoffs:
    cutoff_file = open(args.dirname+"/cutoffs.txt", "w")
    cutoff_file.write("# measured at |k| <= {:d} * 2 pi/L, i.e. k*l_D <= {:.4g} and c*k/w_pe <= {:.4g}\n".format(
        args.cutoff_kbins, args.cutoff_kbins*2.*np.pi/(xsize*dx)*lD, args.cutoff_kbins*2.*np.pi/(xsize*dx)*SI.c/wpe))
    cutoff_file.write("# component, prediction, predicted w/w_pe at k = 0, cold-plasma prediction at the measured k "
                      "(nan: no single branch), measured w/w_pe, frequency bin dw/w_pe, peak power / median power over all w > 0\n")
    if geometry == "oblique":
        print("No cutoff predictions for oblique B0")

def cold_RLS(om):
    """Stix parameters R, L, S for electrons and protons (Kilian et al. 2017, Eqs. 3.10-3.13)."""
    R = 1. - wpe**2/(om*(om - Wce)) - wpi**2/(om*(om + Wci))
    L = 1. - wpe**2/(om*(om + Wce)) - wpi**2/(om*(om - Wci))
    return R, L, 0.5*(R + L)

def predicted_at_k(name, comp, k):
    """Cold-plasma frequency of the branch that starts at prediction `name`, at wavenumber k, as seen
    in component `comp`; None where no single branch applies."""
    from scipy.optimize import brentq
    ck2 = (SI.c*k)**2
    if geometry == "unmagnetised" and name == "w_p":
        return wp*np.sqrt(1. + 3.*(k*lD)**2) if comp == "x" else np.sqrt(wp**2 + ck2)
    if geometry == "parallel":
        if name == "w_p" and comp == "x":
            return wp*np.sqrt(1. + 3.*(k*lD)**2)
        if name in ("L cutoff", "R cutoff") and comp != "x":
            w0, idx = (wLcutoff, 1) if name == "L cutoff" else (wRcutoff, 0)
            f = lambda om: om**2*cold_RLS(om)[idx] - ck2
            return brentq(f, w0*(1. + 1e-12), w0 + np.sqrt(ck2) + wpe)
    if geometry == "perpendicular":
        par = 1 if abs(B0vec[1]) > abs(B0vec[2]) else 2      # field component along B0: O mode
        if comp == "xyz"[par]:
            return np.sqrt(wp**2 + ck2) if name == "w_p" else None
        if comp in ("x", "xyz"[3 - par]) and name in ("L cutoff", "R cutoff"):
            def f(om):
                R, L, S = cold_RLS(om)
                return om**2*R*L/S - ck2
            if name == "L cutoff":   # lower X branch, between the L cutoff and the upper-hybrid resonance
                from scipy.optimize import brentq as bq
                wres = bq(lambda om: cold_RLS(om)[2], wLcutoff*1.0001, wRcutoff)
                return brentq(f, wLcutoff*(1. + 1e-12), wres*(1. - 1e-9))
            return brentq(f, wRcutoff*(1. + 1e-12), wRcutoff + np.sqrt(ck2) + wpe)
    return None

def measure_cutoffs(power, kx, w, label):
    """Peak frequencies at the smallest nonzero |k| near each predicted cutoff."""
    dk = 2.*np.pi/(xsize*dx)
    n = np.rint(kx/dk).astype(int)
    # k = 0 is left out: it also holds the broadband power of uniform fields
    spec = power[:, (abs(n) >= 1) & (abs(n) <= args.cutoff_kbins)].mean(axis=1)
    dw = w[1] - w[0]
    names = list(predicted_cutoffs.keys())
    for name in names:
        wc = predicted_cutoffs[name]
        others = [abs(predicted_cutoffs[m] - wc) for m in names if m != name]
        half = min([0.15*wpe] + [0.5*d for d in others])
        sel = np.where((w > max(0., wc - half)) & (w < wc + half))[0]
        if wc + half > w[-1]:
            msg = "above the temporal Nyquist frequency"
        elif len(sel) < 3:
            msg = "search window narrower than three frequency bins"
        else:
            j = sel[np.argmax(spec[sel])]
            if j in (sel[0], sel[-1]):
                msg = "no peak inside the search window"
            else:
                l = np.log(spec[j-1:j+2])
                delta = 0.5*(l[0] - l[2])/(l[0] - 2.*l[1] + l[2])   # parabolic interpolation of the log
                wm = w[j] + delta*dw
                contrast = spec[j]/np.median(spec[w > 0])
                comp = label[1:] if label[1:] in ("x", "y", "z") else "lr"
                kk = [m*dk for m in range(1, args.cutoff_kbins + 1)]
                pk = [predicted_at_k(name, comp, q) for q in kk]
                wk = np.mean(pk) if None not in pk else np.nan
                msg = "{:.4f} (predicted {:.4f} at k = 0, {} at the measured k; bin {:.4f}, contrast {:.1f})".format(
                    wm/wpe, wc/wpe, "n/a" if np.isnan(wk) else "{:.4f}".format(wk/wpe), dw/wpe, contrast)
                cutoff_file.write("{}, {}, {:.5f}, {}, {:.5f}, {:.5f}, {:.2f}\n".format(
                    label, name, wc/wpe, "nan" if np.isnan(wk) else "{:.5f}".format(wk/wpe), wm/wpe, dw/wpe, contrast))
        print("  {} {}: {}".format(label, name, msg))
    if show_lccl_curve and "w_p" in predicted_cutoffs:
        print("  ({}: the LCCL scheme moves w_p to {:.4f} w_pe at k = 0)".format(label, lccl_numerical_omega(0., dx, dt, wp, theta)/wpe))

# Kinetic reference curves: Maxwellian electrons and protons with v_th = sqrt(kB T/m), zeta = (...)/(sqrt(2) k v_th)
show_kinetic_langmuir = args.kinetic and geometry in ("unmagnetised", "parallel")
show_warm_parallel = args.kinetic and geometry == "parallel"
species = [(wpe, vthe, -Wce), (wpi, vthi, Wci)]   # (plasma frequency, thermal speed, signed gyrofrequency)
kkin = np.linspace(1e-3, 1., 400)*min(np.pi/dx, 3./lD)

def D_langmuir(om, k):
    """Longitudinal dielectric function (Stix 1992, Ch. 8), unchanged by B0 for k || B0."""
    return 1. + sum(wps**2/(k*vts)**2*(1. + om/(np.sqrt(2.)*k*vts)*plasma_Z(om/(np.sqrt(2.)*k*vts)))
                    for wps, vts, _ in species)

def D_parallel(om, k, sign):
    """Circularly polarised modes for k || B0 (Stix 1992, Ch. 11): sign=+1 is R, sign=-1 is L. Divided by w_pe^2."""
    d = om**2 - (SI.c*k)**2
    for wps, vts, Ws in species:
        ws = np.sqrt(2.)*k*vts
        d += wps**2*om/ws*plasma_Z((om + sign*Ws)/ws)
    return d/wpe**2

if show_kinetic_langmuir:
    print("Tracking the kinetic Langmuir branch")
    wLangmuir = track_roots(D_langmuir, kkin, wpe*np.sqrt(1. + 3.*(kkin[0]*lD)**2), wpe)
if show_warm_parallel:
    print("Tracking the warm parallel L and R branches")
    def cold_root(kfun, wlo, whi, k):
        """Cold-plasma frequency on the branch between wlo and whi at wavenumber k."""
        ws = np.linspace(wlo, whi, 20000)
        with np.errstate(invalid="ignore"):
            ks = kfun(ws)
        good = np.isfinite(ks)
        return np.interp(k, ks[good], ws[good])
    def kR_cold(w):
        return 1./SI.c * np.sqrt(w**2 - wpe**2/(1.-Wce/w) - wpi**2/(1.+Wci/w))
    def kL_cold(w):
        return 1./SI.c * np.sqrt(w**2 - wpe**2/(1.+Wce/w) - wpi**2/(1.-Wci/w))
    whi = 1.01*np.sqrt(wRcutoff**2 + (SI.c*kkin[-1])**2) + wpe
    wRhf_warm = track_roots(lambda om, k: D_parallel(om, k, +1), kkin,
                            cold_root(kR_cold, wRcutoff*1.0001, whi, kkin[0]), wpe)
    wLhf_warm = track_roots(lambda om, k: D_parallel(om, k, -1), kkin,
                            cold_root(kL_cold, wLcutoff*1.0001, whi, kkin[0]), wpe)
    # whistler: start where the cold branch has reached 5 % of W_ce, below which ions matter
    k_start = kR_cold(np.array([0.05*Wce]))[0]
    kwh = kkin[kkin >= k_start]
    wRlf_warm = track_roots(lambda om, k: D_parallel(om, k, +1), kwh,
                            cold_root(kR_cold, 0.01*Wce, 0.9999*Wce, kwh[0]), wpe) if len(kwh) else np.array([])

# The shape of these fields is: time, space, field component (including left and right handed)
if have_B:
    B = np.zeros( (len(timesteps), xsize, 5) , dtype=complex)
if have_E:
    E = np.zeros( (len(timesteps), xsize, 5) , dtype=complex)
if have_Phi:
    Phi = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)
if have_rho:
    rho = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)
if have_Eje:
    Eje = np.zeros( (len(timesteps), xsize, 5) , dtype=complex)
if have_ne:
    ne = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)
    delta_ne = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)
if have_ni:
    ni = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)
    delta_ni = np.zeros( (len(timesteps), xsize, 1) , dtype=complex)

print("Loading data")
for i in tqdm(range(len(timesteps))):
    t = timesteps[i]
    if not have_tqdm:
        print("Output step "+str(i)+"/"+str(len(timesteps))+" at time "+str(t))

    #f = analysator.vlsvfile.VlsvReader(args.dirname+"/bulk."+"{:07d}".format(t)+".vlsv", fsGridDecomposition=[1,1,1])
    f = analysator.vlsvfile.VlsvReader(args.dirname+"/bulk."+"{:07d}".format(t)+".vlsv")

    if have_B:
        B[i,:,:3] = read_vec(f, Bname)
    if have_E:
        E[i,:,:3] = read_vec(f, Ename)
    if have_Phi:
        fg_phi = f.read_fsgrid_variable("fg_phi")
        Phi[i,:,0] = auto_squeeze_scalar(fg_phi)
    if have_rho:
        fg_rhoq = f.read_fsgrid_variable("fg_rhoq")
        rho[i,:,0] = auto_squeeze_scalar(fg_rhoq)
    if have_Eje:
        cellids = f.read_variable('cellid')
        vg_eje = f.read_variable('vg_eje')
        Eje[i,:,:3]= auto_squeeze_vec(vg_eje[cellids.argsort()].reshape([1,zsize*ysize,xsize,3]))
    if have_ne:
        cellids = f.read_variable('cellid')
        vg_ne = f.read_variable('electron/vg_rho')
        # there must be a better way of doing this
        ne_ = vg_ne[cellids.argsort()].reshape([zsize*ysize,xsize]).T.reshape([xsize,zsize*ysize,1])
        ne[i,:,0] = auto_squeeze_scalar(ne_)
        delta_ne[i,:,0] = ne[i,:,0] - ne0
    if have_ni:
        cellids = f.read_variable('cellid')
        vg_ni = f.read_variable('proton/vg_rho')
        ni_ = vg_ni[cellids.argsort()].reshape([zsize*ysize,xsize]).T.reshape([xsize,zsize*ysize,1])
        ni[i,:,0] = auto_squeeze_scalar(ni_)
        delta_ni[i,:,0] = ni[i,:,0] - ni0

if have_ni:
    delta_ni[:,:,0] -= np.average(delta_ni[:,:,0])
if have_ne:
    delta_ne[:,:,0] -= np.average(delta_ne[:,:,0])

# form left and right handed circular component by complex addition
if have_B:
    B[:,:,3] = B[:,:,1] - complex("j")*B[:,:,2]
    B[:,:,4] = B[:,:,1] + complex("j")*B[:,:,2]
if have_E:
    E[:,:,3] = E[:,:,1] - complex("j")*E[:,:,2]
    E[:,:,4] = E[:,:,1] + complex("j")*E[:,:,2]
if have_Eje:
    Eje[:,:,3] = Eje[:,:,1] - complex("j")*Eje[:,:,2]
    Eje[:,:,4] = Eje[:,:,1] + complex("j")*Eje[:,:,2]

# window as required
if do_windowing=="spatial" or do_windowing=="both":
    spatial_window = np.hamming(xsize)
else:
    spatial_window = np.ones(xsize)
if do_windowing=="temporal" or do_windowing=="both":
    temporal_window = np.hamming(tsize)
else:
    temporal_window = np.ones(tsize)
window = np.outer(spatial_window, temporal_window).T

vectorcomponentnames = ["x","y","z", "left", "right"]
scalarcomponentnames = [""]


if have_E:
    print("Ex", np.average(E[:,:,0]), np.std(E[:,:,0]))
    print("Ey", np.average(E[:,:,1]), np.std(E[:,:,1]))
    print("Ez", np.average(E[:,:,2]), np.std(E[:,:,2]))


print("Plotting data")

if args.freqnorm == "wpe":
    freqnorm = wpe
    if have_latex:
        freqnormlabel = r"$\omega_{pe}$"
    else:
        freqnormlabel = "w_pe"
elif args.freqnorm == "wpi":
    freqnorm = wpi
    if have_latex:
        freqnormlabel = r"$\omega_{pi}$"
    else:
        freqnormlabel = "w_pi"
elif args.freqnorm == "Wce":
    freqnorm = Wce
    if have_latex:
        freqnormlabel = r"$\Omega_{ce}$"
    else:
        freqnormlabel = "W_ce"
elif args.freqnorm == "Wci":
    freqnorm = Wci
    if have_latex:
        freqnormlabel = r"$\Omega_{ci}$"
    else:
        freqnormlabel = "W_ci"
elif args.freqnorm == "dt":
    freqnorm = np.pi/dt
    if have_latex:
        freqnormlabel = r"$\pi/\Delta t$"
    else:
        freqnormlabel = "pi/dt"

if args.lengthnorm == "lD":
    lengthnorm = lD/np.pi
    if have_latex:
        lengthnormlabel = r"$\lambda_D / \pi$"
    else:
        lengthnormlabel = "l_D / pi"
elif args.lengthnorm == "re":
    lengthnorm = re/np.pi
    if have_latex:
        lengthnormlabel = r"$r_e / \pi$"
    else:
        lengthnormlabel = "r_e / pi"
elif args.lengthnorm == "ri":
    lengthnorm = ri/np.pi
    if have_latex:
        lengthnormlabel = r"$r_i / \pi$"
    else:
        lengthnormlabel = "r_i / pi"
elif args.lengthnorm == "de":
    lengthnorm = de/np.pi
    if have_latex:
        lengthnormlabel = r"$d_e / \pi$"
    else:
        lengthnormlabel = "d_e / pi"
elif args.lengthnorm == "di":
    lengthnorm = di/np.pi
    if have_latex:
        lengthnormlabel = r"$d_i / \pi$"
    else:
        lengthnormlabel = "d_i / pi"
elif args.lengthnorm == "dx":
    lengthnorm = dx/np.pi
    if have_latex:
        lengthnormlabel = r"$\Delta_x / \pi$"
    else:
        lengthnormlabel = "dx / pi"

total = 0
if have_B:
    variables["B"] = B
    total += 3+5+5
if have_E:
    variables["E"] = E
    total += 3+5+5
if have_Phi:
    variables["Phi"] = Phi
    total += 3
if have_rho:
    variables["rho"] = rho
    total += 3
if have_Eje:
    variables["Eje"] = Eje
    total += 3+5+5
if have_ne:
    variables["ne"] = ne
    variables["deltane"] = delta_ne
    total += 3+3
if have_ni:
    variables["ni"] = ni
    variables["deltani"] = delta_ni
    total += 3+3

if have_tqdm:
    pbar = tqdm(total=total)

if args.elec_bernstein:
    maxbernstein = 10
    kbernstein = np.linspace(0.0, np.pi/dx, xsize+1)[1:]
    omegabernstein = np.zeros( (len(kbernstein), maxbernstein) )
    for i in range(maxbernstein):
        omegabernstein[:,i] =  np.array([bernstein_root_in_bracket(k, i, Wce, lD, re, order=2*maxbernstein)
                                         for k in kbernstein])

# plot everything we have loaded and computed
for variablecode,variable in variables.items():
    if variablecode in ["ne", "ni", "deltane", "deltani", "Phi", "rho"]: # the scalar components we know how to plow
        componentnames = scalarcomponentnames
    else:
        componentnames = vectorcomponentnames
    for c in range(len(componentnames)):

        # print(variablecode,componentnames[c],np.amin(variable[:,:,c].real),np.amax(variable[:,:,c].real))

        # plot x-t space
        if c < 3:
            plt.figure(variablecode+componentnames[c], figsize=args.figsize)
            plt.title(args.title)
            X = np.linspace(xmin, xmax, xsize)
            T = np.linspace(timesteps[0], timesteps[-1], len(timesteps))
            vmax = np.amax(abs(variable[:,:,c].real))
            im = plt.pcolormesh(X/lengthnorm, T*dtout*freqnorm, variable[:,:,c].real, shading="gouraud", cmap=args.cmap)#, vmin=-vmax, vmax=vmax)
            if have_latex:
                labelstr = variablename[variablecode]+"$_{"+componentnames[c]+r"} \,/\, $"+variableunit[variablecode]
            else:
                labelstr = variablename[variablecode]+"_{"+componentnames[c]+"} / "+variableunit[variablecode]
            plt.colorbar(im, label=labelstr)
            if have_latex:
                plt.xlabel(r"$x \,/\, $"+lengthnormlabel)
                plt.ylabel(r"$t \, $"+freqnormlabel)
            else:
                plt.xlabel("x / "+lengthnormlabel)
                plt.ylabel("t * "+freqnormlabel)
            finish_plot(plt.gca())
            plt.tight_layout()
            imgname = args.dirname+"/"+variablecode+componentnames[c]
            if args.limit > 1.:
                imgname += "_"+str(int(args.limit_part))+"_"+str(int(args.limit))
            plt.savefig(imgname+".png")
            plt.close()
            if have_tqdm:
                pbar.update()

        # plot k-omega power
        plt.figure("k"+variablecode+componentnames[c], figsize=args.figsize)
        plt.title(args.title)
        kV = np.fft.fftshift(np.fft.fft2(variable[:,:,c]*window))
        w  = 2.*np.pi*np.fft.fftshift(np.fft.fftfreq(tsize, d=dtout))
        kx = 2.*np.pi*np.fft.fftshift(np.fft.fftfreq(xsize, d=dx))

        # L mode
        def kL(w):
            return 1./SI.c * np.sqrt(w**2 - wpe**2/(1.+Wce/w) - wpi**2/(1.-Wci/w))
        # lf L mode
        maskLlf = np.logical_and(w>0, w<Wci)
        wleftlf = w[maskLlf]
        kleftlf = kL(wleftlf)
        # hf L mode
        maskLhf = w>wLcutoff
        wlefthf = w[maskLhf]
        klefthf = kL(wlefthf)

        # R mode
        def kR(w):
            return 1./SI.c * np.sqrt(w**2 - wpe**2/(1.-Wce/w) - wpi**2/(1.+Wci/w))
        # lf R mode
        maskRlf = np.logical_and(w>0, w<Wce)
        wrightlf = w[maskRlf]
        krightlf = kR(wrightlf)
        # hf R mode
        maskRhf = w>wRcutoff
        wrighthf = w[maskRhf]
        krighthf = kR(wrighthf)

        # Langmuir mode
        def kP(w):
            return 1./vthe * np.sqrt(1./3. * (w**2 - wpe**2))
        maskP = w>wpe
        wplasma = w[maskP]
        kplasma = kP(wplasma)


        # Ion acoustic mode
        def kS(w):
            return w/cS

        power = kV.real**2 + kV.imag**2
        if np.amax(power) == 0.:
            power[:,:] = 1e-80
        if args.measure_cutoffs and variablecode == "E":
            measure_cutoffs(power, kx, w, "E"+componentnames[c])

        vmax = np.ceil(np.log10(np.amax(power)))
        vmin = vmax - args.cbrange

        im = plt.pcolormesh(kx*lengthnorm, w/freqnorm, np.log10(power), shading='gouraud', vmin=vmin, vmax=vmax, cmap=args.cmap)
        if have_latex:
            labelstr=r"$\log_{10} \left|\tilde{$"+variablename[variablecode]+"$}_{"+componentnames[c]+r"}\right|^2$"
            labelstr=labelstr.replace("$$", "")
        else:
            labelstr="log10 |~"+variablename[variablecode]+"_{"+componentnames[c]+"}|^2"
        plt.colorbar(im, label=labelstr)
        plt.plot( kplasma*lengthnorm,  wplasma/freqnorm, color="orange", linestyle=":", label="$P_{e}$" if have_latex else "P_e")
        plt.plot( kS(w)*lengthnorm,    w/freqnorm, color="orange", linestyle="--", label=r"$c_{S}$" if have_latex else "c_S")
        if geometry != "perpendicular":   # L and R are the modes for k || B0; for k perp B0 the X and O modes are drawn below
            plt.plot( kleftlf*lengthnorm,  wleftlf/freqnorm, color="red", linestyle=":", label="$L$" if have_latex else "L")
            plt.plot(-kleftlf*lengthnorm,  wleftlf/freqnorm, color="red", linestyle=":")
            plt.plot( klefthf*lengthnorm,  wlefthf/freqnorm, color="red", linestyle=":")
            plt.plot(-klefthf*lengthnorm,  wlefthf/freqnorm, color="red", linestyle=":")
            plt.plot( krightlf*lengthnorm, wrightlf/freqnorm, color="green", linestyle=":", label="$R$" if have_latex else "R")
            plt.plot(-krightlf*lengthnorm, wrightlf/freqnorm, color="green", linestyle=":")
            plt.plot( krighthf*lengthnorm, wrighthf/freqnorm, color="green", linestyle=":")
            plt.plot(-krighthf*lengthnorm, wrighthf/freqnorm, color="green", linestyle=":")

        if componentnames[c] in ("x", ""):
            # thermal noise floor of particle codes, Kilian et al. (2017) Eq. 5.5
            plt.plot( kx*lengthnorm, 3.*np.sqrt(2.)*vthe*kx/freqnorm, color="grey", linestyle=":", linewidth=0.8,
                      label=r"$3\sqrt{2}\,v_{th,e}k$" if have_latex else "3 sqrt(2) v_the k")
            plt.plot(-kx*lengthnorm, 3.*np.sqrt(2.)*vthe*kx/freqnorm, color="grey", linestyle=":", linewidth=0.8)

        if show_kinetic_langmuir:
            plt.plot( kkin*lengthnorm, wLangmuir.real/freqnorm, color="orange", linestyle="-", linewidth=0.8,
                      label="$P_{e}$ kinetic" if have_latex else "P_e kinetic")
            plt.plot(-kkin*lengthnorm, wLangmuir.real/freqnorm, color="orange", linestyle="-", linewidth=0.8)
        if show_warm_parallel:
            for kk, ww, col, lab in ((kkin, wRhf_warm, "green", "R warm"), (kwh, wRlf_warm, "green", None),
                                     (kkin, wLhf_warm, "red", "L warm")):
                if len(kk):
                    plt.plot( kk*lengthnorm, ww.real/freqnorm, color=col, linestyle="-", linewidth=0.8, label=lab)
                    plt.plot(-kk*lengthnorm, ww.real/freqnorm, color=col, linestyle="-", linewidth=0.8)

        if args.elec_bernstein:
            for i in range(maxbernstein):
                plt.plot(kbernstein*lengthnorm, omegabernstein[:,i]/freqnorm, color="white", linestyle=":")

        if geometry == "perpendicular":
            comp = componentnames[c]
            # E along B0 carries the O mode; E_x and the other transverse component carry the X mode
            if comp != perp_par_comp:
                plt.plot( kXdense*lengthnorm, wdense/freqnorm, color="blue", linestyle="--", linewidth=0.9,
                          label="$X$" if have_latex else "X")
                plt.plot(-kXdense*lengthnorm, wdense/freqnorm, color="blue", linestyle="--", linewidth=0.9)
                if not np.isnan(wUH):
                    plt.axhline(wUH/freqnorm, color="white", linestyle="-.", linewidth=0.6,
                                label=r"$\omega_{UH}$" if have_latex else "w_UH")
            if comp in (perp_par_comp, "left", "right"):
                plt.plot( kOdense*lengthnorm, wdense/freqnorm, color="magenta", linestyle="--", linewidth=0.9,
                          label="$O$" if have_latex else "O")
                plt.plot(-kOdense*lengthnorm, wdense/freqnorm, color="magenta", linestyle="--", linewidth=0.9)

        if show_lccl_curve:
            plt.plot(kx*lengthnorm, lccl_numerical_omega(kx, dx, dt, wp, theta)/freqnorm, color="white", linestyle="--", linewidth=0.8,
                     label=(r"LCCL, $\theta=$"+"{:g}".format(theta)) if have_latex else "LCCL, theta={:g}".format(theta))

        if Wci > 0.:
            plt.axhline(Wci/freqnorm, color="cyan", linestyle=":", linewidth=0.5, label=r"$\Omega_{ci}$" if have_latex else "W_ci")
        if Wce > 0.:
            plt.axhline(Wce/freqnorm, color="orange", linestyle=":", linewidth=0.5, label=r"$\Omega_{ce}$" if have_latex else "W_ce")
        plt.axhline(wpe/freqnorm, color="cyan", linestyle="--", linewidth=0.5, label=r"$\omega_{pe}$" if have_latex else "w_pe")
        plt.axhline(wpi/freqnorm, color="orange", linestyle="--", linewidth=0.5, label=r"$\omega_{pi}$" if have_latex else "w_pi")
        if have_latex:
            plt.xlabel(r"$k_x \,$"+lengthnormlabel)
            plt.ylabel(r"$\omega \,/\, $"+freqnormlabel)
        else:
            plt.xlabel("k_x "+lengthnormlabel)
            plt.ylabel("w / "+freqnormlabel)

        if args.kmax != "":
            kmax = float(args.kmax)
            plt.xlim(-kmax, kmax)
        else:
            #plt.xlim(kx[0]*lengthnorm, kx[-1]*lengthnorm)
            plt.xlim(-1.5/lD*lengthnorm, 1.5/lD*lengthnorm)

        if args.wmax != "":
            wmax = float(args.wmax)
            plt.ylim(0, wmax)
        else:
            #plt.ylim(   0., w[-1]/freqnorm)
            plt.ylim(   0.,  3.*wpe/freqnorm)

        if args.legend_loc != "none":
            plt.legend(loc=args.legend_loc)
        finish_plot(plt.gca())
        plt.tight_layout()
        imgname = args.dirname+"/k"+variablecode+componentnames[c]
        if args.limit > 1.:
            imgname += "_"+str(int(args.limit_part))+"_"+str(int(args.limit))
        plt.savefig(imgname+".png")
        plt.close()
        if have_tqdm:
            pbar.update()

        # plot t-k power
        plt.figure("s"+variablecode+componentnames[c], figsize=args.figsize)
        plt.title(args.title)
        # compute the fft only over the x->kx direction, don't apply temporal windowing
        sV = np.fft.fftshift(np.fft.fft(variable[:,:,c]*spatial_window, axis=1), axes=1)
        spower = sV.real**2 + sV.imag**2
        if np.amax(spower) == 0.:
            spower[:,:] = 1e-80
        maskK = kx>=0.
        im = plt.pcolormesh(T*dtout*freqnorm, kx[maskK]*lengthnorm, np.log10(spower[:,maskK].T), shading='gouraud', cmap=args.cmap)
        plt.colorbar(im, label=labelstr)
        if have_latex:
            plt.xlabel(r"$t \, $"+freqnormlabel)
            plt.ylabel(r"$k_x \, $"+lengthnormlabel)
        else:
            plt.xlabel("t * "+freqnormlabel)
            plt.ylabel("k_x "+lengthnormlabel)
        finish_plot(plt.gca())
        plt.tight_layout()
        imgname = args.dirname+"/s"+variablecode+componentnames[c]
        if args.limit > 1.:
            imgname += "_"+str(int(args.limit_part))+"_"+str(int(args.limit))
        plt.savefig(imgname+".png")
        plt.close()
        if have_tqdm:
            pbar.update()

if have_tqdm:
    pbar.close()

if args.measure_cutoffs:
    cutoff_file.close()
    print("wrote "+args.dirname+"/cutoffs.txt")
